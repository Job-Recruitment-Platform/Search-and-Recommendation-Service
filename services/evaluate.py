"""
CF EVALUATION (GATEKEEPER) – PRODUCTION READY
============================================

Mục tiêu:
- So sánh MODEL HIỆN TẠI (đang serve) vs MODEL ỨNG VIÊN (retrain)
- Đánh giá trên data GẦN ĐÂY (temporal holdout)
- Quyết định:
    PASS  → promote candidate → cf_model_latest.pkl
    FAIL  → rollback (giữ model cũ)

--------------------------------------------------
INPUT CSV (EVALUATION WINDOW):
user_id,job_id,event_type,occurred_at

event_type hợp lệ cho evaluate:
- APPLY
- SAVE
- CLICK_FROM_SEARCH
- CLICK_FROM_RECOMMENDED
--------------------------------------------------

OUTPUT:
- Boolean PASS / FAIL
- Log metric để theo dõi drift

--------------------------------------------------
NOTE QUAN TRỌNG (Production mindset):
- Đây là CỬA KIỂM SOÁT DUY NHẤT trước khi deploy
- Không tối ưu hyper-parameter ở đây
- Chỉ trả lời 1 câu hỏi:
  "Model mới có tệ hơn model cũ không?"
--------------------------------------------------
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Dict, Set, Tuple

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


# ======================================================
# CONFIGURATION
# ======================================================

# Event dùng để đánh giá (giống retrain)
ALLOWED_EVENTS = {
    "APPLY",
    "SAVE",
    "CLICK_FROM_SEARCH",
    "CLICK_FROM_RECOMMENDED",
}

# Metric config
TOP_K = 10

# Gatekeeping rule:
# Model mới được phép kém hơn tối đa bao nhiêu
NDCG_TOLERANCE = 0.01

# Paths
CURRENT_MODEL_PATH = "CFModel/models/cf_model.pkl"
CANDIDATE_MODEL_PATH = "CFModel/models/cf_model_candidate.pkl"
EVAL_CSV_PATH = "data/cf_eval.csv"

# Promote target
PROMOTED_MODEL_PATH = "CFModel/models/cf_model.pkl"


# ======================================================
# STEP 1: LOAD MODEL
# ======================================================

def load_model(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    required = {
        "model",
        "user_id_to_index",
        "item_id_to_index",
        "index_to_item_id",
    }
    if not required.issubset(data.keys()):
        raise ValueError("Invalid model format")

    return data


# ======================================================
# STEP 2: LOAD EVALUATION DATA
# ======================================================

def load_eval_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Eval CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"user_id", "job_id", "event_type", "occurred_at"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[df["event_type"].isin(ALLOWED_EVENTS)]

    if df.empty:
        raise ValueError("No valid evaluation data")

    return df


# ======================================================
# STEP 3: BUILD GROUND TRUTH
# ======================================================

def build_ground_truth(df: pd.DataFrame) -> Dict[int, Set[int]]:
    """
    Ground truth:
    - Với mỗi user, tập item họ TƯƠNG TÁC TÍCH CỰC
    """
    gt = defaultdict(set)

    for row in df.itertuples(index=False):
        gt[int(row.user_id)].add(int(row.job_id))

    return gt


# ======================================================
# STEP 4: EVALUATION METRICS
# ======================================================

def ndcg_at_k(recommended: list[int], relevant: Set[int], k: int) -> float:
    """
    Tính NDCG@K cho 1 user.
    """
    dcg = 0.0
    for idx, item_id in enumerate(recommended[:k]):
        if item_id in relevant:
            dcg += 1.0 / np.log2(idx + 2)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg


def evaluate_model(
    model_data: Dict,
    ground_truth: Dict[int, Set[int]],
    k: int,
) -> float:
    """
    Tính NDCG@K trung bình cho toàn bộ user.
    """
    model = model_data["model"]
    user_id_to_index = model_data["user_id_to_index"]
    index_to_item_id = model_data["index_to_item_id"]

    ndcgs = []

    # Dummy matrix để gọi recommend (implicit yêu cầu)
    n_users = len(user_id_to_index)
    n_items = len(index_to_item_id)
    dummy_user_item = csr_matrix((n_users, n_items))

    for user_id, true_items in ground_truth.items():
        if user_id not in user_id_to_index:
            continue

        user_idx = user_id_to_index[user_id]

        rec_indices, _ = model.recommend(
            userid=user_idx,
            user_items=dummy_user_item[user_idx],
            N=k,
            filter_already_liked_items=True,
        )

        rec_items = [
            index_to_item_id[item_idx] for item_idx in rec_indices
        ]

        ndcgs.append(ndcg_at_k(rec_items, true_items, k))

    return float(np.mean(ndcgs)) if ndcgs else 0.0


# ======================================================
# STEP 5: GATEKEEPER LOGIC
# ======================================================

def should_promote(
    old_score: float,
    new_score: float,
    tolerance: float,
) -> bool:
    """
    Quyết định promote hay không.
    """
    return new_score >= old_score - tolerance


# ======================================================
# STEP 6: PROMOTE MODEL
# ======================================================

def promote_model(candidate_path: str, target_path: str) -> None:
    os.replace(candidate_path, target_path)
    print(f"[PROMOTED] Candidate model promoted to: {target_path}")


# ======================================================
# MAIN ENTRYPOINT
# ======================================================

if __name__ == "__main__":
    print("[START] Load evaluation data")
    df_eval = load_eval_data(EVAL_CSV_PATH)
    ground_truth = build_ground_truth(df_eval)

    print("[START] Load current model")
    current_model = load_model(CURRENT_MODEL_PATH)

    print("[START] Load candidate model")
    candidate_model = load_model(CANDIDATE_MODEL_PATH)

    print("[EVAL] Evaluating current model")
    current_ndcg = evaluate_model(
        current_model, ground_truth, TOP_K
    )

    print("[EVAL] Evaluating candidate model")
    candidate_ndcg = evaluate_model(
        candidate_model, ground_truth, TOP_K
    )

    print(
        f"[RESULT] NDCG@{TOP_K} | "
        f"CURRENT={current_ndcg:.4f} "
        f"CANDIDATE={candidate_ndcg:.4f}"
    )

    if should_promote(current_ndcg, candidate_ndcg, NDCG_TOLERANCE):
        promote_model(CANDIDATE_MODEL_PATH, PROMOTED_MODEL_PATH)
        print("[PASS] Candidate model accepted")
    else:
        print("[FAIL] Candidate model rejected (rollback)")
