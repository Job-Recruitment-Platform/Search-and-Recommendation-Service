"""
CF BASE TRAIN (UPDATED BUSINESS SIGNALS) – PRODUCTION READY
==========================================================

Mục tiêu:
- Train BASE Collaborative Filtering model (Implicit ALS)
- Học LONG-TERM preference của user
- Là nền cho retrain + evaluate
- KHÔNG deploy trực tiếp
- KHÔNG đánh giá trong file này

--------------------------------------------------
INPUT CSV (BẮT BUỘC):
user_id,job_id,event_type,occurred_at

CHỈ CHẤP NHẬN 4 EVENT SAU (LONG-TERM SIGNAL):
- APPLY                  : 1.0
- SAVE                   : 0.6
- CLICK_FROM_SEARCH      : 0.7
- CLICK_FROM_RECOMMENDED : 0.5
--------------------------------------------------

OUTPUT:
- CFModel/models/cf_model_base.pkl
--------------------------------------------------
"""

from __future__ import annotations

import os
import time
import math
import pickle
from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


# ======================================================
# CONFIGURATION – BUSINESS DECISION
# ======================================================

EVENT_WEIGHTS: Dict[str, float] = {
    "APPLY": 1.0,
    "SAVE": 0.6,
    "CLICK_FROM_SEARCH": 0.7,
    "CLICK_FROM_RECOMMENDED": 0.5,
}

ALLOWED_EVENTS = set(EVENT_WEIGHTS.keys())

# Time decay: giảm ảnh hưởng của hành vi quá cũ
# final_weight = base_weight * exp(-DECAY_LAMBDA * age_days)
DECAY_LAMBDA = 0.05

# ALS hyper-parameters (ổn định, không tune ở đây)
ALS_FACTORS = 64
ALS_REGULARIZATION = 0.01
ALS_ITERATIONS = 30
ALS_USE_GPU = False
ALS_RANDOM_STATE = 42

# Paths
INPUT_CSV_PATH = "data/cf_train.csv"
OUTPUT_MODEL_PATH = "CFModel/models/cf_model.pkl"


# ======================================================
# STEP 1: LOAD & VALIDATE TRAIN DATA
# ======================================================

def load_training_data(csv_path: str) -> pd.DataFrame:
    """
    - Đọc CSV train
    - Validate schema
    - Lọc đúng event long-term
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"user_id", "job_id", "event_type", "occurred_at"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Chỉ giữ lại long-term events
    df = df[df["event_type"].isin(ALLOWED_EVENTS)]

    if df.empty:
        raise ValueError("No valid training events after filtering")

    return df


# ======================================================
# STEP 2: BUILD USER–ITEM MATRIX (IMPLICIT)
# ======================================================

def build_user_item_matrix(
    df: pd.DataFrame,
) -> Tuple[
    csr_matrix,
    Dict[int, int],
    Dict[int, int],
    Dict[int, int],
    Dict[int, int],
]:
    """
    Convert interaction log → implicit user-item matrix.

    Logic:
    - Gộp interaction theo (user, item)
    - Áp dụng time-decay
    - Mapping ID → index
    """
    now_ts = int(time.time())
    aggregated_weights: Dict[Tuple[int, int], float] = defaultdict(float)

    for row in df.itertuples(index=False):
        user_id = int(row.user_id)
        job_id = int(row.job_id)
        event_type = row.event_type
        occurred_at = int(row.occurred_at)

        base_weight = EVENT_WEIGHTS[event_type]

        # Tính độ cũ (theo ngày)
        age_days = max((now_ts - occurred_at) / 86400.0, 0.0)

        # Time decay
        decay_factor = math.exp(-DECAY_LAMBDA * age_days)

        final_weight = base_weight * decay_factor
        aggregated_weights[(user_id, job_id)] += final_weight

    # Mapping user / item
    user_ids = sorted({u for u, _ in aggregated_weights.keys()})
    item_ids = sorted({i for _, i in aggregated_weights.keys()})

    user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_to_index = {iid: idx for idx, iid in enumerate(item_ids)}
    index_to_user_id = {idx: uid for uid, idx in user_id_to_index.items()}
    index_to_item_id = {idx: iid for iid, idx in item_id_to_index.items()}

    rows, cols, values = [], [], []

    for (user_id, job_id), weight in aggregated_weights.items():
        if weight <= 0:
            continue  # implicit ALS yêu cầu confidence > 0

        rows.append(user_id_to_index[user_id])
        cols.append(item_id_to_index[job_id])
        values.append(float(weight))

    user_item_matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )

    return (
        user_item_matrix,
        user_id_to_index,
        item_id_to_index,
        index_to_user_id,
        index_to_item_id,
    )


# ======================================================
# STEP 3: TRAIN IMPLICIT ALS MODEL
# ======================================================

def train_als_model(user_item_matrix: csr_matrix) -> AlternatingLeastSquares:
    """
    Train Implicit ALS model.

    Lưu ý:
    - implicit library yêu cầu input là (item x user)
    """
    model = AlternatingLeastSquares(
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
        use_gpu=ALS_USE_GPU,
        random_state=ALS_RANDOM_STATE,
        calculate_training_loss=True,
    )

    model.fit(user_item_matrix.T.tocsr(), show_progress=True)
    return model


# ======================================================
# STEP 4: SAVE BASE MODEL
# ======================================================

def save_base_model(
    model: AlternatingLeastSquares,
    user_id_to_index: Dict[int, int],
    item_id_to_index: Dict[int, int],
    index_to_user_id: Dict[int, int],
    index_to_item_id: Dict[int, int],
    output_path: str,
) -> None:
    """
    Lưu BASE model + mapping.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "model": model,
        "user_id_to_index": user_id_to_index,
        "item_id_to_index": item_id_to_index,
        "index_to_user_id": index_to_user_id,
        "index_to_item_id": index_to_item_id,
        "meta": {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_weights": EVENT_WEIGHTS,
            "decay_lambda": DECAY_LAMBDA,
            "als_factors": ALS_FACTORS,
            "als_iterations": ALS_ITERATIONS,
        },
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[OK] Base CF model saved to: {output_path}")


# ======================================================
# MAIN ENTRYPOINT
# ======================================================

if __name__ == "__main__":
    print("[START] Load training data")
    df_train = load_training_data(INPUT_CSV_PATH)

    print("[START] Build user-item matrix")
    (
        user_item_matrix,
        user_id_to_index,
        item_id_to_index,
        index_to_user_id,
        index_to_item_id,
    ) = build_user_item_matrix(df_train)

    print(
        f"[INFO] Matrix shape: users={user_item_matrix.shape[0]}, "
        f"items={user_item_matrix.shape[1]}"
    )

    print("[START] Train ALS base model")
    model = train_als_model(user_item_matrix)

    print("[START] Save base model")
    save_base_model(
        model,
        user_id_to_index,
        item_id_to_index,
        index_to_user_id,
        index_to_item_id,
        OUTPUT_MODEL_PATH,
    )

    print("[DONE] Base training completed successfully")
