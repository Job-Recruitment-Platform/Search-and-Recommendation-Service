"""
CF RETRAIN (INCREMENTAL / TĂNG CƯỜNG) – PRODUCTION READY
======================================================

Long-term events & weights:
- APPLY                  : 1.0
- SAVE                   : 0.6
- CLICK_FROM_SEARCH      : 0.7
- CLICK_FROM_RECOMMENDED : 0.5

--------------------------------------------------
INPUT CSV (DELTA ONLY):
user_id,job_id,event_type,occurred_at
--------------------------------------------------

OUTPUT:
- cf_model_candidate.pkl

NOTE:
- KHÔNG deploy
- KHÔNG evaluate
- Sinh model ứng viên cho bước evaluate
"""

from __future__ import annotations

import os
import time
import pickle
from collections import defaultdict
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


# ======================================================
# CONFIGURATION (BUSINESS-DRIVEN)
# ======================================================

EVENT_WEIGHTS: Dict[str, float] = {
    "APPLY": 1.0,
    "SAVE": 0.6,
    "CLICK_FROM_SEARCH": 0.7,
    "CLICK_FROM_RECOMMENDED": 0.5,
}

ALLOWED_EVENTS = set(EVENT_WEIGHTS.keys())

INCREMENTAL_ITERATIONS = 10
BLEND_FACTOR = 0.7

BASE_MODEL_PATH = "CFModel/models/cf_model.pkl"
INCREMENTAL_CSV_PATH = "data/cf_incremental.csv"
CANDIDATE_MODEL_PATH = "CFModel/models/cf_model_candidate.pkl"


# ======================================================
# STEP 1: LOAD BASE MODEL
# ======================================================

def load_base_model(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Base model not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    required = {
        "model",
        "user_id_to_index",
        "item_id_to_index",
        "index_to_user_id",
        "index_to_item_id",
    }
    if not required.issubset(data.keys()):
        raise ValueError("Invalid base model format")

    return data


# ======================================================
# STEP 2: LOAD INCREMENTAL DATA
# ======================================================

def load_incremental_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Incremental CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"user_id", "job_id", "event_type", "occurred_at"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[df["event_type"].isin(ALLOWED_EVENTS)]

    if df.empty:
        raise ValueError("No valid incremental events after filtering")

    return df


# ======================================================
# STEP 3: BUILD INCREMENTAL MATRIX
# ======================================================

def build_incremental_matrix(
    df: pd.DataFrame,
    user_id_to_index: Dict[int, int],
    item_id_to_index: Dict[int, int],
    index_to_user_id: Dict[int, int],
    index_to_item_id: Dict[int, int],
) -> Tuple[csr_matrix, Dict, Dict, Dict, Dict]:

    aggregated_weights: Dict[Tuple[int, int], float] = defaultdict(float)

    for row in df.itertuples(index=False):
        user_id = int(row.user_id)
        job_id = int(row.job_id)
        event_type = row.event_type

        aggregated_weights[(user_id, job_id)] += EVENT_WEIGHTS[event_type]

    # Mở rộng mapping nếu có user/item mới
    for user_id, job_id in aggregated_weights.keys():
        if user_id not in user_id_to_index:
            new_idx = len(user_id_to_index)
            user_id_to_index[user_id] = new_idx
            index_to_user_id[new_idx] = user_id

        if job_id not in item_id_to_index:
            new_idx = len(item_id_to_index)
            item_id_to_index[job_id] = new_idx
            index_to_item_id[new_idx] = job_id

    n_users = len(user_id_to_index)
    n_items = len(item_id_to_index)

    rows, cols, values = [], [], []

    for (user_id, job_id), weight in aggregated_weights.items():
        rows.append(user_id_to_index[user_id])
        cols.append(item_id_to_index[job_id])
        values.append(max(float(weight), 0.01))  # implicit ALS constraint

    matrix = csr_matrix(
        (values, (rows, cols)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    return (
        matrix,
        user_id_to_index,
        item_id_to_index,
        index_to_user_id,
        index_to_item_id,
    )


# ======================================================
# STEP 4: RETRAIN ALS (WARM-START)
# ======================================================

def retrain_model(
    base_model: AlternatingLeastSquares,
    incremental_matrix: csr_matrix,
) -> AlternatingLeastSquares:

    model = AlternatingLeastSquares(
        factors=base_model.factors,
        regularization=base_model.regularization,
        iterations=INCREMENTAL_ITERATIONS,
        random_state=base_model.random_state,
        calculate_training_loss=True,
    )

    model.fit(incremental_matrix.T.tocsr(), show_progress=True)
    return model


# ======================================================
# STEP 5: BLEND OLD & NEW FACTORS
# ======================================================

def blend_factors(
    base_model: AlternatingLeastSquares,
    new_model: AlternatingLeastSquares,
    old_user_count: int,
    old_item_count: int,
) -> None:

    new_model.user_factors[:old_user_count, :] = (
        BLEND_FACTOR * base_model.user_factors[:old_user_count, :]
        + (1 - BLEND_FACTOR) * new_model.user_factors[:old_user_count, :]
    )

    new_model.item_factors[:old_item_count, :] = (
        BLEND_FACTOR * base_model.item_factors[:old_item_count, :]
        + (1 - BLEND_FACTOR) * new_model.item_factors[:old_item_count, :]
    )


# ======================================================
# STEP 6: SAVE CANDIDATE MODEL
# ======================================================

def save_candidate_model(
    model: AlternatingLeastSquares,
    user_id_to_index: Dict[int, int],
    item_id_to_index: Dict[int, int],
    index_to_user_id: Dict[int, int],
    index_to_item_id: Dict[int, int],
    output_path: str,
) -> None:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "model": model,
        "user_id_to_index": user_id_to_index,
        "item_id_to_index": item_id_to_index,
        "index_to_user_id": index_to_user_id,
        "index_to_item_id": index_to_item_id,
        "meta": {
            "retrained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "incremental_iterations": INCREMENTAL_ITERATIONS,
            "blend_factor": BLEND_FACTOR,
            "event_weights": EVENT_WEIGHTS,
        },
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[OK] Candidate model saved to: {output_path}")


# ======================================================
# MAIN ENTRYPOINT
# ======================================================

if __name__ == "__main__":
    print("[START] Load base model")
    base_data = load_base_model(BASE_MODEL_PATH)

    base_model = base_data["model"]
    user_id_to_index = dict(base_data["user_id_to_index"])
    item_id_to_index = dict(base_data["item_id_to_index"])
    index_to_user_id = dict(base_data["index_to_user_id"])
    index_to_item_id = dict(base_data["index_to_item_id"])

    old_user_count = len(user_id_to_index)
    old_item_count = len(item_id_to_index)

    print("[START] Load incremental CSV")
    df_incremental = load_incremental_data(INCREMENTAL_CSV_PATH)

    print("[START] Build incremental matrix")
    (
        incremental_matrix,
        user_id_to_index,
        item_id_to_index,
        index_to_user_id,
        index_to_item_id,
    ) = build_incremental_matrix(
        df_incremental,
        user_id_to_index,
        item_id_to_index,
        index_to_user_id,
        index_to_item_id,
    )

    print("[START] Retrain ALS model")
    new_model = retrain_model(base_model, incremental_matrix)

    print("[START] Blend latent factors")
    blend_factors(
        base_model,
        new_model,
        old_user_count,
        old_item_count,
    )

    print("[START] Save candidate model")
    save_candidate_model(
        new_model,
        user_id_to_index,
        item_id_to_index,
        index_to_user_id,
        index_to_item_id,
        CANDIDATE_MODEL_PATH,
    )

    print("[DONE] Incremental retrain completed")
