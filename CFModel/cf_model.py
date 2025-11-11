"""
Collaborative Filtering Model - Class-based API
Encapsulates model-related functionality: building matrix, training, recommending, evaluating, saving.
"""

import json
import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from implicit.als import AlternatingLeastSquares
import pickle
import os


class CollaborativeFilteringModel:
    """ALS-based Collaborative Filtering for implicit feedback."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 30,
        use_gpu: bool = False,
        random_state: int = 42,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model: Optional[AlternatingLeastSquares] = None

    @staticmethod
    def load_interactions(filepath: str) -> Tuple[List[Dict], Dict[str, float]]:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['interactions'], data['metadata']['interaction_weights']

    @staticmethod
    def split_interactions_by_time(
        interactions: List[Dict], test_ratio: float = 0.2
    ) -> Tuple[List[Dict], List[Dict]]:
        sorted_interactions = sorted(interactions, key=lambda x: x['timestamp'])
        split_idx = int(len(sorted_interactions) * (1 - test_ratio))
        return sorted_interactions[:split_idx], sorted_interactions[split_idx:]

    @staticmethod
    def build_user_item_matrix(
        interactions: List[Dict],
        interaction_weights: Dict[str, float],
        min_interactions_per_user: int = 3,
        min_interactions_per_item: int = 3,
    ) -> Tuple[csr_matrix, Dict, Dict, Dict, Dict]:
        """
        Build implicit feedback matrix with positivity constraint:
        - Negative and zero weights mapped to small positive 0.01.
        """
        user_item_weights = defaultdict(float)
        for interaction in interactions:
            user_id = interaction['user_id']
            item_id = interaction['job_id']
            itype = interaction['interaction_type']
            weight = interaction_weights.get(itype, 0.0)
            user_item_weights[(user_id, item_id)] += weight

        user_counts = defaultdict(int)
        item_counts = defaultdict(int)
        for (user_id, item_id) in user_item_weights.keys():
            user_counts[user_id] += 1
            item_counts[item_id] += 1

        valid_users = {u for u, c in user_counts.items() if c >= min_interactions_per_user}
        valid_items = {i for i, c in item_counts.items() if c >= min_interactions_per_item}

        filtered_weights = {
            (u, i): w for (u, i), w in user_item_weights.items()
            if u in valid_users and i in valid_items
        }

        unique_users = sorted(valid_users)
        unique_items = sorted(valid_items)
        user_id_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
        item_id_to_index = {iid: idx for idx, iid in enumerate(unique_items)}
        index_to_user_id = {idx: uid for uid, idx in user_id_to_index.items()}
        index_to_item_id = {idx: iid for iid, idx in item_id_to_index.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)
        rows, cols, values = [], [], []

        for (user_id, item_id), weight in filtered_weights.items():
            user_idx = user_id_to_index[user_id]
            item_idx = item_id_to_index[item_id]
            if weight <= 0:
                weight = 0.01
            rows.append(user_idx)
            cols.append(item_idx)
            values.append(weight)

        user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        return (
            user_item_matrix,
            user_id_to_index,
            item_id_to_index,
            index_to_user_id,
            index_to_item_id,
        )

    def train(self, user_item_matrix: csr_matrix) -> None:
        """Train ALS implicit model."""
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            calculate_training_loss=True,
            random_state=self.random_state,
        )
        item_user_matrix = user_item_matrix.T.tocsr()
        self.model.fit(item_user_matrix, show_progress=True)

    def recommend(
        self,
        user_id: int,
        user_id_to_index: Dict[int, int],
        index_to_item_id: Dict[int, int],
        user_item_matrix: csr_matrix,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Recommend top-k items for a user."""
        if self.model is None or user_id not in user_id_to_index:
            return []
        user_idx = user_id_to_index[user_id]
        item_user_matrix = user_item_matrix.T.tocsr()
        ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=item_user_matrix[user_idx],
            N=k,
            filter_already_liked_items=True,
        )
        return [(index_to_item_id[item_idx], float(score)) for item_idx, score in zip(ids, scores)]

    def evaluate(
        self,
        test_interactions: List[Dict],
        user_id_to_index: Dict[int, int],
        item_id_to_index: Dict[int, int],
        index_to_item_id: Dict[int, int],
        user_item_matrix: csr_matrix,
        k: int = 10,
    ) -> Dict[str, float]:
        """Evaluate using Precision@K, Recall@K, NDCG@K on positive interactions."""
        if self.model is None:
            return {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0}

        positive_types = [
            'APPLY', 'SAVE',
            'CLICK_FROM_SEARCH', 'CLICK_FROM_RECOMMENDED', 'CLICK_FROM_SIMILAR',
        ]
        positive_test = [i for i in test_interactions if i['interaction_type'] in positive_types]

        user_test_items = defaultdict(set)
        for interaction in positive_test:
            u = interaction['user_id']
            it = interaction['job_id']
            if u in user_id_to_index and it in item_id_to_index:
                user_test_items[u].add(it)

        precisions, recalls, ndcgs = [], [], []
        for u, true_items in user_test_items.items():
            if not true_items:
                continue
            recs = self.recommend(u, user_id_to_index, index_to_item_id, user_item_matrix, k=k)
            if not recs:
                continue
            rec_items = [iid for iid, _ in recs]
            hits = len(set(rec_items) & true_items)
            precision = hits / k if k > 0 else 0.0
            recall = hits / len(true_items) if len(true_items) > 0 else 0.0
            dcg = sum((1 if item in true_items else 0) / np.log2(i + 2) for i, item in enumerate(rec_items))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

        return {
            'precision': float(np.mean(precisions)) if precisions else 0.0,
            'recall': float(np.mean(recalls)) if recalls else 0.0,
            'ndcg': float(np.mean(ndcgs)) if ndcgs else 0.0,
        }

    @staticmethod
    def save(
        model: "CollaborativeFilteringModel",
        user_id_to_index: Dict[int, int],
        item_id_to_index: Dict[int, int],
        index_to_user_id: Dict[int, int],
        index_to_item_id: Dict[int, int],
        filepath: str = 'models/cf_model.pkl',
    ) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'model': model.model,
            'user_id_to_index': user_id_to_index,
            'item_id_to_index': item_id_to_index,
            'index_to_user_id': index_to_user_id,
            'index_to_item_id': index_to_item_id,
            'params': {
                'factors': model.factors,
                'regularization': model.regularization,
                'iterations': model.iterations,
                'use_gpu': model.use_gpu,
                'random_state': model.random_state,
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
if __name__ == "__main__":
    interactions, interaction_weights = CollaborativeFilteringModel.load_interactions('CFModel\data\cf_interactions.json')
    
    (
        user_item_matrix,
        user_id_to_index,
        item_id_to_index,
        index_to_user_id,
        index_to_item_id,
    ) = CollaborativeFilteringModel.build_user_item_matrix(
        interactions,
        interaction_weights
    )
    
    cf_model = CollaborativeFilteringModel(factors=64, regularization=0.01, iterations=30, use_gpu=False)
    cf_model.train(user_item_matrix)
    CollaborativeFilteringModel.save(cf_model,
        user_id_to_index,
        item_id_to_index,
        index_to_user_id,
        index_to_item_id,
        filepath='CFModel\models\cf_model.pkl',
    )
    evaluate = cf_model.evaluate(
        interactions,
        user_id_to_index,
        item_id_to_index,
        index_to_item_id,
        user_item_matrix,
        k=10,
    )
    print("Evaluation Results:", evaluate)
    