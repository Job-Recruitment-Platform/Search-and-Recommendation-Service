import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from implicit.als import AlternatingLeastSquares
from datetime import datetime


class RetrainService:
    
    DEFAULT_INTERACTION_WEIGHTS = {
        'APPLY': 10.0,
        'SAVE': 6.0,
        'CLICK_FROM_SEARCH': 4.0,
        'CLICK_FROM_RECOMMENDED': 2.5,
        'CLICK_FROM_SIMILAR': 2.0,
        'SKIP_FROM_SEARCH': -0.8,
        'SKIP_FROM_RECOMMENDED': -1.2,
        'SKIP_FROM_SIMILAR': -0.5,
    }
    
    POSITIVE_INTERACTION_TYPES = [
        'APPLY', 'SAVE',
        'CLICK_FROM_SEARCH', 'CLICK_FROM_RECOMMENDED', 'CLICK_FROM_SIMILAR',
    ]
    
    def __init__(
        self,
        base_model_path: str = 'CFModel/models/cf_model.pkl',
        output_model_path: str = 'CFModel/models/retrain_cf_model.pkl',
        interaction_weights: Optional[Dict[str, float]] = None,
    ):
        self.base_model_path = base_model_path
        self.output_model_path = output_model_path
        self.interaction_weights = interaction_weights or self.DEFAULT_INTERACTION_WEIGHTS
        
        self.model: Optional[AlternatingLeastSquares] = None
        self.user_id_to_index: Dict[int, int] = {}
        self.item_id_to_index: Dict[int, int] = {}
        self.index_to_user_id: Dict[int, int] = {}
        self.index_to_item_id: Dict[int, int] = {}
        self.user_item_matrix: Optional[csr_matrix] = None
        self.params: Dict = {}
    
    def _load_base_model(self) -> bool:
        if not os.path.exists(self.base_model_path):
            print(f"Base model not found at {self.base_model_path}")
            return False
        
        with open(self.base_model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.user_id_to_index = data['user_id_to_index']
        self.item_id_to_index = data['item_id_to_index']
        self.index_to_user_id = data['index_to_user_id']
        self.index_to_item_id = data['index_to_item_id']
        self.params = data.get('params', {
            'factors': 64,
            'regularization': 0.01,
            'iterations': 30,
            'use_gpu': False,
            'random_state': 42,
        })
        
        print(f"Loaded base model from {self.base_model_path}")
        print(f"  - Users: {len(self.user_id_to_index)}, Items: {len(self.item_id_to_index)}")
        return True
    
    def _read_csv_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        
        required_columns = ['user_id', 'job_id', 'interaction_type', 'timestamp']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        print(f"Read {len(df)} interactions from {csv_path}")
        return df
    
    def _build_incremental_matrix(
        self, 
        new_interactions: pd.DataFrame
    ) -> Tuple[csr_matrix, Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
        user_item_weights = defaultdict(float)
        for _, row in new_interactions.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['job_id'])
            interaction_type = row['interaction_type']
            weight = self.interaction_weights.get(interaction_type, 0.0)
            user_item_weights[(user_id, item_id)] += weight
        
        existing_users = set(self.user_id_to_index.keys())
        existing_items = set(self.item_id_to_index.keys())
        
        new_user_ids = set()
        new_item_ids = set()
        for (user_id, item_id) in user_item_weights.keys():
            if user_id not in existing_users:
                new_user_ids.add(user_id)
            if item_id not in existing_items:
                new_item_ids.add(item_id)
        
        user_id_to_index = dict(self.user_id_to_index)
        index_to_user_id = dict(self.index_to_user_id)
        next_user_idx = len(user_id_to_index)
        for uid in sorted(new_user_ids):
            user_id_to_index[uid] = next_user_idx
            index_to_user_id[next_user_idx] = uid
            next_user_idx += 1
        
        item_id_to_index = dict(self.item_id_to_index)
        index_to_item_id = dict(self.index_to_item_id)
        next_item_idx = len(item_id_to_index)
        for iid in sorted(new_item_ids):
            item_id_to_index[iid] = next_item_idx
            index_to_item_id[next_item_idx] = iid
            next_item_idx += 1
        
        n_users = len(user_id_to_index)
        n_items = len(item_id_to_index)
        
        rows, cols, values = [], [], []
        for (user_id, item_id), weight in user_item_weights.items():
            user_idx = user_id_to_index[user_id]
            item_idx = item_id_to_index[item_id]
            if weight <= 0:
                weight = 0.01
            rows.append(user_idx)
            cols.append(item_idx)
            values.append(weight)
        
        new_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )
        
        print(f"Built incremental matrix: {n_users} users x {n_items} items")
        print(f"  - New users: {len(new_user_ids)}, New items: {len(new_item_ids)}")
        
        return new_matrix, user_id_to_index, item_id_to_index, index_to_user_id, index_to_item_id
    
    def retrain(
        self, 
        csv_path: str,
        incremental_iterations: int = 10,
    ) -> Dict:
        if not self._load_base_model():
            return {'success': False, 'error': 'Failed to load base model'}
        
        try:
            new_data = self._read_csv_data(csv_path)
        except Exception as e:
            return {'success': False, 'error': f'Failed to read CSV: {str(e)}'}
        
        if len(new_data) == 0:
            return {'success': False, 'error': 'No interactions in CSV file'}
        
        (
            new_matrix,
            user_id_to_index,
            item_id_to_index,
            index_to_user_id,
            index_to_item_id,
        ) = self._build_incremental_matrix(new_data)
        
        old_n_users, old_n_items = len(self.user_id_to_index), len(self.item_id_to_index)
        new_n_users, new_n_items = len(user_id_to_index), len(item_id_to_index)
        
        factors = self.params.get('factors', 64)
        incremental_model = AlternatingLeastSquares(
            factors=factors,
            regularization=self.params.get('regularization', 0.01),
            iterations=incremental_iterations,
            use_gpu=self.params.get('use_gpu', False),
            calculate_training_loss=True,
            random_state=self.params.get('random_state', 42),
        )
        
        if self.model is not None and hasattr(self.model, 'user_factors') and hasattr(self.model, 'item_factors'):
            old_user_factors = self.model.user_factors
            new_user_factors = np.random.normal(0, 0.01, (new_n_users, factors)).astype(np.float32)
            new_user_factors[:old_n_users, :] = old_user_factors[:old_n_users, :]
            
            old_item_factors = self.model.item_factors
            new_item_factors = np.random.normal(0, 0.01, (new_n_items, factors)).astype(np.float32)
            new_item_factors[:old_n_items, :] = old_item_factors[:old_n_items, :]
            
            item_user_matrix = new_matrix.T.tocsr()
            incremental_model.fit(item_user_matrix, show_progress=True)
            
            blend_factor = 0.7
            if hasattr(incremental_model, 'user_factors'):
                incremental_model.user_factors[:old_n_users, :] = (
                    blend_factor * new_user_factors[:old_n_users, :] +
                    (1 - blend_factor) * incremental_model.user_factors[:old_n_users, :]
                )
            if hasattr(incremental_model, 'item_factors'):
                incremental_model.item_factors[:old_n_items, :] = (
                    blend_factor * new_item_factors[:old_n_items, :] +
                    (1 - blend_factor) * incremental_model.item_factors[:old_n_items, :]
                )
        else:
            item_user_matrix = new_matrix.T.tocsr()
            incremental_model.fit(item_user_matrix, show_progress=True)
        
        self.model = incremental_model
        self.user_id_to_index = user_id_to_index
        self.item_id_to_index = item_id_to_index
        self.index_to_user_id = index_to_user_id
        self.index_to_item_id = index_to_item_id
        self.user_item_matrix = new_matrix
        
        self._save_model()
        
        return {
            'success': True,
            'old_users': old_n_users,
            'old_items': old_n_items,
            'new_users': new_n_users,
            'new_items': new_n_items,
            'added_users': new_n_users - old_n_users,
            'added_items': new_n_items - old_n_items,
            'interactions_processed': len(new_data),
            'output_path': self.output_model_path,
        }
    
    def _save_model(self) -> None:
        os.makedirs(os.path.dirname(self.output_model_path), exist_ok=True)
        
        data = {
            'model': self.model,
            'user_id_to_index': self.user_id_to_index,
            'item_id_to_index': self.item_id_to_index,
            'index_to_user_id': self.index_to_user_id,
            'index_to_item_id': self.index_to_item_id,
            'params': self.params,
            'retrained_at': datetime.now().isoformat(),
        }
        
        with open(self.output_model_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved retrained model to {self.output_model_path}")
    
    # for test
    def _recommend(
        self,
        user_id: int,
        k: int = 10,
    ) -> List[Tuple[int, float]]:
        if self.model is None or user_id not in self.user_id_to_index:
            return []
        
        user_idx = self.user_id_to_index[user_id]
        item_user_matrix = self.user_item_matrix.T.tocsr()
        
        ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=item_user_matrix[user_idx],
            N=k,
            filter_already_liked_items=True,
        )
        
        return [(self.index_to_item_id[item_idx], float(score)) for item_idx, score in zip(ids, scores)]
    
    def evaluate(
        self,
        csv_path: str,
        k: int = 10,
    ) -> Dict[str, float]:
        if self.model is None:
            if os.path.exists(self.output_model_path):
                with open(self.output_model_path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.user_id_to_index = data['user_id_to_index']
                self.item_id_to_index = data['item_id_to_index']
                self.index_to_user_id = data['index_to_user_id']
                self.index_to_item_id = data['index_to_item_id']
            else:
                return {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0, 'error': 'No model loaded'}
        
        try:
            test_data = self._read_csv_data(csv_path)
        except Exception as e:
            return {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0, 'error': f'Failed to read CSV: {str(e)}'}
        
        positive_test = test_data[test_data['interaction_type'].isin(self.POSITIVE_INTERACTION_TYPES)]
        
        user_test_items: Dict[int, set] = defaultdict(set)
        for _, row in positive_test.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['job_id'])
            if user_id in self.user_id_to_index and item_id in self.item_id_to_index:
                user_test_items[user_id].add(item_id)
        
        if not user_test_items:
            return {
                'precision': 0.0, 
                'recall': 0.0, 
                'ndcg': 0.0, 
                'error': 'No valid test interactions found',
            }
        
        if self.user_item_matrix is None:
            n_users = len(self.user_id_to_index)
            n_items = len(self.item_id_to_index)
            self.user_item_matrix = csr_matrix((n_users, n_items), dtype=np.float32)
        
        precisions, recalls, ndcgs = [], [], []
        
        for user_id, true_items in user_test_items.items():
            if not true_items:
                continue
            
            recs = self._recommend(user_id, k=k)
            if not recs:
                continue
            
            rec_items = [item_id for item_id, _ in recs]
            
            hits = len(set(rec_items) & true_items)
            precision = hits / k if k > 0 else 0.0
            recall = hits / len(true_items) if len(true_items) > 0 else 0.0
            
            dcg = sum(
                (1.0 if item in true_items else 0.0) / np.log2(i + 2)
                for i, item in enumerate(rec_items)
            )
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)
        
        return {
            'precision': float(np.mean(precisions)) if precisions else 0.0,
            'recall': float(np.mean(recalls)) if recalls else 0.0,
            'ndcg': float(np.mean(ndcgs)) if ndcgs else 0.0,
            'users_evaluated': len(precisions),
            'k': k,
        }


if __name__ == "__main__":
    service = RetrainService(
        base_model_path='CFModel/models/cf_model.pkl',
        output_model_path='CFModel/models/retrain_cf_model.pkl',
    )
