import os
import logging
import time
import requests
import sys
import schedule

# Đảm bảo Python tìm thấy các package trong thư mục hiện tại
sys.path.append(os.getcwd())

from app.config import Config
from CFModel.cf_model import CollaborativeFilteringModel
from utils.db_loader import load_interactions_from_db

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def train_pipeline():
    logger.info("=== Starting Retraining Pipeline ===")
    
    # 1. Load Data
    try:
        data = load_interactions_from_db()
        interactions = data.get('interactions', [])
        weights = data.get('metadata', {}).get('interaction_weights', {})
        
        if not interactions:
            logger.warning("No interactions found in Database. Skipping training.")
            return
        
        if len(interactions) < 100:
            logger.warning(f"Only {len(interactions)} interactions. Need at least 100. Skipping.")
            return
            
        logger.info(f"Loaded {len(interactions)} interactions from DB.")
    except Exception as e:
        logger.error(f"Failed to load data from DB: {e}")
        return

    # 2. Build Matrix
    logger.info("Building User-Item Matrix...")
    try:
        (
            matrix,
            user_map,
            item_map,
            inv_user_map,
            inv_item_map
        ) = CollaborativeFilteringModel.build_user_item_matrix(
            interactions, 
            weights,
            min_interactions_per_user=3,
            min_interactions_per_item=3
        )
        
        if matrix.nnz == 0:
            logger.error("Empty matrix after filtering. Cannot train.")
            return
            
        logger.info(f"Matrix: {matrix.shape}, non-zero: {matrix.nnz}")
        
    except Exception as e:
        logger.error(f"Error building matrix: {e}", exc_info=True)
        return
    
    # 3. Train Model
    logger.info("Training ALS Model...")
    try:
        model = CollaborativeFilteringModel(
            factors=64, 
            regularization=0.01, 
            iterations=30, 
            use_gpu=False
        )
        model.train(matrix)
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return
    
    # 4. Save Model (Atomic Save)
    save_path = getattr(Config, 'CF_MODEL_PATH', 'CFModel/models/cf_model.pkl')
    temp_path = save_path + ".tmp"
    
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        logger.info(f"Saving model to {save_path}...")
        CollaborativeFilteringModel.save(
            model, user_map, item_map, inv_user_map, inv_item_map, filepath=temp_path
        )
        
        if not os.path.exists(temp_path):
            raise FileNotFoundError(f"Temp file not created: {temp_path}")
        
        if os.path.getsize(temp_path) == 0:
            raise ValueError("Saved model file is empty")
        
        # Rename file (Atomic operation)
        if os.path.exists(save_path):
            backup_path = save_path + ".backup"
            os.rename(save_path, backup_path)
        os.rename(temp_path, save_path)
        logger.info("Model saved successfully.")
    except Exception as e:
        logger.error(f"Error saving model: {e}", exc_info=True)
        # Restore backup if exists
        backup_path = save_path + ".backup"
        if os.path.exists(backup_path):
            os.rename(backup_path, save_path)
            logger.info("Restored backup model")
        return
    
    # 5. Trigger Hot Reload
    try:
        api_url = f"http://{Config.API_SERVER_HOST}:{Config.FLASK_PORT}/internal/reload-model"
        headers = {"X-Internal-Token": getattr(Config, "INTERNAL_API_TOKEN", "")}
        
        logger.info(f"Triggering reload at {api_url}...")
        resp = requests.post(api_url, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            logger.info("✅ Service reloaded new model successfully.")
        else:
            logger.warning(f"Service reload returned {resp.status_code}: {resp.text}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not trigger reload (Service might be down): {e}")
    except Exception as e:
        logger.error(f"Unexpected error in reload trigger: {e}", exc_info=True)
    
    logger.info("=== Retraining Pipeline Completed ===")
    
def run_scheduler():
    schedule_time = getattr(Config, 'TRAINING_SCHEDULE_TIME', '02:00')
    schedule.every().day.at(schedule_time).do(train_pipeline)

    logger.info(f"Scheduler started. Training scheduled at: {schedule_time}")
    logger.info("Next run at: %s", schedule.next_run())

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    logger.info("🚀 CF Model Training Worker Starting...")
    logger.info(f"Using model path: {Config.CF_MODEL_PATH}")
    logger.info(f"PostgreSQL: {Config.POSTGRES_HOST}:{Config.POSTGRES_PORT}/{Config.POSTGRES_DB}")
    
    # Run scheduler
    run_scheduler()