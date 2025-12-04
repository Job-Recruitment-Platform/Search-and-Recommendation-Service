"""Pre-download BGE-M3 model on first run"""
import os
import logging

logger = logging.getLogger(__name__)

def ensure_model_downloaded():
    """Download BGE-M3 model if not exists"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_path = os.path.join(cache_dir, "models--BAAI--bge-m3")
    
    if os.path.exists(model_path):
        logger.info("✓ BGE-M3 model already downloaded")
        return
    
    try:
        logger.info("⏳ Downloading BGE-M3 model (first run, ~2GB, may take 3-5 minutes)...")
        
        # Correct API: BGEM3FlagModel() takes model_name_or_path as first positional arg
        from FlagEmbedding import BGEM3FlagModel
        
        ef = BGEM3FlagModel(
            "BAAI/bge-m3",  # model_name_or_path (positional)
            use_fp16=False,
            device="cpu"
        )
        
        logger.info("✓ BGE-M3 model downloaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to download BGE-M3 model: {e}")
        raise