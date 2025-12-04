# utils/db_loader.py
import psycopg2
import psycopg2.extras
import logging
from app.config import Config, INTERACTION_WEIGHTS

logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            database=Config.POSTGRES_DB,
            user=Config.POSTGRES_USERNAME,
            password=Config.POSTGRES_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def load_interactions_from_db():
    """
    Query dữ liệu từ bảng user_interactions và trả về định dạng
    khớp với input của CollaborativeFilteringModel.
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        # Query lấy dữ liệu
        # Lưu ý: Chuyển đổi occurred_at sang timestamp float để tính toán time decay dễ hơn
        # và ép kiểu event_type sang text nếu nó là ENUM trong Postgres
        query = """
            SELECT 
                account_id as user_id,
                job_id,
                event_type::text as interaction_type,
                EXTRACT(EPOCH FROM occurred_at) as timestamp
            FROM user_interactions
            WHERE 
                account_id IS NOT NULL 
                AND job_id IS NOT NULL
        """
        
        logger.info("Executing query to fetch interactions...")
        cursor.execute(query)
        rows = cursor.fetchall()
        
        logger.info(f"Fetched {len(rows)} interaction records from DB.")
        
        # Format dữ liệu trả về
        data = {
            "metadata": {
                "interaction_weights": INTERACTION_WEIGHTS
            },
            "interactions": [dict(row) for row in rows]
        }
        
        return data

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return {"metadata": {}, "interactions": []}
        
    finally:
        cursor.close()
        conn.close()
