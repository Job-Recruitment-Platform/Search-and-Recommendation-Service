## Kiến trúc hệ thống gợi ý (hiện tại)

### Tổng quan
Hệ thống gợi ý đang vận hành theo kiến trúc hybrid 2 giai đoạn:

1. **Candidate Generation** – kết hợp đa nguồn để tăng coverage:
   - **CF (Collaborative Filtering)** dựa trên mô hình ALS implicit (đã train và lưu dạng pickle).
   - **Content-based** dùng vector người dùng (tổng hợp từ hồ sơ và tương tác gần nhất) để truy vấn Milvus.
   - **Popular fallback** bổ sung khi thiếu ứng viên (hỗ trợ cold start).

2. **Ranking & Exploration** – xếp hạng và áp dụng bonus khám phá:
   - Trọng số nguồn (65% CF, 35% content, 25% popular nếu chỉ có popular).
   - Boost cho ứng viên xuất hiện ở nhiều nguồn.
  - Exploration bonus (deterministic) cho job chưa từng tương tác; giảm điểm cho job đã xem.
  - Kết quả cuối cùng chỉ trả về danh sách `job_id`.

### Thành phần chính

- **Recommendation service** (`services/recommend.py`)
  - Hàm chính `RecommendationService.recommend(user_id, top_k, filters=None)` → pipeline: `_generate_candidates` → `_rank_candidates`.
  - `_generate_candidates` thu gom ứng viên từ CF, content, popular và merge thành map duy nhất.
  - `_rank_candidates` tính điểm hybrid + exploration, trả về top-K `job_id`.
  - Cache ngắn hạn: Redis lưu short-term user vector để giảm chi phí tái tính.

- **Milvus service** (`services/milvus_service.py`)
  - Quản lý kết nối Milvus, schema collection `jobs`, `users`.
  - Cung cấp hàm `search`, `get_job_dense_vector`, `upsert_jobs`, `upsert_user_vector`, `generate_embeddings`.

- **CF model layer** (`CFModel/cf_model.py`)
  - Lớp `CollaborativeFilteringModel` bao gói train/inference ALS implicit.
  - Xử lý load dữ liệu tương tác, build ma trận implicit (chuyển mọi giá trị âm/0 thành 0.01), train, evaluate, lưu model/mapping.
  - Model pickle được nạp thông qua `Config.CF_MODEL_PATH`.

- **Sync pipeline** (`sync_service/sync_processor.py`, `utils/data_processor.py`)
  - Nhận sự kiện việc làm, sinh embedding (dense từ description, sparse từ title/skills/location) và upsert vào Milvus.
  - Bảo đảm schema `jobs` luôn đồng bộ với nguồn dữ liệu domain.

- **Cấu hình** (`app/config.py`)
  - Chứa config Milvus, Redis, embedding, search.
  - `CF_MODEL_PATH`, `INTERACTION_WEIGHTS`, `INTERACTION_HALF_LIFE_DAYS`.

### Candidate Generation (chi tiết)

- **CF candidates**
  - `_generate_cf_candidates`: nạp mapping user → index, gọi `ALS.recommend`.
  - Nếu user chưa có vector trong mô hình, bỏ qua nguồn CF.

- **Content candidates**
  - `_generate_content_candidates`: xây user vector (long-term + short-term), truy vấn Milvus chỉ với output `id`.
  - Không áp filter phức tạp; tập trung lấy ứng viên tương tự nhất.

- **Popular candidates**
  - `_generate_popular_candidates`: chuẩn hoá danh sách job phổ biến (từ stub hoặc nguồn thực tế).

### Ranking & Exploration

- `_rank_candidates`:
  - Chuẩn hoá điểm CF (logistic) và content (scale [-1, 1] → [0, 1]).
  - Kết hợp trọng số theo nguồn; bonus multi-source.
  - Exploration bonus (hash-based) cho job chưa tương tác; giảm điểm job đã xem.
  - Sort theo điểm giảm dần, trả về top-K dạng `[{ "job_id": ... }]`.

### Dữ liệu & lưu trữ

- **Milvus – collection `jobs`**
  - Schema: metadata + `dense_vector` + `sparse_vector`.
  - Index: HNSW (COSINE) cho dense, SPARSE_INVERTED_INDEX (IP) cho sparse.

- **Embeddings**
  - BGE-M3 (`BGEM3EmbeddingFunction`) trả về cả dense & sparse.
  - Sync pipeline chuyển sparse matrix sang dict, đảm bảo phù hợp lược đồ Milvus.

- **Redis**
  - Cache vector hành vi ngắn hạn (key `user_vector:short_term:{user_id}`) dạng JSON.

- **Artefact CF**
  - File pickle chứa mô hình ALS, mapping user/item, tham số train.

### Cấu hình liên quan

- `Config`: Milvus, Redis, embedding, search defaults, CF model path.
- `INTERACTION_WEIGHTS`: trọng số tương tác (APPLY, SAVE, CLICK, SKIP…).
- `INTERACTION_HALF_LIFE_DAYS`: điều khiển decay theo thời gian.

### Điểm mở rộng

- Hợp nhất candidate & ranking bằng ranker học máy / learning-to-rank.
- Làm giàu metadata cho kết quả (nếu UI cần) thông qua job service.
- Áp dụng filter nội dung (location, role, salary) khi cần.
- Cá nhân hoá trọng số nguồn, exploration theo từng user.

### Độ tin cậy & fallback

- Redis lỗi → bỏ qua cache, tính lại vector ngắn hạn.
- Không nạp được mô hình CF → hệ thống tự động chuyển sang content-based + popular.
- Milvus lỗi/timeout → fallback sang danh sách popular.

### Tóm tắt

Kiến trúc hiện tại tạo danh sách ứng viên kết hợp CF + content + popular, sau đó xếp hạng với chiến lược hybrid và exploration để tăng coverage/cold-start. Kết quả trả về tối giản (chỉ `job_id`), giúp service downstream quyết định fetch thêm metadata nếu cần. Hệ thống giữ cấu trúc linh hoạt để mở rộng re-ranking, filter nâng cao và cá nhân hoá trong tương lai.
## Recommendation Architecture (Current System)

### Overview
The recommendation stack combines a collaborative filtering (CF) pipeline with a content-based vector search fallback. Requests are served by `RecommendationService`, which first attempts CF recommendations (if a trained model and user mappings are available). If CF is unavailable or the user is unseen, it computes a dense user vector from profile and recent interactions and queries Milvus for similar jobs.

### Components

- Recommendation service (`services/recommend.py`)
  - Entry point `RecommendationService.recommend(user_id, top_k, filters)`
  - CF integration: lazy-loads a pickled ALS model, user/item mappings; produces CF recs when possible
  - Content-based fallback: builds a user dense vector, queries Milvus, formats top results
  - Short-term caching: uses Redis to cache short-term user vectors

- Vector DB (`services/milvus_service.py`)
  - Manages Milvus connection and `jobs` collection
  - Embeddings via BGE-M3 (`BGEM3EmbeddingFunction`) returning dense and sparse vectors
  - Hybrid search support (dense + sparse) and vector retrieval helpers (e.g., `get_job_dense_vector`)
  - Upserts for job entities and user long-term vectors

- CF model (`CFModel/cf_model.py`)
  - `CollaborativeFilteringModel` class wraps implicit ALS training/inference
  - Utilities to load interactions JSON, build user-item matrix (implicit, all-positive), split by time, evaluate, and save model with mappings
  - Persisted model path configured by `Config.CF_MODEL_PATH`

- Sync pipeline (`sync_service/sync_processor.py`, `utils/data_processor.py`)
  - Consumes job events, generates embeddings (dense from description, sparse from title/skills/location)
  - Builds Milvus entities aligned with `jobs` schema and upserts

- Configuration (`app/config.py`)
  - Hosts service ports, Redis, Milvus, embedding model, search defaults
  - `CF_MODEL_PATH` for CF model loading
  - `INTERACTION_WEIGHTS` used when computing behavior vectors

### Request Flow

1. CF-first attempt
   - On `RecommendationService` init, `_load_cf_model()` tries to load pickled ALS model and mappings (`user_id_to_index`, `index_to_item_id`) from `CF_MODEL_PATH`.
   - In `recommend()`, `_recommend_cf(user_id, top_k)` runs first:
     - If user exists in mappings, calls ALS `.recommend()` with an empty user row to obtain top-N item indices.
     - Maps item indices → `job_id` and returns `[{ job_id, score, source: "cf" }]`.
     - If CF unavailable or user unseen, proceeds to fallback.

2. Content-based fallback
   - Fetch user profile and interactions via `_get_user_profile()` and `_get_user_interactions()` (stubs to integrate with your user store).
   - Build user vector:
     - Long-term vector from profile text (`_calculate_long_term_user_vector` → embed + normalize). Persisted to Milvus with `upsert_user_vector`.
     - Short-term vector from recent interactions (`_calculate_short_term_user_vector`):
       - Cache lookup in Redis (`user_vector:short_term:{user_id}`); decode bytes to JSON when present.
       - If miss: compute behavior vector `_compute_behavior_dense(...)`:
         - Iterates allowed interaction types; uses `INTERACTION_WEIGHTS` and exponential time decay with half-life from config.
         - For each interacted job, fetch dense job vector (`get_job_dense_vector`) and accumulate weighted/decayed contributions; normalize.
       - Cache back to Redis with TTL (default 3600s).
     - Combine long-term and short-term with adaptive weights by interaction volume:
       - <5: 90% profile, 10% behavior
       - 5–20: 60% profile, 40% behavior
       - >20: 30% profile, 70% behavior
     - Normalize final vector.
   - Search Milvus:
     - Query `jobs` collection using the dense vector; `search_limit = top_k * 3` to get candidates for simple re-ranking.
     - Apply optional filters via `_build_filter_expr(filters)` (hook point).
     - Format top `top_k` results with metadata when available; tag `source: "content_based"`.

### Data and Storage

- Milvus `jobs` collection (see `services/milvus_service.py`):
  - Schema includes scalar job metadata and vector fields:
    - `dense_vector` (FLOAT_VECTOR, dim from BGE-M3 dense output)
    - `sparse_vector` (SPARSE_FLOAT_VECTOR) from title/skills/location
  - Indexes:
    - Dense: HNSW with COSINE
    - Sparse: SPARSE_INVERTED_INDEX with IP
  - Upsert path writes job records and vectors; query path supports hybrid search and vector fetch.

- Embeddings:
  - Generated by `BGEM3EmbeddingFunction` (`return_dense=True`, `return_sparse=True`).
  - Sync processor composes dense from `description` and sparse from title/skills/location (COO→dict normalization handled in builders).

- Redis:
  - Short-term user vector cache keyed by `user_vector:short_term:{user_id}` storing JSON of normalized dense vector.

- CF artifacts:
  - Pickle file contains `model` (ALS), mappings, and params.
  - Loaded at runtime when present; otherwise service gracefully falls back to content-based mode.

### Configuration

- `Config`:
  - Redis: `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`
  - Milvus: `MILVUS_HOST`, `MILVUS_PORT`
  - Embeddings: `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE`, `EMBEDDING_USE_FP16`
  - Search: `SEARCH_DEFAULT_LIMIT`, `SEARCH_DEFAULT_OFFSET`, `SEARCH_THRESHOLD`
  - Recommendation: `CF_MODEL_PATH`

- `INTERACTION_WEIGHTS`:
  - Positive: APPLY (1.0), SAVE (0.6), CLICK_FROM_SEARCH (0.4), CLICK_FROM_RECOMMENDED (0.25), CLICK_FROM_SIMILAR (0.2)
  - Negative: SKIP_* with small negative weights (ignored by CF at training time where all values are mapped positive; used for behavior vector with decay)

### Extension Points

- CF + Content Hybrid
  - Merge CF and content-based candidates with a learned ranker; currently CF precedes and returns directly if available.

- Metadata Enrichment
  - For CF-only outputs, fetch job metadata (title/company/location) via a job service/DB for richer responses.

- Filters
  - Implement `_build_filter_expr(filters)` to translate filter inputs to Milvus boolean expressions (e.g., location, role, salary).

- Relevance Feedback
  - Adjust `INTERACTION_WEIGHTS`, half-life, and combination weights; consider per-user personalization of decay and weights.

- Cold Start
  - Backfill with popular or role-similar jobs when both CF and vectors are unavailable.

### Error Handling and Resilience

- Redis failures are logged and bypassed (no hard dependency).
- CF load failures are logged; service operates in content-based mode.
- Milvus queries are wrapped; on exceptions, service falls back to popular jobs.

### Summary

The system prioritizes CF when a trained ALS model and mappings exist, otherwise falls back to a robust vector-based approach that blends long-term user profile signals with short-term interaction behavior, cached for performance. Milvus stores job vectors and supports hybrid retrieval. The architecture is modular to support future hybrid ranking, better filtering, and richer metadata enrichment. 
Search Flow:
User Query → BGE-M3 embedding → Hybrid Search (dense+sparse) → Results

Recommendation Flow:  
User ID → User Vector → Dense Search → Personalized Results
```

---

### **Collections trong Milvus:**

1. **items_collection** (dùng chung cho cả search và recommend)
   - dense_vector + sparse_vector
   - metadata

2. **users_collection** (riêng cho recommend)
   - user_id → preference_vector (dense only)

3. **interactions_collection** (optional - có thể dùng PostgreSQL/MongoDB)
   - user-item interactions để build user vector

---

### **Flow hoạt động:**

**Offline (batch job - chạy định kỳ):**
```
1. Collect user interactions (view/click/purchase)
2. Build user vector = weighted_avg(interacted_items_vectors)
3. Upsert vào users_collection
```

**Online (real-time):**
```
GET /recommend/{user_id}
  → Query users_collection lấy user_vector
  → Search items_collection với user_vector (dense only)
  → Filter items đã xem
  → Return top-N
```

**Cold start:**
```
- User mới: recommend trending/popular items
- Sau vài interactions: update user_vector real-time