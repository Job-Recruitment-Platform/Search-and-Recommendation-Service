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