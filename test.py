from pymilvus import connections, Collection


# ======================
# CONFIG
# ======================
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "jobs"
BATCH_SIZE = 1000


# ======================
# FORMATTERS
# ======================
def format_job_response(row: dict) -> dict:
    """
    Format 1 record Milvus -> API Response (KHÔNG trả vector)
    """
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "company": row.get("company"),
        "job_role": row.get("job_role"),
        "location": row.get("location"),
        "work_mode": row.get("work_mode"),
        "seniority": row.get("seniority"),
        "status": row.get("status"),
        "salary": {
            "min": row.get("salary_min"),
            "max": row.get("salary_max"),
            "currency": row.get("currency"),
        },
        "skills": (
            [s.strip() for s in row["skills"].split(",")]
            if row.get("skills")
            else []
        ),
        "max_candidates": row.get("max_candidates"),
        "date_posted": row.get("date_posted"),
        "date_expires": row.get("date_expires"),
        "description": row.get("description"),
    }


# ======================
# FETCH ALL DATA
# ======================
def fetch_all_jobs() -> list[dict]:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    col = Collection(COLLECTION_NAME)
    col.load()

    pk_field = col.schema.primary_field.name
    offset = 0

    formatted_results = []

    while True:
        rows = col.query(
            expr=f"{pk_field} >= 0",
            offset=offset,
            limit=BATCH_SIZE,
            output_fields=["*"],
        )

        if not rows:
            break

        for row in rows:
            formatted_results.append(format_job_response(row))

        offset += BATCH_SIZE

    return formatted_results


# ======================
# RUN TEST
# ======================
if __name__ == "__main__":
    jobs = fetch_all_jobs()
    print(f"Total jobs fetched: {len(jobs)}")

    # Print sample
    if jobs:
        from pprint import pprint
        for job in jobs:
            pprint(job)
