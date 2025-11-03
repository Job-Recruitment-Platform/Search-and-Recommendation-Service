"""Utility script to list all job records (id, title) from Milvus"""
import logging
from services.milvus_service import MilvusService


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    svc = MilvusService()
    coll = svc.jobs_collection

    # Fetch all rows; for large datasets consider batching
    rows = coll.query(expr="id >= 0", output_fields=["id", "title"])  # type: ignore

    print(f"Total rows: {len(rows)}")
    for row in rows:
        print(f"id={row.get('id')}, title={row.get('title')}")


if __name__ == "__main__":
    main()

