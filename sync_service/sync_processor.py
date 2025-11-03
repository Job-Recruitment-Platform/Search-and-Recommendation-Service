"""Sync processor for handling job updates from outbox events"""
import logging
import json
from typing import Dict, Any
from services.milvus_service import MilvusService
from utils.data_processor import DataProcessor
from models.job import Job
from models.event import OutboxEvent
from models.sync import SyncResult

logger = logging.getLogger(__name__)


class SyncProcessor:
    """Processes job sync operations from Redis outbox events"""

    def __init__(self, milvus_service: MilvusService):
        self.milvus_service = milvus_service

    def sync_to_milvus(self, payload: Dict[str, Any]) -> SyncResult:
        """Sync job to Milvus (for CREATED/UPDATED events). Returns a SyncResult."""
        # Normalize payload -> list of job dicts
        if isinstance(payload, dict) and "id" in payload:
            jobs_data = [payload]
        elif isinstance(payload, list):
            jobs_data = payload
        else:
            logger.error(f"Invalid payload format: {type(payload)}")
            return SyncResult(processed=0, inserted=0, deleted=0, error="invalid_payload")

        try:
            jobs = [Job.from_dict(job_data) for job_data in jobs_data]
            jobs_dict = [job.to_dict() for job in jobs]

            combined_texts = [DataProcessor.combine_job_text(j) for j in jobs_dict]
            embeddings = self.milvus_service.generate_embeddings(combined_texts)

            entities = DataProcessor.build_entities(
                dense_vecs=embeddings.get("dense", []),
                sparse_vecs=embeddings.get("sparse", []),
                jobs=jobs_dict,
            )

            upserted = self.milvus_service.upsert_jobs(entities)
            logger.info(f"Synced to Milvus: {len(jobs)} jobs, upserted={upserted}")
            return SyncResult(processed=len(jobs), inserted=upserted, deleted=0)
        except Exception as e:
            logger.exception(f"Failed to sync to Milvus: {e}")
            return SyncResult(processed=0, inserted=0, deleted=0, error=str(e))

    def delete_from_milvus(self, aggregate_id: str) -> SyncResult:
        """Delete job from Milvus (for DELETED events)"""
        try:
            job_id = int(aggregate_id)
        except ValueError:
            logger.error(f"Invalid aggregate_id format: {aggregate_id}")
            return SyncResult(processed=0, inserted=0, deleted=0, error="invalid_aggregate_id")

        try:
            deleted = self.milvus_service.delete_jobs([job_id])
            logger.info(f"Deleted from Milvus: job_id={job_id}, deleted={deleted}")
            return SyncResult(processed=1, inserted=0, deleted=deleted)
        except Exception as e:
            logger.exception(f"Failed to delete from Milvus: {e}")
            return SyncResult(processed=0, inserted=0, deleted=0, error=str(e))

    def process_stream_message(self, fields: Dict[str, str]) -> SyncResult:
        """
        Process a message from Redis outbox stream
        
        Expected fields (8 total):
        - id: Outbox event ID
        - aggregateType: Aggregate type (e.g., "JOB")
        - aggregateId: Aggregate ID (job ID)
        - eventType: Event type ("CREATED", "UPDATED", "DELETED")
        - payload: JSON string with full entity data
        - occurredAt: Timestamp when event occurred
        - traceId: UUID for tracing
        - attempts: Number of retry attempts
        """
        try:
            # Parse event from Redis fields
            event = OutboxEvent.from_redis_fields(fields)

            # Log event metadata
            logger.info(
                f"📥 Received event - id={event.id}, aggregateType={event.aggregate_type.value}, "
                f"aggregateId={event.aggregate_id}, eventType={event.event_type.value}, "
                f"traceId={event.trace_id}, occurredAt={event.occurred_at}, attempts={event.attempts}"
            )
            logger.debug(f"All message fields: {list(fields.keys())}")

            # Filter: Only process JOB aggregate type
            if not event.is_job_event():
                logger.info(
                    f"⏭️  Skipping event: aggregateType='{event.aggregate_type.value}' "
                    f"(only processing JOB events, eventId={event.id})"
                )
                return SyncResult(processed=0, inserted=0, deleted=0)

            # Handle events based on type
            if event.is_created_or_updated():
                # Parse payload for CREATED/UPDATED events
                if not event.payload_str:
                    logger.error(
                        f"Missing payload for {event.event_type.value} event: aggregateId={event.aggregate_id}"
                    )
                    return SyncResult(processed=0, inserted=0, deleted=0)

                try:
                    payload = json.loads(event.payload_str)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse payload JSON for {event.event_type.value} event: {e}, "
                        f"aggregateId={event.aggregate_id}"
                    )
                    return SyncResult(processed=0, inserted=0, deleted=0, error=str(e))

                logger.info(
                    f"Processing {event.event_type.value} event: aggregateId={event.aggregate_id}, traceId={event.trace_id}"
                )
                result = self.sync_to_milvus(payload)
                logger.info(
                    f"Completed {event.event_type.value} event: aggregateId={event.aggregate_id}, traceId={event.trace_id}, result={result.to_dict()}"
                )
                return result

            elif event.is_deleted():
                # For DELETED events, only need aggregateId
                if not event.aggregate_id:
                    logger.error(
                        f"Missing aggregateId for DELETED event: event_id={event.id}"
                    )
                    return SyncResult(processed=0, inserted=0, deleted=0)

                logger.info(
                    f"Processing DELETED event: aggregateId={event.aggregate_id}, "
                    f"traceId={event.trace_id}"
                )
                result = self.delete_from_milvus(event.aggregate_id)
                logger.info(
                    f"Completed DELETED event: aggregateId={event.aggregate_id}, "
                    f"traceId={event.trace_id}, result={result.to_dict()}"
                )
                return result

            else:
                logger.warning(
                    f"Unknown event type: {event.event_type.value}, aggregateId={event.aggregate_id}"
                )
                return SyncResult(processed=0, inserted=0, deleted=0)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse payload JSON: {e}, fields={fields}")
            return SyncResult(processed=0, inserted=0, deleted=0, error=str(e))
        except Exception as e:
            logger.exception(f"Failed to process stream message: {e}, fields={fields}")
            return SyncResult(processed=0, inserted=0, deleted=0, error=str(e))


