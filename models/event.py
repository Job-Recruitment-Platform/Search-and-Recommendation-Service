"""Event models for outbox pattern"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class EventType(str, Enum):
    """Event type enumeration"""
    CREATED = "CREATED"
    UPDATED = "UPDATED"
    DELETED = "DELETED"


class AggregateType(str, Enum):
    """Aggregate type enumeration"""
    JOB = "JOB"


@dataclass
class OutboxEvent:
    """Outbox event model with 8 fields"""
    id: str  # Outbox event ID
    aggregate_type: AggregateType
    aggregate_id: str  # Job ID
    event_type: EventType
    payload: Optional[Dict[str, Any]] = None  # Full entity data (JSON parsed)
    payload_str: Optional[str] = None  # Raw payload string
    occurred_at: Optional[str] = None  # ISO timestamp
    trace_id: Optional[str] = None  # UUID for tracing
    attempts: int = 0  # Number of retry attempts

    @classmethod
    def from_redis_fields(cls, fields: Dict[str, str]) -> "OutboxEvent":
        """Create OutboxEvent from Redis stream message fields"""
        aggregate_type_str = fields.get("aggregateType", "")
        event_type_str = fields.get("eventType", "")

        try:
            aggregate_type = AggregateType(aggregate_type_str)
        except ValueError:
            aggregate_type = AggregateType.JOB  # Default fallback

        try:
            event_type = EventType(event_type_str)
        except ValueError:
            raise ValueError(f"Invalid event type: {event_type_str}")

        attempts = int(fields.get("attempts", "0"))

        return cls(
            id=fields.get("id", ""),
            aggregate_type=aggregate_type,
            aggregate_id=fields.get("aggregateId", ""),
            event_type=event_type,
            payload_str=fields.get("payload"),
            occurred_at=fields.get("occurredAt"),
            trace_id=fields.get("traceId"),
            attempts=attempts,
        )

    def is_job_event(self) -> bool:
        """Check if event is for JOB aggregate type"""
        return self.aggregate_type == AggregateType.JOB

    def is_created_or_updated(self) -> bool:
        """Check if event is CREATED or UPDATED"""
        return self.event_type in (EventType.CREATED, EventType.UPDATED)

    def is_deleted(self) -> bool:
        """Check if event is DELETED"""
        return self.event_type == EventType.DELETED

