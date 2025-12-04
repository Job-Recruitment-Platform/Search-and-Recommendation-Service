"""Event models for outbox pattern"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import json


# ============================================
# Outbox Events (stream: outbox-events)
# ============================================
class OutboxEventType(Enum):
    """Event type enumeration"""
    CREATED = "CREATED"
    UPDATED = "UPDATED"
    DELETED = "DELETED"


class AggregateType(Enum):
    """Aggregate type enumeration"""
    JOB = "JOB"


@dataclass
class OutboxEvent:
    """Event from stream 'outbox-events'"""
    # Required fields (no defaults) - MUST come first
    id: int
    aggregate_type: AggregateType
    aggregate_id: str
    event_type: OutboxEventType
    occurred_at: str
    
    # Optional fields (with defaults) - MUST come last
    payload_str: Optional[str] = None
    trace_id: Optional[str] = None
    attempts: int = 0

    @classmethod
    def from_redis_fields(cls, fields: dict) -> 'OutboxEvent':
        """Parse from Redis message fields"""
        required = ['id', 'aggregateType',
                    'aggregateId', 'eventType', 'occurredAt']
        missing = [f for f in required if f not in fields or not fields[f]]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        try:
            aggregate_type = AggregateType(fields['aggregateType'])
        except ValueError:
            raise ValueError(
                f"Invalid aggregateType: {fields['aggregateType']}")

        try:
            event_type = OutboxEventType(fields['eventType'])
        except ValueError:
            raise ValueError(f"Invalid eventType: {fields['eventType']}")

        return cls(
            id=int(fields['id']),
            aggregate_type=aggregate_type,
            aggregate_id=fields['aggregateId'],
            event_type=event_type,
            payload_str=fields.get('payload'),
            occurred_at=fields['occurredAt'],
            trace_id=fields.get('traceId'),
            attempts=int(fields.get('attempts', 0))
        )

    def is_job_event(self) -> bool:
        """Check if event is for JOB aggregate type"""
        return self.aggregate_type == AggregateType.JOB

    def is_created_or_updated(self) -> bool:
        """Check if event is CREATED or UPDATED"""
        return self.event_type in (OutboxEventType.CREATED, OutboxEventType.UPDATED)

    def is_deleted(self) -> bool:
        """Check if event is DELETED"""
        return self.event_type == OutboxEventType.DELETED

# ============================================
# Interaction Events (stream: user-interactions)
# ============================================

class InteractionEventType(Enum):
    """User interaction events"""
    APPLY = "APPLY"
    SAVE = "SAVE"
    CLICK = "CLICK"
    CLICK_FROM_SEARCH = "CLICK_FROM_SEARCH"
    CLICK_FROM_RECOMMENDED = "CLICK_FROM_RECOMMENDED"
    CLICK_FROM_SIMILAR = "CLICK_FROM_SIMILAR"
    SKIP_FROM_SEARCH = "SKIP_FROM_SEARCH"
    SKIP_FROM_RECOMMENDED = "SKIP_FROM_RECOMMENDED"
    SKIP_FROM_SIMILAR = "SKIP_FROM_SIMILAR"

@dataclass
class InteractionEvent:
    """Event from stream 'user-interactions'"""
    account_id: int
    job_id: int
    event_type: InteractionEventType
    metadata: Optional[dict]
    occurred_at: str

    @classmethod
    def from_redis_fields(cls, fields: dict) -> 'InteractionEvent':
        """Parse from Redis message fields"""
        required = ['accountId', 'jobId', 'eventType', 'occurredAt']
        missing = [f for f in required if f not in fields or not fields[f]]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        try:
            event_type = InteractionEventType(fields['eventType'])
        except ValueError:
            raise ValueError(f"Invalid eventType: {fields['eventType']}")
        
        # Parse metadata JSON if present
        metadata = None
        if 'metadata' in fields and fields['metadata']:
            try:
                metadata = json.loads(fields['metadata'])
            except json.JSONDecodeError:
                pass
        
        return cls(
            account_id=int(fields['accountId']),
            job_id=int(fields['jobId']),
            event_type=event_type,
            metadata=metadata,
            occurred_at=fields['occurredAt']
        )
    
    def get_weight(self) -> float:
        """Get interaction weight from config"""
        from app.config import INTERACTION_WEIGHTS
        return INTERACTION_WEIGHTS.get(self.event_type.value, 0.0)