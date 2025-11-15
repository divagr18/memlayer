from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import uuid
import json
from datetime import datetime, timezone

class TraceEvent(BaseModel):
    """Represents a single, timed event within a larger trace."""
    name: str
    duration_ms: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None

class Trace(BaseModel):
    """
    Holds the complete record of a memory operation, including all timed sub-events.
    This object is what gets returned to the developer for inspection.
    """
    trace_id: str = Field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:12]}")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    events: List[TraceEvent] = Field(default_factory=list)
    final_result: Optional[Any] = None
    final_error: Optional[str] = None

    # --- FIX 1: Add `metadata` parameter to start_event ---
    def start_event(self, name: str, metadata: Dict = None) -> 'TraceContextManager':
        """Creates a context manager to automatically time a block of code."""
        return TraceContextManager(self, name, metadata)

    def add_event(self, name: str, duration_ms: float, metadata: Dict = None, error: Optional[Exception] = None):
        """Manually adds an event to the trace."""
        event = TraceEvent(
            name=name,
            duration_ms=duration_ms,
            metadata=metadata or {},
            error=str(error) if error else None
        )
        self.events.append(event)

    # --- FIX 2: Add `error` parameter to conclude ---
    def conclude(self, result: Any = None, error: Optional[Exception] = None):
        """Finalizes the trace, calculating total duration and storing the result."""
        self.end_time = datetime.now(timezone.utc)
        self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.final_result = result
        if error:
            self.final_error = str(error)

    def summary(self) -> str:
        """Provides a simple, human-readable summary of the trace."""
        s = f"Trace Summary ({self.trace_id}) - Total: {self.total_duration_ms:.2f}ms"
        if self.final_error:
            s += f" - FAILED: {self.final_error}"
        s += "\n"
        
        for event in self.events:
            s += f"  - Event '{event.name}': {event.duration_ms:.2f}ms"
            if event.metadata:
                s += f" (meta: {json.dumps(event.metadata)})"
            if event.error:
                s += f" - FAILED: {event.error}"
            s += "\n"
        return s

class TraceContextManager:
    """A context manager to automatically time events and handle exceptions."""
    # --- FIX 3: Accept and store metadata in __init__ ---
    def __init__(self, trace: Trace, name: str, metadata: Dict = None):
        self.trace = trace
        self.name = name
        self.metadata = metadata or {}
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        error = exc_val if exc_type else None
        # Now, when we add the event, the metadata is correctly passed through.
        self.trace.add_event(self.name, duration_ms, self.metadata, error)