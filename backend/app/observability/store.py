"""Thread-safe bounded storage for recent in-process system traces."""

from collections import OrderedDict
from threading import Lock

from backend.app.observability.models import SystemTrace


class TraceStore:
    """Keep the newest traces in memory without unbounded process growth."""

    def __init__(self, max_records: int = 200) -> None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        self.max_records = max_records
        self._records: OrderedDict[str, SystemTrace] = OrderedDict()
        self._lock = Lock()

    def add(self, trace: SystemTrace) -> None:
        """Store a trace and evict the oldest record when the bound is exceeded."""

        with self._lock:
            self._records[trace.trace_id] = trace
            self._records.move_to_end(trace.trace_id)
            while len(self._records) > self.max_records:
                self._records.popitem(last=False)

    def get(self, trace_id: str) -> SystemTrace | None:
        """Return a trace by identifier without changing recency order."""

        with self._lock:
            return self._records.get(trace_id)

    def recent(self, limit: int = 25) -> list[SystemTrace]:
        """Return newest-first traces up to a validated caller limit."""

        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            records = list(self._records.values())
        return list(reversed(records[-limit:]))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
