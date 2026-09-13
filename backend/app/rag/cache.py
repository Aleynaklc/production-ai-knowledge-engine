"""Bounded, process-local LRU cache for validated answers; caller owns synchronization."""

from collections import OrderedDict
from time import monotonic

from backend.app.rag.models import RAGAnswer

type AnswerCacheKey = tuple[str, str, int]


class AnswerCache:
    """Store private copies with a fixed TTL measured from insertion, not last access."""

    def __init__(self, max_entries: int, ttl_seconds: int) -> None:
        if max_entries < 0 or ttl_seconds <= 0:
            raise ValueError("Cache capacity must be non-negative and TTL must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[AnswerCacheKey, tuple[float, RAGAnswer]] = OrderedDict()

    def get(self, key: AnswerCacheKey) -> RAGAnswer | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, answer = entry
        if monotonic() >= expires_at:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return answer.model_copy(deep=True)

    def put(self, key: AnswerCacheKey, answer: RAGAnswer) -> None:
        if (
            not self.max_entries
            or answer.status != "answered"
            or not answer.validation.valid
            or answer.validation.abstained
        ):
            return
        now = monotonic()
        for expired in [key for key, (expiry, _) in self._entries.items() if expiry <= now]:
            del self._entries[expired]
        self._entries[key] = (now + self.ttl_seconds, answer.model_copy(deep=True))
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
