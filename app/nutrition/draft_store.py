from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from app.nutrition.meallog import CanonicalMealLog


@dataclass
class MealDraftRecord:
    draft_id: str
    user_id: str
    chat_id: str
    meal_log: CanonicalMealLog
    status: str
    created_at: datetime
    updated_at: datetime
    confirmation_prompted: bool = False
    bridge_dispatched: bool = False
    bridge_attempt_count: int = 0
    bridge_claim_token: Optional[str] = None
    bridge_claimed_at: Optional[datetime] = None
    bridge_claim_expires_at: Optional[datetime] = None
    last_claim_token: Optional[str] = None
    write_result: Optional[dict] = None
    last_error: Optional[str] = None


class InMemoryMealDraftStore:
    """
    Short-lived runtime store for confirmation flow.
    This is not long-term user memory.
    """

    def __init__(self, ttl_hours: int = 24):
        self._ttl = timedelta(hours=max(1, ttl_hours))
        self._records: Dict[str, MealDraftRecord] = {}
        self._lock = Lock()

    def _prune_locked(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [draft_id for draft_id, rec in self._records.items() if now - rec.updated_at > self._ttl]
        for draft_id in expired:
            self._records.pop(draft_id, None)

    @staticmethod
    def _claim_is_active(rec: MealDraftRecord, now: datetime) -> bool:
        if not rec.bridge_claim_token or not rec.bridge_claim_expires_at:
            return False
        return rec.bridge_claim_expires_at > now

    def save(self, record: MealDraftRecord) -> MealDraftRecord:
        with self._lock:
            self._prune_locked()
            self._records[record.draft_id] = record
            return record

    def get(self, draft_id: str) -> Optional[MealDraftRecord]:
        with self._lock:
            self._prune_locked()
            return self._records.get(draft_id)

    def update(self, draft_id: str, **fields) -> Optional[MealDraftRecord]:
        with self._lock:
            self._prune_locked()
            rec = self._records.get(draft_id)
            if not rec:
                return None
            for key, value in fields.items():
                setattr(rec, key, value)
            rec.updated_at = datetime.now(timezone.utc)
            return rec

    def claim_for_user(self, user_id: str, *, status: str, limit: int = 20, lease_seconds: int = 300) -> List[MealDraftRecord]:
        with self._lock:
            self._prune_locked()
            now = datetime.now(timezone.utc)
            lease = timedelta(seconds=max(30, lease_seconds))
            rows = [
                rec for rec in self._records.values()
                if rec.user_id == user_id and rec.status == status and not self._claim_is_active(rec, now)
            ]
            rows.sort(key=lambda x: x.updated_at, reverse=True)

            claimed: List[MealDraftRecord] = []
            for rec in rows[:max(1, limit)]:
                claim_token = uuid4().hex
                rec.bridge_dispatched = True
                rec.bridge_attempt_count += 1
                rec.bridge_claim_token = claim_token
                rec.bridge_claimed_at = now
                rec.bridge_claim_expires_at = now + lease
                rec.last_claim_token = claim_token
                rec.updated_at = now
                claimed.append(rec)
            return claimed

    def latest_pending_for_chat(self, user_id: str, chat_id: str) -> Optional[MealDraftRecord]:
        with self._lock:
            self._prune_locked()
            pending = [
                rec for rec in self._records.values()
                if rec.user_id == user_id and rec.chat_id == chat_id and rec.status == "draft"
            ]
            if not pending:
                return None
            pending.sort(key=lambda x: x.created_at, reverse=True)
            return pending[0]

    def list_for_user(self, user_id: str, status: Optional[str] = None, limit: int = 20) -> List[MealDraftRecord]:
        with self._lock:
            self._prune_locked()
            rows = [rec for rec in self._records.values() if rec.user_id == user_id]
            if status:
                rows = [rec for rec in rows if rec.status == status]
            rows.sort(key=lambda x: x.updated_at, reverse=True)
            return rows[:max(1, limit)]


_MEAL_DRAFT_STORE = InMemoryMealDraftStore(ttl_hours=24)


def get_meal_draft_store() -> InMemoryMealDraftStore:
    return _MEAL_DRAFT_STORE
