import os
import time
from uuid import uuid4

import pytest
import psycopg

from app.memory.store import PostgresMemoryStore


def _dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://ai_agent:ai_agent_dev@localhost:5432/ai_agent",
    )


def _cleanup_user(user_id: str) -> None:
    with psycopg.connect(_dsn(), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_memory WHERE user_id = %s;", (user_id,))


@pytest.mark.integration
def test_cross_session_memory_retrieval_by_user_id():
    """
    Verify long-term memory survives across different chat_id values
    for the same user_id.
    """
    user_id = f"it-user-{uuid4().hex[:10]}"
    chat_id_1 = f"chat-{uuid4().hex[:8]}"
    chat_id_2 = f"chat-{uuid4().hex[:8]}"

    store = PostgresMemoryStore(
        dsn=_dsn(),
        retention_days=90,
        max_records_per_user=200,
    )
    store.initialize()
    assert store.enabled is True

    _cleanup_user(user_id)
    try:
        store.upsert_memory(
            user_id=user_id,
            memory_key="pref-high-protein-chicken",
            summary="User prefers high-protein low-fat chicken dinners.",
            keywords=["high-protein", "low-fat", "chicken", "dinner"],
            source_chat_id=chat_id_1,
        )

        # Simulate a new window/new conversation using the same user_id.
        items = store.retrieve(
            user_id=user_id,
            query="any chicken dinner recommendation?",
            limit=5,
        )

        assert len(items) >= 1
        assert any("chicken" in item.summary.lower() for item in items)
        assert any(item.source_chat_id == chat_id_1 for item in items)
        assert all(item.source_chat_id != chat_id_2 for item in items if item.source_chat_id)

    finally:
        _cleanup_user(user_id)


@pytest.mark.integration
def test_memory_cap_enforces_max_records_per_user():
    """
    Verify store pruning keeps at most N records per user.
    """
    user_id = f"it-cap-{uuid4().hex[:10]}"
    cap = 3

    store = PostgresMemoryStore(
        dsn=_dsn(),
        retention_days=90,
        max_records_per_user=cap,
    )
    store.initialize()
    assert store.enabled is True

    _cleanup_user(user_id)
    try:
        for idx in range(5):
            store.upsert_memory(
                user_id=user_id,
                memory_key=f"mk-{idx}",
                summary=f"Memory item {idx}",
                keywords=[f"k{idx}"],
                source_chat_id=f"chat-{idx}",
            )
            time.sleep(0.01)

        with psycopg.connect(_dsn(), autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM user_memory WHERE user_id = %s;",
                    (user_id,),
                )
                (count,) = cur.fetchone()
                assert count == cap
    finally:
        _cleanup_user(user_id)
