import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import psycopg


@dataclass
class MemoryItem:
    summary: str
    keywords: List[str]
    source_chat_id: Optional[str]
    updated_at: datetime


class PostgresMemoryStore:
    def __init__(
        self,
        dsn: str,
        retention_days: int = 90,
        max_records_per_user: int = 200,
    ):
        self.dsn = dsn
        self.retention_days = retention_days
        self.max_records_per_user = max_records_per_user
        self.enabled = False

    def initialize(self) -> None:
        try:
            with psycopg.connect(self.dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS user_memory (
                            id BIGSERIAL PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            memory_key TEXT NOT NULL,
                            summary TEXT NOT NULL,
                            keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
                            source_chat_id TEXT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            last_used_at TIMESTAMPTZ NULL,
                            UNIQUE(user_id, memory_key)
                        );
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_user_memory_user_updated "
                        "ON user_memory(user_id, updated_at DESC);"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_user_memory_fts "
                        "ON user_memory USING GIN (to_tsvector('simple', summary));"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_user_memory_keywords "
                        "ON user_memory USING GIN (keywords);"
                    )
            self.enabled = True
        except Exception:
            logging.exception("[MemoryStore] Failed to initialize Postgres memory store.")
            self.enabled = False

    def upsert_memory(
        self,
        *,
        user_id: str,
        memory_key: str,
        summary: str,
        keywords: List[str],
        source_chat_id: Optional[str],
    ) -> None:
        if not self.enabled:
            return

        try:
            with psycopg.connect(self.dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO user_memory (user_id, memory_key, summary, keywords, source_chat_id)
                        VALUES (%s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (user_id, memory_key)
                        DO UPDATE SET
                            summary = EXCLUDED.summary,
                            keywords = EXCLUDED.keywords,
                            source_chat_id = EXCLUDED.source_chat_id,
                            updated_at = NOW();
                        """,
                        (user_id, memory_key, summary, json.dumps(keywords), source_chat_id),
                    )
                    self._prune(cur, user_id)
        except Exception:
            logging.exception("[MemoryStore] Failed to upsert user memory.")

    def retrieve(self, *, user_id: str, query: str, limit: int = 5) -> List[MemoryItem]:
        if not self.enabled:
            return []

        limit = max(1, limit)
        query = (query or "").strip()
        query_tokens = [tok.lower() for tok in query.split() if tok.strip()]

        try:
            with psycopg.connect(self.dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    if query:
                        cur.execute(
                            """
                            SELECT
                                id,
                                summary,
                                keywords,
                                source_chat_id,
                                updated_at,
                                (
                                    CASE
                                        WHEN to_tsvector('simple', summary) @@ plainto_tsquery('simple', %s)
                                        THEN 2 ELSE 0
                                    END
                                    +
                                    CASE
                                        WHEN EXISTS (
                                            SELECT 1
                                            FROM jsonb_array_elements_text(keywords) kw
                                            WHERE lower(kw) = ANY(%s)
                                        )
                                        THEN 1 ELSE 0
                                    END
                                ) AS score
                            FROM user_memory
                            WHERE user_id = %s
                              AND created_at >= NOW() - (%s || ' days')::interval
                            ORDER BY score DESC, updated_at DESC
                            LIMIT %s;
                            """,
                            (query, query_tokens, user_id, self.retention_days, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, summary, keywords, source_chat_id, updated_at, 0 AS score
                            FROM user_memory
                            WHERE user_id = %s
                              AND created_at >= NOW() - (%s || ' days')::interval
                            ORDER BY updated_at DESC
                            LIMIT %s;
                            """,
                            (user_id, self.retention_days, limit),
                        )

                    rows = cur.fetchall()
                    if not rows:
                        return []

                    ids: List[int] = [row[0] for row in rows]
                    cur.execute(
                        "UPDATE user_memory SET last_used_at = NOW() WHERE id = ANY(%s);",
                        (ids,),
                    )

                    return [
                        MemoryItem(
                            summary=row[1],
                            keywords=row[2] if isinstance(row[2], list) else [],
                            source_chat_id=row[3],
                            updated_at=row[4],
                        )
                        for row in rows
                        if row[5] > 0 or not query
                    ]
        except Exception:
            logging.exception("[MemoryStore] Failed to retrieve user memory.")
            return []

    def _prune(self, cur: psycopg.Cursor, user_id: str) -> None:
        cur.execute(
            """
            DELETE FROM user_memory
            WHERE user_id = %s
              AND created_at < NOW() - (%s || ' days')::interval;
            """,
            (user_id, self.retention_days),
        )
        cur.execute(
            """
            DELETE FROM user_memory
            WHERE id IN (
                SELECT id
                FROM user_memory
                WHERE user_id = %s
                ORDER BY updated_at DESC
                OFFSET %s
            );
            """,
            (user_id, self.max_records_per_user),
        )
