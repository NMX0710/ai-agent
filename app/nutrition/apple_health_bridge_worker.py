from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Protocol

import httpx


@dataclass(frozen=True)
class AppleHealthWriteTask:
    draft_id: str
    user_id: str
    claim_token: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AppleHealthWriteOutcome:
    success: bool
    external_id: str | None = None
    error: str | None = None


class AppleHealthWriter(Protocol):
    def write(self, task: AppleHealthWriteTask) -> AppleHealthWriteOutcome:
        ...


class AppleHealthBridgeClient:
    def __init__(
        self,
        *,
        base_url: str,
        bridge_token: str = "",
        timeout: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)
        self._headers = {"X-Apple-Bridge-Token": bridge_token} if bridge_token else {}

    def close(self) -> None:
        self._client.close()

    def fetch_pending_writes(
        self,
        *,
        user_id: str,
        limit: int = 20,
        lease_seconds: int = 300,
    ) -> list[AppleHealthWriteTask]:
        resp = self._client.post(
            "/integrations/apple-health/pending-writes",
            json={
                "user_id": user_id,
                "limit": limit,
                "lease_seconds": lease_seconds,
            },
            headers=self._headers,
        )
        resp.raise_for_status()
        payload = resp.json()
        tasks: list[AppleHealthWriteTask] = []
        for item in payload.get("items", []):
            claim_token = item.get("claim_token")
            draft_id = item.get("draft_id")
            payload_body = item.get("payload")
            item_user_id = item.get("user_id")
            if not isinstance(claim_token, str) or not isinstance(draft_id, str):
                continue
            if not isinstance(payload_body, dict) or not isinstance(item_user_id, str):
                continue
            tasks.append(
                AppleHealthWriteTask(
                    draft_id=draft_id,
                    user_id=item_user_id,
                    claim_token=claim_token,
                    payload=payload_body,
                )
            )
        return tasks

    def report_write_result(self, task: AppleHealthWriteTask, outcome: AppleHealthWriteOutcome) -> dict[str, Any]:
        resp = self._client.post(
            "/integrations/apple-health/write-result",
            json={
                "user_id": task.user_id,
                "draft_id": task.draft_id,
                "success": outcome.success,
                "claim_token": task.claim_token,
                "external_id": outcome.external_id,
                "error": outcome.error,
            },
            headers=self._headers,
        )
        resp.raise_for_status()
        return resp.json()


def run_apple_health_bridge_cycle(
    *,
    client: AppleHealthBridgeClient,
    writer: AppleHealthWriter,
    user_id: str,
    limit: int = 20,
    lease_seconds: int = 300,
) -> dict[str, int]:
    tasks = client.fetch_pending_writes(user_id=user_id, limit=limit, lease_seconds=lease_seconds)
    processed = 0
    succeeded = 0
    failed = 0

    for task in tasks:
        processed += 1
        try:
            outcome = writer.write(task)
        except Exception as exc:
            outcome = AppleHealthWriteOutcome(success=False, error=str(exc) or exc.__class__.__name__)
        report = client.report_write_result(task, outcome)
        logging.info(
            "[AppleHealthBridgeRunner] draft_id=%s write_result_status=%s ok=%s",
            task.draft_id,
            report.get("status"),
            report.get("ok"),
        )
        if outcome.success:
            succeeded += 1
        else:
            failed += 1

    return {
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
    }


def run_apple_health_bridge_loop(
    *,
    client: AppleHealthBridgeClient,
    writer: AppleHealthWriter,
    user_id: str,
    limit: int = 20,
    lease_seconds: int = 300,
    poll_interval_seconds: float = 5.0,
    run_once: bool = False,
) -> None:
    while True:
        result = run_apple_health_bridge_cycle(
            client=client,
            writer=writer,
            user_id=user_id,
            limit=limit,
            lease_seconds=lease_seconds,
        )
        logging.info(
            "[AppleHealthBridgeRunner] user_id=%s processed=%s succeeded=%s failed=%s",
            user_id,
            result["processed"],
            result["succeeded"],
            result["failed"],
        )
        if run_once:
            return
        time.sleep(max(0.2, poll_interval_seconds))
