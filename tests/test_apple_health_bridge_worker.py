import sys
from pathlib import Path

import httpx

project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

from app import apple_health_bridge_runner
from app.nutrition.apple_health_bridge_worker import (
    AppleHealthBridgeClient,
    AppleHealthWriteOutcome,
    AppleHealthWriteTask,
    run_apple_health_bridge_cycle,
)


def test_bridge_worker_cycle_reports_success_and_failure():
    reported = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/integrations/apple-health/pending-writes":
            body = {
                "ok": True,
                "count": 2,
                "items": [
                    {
                        "draft_id": "draft-1",
                        "user_id": "tg:1",
                        "claim_token": "claim-1",
                        "payload": {"sync_identifier": "meal:1", "samples": [{"identifier": "x"}]},
                    },
                    {
                        "draft_id": "draft-2",
                        "user_id": "tg:1",
                        "claim_token": "claim-2",
                        "payload": {"sync_identifier": "meal:2", "samples": [{"identifier": "y"}]},
                    },
                ],
            }
            return httpx.Response(200, json=body)

        if request.url.path == "/integrations/apple-health/write-result":
            reported.append(request.read().decode("utf-8"))
            return httpx.Response(200, json={"ok": True})

        return httpx.Response(404, json={"ok": False})

    class FakeWriter:
        def write(self, task: AppleHealthWriteTask) -> AppleHealthWriteOutcome:
            if task.draft_id == "draft-1":
                return AppleHealthWriteOutcome(success=True, external_id="hk-1")
            return AppleHealthWriteOutcome(success=False, error="hk_write_failed")

    client = AppleHealthBridgeClient(
        base_url="https://bridge.test",
        bridge_token="secret-bridge",
        transport=httpx.MockTransport(handler),
    )

    result = run_apple_health_bridge_cycle(
        client=client,
        writer=FakeWriter(),
        user_id="tg:1",
        limit=10,
        lease_seconds=180,
    )

    client.close()

    assert result == {"processed": 2, "succeeded": 1, "failed": 1}
    assert len(reported) == 2
    assert '"claim_token":"claim-1"' in reported[0]
    assert '"external_id":"hk-1"' in reported[0]
    assert '"claim_token":"claim-2"' in reported[1]
    assert '"error":"hk_write_failed"' in reported[1]


def test_bridge_worker_cycle_turns_writer_exception_into_failure_report():
    reported = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/integrations/apple-health/pending-writes":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "count": 1,
                    "items": [
                        {
                            "draft_id": "draft-err",
                            "user_id": "tg:2",
                            "claim_token": "claim-err",
                            "payload": {"sync_identifier": "meal:err", "samples": []},
                        }
                    ],
                },
            )
        if request.url.path == "/integrations/apple-health/write-result":
            reported.append(request.read().decode("utf-8"))
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"ok": False})

    class ExplodingWriter:
        def write(self, task: AppleHealthWriteTask) -> AppleHealthWriteOutcome:
            raise RuntimeError("healthkit unavailable")

    client = AppleHealthBridgeClient(
        base_url="https://bridge.test",
        transport=httpx.MockTransport(handler),
    )

    result = run_apple_health_bridge_cycle(
        client=client,
        writer=ExplodingWriter(),
        user_id="tg:2",
    )

    client.close()

    assert result == {"processed": 1, "succeeded": 0, "failed": 1}
    assert len(reported) == 1
    assert '"claim_token":"claim-err"' in reported[0]
    assert '"error":"healthkit unavailable"' in reported[0]


def test_bridge_runner_cli_once_uses_mock_failure_writer(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *, base_url: str, bridge_token: str = ""):
            captured["base_url"] = base_url
            captured["bridge_token"] = bridge_token

        def close(self) -> None:
            captured["closed"] = True

    def fake_loop(*, client, writer, user_id, limit, lease_seconds, poll_interval_seconds, run_once):
        outcome = writer.write(
            AppleHealthWriteTask(
                draft_id="draft-cli",
                user_id=user_id,
                claim_token="claim-cli",
                payload={"sync_identifier": "meal:cli"},
            )
        )
        captured["client"] = client
        captured["user_id"] = user_id
        captured["limit"] = limit
        captured["lease_seconds"] = lease_seconds
        captured["poll_interval_seconds"] = poll_interval_seconds
        captured["run_once"] = run_once
        captured["outcome"] = outcome

    monkeypatch.setattr(apple_health_bridge_runner, "AppleHealthBridgeClient", FakeClient)
    monkeypatch.setattr(apple_health_bridge_runner, "run_apple_health_bridge_loop", fake_loop)

    exit_code = apple_health_bridge_runner.main(
        [
            "--base-url",
            "http://127.0.0.1:8000",
            "--bridge-token",
            "secret-1",
            "--user-id",
            "tg:runner",
            "--writer",
            "mock-failure",
            "--limit",
            "3",
            "--lease-seconds",
            "90",
            "--poll-interval-seconds",
            "0.5",
            "--once",
        ]
    )

    assert exit_code == 0
    assert captured["base_url"] == "http://127.0.0.1:8000"
    assert captured["bridge_token"] == "secret-1"
    assert captured["user_id"] == "tg:runner"
    assert captured["limit"] == 3
    assert captured["lease_seconds"] == 90
    assert captured["poll_interval_seconds"] == 0.5
    assert captured["run_once"] is True
    assert captured["outcome"] == AppleHealthWriteOutcome(success=False, error="mock_writer_failure")
    assert captured["closed"] is True
