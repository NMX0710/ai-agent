from __future__ import annotations

import argparse
import logging
import os
import sys

from app.nutrition.apple_health_bridge_worker import (
    AppleHealthBridgeClient,
    AppleHealthWriteOutcome,
    AppleHealthWriteTask,
    run_apple_health_bridge_loop,
)


class MockSuccessWriter:
    def write(self, task: AppleHealthWriteTask) -> AppleHealthWriteOutcome:
        sync_identifier = str(task.payload.get("sync_identifier") or task.draft_id)
        external_id = f"mock-apple-health:{sync_identifier}"
        logging.info(
            "[AppleHealthMockWriter] draft_id=%s result=success external_id=%s",
            task.draft_id,
            external_id,
        )
        return AppleHealthWriteOutcome(success=True, external_id=external_id)


class MockFailureWriter:
    def write(self, task: AppleHealthWriteTask) -> AppleHealthWriteOutcome:
        logging.info("[AppleHealthMockWriter] draft_id=%s result=failure", task.draft_id)
        return AppleHealthWriteOutcome(success=False, error="mock_writer_failure")


def _build_writer(mode: str):
    if mode == "mock-success":
        return MockSuccessWriter()
    if mode == "mock-failure":
        return MockFailureWriter()
    raise ValueError(f"Unsupported writer mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Apple Health bridge worker.")
    parser.add_argument("--base-url", default=os.getenv("APPLE_HEALTH_BRIDGE_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--bridge-token", default=os.getenv("APPLE_HEALTH_BRIDGE_TOKEN", ""))
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--writer", default=os.getenv("APPLE_HEALTH_WRITER_MODE", "mock-success"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("APPLE_HEALTH_BRIDGE_LIMIT", "20")))
    parser.add_argument("--lease-seconds", type=int, default=int(os.getenv("APPLE_HEALTH_BRIDGE_LEASE_SECONDS", "300")))
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=float(os.getenv("APPLE_HEALTH_BRIDGE_POLL_INTERVAL_SECONDS", "5")),
    )
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    writer = _build_writer(args.writer)
    client = AppleHealthBridgeClient(
        base_url=args.base_url,
        bridge_token=args.bridge_token,
    )

    try:
        logging.info(
            "[AppleHealthBridgeRunner] base_url=%s user_id=%s writer=%s once=%s",
            args.base_url,
            args.user_id,
            args.writer,
            args.once,
        )
        run_apple_health_bridge_loop(
            client=client,
            writer=writer,
            user_id=args.user_id,
            limit=args.limit,
            lease_seconds=args.lease_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            run_once=args.once,
        )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
