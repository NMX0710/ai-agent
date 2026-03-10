import logging
from dataclasses import dataclass


@dataclass
class ImageGovernanceEvent:
    source: str
    user_id: str
    chat_id: str
    update_id: int
    message_id: int | None
    event_ts: int | None
    caption: str
    file_id: str
    file_unique_id: str
    file_path: str
    mime_type: str
    file_size_bytes: int
    sha256: str


async def process_image_event(event: ImageGovernanceEvent) -> dict:
    """
    Placeholder governance intake for channel images.
    A later iteration can enqueue this event to a real governance pipeline.
    """
    tracking_id = f"{event.source}-{event.update_id}-{event.file_unique_id or event.file_id}"
    logging.info(
        "[ModelGovernance] Accepted image event tracking_id=%s source=%s user_id=%s chat_id=%s "
        "file_size=%s mime=%s sha256=%s",
        tracking_id,
        event.source,
        event.user_id,
        event.chat_id,
        event.file_size_bytes,
        event.mime_type,
        event.sha256[:16],
    )
    return {"status": "accepted", "tracking_id": tracking_id}
