"""Secure server-to-server forwarding of Python chatbot crisis events."""

import asyncio
import logging
import os
from typing import Any
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)

DEFAULT_WEBHOOK_URL = (
    "https://api.mindora.rw/api/v1/ai/internal/chatbot-crisis-events"
)

SEVERITY_LEVELS = {
    "low": 2,
    "medium": 3,
    "high": 4,
    "severe": 5,
}


def crisis_level_from_assessment(assessment: Any) -> int:
    """Map the pipeline's typed assessment to the shared 0–5 escalation scale."""
    if assessment is None or not getattr(assessment, "is_crisis", False):
        return 0

    severity = getattr(assessment, "crisis_severity", "medium")
    severity = getattr(severity, "value", severity)
    return SEVERITY_LEVELS.get(str(severity).lower(), 3)


async def send_crisis_webhook(
    *,
    event_id: str,
    occurred_at: str,
    crisis_level: int,
    conversation_id: str,
    external_user_ref: str,
    interaction_id: str,
    message_excerpt: str,
    conversation_excerpt: str,
) -> None:
    """Send a bounded, authenticated crisis event to the MINDORA notification bus.

    The webhook token is separate from the chatbot's user-token signing secret
    and the main-app integration key. No message or transcript text is logged.
    """
    token = os.getenv("CHATBOT_CRISIS_WEBHOOK_TOKEN", "").strip()
    if not token:
        logger.error(
            "Crisis webhook is not configured; event %s was not forwarded",
            event_id,
        )
        return

    try:
        UUID(event_id)
        UUID(interaction_id)
    except (TypeError, ValueError):
        logger.error("Crisis webhook event %s has invalid identifiers", event_id)
        return

    if crisis_level < 3 or crisis_level > 5:
        return

    url = os.getenv("CHATBOT_CRISIS_WEBHOOK_URL", DEFAULT_WEBHOOK_URL).strip()
    payload = {
        "eventId": event_id,
        "occurredAt": occurred_at,
        "crisisLevel": crisis_level,
        "conversationId": conversation_id[:120],
        "externalUserRef": external_user_ref[:120],
        "interactionId": interaction_id,
        "messageExcerpt": message_excerpt[:1500],
        "conversationExcerpt": conversation_excerpt[:5000],
    }
    headers = {"Authorization": f"Bearer {token}"}

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json=payload, headers=headers)

            if response.status_code in (200, 202):
                logger.info("Crisis event %s accepted by MINDORA webhook", event_id)
                return

            if 400 <= response.status_code < 500 and response.status_code != 429:
                logger.error(
                    "MINDORA crisis webhook rejected event %s (HTTP %s)",
                    event_id,
                    response.status_code,
                )
                return

            logger.warning(
                "MINDORA crisis webhook temporarily failed for event %s (HTTP %s), attempt %s/3",
                event_id,
                response.status_code,
                attempt + 1,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "MINDORA crisis webhook request failed for event %s (%s), attempt %s/3",
                event_id,
                type(exc).__name__,
                attempt + 1,
            )

        if attempt < 2:
            await asyncio.sleep(0.5 * (2**attempt))

    logger.error("MINDORA crisis webhook exhausted retries for event %s", event_id)
