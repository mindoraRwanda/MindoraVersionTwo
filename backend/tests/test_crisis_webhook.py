import asyncio
from types import SimpleNamespace

from app.services.crisis_webhook import (
    crisis_level_from_assessment,
    send_crisis_webhook,
)


def test_crisis_severity_maps_to_shared_escalation_levels():
    assert crisis_level_from_assessment(
        SimpleNamespace(is_crisis=True, crisis_severity="medium")
    ) == 3
    assert crisis_level_from_assessment(
        SimpleNamespace(is_crisis=True, crisis_severity="high")
    ) == 4
    assert crisis_level_from_assessment(
        SimpleNamespace(is_crisis=True, crisis_severity="severe")
    ) == 5
    assert crisis_level_from_assessment(
        SimpleNamespace(is_crisis=False, crisis_severity="severe")
    ) == 0


def test_webhook_sends_only_bounded_authenticated_payload(monkeypatch):
    monkeypatch.setenv("CHATBOT_CRISIS_WEBHOOK_TOKEN", "test-secret")
    calls = {}

    class Response:
        status_code = 202

    class Client:
        def __init__(self, timeout):
            calls["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            calls.update(url=url, payload=json, headers=headers)
            return Response()

    monkeypatch.setattr("app.services.crisis_webhook.httpx.AsyncClient", Client)

    asyncio.run(
        send_crisis_webhook(
            event_id="432706a2-d868-412e-837a-62044072e586",
            occurred_at="2026-11-25T10:00:00+00:00",
            crisis_level=5,
            conversation_id="conversation-1",
            external_user_ref="main-user-1",
            interaction_id="f406523f-a0fd-42e1-a488-9d97f903c679",
            message_excerpt="x" * 2000,
            conversation_excerpt="y" * 6000,
        )
    )

    assert calls["timeout"] == 5.0
    assert calls["headers"] == {"Authorization": "Bearer test-secret"}
    assert calls["url"].endswith("/api/v1/ai/internal/chatbot-crisis-events")
    assert len(calls["payload"]["messageExcerpt"]) == 1500
    assert len(calls["payload"]["conversationExcerpt"]) == 5000
    assert calls["payload"]["crisisLevel"] == 5


def test_webhook_fails_closed_when_secret_is_missing(monkeypatch):
    monkeypatch.delenv("CHATBOT_CRISIS_WEBHOOK_TOKEN", raising=False)

    async def unexpected_client(*args, **kwargs):
        raise AssertionError("HTTP client must not be created without a token")

    monkeypatch.setattr("app.services.crisis_webhook.httpx.AsyncClient", unexpected_client)

    asyncio.run(
        send_crisis_webhook(
            event_id="432706a2-d868-412e-837a-62044072e586",
            occurred_at="2026-11-25T10:00:00+00:00",
            crisis_level=4,
            conversation_id="conversation-1",
            external_user_ref="main-user-1",
            interaction_id="f406523f-a0fd-42e1-a488-9d97f903c679",
            message_excerpt="urgent",
            conversation_excerpt="",
        )
    )
