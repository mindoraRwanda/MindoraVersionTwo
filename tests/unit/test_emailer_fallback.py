from unittest.mock import Mock

import pytest

from backend.app.services import emailer


@pytest.mark.parametrize("status_code", [401, 403])
def test_sendgrid_auth_or_quota_rejection_falls_back_to_resend(
    monkeypatch, status_code
):
    monkeypatch.setattr(emailer, "SENDGRID_API_KEY", "sendgrid-test-key")
    monkeypatch.setattr(emailer, "RESEND_API_KEY", "resend-test-key")
    response = Mock(status_code=status_code)
    monkeypatch.setattr(emailer.requests, "post", Mock(return_value=response))
    resend = Mock(return_value=True)
    monkeypatch.setattr(emailer, "_send_via_resend", resend)

    result = emailer._send_via_sendgrid(
        to_email="patient@example.com",
        subject="Test alert",
        text="Test body",
        html=None,
    )

    assert result is True
    resend.assert_called_once_with(
        to_email="patient@example.com",
        subject="Test alert",
        text="Test body",
        html=None,
    )


def test_sendgrid_server_error_does_not_fallback_and_risk_duplicate_email(
    monkeypatch,
):
    monkeypatch.setattr(emailer, "RESEND_API_KEY", "resend-test-key")
    monkeypatch.setattr(
        emailer.requests, "post", Mock(return_value=Mock(status_code=500))
    )
    resend = Mock(return_value=True)
    monkeypatch.setattr(emailer, "_send_via_resend", resend)

    result = emailer._send_via_sendgrid(
        to_email="patient@example.com",
        subject="Test alert",
        text="Test body",
        html=None,
    )

    assert result is False
    resend.assert_not_called()


def test_send_email_logs_do_not_include_recipient_or_subject(monkeypatch, caplog):
    monkeypatch.setattr(emailer, "EMAILS_ENABLED", True)
    monkeypatch.setattr(emailer, "SENDGRID_API_KEY", "")
    monkeypatch.setattr(emailer, "RESEND_API_KEY", "")
    monkeypatch.setattr(emailer, "_send_via_smtp", Mock(return_value=True))

    emailer.send_therapist_alert(
        to_email="patient.private@example.com",
        subject="URGENT: patient private crisis details",
        text="Sensitive clinical excerpt",
    )

    assert "patient.private@example.com" not in caplog.text
    assert "patient private crisis details" not in caplog.text
    assert "Sensitive clinical excerpt" not in caplog.text
