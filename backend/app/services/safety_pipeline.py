# backend/app/services/safety_pipeline.py
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Dict, Any
import os
import logging
from datetime import datetime
from ..db.models import (
    CrisisLog, UserTherapist, Therapist, User,
    CrisisType, CrisisSeverity, CrisisStatus

)
from .emailer import send_therapist_alert, render_crisis_email

_FALLBACK_ALERT_EMAIL = os.getenv("ALERT_FALLBACK_EMAIL", "")

def _map_label(label: str) -> CrisisType:
    try:
        return CrisisType[label]
    except Exception:
        return CrisisType.other

def _map_severity(sev: str) -> CrisisSeverity:
    sev = (sev or "low").lower()
    return {
        "low": CrisisSeverity.low,
        "moderate": CrisisSeverity.moderate,
        "high": CrisisSeverity.high,
        "imminent": CrisisSeverity.imminent,
    }.get(sev, CrisisSeverity.low)

def _primary_therapist(db: Session, user_id) -> Therapist | None:
    # Ensure user_id is a UUID object if it's a string
    import uuid
    if isinstance(user_id, str):
        try:
            user_id = uuid.UUID(user_id)
        except ValueError:
            pass # Let SQLAlchemy handle it or fail gracefully

    link = db.execute(
        select(UserTherapist).where(
            UserTherapist.user_id == user_id,
            UserTherapist.status == "active"
        ).order_by(UserTherapist.is_primary.desc(), UserTherapist.assigned_at.asc())
    ).scalars().first()
    return link.therapist if link else None

def _format_email(patient_name: str, crisis_type: str, severity: str, snippet: str, case_url: str) -> str:
    return f"""URGENT: {severity.upper()} {crisis_type.replace('_',' ').title()} signal detected

Patient: {patient_name}
Severity: {severity}
Signal: {crisis_type}

Excerpt:
\"\"\"{snippet[:600]}\"\"\"

Please review and follow escalation protocol: {case_url}
— Mindora Safety Agent
"""

def log_crisis_and_notify(
    db: Session,
    background: BackgroundTasks,
    user_id,
    conversation_id,
    message_id,
    text: str,
    crisis_result: Dict[str, Any],
    classifier_model: str,
    classifier_version: str
):
    logger = logging.getLogger(__name__)
    logger.info(f"🚨 log_crisis_and_notify: Starting crisis logging for user {user_id}")

    label = str(crisis_result.get("label", "other"))
    severity = str(crisis_result.get("severity", "low"))
    confidence = float(crisis_result.get("confidence", 0.5))
    rationale = crisis_result.get("rationale", "")

    logger.info(f"🚨 log_crisis_and_notify: Crisis details - label: {label}, severity: {severity}, confidence: {confidence}")

    therapist = _primary_therapist(db, user_id)
    logger.info(f"🚨 log_crisis_and_notify: Primary therapist found: {therapist.full_name if therapist else 'None'}")

    # Determine alert destination
    if therapist and therapist.active and therapist.email:
        email_to = therapist.email
        logger.info(f"🚨 log_crisis_and_notify: Sending alert to assigned therapist: {therapist.full_name} ({email_to})")
    elif _FALLBACK_ALERT_EMAIL:
        email_to = _FALLBACK_ALERT_EMAIL
        logger.warning(
            f"🚨 log_crisis_and_notify: No active therapist assigned for user {user_id}. "
            f"Falling back to ALERT_FALLBACK_EMAIL: {email_to}"
        )
    else:
        email_to = None
        logger.warning(
            f"🚨 log_crisis_and_notify: Email SKIPPED — therapist: {therapist}, "
            f"active: {therapist.active if therapist else 'N/A'}, "
            f"email: {therapist.email if therapist else 'N/A'}. "
            f"Set ALERT_FALLBACK_EMAIL in .env to receive alerts when no therapist is assigned."
        )

    crisis = CrisisLog(
        user_id=user_id,
        conversation_id=conversation_id,
        message_id=message_id,
        detected_type=_map_label(label),
        severity=_map_severity(severity),
        confidence=confidence,
        excerpt=text[:1000],
        rationale=rationale,
        classifier_model=classifier_model,
        classifier_version=classifier_version,
        status=CrisisStatus.new
    )
    if therapist:
        crisis.notified_therapist_id = therapist.id

    db.add(crisis)
    db.flush()  # crisis.id now available
    logger.info(f"🚨 log_crisis_and_notify: Crisis log created with ID: {crisis.id}")

    if email_to:
        import uuid
        resolved_user_id = uuid.UUID(user_id) if isinstance(user_id, str) else user_id
        patient = db.get(User, resolved_user_id)
        case_url = f"{os.getenv('ADMIN_DASHBOARD_URL', 'https://your-admin.app')}/cases/{crisis.id}"
        logger.info(f"🚨 log_crisis_and_notify: Case URL: {case_url}")

        subject, text_body, html_body = render_crisis_email(
            patient_name=patient.username if patient else str(user_id),
            crisis_type=label,
            severity=severity,
            snippet=text,
            case_url=case_url,
            confidence=confidence,
            detected_at=datetime.now(),
        )
        logger.info(f"🚨 log_crisis_and_notify: Email rendered — subject: {subject}")

        background.add_task(
            send_therapist_alert,
            to_email=email_to,
            subject=subject,
            text=text_body,
            html=html_body,
        )
        crisis.status = CrisisStatus.notified
        logger.info(f"🚨 log_crisis_and_notify: Crisis alert queued → {email_to}")

    # caller commits
    return crisis.id
