# backend/app/services/safety_pipeline.py
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Dict, Any
import os
from datetime import datetime
from backend.app.db.models import (
    CrisisLog, UserTherapist, Therapist, User,
    CrisisType, CrisisSeverity, CrisisStatus
    
)
from backend.app.services.emailer import send_therapist_alert,render_crisis_email

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

def _primary_therapist(db: Session, user_id: int) -> Therapist | None:
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
    user_id: int,
    conversation_id: int,
    message_id: int,
    text: str,
    crisis_result: Dict[str, Any],
    classifier_model: str,
    classifier_version: str
) -> int | None:
    label = str(crisis_result.get("label", "other"))
    severity = str(crisis_result.get("severity", "low"))
    confidence = float(crisis_result.get("confidence", 0.5))
    rationale = crisis_result.get("rationale", "")

    therapist = _primary_therapist(db, user_id)
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

    if therapist and therapist.active and therapist.email:
        patient = db.get(User, user_id)
        case_url = f"{os.getenv('ADMIN_DASHBOARD_URL', 'https://your-admin.app')}/cases/{crisis.id}"
        subject, text_body, html_body = render_crisis_email(
        patient_name=patient.username,
        crisis_type=label,
        severity=severity,
        snippet=text,
        case_url=case_url,
        confidence=confidence,
        detected_at=datetime.utcnow(),
        )

        background.add_task(
    send_therapist_alert,
    to_email=therapist.email,
    subject=subject,
    text=text_body,
    html=html_body,
)
        crisis.status = CrisisStatus.notified

    # caller commits
    return crisis.id
