import os
import ssl
import smtplib
import logging
import requests
from email.message import EmailMessage
from datetime import datetime

# ---- SMTP / Mailtrap defaults ----
SMTP_HOST = os.getenv("SMTP_HOST", "sandbox.smtp.mailtrap.io")
SMTP_PORT = int(os.getenv("SMTP_PORT", "2525"))
SMTP_USER = os.getenv("SMTP_USER", "1ae11aaecfabd9")
SMTP_PASS = os.getenv("SMTP_PASS", "4bc4abeba1f55c")
FROM_EMAIL = os.getenv("ALERTS_FROM", "alerts@mindora.local")
EMAILS_ENABLED = os.getenv("EMAILS_ENABLED", "1") == "1"

# ---- Resend (HTTP API) ----
# Many hosts (Render included) block outbound SMTP ports (25/465/587)
# entirely, regardless of credentials — raw smtplib connections fail there
# with "Network is unreachable". Resend sends over HTTPS instead, which
# isn't blocked. When RESEND_API_KEY is set, it's used in place of SMTP;
# otherwise this falls back to SMTP unchanged (e.g. for local dev).
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_API_URL = "https://api.resend.com/emails"

# Branding / links
ORG_NAME = os.getenv("ORG_NAME", "Mindora")
ORG_LOGO_URL = os.getenv("ORG_LOGO_URL", "")
ADMIN_DASHBOARD_URL = os.getenv("ADMIN_DASHBOARD_URL", "https://your-admin.app")

SEV_COLORS = {
    "imminent": "#b91c1c",
    "high":     "#dc2626",
    "moderate": "#f59e0b",
    "low":      "#2563eb",
}

def _title(s: str) -> str:
    return (s or "").replace("_", " ").title()

def render_crisis_email(
    *,
    patient_name: str,
    crisis_type: str,
    severity: str,
    snippet: str,
    case_url: str,
    confidence: float | None = None,
    detected_at: datetime | None = None,
) -> tuple[str, str, str]:
    """Return (subject, text_body, html_body) for the therapist alert."""
    sev = (severity or "low").lower()
    sev_color = SEV_COLORS.get(sev, "#2563eb")
    sev_label = sev.upper()
    crisis_label = _title(crisis_type)

    subject = "[{org}] URGENT: {sev} {ctype} signal for {patient}".format(
        org=ORG_NAME, sev=sev, ctype=crisis_type, patient=patient_name
    )

    # Plain-text fallback
    text_body = (
        "URGENT: {sev_label} {crisis_label} signal detected\n\n"
        "Patient: {patient}\n"
        "Severity: {sev}\n"
        "Signal: {ctype}{conf}{ts}\n\n"
        "Excerpt:\n"
        "\"\"\"{excerpt}\"\"\"\n\n"
        "Open case: {case_url}\n\n"
        "— {org} Safety Agent\n"
    ).format(
        sev_label=sev_label,
        crisis_label=crisis_label,
        patient=patient_name,
        sev=sev,
        ctype=crisis_type,
        conf=("\nConfidence: {:.2f}".format(confidence) if confidence is not None else ""),
        ts=("\nDetected at: {}".format(detected_at.isoformat()) if detected_at else ""),
        excerpt=(snippet or "").strip()[:600],
        case_url=case_url,
        org=ORG_NAME,
    )

    # HTML primary (note the doubled braces {{ }} so .format() keeps CSS braces)
    logo_html = "<img src='{url}' alt='{org} logo' style='height:28px;display:block;'/>".format(
        url=ORG_LOGO_URL, org=ORG_NAME
    ) if ORG_LOGO_URL else ""

    HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{subject}</title>
<style>
  .container {{ max-width:640px; margin:0 auto; background:#fff; border-radius:12px; overflow:hidden;
               box-shadow:0 2px 8px rgba(0,0,0,.06); font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif; }}
  .header {{ padding:20px 24px; background:#0f172a; color:#fff; }}
  .brand {{ display:flex; align-items:center; gap:12px; }}
  .title {{ margin:0; font-size:18px; font-weight:600; }}
  .pill {{ display:inline-block; padding:4px 10px; border-radius:9999px; background:{sev_color}; color:#fff;
           font-size:12px; letter-spacing:.3px; font-weight:600; vertical-align:middle; }}
  .body {{ padding:24px; color:#0f172a; line-height:1.55; }}
  .kv {{ margin:14px 0; }}
  .row {{ margin:6px 0; }}
  .key {{ display:inline-block; min-width:110px; color:#334155; }}
  .excerpt {{ white-space:pre-wrap; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px;
              font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono","Courier New", monospace; }}
  .btn {{ display:inline-block; background:#0ea5e9; color:#fff; text-decoration:none; padding:12px 16px; border-radius:10px; font-weight:600; }}
  .meta {{ color:#64748b; font-size:12px; margin-top:16px; }}
  .footer {{ padding:18px 24px; color:#64748b; font-size:12px; border-top:1px solid #e2e8f0; }}
</style>
</head>
<body style="background:#f1f5f9; padding:20px;">
  <div class="container">
    <div class="header">
      <div class="brand">
        {logo_html}
        <h1 class="title">{org} · Safety Alert</h1>
      </div>
    </div>

    <div class="body">
      <p><span class="pill">{sev_label}</span> <strong style="margin-left:8px;">{crisis_label}</strong> signal detected</p>

      <div class="kv">
        <div class="row"><span class="key">Patient:</span> <strong>{patient}</strong></div>
        <div class="row"><span class="key">Severity:</span> {sev_title}</div>
        <div class="row"><span class="key">Signal:</span> {crisis_title}</div>
        {confidence_html}
        {timestamp_html}
      </div>

      <div class="row"><span class="key">Excerpt:</span></div>
      <div class="excerpt">{excerpt}</div>

      <p style="margin:18px 0 6px;">
        <a class="btn" href="{case_url}" target="_blank" rel="noopener">Open Case</a>
      </p>
      <div class="meta">
        If the button doesn't work, copy and paste this link into your browser:<br/>
        <span>{case_url}</span>
      </div>
    </div>

    <div class="footer">
      This notification was generated by the {org} Safety Agent to assist with urgent care routing.
    </div>
  </div>
</body>
</html>
"""
    confidence_html = (
        "<div class='row'><span class='key'>Confidence:</span> {:.2f}</div>".format(confidence)
        if confidence is not None else ""
    )
    timestamp_html = (
        "<div class='row'><span class='key'>Detected at:</span> {}</div>".format(
            detected_at.strftime("%Y-%m-%d %H:%M")
        ) if detected_at else ""
    )

    html_body = HTML_TEMPLATE.format(
        subject=subject,
        org=ORG_NAME,
        logo_html=logo_html,
        sev_color=sev_color,
        sev_label=sev_label,
        crisis_label=crisis_label,
        patient=patient_name,
        sev_title=_title(sev),
        crisis_title=_title(crisis_type),
        confidence_html=confidence_html,
        timestamp_html=timestamp_html,
        excerpt=(snippet or "").strip()[:1000],
        case_url=case_url,
    )

    return subject, text_body, html_body


def render_password_reset_email(reset_link: str, expire_minutes: int) -> tuple[str, str, str]:
    """Return (subject, text_body, html_body) for a password-reset email."""
    subject = f"[{ORG_NAME}] Reset your password"

    text_body = (
        "We received a request to reset your {org} password.\n\n"
        "Reset it here (this link expires in {mins} minutes):\n{link}\n\n"
        "If you didn't request this, you can safely ignore this email — "
        "your password won't be changed.\n\n"
        "— {org}\n"
    ).format(org=ORG_NAME, mins=expire_minutes, link=reset_link)

    html_body = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="background:#f1f5f9; padding:20px; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Arial,sans-serif;">
  <div style="max-width:480px; margin:0 auto; background:#fff; border-radius:12px; padding:32px; box-shadow:0 2px 8px rgba(0,0,0,.06);">
    <h2 style="margin-top:0; color:#0f172a;">Reset your password</h2>
    <p style="color:#334155; line-height:1.55;">
      We received a request to reset your {org} password. This link expires in {mins} minutes.
    </p>
    <p style="margin:24px 0;">
      <a href="{link}" style="display:inline-block; background:#6d28d9; color:#fff; text-decoration:none; padding:12px 20px; border-radius:10px; font-weight:600;">
        Reset Password
      </a>
    </p>
    <p style="color:#64748b; font-size:13px;">
      If the button doesn't work, copy and paste this link into your browser:<br/>
      <span>{link}</span>
    </p>
    <p style="color:#94a3b8; font-size:12px; margin-top:24px;">
      If you didn't request this, you can safely ignore this email — your password won't be changed.
    </p>
  </div>
</body>
</html>
""".format(org=ORG_NAME, mins=expire_minutes, link=reset_link)

    return subject, text_body, html_body


def _send_via_resend(*, to_email: str, subject: str, text: str, html: str | None) -> bool:
    """Send via Resend's HTTPS API — used instead of SMTP when RESEND_API_KEY
    is set, since it works on hosts that block outbound SMTP ports."""
    payload = {
        "from": FROM_EMAIL,
        "to": [to_email],
        "subject": subject,
        "text": text,
    }
    if html:
        payload["html"] = html

    try:
        resp = requests.post(
            RESEND_API_URL,
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        if resp.status_code >= 400:
            logging.error(f"[email failed] to={to_email} via Resend: {resp.status_code} {resp.text}")
            return False
        logging.info(f"[email sent] to={to_email} via Resend")
        return True
    except Exception as e:
        logging.exception(f"[email failed] to={to_email} via Resend: {e}")
        return False


def _send_via_smtp(*, to_email: str, subject: str, text: str, html: str | None) -> bool:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    msg["X-Priority"] = "1"
    msg["X-MSMail-Priority"] = "High"
    msg["Importance"] = "High"

    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as s:
            s.starttls(context=context)
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        logging.info(f"[email sent] to={to_email} via {SMTP_HOST}:{SMTP_PORT}")
        return True
    except Exception as e:
        logging.exception(f"[email failed] to={to_email}: {e}")
        return False


def send_therapist_alert(*, to_email: str, subject: str, text: str, html: str | None = None) -> bool:
    """Send multipart (text + optional HTML) email.

    Uses Resend's HTTP API when RESEND_API_KEY is set (required on hosts that
    block outbound SMTP, e.g. Render) — otherwise falls back to direct SMTP.
    """
    logging.info(f"🚨 send_therapist_alert: Attempting to send email to {to_email} with subject: {subject}")
    logging.info(f"🚨 send_therapist_alert: EMAILS_ENABLED = {EMAILS_ENABLED}")

    if not EMAILS_ENABLED:
        logging.info(f"[email disabled] would send to={to_email} subj={subject!r}")
        return False

    if RESEND_API_KEY:
        return _send_via_resend(to_email=to_email, subject=subject, text=text, html=html)

    return _send_via_smtp(to_email=to_email, subject=subject, text=text, html=html)


# Despite the name, send_therapist_alert is a plain generic multipart-email
# sender — reuse it for any transactional email (password reset, etc.) rather
# than duplicating the SMTP logic.
send_email = send_therapist_alert