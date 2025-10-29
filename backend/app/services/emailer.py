import os
import ssl
import smtplib
import logging
from email.message import EmailMessage
from datetime import datetime

# ---- SMTP / Mailtrap defaults ----
SMTP_HOST = os.getenv("SMTP_HOST", "sandbox.smtp.mailtrap.io")
SMTP_PORT = int(os.getenv("SMTP_PORT", "2525"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_EMAIL = os.getenv("ALERTS_FROM", "alerts@mindora.local")
EMAILS_ENABLED = os.getenv("EMAILS_ENABLED", "1") == "1"

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


def send_therapist_alert(*, to_email: str, subject: str, text: str, html: str | None = None) -> bool:
    """Send multipart (text + optional HTML) email."""
    if not EMAILS_ENABLED:
        logging.info(f"[email disabled] would send to={to_email} subj={subject!r}")
        return False

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
