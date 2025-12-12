"""Slot management utilities."""
from typing import Dict, Any
from enum import Enum


class YesNoUnknown(str, Enum):
    yes = "yes"
    no = "no"
    unknown = "unknown"


class AppetiteDelta(str, Enum):
    increase = "increase"
    decrease = "decrease"
    none = "none"
    unknown = "unknown"


class SocialSupportLevel(str, Enum):
    none = "none"
    low = "low"
    moderate = "moderate"
    strong = "strong"
    unknown = "unknown"


class Stressor(str, Enum):
    exams = "exams"
    assignments = "assignments"
    family_conflict = "family_conflict"
    bullying = "bullying"
    financial = "financial"
    grief = "grief"
    health = "health"
    relationship = "relationship"
    other = "other"


def coerce_yes_no_unknown(v) -> str:
    """Coerce value to yes/no/unknown."""
    if v is None:
        return YesNoUnknown.unknown.value
    s = str(v).strip().lower()
    if s in ("yes", "true", "y", "1", "yeah"):
        return YesNoUnknown.yes.value
    if s in ("no", "false", "n", "0", "nope"):
        return YesNoUnknown.no.value
    return YesNoUnknown.unknown.value


def coerce_appetite(v) -> str:
    """Coerce value to appetite change enum."""
    if v is None:
        return AppetiteDelta.unknown.value
    s = str(v).strip().lower()
    if "increase" in s or "more" in s or "gain" in s:
        return AppetiteDelta.increase.value
    if "decrease" in s or "less" in s or "low" in s or "skip" in s:
        return AppetiteDelta.decrease.value
    if "none" in s or "normal" in s or "same" in s:
        return AppetiteDelta.none.value
    return AppetiteDelta.unknown.value


def coerce_social(v) -> str:
    """Coerce value to social support level."""
    if v is None:
        return SocialSupportLevel.unknown.value
    s = str(v).strip().lower()
    if "none" in s or "no friends" in s or "alone" in s:
        return SocialSupportLevel.none.value
    if "low" in s or "few" in s or "weak" in s:
        return SocialSupportLevel.low.value
    if "moderate" in s or "some" in s or "okay" in s:
        return SocialSupportLevel.moderate.value
    if "strong" in s or "good" in s or "church" in s or "family" in s or "mentor" in s:
        return SocialSupportLevel.strong.value
    return SocialSupportLevel.unknown.value


def coerce_stressors(arr) -> list[str]:
    """Coerce array to stressor enum values."""
    if not arr:
        return []
    out = []
    for item in arr:
        s = str(item).strip().lower()
        mapping = {
            "exam": "exams", "exams": "exams",
            "assignment": "assignments", "assignments": "assignments",
            "family": "family_conflict", "family conflict": "family_conflict",
            "bully": "bullying", "bullying": "bullying",
            "money": "financial", "finance": "financial", "financial": "financial",
            "grief": "grief", "loss": "grief",
            "health": "health", "sick": "health",
            "relationship": "relationship", "breakup": "relationship"
        }
        val = mapping.get(s, "other")
        try:
            Stressor(val)  # Validate
            out.append(val)
        except Exception:
            out.append(Stressor.other.value)
    # Dedupe but keep order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def bound_int(v, lo, hi):
    """Bound integer value between lo and hi."""
    if v is None:
        return None
    try:
        n = int(v)
        return max(lo, min(hi, n))
    except Exception:
        return None


def apply_slot_updates(current_slots: Dict[str, Any], slot_updates: Dict[str, Any]) -> Dict[str, Any]:
    """Apply slot updates to current slots, returning updated slots."""
    updated = current_slots.copy()
    
    if "sleepIssues" in slot_updates:
        updated["sleepIssues"] = coerce_yes_no_unknown(slot_updates["sleepIssues"])
    
    if "lastSleptHours" in slot_updates:
        v = bound_int(slot_updates["lastSleptHours"], 0, 24)
        updated["lastSleptHours"] = v
        # If hours are small/zero, also mark sleepIssues=yes
        if v is not None and v <= 4:
            updated["sleepIssues"] = YesNoUnknown.yes.value
    
    if "exerciseWeekly" in slot_updates:
        updated["exerciseWeekly"] = bound_int(slot_updates["exerciseWeekly"], 0, 14)
    
    if "appetiteChange" in slot_updates:
        updated["appetiteChange"] = coerce_appetite(slot_updates["appetiteChange"])
    
    if "concentrationIssues" in slot_updates:
        updated["concentrationIssues"] = coerce_yes_no_unknown(slot_updates["concentrationIssues"])
    
    if "socialSupport" in slot_updates:
        updated["socialSupport"] = coerce_social(slot_updates["socialSupport"])
    
    if "stressors" in slot_updates:
        # Overwrite with latest deduped list (last-write-wins)
        updated["stressors"] = coerce_stressors(slot_updates["stressors"])
    
    return updated


def get_default_slots() -> Dict[str, Any]:
    """Get default diagnostic slots."""
    return {
        "sleepIssues": YesNoUnknown.unknown.value,
        "appetiteChange": AppetiteDelta.unknown.value,
        "concentrationIssues": YesNoUnknown.unknown.value,
        "socialSupport": SocialSupportLevel.unknown.value,
        "stressors": [],
        "exerciseWeekly": None,
        "lastSleptHours": None
    }

