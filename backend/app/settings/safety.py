from typing import List
import re
from pydantic import validator
from .base import BaseAppSettings

class SafetySettings(BaseAppSettings):
    """Configuration for safety and content filtering."""
    
    # Crisis detection keywords
    crisis_keywords: List[str] = [
        "kill myself", "end my life", "suicide", "hurt myself", "harm myself",
        "i want to die", "i dont want to live", "take my life", "end it all",
        "life isn't worth", "no reason to live", "better off dead",
        "can't go on", "want to disappear", "nothing matters", "hopeless"
    ]
    
    # Other keyword lists
    substance_abuse_keywords: List[str] = [
        "overdose", "pills", "drugs", "alcohol abuse", "drinking problem",
        "getting high", "substance abuse", "addiction", "withdrawal"
    ]
    
    self_injury_keywords: List[str] = [
        "cutting", "burning", "scratching", "hitting myself",
        "self injury", "self harm", "hurting myself"
    ]
    
    illegal_content_keywords: List[str] = ["hack", "bomb", "weapon", "violence", "revenge"]
    
    jailbreak_keywords: List[str] = ["ignore instructions", "jailbreak", "pretend to be", "roleplay as", "act as if"]
    
    inappropriate_relationship_keywords: List[str] = ["romantic", "dating", "love you", "marry me", "kiss"]
    
    medical_advice_keywords: List[str] = ["diagnose", "medication dosage", "stop taking", "medical advice"]
    
    mental_health_indicators: List[str] = [
        "feel", "sad", "happy", "angry", "anxious", "stressed", "depressed", "worried",
        "help", "support", "talk", "problem", "difficult", "struggle", "emotion",
        "mental health", "therapy", "counseling", "coping"
    ]
    
    simple_greetings: List[str] = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you']
    
    # Regex patterns
    injection_patterns: List[str] = [
        r'ignore\s+previous\s+instructions',
        r'forget\s+everything',
        r'system\s*:',
        r'assistant\s*:',
        r'user\s*:',
        r'<\s*system\s*>',
        r'<\s*/\s*system\s*>',
        r'```\s*system',
        r'```\s*prompt'
    ]
    
    unsafe_output_patterns: List[str] = [
        r'how\s+to\s+make\s+(bomb|weapon|drug)',
        r'suicide\s+method',
        r'kill\s+yourself',
        r'harm\s+yourself',
        r'end\s+your\s+life'
    ]
    
    @validator('crisis_keywords', 'substance_abuse_keywords', 'self_injury_keywords', 
               'illegal_content_keywords', 'jailbreak_keywords', 'inappropriate_relationship_keywords',
               'medical_advice_keywords', 'mental_health_indicators', 'simple_greetings',
               pre=True)
    def parse_string_list(cls, v):
        """Parse comma-separated string into list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v
    
    @validator('injection_patterns', 'unsafe_output_patterns', pre=True)
    def parse_regex_list(cls, v):
        """Parse pipe-separated string into list of regex patterns."""
        if isinstance(v, str):
            return [item.strip() for item in v.split('|') if item.strip()]
        return v
    
    def get_compiled_patterns(self, pattern_type: str) -> List[re.Pattern]:
        """Get compiled regex patterns for a specific type.

        Accepts either a short key (e.g. "injection", "unsafe_output") or the full
        attribute name (e.g. "injection_patterns", "unsafe_output_patterns").
        """
        attr_map = {
            "injection": "injection_patterns",
            "injection_patterns": "injection_patterns",
            "unsafe_output": "unsafe_output_patterns",
            "unsafe_output_patterns": "unsafe_output_patterns",
        }
        attr = attr_map.get(pattern_type, f"{pattern_type}_patterns")
        patterns = getattr(self, attr, []) or []
        return [re.compile(p, re.IGNORECASE) for p in patterns]
    
    class Config:
        extra = "allow"  # Allow extra fields from environment