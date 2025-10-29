from typing import Dict
from .base import BaseAppSettings

class EmotionalResponseSettings(BaseAppSettings):
    """Configuration for emotion-specific response guidance."""
    
    emotion_guidance: Dict[str, Dict[str, str]] = {
        "sadness": {
            "response_style": "gentle_presence",
            "validation_approach": "Show that you understand their sadness is real and valid",
            "exploration_style": "Ask gently about what's making them feel this way",
            "support_style": "Be there to listen without trying to fix everything at once",
            "natural_tone": "Like a caring friend who gets that sometimes life just hurts"
        },
        "anxiety": {
            "response_style": "calming_reassurance",
            "validation_approach": "Let them know anxiety is common and their feelings make sense",
            "exploration_style": "Help them identify what's triggering these anxious feelings",
            "support_style": "Offer practical ways to feel more grounded and less overwhelmed",
            "natural_tone": "Like a friend who understands how anxiety can make everything feel too much"
        },
        "stress": {
            "response_style": "understanding_support",
            "validation_approach": "Acknowledge how heavy their load feels right now",
            "exploration_style": "Help them figure out what's weighing on them most",
            "support_style": "Offer to help break things down into smaller, manageable pieces",
            "natural_tone": "Like a friend who sees they're struggling and wants to help share the weight"
        },
        "anger": {
            "response_style": "patient_understanding",
            "validation_approach": "Recognize that anger often comes from feeling hurt or scared",
            "exploration_style": "Gently help them understand what set off these angry feelings",
            "support_style": "Support them in expressing and processing their anger in healthy ways",
            "natural_tone": "Like a friend who knows anger is usually about something deeper that hurts"
        },
        "fear": {
            "response_style": "reassuring_presence",
            "validation_approach": "Let them know it's okay to feel scared about uncertain things",
            "exploration_style": "Help them talk about what specifically feels scary or unsafe",
            "support_style": "Offer steady reassurance and help them feel braver about facing fears",
            "natural_tone": "Like a friend who understands that fear is normal and wants to help them feel safer"
        }
    }
    
    topic_guidance: Dict[str, Dict[str, str]] = {
        "school": {
            "context_understanding": "School can be really tough with all the pressure and expectations",
            "exploration_approach": "Ask how school stuff is affecting their daily life and feelings",
            "natural_approach": "Talk about school like any friend would - no judgment, just understanding",
            "support_focus": "Help find ways to manage school stress while taking care of themselves"
        },
        "university": {
            "context_understanding": "University brings a lot of changes and new kinds of pressure",
            "exploration_approach": "Explore how uni life is affecting their emotional and social world",
            "natural_approach": "Relate to university challenges like a friend who's been through it",
            "support_focus": "Help navigate the ups and downs of university life and independence"
        },
        "family": {
            "context_understanding": "Family stuff can be complicated and affect everything else",
            "exploration_approach": "Help them talk about family relationships and conflicts naturally",
            "natural_approach": "Approach family topics with warmth and without taking sides",
            "support_focus": "Support them in figuring out family dynamics while honoring their own needs"
        },
        "work": {
            "context_understanding": "Work stress can make everything else feel harder",
            "exploration_approach": "Explore how work is impacting their well-being and relationships",
            "natural_approach": "Talk about work challenges like friends sharing about tough days",
            "support_focus": "Help find balance between work responsibilities and personal well-being"
        },
        "relationships": {
            "context_understanding": "Relationships can bring joy but also hurt and confusion",
            "exploration_approach": "Help them explore relationship dynamics with care and openness",
            "natural_approach": "Talk about relationships like a trusted friend giving advice",
            "support_focus": "Support healthy relationship patterns while respecting their feelings"
        }
    }
    
    response_templates: Dict[str, str] = {
        "crisis_validation": "Generate a response that validates their crisis feelings while emphasizing immediate help-seeking",
        "support_offering": "Create an offer of support that feels genuine and natural",
        "question_asking": "Ask questions that show genuine interest in understanding their experience",
        "resource_mentioning": "Mention local resources naturally when they would be helpful, not as a default response"
    }
    
    @property
    def emotion_responses(self) -> Dict[str, Dict[str, str]]:
        """Backward compatibility property for emotion_responses."""
        return self.emotion_guidance
    
    @property
    def topic_adjustments(self) -> Dict[str, Dict[str, str]]:
        """Backward compatibility property for topic_adjustments."""
        return self.topic_guidance
    
    class Config:
        extra = "allow"  # Allow extra fields from environment