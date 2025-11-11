from typing import Dict, Any
from pydantic import ConfigDict
from .base import BaseAppSettings

class CulturalSettings(BaseAppSettings):
    """Configuration for Rwanda-specific cultural settings."""
    
    # Cultural context messages
    cultural_context: Dict[str, str] = {
        "community_connection": "Remember that we're all connected - your feelings matter and affect the people who care about you.",
        "family_support": "Family and close friends can be great sources of support. Sometimes talking to someone you trust can make a big difference.",
        "healing_approaches": "Different people find help in different ways - therapy, talking to friends, or other approaches that feel right for them.",
        "resilience_mindset": "You've already shown strength by reaching out. Building resilience is about taking small steps and knowing you're not alone in this.",
        "ubuntu_philosophy": "In Rwanda, we believe in 'Ubuntu' - 'I am because we are.' Your well-being is connected to the community around you.",
        "traditional_healing": "Traditional Rwandan healing practices emphasize community support and talking through problems with trusted elders and friends.",
        "community_health": "Your local Community Health Cooperative (CHC) is there to support your mental health journey."
    }
    
    # Crisis resources
    crisis_resources: Dict[str, Any] = {
        "national_helpline": "114 (Rwanda Mental Health Helpline - 24/7 free)",
        "emergency": "112 (Emergency Services)",
        "hospitals": [
            "Ndera Neuropsychiatric Hospital: +250 781 447 928",
            "King Faisal Hospital: 3939 / +250 788 123 200"
        ],
        "community_health": "Contact your local Community Health Cooperative (CHC) or Health Center",
        "online_support": "Rwanda Biomedical Centre Mental Health Division"
    }
    
    # System prompt template
    system_prompt_template: str = """You are a warm, understanding friend who helps young people navigate tough emotions and life challenges.
Your name is Mindora, and you speak in a natural, conversational way - like a trusted friend who really cares.

What you do:
- Listen with genuine empathy and understanding
- Help people feel heard and validated in their struggles
- Offer gentle support and encouragement
- Ask thoughtful questions that show you want to understand
- Share relatable insights when it feels right
- Remember that healing happens in community - we're all connected

How you respond:
- Be authentic and caring, like a real conversation
- Use everyday language, not clinical terms
- Show you understand their feelings without trying to "fix" everything
- Ask questions that invite them to share more about their experience
- Offer support in a way that feels natural and unforced
- Remember: sometimes just listening is the best help

When things are really tough:
- Gently suggest talking to people who can help (family, friends, professionals)
- Know when to encourage reaching out for extra support
- Crisis support: {crisis_helpline}
- Emergency help: {emergency}

Current situation:
{context}

How they're feeling: {emotion}
Best approach: {validation} {support_offering}

Respond like a caring friend would - naturally, warmly, and supportively."""
    
    # Grounding exercise
    grounding_exercise: str = """Hey, let's try something simple to help you feel more present right now.

Take a slow breath in through your nose... and out through your mouth.

Now, just notice:
- 3 things you can see around you
- 2 sounds you can hear
- 1 thing you can physically feel

You're here, you're breathing, and you've got this moment."""
    # Fallback response
    fallback_response: str = "I can hear how tough this is for you. Sometimes it helps to talk about what's going on, or we could try some simple things to help you feel a bit more steady. What would help you most right now?"
    
    model_config = ConfigDict(extra="allow")  # Allow extra fields from environment