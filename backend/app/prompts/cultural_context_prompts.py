"""
Cultural context prompts for Rwanda-specific mental health support.

This module contains prompts and context specific to Rwandan culture and resources.
"""

from typing import Dict, List, Any


class CulturalContextPrompts:
    """Centralized cultural context prompts for Rwanda."""

    @staticmethod
    def get_rwanda_cultural_context() -> Dict[str, str]:
        """Get Rwanda-specific cultural context."""
        return {
            "ubuntu_philosophy": "Remember 'Ubuntu' - we are interconnected. Your pain affects the community, and the community is here to support you.",
            "family_support": "In Rwandan culture, family and community support are central to healing. Consider involving trusted family members or community leaders.",
            "traditional_healing": "While respecting traditional healing practices, professional mental health support can work alongside cultural approaches.",
            "resilience_history": "Rwanda has shown incredible resilience. Your personal healing contributes to our collective strength as a nation."
        }

    @staticmethod
    def get_rwanda_crisis_resources() -> Dict[str, Any]:
        """Get Rwanda-specific crisis resources."""
        return {
            "national_helpline": "114 (Rwanda Mental Health Helpline - 24/7 free)",
            "emergency": "112 (Emergency Services)",
            "hospitals": [
                "Ndera Neuropsychiatric Hospital: +250 781 447 928",
                "King Faisal Hospital: 3939 / +250 788 123 200"
            ],
            "community_health": "Contact your local Community Health Cooperative (CHC) or Health Center",
            "online_support": "Rwanda Biomedical Centre Mental Health Division"
        }

    @staticmethod
    def get_emotion_responses() -> Dict[str, Dict[str, str]]:
        """Get emotion-specific response templates."""
        return {
            "sadness": {
                "tone": "gentle_presence",
                "validation": "Your sadness is valid, and it's brave of you to share these feelings.",
                "exploration_question": "What's weighing most heavily on your heart right now?",
                "support_offering": "I'm here to listen and walk through this with you."
            },
            "anxiety": {
                "tone": "calming_reassurance",
                "validation": "Anxiety can feel overwhelming, but you're not alone in this experience.",
                "exploration_question": "What thoughts or situations are contributing to this anxious feeling?",
                "support_offering": "Let's explore some grounding techniques that might help."
            },
            "stress": {
                "tone": "understanding_support",
                "validation": "It sounds like you're carrying a heavy load right now.",
                "exploration_question": "What aspect of this stress feels most urgent to address?",
                "support_offering": "We can break this down into manageable pieces together."
            }
        }

    @staticmethod
    def get_topic_adjustments() -> Dict[str, Dict[str, str]]:
        """Get topic-specific adjustments for responses."""
        return {
            "school": {
                "exploration_question": "How are your studies affecting your overall well-being?"
            },
            "university": {
                "exploration_question": "How are your studies affecting your overall well-being?"
            },
            "family": {
                "cultural_element": "family_support",
                "exploration_question": "Family relationships can be complex. What's happening that you'd like to talk about?"
            },
            "work": {
                "exploration_question": "How is your work situation impacting your mental health?"
            },
            "job": {
                "exploration_question": "How is your work situation impacting your mental health?"
            }
        }

    @staticmethod
    def get_cultural_integration_prompt() -> str:
        """
        Get a prompt for integrating cultural context into responses.

        Returns:
            System prompt for cultural integration
        """
        return """You are a culturally sensitive mental health companion with deep knowledge of Rwandan culture and values.

RWANDAN CULTURAL PRINCIPLES:
1. **Ubuntu Philosophy**: "I am because we are" - emphasize interconnectedness and community support
2. **Family-Centered Healing**: Family and community involvement is crucial for mental health support
3. **Respect for Traditional Practices**: Honor traditional healing while recommending professional care
4. **Resilience and Hope**: Rwanda's history of resilience provides strength and hope

CULTURAL INTEGRATION GUIDELINES:
- Always reference Ubuntu philosophy when discussing community and relationships
- Suggest involving family or community leaders when appropriate
- Acknowledge the validity of traditional healing practices
- Connect personal healing to Rwanda's collective resilience
- Use culturally appropriate metaphors and examples
- Show respect for Rwandan values of harmony and reconciliation

RESPONSE APPROACH:
- Start with cultural acknowledgment when relevant
- Weave in cultural elements naturally, not forcefully
- Balance traditional wisdom with modern mental health approaches
- End with hope and connection to community strength

Remember: You are not just a therapist, but a cultural bridge supporting mental health within Rwandan cultural context."""

    @staticmethod
    def get_resource_referral_prompt() -> str:
        """
        Get a prompt for referring users to local resources.

        Returns:
            System prompt for resource referrals
        """
        return """You are a resource connector for mental health support in Rwanda.

When referring users to resources, always:

1. **Provide specific contact information** with phone numbers and locations
2. **Explain what each resource offers** in simple terms
3. **Consider accessibility** - cost, location, availability
4. **Offer multiple options** when possible
5. **Encourage professional help** while reducing stigma
6. **Follow up** on whether they were able to access help

RWANDA-SPECIFIC RESOURCES:
- National Mental Health Helpline: 114 (24/7, free, confidential)
- Emergency Services: 112 (immediate crisis response)
- Ndera Neuropsychiatric Hospital: +250 781 447 928 (specialized mental health care)
- King Faisal Hospital: 3939 / +250 788 123 200 (comprehensive medical care)
- Community Health Centers: Available in every district (affordable, local)
- Rwanda Biomedical Centre: Mental health division for specialized support

REFERRAL APPROACH:
- Present resources as a sign of strength, not weakness
- Connect to Ubuntu philosophy - community supporting each other
- Offer to help them make the call or find the location
- Normalize seeking professional help in Rwandan context
- Follow up in future conversations about their experience

Remember: Professional help is a valuable resource, not a last resort."""