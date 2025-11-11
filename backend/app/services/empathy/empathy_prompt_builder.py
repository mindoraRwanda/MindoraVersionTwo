"""
Empathy Prompt Builder
======================
Creates natural, emotion-aware empathy guidance for LLM responses.

This module uses prompt engineering to teach the LLM empathy principles
rather than scripting rigid template responses. The LLM generates natural
responses informed by emotion-specific guidance, cultural context, and
intensity-based variation.

Based on:
- Motivational Interviewing (Miller & Rollnick, 2012)
- Person-Centered Therapy (Rogers, 1951)
- Ubuntu Philosophy (Rwandan cultural values)
- Crisis Intervention Best Practices
"""

from typing import Dict, Optional
from enum import Enum

from ..emotion.schemas import EmotionType, EmotionIntensity


# ============================================================================
# CULTURAL INTEGRATION: RWANDAN KINYARWANDA PHRASES
# ============================================================================

CULTURAL_PHRASES = {
    "turi_kumwe": {
        "phrase": "Turi kumwe",
        "meaning": "We are together",
        "usage": "Use for solidarity during sadness, anxiety, despair. Shows community support."
    },
    "buhoro_buhoro": {
        "phrase": "Buhoro buhoro",
        "meaning": "Slowly, slowly / Take your time",
        "usage": "Use for anxiety, fear, overwhelm. Encourages patience and self-compassion."
    },
    "imana_irabona": {
        "phrase": "Imana irabona",
        "meaning": "God sees (you are not alone)",
        "usage": "Use for despair, hopelessness, spiritual pain. Offers spiritual comfort."
    },
    "ejo_hazaza": {
        "phrase": "Ejo hazaza",
        "meaning": "Tomorrow will come",
        "usage": "Use for despair, hopelessness, when future seems bleak. Gentle hope."
    },
    "komera": {
        "phrase": "Komera",
        "meaning": "Be strong / Courage",
        "usage": "Use for anger, fear, challenging situations. Acknowledges inner strength."
    },
    "ubuntu": {
        "phrase": "Ubuntu (Umuntu ni umuntu ku bantu)",
        "meaning": "I am because we are / A person is a person through people",
        "usage": "Use for isolation, disconnection, identity struggles. Core cultural value."
    }
}


# ============================================================================
# EMPATHY PRINCIPLES: ALL 10 EMOTION TYPES
# ============================================================================

EMPATHY_PRINCIPLES = {
    EmotionType.SADNESS: {
        "core_principle": "Sadness is a valid emotional response that deserves space and acknowledgment, not dismissal or quick fixing.",
        "what_to_do": [
            "Acknowledge the weight of sadness without minimizing ('This sounds really hard')",
            "Validate that feeling sad makes sense given the situation",
            "Use 'Turi kumwe' (We are together) to show solidarity",
            "Gently explore what the sadness is connected to",
            "Offer reassurance that sadness doesn't mean weakness"
        ],
        "what_to_avoid": [
            "Rushing to 'cheer them up' or 'look on the bright side'",
            "Comparing their sadness to others ('others have it worse')",
            "Implying they should be over it by now",
            "Offering solutions before understanding the pain",
            "Using overly cheerful language that dismisses the emotion"
        ],
        "cultural_note": "In Rwandan culture, grief and sadness are honored through community presence. The phrase 'Turi kumwe' (we are together) reflects Ubuntu - you don't carry sadness alone.",
        "intensity_guidance": {
            "LOW": "Light acknowledgment, brief validation, gentle check-in",
            "MEDIUM": "Deeper validation, explore connections, offer cultural comfort",
            "HIGH": "Strong presence, extended validation, community solidarity ('Turi kumwe'), assess support needs",
            "CRITICAL": "Crisis assessment, immediate safety check, offer concrete support resources"
        }
    },
    
    EmotionType.ANXIETY: {
        "core_principle": "Anxiety is the body's alarm system trying to protect, but sometimes it fires when there's no real danger. Honor the feeling while gently grounding.",
        "what_to_do": [
            "Acknowledge the worry without amplifying it ('I hear how worried you are')",
            "Validate that anxiety makes sense (even if the threat isn't immediate)",
            "Use 'Buhoro buhoro' (slowly, slowly) to encourage patience",
            "Gently ground them in the present moment",
            "Explore what specific worries are underneath the anxiety"
        ],
        "what_to_avoid": [
            "Saying 'don't worry' or 'calm down' (invalidates the experience)",
            "Feeding the anxiety by catastrophizing with them",
            "Rushing them to 'just relax'",
            "Dismissing worries as 'not a big deal'",
            "Overwhelming with too many coping strategies at once"
        ],
        "cultural_note": "Rwandan wisdom teaches 'Buhoro buhoro' (slowly, slowly) - healing and peace come gradually, not by force. Patience with yourself is strength.",
        "intensity_guidance": {
            "LOW": "Brief acknowledgment, gentle grounding, normalize the feeling",
            "MEDIUM": "Deeper validation, explore specific worries, teach 'Buhoro buhoro'",
            "HIGH": "Strong grounding presence, validate intensity, offer calming techniques, check for panic",
            "CRITICAL": "Crisis assessment, immediate grounding, assess for panic attack or acute distress"
        }
    },
    
    EmotionType.ANGER: {
        "core_principle": "Anger is a messenger telling you something important - a boundary was crossed, a need wasn't met, or injustice occurred. Honor the message.",
        "what_to_do": [
            "Acknowledge the anger without judgment ('I can feel how angry this makes you')",
            "Validate that anger is a natural response to being hurt or wronged",
            "Use 'Komera' (be strong/courage) to acknowledge their inner strength",
            "Gently explore what's underneath the anger (hurt, fear, betrayal)",
            "Help channel anger toward understanding rather than destruction"
        ],
        "what_to_avoid": [
            "Telling them to 'calm down' or 'let it go'",
            "Judging anger as bad or inappropriate",
            "Escalating by agreeing with vengeful thoughts",
            "Dismissing the situation that caused the anger",
            "Rushing to problem-solving before validating the emotion"
        ],
        "cultural_note": "In Rwandan culture, 'Komera' (be strong) acknowledges that anger often comes from situations requiring courage. Your strength is not in suppressing anger, but in honoring what it teaches.",
        "intensity_guidance": {
            "LOW": "Brief acknowledgment, explore the trigger gently",
            "MEDIUM": "Deep validation, explore hurt underneath, teach 'Komera' principle",
            "HIGH": "Strong validation, safety assessment (harm to self/others), help channel anger constructively",
            "CRITICAL": "Crisis assessment, immediate safety check, de-escalation techniques, assess for violence risk"
        }
    },
    
    EmotionType.FEAR: {
        "core_principle": "Fear is your body's protection system. Sometimes it protects from real danger, sometimes from perceived threat. Either way, the fear is real and valid.",
        "what_to_do": [
            "Acknowledge the fear without amplifying it ('I hear how scared you are')",
            "Validate that fear makes sense given how they perceive the situation",
            "Use 'Komera' (be strong) to remind them of inner courage",
            "Gently assess if the fear is about present danger or future worry",
            "Help distinguish between feeling unsafe and being unsafe"
        ],
        "what_to_avoid": [
            "Saying 'there's nothing to be afraid of' (invalidates experience)",
            "Catastrophizing with them or feeding the fear",
            "Rushing them to 'just face it'",
            "Dismissing their perception of threat",
            "Overwhelming with exposure suggestions before building safety"
        ],
        "cultural_note": "Rwandan wisdom teaches 'Komera' (be strong) - courage is not the absence of fear, but moving forward even when afraid. Community stands with you in fear.",
        "intensity_guidance": {
            "LOW": "Brief acknowledgment, normalize fear response, gentle exploration",
            "MEDIUM": "Deeper validation, assess threat perception, teach 'Komera', build sense of safety",
            "HIGH": "Strong validation, assess if in immediate danger, grounding techniques, safety planning",
            "CRITICAL": "Crisis assessment, immediate safety check, assess for acute trauma, consider emergency resources"
        }
    },
    
    EmotionType.JOY: {
        "core_principle": "Joy is precious and deserves celebration. Honor happiness without diminishing it or rushing to worry about when it might end.",
        "what_to_do": [
            "Celebrate the joy with genuine warmth ('This is wonderful to hear!')",
            "Validate that they deserve this happiness",
            "Encourage savoring the moment rather than worrying about the future",
            "Gently explore what brought this joy (to help repeat it)",
            "Use cultural phrases like 'Ejo hazaza' (tomorrow will come) to affirm hope"
        ],
        "what_to_avoid": [
            "Immediately bringing up potential problems or worries",
            "Comparing their joy to others' struggles",
            "Minimizing the joy ('well, it's not that big a deal')",
            "Assuming they no longer need support because they feel good",
            "Being less engaged just because the emotion is positive"
        ],
        "cultural_note": "In Rwandan culture, joy is communal - 'Turi kumwe' (we are together) in celebration as in sorrow. Your happiness strengthens the whole community.",
        "intensity_guidance": {
            "LOW": "Warm acknowledgment, brief celebration",
            "MEDIUM": "Deep celebration, explore sources of joy, encourage savoring",
            "HIGH": "Full celebration, validate the feeling, help ground joy sustainably",
            "CRITICAL": "Assess for mania if joy seems disconnected from reality, otherwise celebrate fully"
        }
    },
    
    EmotionType.NEUTRAL: {
        "core_principle": "Neutral isn't 'nothing' - it can be peace, numbness, calm before exploring, or protection after pain. Meet them where they are.",
        "what_to_do": [
            "Acknowledge where they are without pushing for emotion ('I'm here with you')",
            "Gently explore if neutral is peaceful calm or protective numbness",
            "Create space for whatever might emerge without forcing it",
            "Validate that not every moment needs intense emotion",
            "Check in on their overall wellbeing without assuming neutral means 'fine'"
        ],
        "what_to_avoid": [
            "Assuming neutral means everything is okay",
            "Pushing them to feel something or open up before they're ready",
            "Filling silence with forced cheerfulness",
            "Dismissing the session as 'nothing to talk about'",
            "Interpreting neutral as disengagement or lack of need"
        ],
        "cultural_note": "Rwandan wisdom teaches 'Buhoro buhoro' (slowly, slowly) - sometimes neutral is the rest between storms, a necessary pause. Peace has its own value.",
        "intensity_guidance": {
            "LOW": "Light presence, gentle check-in, respect the space",
            "MEDIUM": "Deeper exploration of what neutral means for them, assess for numbness vs peace",
            "HIGH": "Assess if neutral is dissociation or emotional shutdown, check for underlying crisis",
            "CRITICAL": "Assess for severe dissociation, emotional detachment as crisis response"
        }
    },
    
    EmotionType.DESPAIR: {
        "core_principle": "Despair is profound hopelessness that needs deep validation and gentle, patient hope-building. Never rush it, never minimize it.",
        "what_to_do": [
            "Acknowledge the depth of pain without flinching ('This feels unbearable right now')",
            "Validate that despair makes sense given what they're experiencing",
            "Use 'Imana irabona' (God sees) for spiritual comfort, 'Turi kumwe' for solidarity",
            "CRITICAL: Assess for suicidal ideation immediately",
            "Offer very small, gentle reminders that feelings can shift (not that things aren't bad)"
        ],
        "what_to_avoid": [
            "Any form of toxic positivity ('it could be worse', 'think positive')",
            "Rushing them out of despair before validating the depth",
            "Quoting inspirational sayings that minimize pain",
            "Assuming they'll be fine without crisis assessment",
            "Leaving them alone in the despair without support resources"
        ],
        "cultural_note": "In Rwandan culture, 'Imana irabona' (God sees) acknowledges that even in deepest despair, you are not forgotten. 'Ejo hazaza' (tomorrow will come) offers hope without dismissing today's pain.",
        "intensity_guidance": {
            "LOW": "Deep validation, gentle hope-building, cultural comfort phrases",
            "MEDIUM": "Extended validation, explore sources of despair, 'Imana irabona', assess suicidal thoughts",
            "HIGH": "Crisis assessment required, suicide risk screening, 'Turi kumwe', immediate support resources",
            "CRITICAL": "IMMEDIATE crisis intervention, suicide prevention protocol, emergency resources, do not leave unsupported"
        }
    },
    
    EmotionType.SURPRISE: {
        "core_principle": "Surprise signals the unexpected - it can be positive, negative, or disorienting. Help process what happened and what it means.",
        "what_to_do": [
            "Acknowledge the unexpected nature of the situation ('That must have been surprising')",
            "Give space to process what just happened without rushing to judgment",
            "Gently explore if the surprise is positive, negative, or mixed",
            "Help them integrate the new information into their understanding",
            "Validate any confusion or disorientation that comes with surprise"
        ],
        "what_to_avoid": [
            "Immediately labeling the surprise as good or bad",
            "Rushing them to decide how they feel about it",
            "Dismissing the surprise as 'not a big deal'",
            "Overwhelming with analysis before they've processed the event",
            "Assuming surprise is always positive excitement"
        ],
        "cultural_note": "Rwandan wisdom teaches 'Buhoro buhoro' (slowly, slowly) - when life surprises you, take time to understand. Community helps make sense of the unexpected.",
        "intensity_guidance": {
            "LOW": "Brief acknowledgment, light processing of the unexpected",
            "MEDIUM": "Deeper exploration, help process implications, validate confusion",
            "HIGH": "Extended processing time, assess if surprise is traumatic shock, grounding if needed",
            "CRITICAL": "Assess for traumatic shock, acute stress response, dissociation from overwhelming surprise"
        }
    },
    
    EmotionType.DISGUST: {
        "core_principle": "Disgust signals a violation of values or boundaries - something feels deeply wrong. Honor this moral/visceral alarm without judgment.",
        "what_to_do": [
            "Acknowledge the strong reaction without judgment ('I can feel how wrong this feels to you')",
            "Validate that disgust is your body's way of protecting your values",
            "Gently explore what specific value or boundary was violated",
            "Help distinguish between the feeling and appropriate action",
            "Use 'Komera' (be strong) to acknowledge their moral clarity"
        ],
        "what_to_avoid": [
            "Judging them for having a strong reaction",
            "Telling them to 'get over it' or 'not be so sensitive'",
            "Agreeing with harsh judgments about people (vs behaviors)",
            "Dismissing the value violation that triggered disgust",
            "Pushing them to reconcile before processing the boundary violation"
        ],
        "cultural_note": "In Rwandan culture, 'Komera' (be strong) acknowledges that protecting your values requires courage. Disgust is your inner compass showing what you stand for.",
        "intensity_guidance": {
            "LOW": "Acknowledge the reaction, briefly explore the trigger",
            "MEDIUM": "Deeper validation, explore value violation, teach moral clarity vs judgment",
            "HIGH": "Strong validation, assess if trauma-related, help process boundary violation",
            "CRITICAL": "Assess if disgust is related to abuse, assault, or severe trauma - may require crisis intervention"
        }
    },
    
    EmotionType.HOPE: {
        "core_principle": "Hope is fragile and precious, especially after pain. Nurture it authentically without false promises or toxic positivity.",
        "what_to_do": [
            "Celebrate the emergence of hope with genuine warmth ('I hear hope in your words')",
            "Validate that hope is courage after difficulty",
            "Use 'Ejo hazaza' (tomorrow will come) to affirm hopeful outlook",
            "Gently explore what's nurturing this hope (to strengthen it)",
            "Acknowledge that hope and fear can coexist"
        ],
        "what_to_avoid": [
            "Immediately warning about potential disappointment",
            "Making promises you can't keep to amplify hope",
            "Assuming hope means all problems are solved",
            "Comparing their hope to past disappointments",
            "Rushing them to action before the hope is grounded"
        ],
        "cultural_note": "In Rwandan culture, 'Ejo hazaza' (tomorrow will come) acknowledges that hope is about trusting the future even when today is hard. Faith and community sustain hope.",
        "intensity_guidance": {
            "LOW": "Warm acknowledgment, brief encouragement",
            "MEDIUM": "Deeper celebration, explore sources of hope, connect to faith/community",
            "HIGH": "Full affirmation, help ground hope realistically, strengthen hope-building practices",
            "CRITICAL": "If hope seems disconnected from reality, assess for mania; otherwise celebrate authentically"
        }
    }
}


# ============================================================================
# EMPATHY PROMPT BUILDER CLASS
# ============================================================================

class EmpathyPromptBuilder:
    """
    Builds natural, emotion-aware empathy guidance for LLM responses.
    
    This class does NOT generate template responses. Instead, it creates
    prompts that teach the LLM how to respond empathetically based on:
    - Detected emotion type and intensity
    - Cultural context (Rwandan Kinyarwanda phrases)
    - Best practices from Motivational Interviewing and Person-Centered Therapy
    - 5-part empathy structure (acknowledge, validate, cultural, probe, reassure)
    
    The LLM generates natural language responses informed by these principles.
    """
    
    @staticmethod
    def build_empathy_prompt(
        emotion: EmotionType,
        intensity: EmotionIntensity,
        user_message: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Build emotion-aware empathy guidance prompt for LLM.
        
        Args:
            emotion: Detected emotion type (from classifier)
            intensity: Emotion intensity (LOW, MEDIUM, HIGH, CRITICAL)
            user_message: The actual message user sent
            context: Optional context (gender, language, conversation history)
        
        Returns:
            Natural language prompt teaching LLM how to respond empathetically
        """
        # Get empathy principles for this emotion
        principles = EMPATHY_PRINCIPLES.get(emotion)
        
        # Fallback if emotion not found (should not happen with all 10 covered)
        if not principles:
            principles = {
                "core_principle": "This emotion deserves acknowledgment and validation.",
                "what_to_do": ["Acknowledge the emotion", "Validate the experience", "Explore gently"],
                "what_to_avoid": ["Dismissing the feeling", "Rushing to solutions"],
                "cultural_note": "In Rwandan culture, 'Turi kumwe' (we are together) - no one faces emotions alone.",
                "intensity_guidance": {
                    "LOW": "Brief acknowledgment and validation",
                    "MEDIUM": "Deeper exploration and cultural connection",
                    "HIGH": "Strong presence and support",
                    "CRITICAL": "Crisis assessment and immediate support"
                }
            }
        
        # Get intensity-specific guidance
        intensity_guide = principles["intensity_guidance"].get(intensity.value, "Acknowledge and validate the emotion")
        
        # Select appropriate cultural phrases for this emotion
        cultural_suggestions = EmpathyPromptBuilder._select_cultural_phrases(emotion)
        
        # Build the empathy guidance prompt
        prompt = f"""
You are responding to a user experiencing {emotion.value} at {intensity.value} intensity.

USER'S MESSAGE: "{user_message}"

EMPATHY PRINCIPLES FOR {emotion.value.upper()}:
Core Understanding: {principles['core_principle']}

What to do:
{EmpathyPromptBuilder._format_list(principles['what_to_do'])}

What to avoid:
{EmpathyPromptBuilder._format_list(principles['what_to_avoid'])}

Cultural Context: {principles['cultural_note']}

Intensity Guidance ({intensity.value}): {intensity_guide}

CULTURAL PHRASES (Use naturally, not forced):
{cultural_suggestions}

RESPONSE STRUCTURE (5-part empathy):
1. ACKNOWLEDGE: Recognize their emotion without judgment
2. VALIDATE: Show their feeling makes sense
3. CULTURAL CONNECTION: Weave in Kinyarwanda phrase naturally if appropriate
4. GENTLE PROBE: Ask exploratory question (not interrogating)
5. REASSURANCE: Offer hope/support without false promises

REMEMBER:
- Generate a natural, human response - NOT a template or script
- Let empathy principles GUIDE you, but speak naturally
- Use Kinyarwanda phrases authentically, not decoratively
- Match intensity - don't be overly cheerful for HIGH intensity sadness
- For CRITICAL intensity: Prioritize safety assessment

Generate your empathetic response now:
"""
        
        return prompt.strip()
    
    @staticmethod
    def _select_cultural_phrases(emotion: EmotionType) -> str:
        """
        Select appropriate Kinyarwanda phrases for this emotion.
        
        Args:
            emotion: The detected emotion type
        
        Returns:
            Formatted string with relevant cultural phrases
        """
        # Map emotions to relevant cultural phrases
        emotion_phrase_map = {
            EmotionType.SADNESS: ["turi_kumwe", "buhoro_buhoro", "ejo_hazaza"],
            EmotionType.ANXIETY: ["buhoro_buhoro", "turi_kumwe"],
            EmotionType.ANGER: ["komera", "turi_kumwe"],
            EmotionType.FEAR: ["komera", "buhoro_buhoro", "turi_kumwe"],
            EmotionType.JOY: ["turi_kumwe", "ejo_hazaza"],
            EmotionType.NEUTRAL: ["buhoro_buhoro", "ubuntu"],
            EmotionType.DESPAIR: ["imana_irabona", "turi_kumwe", "ejo_hazaza"],
            EmotionType.SURPRISE: ["buhoro_buhoro", "turi_kumwe"],
            EmotionType.DISGUST: ["komera", "turi_kumwe"],
            EmotionType.HOPE: ["ejo_hazaza", "turi_kumwe", "imana_irabona"]
        }
        
        relevant_phrases = emotion_phrase_map.get(emotion, ["turi_kumwe"])
        
        phrase_descriptions = []
        for phrase_key in relevant_phrases:
            phrase_data = CULTURAL_PHRASES.get(phrase_key, {})
            phrase_descriptions.append(
                f"• {phrase_data.get('phrase', '')} ({phrase_data.get('meaning', '')}): {phrase_data.get('usage', '')}"
            )
        
        return "\n".join(phrase_descriptions)
    
    @staticmethod
    def _format_list(items: list) -> str:
        """Format list items with bullet points."""
        return "\n".join([f"• {item}" for item in items])
