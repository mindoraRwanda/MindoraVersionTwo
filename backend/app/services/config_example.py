#!/usr/bin/env python3
"""
Example script demonstrating the new generative configuration system.

This script shows how to:
1. Use the new structured configuration classes for generative responses
2. Load configuration from external files
3. Override configuration with environment variables
4. Understand how the LLM uses configuration as guidance for custom responses
"""

import os
from llm_config import (
    ConfigManager, ModelConfig, PerformanceConfig, SafetyConfig,
    EmotionalResponseConfig, model_config, performance_config,
    safety_config, emotional_config, rwanda_config
)


def demonstrate_generative_configuration():
    """Demonstrate how configuration drives generative responses."""

    print("🎭 Gender-Aware Rwandan Cultural Wisdom & Support System Demo")
    print("=" * 60)

    # 1. Show how configuration provides guidance for custom responses
    print("\n🧠 Configuration as Response Guidance:")
    print("   The configuration no longer contains response templates,")
    print("   but provides guidance for the LLM to generate custom responses.")

    # Show emotion guidance structure
    sadness_guidance = emotional_config.emotion_guidance["sadness"]
    print(f"\n   📋 Sadness Response Guidance:")
    print(f"      Response Style: {sadness_guidance['response_style']}")
    print(f"      Validation Approach: {sadness_guidance['validation_approach']}")
    print(f"      Natural Tone: {sadness_guidance['natural_tone']}")

    # 2. Demonstrate configuration manager
    print("\n🏭 Configuration Manager:")
    config_manager = ConfigManager()

    # Show how to access structured configs
    emotion_cfg = config_manager.get_emotional_config()
    rwanda_cfg = config_manager.get_rwanda_config()

    print(f"   Emotion Guidance Keys: {len(emotion_cfg.emotion_guidance)}")
    print(f"   Topic Guidance Keys: {len(emotion_cfg.topic_guidance)}")
    print(f"   Response Guidance Keys: {len(rwanda_cfg.cultural_context)}")

    # 3. Show how LLM uses this for generative responses
    print("\n🤖 How LLM Uses Configuration:")

    system_prompt = rwanda_cfg.system_prompt_template
    print("   📝 System Prompt instructs LLM to:")
    print("      • Generate original, personalized responses")
    print("      • Use emotional context as guidance")
    print("      • Speak like a culturally aware Rwandan elder (gender-matched)")
    print("      • Use gender-appropriate Kinyarwanda addressing (murumuna for brothers, murumuna wanjye for sisters)")
    print("      • Draw on Ubuntu wisdom and cultural understanding")
    print("      • Keep responses conversational, like talking to an elder or trusted friend")

    # 4. Demonstrate emotion-specific guidance
    print("\n🎭 Emotion-Specific Guidance Examples:")

    for emotion, guidance in list(emotion_cfg.emotion_guidance.items())[:3]:
        print(f"\n   {emotion.upper()} Guidance:")
        print(f"      Style: {guidance['response_style']}")
        print(f"      Cultural Note: {guidance['cultural_consideration']}")

    # 5. Show topic-specific guidance
    print("\n📚 Topic-Specific Guidance Examples:")

    for topic, guidance in list(emotion_cfg.topic_guidance.items())[:2]:
        print(f"\n   {topic.upper()} Guidance:")
        print(f"      Context: {guidance['context_understanding']}")
        print(f"      Cultural Perspective: {guidance['cultural_perspective']}")

    # 6. Demonstrate configuration flexibility
    print("\n🔧 Configuration Flexibility:")

    # Show how to modify configuration at runtime
    custom_emotion_config = EmotionalResponseConfig()
    custom_emotion_config.emotion_guidance["joy"] = {
        "response_style": "celebratory_appreciation",
        "validation_approach": "Celebrate positive emotions as part of healing journey",
        "exploration_style": "Explore what specifically brought joy and how to cultivate more",
        "support_style": "Encourage maintaining sources of joy and building resilience",
        "cultural_consideration": "Connect joy to Ubuntu - shared happiness strengthens community bonds"
    }

    print(f"   ✅ Added custom emotion guidance for 'joy'")
    print(f"   Total Emotion Types: {len(custom_emotion_config.emotion_guidance)}")

    # 7. Demonstrate configuration export
    print("\n📤 Configuration Export:")

    # Export current configuration to a new file
    export_path = "backend/app/services/generative_config.json"
    config_manager.export_config(export_path)
    print(f"   ✅ Generative configuration exported to {export_path}")

    print("\n🎯 Key Benefits of Generative Configuration:")
    print("   ✅ LLM generates original responses based on guidance")
    print("   ✅ Configuration provides context, not templates")
    print("   ✅ Emotionally intelligent response generation")
    print("   ✅ Gender-aware Rwandan cultural wisdom and personalized connection")
    print("   ✅ Flexible and extensible configuration system")
    print("   ✅ Type-safe configuration management")
    print("   ✅ Easy customization for different use cases")

    print("\n💡 How It Works:")
    print("   1. Configuration provides response guidance and emotional context")
    print("   2. LLM receives user's message and emotional analysis")
    print("   3. System prompt instructs LLM to generate custom responses")
    print("   4. Configuration guidance informs the tone and approach")
    print("   5. Result: Personalized, culturally resonant responses")

    print("\n🚀 Usage Examples:")
    print("   # Set custom config file")
    print("   export LLM_CONFIG_FILE='path/to/generative_config.json'")
    print("   ")
    print("   # Override specific values")
    print("   export LLM_TEMPERATURE='0.8'")
    print("   ")
    print("   # Use in code")
    print("   from llm_config import config_manager")
    print("   emotion_cfg = config_manager.get_emotional_config()")
    print("   rwanda_cfg = config_manager.get_rwanda_config()")


def show_response_generation_example():
    """Show how the LLM would use configuration to generate responses."""

    print("\n\n🗣️  Response Generation Example:")
    print("=" * 60)

    # Simulate user input
    user_message = "I'm feeling really sad about my exam results"
    detected_emotion = "sadness"

    print(f"User Message: '{user_message}'")
    print(f"Detected Emotion: {detected_emotion}")

    # Show how configuration would guide the response
    sadness_guidance = emotional_config.emotion_guidance[detected_emotion]

    print(f"\n📋 Configuration Guidance for Response:")
    print(f"   Response Style: {sadness_guidance['response_style']}")
    print(f"   Validation: {sadness_guidance['validation_approach']}")
    print(f"   Natural Tone: {sadness_guidance['natural_tone']}")

    print(f"\n🤖 Gender-Aware Examples:")
    print("   For Male User: 'Murumuna, I can really hear how disappointing those exam results feel...")
    print("   For Female User: 'Murumuna wanjye, I can really hear how disappointing those exam results feel...")
    print("   ")
    print("   'It's completely okay to feel sad about this - it shows how much you care about doing well.")
    print("    What's been weighing on you most about these results?'")

    print(f"\n✅ Result:")
    print("   • Original response (not copied from template)")
    print("   • Natural and conversational")
    print("   • Emotionally validating and supportive")
    print("   • Guided by configuration but generated by LLM")


if __name__ == "__main__":
    demonstrate_generative_configuration()
    show_response_generation_example()