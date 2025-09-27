# from langchain_community.chat_models import ChatOllama 
from langchain_ollama import ChatOllama # use this instead of the above import because of deprecation
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from typing import Dict, List, Any, Optional
import requests
import re
import os
from backend.app.services.retriever_service import RetrieverService
from backend.app.services.emotion_classifier import classify_emotion as classify_emotion_legacy
from backend.app.services.emotion_service import classify_emotion
from backend.app.services.chatbot_insights_pipeline import detect_medication_mentions, detect_suicide_risk
from backend.app.db.database import get_db
from backend.app.db.models import Message, Conversation
from sqlalchemy.orm import Session

# Guardrails
from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.actions import action


# --- Rwanda-Specific Mental Health Resources ---
class RwandaMentalHealthResources:
    @staticmethod
    def get_crisis_resources():
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
    def get_cultural_context():
        return {
            "ubuntu_philosophy": "Remember 'Ubuntu' - we are interconnected. Your pain affects the community, and the community is here to support you.",
            "family_support": "In Rwandan culture, family and community support are central to healing. Consider involving trusted family members or community leaders.",
            "traditional_healing": "While respecting traditional healing practices, professional mental health support can work alongside cultural approaches.",
            "resilience_history": "Rwanda has shown incredible resilience. Your personal healing contributes to our collective strength as a nation."
        }

# --- Enhanced Intent Classifier for Guardrails ---
@action()
async def classify_intent_with_ollama(last_user_message: str):
    lowered = last_user_message.lower().strip()
    
    # Enhanced self-harm detection
    crisis_keywords = [
        "kill myself", "end my life", "suicide", "hurt myself", "harm myself",
        "i want to die", "i dont want to live", "take my life", "end it all",
        "life isn't worth", "no reason to live", "better off dead",
        "can't go on", "want to disappear", "nothing matters", "hopeless",

    ]
    
    # Substance abuse indicators
    substance_keywords = [
        "overdose", "pills", "drugs", "alcohol abuse", "drinking problem",
        "getting high", "substance abuse", "addiction", "withdrawal"
    ]
    
    # Self-injury patterns
    self_injury_keywords = [
        "cutting", "burning", "scratching", "hitting myself",
        "self injury", "self harm", "hurting myself"
    ]
    
    # Check for crisis situations
    for kw in crisis_keywords:
        if kw in lowered:
            return "self_harm"
    
    for kw in substance_keywords:
        if kw in lowered:
            return "substance_abuse"
            
    for kw in self_injury_keywords:
        if kw in lowered:
            return "self_injury"
    
    # Enhanced illegal content detection
    if any(k in lowered for k in ["hack", "bomb", "weapon", "violence", "revenge"]):
        return "illegal"
    
    # Enhanced jailbreak detection
    if any(k in lowered for k in ["ignore instructions", "jailbreak", "pretend to be", "roleplay as", "act as if"]):
        return "jailbreak"
    
    # Inappropriate relationship boundaries
    if any(k in lowered for k in ["romantic", "dating", "love you", "marry me", "kiss"]):
        return "inappropriate_relationship"
    
    # Medical advice seeking
    if any(k in lowered for k in ["diagnose", "medication dosage", "stop taking", "medical advice"]):
        return "medical_advice"
    
    # Off-topic requests (non-mental health) - simple boundary check
    mental_health_indicators = [
        "feel", "sad", "happy", "angry", "anxious", "stressed", "depressed", "worried",
        "help", "support", "talk", "problem", "difficult", "struggle", "emotion",
        "mental health", "therapy", "counseling", "coping"
    ]
    
    # If message is long and contains no mental health indicators, likely off-topic
    if len(lowered) > 30 and not any(indicator in lowered for indicator in mental_health_indicators):
        return "off_topic"
    
    return "general"

class LLMService:
    #def __init__(self, model_name: str = "llama3", use_vllm: bool = False):
    def __init__(self, model_name: str = None, use_vllm: bool = False):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "gemma3:1b")  # Default to "gemma3:1b" if not specified
        self.use_vllm = use_vllm
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.vllm_base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")  #  # vLLM OpenAI-compatible endpoint
        self.chat_model = None
        self.retriever = RetrieverService()
        self.is_initialized = False
        self.rails = None
        self._initialize_model()
        self._initialize_guardrails()

    def _initialize_model(self):
        try:
            if self.use_vllm:
                if not self._is_vllm_running():
                    print("vLLM is not running. Please start: docker-compose up vllm")
                    return
                # Use ChatOllama with vLLM's OpenAI-compatible API
                self.chat_model = ChatOllama(
                    base_url=self.vllm_base_url,
                    model=self.model_name,
                    temperature=0.7
                )
                self.is_initialized = True
                print("LLM initialized with vLLM (Llama-3-8B-Instruct)")
            else:
                if not self._is_ollama_running():
                    print("Ollama is not running. Please start the Ollama server.")
                    return
                if not self._is_model_available():
                    #print(f"Model '{self.model_name}' is not available. Please run: ollama pull {self.model_name}")
                    print(f"Model '{self.model_name}' is not available. Please run: ollama pull {self.model_name}")
                    return
                #self.chat_model = ChatOllama(model=self.model_name, temperature=0.7)
                self.chat_model = ChatOllama(
                    base_url=self.ollama_base_url,
                    model=self.model_name,
                    temperature=0.7
                )
                self.is_initialized = True
                print(f"LLM initialized with Ollama model: {self.model_name}")
        except Exception as e:
            print(f"Error during model initialization: {e}")

    def _initialize_guardrails(self):
        rwanda_resources = RwandaMentalHealthResources()
        crisis_resources = rwanda_resources.get_crisis_resources()
        cultural_context = rwanda_resources.get_cultural_context()
        
        flows_config = f"""
        colang: |
          flow safety_and_boundary_check
            user ".*"
            call classify_intent_with_ollama $last_user_message
            
            if $result == "self_harm":
              bot "**I'm deeply concerned** about what you're sharing. Your life has **value** and you don't have to face this alone.\n\n**Please reach out for immediate professional help:**\n- {crisis_resources['national_helpline']}\n- Emergency: {crisis_resources['emergency']}\n- Visit: {crisis_resources['hospitals'][0]}\n\n> {cultural_context['ubuntu_philosophy']}\n\nI'm here to support you, but professional crisis support is essential right now."
              stop
            elif $result == "substance_abuse":
              bot "I understand you're dealing with substance-related concerns. This is a **serious matter** that needs professional support.\n\n**Please contact:**\n- {crisis_resources['national_helpline']} for mental health and addiction support\n- Visit your nearest health center\n\n> {cultural_context['family_support']}\n\nProfessional treatment combined with community support can make a real difference."
              stop
            elif $result == "self_injury":
              bot "I'm concerned about the self-harm you're describing. These feelings are **valid**, but hurting yourself isn't the solution.\n\n**Please reach out:**\n- {crisis_resources['national_helpline']}\n- {crisis_resources['emergency']} for immediate support\n\n> {cultural_context['traditional_healing']}\n\nLet's work together on healthier ways to manage these difficult emotions."
              stop
            elif $result == "illegal":
              bot "**I can't help with that.** I'm here only to support mental health and well-being in a safe, constructive way."
              stop
            elif $result == "jailbreak":
              bot "**I can't override my safety instructions.** I'm here only to support mental health and your well-being.\n\nLet's focus on how I can help you today."
              stop
            elif $result == "inappropriate_relationship":
              bot "I'm here as a **supportive mental health companion**, not for romantic or personal relationships.\n\nLet's keep our conversation focused on supporting your well-being and mental health. How can I help you today?"
              stop
            elif $result == "medical_advice":
              bot "**I can't provide medical diagnosis or medication advice.**\n\nFor medical concerns, please consult qualified healthcare providers at:\n- {crisis_resources['hospitals'][0]}\n- {crisis_resources['hospitals'][1]}\n- Your local health center\n\n> {crisis_resources['community_health']}\n\nI can offer emotional support and coping strategies while you seek professional medical care."
              stop
            elif $result == "off_topic":
              bot "I'm a **mental health companion** designed specifically to support your emotional well-being and mental health.\n\nWhile I'd love to help with other topics, my expertise is in providing:\n- Mental health support\n- Coping strategies\n- Connecting you with resources in Rwanda\n\nIs there anything related to your mental health or well-being I can help you with today?"
              stop
        """
        try:
            if self.chat_model:
                config = RailsConfig.from_content(flows_config)
                self.rails = LLMRails(config, llm=self.chat_model)
                print("Guardrails loaded successfully.")
            else:
                print("Guardrails not initialized because chat_model is missing.")
        except Exception as e:
            print(f"Error loading guardrails: {e}")
            self.rails = None

    def _is_ollama_running(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _is_model_available(self) -> bool:
        try:
            #models = requests.get(f"{self.ollama_base_url}/api/tags").json().get("models", [])
            r = requests.get(f"{self.ollama_base_url}/api/tags")
            r.raise_for_status()
            models = r.json().get("models", [])
            names = [m.get("name", "") for m in models]
            print(f"[Ollama] Available models: {names}")
            # treat exact or prefix (llama3 matches llama3.1)
            return any(self.model_name == n or n.startswith(self.model_name) for n in names)
            #return any(self.model_name in m.get("name", "") for m in models)
        
        except:
            return False

    def _is_vllm_running(self) -> bool:
        try:
            response = requests.get(f"{self.vllm_base_url}/models")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def _sanitize_input(self, user_message: str) -> str:
        """Sanitize user input to prevent injection attacks and clean malicious content"""
        # Remove excessive whitespace and normalize
        sanitized = re.sub(r'\s+', ' ', user_message.strip())
        
        # Remove potential prompt injection patterns
        injection_patterns = [
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
        
        for pattern in injection_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        # Limit length to prevent context overflow
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000] + "..."
            
        return sanitized

    def _is_safe_output(self, response: str) -> bool:
        """Check if model output contains unsafe content"""
        unsafe_patterns = [
            r'how\s+to\s+make\s+(bomb|weapon|drug)',
            r'suicide\s+method',
            r'kill\s+yourself',
            r'harm\s+yourself',
            r'end\s+your\s+life'
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False
        return True

    def _get_contextual_response_approach(self, emotion: str, user_message: str, conversation_context: List[Dict[str, str]]) -> Dict[str, str]:
        """Generate contextually appropriate response approach based on emotion, message content, and conversation history"""
        lowered = user_message.lower()
        
        
        # Rwanda-specific cultural considerations
        cultural_elements = RwandaMentalHealthResources.get_cultural_context()
        
        approach = {
            "tone": "empathetic",
            "cultural_element": "",
            "validation": "",
            "exploration_question": "",
            "support_offering": ""
        }
        
        # Emotion-specific approaches with cultural integration
        if emotion == "sadness":
            approach.update({
                "tone": "gentle_presence",
                "cultural_element": cultural_elements["ubuntu_philosophy"],
                "validation": "Your sadness is valid, and it's brave of you to share these feelings.",
                "exploration_question": "What's weighing most heavily on your heart right now?",
                "support_offering": "I'm here to listen and walk through this with you."
            })
        elif emotion == "anxiety":
            approach.update({
                "tone": "calming_reassurance",
                "cultural_element": cultural_elements["resilience_history"],
                "validation": "Anxiety can feel overwhelming, but you're not alone in this experience.",
                "exploration_question": "What thoughts or situations are contributing to this anxious feeling?",
                "support_offering": "Let's explore some grounding techniques that might help."
            })
        elif emotion == "stress":
            approach.update({
                "tone": "understanding_support",
                "cultural_element": cultural_elements["family_support"],
                "validation": "It sounds like you're carrying a heavy load right now.",
                "exploration_question": "What aspect of this stress feels most urgent to address?",
                "support_offering": "We can break this down into manageable pieces together."
            })
        else:  # neutral or other emotions
            approach.update({
                "validation": "Thank you for sharing what's on your mind.",
                "exploration_question": "What would be most helpful for us to focus on today?",
                "support_offering": "I'm here to support you in whatever way feels right."
            })
        
        # Contextual topic-specific adjustments
        if "school" in lowered or "university" in lowered:
            approach["exploration_question"] = "How are your studies affecting your overall well-being?"
        elif "family" in lowered:
            approach["cultural_element"] = cultural_elements["family_support"]
            approach["exploration_question"] = "Family relationships can be complex. What's happening that you'd like to talk about?"
        elif "work" in lowered or "job" in lowered:
            approach["exploration_question"] = "How is your work situation impacting your mental health?"
        
        return approach

    def _get_rwanda_appropriate_grounding(self) -> str:
        """Rwanda-culturally appropriate grounding exercise"""
        return (
            "🌿 Let's ground ourselves together:\n"
            "Breathe slowly... in through your nose, out through your mouth.\n"
            "Now, notice your surroundings:\n"
            "- 3 things you can see around you\n"
            "- 2 sounds you can hear (maybe birds, voices, or wind)\n" 
            "- 1 thing you can feel (your feet on the ground, air on your skin)\n"
            "Remember: You are here, you are present, and you belong to this community. "
        )

    def _fetch_recent_conversation(self, user_id: str, limit: int = 15) -> List[Dict[str, str]]:
        """Fetch recent conversation history with enhanced context"""
        try:
            db: Session = next(get_db())
            convo = db.query(Conversation).filter_by(user_id=user_id).order_by(Conversation.last_activity_at.desc()).first()
            if not convo: return []
            
            # Get more messages for better context (increased from 5 to 15)
            messages = db.query(Message).filter_by(conversation_id=convo.id).order_by(Message.timestamp.desc()).limit(limit).all()
            conversation_messages = [{"role": m.sender, "text": m.content, "timestamp": m.timestamp} for m in reversed(messages)]
            
            # Filter out very short or system messages for better context quality
            meaningful_messages = []
            for msg in conversation_messages:
                if len(msg["text"].strip()) >= 3:  # Keep messages with meaningful content
                    meaningful_messages.append({"role": msg["role"], "text": msg["text"]})
            
            return meaningful_messages
        except Exception as e:
            print(f"[DB Error] {e}")
            return []


    async def generate_response(self, user_message: str, conversation_history: Optional[List[Dict[str, Any]]] = None, user_id: Optional[str] = None, skip_analysis: bool = False, emotion_data: Optional[Dict[str, Any]] = None) -> str:
        import time
        llm_pipeline_start = time.time()
        
        if not self.is_initialized:
            return "Ollama not initialized."

        # Sanitize input first
        sanitized_message = self._sanitize_input(user_message)
        print(f"    🧹 LLM: Input sanitized ({len(user_message)} -> {len(sanitized_message)} chars)")

        # Apply guardrails first
        if self.rails:
            try:
                guardrails_start = time.time()
                response = await self.rails.generate_async(messages=[{"role": "user", "content": sanitized_message}])
                guardrails_time = time.time() - guardrails_start
                print(f"    🛡️  LLM: Guardrails check: {guardrails_time:.3f}s")
                
                # If guardrails blocked the message, return the safety response
                if response and hasattr(response, 'content') and response.content:
                    return response.content.strip()
            except Exception as e:
                print(f"    ❌ LLM: Guardrails Error: {e}")
                # Continue with normal processing if guardrails fail

        # Use sanitized message for the rest of processing
        user_message = sanitized_message

        # Initialize variables
        suicide_flag = False
        meds_mentioned = []
        retrieved_text = ""

        # Use emotion data from LangGraph if provided, otherwise detect locally
        if emotion_data:
            emotion = emotion_data.get("detected_emotion", "neutral")
            emotion_confidence = emotion_data.get("confidence", 0.5)
            emotion_intensity = emotion_data.get("intensity", "low")
            emotion_reasoning = emotion_data.get("reasoning", "Emotion detected by LangGraph")
            print(f"    🎭 LLM: Using emotion data from LangGraph: {emotion} (confidence: {emotion_confidence}, intensity: {emotion_intensity})")
        else:
            # Fast path - skip heavy analysis for simple/short messages
            if skip_analysis or len(user_message.strip()) < 10:
                emotion = "neutral"
                emotion_confidence = 0.5
                emotion_intensity = "low"
                emotion_reasoning = "Fast path - short message"
                print(f"    📝 LLM: Using fast path (short message)")
            else:
                # Only run expensive operations for longer messages
                analysis_start = time.time()
                # Use legacy classifier for now to avoid async/await issues in sync context
                emotion = classify_emotion_legacy(user_message)
                emotion_confidence = 0.8  # Default confidence for local detection
                emotion_intensity = "medium"  # Default intensity for local detection
                emotion_reasoning = "Detected by legacy emotion classifier (sync)"
                suicide_flag = detect_suicide_risk(user_message)
                meds_mentioned = detect_medication_mentions(user_message)
                analysis_time = time.time() - analysis_start
                print(f"    📊 LLM: Analysis pipeline: {analysis_time:.3f}s")

            # RAG decision logic - skip only for very basic interactions
            simple_greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you']
            is_simple_greeting = len(user_message.strip()) < 15 and any(greeting in user_message.lower() for greeting in simple_greetings)
            
            if is_simple_greeting:
                retrieved_text = ""
                print(f"    🚫 LLM: Skipping RAG (simple greeting)")
            else:
                rag_start = time.time()
                try:
                    # Increase RAG results for better context (from 2 to 3)
                    retrieved_chunks = self.retriever.search(query=user_message, top_k=3)
                    retrieved_text = "\n\n".join(chunk for chunk in retrieved_chunks if isinstance(chunk, str)) if retrieved_chunks else ""
                    rag_time = time.time() - rag_start
                    print(f"    🔍 LLM: RAG search: {rag_time:.3f}s ({len(retrieved_chunks)} chunks)")
                except Exception as e:
                    print(f"    ❌ LLM: RAG Error: {e}")
                    retrieved_text = ""

        if not conversation_history and user_id:
            conversation_history = self._fetch_recent_conversation(user_id)

        # Enhanced context building and response approach
        prompt_start = time.time()
        
        # Build richer memory block with more context (up to 15 messages for better continuity)
        memory_block = ""
        if conversation_history:
            recent_messages = conversation_history[-15:]  # Increased from 10 to 15
            memory_block = "\n".join(f"{m['role'].title()}: {m['text']}" for m in recent_messages)
        
        # Get contextual response approach
        response_approach = self._get_contextual_response_approach(emotion, user_message, conversation_history)
        
        # Rwanda-specific cultural and resource context
        rwanda_resources = RwandaMentalHealthResources()
        cultural_context = rwanda_resources.get_cultural_context()
        crisis_resources = rwanda_resources.get_crisis_resources()
        
        # Build contextual system prompt - avoid hallucination
        context_parts = []
        
        if memory_block.strip():
            context_parts.append(f"Previous conversation context:\n{memory_block}")
        else:
            context_parts.append("This appears to be a new conversation.")
            
        if suicide_flag:
            context_parts.append("⚠️ CRISIS INDICATOR: Suicide risk detected - prioritize safety and professional referral")
            
        if meds_mentioned:
            context_parts.append(f"📋 Medications mentioned: {', '.join(meds_mentioned)} - be mindful of medication safety")
            
        if retrieved_text:
            context_parts.append(f"Relevant knowledge: {retrieved_text[:500]}")

        # Enhanced emotion context from LangGraph
        emotion_context = ""
        if emotion_data:
            emotion_context = f"""
Advanced emotion analysis:
- Primary emotion: {emotion}
- Confidence: {emotion_confidence}
- Intensity: {emotion_intensity}
- Analysis: {emotion_reasoning}
- Emotion scores: {emotion_data.get('emotion_score', {})}
- Context relevance: {emotion_data.get('context_relevance', 'medium')}"""

        system_prompt = f"""You are a compassionate mental health companion for young people in Rwanda.
        You only use the English language even for greetings. Your name is Mindora Chat Companion.

Key principles:
- Ubuntu philosophy: "I am because we are" - emphasize community support
- Respect Rwandan culture and family-centered healing
- Provide emotional support, not medical diagnosis
- Connect to local resources when needed

Current context:
{chr(10).join(context_parts)}

User's emotional state: {emotion}
Response approach: {response_approach['validation']} {response_approach['support_offering']}
{emotion_context}

Available resources:
- Crisis: {crisis_resources['national_helpline']}
- Emergency: {crisis_resources['emergency']}
- Local health centers available

Respond with warmth and cultural sensitivity. Only reference what the user has actually shared - never invent conversation history."""

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
        prompt_time = time.time() - prompt_start
        print(f"    📝 LLM: Prompt building: {prompt_time:.3f}s ({len(system_prompt)} chars)")
        
        model_start = time.time()
        response = await self.chat_model.ainvoke(messages)
        model_time = time.time() - model_start
        print(f"    🤖 LLM: Model inference: {model_time:.3f}s")
        
        # Apply output safety filtering
        output_content = response.content.strip()
        if not self._is_safe_output(output_content):
            print(f"    ⚠️  LLM: Unsafe output detected, using fallback response")
            output_content = "**I understand** you're going through a difficult time.\n\nLet's focus on **healthier ways to cope** with what you're feeling.\n\nWould you like to:\n- Talk about what's troubling you?\n- Try some grounding techniques that might help?\n\nI'm here to support you through this."
        
        llm_total_time = time.time() - llm_pipeline_start
        print(f"      LLM: Total pipeline: {llm_total_time:.3f}s")
        
        return output_content
