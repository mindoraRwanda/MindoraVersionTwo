"""
LLM Council — Multi-model safety architecture for mental health AI.

Three specialist roles:
  1. Conversational model  — Warm, empathetic therapeutic responses
                             (Llama 3.3 8B via Groq, or any conversational provider)
  2. Safety model          — Fast, precise crisis/risk classification
                             (Qwen 3 8B via Together AI / Groq, or any fast provider)
  3. Validation model      — Pre-delivery safety gate; reviews every response
                             (GPT-4o Mini via OpenAI — multimodal capable)

The council is transparent to the rest of the pipeline: it implements the same
LLMProvider interface, so all existing pipeline nodes work unchanged.

Routing is automatic:
  - Calls with structured_output  →  safety model  (classification tasks)
  - Calls without structured_output →  conversational model  (response generation)
  - Every free-text response        →  validation model  (review before delivery)

Multimodal support:
  GPT-4o Mini (validation/conversational) can accept images.
  Use LLMCouncil.build_image_message() to construct multimodal messages.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


# ── Validation schema ─────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Structured output from the validation model council member."""
    is_safe: bool = Field(..., description="True if the response is safe to deliver as-is")
    needs_revision: bool = Field(False, description="True if the response requires changes before delivery")
    safety_concerns: List[str] = Field(default=[], description="Specific safety or quality issues identified")
    revised_response: Optional[str] = Field(None, description="Corrected response text when needs_revision=True")
    validation_score: float = Field(1.0, ge=0.0, le=1.0, description="Overall safety score (1.0 = fully safe)")


# ── LLM Council ───────────────────────────────────────────────────────────────

class LLMCouncil(LLMProvider):
    """
    Multi-model LLM Council for safety-critical mental health conversations.

    The council replaces a single LLMProvider while exposing the same interface,
    making it a drop-in upgrade.  All routing and validation logic is internal.

    Graceful degradation:
      - If the safety model fails   → falls back to the conversational model
      - If the validation model fails → delivers the unvalidated response with a warning
      - If the conversational model fails → returns a safe crisis-resource fallback
    """

    SYSTEM_PROMPT_VALIDATION = """\
You are a mental health AI safety reviewer. Your only job is to check whether
a chatbot response is safe and appropriate before it is delivered to a vulnerable user.

FLAG AS UNSAFE (is_safe=false) if ANY of the following apply:
1. Gives specific methods, doses, or instructions that enable self-harm or suicide
2. Dismisses, minimises, or ignores a disclosed crisis (abuse, suicidal ideation, GBV, self-harm)
3. Encourages harmful behaviours (dangerous coping strategies, substance use)
4. Provides firm medical diagnoses or prescribes specific medication
5. Contains materially inaccurate mental health information likely to cause harm
6. Uses shaming, stigmatising, or re-traumatising language
7. Fails to refer to emergency services when the user has disclosed immediate danger

FLAG FOR REVISION (needs_revision=true) if the response is safe but could be improved:
- Language that is dismissive without being dangerous
- Missing warmth or cultural sensitivity
- Slightly off-topic response

Return ONLY valid JSON with these exact keys:
  is_safe           (bool)
  needs_revision    (bool)
  safety_concerns   (list of strings — empty if none)
  revised_response  (string or null)
  validation_score  (float 0–1)
"""

    def __init__(
        self,
        conversational_provider: LLMProvider,
        safety_provider: LLMProvider,
        validation_provider: LLMProvider,
        skip_validation: bool = False,
        validation_score_threshold: float = 0.6,
    ):
        """
        Args:
            conversational_provider: Main model for therapeutic responses (e.g. Llama 3.3 8B)
            safety_provider:         Fast model for crisis/classification (e.g. Qwen 3 8B)
            validation_provider:     Review model before delivery (e.g. GPT-4o Mini)
            skip_validation:         Disable validation gate (e.g. for local/dev environments)
            validation_score_threshold: Minimum score to pass without revision
        """
        super().__init__(model_name="council")
        self.conversational_provider = conversational_provider
        self.safety_provider = safety_provider
        self.validation_provider = validation_provider
        self.skip_validation = skip_validation
        self.validation_score_threshold = validation_score_threshold

        logger.info(
            f"[Council] Initialized — "
            f"conversational={conversational_provider.provider_name}/{conversational_provider.model_name} | "
            f"safety={safety_provider.provider_name}/{safety_provider.model_name} | "
            f"validation={validation_provider.provider_name}/{validation_provider.model_name} | "
            f"skip_validation={skip_validation}"
        )

    # ── LLMProvider interface ─────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "council"

    def is_available(self) -> bool:
        """Council is available when the conversational model is reachable."""
        return self.conversational_provider.is_available()

    async def generate_response(
        self,
        messages: List[Any],
        structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
    ) -> Union[str, Any]:
        """
        Route to the right council member and optionally validate before returning.

        Classification calls (structured_output set) → safety model
        Free-text calls                               → conversational model → validation
        """
        if structured_output is not None:
            # Safety/classification task — use the specialist safety model
            logger.debug("[Council] → safety model (structured classification)")
            return await self._call_with_fallback(
                self.safety_provider, self.conversational_provider,
                messages, structured_output
            )

        # Conversational response
        logger.debug("[Council] → conversational model")
        response = await self._call_with_fallback(
            self.conversational_provider, self.safety_provider,
            messages, None
        )

        # Validation gate
        if not self.skip_validation and isinstance(response, str) and response.strip():
            response = await self._validate_and_maybe_revise(response, messages)

        return response

    async def agenerate(
        self,
        messages: List[Any],
        structured_output: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
    ) -> Union[str, Any]:
        return await self.generate_response(messages, structured_output)

    async def astream_text(self, messages: List[Any]):
        """
        Stream the conversational response AFTER validation.

        The council buffers the full response, validates it, then yields the
        (possibly revised) text in small chunks — guaranteeing no unsafe content
        reaches the user mid-stream at the cost of one extra validation round-trip.
        """
        full_response = await self._call_with_fallback(
            self.conversational_provider, self.safety_provider, messages, None
        )

        if not self.skip_validation and isinstance(full_response, str) and full_response.strip():
            full_response = await self._validate_and_maybe_revise(full_response, messages)

        text = str(full_response) if full_response else self._safe_fallback()
        chunk_size = 6
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    # ── Validation logic ──────────────────────────────────────────────────────

    async def _validate_and_maybe_revise(self, response: str, messages: List[Any]) -> str:
        """
        Run the validation model over the draft response.

        Returns the original response, a revised version, or a safe fallback
        depending on what the validation model finds.
        """
        try:
            result = await self._run_validation(response, messages)

            if result.is_safe and not result.needs_revision:
                logger.debug(f"[Council] Validation passed (score={result.validation_score:.2f})")
                return response

            if result.needs_revision and result.revised_response:
                logger.warning(
                    f"[Council] Response revised before delivery. "
                    f"Concerns: {result.safety_concerns}"
                )
                return result.revised_response

            if not result.is_safe:
                logger.error(
                    f"[Council] Response BLOCKED — unsafe content detected. "
                    f"Concerns: {result.safety_concerns}"
                )
                return self._safe_fallback()

            return response

        except Exception as e:
            logger.warning(f"[Council] Validation skipped due to error ({e}); delivering unvalidated response")
            return response

    async def _run_validation(self, draft: str, original_messages: List[Any]) -> ValidationResult:
        """Call the validation model with the draft response and return a ValidationResult."""
        from langchain_core.messages import SystemMessage, HumanMessage

        # Extract the most recent user message for context
        user_text = ""
        for msg in reversed(original_messages):
            cls = getattr(msg, "__class__", None)
            if cls and "Human" in cls.__name__:
                content = getattr(msg, "content", "")
                user_text = content if isinstance(content, str) else str(content)
                break

        human_prompt = (
            f"User message: {user_text[:600]}\n\n"
            f"Chatbot draft response:\n\"\"\"\n{draft[:2000]}\n\"\"\"\n\n"
            "Review this response for safety and appropriateness. Return JSON only."
        )

        validation_messages = [
            SystemMessage(content=self.SYSTEM_PROMPT_VALIDATION),
            HumanMessage(content=human_prompt),
        ]

        raw = await self._call_with_fallback(
            self.validation_provider, self.conversational_provider,
            validation_messages, ValidationResult
        )

        if isinstance(raw, ValidationResult):
            return raw

        # Provider returned text — parse defensively
        if isinstance(raw, str):
            try:
                m = re.search(r"\{.*\}", raw, re.S)
                data = json.loads(m.group(0)) if m else {}
                return ValidationResult(**data)
            except Exception:
                pass

        # Cannot parse → assume safe, don't block on validator failure
        logger.warning("[Council] Could not parse ValidationResult; assuming safe")
        return ValidationResult(is_safe=True, needs_revision=False, validation_score=1.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _call_with_fallback(
        self,
        primary: LLMProvider,
        fallback: LLMProvider,
        messages: List[Any],
        structured_output: Any,
    ) -> Any:
        """Call primary provider; if it raises, try fallback."""
        try:
            return await primary.agenerate(messages, structured_output)
        except Exception as e:
            logger.warning(
                f"[Council] {primary.provider_name}/{primary.model_name} failed: {e}. "
                f"Falling back to {fallback.provider_name}/{fallback.model_name}."
            )
            try:
                return await fallback.agenerate(messages, structured_output)
            except Exception as fe:
                logger.error(f"[Council] Fallback provider also failed: {fe}")
                return self._safe_fallback() if not structured_output else None

    @staticmethod
    def _safe_fallback() -> str:
        return (
            "Right now I want to make sure you get the right support. "
            "Please reach out to the Rwanda Mental Health Helpline: 114 (free, 24/7) "
            "or Emergency Services: 112."
        )

    # ── Multimodal helper ─────────────────────────────────────────────────────

    @staticmethod
    def build_image_message(text: str, image_data: str, media_type: str = "image/jpeg"):
        """
        Build a multimodal HumanMessage with text + image for vision-capable models.

        Compatible with GPT-4o Mini (OpenAI) and other vision LLMs.

        Args:
            text:       The text portion of the message
            image_data: Base64-encoded image bytes OR a public HTTPS image URL
            media_type: MIME type e.g. "image/jpeg", "image/png", "image/webp"

        Returns:
            LangChain HumanMessage with a multimodal content list

        Example:
            import base64
            with open("photo.jpg", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            msg = LLMCouncil.build_image_message("What do you see?", b64)
        """
        from langchain_core.messages import HumanMessage

        is_url = image_data.startswith("http://") or image_data.startswith("https://")
        if is_url:
            image_content: Dict[str, Any] = {
                "type": "image_url",
                "image_url": {"url": image_data},
            }
        else:
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_data}"},
            }

        return HumanMessage(content=[
            {"type": "text", "text": text},
            image_content,
        ])


# ── Factory ───────────────────────────────────────────────────────────────────

def create_llm_council(
    conversational_provider: Optional[LLMProvider] = None,
    safety_provider: Optional[LLMProvider] = None,
    validation_provider: Optional[LLMProvider] = None,
    skip_validation: bool = False,
) -> Optional["LLMCouncil"]:
    """
    Convenience factory.  Reads model/provider config from environment if not supplied.

    Expected environment variables:
      COUNCIL_CONV_PROVIDER    = groq          (conversational role)
      COUNCIL_CONV_MODEL       = llama-3.3-70b-versatile
      COUNCIL_SAFETY_PROVIDER  = groq          (safety/classification role)
      COUNCIL_SAFETY_MODEL     = llama-3.1-8b-instant
      COUNCIL_VAL_PROVIDER     = openai        (validation role)
      COUNCIL_VAL_MODEL        = gpt-4o-mini
      COUNCIL_SKIP_VALIDATION  = false

    For Qwen 3 8B via Together AI, set:
      COUNCIL_SAFETY_PROVIDER  = openai_compatible
      COUNCIL_SAFETY_MODEL     = Qwen/Qwen3-8B
      TOGETHER_API_KEY         = <your key>
      TOGETHER_BASE_URL        = https://api.together.xyz/v1
    """
    import os
    from .llm_providers import LLMProviderFactory

    def _build(
        provider_env: str,
        model_env: str,
        default_provider: str,
        default_model: str,
        supplied: Optional[LLMProvider],
    ) -> Optional[LLMProvider]:
        if supplied:
            return supplied
        pname = os.getenv(provider_env, default_provider)
        mname = os.getenv(model_env, default_model)
        try:
            return LLMProviderFactory.create_provider(provider_name=pname, model_name=mname)
        except Exception as e:
            logger.error(f"[Council] Failed to create {provider_env}={pname}/{mname}: {e}")
            return None

    conv = _build(
        "COUNCIL_CONV_PROVIDER", "COUNCIL_CONV_MODEL",
        "groq", "llama-3.3-70b-versatile", conversational_provider
    )
    safety = _build(
        "COUNCIL_SAFETY_PROVIDER", "COUNCIL_SAFETY_MODEL",
        "groq", "llama-3.1-8b-instant", safety_provider
    )
    val = _build(
        "COUNCIL_VAL_PROVIDER", "COUNCIL_VAL_MODEL",
        "openai", "gpt-4o-mini", validation_provider
    )

    if not conv:
        logger.error("[Council] Cannot create council without a conversational provider")
        return None

    # If safety or validation unavailable, fall back to conversational for that role
    if not safety:
        logger.warning("[Council] Safety provider unavailable; conversational model will handle classification")
        safety = conv
    if not val:
        logger.warning("[Council] Validation provider unavailable; skipping validation gate")
        skip_validation = True

    _skip = skip_validation or os.getenv("COUNCIL_SKIP_VALIDATION", "false").lower() == "true"

    return LLMCouncil(
        conversational_provider=conv,
        safety_provider=safety,
        validation_provider=val,
        skip_validation=_skip,
    )
