# # services/crisis_alert_service.py
# """
# Crisis alert service for mental health chatbot.

# This service handles crisis detection alerts and notifications, integrating
# with existing cultural crisis resources and support systems.
# """

# from typing import Dict, Any, List, Optional
# from datetime import datetime
# import logging
# import asyncio

# from ..settings.cultural import CulturalSettings
# from .pipeline_state import StatefulPipelineState, CrisisSeverity
# from .llm_cultural_context import RwandaCulturalManager

# logger = logging.getLogger(__name__)


# class CrisisAlertService:
#     """
#     Service for handling crisis alerts and interventions.

#     Integrates with existing cultural crisis resources and provides
#     appropriate notifications and support pathways.
#     """

#     def __init__(self):
#         """Initialize the crisis alert service."""
#         self.cultural_settings = CulturalSettings()
#         self.crisis_resources = self.cultural_settings.crisis_resources
#         self.cultural_manager = RwandaCulturalManager()
#         logger.info("🚨 CrisisAlertService initialized with cultural context integration")

#     async def handle_crisis_alert(
#         self,
#         state: StatefulPipelineState,
#         crisis_severity: CrisisSeverity,
#         immediate_action_required: bool = True
#     ) -> Dict[str, Any]:
#         """
#         Handle a crisis alert based on detected severity.

#         Args:
#             state: Current pipeline state
#             crisis_severity: Detected crisis severity
#             immediate_action_required: Whether immediate action is needed

#         Returns:
#             Alert handling results
#         """
#         logger.info("🚨 [CRISIS_ALERT] Handling crisis alert")
#         logger.info(f"   Severity: {crisis_severity.value}")
#         logger.info(f"   User ID: {state.get('user_id', 'unknown')}")
#         logger.info(f"   Query: '{state.get('user_query', '')[:50]}{'...' if len(state.get('user_query', '')) > 50 else ''}'")
#         logger.info(f"   Immediate action required: {immediate_action_required}")

#         try:
#             alert_result = {
#                 "alert_sent": False,
#                 "notifications_dispatched": [],
#                 "resources_provided": [],
#                 "followup_required": False,
#                 "timestamp": datetime.now().isoformat()
#             }

#             # Determine alert level based on severity
#             logger.info(f"   Processing {crisis_severity.value} severity crisis...")

#             if crisis_severity == CrisisSeverity.SEVERE:
#                 logger.warning("🔴 [CRISIS_ALERT] SEVERE severity - activating emergency protocols")
#                 alert_result.update(await self._handle_critical_crisis(state))
#             elif crisis_severity == CrisisSeverity.HIGH:
#                 logger.warning("🟠 [CRISIS_ALERT] HIGH severity - activating urgent intervention")
#                 alert_result.update(await self._handle_high_crisis(state))
#             elif crisis_severity == CrisisSeverity.MEDIUM:
#                 logger.info("🟡 [CRISIS_ALERT] MEDIUM severity - monitoring and support")
#                 alert_result.update(await self._handle_medium_crisis(state))
#             else:
#                 logger.info("🟢 [CRISIS_ALERT] LOW severity - providing resources only")
#                 # Low severity - just provide resources
#                 alert_result["resources_provided"] = self._get_support_resources()

#             # Update state with alert results
#             state["crisis_alert_sent"] = alert_result["alert_sent"]
#             state["crisis_followup_needed"] = alert_result["followup_required"]

#             logger.info("✅ [CRISIS_ALERT] Alert handling completed")
#             logger.info(f"   Alert sent: {alert_result['alert_sent']}")
#             logger.info(f"   Notifications dispatched: {len(alert_result['notifications_dispatched'])}")
#             logger.info(f"   Resources provided: {len(alert_result['resources_provided'])}")
#             logger.info(f"   Follow-up required: {alert_result['followup_required']}")

#             return alert_result

#         except Exception as e:
#             logger.error(f"❌ [CRISIS_ALERT] Failed to handle crisis alert: {e}")
#             logger.error(f"   Error type: {type(e).__name__}")
#             return {
#                 "alert_sent": False,
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }

#     async def _handle_critical_crisis(self, state: StatefulPipelineState) -> Dict[str, Any]:
#         """
#         Handle critical crisis situations requiring immediate emergency response.

#         Args:
#             state: Current mental health state

#         Returns:
#             Critical crisis handling results
#         """
#         # In a real implementation, this would:
#         # 1. Send immediate alerts to emergency services
#         # 2. Contact crisis intervention teams
#         # 3. Log crisis event with full context
#         # 4. Provide immediate emergency resources

#         alert_result = {
#             "alert_sent": True,
#             "notifications_dispatched": [
#                 "Emergency Services Alert",
#                 "Crisis Intervention Team Notification",
#                 "Medical Emergency Dispatch"
#             ],
#             "resources_provided": self._get_emergency_resources(),
#             "followup_required": True,
#             "immediate_actions": [
#                 "Emergency contact initiated",
#                 "Crisis team alerted",
#                 "Location tracking enabled (if consented)"
#             ]
#         }

#         # Log critical crisis event
#         await self._log_crisis_event(state, "SEVERE", alert_result)

#         return alert_result

#     async def _handle_high_crisis(self, state: StatefulPipelineState) -> Dict[str, Any]:
#         """
#         Handle high-risk crisis situations requiring urgent intervention.

#         Args:
#             state: Current mental health state

#         Returns:
#             High crisis handling results
#         """
#         alert_result = {
#             "alert_sent": True,
#             "notifications_dispatched": [
#                 "Crisis Support Team Alert",
#                 "Mental Health Professional Notification"
#             ],
#             "resources_provided": self._get_crisis_resources(),
#             "followup_required": True,
#             "immediate_actions": [
#                 "Crisis counselor contacted",
#                 "24/7 support line prioritized",
#                 "Family notification (if appropriate)"
#             ]
#         }

#         # Log high crisis event
#         await self._log_crisis_event(state, "HIGH", alert_result)

#         return alert_result

#     async def _handle_medium_crisis(self, state: StatefulPipelineState) -> Dict[str, Any]:
#         """
#         Handle medium-risk crisis situations requiring monitoring and support.

#         Args:
#             state: Current mental health state

#         Returns:
#             Medium crisis handling results
#         """
#         alert_result = {
#             "alert_sent": False,  # No immediate alert, but monitoring
#             "notifications_dispatched": [
#                 "Support Team Monitoring Alert"
#             ],
#             "resources_provided": self._get_support_resources(),
#             "followup_required": True,
#             "immediate_actions": [
#                 "Increased monitoring initiated",
#                 "Daily check-in scheduled",
#                 "Additional resources provided"
#             ]
#         }

#         # Log medium crisis event
#         await self._log_crisis_event(state, "MEDIUM", alert_result)

#         return alert_result

#     def _get_emergency_resources(self) -> List[Dict[str, str]]:
#         """
#         Get emergency resources from cultural settings.

#         Returns:
#             List of emergency resources
#         """
#         resources = []

#         # National helpline
#         helpline = self.crisis_resources.get("national_helpline")
#         if helpline:
#             resources.append({
#                 "type": "Emergency Helpline",
#                 "contact": helpline,
#                 "description": "24/7 mental health crisis support",
#                 "priority": "immediate"
#             })

#         # Emergency services
#         emergency = self.crisis_resources.get("emergency")
#         if emergency:
#             resources.append({
#                 "type": "Emergency Services",
#                 "contact": emergency,
#                 "description": "General emergency response",
#                 "priority": "immediate"
#             })

#         # Hospitals
#         hospitals = self.crisis_resources.get("hospitals", [])
#         for hospital in hospitals[:2]:  # Limit to top 2
#             resources.append({
#                 "type": "Emergency Hospital",
#                 "contact": hospital,
#                 "description": "Mental health emergency care",
#                 "priority": "immediate"
#             })

#         return resources

#     def _get_crisis_resources(self) -> List[Dict[str, str]]:
#         """
#         Get crisis intervention resources from cultural settings.

#         Returns:
#             List of crisis resources
#         """
#         resources = []

#         # National helpline
#         helpline = self.crisis_resources.get("national_helpline")
#         if helpline:
#             resources.append({
#                 "type": "Crisis Helpline",
#                 "contact": helpline,
#                 "description": "24/7 mental health support",
#                 "priority": "urgent"
#             })

#         # Hospitals
#         hospitals = self.crisis_resources.get("hospitals", [])
#         for hospital in hospitals[:1]:  # Limit to top 1
#             resources.append({
#                 "type": "Mental Health Hospital",
#                 "contact": hospital,
#                 "description": "Specialized mental health care",
#                 "priority": "urgent"
#             })

#         # Community health
#         community = self.crisis_resources.get("community_health")
#         if community:
#             resources.append({
#                 "type": "Community Health Center",
#                 "contact": community,
#                 "description": "Local mental health support",
#                 "priority": "high"
#             })

#         return resources

#     def _get_support_resources(self) -> List[Dict[str, str]]:
#         """
#         Get general support resources from cultural settings.

#         Returns:
#             List of support resources
#         """
#         resources = []

#         # National helpline
#         helpline = self.crisis_resources.get("national_helpline")
#         if helpline:
#             resources.append({
#                 "type": "Mental Health Helpline",
#                 "contact": helpline,
#                 "description": "24/7 mental health support",
#                 "priority": "high"
#             })

#         # Community health
#         community = self.crisis_resources.get("community_health")
#         if community:
#             resources.append({
#                 "type": "Community Health Center",
#                 "contact": community,
#                 "description": "Local mental health services",
#                 "priority": "medium"
#             })

#         # Online support
#         online = self.crisis_resources.get("online_support")
#         if online:
#             resources.append({
#                 "type": "Online Mental Health Support",
#                 "contact": online,
#                 "description": "Additional mental health resources",
#                 "priority": "medium"
#             })

#         return resources

#     async def _log_crisis_event(
#         self,
#         state: StatefulPipelineState,
#         severity: str,
#         alert_result: Dict[str, Any]
#     ) -> None:
#         """
#         Log crisis event for monitoring and follow-up.

#         Args:
#             state: Current mental health state
#             severity: Crisis severity level
#             alert_result: Alert handling results
#         """
#         # In a real implementation, this would log to a database or monitoring system
#         crisis_log = {
#             "timestamp": datetime.now().isoformat(),
#             "severity": severity,
#             "user_id": state.get("user_id"),
#             "query": state.get("user_query", "")[:200],  # Truncate for privacy
#             "alert_sent": alert_result.get("alert_sent", False),
#             "notifications": alert_result.get("notifications_dispatched", []),
#             "resources_provided": len(alert_result.get("resources_provided", [])),
#             "cultural_context": state.get("detected_language", "unknown"),
#             "crisis_indicators": state.get("crisis_assessment", {}).crisis_keywords if hasattr(state.get("crisis_assessment", {}), 'crisis_keywords') else []
#         }

#         logger.warning(f"CRISIS EVENT LOGGED: {crisis_log}")

#         # TODO: Implement actual logging to database/monitoring system
#         # await self._save_crisis_log(crisis_log)

#     def get_crisis_response_template(self, severity: CrisisSeverity) -> str:
#         """
#         Get appropriate crisis response template based on severity.

#         Args:
#             severity: Crisis severity level

#         Returns:
#             Crisis response template
#         """
#         templates = {
#             CrisisSeverity.SEVERE: """I can hear that you're in immediate danger and need urgent help right now. Your safety is the most important thing.

# **EMERGENCY CONTACTS:**
# {emergency_contacts}

# Please call emergency services immediately or go to the nearest emergency room. Help is available and you don't have to face this alone.""",

#             CrisisSeverity.HIGH: """I hear how much pain you're experiencing, and I want to make sure you get the help you need right away.

# **URGENT CRISIS SUPPORT:**
# {crisis_contacts}

# These services are available 24/7 and trained to help in crisis situations. Please reach out to them now - they can provide immediate support.""",

#             CrisisSeverity.MEDIUM: """I can sense this is really difficult for you right now. While it may not be an immediate emergency, getting support is important.

# **MENTAL HEALTH SUPPORT:**
# {support_contacts}

# Consider reaching out to these resources for additional support and guidance.""",
#         }

#         template = templates.get(severity, templates[CrisisSeverity.MEDIUM])

#         # Fill in contact information
#         emergency_contacts = "\n".join([
#             f"- {resource['contact']} ({resource['description']})"
#             for resource in self._get_emergency_resources()
#         ])

#         crisis_contacts = "\n".join([
#             f"- {resource['contact']} ({resource['description']})"
#             for resource in self._get_crisis_resources()
#         ])

#         support_contacts = "\n".join([
#             f"- {resource['contact']} ({resource['description']})"
#             for resource in self._get_support_resources()
#         ])

#         return template.format(
#             emergency_contacts=emergency_contacts,
#             crisis_contacts=crisis_contacts,
#             support_contacts=support_contacts
#         )

#     def should_trigger_alert(self, state: StatefulPipelineState) -> bool:
#         """
#         Determine if an alert should be triggered based on state.

#         Args:
#             state: Current mental health state

#         Returns:
#             True if alert should be triggered
#         """
#         crisis_assessment = state.get("crisis_assessment")
#         if not crisis_assessment:
#             return False

#         # Trigger alert for high or severe crises
#         crisis_severity = crisis_assessment.crisis_severity if hasattr(crisis_assessment, 'crisis_severity') else None
#         return crisis_severity in [CrisisSeverity.HIGH, CrisisSeverity.SEVERE]

#     def get_followup_schedule(self, severity: CrisisSeverity) -> List[str]:
#         """
#         Get appropriate follow-up schedule based on crisis severity.

#         Args:
#             severity: Crisis severity level

#         Returns:
#             List of follow-up timeframes
#         """
#         schedules = {
#             CrisisSeverity.SEVERE: [
#                 "Immediate (within 1 hour)",
#                 "4 hours after initial contact",
#                 "24 hours after initial contact",
#                 "Weekly for 4 weeks"
#             ],
#             CrisisSeverity.HIGH: [
#                 "Within 24 hours",
#                 "3 days after initial contact",
#                 "1 week after initial contact",
#                 "Bi-weekly for 4 weeks"
#             ],
#             CrisisSeverity.MEDIUM: [
#                 "Within 3 days",
#                 "1 week after initial contact",
#                 "Bi-weekly for 2 weeks"
#             ]
#         }

#         return schedules.get(severity, [])
    
#     def get_culturally_appropriate_resources(
#         self, 
#         state: StatefulPipelineState, 
#         severity: CrisisSeverity
#     ) -> List[Dict[str, str]]:
#         """
#         Get culturally appropriate crisis resources based on language and RAG knowledge.
        
#         Args:
#             state: Current pipeline state
#             severity: Crisis severity level
            
#         Returns:
#             List of culturally appropriate crisis resources
#         """
#         language = state.get("detected_language", "en")
#         rag_applied = state.get("rag_enhancement_applied", False)
        
#         # Get base resources
#         if severity == CrisisSeverity.SEVERE:
#             base_resources = self._get_emergency_resources()
#         elif severity == CrisisSeverity.HIGH:
#             base_resources = self._get_crisis_resources()
#         else:
#             base_resources = self._get_support_resources()
        
#         # Enhance with cultural context
#         cultural_resources = self.cultural_manager.get_crisis_resources(language)
        
#         # Add cultural context to resources
#         enhanced_resources = []
#         for resource in base_resources:
#             enhanced_resource = resource.copy()
            
#             # Add cultural context if available
#             if language in cultural_resources:
#                 cultural_info = cultural_resources[language]
#                 if resource["type"] == "Emergency Helpline" and "national_helpline" in cultural_info:
#                     enhanced_resource["cultural_note"] = f"Available in {language}"
#                 elif resource["type"] == "Emergency Services" and "emergency" in cultural_info:
#                     enhanced_resource["cultural_note"] = f"Emergency services in {language}"
            
#             enhanced_resources.append(enhanced_resource)
        
#         # Add RAG-enhanced resources if available
#         if rag_applied:
#             knowledge_context = state.get("knowledge_context", "")
#             if knowledge_context:
#                 # Extract relevant crisis resources from RAG knowledge
#                 rag_resources = self._extract_crisis_resources_from_rag(knowledge_context, language)
#                 enhanced_resources.extend(rag_resources)
        
#         return enhanced_resources
    
#     def _extract_crisis_resources_from_rag(self, knowledge_context: str, language: str) -> List[Dict[str, str]]:
#         """
#         Extract crisis resources from RAG knowledge context.
        
#         Args:
#             knowledge_context: Retrieved knowledge context
#             language: User's language
            
#         Returns:
#             List of crisis resources extracted from knowledge
#         """
#         resources = []
        
#         # Simple extraction of crisis resources from knowledge context
#         # In a real implementation, this would use NLP to extract contact information
#         if "helpline" in knowledge_context.lower() or "hotline" in knowledge_context.lower():
#             resources.append({
#                 "type": "Knowledge-Based Crisis Support",
#                 "contact": "See knowledge context for details",
#                 "description": "Crisis support information from mental health knowledge base",
#                 "priority": "high",
#                 "source": "RAG Knowledge"
#             })
        
#         if "hospital" in knowledge_context.lower() or "emergency" in knowledge_context.lower():
#             resources.append({
#                 "type": "Knowledge-Based Emergency Care",
#                 "contact": "See knowledge context for details", 
#                 "description": "Emergency care information from mental health knowledge base",
#                 "priority": "high",
#                 "source": "RAG Knowledge"
#             })
        
#         return resources
    
#     def get_cultural_crisis_response_template(
#         self, 
#         severity: CrisisSeverity, 
#         state: StatefulPipelineState
#     ) -> str:
#         """
#         Get culturally appropriate crisis response template.
        
#         Args:
#             severity: Crisis severity level
#             state: Current pipeline state
            
#         Returns:
#             Culturally appropriate crisis response template
#         """
#         language = state.get("detected_language", "en")
#         gender_addressing = state.get("gender_aware_addressing", "friend")
        
#         # Get base template
#         base_template = self.get_crisis_response_template(severity)
        
#         # Enhance with cultural context
#         if language == "rw":
#             # Kinyarwanda cultural adaptation
#             cultural_enhancement = f"\n\nMuraho {gender_addressing}, ndabizi ko urakomeje. Turi kumwe, kandi dushobora gufasha."
#             base_template = cultural_enhancement + base_template
#         elif language == "fr":
#             # French cultural adaptation
#             cultural_enhancement = f"\n\nBonjour {gender_addressing}, je comprends que vous traversez une période difficile. Nous sommes là pour vous aider."
#             base_template = cultural_enhancement + base_template
        
#         return base_template