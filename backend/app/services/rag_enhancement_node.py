"""
RAG Enhancement Node for Stateful Pipeline
Enhances responses with retrieved knowledge from the vector database.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from .pipeline_state import StatefulPipelineState, add_processing_metadata, add_error
from .kb import retrieve_kb_hybrid, initialize_kb

logger = logging.getLogger(__name__)


class RAGEnhancementNode:
    """
    Node for enhancing responses with retrieved knowledge from RAG system.
    
    This node:
    1. Searches for relevant knowledge based on user query
    2. Filters and ranks retrieved content
    3. Integrates knowledge into response context
    4. Tracks retrieval metadata for explainability
    """
    
    def __init__(self, rag_service=None):
        """Initialize the RAG enhancement node.

        The previous implementation depended on a dedicated vector-database
        service (UnifiedRAGService). We now use the KB cards + TF-IDF (and
        optional local Qdrant) via the kb module, keeping the public behavior
        similar but dropping the heavy runtime vector DB dependency.
        """
        # The rag_service parameter is kept for backward-compatibility but
        # is no longer required; retrieval is done via retrieve_kb_hybrid.
        self.rag_service = rag_service
        self.logger = logger
        
        # Configuration
        self.max_retrieved_chunks = 5
        self.min_relevance_score = 0.3
        self.max_context_length = 2000  # Max characters for context
        
        self.logger.info("🔍 RAGEnhancementNode initialized")
    
    async def execute(self, state: StatefulPipelineState) -> StatefulPipelineState:
        """
        Execute RAG enhancement to retrieve relevant knowledge.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with retrieved knowledge
        """
        start_time = time.time()
        query = state["user_query"]
        
        self.logger.info(f"🔍 Starting RAG enhancement for query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
        
        try:
            # Skip RAG for simple greetings or very short queries
            if self._should_skip_rag(query, state):
                self.logger.info("⏭️ Skipping RAG enhancement for simple query")
                state["retrieved_knowledge"] = []
                state["rag_enhancement_applied"] = False
                return state
            
            # Retrieve relevant knowledge from KB (cards)
            retrieved_results = await self._retrieve_knowledge(query, state)
            
            # Filter and rank results
            filtered_results = self._filter_and_rank_results(retrieved_results, query)
            
            # Format knowledge for context
            knowledge_context = self._format_knowledge_context(filtered_results, state)
            
            # Update state with retrieved knowledge
            state["retrieved_knowledge"] = filtered_results
            state["knowledge_context"] = knowledge_context
            state["rag_enhancement_applied"] = True
            state["rag_relevance_score"] = self._calculate_avg_relevance(filtered_results)
            
            processing_time = time.time() - start_time
            
            # Add processing metadata
            state = add_processing_metadata(
                state,
                "rag_enhancement",
                0.8 if filtered_results else 0.3,
                f"Retrieved {len(filtered_results)} relevant knowledge chunks",
                ["rag", "knowledge_retrieval", "context_enhancement"],
                processing_time
            )
            
            self.logger.info(f"✅ RAG enhancement completed: {len(filtered_results)} chunks, {processing_time:.3f}s")
            
        except Exception as e:
            self.logger.error(f"❌ RAG enhancement failed: {e}")
            add_error(state, f"RAG enhancement error: {str(e)}")
            state["retrieved_knowledge"] = []
            state["knowledge_context"] = ""
            state["rag_enhancement_applied"] = False
        
        return state
    
    def _should_skip_rag(self, query: str, state: StatefulPipelineState) -> bool:
        """Determine if RAG should be skipped for this query."""
        # Skip for very short queries
        if len(query.strip()) < 10:
            return True
        
        # Skip for simple greetings
        simple_greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if query.lower().strip() in simple_greetings:
            return True
        
        # Skip if query validation indicates it's random/off-topic
        query_validation = state.get("query_validation")
        if query_validation and hasattr(query_validation, 'is_random') and query_validation.is_random:
            return True
        
        return False
    
    async def _retrieve_knowledge(self, query: str, state: StatefulPipelineState) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge from the KB system."""
        try:
            # Ensure KB is initialized
            initialize_kb()

            # Enhance query with context from conversation history
            enhanced_query = self._enhance_query_with_context(query, state)

            kb_hits, method, elapsed = retrieve_kb_hybrid(
                enhanced_query,
                k=self.max_retrieved_chunks,
                username=str(state.get("user_id")) if state.get("user_id") else None,
                conversation_id=str(state.get("conversation_id")) if state.get("conversation_id") else None,
            )

            self.logger.info(
                f"🔍 Retrieved {len(kb_hits)} KB cards using {method} in {elapsed:.3f}s"
            )

            # Adapt KB cards into the generic 'results' format used downstream
            results: List[Dict[str, Any]] = []
            for card in kb_hits:
                results.append(
                    {
                        "id": card.get("id"),
                        "score": 1.0,  # TF-IDF / semantic scores not exposed; treat as relevant
                        "text": card.get("bot_say", ""),
                        "source": card.get("title", card.get("id", "KB")),
                        "chunk_id": 0,
                        "file_size": 0,
                        "processed_at": 0,
                    }
                )

            return results

        except Exception as e:
            self.logger.error(f"❌ Knowledge retrieval failed: {e}")
            return []
    
    def _enhance_query_with_context(self, query: str, state: StatefulPipelineState) -> str:
        """Enhance the search query with context from conversation history."""
        # Start with the original query
        enhanced_query = query
        
        # Add emotion context if available
        emotion_detection = state.get("emotion_detection")
        if emotion_detection and hasattr(emotion_detection, 'selected_emotion'):
            emotion = emotion_detection.selected_emotion
            if emotion != "neutral":
                enhanced_query += f" {emotion} emotional state"
        
        # Add crisis context if available
        crisis_assessment = state.get("crisis_assessment")
        if crisis_assessment and hasattr(crisis_assessment, 'crisis_severity'):
            crisis_severity = crisis_assessment.crisis_severity.value
            if crisis_severity != "none":
                enhanced_query += f" {crisis_severity} crisis level"
        
        # Add cultural context if available
        detected_language = state.get("detected_language", "en")
        if detected_language != "en":
            enhanced_query += f" {detected_language} language context"
        
        return enhanced_query
    
    def _filter_and_rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter and rank retrieved results based on relevance."""
        if not results:
            return []
        
        # Filter by minimum relevance score
        filtered_results = [
            result for result in results 
            if result.get("score", 0) >= self.min_relevance_score
        ]
        
        # Sort by relevance score (descending)
        filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Limit context length
        filtered_results = self._limit_context_length(filtered_results)
        
        self.logger.info(f"📊 Filtered {len(results)} results to {len(filtered_results)} relevant chunks")
        
        return filtered_results
    
    def _limit_context_length(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Limit the total context length to avoid overwhelming the LLM."""
        limited_results = []
        total_length = 0
        
        for result in results:
            text_length = len(result.get("text", ""))
            if total_length + text_length <= self.max_context_length:
                limited_results.append(result)
                total_length += text_length
            else:
                # Truncate the last result if needed
                remaining_length = self.max_context_length - total_length
                if remaining_length > 100:  # Only add if meaningful length remains
                    truncated_result = result.copy()
                    truncated_result["text"] = result["text"][:remaining_length] + "..."
                    limited_results.append(truncated_result)
                break
        
        return limited_results
    
    def _format_knowledge_context(self, results: List[Dict[str, Any]], state: StatefulPipelineState) -> str:
        """Format retrieved knowledge into context for the LLM."""
        if not results:
            return ""
        
        # Get cultural context for formatting
        detected_language = state.get("detected_language", "en")
        
        # Format based on language
        if detected_language == "rw":
            context_header = "Ubwenge bw'ubuzima bwo mu mutwe:"
        elif detected_language == "fr":
            context_header = "Connaissances en santé mentale:"
        else:
            context_header = "Mental Health Knowledge:"
        
        context_parts = [context_header]
        
        for i, result in enumerate(results, 1):
            source = result.get("source", "Unknown")
            text = result.get("text", "")
            score = result.get("score", 0)
            
            # Format each knowledge chunk
            context_parts.append(f"\n{i}. From {source} (relevance: {score:.2f}):")
            context_parts.append(f"   {text}")
        
        return "\n".join(context_parts)
    
    def _calculate_avg_relevance(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average relevance score of retrieved results."""
        if not results:
            return 0.0
        
        total_score = sum(result.get("score", 0) for result in results)
        return total_score / len(results)
    
    def get_retrieval_summary(self, state: StatefulPipelineState) -> Dict[str, Any]:
        """Get a summary of the RAG retrieval process."""
        retrieved_knowledge = state.get("retrieved_knowledge", [])
        rag_enhancement_applied = state.get("rag_enhancement_applied", False)
        rag_relevance_score = state.get("rag_relevance_score", 0.0)
        
        return {
            "rag_applied": rag_enhancement_applied,
            "chunks_retrieved": len(retrieved_knowledge),
            "avg_relevance_score": rag_relevance_score,
            "sources": list(set(result.get("source", "Unknown") for result in retrieved_knowledge)),
            "knowledge_context_length": len(state.get("knowledge_context", "")),
            "top_sources": [
                {
                    "source": result.get("source", "Unknown"),
                    "score": result.get("score", 0),
                    "text_preview": result.get("text", "")[:100] + "..." if len(result.get("text", "")) > 100 else result.get("text", "")
                }
                for result in retrieved_knowledge[:3]  # Top 3 sources
            ]
        }
