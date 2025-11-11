"""
Pipeline execution metrics tracker.

Tracks overall pipeline performance and system health.
"""

import logging
from typing import Dict, List
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


class PipelineMetricsTracker:
    """
    Lightweight metrics tracker for pipeline execution.
    
    Tracks:
    - Response time and latency
    - LLM call efficiency
    - Error rates
    - Strategy distribution
    """
    
    def __init__(self):
        """Initialize metrics storage."""
        self.metrics: List[Dict] = []
        self.running_stats = {
            "total": 0,
            "total_time_ms": 0.0,
            "total_llm_calls": 0,
            "total_errors": 0,
            "strategies": Counter(),
            "crisis_levels": Counter()
        }
        logger.info("📊 PipelineMetricsTracker initialized")
    
    def track(
        self,
        user_id: str,
        message_id: str,
        query_length: int,
        total_time_ms: float,
        llm_calls: int,
        errors: int,
        steps_completed: List[str],
        response_length: int,
        response_confidence: float,
        strategy: str,
        crisis_severity: str,
        detected_emotion: str
    ):
        """
        Track a single pipeline execution.
        
        Args:
            user_id: User identifier
            message_id: Message identifier
            query_length: Length of user query
            total_time_ms: Total pipeline time
            llm_calls: Number of LLM calls
            errors: Number of errors
            steps_completed: List of completed steps
            response_length: Response text length
            response_confidence: Response confidence
            strategy: Strategy used
            crisis_severity: Crisis level detected
            detected_emotion: Emotion detected
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "message_id": message_id,
            "query_length": query_length,
            "total_time_ms": total_time_ms,
            "llm_calls": llm_calls,
            "errors": errors,
            "steps": len(steps_completed),
            "response_length": response_length,
            "response_confidence": response_confidence,
            "strategy": strategy,
            "crisis_severity": crisis_severity,
            "emotion": detected_emotion
        }
        
        # Update running stats
        self.running_stats["total"] += 1
        self.running_stats["total_time_ms"] += total_time_ms
        self.running_stats["total_llm_calls"] += llm_calls
        self.running_stats["total_errors"] += errors
        self.running_stats["strategies"][strategy] += 1
        self.running_stats["crisis_levels"][crisis_severity] += 1
        
        # Store metric
        self.metrics.append(metric)
        
        # Log summary every 10 executions
        if self.running_stats["total"] % 10 == 0:
            self._log_summary()
    
    def get_summary(self) -> Dict:
        """Get current metrics summary."""
        total = self.running_stats["total"]
        if total == 0:
            return {"total_executions": 0}
        
        return {
            "total_executions": total,
            "avg_response_time_ms": self.running_stats["total_time_ms"] / total,
            "avg_llm_calls": self.running_stats["total_llm_calls"] / total,
            "total_errors": self.running_stats["total_errors"],
            "error_rate": self.running_stats["total_errors"] / total if total > 0 else 0,
            "strategy_distribution": dict(self.running_stats["strategies"].most_common()),
            "crisis_distribution": dict(self.running_stats["crisis_levels"]),
            "last_100_metrics": self.metrics[-100:] if len(self.metrics) > 100 else self.metrics
        }
    
    def _log_summary(self):
        """Log periodic summary."""
        stats = self.running_stats
        total = stats["total"]
        
        avg_time = stats["total_time_ms"] / total
        avg_llm = stats["total_llm_calls"] / total
        error_rate = stats["total_errors"] / total
        
        logger.info("=" * 70)
        logger.info("📊 PIPELINE METRICS SUMMARY")
        logger.info(f"   Total: {total} | Avg Time: {avg_time:.1f}ms | LLM: {avg_llm:.1f}")
        logger.info(f"   Errors: {stats['total_errors']} ({error_rate:.1%})")
        logger.info(f"   Strategies: {dict(stats['strategies'].most_common(3))}")
        logger.info("=" * 70)
    
    def export_json(self, filepath: str):
        """Export metrics to JSON."""
        import json
        with open(filepath, 'w') as f:
            json.dump({
                "exported_at": datetime.now().isoformat(),
                "summary": self.get_summary(),
                "metrics": self.metrics
            }, f, indent=2)
        logger.info(f"📁 Exported to {filepath}")
