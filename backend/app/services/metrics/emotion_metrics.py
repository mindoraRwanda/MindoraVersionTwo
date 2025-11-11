"""
Emotion classification metrics tracker.

Tracks ML emotion classifier performance, accuracy, and cultural awareness.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter, defaultdict
import time

logger = logging.getLogger(__name__)


class EmotionMetricsTracker:
    """
    Lightweight metrics tracker for emotion detection.
    
    Tracks:
    - Classification performance (latency, confidence)
    - Emotion distribution
    - Cultural marker detection rate
    - Running statistics for real-time monitoring
    """
    
    def __init__(self):
        """Initialize metrics storage."""
        self.metrics: List[Dict] = []
        self.running_stats = {
            "total": 0,
            "total_time_ms": 0.0,
            "total_confidence": 0.0,
            "cultural_detected": 0,
            "emotions": Counter()
        }
        logger.info("📊 EmotionMetricsTracker initialized")
    
    def track(
        self,
        user_id: str,
        message_id: str,
        text: str,
        primary_emotion: str,
        confidence: float,
        intensity: str,
        processing_time_ms: float,
        secondary_emotions: Optional[Dict[str, float]] = None,
        cultural_context: Optional[str] = None
    ):
        """
        Track a single emotion detection event.
        
        Args:
            user_id: User identifier
            message_id: Message identifier
            text: Input text
            primary_emotion: Detected emotion
            confidence: Confidence score (0-1)
            intensity: Intensity level (low/medium/high/critical)
            processing_time_ms: Processing time in milliseconds
            secondary_emotions: Dict of secondary emotions and scores
            cultural_context: Cultural markers detected
        """
        metric = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "message_id": message_id,
            "text_length": len(text),
            "primary_emotion": primary_emotion,
            "confidence": confidence,
            "intensity": intensity,
            "processing_ms": processing_time_ms,
            "secondary_emotions": secondary_emotions or {},
            "cultural_detected": bool(cultural_context),
            "cultural_context": cultural_context
        }
        
        # Update running stats
        self.running_stats["total"] += 1
        self.running_stats["total_time_ms"] += processing_time_ms
        self.running_stats["total_confidence"] += confidence
        self.running_stats["emotions"][primary_emotion] += 1
        if cultural_context:
            self.running_stats["cultural_detected"] += 1
        
        # Store metric
        self.metrics.append(metric)
        
        # Log summary every 10 classifications
        if self.running_stats["total"] % 10 == 0:
            self._log_summary()
    
    def get_summary(self) -> Dict:
        """Get current metrics summary."""
        total = self.running_stats["total"]
        if total == 0:
            return {"total_classifications": 0}
        
        return {
            "total_classifications": total,
            "avg_confidence": self.running_stats["total_confidence"] / total,
            "avg_processing_ms": self.running_stats["total_time_ms"] / total,
            "cultural_detection_rate": self.running_stats["cultural_detected"] / total,
            "emotion_distribution": dict(self.running_stats["emotions"].most_common()),
            "last_100_metrics": self.metrics[-100:] if len(self.metrics) > 100 else self.metrics
        }
    
    def _log_summary(self):
        """Log periodic summary."""
        stats = self.running_stats
        total = stats["total"]
        
        avg_time = stats["total_time_ms"] / total
        avg_conf = stats["total_confidence"] / total
        cultural_rate = stats["cultural_detected"] / total
        
        logger.info("=" * 70)
        logger.info("📊 EMOTION METRICS SUMMARY")
        logger.info(f"   Total: {total} | Avg Confidence: {avg_conf:.2f}")
        logger.info(f"   Avg Latency: {avg_time:.1f}ms | Cultural Rate: {cultural_rate:.1%}")
        logger.info(f"   Top Emotions: {dict(stats['emotions'].most_common(3))}")
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
