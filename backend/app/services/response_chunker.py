"""
Response Chunker
================
Simple, dependency-free chunking helper used to split LLM responses into
progressive message chunks suitable for front-end progressive delivery.

Strategy:
- Prefer paragraph splits (double newline).
- Fallback to sentence splitting via punctuation.
- Group sentences to produce between 2 and 5 chunks (configurable).
- Attach recommended delays per chunk based on emotion intensity (research-backed ranges).

Return: list[dict] where each dict has {"text": str, "delay": float}
"""
from typing import List
import re

# Delay mapping based on research and product decision
_DELAY_MAPPING = {
    "LOW": 0.8,
    "MEDIUM": 1.6,
    "HIGH": 2.0,
    "CRITICAL": 1.2
}


def _split_into_paragraphs(text: str) -> List[str]:
    # Normalize newlines and split on two or more newlines
    paras = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
    return paras


def _split_into_sentences(text: str) -> List[str]:
    # Very lightweight sentence splitter using punctuation
    # Keep punctuation with sentence.
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def _group_into_n_chunks(items: List[str], n: int) -> List[str]:
    if n <= 0:
        return []
    if len(items) <= n:
        return items
    # Distribute items as evenly as possible
    avg = len(items) / n
    chunks = []
    start = 0
    for i in range(n):
        end = int(round((i + 1) * avg))
        if end <= start:
            end = start + 1
        chunk = " ".join(items[start:end])
        chunks.append(chunk.strip())
        start = end
        if start >= len(items):
            # append remaining as empty strings if needed
            for _ in range(i + 1, n):
                chunks.append("")
            break
    # Remove empties
    return [c for c in chunks if c]


class ResponseChunker:
    """Utility for splitting responses into progressive chunks.

    Methods
    -------
    chunk_response(text, emotion_intensity, max_chunks=5)
        Returns list of chunk dicts: {"text": str, "delay": float}
    """

    @staticmethod
    def chunk_response(text: str, emotion_intensity: str, max_chunks: int = 5) -> List[dict]:
        if not text:
            return []

        # Prefer paragraph split
        paragraphs = _split_into_paragraphs(text)
        if len(paragraphs) >= 2:
            # Use up to max_chunks paragraphs (merge if more)
            num = min(len(paragraphs), max_chunks)
            chunks = _group_into_n_chunks(paragraphs, num)
        else:
            # Fall back to sentence splitting
            sentences = _split_into_sentences(text)
            # Choose number of chunks: between 2 and max_chunks but not more than sentences
            desired = min(max(2, min(max_chunks, len(sentences))), max_chunks)
            chunks = _group_into_n_chunks(sentences, desired)

        # Final clean: ensure chunks are non-empty and reasonably sized
        cleaned = []
        for c in chunks:
            s = ' '.join([line.strip() for line in c.splitlines() if line.strip()])
            if s:
                cleaned.append(s)

        # Map intensity to delay
        delay_key = str(emotion_intensity).upper() if emotion_intensity else "MEDIUM"
        delay = _DELAY_MAPPING.get(delay_key, 1.6)

        # Build chunk dicts
        chunk_objs = [{"text": c, "delay": delay} for c in cleaned]

        # If only one chunk, still provide a small delay for UI smoothing
        if len(chunk_objs) == 1:
            chunk_objs[0]["delay"] = max(0.6, delay / 2)

        return chunk_objs
