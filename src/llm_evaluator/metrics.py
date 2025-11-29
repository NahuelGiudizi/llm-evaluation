"""
Performance and Quality Metrics for LLM Evaluation
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Performance-related metrics"""

    @staticmethod
    def calculate_avg_response_time(times: List[float]) -> float:
        """Calculate average response time"""
        if not times:
            raise ValueError("No response times provided")
        return sum(times) / len(times)

    @staticmethod
    def calculate_tokens_per_second(tokens: int, time_seconds: float) -> float:
        """Calculate token generation speed"""
        if time_seconds <= 0:
            raise ValueError("Time must be greater than 0")
        return tokens / time_seconds

    @staticmethod
    def calculate_throughput(requests: int, time_seconds: float) -> float:
        """Calculate requests per second"""
        if time_seconds <= 0:
            raise ValueError("Time must be greater than 0")
        return requests / time_seconds

    @staticmethod
    def calculate_latency_percentile(latencies: List[float], percentile: int = 95) -> float:
        """Calculate latency percentile (e.g., P95)"""
        sorted_latencies = sorted(latencies)
        index = int(len(sorted_latencies) * (percentile / 100))
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]


@dataclass
class QualityMetrics:
    """Quality-related metrics"""

    @staticmethod
    def calculate_accuracy(correct: int, total: int) -> float:
        """Calculate simple accuracy"""
        if total <= 0:
            raise ValueError("Total must be greater than 0")
        return correct / total

    @staticmethod
    def calculate_coherence_score(responses: List[str]) -> float:
        """Calculate coherence score based on response length and structure"""
        if not responses:
            return 0.0
        # Simple coherence: average sentence length and punctuation usage
        scores = []
        for response in responses:
            sentences = response.split(".")
            avg_sentence_length = (
                sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            )
            # Normalize: ideal sentence length ~15-20 words
            score = min(avg_sentence_length / 20.0, 1.0)
            scores.append(score)
        return sum(scores) / len(scores)

    @staticmethod
    def calculate_hallucination_rate(hallucinations: int, total: int) -> float:
        """Calculate hallucination rate"""
        if total <= 0:
            return 0.0
        return hallucinations / total

    @staticmethod
    def calculate_precision_recall_f1(
        true_positives: int, false_positives: int, false_negatives: int
    ) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score"""
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision, "recall": recall, "f1_score": f1}

    @staticmethod
    def calculate_bleu_score(reference: str, candidate: str) -> float:
        """
        Simplified BLEU score calculation (unigram only)
        For production, use nltk.translate.bleu_score
        """
        ref_tokens = set(reference.lower().split())
        cand_tokens = candidate.lower().split()

        matches = sum(1 for token in cand_tokens if token in ref_tokens)
        return matches / len(cand_tokens) if cand_tokens else 0.0
