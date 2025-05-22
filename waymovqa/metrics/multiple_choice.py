from abc import ABC, abstractmethod
from typing import Dict, Any, List

from pathlib import Path

from waymovqa.questions.base import BaseQuestion
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer

from .base import BaseMetric


class MultipleChoiceMetric(BaseMetric[MultipleChoiceAnswer]):
    """Evaluates multiple choice answers."""

    def __init__(self):
        super().__init__(MultipleChoiceAnswer)

    def evaluate(
        self,
        prediction: MultipleChoiceAnswer,
        ground_truth: MultipleChoiceAnswer,
        question,
    ) -> Dict[str, bool]:
        """
        Evaluate predicted multiple choice answwer.
        """
        assert prediction.choices == ground_truth.choices

        choices = prediction.choices
        if prediction.answer not in choices:
            return dict(valid=False, correct=False)
        else:
            return dict(valid=True, correct=ground_truth.answer == prediction.answer)

    def _f_score(self, precision: float, recall: float, beta: float):
        """
        Calculates F-score.

        Arguments:
            precision: float
            recall: float
            beta: float
        """
        return (
            (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)
            if (precision + recall) > 0
            else 0
        )

    def summarise(self, metric_results: List[Dict]) -> Dict[str, Any]:
        total_valid, total_correct, total = 0, 0, 0
        for res in metric_results:
            total_valid += res["valid"] * 1
            total_correct += res["correct"] * 1
            total += 1

        # Calculate basic metrics
        validity_rate = total_valid / total if total > 0 else 0
        accuracy = total_correct / total if total > 0 else 0

        # Calculate precision, recall, and F-scores
        # Precision: Of the valid answers, how many are correct
        precision = total_correct / total_valid if total_valid > 0 else 0

        # Recall: How many correct answers out of total questions
        recall = total_correct / total if total > 0 else 0

        # F1 score (balanced precision and recall)
        f1 = self._f_score(precision, recall, beta=1.0)

        # F2 score (emphasizes recall over precision)
        f2 = self._f_score(precision, recall, beta=2.0)

        # F0.5 score (emphasizes precision over recall)
        f0_5 = self._f_score(precision, recall, beta=0.5)

        return {
            "total": total,
            "valid": total_valid,
            "valid_rate": validity_rate,
            "correct": total_correct,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
            "f0.5": f0_5,
        }
