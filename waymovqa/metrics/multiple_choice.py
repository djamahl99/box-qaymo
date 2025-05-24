from abc import ABC, abstractmethod
from typing import Dict, Any, List

from pathlib import Path

from waymovqa.questions.base import BaseQuestion
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer

from .base import BaseMetric

from collections import defaultdict, Counter
import numpy as np


class MultipleChoiceMetric(BaseMetric[MultipleChoiceAnswer]):
    """Evaluates multiple choice answers with comprehensive per-class metrics."""

    def __init__(self):
        super().__init__(MultipleChoiceAnswer)

    def evaluate(
        self,
        prediction: MultipleChoiceAnswer,
        ground_truth: MultipleChoiceAnswer,
        question,
    ) -> Dict[str, Any]:
        """
        Evaluate predicted multiple choice answer.

        Returns:
            Dictionary with evaluation results including validity, correctness,
            predicted and ground truth answers for per-class analysis.
        """
        assert prediction.choices == ground_truth.choices

        choices = prediction.choices
        is_valid = prediction.answer in choices
        is_correct = ground_truth.answer == prediction.answer if is_valid else False

        return {
            "valid": is_valid,
            "correct": is_correct,
            "predicted_answer": prediction.answer if is_valid else None,
            "ground_truth_answer": ground_truth.answer,
            "choices": choices,
        }

    def _f_score(self, precision: float, recall: float, beta: float) -> float:
        """
        Calculates F-score.

        Arguments:
            precision: Precision score
            recall: Recall score
            beta: Beta parameter for F-score weighting

        Returns:
            F-score value
        """
        if precision + recall == 0:
            return 0.0
        return (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)

    def _calculate_per_class_metrics(
        self, metric_results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F-scores for each answer choice.

        Arguments:
            metric_results: List of evaluation results

        Returns:
            Dictionary mapping each choice to its metrics
        """
        # Collect all possible choices
        all_choices = set()
        for result in metric_results:
            all_choices.update(result["choices"])

        per_class_metrics = {}

        for choice in sorted(all_choices):
            # Calculate confusion matrix elements for this class
            tp = sum(
                1
                for r in metric_results
                if r["ground_truth_answer"] == choice
                and r["predicted_answer"] == choice
            )
            fp = sum(
                1
                for r in metric_results
                if r["ground_truth_answer"] != choice
                and r["predicted_answer"] == choice
            )
            fn = sum(
                1
                for r in metric_results
                if r["ground_truth_answer"] == choice
                and r["predicted_answer"] != choice
            )
            tn = sum(
                1
                for r in metric_results
                if r["ground_truth_answer"] != choice
                and r["predicted_answer"] != choice
            )

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Support (number of true instances for this class)
            support = tp + fn

            # F-scores
            f1 = self._f_score(precision, recall, beta=1.0)
            f2 = self._f_score(precision, recall, beta=2.0)
            f0_5 = self._f_score(precision, recall, beta=0.5)

            per_class_metrics[choice] = {
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "f2": f2,
                "f0.5": f0_5,
                "support": support,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
            }

        return per_class_metrics

    def _calculate_macro_metrics(
        self, per_class_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate macro-averaged metrics across all classes.

        Arguments:
            per_class_metrics: Per-class metrics dictionary

        Returns:
            Dictionary of macro-averaged metrics
        """
        if not per_class_metrics:
            return {}

        metrics_to_average = ["precision", "recall", "specificity", "f1", "f2", "f0.5"]
        macro_metrics = {}

        for metric in metrics_to_average:
            values = [
                class_metrics[metric] for class_metrics in per_class_metrics.values()
            ]
            macro_metrics[f"macro_{metric}"] = np.mean(values)

        return macro_metrics

    def _calculate_weighted_metrics(
        self, per_class_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate weighted-averaged metrics across all classes.

        Arguments:
            per_class_metrics: Per-class metrics dictionary

        Returns:
            Dictionary of weighted-averaged metrics
        """
        if not per_class_metrics:
            return {}

        total_support = sum(
            class_metrics["support"] for class_metrics in per_class_metrics.values()
        )
        if total_support == 0:
            return {}

        metrics_to_average = ["precision", "recall", "specificity", "f1", "f2", "f0.5"]
        weighted_metrics = {}

        for metric in metrics_to_average:
            weighted_sum = sum(
                class_metrics[metric] * class_metrics["support"]
                for class_metrics in per_class_metrics.values()
            )
            weighted_metrics[f"weighted_{metric}"] = weighted_sum / total_support

        return weighted_metrics

    def _calculate_confusion_matrix(self, metric_results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate and format confusion matrix.

        Arguments:
            metric_results: List of evaluation results

        Returns:
            Dictionary containing confusion matrix data
        """
        # Get all choices
        all_choices = set()
        for result in metric_results:
            all_choices.update(result["choices"])
        choices = sorted(all_choices)

        # Initialize confusion matrix
        matrix = defaultdict(lambda: defaultdict(int))

        # Fill confusion matrix
        for result in metric_results:
            true_label = result["ground_truth_answer"]
            pred_label = result["predicted_answer"]

            if pred_label is not None:  # Only count valid predictions
                matrix[true_label][pred_label] += 1

        # Convert to regular dict for JSON serialization
        confusion_matrix = {
            true_label: dict(matrix[true_label]) for true_label in choices
        }

        return {
            "matrix": confusion_matrix,
            "labels": choices,
        }

    def summarise(self, metric_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive summary of multiple choice evaluation results.

        Arguments:
            metric_results: List of evaluation results from evaluate() method

        Returns:
            Dictionary containing overall metrics, per-class metrics, and confusion matrix
        """
        if not metric_results:
            return {"error": "No results to summarize"}

        # Basic counts
        total = len(metric_results)
        total_valid = sum(1 for res in metric_results if res["valid"])
        total_correct = sum(1 for res in metric_results if res["correct"])

        # Overall metrics
        validity_rate = total_valid / total if total > 0 else 0
        accuracy = total_correct / total if total > 0 else 0

        # For overall precision/recall, we treat this as a binary classification problem
        # where we're predicting "correct" vs "incorrect"
        precision = total_correct / total_valid if total_valid > 0 else 0
        recall = total_correct / total if total > 0 else 0

        # Overall F-scores
        f1 = self._f_score(precision, recall, beta=1.0)
        f2 = self._f_score(precision, recall, beta=2.0)
        f0_5 = self._f_score(precision, recall, beta=0.5)

        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(metric_results)
        macro_metrics = self._calculate_macro_metrics(per_class_metrics)
        weighted_metrics = self._calculate_weighted_metrics(per_class_metrics)

        # Confusion matrix
        confusion_matrix = self._calculate_confusion_matrix(metric_results)

        # Answer distribution
        ground_truth_dist = Counter(r["ground_truth_answer"] for r in metric_results)
        prediction_dist = Counter(
            r["predicted_answer"]
            for r in metric_results
            if r["predicted_answer"] is not None
        )

        return {
            # Overall metrics
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
            # Per-class metrics
            "per_class_metrics": per_class_metrics,
            "macro_metrics": macro_metrics,
            "weighted_metrics": weighted_metrics,
            # Additional analysis
            "confusion_matrix": confusion_matrix,
            "ground_truth_distribution": dict(ground_truth_dist),
            "prediction_distribution": dict(prediction_dist),
            # Class-specific summary
            "num_classes": len(per_class_metrics),
            "best_performing_class": (
                max(per_class_metrics.keys(), key=lambda x: per_class_metrics[x]["f1"])
                if per_class_metrics
                else None
            ),
            "worst_performing_class": (
                min(per_class_metrics.keys(), key=lambda x: per_class_metrics[x]["f1"])
                if per_class_metrics
                else None
            ),
        }
