from abc import ABC, abstractmethod
import re
from typing import Dict, Any, List, Union

from pathlib import Path

from waymovqa.metrics.analysis import create_confusion_matrix_plotly
from waymovqa.questions.single_image_multi_choice import (
    SingleImageMultipleChoiceQuestion,
)
from waymovqa.questions.multi_image_multi_choice import (
    MultipleImageMultipleChoiceQuestion,
)
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.answers.raw_text import RawTextAnswer

from .base import BaseMetric

from collections import defaultdict, Counter
import numpy as np

import pprint

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any

import string

SIMILARITY_THRESH = 0.9


def remove_punctuation(text: str):
    """Remove punctuation"""
    return text.translate(str.maketrans("", "", string.punctuation))


def detect_numbered_choice_listing(response_text, choices):
    """
    Detects if the response is listing choices in numbered format like "1 choice1 2 choice2 3 choice3"

    Returns:
        (is_listing, matched_pairs) where:
        - is_listing: True if this appears to be a choice listing format
        - matched_pairs: list of (number, choice) tuples that were matched
    """
    # Pattern to find "number choice" pairs (with optional periods/dots)
    # Matches: "1 yes", "2. no", "3 maybe", etc.
    pattern = r"(\d+)\.?\s+([a-zA-Z][a-zA-Z\s]*?)(?=\s*\d+\.?\s+[a-zA-Z]|\s*$)"

    matches = re.findall(pattern, response_text.lower().strip())

    if len(matches) < 2:  # Need at least 2 numbered items to be considered "listing"
        return False, []

    # Clean up the choices for comparison
    cleaned_choices = [choice.lower().strip() for choice in choices]

    matched_pairs = []
    choice_matches = 0

    for number, text in matches:
        # Clean the matched text
        cleaned_text = text.strip()

        # Check if this text matches any of our actual choices
        for original_choice in cleaned_choices:
            if cleaned_text == original_choice:
                matched_pairs.append((int(number), original_choice))
                choice_matches += 1
                break

    # Consider it a listing if:
    # 1. We found at least 2 numbered items AND
    # 2. At least half of them match our actual choices AND
    # 3. The numbers are sequential starting from 1
    numbers_found = [pair[0] for pair in matched_pairs]
    is_sequential = len(numbers_found) >= 2 and numbers_found == list(
        range(1, len(numbers_found) + 1)
    )

    is_listing = (
        len(matched_pairs) >= 2
        and choice_matches >= len(matched_pairs) * 0.5
        and is_sequential
    )

    return is_listing, matched_pairs


def match_senna_common_responses(response_text):
    senna_common_responses = [
        (
            "the object in the red box is facing towards the right side of the image",
            "right",
        ),
        (
            "the object highlighted in red is facing towards the right side of the image",
            "right",
        ),
        ("no there is no stop sign visible in the image", "no"),
        ("the pedestrian is walking", "walking"),
        (
            "yes there are pedestrians walking and facing towards the right of the camera",
            "yes",
        ),
        (
            "yes there are pedestrians walking and facing towards the left of the camera",
            "yes",
        ),
        ("no pedestrians are visible in the image", "no"),
        ("the adult pedestrian is moving faster than the ego vehicle", "yes"),
        ("the black suv is moving faster than the ego vehicle", "yes"),
        ("the white sedan is moving faster than the ego vehicle", "yes"),
        (
            "the pedestrian is standing on the sidewalk not currently in motion",
            "stationary",
        ),
        (
            "the pedestrian appears to be stationary as they are standing on the sidewalk and not in motion",
            "stationary",
        ),
        ("the pedestrian is standing on the sidewalk not in motion", "stationary"),
        ("the object in the red box is facing towards", "towards"),
        (
            "the pedestrian appears to be stationary possibly waiting to cross the street or at a bus stop",
            "stationary",
        ),
        (
            " the pedestrian is crossing the street at a crosswalk indicating that they are likely in the process of walking the speed of the pedestrian is not clear from the image but it appears to be a normal pace for crossing the street",
            "walking",
        ),
        ("the vehicle is moving faster than the ego vehicle", "yes"),
        ("the black sedan is moving faster than the ego vehicle", "yes"),
        ("the image shows a stop sign that the ego vehicle is approaching", "yes"),
        ("the silver suv is moving faster than the ego vehicle", "yes"),
        ("the gray suv is moving faster than the ego vehicle", "yes"),
        ("the vehicle is moving faster than the ego vehicle", "yes"),
        ("the pedestrian is standing on the sidewalk not in motion", "stationary"),
        ("the object in the red box is facing towards", "towards"),
        (
            "the pedestrian appears to be stationary standing on the sidewalk",
            "stationary",
        ),
        ("the gray sedan is moving faster than the ego vehicle", "yes"),
        ("the white suv is moving faster than the ego vehicle", "yes"),
        ("the vehicle in the image is moving faster than the ego vehicle", "yes"),
        ("the pedestrian is moving faster than the ego vehicle", "yes"),
        ("the silver sedan is moving faster than the ego vehicle", "yes"),
        (
            "the pedestrian is walking on the sidewalk and there is no indication of any sudden movements or high speeds the environment suggests a calm urban setting with no immediate signs of urgency or danger",
            "walking",
        ),
        ("the pedestrian is stationary standing on the sidewalk", "stationary"),
    ]

    best_value, best_common_response, best_similarity = response_text, None, 0.0

    for common_response, value in senna_common_responses:
        sim = MultipleChoiceMetric.compute_text_similarity(
            common_response, response_text
        )

        if sim > best_similarity:
            best_similarity = sim
            best_value = value
            best_common_response = common_response

    valid = (
        best_similarity > SIMILARITY_THRESH
    ) or response_text == best_common_response

    if valid:
        print(
            f"best_similarity={best_similarity}\nresponse_text={response_text}\nbest_common_response={best_common_response}\nbest_value={best_value}"
        )
        # exit()

    return valid, best_value


class MultipleChoiceMetric(BaseMetric[MultipleChoiceAnswer]):
    """Evaluates multiple choice answers with comprehensive per-class metrics."""

    def __init__(self):
        super().__init__(MultipleChoiceAnswer)

        # Question type mappings based on your templates
        self.question_type_mappings = {
            # Temporal Trajectory Questions
            "_prompt_faster_than_ego": ("motion", "Ego Relative Speed"),
            "_prompt_moving_towards_ego": ("motion", "Approach/Divergence"),
            "_prompt_parallel_motion": ("motion", "Parallel/Perpendicular"),
            "_prompt_approaching_stop_sign": ("motion", "Approaching Stop Sign"),
            "_prompt_vehicle_future_path": ("motion", "Ego Collision Prediction"),
            "_prompt_ego_following": ("motion", "Following Behavior"),
            # Instance-Referenced Questions (with bounding boxes)
            "_color_prompt": ("attribute", "Color"),
            "_label_prompt": ("attribute", "Object Type"),
            "_heading_prompt": ("attribute", "Orientation"),
            "_speed_prompt": ("motion", "Object Speed"),
            "_movement_direction_prompt": ("attribute", "Movement Direction"),
            # Binary Characteristic Questions
            "_object_facing_type": ("binary", "Facing Direction"),
            "_object_movement_direction_type": ("binary", "Movement Direction"),
            # Speed category questions will be detected by pattern matching
        }

        self.invalid_responses = defaultdict(int)
        self.invalid_pairs = defaultdict(int)
        self.parse_type_counts = defaultdict(int)

    def _classify_question_type(self, question_name: str) -> tuple:
        """Classify question into main category and subcategory."""
        if question_name in self.question_type_mappings:
            return self.question_type_mappings[question_name]

        return ("other", "Unknown")

    @staticmethod
    def compute_text_similarity(text1: str, text2: str) -> float:
        """
        Compute simple text similarity between two strings.
        """
        # This is a simple implementation - you might want to use a more sophisticated approach
        text1_tokens = set(text1.lower().split())
        text2_tokens = set(text2.lower().split())
        intersection = text1_tokens.intersection(text2_tokens)
        union = text1_tokens.union(text2_tokens)
        return len(intersection) / len(union) if union else 0

    def parse_response(self, prediction, ground_truth):
        choices = ground_truth.choices
        response_text = prediction.text

        # Preprocessing
        cleaned_response = remove_punctuation(response_text).lower()
        cleaned_choices = [x.lower() for x in choices]

        # Check choice containment
        present_choices = [
            choice for choice in cleaned_choices if choice in cleaned_response
        ]

        # Stage 1: Exact single match
        if len(present_choices) == 1 and cleaned_response in cleaned_choices:
            self.parse_type_counts["exact_matching"] += 1
            return choices[cleaned_choices.index(cleaned_response)], True

        # # Stage 2: Common response patterns
        # is_senna_common, senna_mapping = match_senna_common_responses(cleaned_response)
        # if is_senna_common:
        #     self.parse_type_counts["common_response"] += 1
        #     return senna_mapping, True

        # Stage 3: Ranked listing detection
        if len(present_choices) > 1:
            is_listing, matched_pairs = detect_numbered_choice_listing(
                cleaned_response, cleaned_choices
            )
            if is_listing:
                self.parse_type_counts["ranked_listing"] += 1

                return sorted(matched_pairs, key=lambda x: x[0])[0][1], True

        # Stage 4: Similarity fallback
        # if len(present_choices) != 1:
        #     chosen_choice = max(choices,
        #         key=lambda x: self.compute_text_similarity(x.lower(), response_text))
        #     similarity_score = self.compute_text_similarity(
        #         chosen_choice.lower(), response_text)

        #     if similarity_score >= SIMILARITY_THRESH:
        #         self.parse_type_counts["text_similarity"] += 1

        #         return chosen_choice, True

        self.parse_type_counts["not_parsed"] += 1

        return None, False

    def evaluate(
        self,
        prediction: Union[MultipleChoiceAnswer, RawTextAnswer],
        ground_truth: MultipleChoiceAnswer,
        question: Union[
            SingleImageMultipleChoiceQuestion, MultipleImageMultipleChoiceQuestion
        ],
    ) -> Dict[str, Any]:
        """
        Evaluate predicted multiple choice answer.

        Returns:
            Dictionary with evaluation results including validity, correctness,
            predicted and ground truth answers for per-class analysis.
        """
        if isinstance(prediction, MultipleChoiceAnswer):
            assert prediction.choices == ground_truth.choices
            choices = prediction.choices
            is_valid = prediction.answer in choices
            predicted_answer = prediction.answer if is_valid else None

        elif isinstance(prediction, RawTextAnswer):
            predicted_answer, is_valid = self.parse_response(prediction, ground_truth)
            choices = ground_truth.choices
            cleaned_response = remove_punctuation(prediction.text).lower()

            if not is_valid:
                self.invalid_responses[cleaned_response] += 1
                self.invalid_pairs[(question.question, cleaned_response)] += 1
        else:
            raise TypeError(
                f"Prediction should be RawTextAnswer or MultipleChoiceAnswer. Got {type(prediction)}"
            )

        is_correct = (ground_truth.answer == predicted_answer) if is_valid else False

        return {
            "valid": is_valid,
            "correct": is_correct,
            "predicted_answer": predicted_answer,  # None for invalid predictions
            "ground_truth_answer": ground_truth.answer,
            "choices": choices,
            "question_name": question.question_name,
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

    def _calculate_per_question_name_metrics(
        self, metric_results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate precision, recall, and F-scores for each question_name.

        Arguments:
            metric_results: List of evaluation results

        Returns:
            Dictionary mapping each choice to its metrics
        """
        # Collect all possible question_name values
        all_question_names = set()
        for result in metric_results:
            all_question_names.add(result["question_name"])

        per_question_metrics = {}

        for question_name in sorted(all_question_names):
            # Calculate confusion matrix elements for this class
            question_metric_results = [
                x for x in metric_results if x["question_name"] == question_name
            ]
            per_class_metrics = self._calculate_per_class_metrics(
                question_metric_results
            )
            macro_metrics = self._calculate_macro_metrics(per_class_metrics)  # type: ignore
            weighted_metrics = self._calculate_weighted_metrics(per_class_metrics)  # type: ignore
            per_question_metrics[question_name] = dict(
                per_class_metrics=per_class_metrics,
                macro_metric=macro_metrics,
                weighted_metrics=weighted_metrics,
            )

        return per_question_metrics

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

        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(metric_results)
        macro_metrics = self._calculate_macro_metrics(per_class_metrics)
        weighted_metrics = self._calculate_weighted_metrics(per_class_metrics)

        per_question_name_metrics = self._calculate_per_question_name_metrics(
            metric_results
        )

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
            # Per-class metrics
            "per_class_metrics": per_class_metrics,
            "macro_metrics": macro_metrics,
            "weighted_metrics": weighted_metrics,
            "per_question_name_metrics": per_question_name_metrics,
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

    def generate_latex_tables(
        self, metric_results: List[Dict], model_name: str
    ) -> Dict[str, str]:
        """
        Generate simplified LaTeX table rows for question subtypes.

        Arguments:
            metric_results: List of evaluation results from evaluate() method
            model_name: Name of the model being evaluated

        Returns:
            Dictionary containing LaTeX table rows for each table type
        """
        if not metric_results:
            return {"error": "No results to generate tables"}

        # Group results by question type and subtype
        subtype_groups = defaultdict(list)

        for result in metric_results:
            main_type, sub_type = self._classify_question_type(result["question_name"])
            subtype_groups[(main_type, sub_type)].append(result)

        tables = {}

        # Generate tables for each main type with subtypes
        tables["binary"] = self._generate_subtype_table(
            subtype_groups, "binary", model_name
        )
        tables["attribute"] = self._generate_subtype_table(
            subtype_groups, "attribute", model_name
        )
        tables["motion"] = self._generate_subtype_table(
            subtype_groups, "motion", model_name
        )

        for key in subtype_groups.keys():
            main_type, sub_type = key
            subtype_summary = self.summarise(subtype_groups[key])

            fig = create_confusion_matrix_plotly(
                subtype_summary["confusion_matrix"],
                question_name=f"{main_type.title()} {sub_type.title()}",
                normalize="true",  # Normalize by true class
                show_percentages=True,
            )
            fig.write_image(
                f"figures/confmat_{model_name}_{main_type.lower().replace(' ', '')}_{sub_type.lower().replace(' ', '').replace('/', '-')}.png"
            )

        return tables

    def generate_tables_values(self, metric_results: List[Dict], model_name: str):
        """
        Generate simplified LaTeX table rows for question subtypes.

        Arguments:
            metric_results: List of evaluation results from evaluate() method
            model_name: Name of the model being evaluated

        Returns:
            Dictionary containing LaTeX table rows for each table type
        """
        if not metric_results:
            return {"error": "No results to generate tables"}

        # Group results by question type and subtype
        subtype_groups = defaultdict(list)

        for result in metric_results:
            main_type, sub_type = self._classify_question_type(result["question_name"])
            subtype_groups[(main_type, sub_type)].append(result)

        tables = {}

        # Generate tables for each main type with subtypes
        overall_summary = self.summarise(metric_results)

        macro_precision = overall_summary["macro_metrics"]["macro_precision"] * 100
        macro_recall = overall_summary["macro_metrics"]["macro_recall"] * 100
        macro_f1 = overall_summary["macro_metrics"]["macro_f1"] * 100

        tables["overall"] = [(model_name, macro_precision, macro_recall, macro_f1)]

        tables["binary"] = self._generate_subtype_values(
            subtype_groups, "binary", model_name
        )
        tables["attribute"] = self._generate_subtype_values(
            subtype_groups, "attribute", model_name
        )
        tables["motion"] = self._generate_subtype_values(
            subtype_groups, "motion", model_name
        )

        return tables

    def _generate_subtype_values(
        self,
        subtype_groups: Dict[tuple, List[Dict]],
        question_type: str,
        model_name: str,
    ):
        """Generate subtype performance rows for a given question type."""
        rows = []

        # Get subtypes for this question type
        subtypes = [
            key
            for key in self.question_type_mappings.values()
            if key[0] == question_type
        ]

        for key in subtypes:
            if key in subtype_groups and subtype_groups[key]:
                subtype_summary = self.summarise(subtype_groups[key])
                macro_precision = (
                    subtype_summary["macro_metrics"]["macro_precision"] * 100
                )
                macro_recall = subtype_summary["macro_metrics"]["macro_recall"] * 100
                macro_f1 = subtype_summary["macro_metrics"]["macro_f1"] * 100

                # Get display name
                _, display_name = key

                rows.append(
                    (model_name, display_name, macro_precision, macro_recall, macro_f1)
                )

        return rows

    def _generate_subtype_table(
        self,
        subtype_groups: Dict[tuple, List[Dict]],
        question_type: str,
        model_name: str,
    ) -> str:
        """Generate subtype performance rows for a given question type."""
        rows = []

        # Get subtypes for this question type
        subtypes = [
            key
            for key in self.question_type_mappings.values()
            if key[0] == question_type
        ]

        for key in subtypes:
            if key in subtype_groups and subtype_groups[key]:
                subtype_summary = self.summarise(subtype_groups[key])
                macro_precision = (
                    subtype_summary["macro_metrics"]["macro_precision"] * 100
                )
                macro_recall = subtype_summary["macro_metrics"]["macro_recall"] * 100
                macro_f1 = subtype_summary["macro_metrics"]["macro_f1"] * 100

                # Get display name
                _, display_name = key

                rows.append(
                    f"{model_name} & {display_name} & {macro_precision:.1f} & {macro_recall:.1f} & {macro_f1:.1f}  \\\\"
                )

        return "\n".join(rows)

    def print_latex_tables(self, metric_results: List[Dict], model_name: str):
        """
        Print simplified LaTeX tables ready for copy-paste into paper.

        Arguments:
            metric_results: List of evaluation results from evaluate() method
            model_name: Name of the model being evaluated
        """
        tables = self.generate_latex_tables(metric_results, model_name)

        if "error" in tables:
            print(f"Error: {tables['error']}")
            return

        # Table configurations
        table_configs = [
            {
                "key": "binary",
                "title": "Binary Questions by Subtype",
                "caption": "Performance on binary characteristic questions by subtype.",
                "label": "tab:binary_subtypes",
            },
            {
                "key": "attribute",
                "title": "Attribute Questions by Subtype",
                "caption": "Performance on attribute questions by subtype.",
                "label": "tab:attribute_subtypes",
            },
            {
                "key": "motion",
                "title": "Motion Questions by Subtype",
                "caption": "Performance on motion questions by subtype.",
                "label": "tab:motion_subtypes",
            },
        ]

        out = ""

        # Generate each table
        for config in table_configs:
            if config["key"] in tables and tables[config["key"]]:
                out += f"\n% {config['title']}\n"
                out += "\\begin{table}[t]\n"
                out += "\\centering\n"
                out += f"\\caption{{{config['caption']}}}\n"
                out += f"\\label{{{config['label']}}}\n"
                out += "\\adjustbox{width=\\columnwidth,center}{% \n"
                out += "\\begin{tabular}{llccc}\n"
                out += "\\toprule\n"
                # out += "Model & Question Type & Accuracy (\\%) & Precision (\\%) & Recall (\\%) \\\\\n"
                out += "Model & Question Type & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\\n"
                out += "\\midrule\n"
                out += tables[config["key"]] + "\n"
                out += "\\bottomrule\n"
                out += "\\end{tabular}\n"
                out += "}\n"
                out += "\\end{table}\n"

        print("\n" + "=" * 80)
        print("Simplified tables are ready to copy-paste into your LaTeX document!")
        print("=" * 80)

    def save_latex_tables(
        self, metric_results: List[Dict], model_name: str, save_path: str
    ):
        """
        Save simplified LaTeX tables to file.

        Arguments:
            metric_results: List of evaluation results from evaluate() method
            model_name: Name of the model being evaluated
            save_path: Path to save the LaTeX tables
        """
        tables = self.generate_latex_tables(metric_results, model_name)

        if "error" in tables:
            print(f"Error: {tables['error']}")
            return

        out = ""

        # Table configurations
        table_configs = [
            {
                "key": "binary",
                "title": "Binary Questions by Subtype",
                "caption": "Performance on binary characteristic questions by subtype.",
                "label": "tab:binary_subtypes",
            },
            {
                "key": "attribute",
                "title": "Attribute Questions by Subtype",
                "caption": "Performance on attribute questions by subtype.",
                "label": "tab:attribute_subtypes",
            },
            {
                "key": "motion",
                "title": "Motion Questions by Subtype",
                "caption": "Performance on motion questions by subtype.",
                "label": "tab:motion_subtypes",
            },
        ]

        # Generate each table
        for config in table_configs:
            if config["key"] in tables and tables[config["key"]]:
                out += f"\n% {config['title']}\n"
                out += "\\begin{table}[t]\n"
                out += "\\centering\n"
                out += f"\\caption{{{config['caption']}}}\n"
                out += f"\\label{{{config['label']}}}\n"
                out += "\\begin{tabular}{llccc}\n"
                out += "\\toprule\n"
                # out += "Model & Question Type & Accuracy (\\%) & Precision (\\%) & Recall (\\%) \\\\\n"
                # out += "Model & Question Type & Precision (\\%) & Recall (\\%) \\\\\n"
                out += "Model & Question Type & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\\n"
                out += "\\midrule\n"
                out += tables[config["key"]] + "\n"
                out += "\\bottomrule\n"
                out += "\\end{tabular}\n"
                out += "\\end{table}\n"

        out += "\n" + "=" * 80 + "\n"
        out += "Simplified tables are ready to copy-paste into your LaTeX document!\n"
        out += "=" * 80 + "\n"

        with open(save_path, "w") as f:
            f.write(out)

        print(f"LaTeX tables saved to: {save_path}")
