from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union

from pathlib import Path

from waymovqa.questions.single_image_multi_choice import SingleImageMultipleChoiceQuestion
from waymovqa.questions.multi_image_multi_choice import MultipleImageMultipleChoiceQuestion
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer

from .base import BaseMetric

from collections import defaultdict, Counter
import numpy as np

class MultipleChoiceMetric(BaseMetric[MultipleChoiceAnswer]):
    """Evaluates multiple choice answers with comprehensive per-class metrics."""

    def __init__(self):
        super().__init__(MultipleChoiceAnswer)
        
        # Question type mappings based on your templates
        self.question_type_mappings = {
            # Temporal Trajectory Questions
            "_prompt_faster_than_ego": ("motion", "Relative Speed"),
            "_prompt_moving_towards_ego": ("motion", "Approach/Divergence"),
            "_prompt_parallel_motion": ("motion", "Approach/Divergence"),
            "_prompt_approaching_stop_sign": ("motion", "Path Prediction"),
            "_prompt_vehicle_future_path": ("motion", "Path Prediction"),
            "_prompt_ego_following": ("motion", "Following Behavior"),
            
            # Instance-Referenced Questions (with bounding boxes)
            "_color_prompt": ("attribute", "Color"),
            "_label_prompt": ("attribute", "Object Type"),
            "_heading_prompt": ("attribute", "Orientation"),
            "_speed_prompt": ("attribute", "Speed"),
            "_movement_direction_prompt": ("attribute", "Movement Direction"),
            
            # Binary Characteristic Questions
            "_object_facing_type": ("binary", "Facing Direction"),
            "_object_movement_direction_type": ("binary", "Movement Direction"),
            # Speed category questions will be detected by pattern matching
        }

    def _classify_question_type(self, question_name: str) -> tuple:
        """Classify question into main category and subcategory."""
        if question_name in self.question_type_mappings:
            return self.question_type_mappings[question_name]
        
        # Pattern matching for speed category questions
        if any(speed_cat in question_name.lower() for speed_cat in 
               ["stationary", "slow", "medium", "highway", "walking", "jogging", "running"]):
            return ("binary", "Speed Categories")
        
        # Default classification for object presence questions
        if any(obj_type in question_name.lower() for obj_type in 
               ["vehicle", "pedestrian", "cyclist", "sign"]):
            return ("binary", "Object Presence")
        
        return ("other", "Unknown")

    def evaluate(
        self,
        prediction: MultipleChoiceAnswer,
        ground_truth: MultipleChoiceAnswer,
        question: Union[SingleImageMultipleChoiceQuestion, MultipleImageMultipleChoiceQuestion],
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
            "question_name": question.question_name
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
            question_metric_results = [x for x in metric_results if x['question_name'] == question_name]
            per_class_metrics = self._calculate_per_class_metrics(question_metric_results)
            macro_metrics = self._calculate_macro_metrics(per_class_metrics) # type: ignore
            weighted_metrics = self._calculate_weighted_metrics(per_class_metrics) # type: ignore
            per_question_metrics[question_name] = dict(
                per_class_metrics=per_class_metrics,
                macro_metric=macro_metrics,
                weighted_metrics=weighted_metrics
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
        
        per_question_name_metrics = self._calculate_per_question_name_metrics(metric_results)

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

    def generate_latex_tables(self, metric_results: List[Dict], model_name: str) -> Dict[str, str]:
        """
        Generate LaTeX table rows for different question categories.
        
        Arguments:
            metric_results: List of evaluation results from evaluate() method
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary containing LaTeX table rows for each table type
        """
        if not metric_results:
            return {"error": "No results to generate tables"}
        
        summary = self.summarise(metric_results)
        
        # Group results by question type
        type_groups = defaultdict(list)
        subtype_groups = defaultdict(list)
        
        for result in metric_results:
            main_type, sub_type = self._classify_question_type(result["question_name"])
            type_groups[main_type].append(result)
            subtype_groups[(main_type, sub_type)].append(result)
        
        tables = {}
        
        # Overall Performance Table
        tables["overall"] = self._generate_overall_table_row(summary, model_name)
        
        # Binary Questions Tables (both overall and subtypes)
        if "binary" in type_groups:
            tables["binary_overall"] = self._generate_binary_overall_row(type_groups["binary"], model_name)
            tables["binary_subtypes"] = self._generate_binary_subtype_rows(subtype_groups, model_name)
        
        # Attribute Questions Tables (both overall and subtypes)
        if "attribute" in type_groups:
            tables["attribute_overall"] = self._generate_attribute_overall_row(type_groups["attribute"], model_name)
            tables["attribute_subtypes"] = self._generate_attribute_subtype_rows(subtype_groups, model_name)
        
        # Motion Questions Tables (both overall and subtypes)
        if "motion" in type_groups:
            tables["motion_overall"] = self._generate_motion_overall_row(type_groups["motion"], model_name)
            tables["motion_subtypes"] = self._generate_motion_subtype_rows(subtype_groups, model_name)
        
        return tables

    def _generate_overall_table_row(self, summary: Dict[str, Any], model_name: str) -> str:
        """Generate overall performance table row."""
        accuracy = summary["accuracy"] * 100
        valid_rate = summary["valid_rate"] * 100
        f1 = summary["f1"]
        
        return f"{model_name} & {accuracy:.1f} & {valid_rate:.1f} & {f1:.2f} \\\\"

    def _generate_binary_overall_row(self, binary_results: List[Dict], model_name: str) -> str:
        """Generate binary questions overall performance row."""
        binary_summary = self.summarise(binary_results)
        accuracy = binary_summary["accuracy"] * 100
        precision = binary_summary["precision"]
        recall = binary_summary["recall"]
        
        return f"{model_name} & {accuracy:.1f} & {precision:.2f} & {recall:.2f} \\\\"

    def _generate_binary_subtype_rows(self, subtype_groups: Dict[tuple, List[Dict]], model_name: str) -> str:
        """Generate binary questions subtype performance rows."""
        rows = []
        
        binary_subtypes = [
            ("Object Presence", ("binary", "Object Presence")),
            ("Movement Direction", ("binary", "Movement Direction")),
            ("Speed Categories", ("binary", "Speed Categories")),
        ]
        
        for display_name, key in binary_subtypes:
            if key in subtype_groups and subtype_groups[key]:
                subtype_summary = self.summarise(subtype_groups[key])
                sub_accuracy = subtype_summary["accuracy"] * 100
                sub_precision = subtype_summary["precision"]
                sub_recall = subtype_summary["recall"]
                rows.append(f"{model_name} & {display_name} & {sub_accuracy:.1f} & {sub_precision:.2f} & {sub_recall:.2f} \\\\")
        
        return "\n".join(rows)

    def _generate_attribute_overall_row(self, attribute_results: List[Dict], model_name: str) -> str:
        """Generate attribute questions overall performance row."""
        attribute_summary = self.summarise(attribute_results)
        accuracy = attribute_summary["accuracy"] * 100
        precision = attribute_summary["precision"]
        recall = attribute_summary["recall"]
        
        return f"{model_name} & {accuracy:.1f} & {precision:.2f} & {recall:.2f} \\\\"

    def _generate_attribute_subtype_rows(self, subtype_groups: Dict[tuple, List[Dict]], model_name: str) -> str:
        """Generate attribute questions subtype performance rows."""
        rows = []
        
        attribute_subtypes = [
            ("Object Type", ("attribute", "Object Type")),
            ("Color", ("attribute", "Color")),
            ("Orientation", ("attribute", "Orientation")),
            ("Speed", ("attribute", "Speed")),
        ]
        
        for display_name, key in attribute_subtypes:
            if key in subtype_groups and subtype_groups[key]:
                subtype_summary = self.summarise(subtype_groups[key])
                sub_accuracy = subtype_summary["accuracy"] * 100
                sub_precision = subtype_summary["precision"]
                sub_recall = subtype_summary["recall"]
                rows.append(f"{model_name} & {display_name} & {sub_accuracy:.1f} & {sub_precision:.2f} & {sub_recall:.2f} \\\\")
        
        return "\n".join(rows)

    def _generate_motion_overall_row(self, motion_results: List[Dict], model_name: str) -> str:
        """Generate motion questions overall performance row."""
        motion_summary = self.summarise(motion_results)
        accuracy = motion_summary["accuracy"] * 100
        precision = motion_summary["precision"]
        recall = motion_summary["recall"]
        
        return f"{model_name} & {accuracy:.1f} & {precision:.2f} & {recall:.2f} \\\\"

    def _generate_motion_subtype_rows(self, subtype_groups: Dict[tuple, List[Dict]], model_name: str) -> str:
        """Generate motion questions subtype performance rows."""
        rows = []
        
        motion_subtypes = [
            ("Relative Speed", ("motion", "Relative Speed")),
            ("Approach/Divergence", ("motion", "Approach/Divergence")),
            ("Path Prediction", ("motion", "Path Prediction")),
            ("Following Behavior", ("motion", "Following Behavior")),
        ]
        
        for display_name, key in motion_subtypes:
            if key in subtype_groups and subtype_groups[key]:
                subtype_summary = self.summarise(subtype_groups[key])
                sub_accuracy = subtype_summary["accuracy"] * 100
                sub_precision = subtype_summary["precision"]
                sub_recall = subtype_summary["recall"]
                rows.append(f"{model_name} & {display_name} & {sub_accuracy:.1f} & {sub_precision:.2f} & {sub_recall:.2f} \\\\")
        
        return "\n".join(rows)

    def print_latex_tables(self, metric_results: List[Dict], model_name: str):
        """
        Print formatted LaTeX tables ready for copy-paste into paper.
        
        Arguments:
            metric_results: List of evaluation results from evaluate() method
            model_name: Name of the model being evaluated
        """
        tables = self.generate_latex_tables(metric_results, model_name)
        
        if "error" in tables:
            print(f"Error: {tables['error']}")
            return
        
        print("=" * 80)
        print(f"LATEX TABLES FOR MODEL: {model_name}")
        print("=" * 80)
        
        if "overall" in tables:
            print("\n📊 OVERALL PERFORMANCE TABLE ROW:")
            print("-" * 50)
            print(tables["overall"])
        
        if "binary_overall" in tables:
            print("\n🔄 BINARY QUESTIONS - OVERALL TABLE ROW:")
            print("-" * 50)
            print(tables["binary_overall"])
        
        if "binary_subtypes" in tables:
            print("\n🔄 BINARY QUESTIONS - SUBTYPES TABLE ROWS:")
            print("-" * 50)
            print(tables["binary_subtypes"])
        
        if "attribute_overall" in tables:
            print("\n🎯 ATTRIBUTE QUESTIONS - OVERALL TABLE ROW:")
            print("-" * 50)
            print(tables["attribute_overall"])
        
        if "attribute_subtypes" in tables:
            print("\n🎯 ATTRIBUTE QUESTIONS - SUBTYPES TABLE ROWS:")
            print("-" * 50)
            print(tables["attribute_subtypes"])
        
        if "motion_overall" in tables:
            print("\n🚗 MOTION QUESTIONS - OVERALL TABLE ROW:")
            print("-" * 50)
            print(tables["motion_overall"])
        
        if "motion_subtypes" in tables:
            print("\n🚗 MOTION QUESTIONS - SUBTYPES TABLE ROWS:")
            print("-" * 50)
            print(tables["motion_subtypes"])
        
        print("\n" + "=" * 80)
        print("LATEX TABLE TEMPLATES:")
        print("=" * 80)
        
        print("""
    % Overall Performance Table
    \\begin{table}[t]
    \\centering
    \\caption{Overall performance across all question types.}
    \\label{tab:overall}
    \\begin{tabular}{lccc}
    \\toprule
    Model & Accuracy (\\%) & Valid Rate (\\%) & F1 Score \\\\
    \\midrule
    % Insert overall rows here
    \\bottomrule
    \\end{tabular}
    \\end{table}

    % Binary Questions - Overall
    \\begin{table}[t]
    \\centering
    \\caption{Performance on binary characteristic questions.}
    \\label{tab:binary_overall}
    \\begin{tabular}{lccc}
    \\toprule
    Model & Accuracy (\\%) & Precision & Recall \\\\
    \\midrule
    % Insert binary_overall rows here
    \\bottomrule
    \\end{tabular}
    \\end{table}

    % Binary Questions - By Subtype
    \\begin{table}[t]
    \\centering
    \\caption{Performance on binary characteristic questions by subtype.}
    \\label{tab:binary_subtypes}
    \\begin{tabular}{llccc}
    \\toprule
    Model & Question Type & Accuracy (\\%) & Precision & Recall \\\\
    \\midrule
    % Insert binary_subtypes rows here
    \\bottomrule
    \\end{tabular}
    \\end{table}
    """)
        
        print("Copy the rows above and paste them into your LaTeX tables!")
        print("=" * 80)

    def save_latex_tables(self, metric_results: List[Dict], model_name: str, save_path):
        """
        Print formatted LaTeX tables ready for copy-paste into paper.
        
        Arguments:
            metric_results: List of evaluation results from evaluate() method
            model_name: Name of the model being evaluated
        """
        tables = self.generate_latex_tables(metric_results, model_name)
        
        if "error" in tables:
            print(f"Error: {tables['error']}")
            return
        
        out = ""

        out += ("=" * 80)
        out += (f"LATEX TABLES FOR MODEL: {model_name}")
        out += ("=" * 80)
        
        if "overall" in tables:
            out += ("\n📊 OVERALL PERFORMANCE TABLE ROW:")
            out += ("-" * 50)
            out += (tables["overall"])
        
        if "binary_overall" in tables:
            out += ("\n🔄 BINARY QUESTIONS - OVERALL TABLE ROW:")
            out += ("-" * 50)
            out += (tables["binary_overall"])
        
        if "binary_subtypes" in tables:
            out += ("\n🔄 BINARY QUESTIONS - SUBTYPES TABLE ROWS:")
            out += ("-" * 50)
            out += (tables["binary_subtypes"])
        
        if "attribute_overall" in tables:
            out += ("\n🎯 ATTRIBUTE QUESTIONS - OVERALL TABLE ROW:")
            out += ("-" * 50)
            out += (tables["attribute_overall"])
        
        if "attribute_subtypes" in tables:
            out += ("\n🎯 ATTRIBUTE QUESTIONS - SUBTYPES TABLE ROWS:")
            out += ("-" * 50)
            out += (tables["attribute_subtypes"])
        
        if "motion_overall" in tables:
            out += ("\n🚗 MOTION QUESTIONS - OVERALL TABLE ROW:")
            out += ("-" * 50)
            out += (tables["motion_overall"])
        
        if "motion_subtypes" in tables:
            out += ("\n🚗 MOTION QUESTIONS - SUBTYPES TABLE ROWS:")
            out += ("-" * 50)
            out += (tables["motion_subtypes"])
        
        out += ("\n" + "=" * 80)
        out += ("LATEX TABLE TEMPLATES:")
        out += ("=" * 80)
        
        out += ("""
    % Overall Performance Table
    \\begin{table}[t]
    \\centering
    \\caption{Overall performance across all question types.}
    \\label{tab:overall}
    \\begin{tabular}{lccc}
    \\toprule
    Model & Accuracy (\\%) & Valid Rate (\\%) & F1 Score \\\\
    \\midrule
    % Insert overall rows here
    \\bottomrule
    \\end{tabular}
    \\end{table}

    % Binary Questions - Overall
    \\begin{table}[t]
    \\centering
    \\caption{Performance on binary characteristic questions.}
    \\label{tab:binary_overall}
    \\begin{tabular}{lccc}
    \\toprule
    Model & Accuracy (\\%) & Precision & Recall \\\\
    \\midrule
    % Insert binary_overall rows here
    \\bottomrule
    \\end{tabular}
    \\end{table}

    % Binary Questions - By Subtype
    \\begin{table}[t]
    \\centering
    \\caption{Performance on binary characteristic questions by subtype.}
    \\label{tab:binary_subtypes}
    \\begin{tabular}{llccc}
    \\toprule
    Model & Question Type & Accuracy (\\%) & Precision & Recall \\\\
    \\midrule
    % Insert binary_subtypes rows here
    \\bottomrule
    \\end{tabular}
    \\end{table}
    """)
        
        out += ("Copy the rows above and paste them into your LaTeX tables!")
        out += ("=" * 80)

        with open(save_path, 'w') as f:
            f.write(out)