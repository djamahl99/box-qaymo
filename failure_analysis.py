from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import importlib
import os
from collections import defaultdict, Counter
import pandas as pd
import json
from pprint import pprint

from box_qaymo.data.vqa_dataset import VQADataset, VQASample
from box_qaymo.metrics.multiple_choice import MultipleChoiceMetric
from box_qaymo.questions.base import BaseQuestion
from box_qaymo.waymo_loader import WaymoDatasetLoader
import string


MODEL_SUFFIX_MAP = {
    # "llava-v1.5-7b": "LLaVA",
    # "llava-v1.5-7b_nobbox": "LLaVA - No Drawn Box",
    "llava-v1.5-7b_raw_wchoices": "LLaVA",
    "llava-v1.5-7b_raw_wchoices_nodrawnbbox": "LLaVA - No Drawn Box",
    "qwen-vlnobbox_wchoices": "Qwen-VL - No Drawn Box",
    "qwen-vlwchoices": "Qwen-VL",
    "llava-v1.5-7b-lorafinetuned_wchoices": "LLaVA†",
    # "Senna_results_0527": "SENNA - No Choices",
    # "Senna_results_0528__FT_senna_pred": "Senna Finetuned_",
    # "Senna_results_0528__senna_pred": "Senna",
    # "senna_multi_frame_results_0529": "Senna",
    "single_frame_results_0528__senna_pred": "Senna",
    "single_frame_results_0528__FT_senna_pred": "Senna†",
    # "single_frame_results_0528_multiframe__senna_pred": "Senna_multi",
    # "multi_frame_results_0529__multiframeconverted_senna_pred": "Senna_multi",
    # "llava-multiframe-wbbox-wchoices": "LLaVA_multi",
}

QUESTION_TYPE_MAPPINGS = {
    # Binary Characteristic Questions
    "_object_movement_direction_type": ("binary", "Movement Status"),
    "_object_facing_type": ("binary", "Orientation"),
    
    # Instance-Referenced Questions (with bounding boxes)
    "_label_prompt": ("attribute", "Fine-grained Classification"),
    "_color_prompt": ("attribute", "Color Recognition"),
    "_heading_prompt": ("attribute", "Facing Direction"),
    
    # Temporal Trajectory Questions
    "_speed_prompt": ("motion", "Speed Assessment"),
    "_movement_direction_prompt": ("motion", "Movement Direction"),
    "_prompt_faster_than_ego": ("motion", "Relative Motion Analysis"),
    "_prompt_approaching_stop_sign": ("motion", "Traffic Element Recognition"),
    "_prompt_moving_towards_ego": ("motion", "Trajectory Analysis"),
    "_prompt_parallel_motion": ("motion", "Relative Motion Direction"),
    "_prompt_vehicle_future_path": ("motion", "Path Conflict Detection"),
    # "_prompt_ego_following": ("motion", "Following Behavior"),
    
}

import plotly.graph_objects as go
from PIL import Image
import textwrap
from typing import Optional, Dict

def visualise_sample_plotly(
    question_obj,
    answer_obj,
    save_path: str,
    model_predictions: Optional[Dict[str, str]] = None,
):
    """
    Minimal Plotly visualization showing question, multiple choices, and image with bounding box.
    Shows model predictions as votes next to each choice.
    
    Args:
        question_obj: Question object with image_path, question text, and optional bbox data
        answer_obj: Answer object with choices and correct answer
        save_path: Path to save the visualization
        model_predictions: Dict mapping model names to their predicted answers
    """
    
    # Load image
    try:
        image_path = None
        if hasattr(question_obj, 'image_paths'):
            # choose the image_path
            image_path = question_obj.image_paths[question_obj.camera_names.index(question_obj.data['best_camera_name'])]
        else:
            image_path = question_obj.image_path
        
        img = Image.open(image_path)
        width, height = img.size
    except Exception as e:
        print("e", e)
        print(question_obj.image_path)
        
        raise e
        # Create a placeholder if image can't be loaded
        img = Image.new('RGB', (600, 400), color='white')
        width, height = img.size
    
    # Create figure
    fig = go.Figure()
    
    # Add image
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=height,
            sizex=width,
            sizey=height,
            sizing="stretch",
            layer="below"
        )
    )
    
    # Add bounding box if available
    if isinstance(question_obj.data, dict) and "bbox" in question_obj.data:
        x1, y1, x2, y2 = question_obj.data['bbox']
        fig.add_shape(
            type="rect",
            x0=x1, y0=height-y2,  # Flip y-coordinate for plotly
            x1=x2, y1=height-y1,
            line=dict(color="red", width=6),
            fillcolor="rgba(0,0,0,0)"  # No fill
        )
    
    question_text = question_obj.question.split("?")[0] + "?"
    
    # Add question text at top
    question_text = textwrap.fill(
        question_text, 
        width=10,
        break_long_words=False,  # Don't break in middle of words
        break_on_hyphens=False   # Don't break on hyphens
    )
    fig.add_annotation(
        text=f"Q: {question_text}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=48, color="black", family="Arial Black"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="white",
        borderwidth=1
    )
    
    # Process model predictions
    model_predictions = model_predictions or {}
    correct_answer = answer_obj.answer
    
    choices = answer_obj.choices + ["OTHER"]
    
    models_matched = set()
    
    y_off = 0.16
    
    for idx, choice in enumerate(choices):
        is_correct = (choice == correct_answer)
        
        # Base choice styling
        if is_correct:
            choice_text = f"{choice}"
            bg_color = "rgba(144,238,144,0.8)"  # Light green
            text_color = "darkgreen"
        else:
            choice_text = f"{choice}"
            bg_color = "rgba(240,128,128,0.8)"  # Light coral
            text_color = "darkred"
        
        # Add choice annotation
        fig.add_annotation(
            text=choice_text,
            xref="paper", yref="paper",
            x=0.02, y=0.1 + idx * y_off,
            xanchor="left", yanchor="bottom",
            showarrow=False,
            font=dict(size=48, color=text_color, family="Arial Black"),
            bgcolor=bg_color,
            bordercolor=text_color,
            borderwidth=2
        )
        
        # Add model votes to the right of each choice
        vote_x_start = 0.2  # Start votes to the right of choices
        vote_spacing = 0.12  # Space between each model vote
        
        for vote_idx, (model_name, predicted_answer) in enumerate(model_predictions.items()):
            matches_model_output = (remove_punctuation(predicted_answer.lower()) == choice) or (choice == "OTHER" and model_name not in models_matched)
            
            if matches_model_output:
                models_matched.add(model_name)
                # Model chose this option
                if is_correct:
                    # Correct prediction
                    vote_symbol = "✓"
                    vote_bg_color = "rgba(0,128,0,0.9)"  # Green
                    vote_text_color = "white"
                else:
                    # Incorrect prediction
                    vote_symbol = "✗"
                    vote_bg_color = "rgba(255,0,0,0.9)"  # Red
                    vote_text_color = "white"
                
                # Add vote symbol
                fig.add_annotation(
                    text=vote_symbol,
                    xref="paper", yref="paper",
                    x=vote_x_start + vote_idx * vote_spacing,
                    y=0.1 + idx * y_off,
                    xanchor="center", yanchor="bottom",
                    showarrow=False,
                    font=dict(size=36, color=vote_text_color, family="Arial Black"),
                    bgcolor=vote_bg_color,
                    bordercolor=vote_text_color,
                    borderwidth=2
                )
                
                # Add model name below the vote
                fig.add_annotation(
                    text=f"{model_name}: '{predicted_answer if len(predicted_answer) < 100 else predicted_answer[:100] + '.....'}'",
                    xref="paper", yref="paper",
                    x=vote_x_start + vote_idx * vote_spacing,
                    y=0.09 + idx * y_off,
                    xanchor="center", yanchor="top",
                    showarrow=False,
                    font=dict(size=36, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
    
    # Update layout for minimal appearance
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, width]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[0, height],
            scaleanchor="x",
            scaleratio=1
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    # Save the figure
    fig.write_image(save_path)
    
    return fig

class FailureAnalyzer:
    """Enhanced failure analysis for VQA models with cross-model comparison."""
    
    def __init__(self, output_dir: str = "./failure_analysis_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.topk = 25
        
    def analyze_cross_model_failures(
        self,
        model_results: Dict[str, List[Dict]],
        questions_list: List[BaseQuestion],
        gt_answers: List,
        pred_answers_by_model: Dict[str, List],
        min_models_failing: int = 2,
        min_failure_count: int = 3
    ) -> Dict[str, Any]:
        """
        Find failure patterns that are common across multiple models.
        
        Args:
            model_results: Dict mapping model names to their evaluation results
            questions_list: List of question objects
            gt_answers: Ground truth answers
            pred_answers_by_model: Dict mapping model names to their predictions
            min_models_failing: Minimum number of models that must fail on the same pattern
            min_failure_count: Minimum number of times a pattern must occur
            
        Returns:
            Dictionary with cross-model failure analysis
        """
        print("🔍 Analyzing cross-model failure patterns...")
        
        # Track failures by pattern across models
        cross_model_failures = defaultdict(lambda: defaultdict(int))  # pattern -> model -> count
        failure_samples = defaultdict(lambda: defaultdict(list))  # pattern -> model -> samples
        
        # for model_name, results in model_results.items():
        #     for i, result in enumerate(results):
        #         if not result["correct"] and result["valid"]:
        #             gt = result["ground_truth_answer"]
        #             pred = result["predicted_answer"]
        #             question_name = questions_list[i].question_name
        #             pattern = (question_name, gt, pred)
                    
        #             cross_model_failures[pattern][model_name] += 1
        #             failure_samples[pattern][model_name].append(i)

        metric = MultipleChoiceMetric()

        for model_name, pred_answers in pred_answers_by_model.items():
            for idx, (question, gt_answer, pred_answer) in enumerate(zip(questions_list, gt_answers, pred_answers)):

                # if question.question_name != "_prompt_moving_towards_ego" or "pedestrian" in question.question:
                if question.question_name not in ["_prompt_vehicle_future_path", "_prompt_moving_towards_ego"] or "pedestrian" in question.question:
                    continue
                
                result = metric.evaluate(pred_answer, gt_answer, question)
                
                if not result["correct"] and result["valid"]:
                    gt = result["ground_truth_answer"]
                    pred = result["predicted_answer"]
                    question_txt = question.question
                    pattern = (question_txt, gt, pred)
                    
                    # print("pattern", pattern)
                    # print("question", question)
                    # exit()
                    
                    cross_model_failures[pattern][model_name] += 1
                    failure_samples[pattern][model_name].append(idx)
        
        # Filter patterns that occur across multiple models
        common_patterns = {}
        for pattern, model_counts in cross_model_failures.items():
            models_failing = len(model_counts)
            total_failures = sum(model_counts.values())
            
            if models_failing >= min_models_failing and total_failures >= min_failure_count:
                common_patterns[pattern] = {
                    'models_failing': models_failing,
                    'total_failures': total_failures,
                    'model_counts': dict(model_counts),
                    'sample_indices': dict(failure_samples[pattern])
                }
        
        # Sort by impact (models failing * total failures)
        sorted_patterns = sorted(
            common_patterns.items(),
            key=lambda x: x[1]['models_failing'] * x[1]['total_failures'],
            reverse=True
        )
        
        print(f"Found {len(sorted_patterns)} cross-model failure patterns")
        for pattern, info in sorted_patterns[:5]:
            question_name, gt, pred = pattern
            print(f"  {question_name}: {gt} → {pred}: {info['models_failing']} models, {info['total_failures']} total failures")
        
        return {
            'patterns': dict(sorted_patterns),
            'summary': self._create_cross_model_summary(sorted_patterns, model_results.keys())
        }
    
    def select_best_confusion_matrices(
        self,
        model_results: Dict[str, List[Dict]],
        questions_list: List[BaseQuestion],
        top_k: int = 3,
        criteria: str = 'diversity'  # 'diversity', 'performance', 'error_rate'
    ) -> List[Tuple[str, Dict]]:
        """
        Select the most informative confusion matrices to display.
        
        Args:
            model_results: Dict mapping model names to their evaluation results
            questions_list: List of question objects
            top_k: Number of matrices to select
            criteria: Selection criteria
            
        Returns:
            List of (model_name, confusion_data) tuples
        """
        print(f"🎯 Selecting top {top_k} confusion matrices based on {criteria}...")
        
        model_stats = {}
        
        for model_name, results in model_results.items():
            # Group by question type
            question_type_results = defaultdict(list)
            for i, result in enumerate(results):
                q_type = questions_list[i].question_name
                
                question_type_results[q_type].append(result)
                
            
            # Calculate metrics for each question type
            type_metrics = {}
            for q_type, type_results in question_type_results.items():
                confusion_data = self._build_confusion_matrix(type_results)
                metrics = self._calculate_matrix_metrics(confusion_data)
                type_metrics[q_type] = {
                    'confusion_data': confusion_data,
                    'metrics': metrics
                }
            
            model_stats[model_name] = type_metrics
        
        # Select based on criteria
        if criteria == 'diversity':
            selected = self._select_by_diversity(model_stats, top_k)
        elif criteria == 'performance':
            selected = self._select_by_performance(model_stats, top_k)
        elif criteria == 'error_rate':
            selected = self._select_by_error_patterns(model_stats, top_k)
        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        return selected
    
    def create_model_comparison_plot(
        self,
        model_results: Dict[str, List[Dict]],
        questions_list: List[BaseQuestion],
        metric: str = 'accuracy'
    ) -> go.Figure:
        """Create a comparison plot across models and question types."""
        
        # Calculate metrics by model and question type
        comparison_data = []
        
        for model_name, results in model_results.items():
            question_type_results = defaultdict(list)
            for i, result in enumerate(results):
                q_type = questions_list[i].question_name
                question_type_results[q_type].append(result)
            
            for q_type, type_results in question_type_results.items():
                if metric == 'accuracy':
                    score = sum(r['correct'] for r in type_results) / len(type_results)
                elif metric == 'error_diversity':
                    # Calculate how diverse the errors are
                    errors = [(r['ground_truth_answer'], r['predicted_answer']) 
                             for r in type_results if not r['correct']]
                    score = len(set(errors)) / len(errors) if errors else 0
                
                comparison_data.append({
                    'Model': model_name,
                    'Question Type': q_type,
                    'Score': score,
                    'Sample Count': len(type_results)
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Create heatmap
        pivot_df = df.pivot(index='Question Type', columns='Model', values='Score')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlBu_r',
            text=np.round(pivot_df.values, 3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='Model: %{x}<br>Question Type: %{y}<br>Score: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Model Performance Comparison ({metric.title()})',
            xaxis_title='Model',
            yaxis_title='Question Type',
            width=800,
            height=600
        )
        
        return fig
    
    def generate_failure_report(
        self,
        cross_model_analysis: Dict,
        model_results: Dict[str, List[Dict]],
        questions_list: List[BaseQuestion],
        gt_answers: List,
        pred_answers_by_model: Dict[str, List],
        max_examples: int = 20
    ) -> str:
        """Generate a comprehensive failure analysis report."""
        
        report_path = self.output_dir / "failure_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# VQA Model Failure Analysis Report\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            patterns = cross_model_analysis['patterns']
            f.write(f"- **Total cross-model failure patterns identified**: {len(patterns)}\n")
            f.write(f"- **Models analyzed**: {list(model_results.keys())}\n")
            f.write(f"- **Total questions analyzed**: {len(questions_list)}\n\n")
            
            # Top failure patterns
            f.write("## Most Common Cross-Model Failures\n\n")
            for i, (pattern, info) in enumerate(list(patterns.items())):
                question_name, gt, pred = pattern
                f.write(f"### {i+1}. {question_name}: {gt} → {pred}\n")
                f.write(f"- **Models affected**: {info['models_failing']}/{len(model_results)}\n")
                f.write(f"- **Total failures**: {info['total_failures']}\n")
                f.write(f"- **Per-model breakdown**: {info['model_counts']}\n\n")
                
                # Add specific examples
                f.write("**Example failures:**\n")
                for model_suffix, sample_indices in list(info['sample_indices'].items())[:2]:
                    for idx in sample_indices[:max_examples]:
                        # question = questions_list[idx]
                        pred_answer = pred_answers_by_model[model_suffix][idx]
                        f.write(f"- *{MODEL_SUFFIX_MAP[model_suffix]}*: {pred_answer.get_answer_text()}\n")
                f.write("\n")
                
                sanitized_question_name = remove_punctuation(question_name.lower().replace(" ", "_"))
                failure_path = f"./comprehensive_failure_analysis/failure_examples_{sanitized_question_name}"
                os.makedirs(failure_path, exist_ok=True)
                
                
                for model_suffix, sample_indices in list(info['sample_indices'].items())[:2]:
                    # some model got it wrong
                    for idx in sample_indices[:max_examples]:
                        model_predictions = {MODEL_SUFFIX_MAP[model_suffix]: answers[idx].get_answer_text() for model_suffix, answers in pred_answers_by_model.items()}
                        question_obj = questions_list[idx]
                        answer_obj = gt_answers[idx]
                        visualise_sample_plotly(
                            question_obj,
                            answer_obj,
                            str(Path(failure_path) / f"question_{question_obj.question_id}.jpg"),
                            model_predictions
                        )
                
                # pattern_dir = visualize_failure_samples(
                #     (gt, pred, info['total_failures']),
                #     questions_list,
                #     gt_answers,
                #     pred_answers_by_model,
                #     failure_path,
                #     10,
                # )
            
            # Model-specific insights
            f.write("## Model-Specific Insights\n\n")
            for model_name, results in model_results.items():
                accuracy = sum(r['correct'] for r in results) / len(results)
                f.write(f"### {model_name}\n")
                f.write(f"- **Overall accuracy**: {accuracy:.3f}\n")
                
                # Top individual failures
                individual_failures = Counter()
                for result in results:
                    if not result['correct'] and result['valid']:
                        pattern = (result['ground_truth_answer'], result['predicted_answer'])
                        individual_failures[pattern] += 1
                
                f.write("- **Top individual failure patterns**:\n")
                for pattern, count in individual_failures.most_common(3):
                    gt, pred = pattern
                    f.write(f"  - {gt} → {pred}: {count} times\n")
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the failure analysis:\n\n")
            f.write("1. **Focus on cross-model failures** - These represent systematic challenges\n")
            f.write("2. **Prioritize high-impact patterns** - Address failures affecting multiple models\n")
            f.write("3. **Investigate question types** - Some question types may be inherently more challenging\n")
            f.write("4. **Consider data augmentation** - For patterns with consistent failures across models\n\n")
        
        print(f"📊 Failure analysis report saved to: {report_path}")
        return str(report_path)
    
    def _build_confusion_matrix(self, results: List[Dict]) -> Dict[str, Any]:
        """Build confusion matrix data from results."""
        matrix_dict = defaultdict(lambda: defaultdict(int))
        labels = set()
        
        for result in results:
            if result['valid']:
                gt = result['ground_truth_answer']
                pred = result['predicted_answer']
                matrix_dict[gt][pred] += 1
                labels.add(gt)
                labels.add(pred)
        
        return {
            'matrix': dict(matrix_dict),
            'labels': sorted(list(labels))
        }
    
    def _calculate_matrix_metrics(self, confusion_data: Dict) -> Dict[str, float]:
        """Calculate metrics for a confusion matrix."""
        matrix_dict = confusion_data['matrix']
        labels = confusion_data['labels']
        
        # Convert to numpy array
        n_classes = len(labels)
        matrix = np.zeros((n_classes, n_classes))
        
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                matrix[i, j] = matrix_dict.get(true_label, {}).get(pred_label, 0)
        
        # Calculate metrics
        total = matrix.sum()
        accuracy = np.trace(matrix) / total if total > 0 else 0
        
        # Error diversity (number of unique error patterns)
        error_patterns = 0
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and matrix[i, j] > 0:
                    error_patterns += 1
        
        error_diversity = error_patterns / (n_classes * (n_classes - 1)) if n_classes > 1 else 0
        
        return {
            'accuracy': accuracy,
            'error_diversity': error_diversity,
            'total_samples': int(total),
            'num_classes': n_classes
        }
    
    def _name_matrix(self, model_suffix, q_type):
        model_name = MODEL_SUFFIX_MAP[model_suffix]
        hierarchy_name, question_name = QUESTION_TYPE_MAPPINGS[q_type]
        
        return f"{model_name}: {question_name}"
    
    def _select_by_diversity(self, model_stats: Dict, top_k: int) -> List[Tuple[str, Dict]]:
        """Select matrices with highest error diversity."""
        candidates = []
        
        for model_suffix, type_metrics in model_stats.items():
            for q_type, data in type_metrics.items():
                diversity = data['metrics']['error_diversity']
                total_samples = data['metrics']['total_samples']
                candidates.append((
                    self._name_matrix(model_suffix, q_type),
                    data['confusion_data'],
                    diversity * np.log(total_samples + 1)  # Weight by sample size
                ))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(name, data) for name, data, _ in candidates[:top_k]]
    
    def _select_by_performance(self, model_stats: Dict, top_k: int) -> List[Tuple[str, Dict]]:
        """Select matrices showing interesting performance patterns."""
        candidates = []
        
        for model_suffix, type_metrics in model_stats.items():
            for q_type, data in type_metrics.items():
                accuracy = data['metrics']['accuracy']
                # Select moderate performers (not too high or too low)
                interestingness = 1 - abs(accuracy - 0.5) * 2  # Peak at 0.5 accuracy
                candidates.append((
                    self._name_matrix(model_suffix, q_type),
                    data['confusion_data'],
                    interestingness
                ))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(name, data) for name, data, _ in candidates[:top_k]]
    
    def _select_by_error_patterns(self, model_stats: Dict, top_k: int) -> List[Tuple[str, Dict]]:
        """Select matrices with most interesting error patterns."""
        candidates = []
        
        for model_suffix, type_metrics in model_stats.items():
            for q_type, data in type_metrics.items():
                metrics = data['metrics']
                # Combine low accuracy with high diversity
                score = (1 - metrics['accuracy']) * metrics['error_diversity']
                candidates.append((
                    self._name_matrix(model_suffix, q_type),
                    data['confusion_data'],
                    score
                ))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(name, data) for name, data, _ in candidates[:top_k]]
    
    def _create_cross_model_summary(self, sorted_patterns: List, model_names: List) -> Dict:
        """Create summary statistics for cross-model analysis."""
        total_patterns = len(sorted_patterns)
        patterns_by_model_count = defaultdict(int)
        
        for pattern, info in sorted_patterns:
            models_failing = info['models_failing']
            patterns_by_model_count[models_failing] += 1
        
        return {
            'total_patterns': total_patterns,
            'patterns_by_model_count': dict(patterns_by_model_count),
            'models_analyzed': list(model_names),
            'average_models_per_pattern': np.mean([info['models_failing'] for _, info in sorted_patterns]) if sorted_patterns else 0
        }


# Keep the original confusion matrix function with minor improvements
def create_confusion_matrix_plotly(
    confusion_data: Dict[str, Any],
    question_name: str,
    title: Optional[str] = None,
    show_percentages: bool = True,
    normalize: Optional[str] = None,
    colorscale: str = "Blues",
    width: int = 600,
    height: int = 500,
) -> go.Figure:
    """Enhanced confusion matrix with better formatting and metrics."""
    
    matrix_dict = confusion_data["matrix"]
    labels = confusion_data["labels"]

    # Convert dictionary matrix to numpy array
    n_classes = len(labels)
    matrix = np.zeros((n_classes, n_classes))

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = matrix_dict.get(true_label, {}).get(pred_label, 0)

    # Store original matrix for hover text
    original_matrix = matrix.copy()

    # Normalize if requested
    normalized_matrix = matrix.copy()
    if normalize == "true":
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized_matrix = matrix / row_sums
    elif normalize == "pred":
        col_sums = matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        normalized_matrix = matrix / col_sums
    elif normalize == "all":
        total = matrix.sum()
        normalized_matrix = matrix / total if total > 0 else matrix

    # Create hover text with enhanced information
    hover_text = []
    for i in range(n_classes):
        hover_row = []
        for j in range(n_classes):
            count = int(original_matrix[i, j])
            
            if i == j:  # Diagonal (correct predictions)
                status = "✓ Correct"
            else:  # Off-diagonal (errors)
                status = "✗ Error"
            
            if normalize:
                percentage = normalized_matrix[i, j] * 100
                hover_info = (
                    f"{status}<br>"
                    f"True: {labels[i]}<br>"
                    f"Predicted: {labels[j]}<br>"
                    f"Count: {count}<br>"
                    f"Percentage: {percentage:.1f}%"
                )
            else:
                hover_info = (
                    f"{status}<br>"
                    f"True: {labels[i]}<br>"
                    f"Predicted: {labels[j]}<br>"
                    f"Count: {count}"
                )
            hover_row.append(hover_info)
        hover_text.append(hover_row)

    # Create annotations for cell text
    annotations = []
    max_classes_for_annotations = 12  # Increased threshold

    if n_classes <= max_classes_for_annotations:
        for i in range(n_classes):
            for j in range(n_classes):
                count = int(original_matrix[i, j])
                
                # Determine text color based on background
                bg_intensity = normalized_matrix[i, j]
                text_color = "white" if bg_intensity > 0.5 else "black"
                
                if n_classes <= 8:
                    if normalize and show_percentages:
                        percentage = normalized_matrix[i, j] * 100
                        text = f"{count}<br>({percentage:.1f}%)"
                    else:
                        text = str(count)
                else:
                    text = str(count) if count > 0 else ""

                if text:  # Only add annotation if there's text to show
                    annotations.append(
                        dict(
                            x=j, y=i, text=text, showarrow=False,
                            font=dict(
                                color=text_color,
                                size=10 if n_classes > 8 else 12,
                                family="Arial"
                            ),
                            xref="x", yref="y"
                        )
                    )

    # Create the heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=normalized_matrix,
            x=labels,
            y=labels,
            colorscale=colorscale,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_text,
            showscale=True,
            colorbar=dict(
                title="Normalized Value" if normalize else "Count",
            ),
        )
    )

    # Add annotations
    fig.update_layout(annotations=annotations)

    # Enhanced metrics calculation
    correct_predictions = np.trace(original_matrix)
    total_predictions = original_matrix.sum()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Calculate per-class metrics
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(n_classes):
        tp = original_matrix[i, i]
        fp = original_matrix[:, i].sum() - tp
        fn = original_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = np.mean(precisions) * 100
    macro_recall = np.mean(recalls) * 100
    macro_f1 = np.mean(f1_scores) * 100
    accuracy = accuracy *100

    # Set title
    if title is None:
        title = f"Confusion Matrix: {question_name}"
        if normalize:
            title += f" (Normalized by {normalize})"

    subtitle = (
        f"Acc: {accuracy:.3f}% | "
        f"Prec: {macro_precision:.3f}% | "
        f"Rec: {macro_recall:.3f}% | "
        f"F1: {macro_f1:.3f}% | "
        f"N: {int(total_predictions)}"
    )

    # Update layout with better formatting
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{subtitle}</sub>", 
            x=0.5, 
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Predicted Class", 
            tickangle=45, 
            side="bottom",
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="True Class",
            autorange="reversed",
            tickfont=dict(size=16)
        ),
        width=width,
        height=height,
        font=dict(size=16),
        margin=dict(l=80, r=80, t=80, b=80),
    )

    return fig



def remove_punctuation(text: str):
    """Remove punctuation"""
    return text.translate(str.maketrans("", "", string.punctuation))

# Example usage function
def run_comprehensive_failure_analysis(
    model_results: Dict[str, List[Dict]],
    questions_list: List[BaseQuestion],
    gt_answers: List,
    pred_answers_by_model: Dict[str, List],
    output_dir: str = "./comprehensive_failure_analysis"
) -> Dict[str, Any]:
    """
    Run the complete failure analysis pipeline.
    
    Returns:
        Dictionary with all analysis results and file paths
    """
    
    analyzer = FailureAnalyzer(output_dir)
    
    # 1. Cross-model failure analysis
    print("Step 1: Cross-model failure analysis...")
    cross_model_analysis = analyzer.analyze_cross_model_failures(
        model_results, questions_list, gt_answers, pred_answers_by_model
    )
    
    # 2. Select best confusion matrices to show
    print("Step 2: Selecting representative confusion matrices...")
    selected_matrices = analyzer.select_best_confusion_matrices(
        model_results, questions_list, top_k=25, criteria='diversity'
    )
    selected_matrices2 = analyzer.select_best_confusion_matrices(
        model_results, questions_list, top_k=25, criteria='error_rate'
    )
    selected_matrices3 = analyzer.select_best_confusion_matrices(
        model_results, questions_list, top_k=25, criteria='performance'
    )
    
    selected_matrices = selected_matrices + selected_matrices2 + selected_matrices3
    
    # 3. Create model comparison plot
    print("Step 3: Creating model comparison visualization...")
    comparison_fig = analyzer.create_model_comparison_plot(
        model_results, questions_list, metric='accuracy'
    )
    comparison_fig.write_html(str(analyzer.output_dir / "model_comparison.html"))
    
    # 4. Generate comprehensive report
    print("Step 4: Generating failure analysis report...")
    report_path = analyzer.generate_failure_report(
        cross_model_analysis, model_results, questions_list, 
        gt_answers, pred_answers_by_model, max_examples=5
    )
    
    # 5. Create selected confusion matrices
    print("Step 5: Creating selected confusion matrices...")
    matrix_files = []
    for name, confusion_data in selected_matrices:
        fig = create_confusion_matrix_plotly(
            confusion_data, name, 
            title=name,
            show_percentages=True
        )
        name_sanitized = remove_punctuation(name.lower().replace(" ", "_"))
        file_path = analyzer.output_dir / f"confusion_matrix_{name_sanitized}.html"
        # fig.write_html(str(file_path))
        fig.write_image(str(file_path.with_suffix(".jpg")))
        matrix_files.append(str(file_path))
    
    return {
        'cross_model_analysis': cross_model_analysis,
        'selected_matrices': selected_matrices,
        'report_path': report_path,
        'comparison_plot': str(analyzer.output_dir / "model_comparison.html"),
        'matrix_files': matrix_files,
        'output_directory': str(analyzer.output_dir)
    }
    
def get_ground_truth(generated_samples_path: Path, save_prefix: str):
    dataset = VQADataset(tag=f"{save_prefix}_all")
    seen_questions = set()

    # Get all prompt generators (you might need to adjust this based on your structure)
    all_prompt_generators = [
        "ObjectBinaryPromptGenerator",
        "ObjectDrawnBoxPromptGenerator",
        "EgoRelativeObjectTrajectoryPromptGenerator",
    ]

    # Process all prompt generator files
    for prompt_generator in all_prompt_generators:
        path = (
            generated_samples_path / f"{save_prefix}_{prompt_generator}_validation.json"
        )

        if not path.exists():
            raise FileNotFoundError(f"path={path} should exist")

        ds1 = VQADataset.load_dataset(str(path))

        for sample in ds1.samples:
            if sample.question.question_id not in seen_questions:
                dataset.add_sample(
                    sample.question, sample.answer
                )

    return dataset

def get_model_predictions(model_output_path: Path, save_prefix: str, model_suffix: str, gt_dataset: VQADataset):
    # Get all prompt generators (you might need to adjust this based on your structure)
    all_prompt_generators = [
        "ObjectBinaryPromptGenerator",
        "ObjectDrawnBoxPromptGenerator",
        "EgoRelativeObjectTrajectoryPromptGenerator",
    ]

    gt_question_mapping = {x.question.question_id: idx for idx, x in enumerate(gt_dataset.samples)}

    # Create prediction dataset by collecting from all relevant prediction files
    pred_dataset = VQADataset(tag=f"{save_prefix}_pred")
    pred_sample_queue = [None for _ in range(len(gt_dataset.samples))]


    # Process all prediction files to find samples that belong to this category
    for prompt_generator in all_prompt_generators:
        pred_path = (
            model_output_path
            / f"{save_prefix}_{prompt_generator}_validation_{model_suffix}.json"
        )

        if pred_path.exists():
            print(f"    Loading predictions from {pred_path.name}")
            pred_ds = VQADataset.load_dataset(str(pred_path))

            for sample in pred_ds.samples:
                question_name = sample.question.question_name

                # Check if this question belongs to the current category
                if sample.question.question_id in gt_question_mapping:
                    sample_idx = gt_question_mapping[sample.question.question_id]
                    pred_sample_queue[sample_idx] = sample
    # reorder preds according to gt
    for sample in pred_sample_queue:
        assert sample is not None
        pred_dataset.add_sample(sample.question, sample.answer)
        
    return pred_dataset
    
def main():
    generated_samples_path = Path(
        "/media/local-data/uqdetche/waymo_vqa_dataset/generated_vqa_samples/"
    )
    model_output_path = Path(
        "/media/local-data/uqdetche/waymo_vqa_dataset/model_outputs"
    )
    
    model_suffix_map = {
        "llava-v1.5-7b_raw_wchoices": "LLaVA",
        "qwen-vlwchoices": "Qwen-VL",
        "llava-v1.5-7b-lorafinetuned_wchoices": "LLaVA†",
        "single_frame_results_0528__senna_pred": "Senna",
        "single_frame_results_0528__FT_senna_pred": "Senna†",
    }
    
    save_prefix = "26_05_2025_export"
    gt_dataset = get_ground_truth(generated_samples_path, save_prefix)
    
    questions_list = []
    gt_answers = []
    
    for sample in gt_dataset.samples:
        questions_list.append(sample.question)
        gt_answers.append(sample.answer)

    pred_answers_by_model = {}
    model_results = {}
    model_datasets = {}

    for model_suffix, model_name in model_suffix_map.items():
        
        model_datasets[model_suffix] = get_model_predictions(model_output_path, save_prefix, model_suffix, gt_dataset)
        
        metric = MultipleChoiceMetric()
        
        model_results[model_suffix] = metric.evaluate_dataset(model_datasets[model_suffix], gt_dataset)

        pred_answers_by_model[model_suffix] = [x.answer for x in model_datasets[model_suffix].samples]
    
    
    # Run the complete analysis
    results = run_comprehensive_failure_analysis(
        model_results=model_results,
        questions_list=questions_list,
        gt_answers=gt_answers,
        pred_answers_by_model=pred_answers_by_model
    )
    
    pprint(results)
    
if __name__ == "__main__":
    main()