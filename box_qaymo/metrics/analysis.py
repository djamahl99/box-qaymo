from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import importlib
import os
from collections import defaultdict

from box_qaymo.data.vqa_dataset import VQASample
from box_qaymo.questions.base import BaseQuestion
from box_qaymo.waymo_loader import WaymoDatasetLoader


def create_confusion_matrix_plotly(
    confusion_data: Dict[str, Any],
    question_name: str,
    title: Optional[str] = None,
    show_percentages: bool = True,
    normalize: Optional[str] = None,  # None, 'true', 'pred', or 'all'
    colorscale: str = "Blues",
    width: int = 600,
    height: int = 500,
) -> go.Figure:
    """
    Create an interactive confusion matrix visualization using Plotly.

    Args:
        confusion_data: Dictionary with 'matrix' and 'labels' keys
                       matrix format: {'true_class': {'pred_class': count, ...}, ...}
        question_name: Name of the question type for labeling
        title: Custom title for the plot
        show_percentages: Whether to show percentages alongside counts
        normalize: How to normalize the matrix:
                  - None: Raw counts
                  - 'true': Normalize by true class (rows sum to 1)
                  - 'pred': Normalize by predicted class (columns sum to 1)
                  - 'all': Normalize by total samples
        colorscale: Plotly colorscale name
        width: Figure width in pixels
        height: Figure height in pixels

    Returns:
        Plotly Figure object
    """

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
        # Normalize by row (true class)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        normalized_matrix = matrix / row_sums
    elif normalize == "pred":
        # Normalize by column (predicted class)
        col_sums = matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        normalized_matrix = matrix / col_sums
    elif normalize == "all":
        # Normalize by total
        total = matrix.sum()
        normalized_matrix = matrix / total if total > 0 else matrix

    # Create hover text
    hover_text = []
    for i in range(n_classes):
        hover_row = []
        for j in range(n_classes):
            count = int(original_matrix[i, j])

            if normalize:
                percentage = normalized_matrix[i, j] * 100
                hover_info = (
                    f"True: {labels[i]}<br>"
                    f"Predicted: {labels[j]}<br>"
                    f"Count: {count}<br>"
                    f"Percentage: {percentage:.1f}%"
                )
            else:
                hover_info = (
                    f"True: {labels[i]}<br>"
                    f"Predicted: {labels[j]}<br>"
                    f"Count: {count}"
                )
            hover_row.append(hover_info)
        hover_text.append(hover_row)

    # Create annotations for cell text (only if not too many classes)
    annotations = []
    max_classes_for_annotations = 10  # Threshold for showing text in cells

    if n_classes <= max_classes_for_annotations:
        for i in range(n_classes):
            for j in range(n_classes):
                count = int(original_matrix[i, j])

                # Simplify text display for larger matrices
                if n_classes <= 5:
                    # Show full details for small matrices
                    if normalize and show_percentages:
                        percentage = normalized_matrix[i, j] * 100
                        text = f"{count}<br>({percentage:.1f}%)"
                    else:
                        text = str(count)
                else:
                    # Show only counts for medium-sized matrices (6-10 classes)
                    text = str(count)

                # Use black text with white outline for maximum readability
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=text,
                        showarrow=False,
                        font=dict(
                            color="black",
                            size=(
                                12 if n_classes > 5 else 14
                            ),  # Smaller text for more classes
                            family="Arial Black",  # Bold font for better readability
                        ),
                        # Add white border/stroke effect
                        bordercolor="white",
                        borderwidth=2,
                        bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent white background
                        borderpad=(
                            3 if n_classes > 5 else 4
                        ),  # Less padding for more classes
                        xref="x",
                        yref="y",
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
                # titleside="right"
            ),
        )
    )

    # Add annotations
    fig.update_layout(annotations=annotations)

    # Calculate overall metrics for subtitle
    correct_predictions = np.trace(original_matrix)
    total_predictions = original_matrix.sum()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Calculate macro-averaged precision and recall
    precisions = []
    recalls = []

    for i in range(n_classes):
        # For class i
        tp = original_matrix[i, i]
        fp = original_matrix[:, i].sum() - tp  # Column sum minus TP
        fn = original_matrix[i, :].sum() - tp  # Row sum minus TP

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)

    # Set title
    if title is None:
        title = f"Confusion Matrix: {question_name}"
        if normalize:
            title += f" (Normalized by {normalize})"

    subtitle = (
        f"Accuracy: {accuracy:.3f} | "
        f"Precision: {macro_precision:.3f} | "
        f"Recall: {macro_recall:.3f} | "
        f"Samples: {int(total_predictions)}"
    )

    # Update layout
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>{subtitle}</sub>", x=0.5, font=dict(size=16)),
        xaxis=dict(title="Predicted Class", tickangle=45, side="bottom"),
        yaxis=dict(
            title="True Class",
            autorange="reversed",  # This puts the first class at the top
        ),
        width=width,
        height=height,
        font=dict(size=10),
        margin=dict(l=100, r=100, t=100, b=100),
    )

    return fig


def find_common_failures(
    metric_results: List[Dict], min_failure_count: int = 3, top_k_failures: int = 2
) -> List[Tuple[str, str, int]]:
    """
    Find common failure patterns in the confusion matrix.

    Args:
        metric_results: List of evaluation results from MultipleChoiceMetric
        min_failure_count: Minimum number of failures to be considered "common"
        top_k_failures: Number of top failure patterns to return

    Returns:
        List of tuples: (ground_truth, predicted, count, sample_results)
        where sample_results contains the actual failed samples
    """

    # Group failures by (ground_truth, predicted) pattern
    failure_patterns = defaultdict(int)

    for result in metric_results:
        if (
            not result["correct"] and result["valid"]
        ):  # Only valid but incorrect predictions
            gt = result["ground_truth_answer"]
            pred = result["predicted_answer"]
            if gt != pred:  # Sanity check
                failure_patterns[(gt, pred)] += 1

    # Filter by minimum count and sort by frequency
    common_failures = [
        (gt, pred, count)
        for (gt, pred), count in failure_patterns.items()
        if count >= min_failure_count
    ]

    # Sort by failure count (descending) and take top k
    common_failures.sort(key=lambda x: x[2], reverse=True)

    return common_failures[:top_k_failures]


def visualize_failure_samples(
    failure_pattern: Tuple[str, str, int],
    questions_list,
    gt_answers,
    pred_answers,
    output_dir: str = "./failure_analysis",
    max_samples: int = 5,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = 150,
) -> str:
    """
    Visualize samples for a specific failure pattern.

    Args:
        failure_pattern: Tuple from find_common_failures (gt, pred, count)
        questions_data: List of original question objects/dicts
        answers_data: List of original answer objects/dicts
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize
        figsize: Figure size for visualizations
        dpi: DPI for saved images

    Returns:
        Path to the created visualization directory
    """

    ground_truth, predicted, count = failure_pattern

    # Create output directory
    pattern_dir = os.path.join(output_dir, f"GT_{ground_truth}_PRED_{predicted}")
    os.makedirs(pattern_dir, exist_ok=True)
    
    # check if we have enough samples
    samples = list(Path(pattern_dir).rglob("*.png"))
    
    if len(samples) >= max_samples:
        return pattern_dir

    print(
        f"Analyzing failure pattern: GT='{ground_truth}' → PRED='{predicted}' ({count} samples)"
    )

    # Take a sample of failures to visualize
    loader = WaymoDatasetLoader(
        Path("/media/local-data/uqdetche/waymo_vqa_dataset/")
    )  # TODO constant

    sampled = 0
    
    # import here to avoid circular import
    from box_qaymo.metrics.multiple_choice import MultipleChoiceMetric
    from box_qaymo.answers.multiple_choice import MultipleChoiceAnswer
    from box_qaymo.answers.raw_text import RawTextAnswer

    for question_obj, gt_answer, pred_answer in zip(
        questions_list, gt_answers, pred_answers
    ):
        if (
            gt_answer.answer != ground_truth
            or (
                isinstance(pred_answer, MultipleChoiceAnswer)
                and pred_answer.answer != predicted
            )
            or (
                isinstance(pred_answer, RawTextAnswer)
                and max(
                    gt_answer.choices,
                    key=lambda x: MultipleChoiceMetric.compute_text_similarity(
                        x.lower(), pred_answer.text
                    ),
                )
                != predicted
            )
        ):
            continue

        # Get generator from question object
        generator_name = question_obj.generator_name
        print("generator_name", generator_name)
        if generator_name is None:
            print(f"Warning: No generator_name found for sample {question_obj}")
            continue

        # Dynamically import and instantiate generator
        try:
            # Assuming generators are in a module like 'generators' or similar
            # You might need to adjust the import path
            # module_path = f"generators.{generator_name}"  # Adjust path as needed
            module_path_split = generator_name.split(".")
            module_path, class_name = (
                ".".join(module_path_split[:-1]),
                module_path_split[-1],
            )
            print("module_path, class_name", module_path, class_name)
            generator_module = importlib.import_module(module_path)
            generator_class = getattr(generator_module, class_name)

            generator = generator_class()
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import generator {generator_name}: {e}")
            continue

        # Get frames for this sample
        frames = []

        if "EgoRelativeObjectTrajectoryPromptGenerator" in generator_name:
            # continue # don't want to vis these aha
            print("loading frames for EgoRelativeObjectTrajectoryPromptGenerator")
            timestamps = loader.get_frame_timestamps(question_obj.scene_id)

            for timestamp in timestamps:
                frames.append(loader.load_frame(question_obj.scene_id, timestamp))

        # Create save path
        save_path = os.path.join(pattern_dir, f"sample_{question_obj.question_id}.png")

        # Call generator.visualise_sample
        generator.visualise_sample(
            question_obj=question_obj,
            answer_obj=gt_answer,
            pred_answer_obj=pred_answer,
            save_path=save_path,
            frames=frames,
            figsize=figsize,
            text_fontsize=12,
            title_fontsize=14,
            dpi=dpi,
        )

        print(f"  Saved visualization: {save_path}")

        if sampled > max_samples:
            return pattern_dir

        sampled += 1

    return pattern_dir


def analyze_all_common_failures(
    metric_results: List[Dict],
    questions_list: List[BaseQuestion],
    gt_answers: List,
    pred_answers: List,
    output_dir: str = "./failure_analysis",
    min_failure_count: int = 3,
    top_k_failures: int = 5,
    max_samples_per_pattern: int = 5,
) -> Dict[str, str]:
    """
    Complete failure analysis pipeline.

    Args:
        metric_results: Results from MultipleChoiceMetric evaluation
        questions_data: Original question objects/dicts
        answers_data: Original answer objects/dicts
        frames_data: Dictionary mapping sample IDs to frame data
        output_dir: Base directory for all failure analysis
        min_failure_count: Minimum failures to be considered common
        top_k_failures: Number of top failure patterns to analyze
        max_samples_per_pattern: Max samples to visualize per pattern

    Returns:
        Dictionary mapping failure patterns to their output directories
    """

    # Find common failures
    common_failures = find_common_failures(
        metric_results,
        min_failure_count=min_failure_count,
        top_k_failures=top_k_failures,
    )

    if not common_failures:
        print("No common failure patterns found!")
        return {}

    print(f"Found {len(common_failures)} common failure patterns:")
    for gt, pred, count in common_failures:
        print(f"  {gt} → {pred}: {count} failures")

    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)

    # Visualize each failure pattern
    pattern_dirs = {}
    for failure_pattern in common_failures:
        gt, pred, count = failure_pattern
        pattern_key = f"{gt}_to_{pred}"

        pattern_dir = visualize_failure_samples(
            failure_pattern,
            questions_list,
            gt_answers,
            pred_answers,
            output_dir,
            max_samples_per_pattern,
        )
        pattern_dirs[pattern_key] = pattern_dir

    # Create overall summary
    summary_path = os.path.join(output_dir, "overall_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Common Failure Patterns Analysis\n")
        f.write("================================\n\n")
        f.write(f"Total patterns analyzed: {len(pattern_dirs)}\n")
        f.write(f"Minimum failure threshold: {min_failure_count}\n\n")

        f.write("Failure Patterns (by frequency):\n")
        for gt, pred, count in common_failures:
            f.write(f"  {gt} → {pred}: {count} failures\n")

        f.write(f"\nVisualization directories:\n")
        for pattern_key, pattern_dir in pattern_dirs.items():
            f.write(f"  {pattern_key}: {pattern_dir}\n")

    print(f"\nFailure analysis complete! Summary saved to: {summary_path}")
    return pattern_dirs
