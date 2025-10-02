import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import random

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

pio.kaleido.scope.mathjax = None

from failure_analysis import run_comprehensive_failure_analysis
from box_qaymo.data.vqa_dataset import VQADataset
from box_qaymo.metrics.analysis import (
    analyze_all_common_failures,
    create_confusion_matrix_plotly,
)
from box_qaymo.metrics.multiple_choice import MultipleChoiceMetric

CATEGORY_TO_PROMPT_GENERATOR = {
    "binary": ["ObjectBinaryPromptGenerator"],
    "attribute": ["ObjectDrawnBoxPromptGenerator"],
    "motion": ["EgoRelativeObjectTrajectoryPromptGenerator"],
    "overall": [
        "ObjectBinaryPromptGenerator",
        "ObjectDrawnBoxPromptGenerator",
        "EgoRelativeObjectTrajectoryPromptGenerator",
    ],
}


def format_value(value, best_val, second_best_val, is_percentage=True):
    """Format a value with appropriate highlighting"""
    if value is None or pd.isna(value):
        return "N/A"

    formatted = f"{value:.2f}" if is_percentage else str(value)

    if value == best_val:
        return f"\\textbf{{{formatted}}}"  # bold the best
    elif value == second_best_val:
        return f"\\underline{{{formatted}}}"  # underline the second best
    else:
        return formatted


def create_category_datasets(generated_samples_path: Path, save_prefix: str):
    """Create combined datasets for each category from all save_prefixes using question_type_mappings"""

    metric = MultipleChoiceMetric()
    question_type_mappings = metric.question_type_mappings

    del metric

    # Get all unique categories from question_type_mappings
    categories = set(category for category, _ in question_type_mappings.values())
    category_datasets = {}
    category_datasets_seen_ids = {}

    # Initialize datasets for each category
    for category in categories:
        category_datasets[category] = VQADataset(tag=f"{category}_combined")
        category_datasets_seen_ids[category] = set()

    # Get all prompt generators (you might need to adjust this based on your structure)
    all_prompt_generators = set()
    for prompt_list in CATEGORY_TO_PROMPT_GENERATOR.values():
        all_prompt_generators.update(prompt_list)

    print(f"\nCreating category datasets using question_type_mappings...")

    # Process all prompt generator files
    for prompt_generator in all_prompt_generators:
        path = (
            generated_samples_path / f"{save_prefix}_{prompt_generator}_validation.json"
        )

        if not path.exists():
            continue

        print(f"  Processing {path.name}")
        ds1 = VQADataset.load_dataset(str(path))

        for sample in ds1.samples:
            question_name = sample.question.question_name

            # Map question to category using question_type_mappings
            if question_name in question_type_mappings:
                category, question_type = question_type_mappings[question_name]

                if category in category_datasets:
                    if sample.question.question_id not in category_datasets_seen_ids[category]:
                        category_datasets[category].add_sample(
                            sample.question, sample.answer
                        )
                        category_datasets_seen_ids[category].add(sample.question.question_id)
                else:
                    print(
                        f"    Warning: Unknown category '{category}' for question '{question_name}'"
                    )
            else:
                print(
                    f"    Warning: Question '{question_name}' not found in question_type_mappings"
                )

    # Save and report results
    final_datasets = {}
    for category, ds in category_datasets.items():
        if len(ds.samples) > 0:
            final_datasets[category] = ds
            print(f"  {category}: {len(ds.samples)} samples")

            # Create directory if it doesn't exist
            os.makedirs("category_datasets", exist_ok=True)
            ds.save_dataset(
                f"category_datasets/{save_prefix}_validation_{category}.json"
            )
        else:
            print(f"  Warning: No samples found for {category}")

    # overall
    if "overall" not in final_datasets:
        final_datasets["overall"] = VQADataset(tag=f"{category}_combined")

        ids_seen = set()
        for cat_ds in final_datasets.values():
            for sample in cat_ds.samples:
                if sample.question.question_id not in ids_seen:
                    final_datasets["overall"].add_sample(sample.question, sample.answer)
                    ids_seen.add(sample.question.question_id)
                    
                    
        print(f"Overall dataet has {len(final_datasets['overall'].samples)} samples")

    return final_datasets


def evaluate_model_on_categories(
    model_suffix: str,
    model_name: str,
    category_datasets: dict,
    model_output_path: Path,
    save_prefix: str,
):
    """Evaluate a single model on all categories using question_type_mappings"""

    model_results = {}
    parse_type_data = []

    metric = MultipleChoiceMetric()
    question_type_mappings = metric.question_type_mappings

    del metric

    for category, gt_dataset in category_datasets.items():
        print(f"  Evaluating {model_name} on {category}...")
        
        gt_question_mapping = {x.question.question_id: idx for idx, x in enumerate(gt_dataset.samples)}

        # Create prediction dataset by collecting from all relevant prediction files
        pred_dataset = VQADataset(tag=f"{category}_pred")
        pred_sample_queue = [None for _ in range(len(gt_dataset.samples))]

        # Get all prompt generators (you might need to adjust this based on your structure)
        all_prompt_generators = set()
        for prompt_list in CATEGORY_TO_PROMPT_GENERATOR.values():
            all_prompt_generators.update(prompt_list)

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
                    # if question_name in question_type_mappings:
                    #     question_category, question_type = question_type_mappings[
                    #         question_name
                    #     ]

                    #     if question_category == category or (sample.question.question_id in gt_question_mapping):
                    #         pred_sample_queue[gt]
                            
                    #     else:
                    #         print("incorrect category?", sample.question.question_id, sample.question.question_id in gt_question_mapping)

        # reorder preds according to gt
        for sample in pred_sample_queue:
            assert sample is not None
            pred_dataset.add_sample(sample.question, sample.answer)

        # Check if we have valid predictions for this category
        if (
            len(pred_dataset.samples) > 0
            and len(pred_dataset.samples) == len(gt_dataset.samples)
            and all(
                x.question.question_id == y.question.question_id
                for x, y in zip(pred_dataset.samples, gt_dataset.samples)
            )
        ):
            # Evaluate this category
            metric = MultipleChoiceMetric()
            metric_results = metric.evaluate_dataset(pred_dataset, gt_dataset)

            # Process invalid pairs if they exist
            if hasattr(metric, "invalid_pairs") and metric.invalid_pairs:
                invalid_counts = np.array(
                    [count for count in metric.invalid_pairs.values()], dtype=int
                )
                top5 = np.argsort(invalid_counts)[::-1]

                out = f"Invalid responses for {model_name} on {category} questions\n"
                out += f"Top invalid responses for {model_name}\n"

                total = sum(v for v in metric.invalid_pairs.values())
                items = list(metric.invalid_pairs.items())

                for idx in top5:
                    (question, response), count = items[idx]
                    out += f"Question: {question}\n Response: {response}\n     {count} occurences - {((count/total)*100):.2f}%\n"

                # Create output directory if it doesn't exist
                os.makedirs("figures/invalid_pairs", exist_ok=True)
                with open(
                    f"figures/invalid_pairs/{category}_{model_suffix}.txt", "w"
                ) as f:
                    f.write(out)

            # Process parse type counts if they exist
            if hasattr(metric, "parse_type_counts") and metric.parse_type_counts:
                for parse_type, count in metric.parse_type_counts.items():
                    parse_type_data.append(
                        {
                            "parse_type": parse_type,
                            "count": count,
                            "category": category,
                            "model_name": model_name,
                        }
                    )

            if len(metric_results) > 0:
                model_results[category] = metric_results
                print(
                    f"    {category}: {len(pred_dataset.samples)} predictions evaluated"
                )
            else:
                print(f"    Warning: Empty metric results for {category}")
        else:
            print(f"    Warning: Prediction/GT mismatch for {model_name} on {category}")
            print(
                f"      Predictions: {len(pred_dataset.samples)}, Ground truth: {len(gt_dataset.samples)}"
            )
            model_results[category] = {}  # Empty results for missing categories

    # Only process parse_type_data if we have data
    if parse_type_data:
        parse_type_df = pd.DataFrame(parse_type_data)

        # Calculate totals per category
        total_per_cat = {
            category: sum(parse_type_df["count"][parse_type_df["category"] == category])
            for category in parse_type_df["category"].unique()
        }

        # Calculate percentages safely
        parse_type_df["percentage"] = parse_type_df.apply(
            lambda row: 100 * row["count"] / max(1, total_per_cat[row["category"]]),
            axis=1,
        )

        out = "\\begin{table}[H]\n"
        out += "\\centering\n"
        out += "\\caption{Valid rates for each parsing method.}\n"
        out += "\\label{tab:parsing_comparison}\n"
        out += "\\adjustbox{width=\\columnwidth,center}{% \n"
        out += "\\begin{tabular}{lccc}\n"
        out += "\\toprule\n"
        out += "Model & Exact Matching (\\%) & Ranked Matching (\\%) & Unmatched (\\%)\\\\\n"
        out += "\\midrule\n"

        model_names = parse_type_df["model_name"].unique()

        for model_name in model_names:
            exact_match_total = sum(
                parse_type_df["count"][
                    (parse_type_df["model_name"] == model_name)
                    & (parse_type_df["parse_type"] == "exact_matching")
                ]
            )
            ranked_match_total = sum(
                parse_type_df["count"][
                    (parse_type_df["model_name"] == model_name)
                    & (parse_type_df["parse_type"] == "ranked_listing")
                ]
            )
            unmatched_total = sum(
                parse_type_df["count"][
                    (parse_type_df["model_name"] == model_name)
                    & (parse_type_df["parse_type"] == "not_parsed")
                ]
            )
            model_total = sum(
                parse_type_df["count"][parse_type_df["model_name"] == model_name]
            )

            if model_total == 0:
                continue  # not worth adding to table

            exact_match_perc = exact_match_total / model_total
            ranked_match_perc = ranked_match_total / model_total
            unmatched_perc = unmatched_total / model_total

            exact_str = format_value(exact_match_perc * 100, 100.0, 100.0)
            ranked_str = format_value(ranked_match_perc * 100, 100.0, 100.0)
            unmatched_perc = format_value(unmatched_perc * 100, 100.0, 100.0)

            out += f"{model_name} & {exact_str} & {ranked_str} & {unmatched_perc}\\\\\n"

        out += "\\bottomrule\n"
        out += "\\end{tabular}\n"
        out += "}\n"
        out += "\\end{table}\n"

        os.makedirs("./figures/parse_type_data/", exist_ok=True)

        with open(f"./figures/parse_type_data/{model_name}.tex", "w") as f:
            f.write(out)

        parse_type_df.to_csv(f"figures/parse_type_data/{model_name}.csv", index=False)
    else:
        print(f"Warning: No parse type data collected for {model_name}")

    return model_results


def create_results_dataframe(all_model_results: dict, category_datasets: dict):
    """Create a comprehensive DataFrame with all results"""

    all_data = []
    metric = MultipleChoiceMetric()

    for model_name, model_category_results in all_model_results.items():
        for category in category_datasets.keys():
            if (
                category in model_category_results
                and len(model_category_results[category]) > 0
            ):
                # Get table values for this category
                metric_results = model_category_results[category]
                tables = metric.generate_tables_values(metric_results, model_name)

                # summarised_results = metric.summarise(metric_results)

                # for question_name in summarised_results['per_question_name_metrics'].keys():
                #     qtype_h, qtype_n = metric.question_type_mappings[question_name]
                #     fig = create_confusion_matrix_plotly(
                #         ["confusion_matrix"],
                #         question_name=f"{category}: {model_name}",
                #         normalize="true",  # Normalize by true class
                #         show_percentages=True,
                #     )
                #     fig.write_image(
                #         f"figures/confmat_{model_name.lower().replace(' ', '')}_{category}.png"
                #     )

                if category in tables:
                    table_data = tables[category]

                    for row in table_data:
                        if len(row) == 5:  # model, question_type, precision, recall, f1
                            _, question_type, precision, recall, f1 = row
                        elif len(row) == 4:  # model, precision, recall, f1 (overall)
                            _, precision, recall, f1 = row
                        else:
                            raise ValueError("Row invalid!")

                        all_data.append(
                            {
                                "Model": model_name,
                                "Category": category,
                                "Question_Type": question_type,
                                "Precision": precision,
                                "Recall": recall,
                                "F1": f1,
                                "Sample_Count": len(
                                    category_datasets[category].samples
                                ),
                            }
                        )
            # else:
            #     # Add N/A entries for missing categories
            #     all_data.append(
            #         {
            #             "Model": model_name,
            #             "Category": category,
            #             "Question_Type": "Overall",
            #             "Precision": None,
            #             "Recall": None,
            #             "Sample_Count": 0,
            #         }
            #     )

    df = pd.DataFrame(all_data)
    df = df.sort_values("Model")

    return df


def export_results_to_csv(df: pd.DataFrame, save_path: str):
    """Export results to CSV files"""

    # Save master CSV with all data
    master_csv_path = save_path.replace(".tex", "_complete_results.csv")
    df.to_csv(master_csv_path, index=False)
    print(f"Complete results saved to: {master_csv_path}")

    if "Category" not in df.keys():
        return

    # Create separate CSVs for each category
    categories = df["Category"].unique()

    for category in categories:
        category_df = df[df["Category"] == category].copy()

        if category_df.empty:
            continue

        # Create pivot table for this category
        pivot_data = []
        models = category_df["Model"].unique()
        question_types = category_df["Question_Type"].unique()

        for model in models:
            row_data = {"Model": model}
            model_df = category_df[category_df["Model"] == model]

            for qt in question_types:
                qt_data = model_df[model_df["Question_Type"] == qt]
                if not qt_data.empty:
                    row_data[f"{qt}_Precision"] = qt_data.iloc[0]["Precision"]
                    row_data[f"{qt}_Recall"] = qt_data.iloc[0]["Recall"]
                else:
                    row_data[f"{qt}_Precision"] = None
                    row_data[f"{qt}_Recall"] = None

            # Add sample count (should be same for all question types in category)
            if not model_df.empty:
                row_data["Total_Samples"] = model_df.iloc[0]["Sample_Count"]

            pivot_data.append(row_data)

        pivot_df = pd.DataFrame(pivot_data)

        # Save category-specific CSV
        category_csv_path = save_path.replace(".tex", f"_{category}_results.csv")
        pivot_df.to_csv(category_csv_path, index=False)
        print(f"{category.title()} results saved to: {category_csv_path}")


def save_latex_tables(df: pd.DataFrame, save_path: str):
    """Generate LaTeX tables from the results DataFrame"""

    if "Category" not in df.keys():
        return

    out = ""
    
    text_md_explanation = "\\textmd{\\textdagger indicates single-frame finetuning. \\textbf{Bold} indicates best performance, \\underline{underline} indicates second-best.}"

    # Table configurations
    table_configs = [
        # {
        #     "key": "overall",
        #     "title": "Overall Performance",
        #     "caption": f"Performance on our validation set. {text_md_explanation}",
        #     "label": "tab:overall_performance",
        #     "adjustbox": False,
        #     "models": ["LLaVA", "Qwen-VL", "Senna", "LLaVA\\textsuperscript{\\textdagger}", "Senna\\textsuperscript{\\textdagger}"]
        # },
        {
            "key": "binary",
            "title": "Binary Questions by Subtype",
            "caption": f"Performance on binary characteristic questions by subcategory. {text_md_explanation}",
            "label": "tab:binary",
            "adjustbox": True,
            "models": ["LLaVA", "Qwen-VL", "Senna", "LLaVA\\textsuperscript{\\textdagger}", "Senna\\textsuperscript{\\textdagger}"]
        },
        {
            "key": "attribute",
            "title": "Attribute Questions by Subtype",
            "caption": f"Performance on attribute questions by subcategory. {text_md_explanation}",
            "label": "tab:attribute",
            "adjustbox": True,
            "models": ["LLaVA", "Qwen-VL", "Senna", "LLaVA\\textsuperscript{\\textdagger}", "Senna\\textsuperscript{\\textdagger}"]
        },
        {
            "key": "motion",
            "title": "Motion Questions by Subtype",
            "caption": f"Performance on motion questions by subcategory. {text_md_explanation}",
            "label": "tab:motion",
            "adjustbox": True,
            "models": ["LLaVA", "Qwen-VL", "Senna", "LLaVA\\textsuperscript{\\textdagger}", "Senna\\textsuperscript{\\textdagger}"]
        },
        {
            "key": "attribute",
            "title": "Referring Box Attribute Ablation",
            "caption": "We evaluate the effectiveness of red bounding boxes at grounding models to specific object instances across different architectures.",
            "label": "tab:box_ablation_attribute",
            "adjustbox": True,
            "models": ["LLaVA", "LLaVA - No Drawn Box", "Qwen-VL", "Qwen-VL - No Drawn Box"]
        },
        # {
        #     "key": "motion",
        #     "title": "Referring Box Motion Ablation",
        #     "caption": f"We investigate the effect of drawing the box on the images.",
        #     "label": "tab:box_ablation_motion",
        #     "adjustbox": True,
        #     "models": ["LLaVA", "LLaVA - No Drawn Box", "Qwen-VL", "Qwen-VL - No Drawn Box"]
        # },
    ]

    # Generate each table
    for config in table_configs:
        category_df = df[df["Category"] == config["key"]].copy()

        # if config["key"] == "binary":
        #     category_df = category_df.copy()
        #     model_names = category_df["Model"].unique()
            
        #     for model_name in model_names:
        #         if " - No Drawn Box" in model_name: # this doesnt make a difference -> keep the box one (as is more recent and using choices prompt)?
        #             category_df = category_df[category_df["Model"] != model_name.replace(" - No Drawn Box", "")] # remove normal version
                    
        # remove rows that don't have the wanted models
        category_df = category_df[df["Model"].isin(config["models"])].copy()
            
        if category_df.empty:
            print(f"table for {config} is empty!")
            exit()
            continue
        
        if config["key"] == "overall":
            import pprint
            pprint.pp(category_df)
            exit()
            

        # Calculate best and second best values for highlighting
        question_types = category_df["Question_Type"].unique()

        question_best_precision = {}
        question_2ndbest_precision = {}
        question_best_recall = {}
        question_2ndbest_recall = {}
        question_best_f1 = {}
        question_2ndbest_f1 = {}

        for question_type in question_types:
            qt_df = category_df[category_df["Question_Type"] == question_type]

            precision_vals = qt_df["Precision"].dropna().values
            recall_vals = qt_df["Recall"].dropna().values
            f1_vals = qt_df["F1"].dropna().values

            if len(precision_vals) > 0:
                sorted_precision = sorted(set(precision_vals), reverse=True)
                question_best_precision[question_type] = sorted_precision[0]
                question_2ndbest_precision[question_type] = (
                    sorted_precision[1] if len(sorted_precision) > 1 else None
                )

            if len(recall_vals) > 0:
                sorted_recall = sorted(set(recall_vals), reverse=True)
                question_best_recall[question_type] = sorted_recall[0]
                question_2ndbest_recall[question_type] = (
                    sorted_recall[1] if len(sorted_recall) > 1 else None
                )

            if len(f1_vals) > 0:
                sorted_f1 = sorted(set(f1_vals), reverse=True)
                question_best_f1[question_type] = sorted_f1[0]
                question_2ndbest_f1[question_type] = (
                    sorted_f1[1] if len(sorted_f1) > 1 else None
                )

        # Generate LaTeX table
        out += f"\n% {config['title']}\n"
        out += "\\begin{table}[!htb]\n"
        out += "\\centering\n"
        out += f"\\caption{{{config['caption']}}}\n"
        out += f"\\label{{{config['label']}}}\n"
        if config["adjustbox"]:
            out += "\\adjustbox{width=\\columnwidth,center}{% \n"
        out += "\\begin{tabular}{llccc}\n"
        out += "\\toprule\n"

        if len(question_types) > 1 or question_types[0] != "Overall":
            out += "Question Type & Model & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\\n"
        else:
            out += "Model & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\\n"

        out += "\\midrule\n"


        # question type then model
        category_df_sorted = category_df.sort_values(
            ["Question_Type", "Model"], ascending=[True, True]
        )

        if len(question_types) > 1 or question_types[0] != "Overall":
            # Group by question type for multirow
            prev_question_type = None
            question_type_counts = category_df_sorted["Question_Type"].value_counts()
            first_question_type = True

            for _, row in category_df_sorted.iterrows():
                model_name = row["Model"]
                question_type = row["Question_Type"]
                precision = row["Precision"]
                recall = row["Recall"]
                f1 = row["F1"]

                # Format values with highlighting
                precision_str = format_value(
                    precision,
                    question_best_precision.get(question_type),
                    question_2ndbest_precision.get(question_type),
                )
                recall_str = format_value(
                    recall,
                    question_best_recall.get(question_type),
                    question_2ndbest_recall.get(question_type),
                )
                f1_str = format_value(
                    f1,
                    question_best_f1.get(question_type),
                    question_2ndbest_f1.get(question_type),
                )

                # Add midrule between question types (but not before the first one)
                if prev_question_type != question_type and not first_question_type:
                    out += "\\midrule\n"

                # Use multirow for question type
                if prev_question_type != question_type:
                    question_type_cell = f"\\multirow{{{question_type_counts[question_type]}}}{{*}}{{{question_type}}}"
                    prev_question_type = question_type
                    first_question_type = False
                else:
                    question_type_cell = ""

                out += f"{question_type_cell} & {model_name} & {precision_str} & {recall_str} & {f1_str}\\\\\n"
        else:
            # Overall table without question types
            for _, row in category_df_sorted.iterrows():
                model_name = row["Model"]
                question_type = row["Question_Type"]
                precision = row["Precision"]
                recall = row["Recall"]
                f1 = row["F1"]

                # Format values with highlighting
                precision_str = format_value(
                    precision,
                    question_best_precision.get(question_type),
                    question_2ndbest_precision.get(question_type),
                )
                recall_str = format_value(
                    recall,
                    question_best_recall.get(question_type),
                    question_2ndbest_recall.get(question_type),
                )
                f1_str = format_value(
                    f1,
                    question_best_f1.get(question_type),
                    question_2ndbest_f1.get(question_type),
                )

                out += (
                    f"{model_name} & {precision_str} & {recall_str} & {f1_str} \\\\\n"
                )

        out += "\\bottomrule\n"
        out += "\\end{tabular}\n"
        if config["adjustbox"]:
            out += "}\n"
        out += "\\end{table}\n"

    # Save LaTeX file
    with open(save_path, "w") as f:
        f.write(out)

    print(f"LaTeX tables saved to: {save_path}")


def save_latex_tables_with_weighted_averages(df: pd.DataFrame, save_path: str):
    """Generate LaTeX tables from the results DataFrame"""

    if "Category" not in df.keys():
        return

    out = ""
    
    text_md_explanation = "\\textmd{\\textdagger indicates single-frame finetuning. \\textbf{Bold} indicates best performance, \\underline{underline} indicates second-best.}"

    # Table configurations
    table_configs = [
        {
            "key": "binary",
            "title": "Binary Questions by Subtype",
            "caption": f"Performance on binary characteristic questions by subcategory. {text_md_explanation}",
            "label": "tab:binary",
            "adjustbox": True,
            "models": ["LLaVA", "Qwen-VL", "Senna", "LLaVA\\textsuperscript{\\textdagger}", "Senna\\textsuperscript{\\textdagger}"]
        },
        {
            "key": "attribute",
            "title": "Attribute Questions by Subtype",
            "caption": f"Performance on attribute questions by subcategory. {text_md_explanation}",
            "label": "tab:attribute",
            "adjustbox": True,
            "models": ["LLaVA", "Qwen-VL", "Senna", "LLaVA\\textsuperscript{\\textdagger}", "Senna\\textsuperscript{\\textdagger}"]
        },
        {
            "key": "motion",
            "title": "Motion Questions by Subtype",
            "caption": f"Performance on motion questions by subcategory. {text_md_explanation}",
            "label": "tab:motion",
            "adjustbox": True,
            "models": ["LLaVA", "Qwen-VL", "Senna", "LLaVA\\textsuperscript{\\textdagger}", "Senna\\textsuperscript{\\textdagger}"]
        },
        {
            "key": "attribute",
            "title": "Referring Box Attribute Ablation",
            "caption": "We evaluate the effectiveness of red bounding boxes at grounding models to specific object instances across different architectures.",
            "label": "tab:box_ablation_attribute",
            "adjustbox": True,
            "models": ["LLaVA", "LLaVA - No Drawn Box", "Qwen-VL", "Qwen-VL - No Drawn Box"]
        },
    ]

    # Generate each table
    for config in table_configs:
        category_df = df[df["Category"] == config["key"]].copy()
        
        # remove rows that don't have the wanted models
        category_df = category_df[category_df["Model"].isin(config["models"])].copy()
            
        if category_df.empty:
            print(f"table for {config} is empty!")
            continue

        # Calculate weighted averages for each model
        model_averages = {}
        for model in config["models"]:
            model_df = category_df[category_df["Model"] == model]
            if not model_df.empty:
                # Weight by sample count for each question type
                total_samples = model_df["Sample_Count"].sum()
                weighted_precision = (model_df["Precision"] * model_df["Sample_Count"]).sum() / total_samples
                weighted_recall = (model_df["Recall"] * model_df["Sample_Count"]).sum() / total_samples
                weighted_f1 = (model_df["F1"] * model_df["Sample_Count"]).sum() / total_samples
                
                model_averages[model] = {
                    "Precision": weighted_precision,
                    "Recall": weighted_recall,
                    "F1": weighted_f1
                }

        # Calculate best and second best values for highlighting
        question_types = category_df["Question_Type"].unique()

        question_best_precision = {}
        question_2ndbest_precision = {}
        question_best_recall = {}
        question_2ndbest_recall = {}
        question_best_f1 = {}
        question_2ndbest_f1 = {}

        # Add calculations for question types
        for question_type in question_types:
            qt_df = category_df[category_df["Question_Type"] == question_type]

            precision_vals = qt_df["Precision"].dropna().values
            recall_vals = qt_df["Recall"].dropna().values
            f1_vals = qt_df["F1"].dropna().values

            if len(precision_vals) > 0:
                sorted_precision = sorted(set(precision_vals), reverse=True)
                question_best_precision[question_type] = sorted_precision[0]
                question_2ndbest_precision[question_type] = (
                    sorted_precision[1] if len(sorted_precision) > 1 else None
                )

            if len(recall_vals) > 0:
                sorted_recall = sorted(set(recall_vals), reverse=True)
                question_best_recall[question_type] = sorted_recall[0]
                question_2ndbest_recall[question_type] = (
                    sorted_recall[1] if len(sorted_recall) > 1 else None
                )

            if len(f1_vals) > 0:
                sorted_f1 = sorted(set(f1_vals), reverse=True)
                question_best_f1[question_type] = sorted_f1[0]
                question_2ndbest_f1[question_type] = (
                    sorted_f1[1] if len(sorted_f1) > 1 else None
                )

        # Calculate best/second best for weighted averages
        if model_averages:
            avg_precisions = [v["Precision"] for v in model_averages.values()]
            avg_recalls = [v["Recall"] for v in model_averages.values()]
            avg_f1s = [v["F1"] for v in model_averages.values()]
            
            sorted_avg_precision = sorted(set(avg_precisions), reverse=True)
            avg_best_precision = sorted_avg_precision[0]
            avg_2ndbest_precision = sorted_avg_precision[1] if len(sorted_avg_precision) > 1 else None
            
            sorted_avg_recall = sorted(set(avg_recalls), reverse=True)
            avg_best_recall = sorted_avg_recall[0]
            avg_2ndbest_recall = sorted_avg_recall[1] if len(sorted_avg_recall) > 1 else None
            
            sorted_avg_f1 = sorted(set(avg_f1s), reverse=True)
            avg_best_f1 = sorted_avg_f1[0]
            avg_2ndbest_f1 = sorted_avg_f1[1] if len(sorted_avg_f1) > 1 else None

        # Generate LaTeX table
        out += f"\n% {config['title']}\n"
        out += "\\begin{table}[!htb]\n"
        out += "\\centering\n"
        out += f"\\caption{{{config['caption']}}}\n"
        out += f"\\label{{{config['label']}}}\n"
        if config["adjustbox"]:
            out += "\\adjustbox{width=\\columnwidth,center}{% \n"
        out += "\\begin{tabular}{llccc}\n"
        out += "\\toprule\n"

        if len(question_types) > 1 or question_types[0] != "Overall":
            out += "Question Type & Model & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\\n"
        else:
            out += "Model & Precision (\\%) & Recall (\\%) & F1 (\\%) \\\\\n"

        out += "\\midrule\n"

        # question type then model
        category_df_sorted = category_df.sort_values(
            ["Question_Type", "Model"], ascending=[True, True]
        )

        if len(question_types) > 1 or question_types[0] != "Overall":
            # Group by question type for multirow
            prev_question_type = None
            question_type_counts = category_df_sorted["Question_Type"].value_counts()
            first_question_type = True

            for _, row in category_df_sorted.iterrows():
                model_name = row["Model"]
                question_type = row["Question_Type"]
                precision = row["Precision"]
                recall = row["Recall"]
                f1 = row["F1"]

                # Format values with highlighting
                precision_str = format_value(
                    precision,
                    question_best_precision.get(question_type),
                    question_2ndbest_precision.get(question_type),
                )
                recall_str = format_value(
                    recall,
                    question_best_recall.get(question_type),
                    question_2ndbest_recall.get(question_type),
                )
                f1_str = format_value(
                    f1,
                    question_best_f1.get(question_type),
                    question_2ndbest_f1.get(question_type),
                )

                # Add midrule between question types (but not before the first one)
                if prev_question_type != question_type and not first_question_type:
                    out += "\\midrule\n"

                # Use multirow for question type
                if prev_question_type != question_type:
                    question_type_cell = f"\\multirow{{{question_type_counts[question_type]}}}{{*}}{{{question_type}}}"
                    prev_question_type = question_type
                    first_question_type = False
                else:
                    question_type_cell = ""

                out += f"{question_type_cell} & {model_name} & {precision_str} & {recall_str} & {f1_str}\\\\\n"

            # Add weighted average row
            out += "\\midrule\n"
            out += "\\midrule\n"  # Double line before averages
            
            # Sort models to maintain consistent order
            sorted_models = [m for m in config["models"] if m in model_averages]
            
            # weighted_avg_precision = sum(model_averages[m]["Precision"] for m in sorted_models) / len(sorted_models)
            # weighted_avg_recall = sum(model_averages[m]["Recall"] for m in sorted_models) / len(sorted_models)
            # weighted_avg_f1 = sum(model_averages[m]["F1"] for m in sorted_models) / len(sorted_models)
            
            # out += f"\\textit{{Weighted Average}} & {model} & {weighted_avg_precision:.2f} & {weighted_avg_recall:.2f} & {weighted_avg_f1:.2f}\\\\\n"
            
            for i, model in enumerate(sorted_models):
                if model in model_averages:
                    avg_data = model_averages[model]
                    
                    # Format average values with highlighting
                    avg_precision_str = format_value(
                        avg_data["Precision"],
                        avg_best_precision,
                        avg_2ndbest_precision,
                    )
                    avg_recall_str = format_value(
                        avg_data["Recall"],
                        avg_best_recall,
                        avg_2ndbest_recall,
                    )
                    avg_f1_str = format_value(
                        avg_data["F1"],
                        avg_best_f1,
                        avg_2ndbest_f1,
                    )
                    
                    if i == 0:
                        # First model gets the "Weighted Average" label
                        num_models = len(sorted_models)
                        out += f"\\multirow{{{num_models}}}{{*}}{{\\textit{{Weighted Average}}}} & {model} & {avg_precision_str} & {avg_recall_str} & {avg_f1_str}\\\\\n"
                    else:
                        out += f" & {model} & {avg_precision_str} & {avg_recall_str} & {avg_f1_str}\\\\\n"

        else:
            # Overall table without question types
            for _, row in category_df_sorted.iterrows():
                model_name = row["Model"]
                question_type = row["Question_Type"]
                precision = row["Precision"]
                recall = row["Recall"]
                f1 = row["F1"]

                # Format values with highlighting
                precision_str = format_value(
                    precision,
                    question_best_precision.get(question_type),
                    question_2ndbest_precision.get(question_type),
                )
                recall_str = format_value(
                    recall,
                    question_best_recall.get(question_type),
                    question_2ndbest_recall.get(question_type),
                )
                f1_str = format_value(
                    f1,
                    question_best_f1.get(question_type),
                    question_2ndbest_f1.get(question_type),
                )

                out += (
                    f"{model_name} & {precision_str} & {recall_str} & {f1_str} \\\\\n"
                )

        out += "\\bottomrule\n"
        out += "\\end{tabular}\n"
        if config["adjustbox"]:
            out += "}\n"
        out += "\\end{table}\n"

    # Save LaTeX file
    with open(save_path, "w") as f:
        f.write(out)

    print(f"LaTeX tables saved to: {save_path}")

def save_bar_chart(df: pd.DataFrame, save_path: str):
    """Generate professional grouped bar charts suitable for academic papers (CVPR/ICCV style) using matplotlib"""

    if "Category" not in df.keys():
        return

    df = df.copy()

    # Replace model name (your existing code)
    actual_model_col = None
    for col in df.columns:
        if col.lower() == "model":
            actual_model_col = col
            break
    if actual_model_col:
        df[actual_model_col] = df[actual_model_col].str.replace(
            "\\textsuperscript{\\textdagger}", "†", regex=False
        )

    # Filter out unwanted categories
    df = df[df["Category"] != "overall"].copy()
    df["Category"] = df["Category"].str.title()

    # Professional muted color palette (academic-friendly)
    # Using muted colorbrewer colors that work well in print
    # academic_colors = [
    #     "#5b9bd5",  # Muted blue
    #     "#f79646",  # Muted orange
    #     "#70ad47",  # Muted green
    #     "#c55a5a",  # Muted red
    #     "#9f9f9f",  # Muted gray
    #     "#c27ba0",  # Muted purple
    #     "#a17c5a",  # Muted brown
    #     "#87ceeb",  # Light sky blue
    #     "#dda0dd",  # Plum
    #     "#98d8c8",  # Mint
    # ]

    plt.style.use('seaborn-v0_8-whitegrid')  # Academic style

    models = df["Model"].unique()

    academic_colors = sns.color_palette("muted", len(models))
    academic_colors = [(r*0.5 + 0.5, g*0.5 + 0.5, b*0.5 + 0.5) 
                            for r, g, b in academic_colors]

    model_colors = {
        model: academic_colors[i % len(academic_colors)]
        for i, model in enumerate(models)
    }



    # Define custom category order
    category_order = ["Binary", "Attribute", "Motion"]

    # Filter dataframe to only include categories we want, in the order we want
    df = df[df["Category"].isin(category_order)].copy()

    # Convert Category to categorical with specified order
    df["Category"] = pd.Categorical(
        df["Category"], categories=category_order, ordered=True
    )

    # Create the figure with academic proportions
    fig, ax = plt.subplots(figsize=(8, 5))  # Good for 2-column papers
    
    # Prepare data for grouped bar chart
    category_labels = category_order
    x_positions = np.arange(len(category_labels))
    bar_width = 0.8 / len(models)  # Width of bars adjusted for number of models
    
    # Plot bars for each model
    for i, model in enumerate(models):
        model_df = df[df["Model"] == model].copy()
        
        # Group by category and get mean F1 score (in case of multiple entries per category)
        category_f1 = model_df.groupby("Category", observed=True)["F1"].mean()
        
        # Reindex to ensure all categories are present (fill missing with 0)
        category_f1 = category_f1.reindex(category_order, fill_value=0)
        
        # Calculate x positions for this model's bars
        x_pos = x_positions + (i - len(models)/2 + 0.5) * bar_width
        
        # Create bars
        bars = ax.bar(x_pos, category_f1.values, 
                     width=bar_width, 
                     label=model,
                     color=model_colors[model],
                     edgecolor='black',
                     linewidth=0.5,
                     alpha=1.0)

                #          bars = ax.barh(y_positions, percentages, color=bar_colors, 
                #    edgecolor='black', linewidth=0.8, alpha=1.0)
    
    # Set x-axis labels and positions
    ax.set_xticks(x_positions)
    ax.set_xticklabels(category_labels, fontsize=12, fontfamily='Arial')
    ax.set_xlabel("", fontsize=14, fontfamily='Arial')  # Empty as in original
    
    # Set y-axis
    ax.set_ylabel("F1 Score (%)", fontsize=14, fontfamily='Arial')
    
    # Academic styling matching the original
    # ax.spines['top'].set_visible(True)
    # ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(True)
    
    # Set spine colors and width
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)
    
    # Grid styling (light gray, only for y-axis like in original)
    ax.grid(True, axis='y', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Remove x-axis grid
    ax.grid(False, axis='x')
    
    # Set tick parameters for academic style
    ax.tick_params(axis='both', which='major', labelsize=12, 
                   width=1, length=5, direction='out', color='black')
    ax.tick_params(axis='x', which='major', bottom=True, top=False)
    ax.tick_params(axis='y', which='major', left=True, right=False)
    
    # Add zero line
    ax.axhline(y=0, color='black', linewidth=1, zorder=0)
    
    legend = ax.legend(loc='upper right',
                      ncol=min(len(models), 3),  # Max 4 columns to prevent overcrowding
                      frameon=True,  # Transparent background like original
                      fancybox=False,
                      shadow=False,
                      title=None,
                      fontsize=12)

    # Set font family for legend
    for text in legend.get_texts():
        text.set_fontfamily('Arial')
    
    # Set background colors
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Adjust layout with tighter margins for academic papers
    plt.tight_layout()
    plt.subplots_adjust(
        left=0.12,    # Left margin (60/500 from original)
        right=0.96,   # Right margin (20/500 from original) 
        top=0.93,     # Top margin (20/300 from original)
        bottom=0.27   # Bottom margin for legend (80/300 from original)
    )
    
    # Save with high DPI for publication quality
    save_path = Path(save_path).with_suffix(".pdf")
    
    plt.savefig(save_path, 
                dpi=300,  # High DPI like original (scale=3)
                bbox_inches='tight',
                facecolor='white', 
                edgecolor='none',
                format='pdf')  # PDF preferred for LaTeX
    
    print(f"Professional bar chart saved to: {save_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return fig


# Alternative version with even more minimal styling
def save_minimal_bar_chart(df: pd.DataFrame, save_path: str):
    """Ultra-minimal version for maximum professionalism"""

    if "Category" not in df.keys():
        return

    df = df.copy()

    # Replace model name (your existing code)
    actual_model_col = None
    for col in df.columns:
        if col.lower() == "model":
            actual_model_col = col
            break
    if actual_model_col:
        df[actual_model_col] = df[actual_model_col].str.replace(
            "\\textsuperscript{\\textdagger}", "†", regex=False
        )

    # Filter out unwanted categories
    df = df[df["Category"] != "overall"].copy()

    # Grayscale palette for ultimate professionalism
    gray_colors = ["#2c2c2c", "#525252", "#737373", "#969696", "#bdbdbd"]
    models = df["Model"].unique()
    model_colors = {
        model: gray_colors[i % len(gray_colors)] for i, model in enumerate(models)
    }

    # Create the bar chart
    fig = go.Figure()

    for model in models:
        model_df = df[df["Model"] == model].copy()
        category_f1 = model_df.groupby("Category")["F1"].mean()

        fig.add_trace(
            go.Bar(
                x=category_f1.index,
                y=category_f1.values,
                name=model,
                marker_color=model_colors[model],
                marker_line=dict(color="white", width=1),
                opacity=1.0,
            )
        )

    # Minimal styling
    fig.update_layout(
        title=None,
        xaxis=dict(
            title="",  # Remove axis titles if they're in caption
            tickfont=dict(size=10),
            linecolor="black",
            linewidth=1,
            ticks="outside",
        ),
        yaxis=dict(
            title="F1 Score (%)",
            tickfont=dict(size=10),
            linecolor="black",
            linewidth=1,
            ticks="outside",
            showgrid=False,
        ),
        barmode="group",
        bargap=0.2,
        bargroupgap=0.05,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=15, t=15, b=60),
        height=250,
        width=400,
        font=dict(family="Times New Roman, serif", size=10),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    save_path = Path(save_path).with_suffix(".pdf")
    fig.write_image(save_path, width=400, height=250, scale=4, format="pdf")
    print(f"Minimal bar chart saved to: {save_path}")


def save_radar_charts(df: pd.DataFrame, save_path_prefix: str):
    """Generate radar charts from the results DataFrame, similar to save_latex_tables"""

    if "Category" not in df.keys():
        return

    df = df.copy()
    # replace model name
    actual_model_col = None
    for col in df.columns:
        if col.lower() == "model":
            actual_model_col = col
            break
    if actual_model_col:
        df[actual_model_col] = df[actual_model_col].str.replace(
            "\\textsuperscript{\\textdagger}", "†", regex=False
        )

    # Filter out unwanted categories
    df = df[df["Category"] != "overall"].copy()

    # Color palette for models (you can customize these)
    colors = px.colors.qualitative.Set1

    # import pprint
    # df['category_question'] = df.apply(
    #     lambda row: pprint.pp(row)
    # )

    # Find and use the actual column name
    actual_category_col = None
    for col in df.columns:
        if col.lower() == "category":
            actual_category_col = col
            break

    print(f"actual_category_col '{actual_category_col}'")

    if actual_category_col:
        df["category_question"] = df.apply(
            lambda row: f"{row[actual_category_col]}: {row['Question_Type']}", axis=1
        )

    # Get unique models for this category
    models = df["Model"].unique()

    # Create subplot grid - adjust based on number of models
    n_models = len(models)
    if n_models <= 2:
        rows, cols = 1, n_models
    elif n_models <= 4:
        rows, cols = 2, 2
    else:
        rows, cols = (n_models + 2) // 3, 3

    # Create subplots with polar coordinates
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{model}" for model in models],
        specs=[[{"type": "polar"}] * cols for _ in range(rows)],
        # vertical_spacing=0.3,
        # horizontal_spacing=0.05
    )

    # # Process each model
    # for idx, model in enumerate(models):
    #     row = (idx // cols) + 1
    #     col = (idx % cols) + 1

    #     model_df = df[df["Model"] == model].copy()

    #     categories = model_df["Question_Type"].tolist()

    #     # Extract values for each metric
    #     f1_vals = model_df["F1"].fillna(0).tolist()

    #     fig.add_trace(
    #         go.Scatterpolar(
    #             r=f1_vals + [f1_vals[0]] if f1_vals else [0],
    #             theta=categories + [categories[0]] if categories else [""],
    #             fill='toself',
    #             name='F1',
    #             line_color=colors[2],
    #             fillcolor=colors[2],
    #             opacity=0.3,
    #             showlegend=(idx == 0)
    #         ),
    #         row=row, col=col
    #     )

    # jsut categories
    for idx, model in enumerate(models):
        row = (idx // cols) + 1
        col = (idx % cols) + 1

        model_df = df[df["Model"] == model].copy()

        categories = model_df["Category"].tolist()

        # Extract values for each metric
        f1_vals = model_df["F1"].fillna(0).tolist()

        fig.add_trace(
            go.Scatterpolar(
                r=f1_vals + [f1_vals[0]] if f1_vals else [0],
                theta=categories + [categories[0]] if categories else [""],
                fill="toself",
                name="F1",
                line_color=colors[2],
                fillcolor=colors[2],
                opacity=0.3,
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        # title=dict(text="Overall Performance", x=0.5, font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        height=400 * rows,
        width=400 * cols,
    )

    # Update polar axes for all subplots
    for i in range(1, n_models + 1):
        fig.update_polars(
            radialaxis=dict(
                visible=True,
                range=[0, 100],  # Assuming percentage values
                tickmode="linear",
                tick0=0,
                dtick=20,
            ),
            angularaxis=dict(tickfont_size=10),
            # subplot=f"polar{i if i > 1 else ''}"
        )

    # Save the chart
    save_path = save_path_prefix
    fig.write_image(save_path, width=400 * cols, height=400 * rows, scale=2)
    print(f"Radar chart saved to: {save_path}")


def save_radar_chart_single(df: pd.DataFrame, save_path_prefix: str):
    """Generate radar charts from the results DataFrame, similar to save_latex_tables"""

    if "Category" not in df.keys():
        print("Warning: 'Category' column not found in DataFrame")
        return

    df = df.copy()

    # Find actual column names dynamically
    actual_model_col = None
    actual_category_col = None
    actual_f1_col = None

    for col in df.columns:
        if col.lower() == "model":
            actual_model_col = col
        elif col.lower() == "category":
            actual_category_col = col
        elif col.lower() == "f1":
            actual_f1_col = col

    # Validate required columns exist
    if not actual_model_col:
        print("Warning: No 'Model' column found")
        return
    if not actual_category_col:
        print("Warning: No 'Category' column found")
        return
    if not actual_f1_col:
        print("Warning: No 'F1' column found")
        return

    print(
        f"Using columns - Model: '{actual_model_col}', Category: '{actual_category_col}', F1: '{actual_f1_col}'"
    )

    # Replace model name symbols
    df[actual_model_col] = df[actual_model_col].str.replace(
        "\\textsuperscript{\\textdagger}", "†", regex=False
    )

    # Filter out unwanted categories
    df = df[df[actual_category_col] != "overall"].copy()

    if df.empty:
        print("Warning: DataFrame is empty after filtering")
        return

    # Color palette for models
    colors = px.colors.qualitative.Set1

    # Create category-question combination for reference
    if "Question_Type" in df.columns:
        df["category_question"] = df.apply(
            lambda row: f"{row[actual_category_col]}: {row['Question_Type']}", axis=1
        )

    # Get unique values
    models = df[actual_model_col].unique()
    categories = df[actual_category_col].unique()

    print(f"Found {len(models)} models and {len(categories)} categories")

    # Calculate relative F1 scores (normalize within each category)
    # FIX: Use actual_f1_col instead of hardcoded "F1"
    min_f1_per_cat = {}
    max_f1_per_cat = {}

    for cat in categories:
        cat_data = df[df[actual_category_col] == cat][actual_f1_col]
        min_f1_per_cat[cat] = cat_data.min()
        max_f1_per_cat[cat] = cat_data.max()

    # Calculate relative F1 scores
    def calculate_relative_f1(row):
        cat = row[actual_category_col]
        f1_val = row[actual_f1_col]
        min_val = min_f1_per_cat[cat]
        max_val = max_f1_per_cat[cat]

        # Avoid division by zero
        if max_val - min_val < 1e-6:
            return 50.0  # Return middle value if no variation

        return 100 * (f1_val - min_val) / (max_val - min_val)

    df["F1_relative"] = df.apply(calculate_relative_f1, axis=1)

    # Create the radar chart
    fig = go.Figure()

    # Add trace for each model
    for idx, model in enumerate(models):
        model_df = df[df[actual_model_col] == model].copy()

        if model_df.empty:
            continue

        categories_list = model_df[actual_category_col].tolist()
        f1_vals = model_df["F1_relative"].fillna(0).tolist()

        # Ensure we close the radar by repeating first values
        if f1_vals and categories_list:
            categories_closed = categories_list + [categories_list[0]]
            f1_vals_closed = f1_vals + [f1_vals[0]]
        else:
            categories_closed = [""]
            f1_vals_closed = [0]

        fig.add_trace(
            go.Scatterpolar(
                r=f1_vals_closed,
                theta=categories_closed,
                fill="toself",
                name=model,
                line=dict(color=colors[idx % len(colors)], width=2),
                fillcolor=colors[idx % len(colors)],
                opacity=0.2,
                showlegend=True,
                hovertemplate=f"<b>{model}</b><br>"
                + "Category: %{theta}<br>"
                + "Relative F1: %{r:.1f}%<br>"
                + "<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text="F1 Relative Performance per Category",
            x=0.5,
            font=dict(size=18, family="Arial, sans-serif"),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        height=500,
        width=500,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Update polar axes
    fig.update_polars(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            tickmode="linear",
            tick0=0,
            dtick=20,
            tickfont=dict(size=10),
            gridcolor="lightgray",
            linecolor="gray",
        ),
        angularaxis=dict(
            tickfont=dict(size=12), gridcolor="lightgray", linecolor="gray"
        ),
    )

    # Save the chart
    try:
        fig.write_image(save_path_prefix, width=500, height=500, scale=2)
        print(f"Radar chart saved to: {save_path_prefix}")
    except Exception as e:
        print(f"Error saving chart: {e}")

    return fig  # Return figure object for further manipulation if needed


def save_radar_charts_individual(df: pd.DataFrame, save_path_prefix: str):
    """Alternative version: Create individual radar chart for each model, showing all categories"""

    if "Category" not in df.keys():
        return

    # Get unique models
    models = df["Model"].unique()

    # Color palette for categories
    colors = px.colors.qualitative.Set1

    # Chart configurations for filtering
    valid_categories = ["overall", "binary", "attribute", "motion"]

    for model in models:
        model_df = df[df["Model"] == model].copy()

        # Filter to valid categories
        model_df = model_df[model_df["Category"].isin(valid_categories)]

        if model_df.empty:
            continue

        fig = go.Figure()

        categories = model_df["Category"].tolist()
        precision_vals = model_df["Precision"].fillna(0).tolist()
        recall_vals = model_df["Recall"].fillna(0).tolist()
        f1_vals = (
            model_df["F1"].fillna(0).tolist()
            if "F1" in model_df.columns
            else [0] * len(categories)
        )

        # Add traces for each metric
        fig.add_trace(
            go.Scatterpolar(
                r=precision_vals,
                theta=categories,
                fill="toself",
                name="Precision",
                line_color=colors[0],
                fillcolor=colors[0],
                opacity=0.3,
            )
        )

        fig.add_trace(
            go.Scatterpolar(
                r=recall_vals,
                theta=categories,
                fill="toself",
                name="Recall",
                line_color=colors[1],
                fillcolor=colors[1],
                opacity=0.3,
            )
        )

        if any(f1_vals):
            fig.add_trace(
                go.Scatterpolar(
                    r=f1_vals,
                    theta=categories,
                    fill="toself",
                    name="F1",
                    line_color=colors[2],
                    fillcolor=colors[2],
                    opacity=0.3,
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=f"{model}",
            showlegend=True,
            width=500,
            height=500,
        )

        # Save individual chart
        save_path = f"{save_path_prefix}_{model.lower().replace(' ', '_')}_radar.png"
        fig.write_image(save_path, width=500, height=500, scale=2)
        print(f"Individual radar chart saved to: {save_path}")


def create_multiframe_comparison_table(
    generated_samples_path: Path, model_output_path: Path, model_suffix_map: dict
):
    """Create comparison between single-frame and multi-frame results"""

    single_frame_path = (
        generated_samples_path
        / "26_05_2025_export_EgoRelativeObjectTrajectoryPromptGenerator_validation.json"
    )

    # multi_frame_path = (
    #     generated_samples_path / "27_05_2025_multiframe_EgoRelativeObjectTrajectoryMultiFramePromptGenerator_validation.json"
    # )

    multi_frame_path = (
        generated_samples_path
        / "26_05_2025_export_EgoRelativeObjectTrajectoryPromptGenerator_multiframeconverted_validation.json"
    )

    if not single_frame_path.exists() or not multi_frame_path.exists():
        print(
            f"Warning: Missing dataset files for multiframe comparison {single_frame_path.exists()} {multi_frame_path.exists()}"
        )
        return None

    single_frame_ds = VQADataset.load_dataset(str(single_frame_path))
    multi_frame_ds = VQADataset.load_dataset(str(multi_frame_path))

    print(f"Single-frame dataset: {len(single_frame_ds.samples)} samples")
    print(f"Multi-frame dataset: {len(multi_frame_ds.samples)} samples")

    # Create mapping between single-frame and multi-frame suffixes
    suffix_pairs = [
        # ("llava-v1.5-7b_raw_wchoices", "llava-multiframe-wbbox-wchoices"),  # LLaVA
        ("llava-v1.5-7b_raw_wchoices", "llava-v1.5-7b_raw_wchoices", "LLaVA"),  # LLaVA
        ("qwen-vlwchoices", "qwen-vlwchoices", "Qwen-VL"),
        (
            "llava-v1.5-7b_raw_wchoices_nodrawnbbox",
            "llava-v1.5-7b_raw_wchoices_nodrawnbbox",
            "LLaVA No Box",
        ),  # one is without bbox but change later...
        # ("Senna_results_0528__senna_pred", "single_frame_results_0528_multiframe__senna_pred", "Senna"),  # Senna
        (
            "Senna_results_0528__senna_pred",
            "senna_multi_frame_results_0529",
            "Senna",
        ),  # Senna
        # (
        #     "Senna_results_0528__senna_pred",
        #     "senna_multiframe_finetuned_0529",
        #     "Senna vs Senna\\textsuperscript{\\textdagger}",
        # ),  # Senna vs finetuned senna
        (
            "single_frame_results_0528__FT_senna_pred",
            "senna_multiframe_finetuned_0529",
            "Senna$''$",
            # "Senna\\textsuperscript{\\textdagger} vs Senna\\textsuperscript{\\textdaggerdbl}",
        ),  # Senna
        # Add more pairs as needed
    ]

    comparison_results = []
    detailed_results = []
    all_data = []

    for single_suffix, multi_suffix, comp_name in suffix_pairs:
        # Get model name from suffix_map
        # model_name = model_suffix_map.get(single_suffix) or model_suffix_map.get(
        #     multi_suffix
        # ) or comp_name
        model_name = comp_name

        print(
            f"Processing {model_name} (single: {single_suffix}, multi: {multi_suffix})..."
        )

        # Single-frame predictions
        single_pred_path = (
            model_output_path
            / f"26_05_2025_export_EgoRelativeObjectTrajectoryPromptGenerator_validation_{single_suffix}.json"
        )

        # Multi-frame predictions
        multi_pred_path = (
            model_output_path
            / f"26_05_2025_export_EgoRelativeObjectTrajectoryPromptGenerator_multiframeconverted_validation_{multi_suffix}.json"
        )

        print(f"    Single path exists: {single_pred_path.exists()}")
        print(f"    Multi path exists: {multi_pred_path.exists()}")

        if not multi_pred_path.exists() or not single_pred_path.exists():
            print("other results?", list(model_output_path.rglob("*{multi_suffix}*")))
            # print("results", )
            prefix = "26_05_2025_export_EgoRelativeObjectTrajectoryPromptGenerator_multiframeconverted_validation"
            for x in list(model_output_path.rglob(f"{prefix}*")):
                print("other suffix", x.stem.replace(prefix, ""))

            exit()

        single_results = None
        multi_results = None

        # Evaluate single-frame
        if single_pred_path.exists():
            single_pred_ds = VQADataset.load_dataset(str(single_pred_path))
            print(f"    Single predictions: {len(single_pred_ds.samples)} samples")
            if len(single_pred_ds.samples) > 0:
                metric = MultipleChoiceMetric()
                single_results = metric.evaluate_dataset(
                    single_pred_ds, single_frame_ds
                )

        if multi_pred_path.exists():
            multi_pred_ds = VQADataset.load_dataset(str(multi_pred_path))
            print(f"    Multi dataset: {len(multi_frame_ds.samples)} samples")
            print(f"    Multi predictions: {len(multi_pred_ds.samples)} samples")
            if len(multi_pred_ds.samples) > 0:
                metric = MultipleChoiceMetric()
                multi_results = metric.evaluate_dataset(multi_pred_ds, multi_frame_ds)

        # Extract results if both exist
        if single_results and multi_results:
            metric = MultipleChoiceMetric()

            # Get table values
            single_tables = metric.generate_tables_values(single_results, single_suffix)
            multi_tables = metric.generate_tables_values(multi_results, multi_suffix)
            
            print("single_tables", single_tables)
            print("multi_tables", multi_tables)

            # Extract overall results
            single_overall = None
            multi_overall = None

            for category in CATEGORY_TO_PROMPT_GENERATOR.keys():
                if category in single_tables and category in multi_tables:
                    single_table_data = single_tables[category]
                    multi_table_data = multi_tables[category]

                    single_rows_question_type = defaultdict(tuple)
                    multi_rows_question_type = defaultdict(tuple)

                    for row in single_table_data:
                        if len(row) == 5:  # model, question_type, precision, recall, f1
                            _, question_type, precision, recall, f1 = row
                            if question_type in single_rows_question_type:
                                raise ValueError(
                                    f"question_type={question_type} already exists in single_rows_question_type single_table_data={single_table_data}"
                                )
                            single_rows_question_type[question_type] = (
                                precision,
                                recall,
                                f1,
                            )

                    for row in multi_table_data:
                        if len(row) == 5:  # model, question_type, precision, recall, f1
                            _, question_type, precision, recall, f1 = row
                            if question_type in multi_rows_question_type:
                                raise ValueError(
                                    f"question_type={question_type} already exists in multi_rows_question_type multi_table_data={multi_table_data}"
                                )
                            multi_rows_question_type[question_type] = (
                                precision,
                                recall,
                                f1,
                            )

                    for question_type in set(
                        list(single_rows_question_type.keys())
                        + list(multi_rows_question_type.keys())
                    ):
                        if (
                            question_type in single_rows_question_type
                            and question_type in multi_rows_question_type
                        ):
                            single_precision, single_recall, single_f1 = (
                                single_rows_question_type[question_type]
                            )
                            multi_precision, multi_recall, multi_f1 = (
                                multi_rows_question_type[question_type]
                            )

                            all_data.append(
                                {
                                    "Category": category,
                                    "question_type": question_type,
                                    "Model": model_name,
                                    "Comparison_Name": comp_name,
                                    "Single_Suffix": single_suffix,
                                    "Multi_Suffix": multi_suffix,
                                    "Single_Precision": single_precision,
                                    "Single_Recall": single_recall,
                                    "Single_F1": single_f1,
                                    "Multi_Precision": multi_precision,
                                    "Multi_Recall": multi_recall,
                                    "Multi_F1": multi_f1,
                                    "Multi_F1_Improvement": multi_f1 - single_f1,
                                    "Single_Samples": len(single_frame_ds.samples),
                                    "Multi_Samples": len(multi_frame_ds.samples),
                                }
                            )

            for category_data in single_tables.values():
                for row in category_data:
                    if len(row) == 4:  # model, precision, recall, f1 (overall)
                        _, single_precision, single_recall, single_f1 = row
                        single_overall = (single_precision, single_recall, single_f1)
                        break
                if single_overall:
                    break

            for category_data in multi_tables.values():
                for row in category_data:
                    if len(row) == 4:  # model, precision, recall, f1 (overall)
                        _, multi_precision, multi_recall, multi_f1 = row
                        multi_overall = (multi_precision, multi_recall, multi_f1)
                        break
                if multi_overall:
                    break

            if single_overall and multi_overall:
                single_f1 = single_overall[2]
                multi_f1 = multi_overall[2]

                # Log detailed results
                detailed_results.append(
                    {
                        "Comparison_Name": comp_name,
                        "Model_Name": model_name,
                        "Single_Suffix": single_suffix,
                        "Multi_Suffix": multi_suffix,
                        "Single_Precision": single_overall[0],
                        "Single_Recall": single_overall[1],
                        "Single_F1": single_overall[2],
                        "Multi_Precision": multi_overall[0],
                        "Multi_Recall": multi_overall[1],
                        "Multi_F1": multi_overall[2],
                        "Multi_F1_Improvement": multi_f1 - single_f1,
                    }
                )

                comparison_results.append(
                    {
                        "Model": model_name,
                        "Single_Suffix": single_suffix,
                        "Multi_Suffix": multi_suffix,
                        "Single_Precision": single_overall[0],
                        "Single_Recall": single_overall[1],
                        "Single_F1": single_overall[2],
                        "Multi_Precision": multi_overall[0],
                        "Multi_Recall": multi_overall[1],
                        "Multi_F1": multi_overall[2],
                        "Multi_F1_Improvement": multi_f1 - single_f1,
                        "Single_Samples": len(single_frame_ds.samples),
                        "Multi_Samples": len(multi_frame_ds.samples),
                    }
                )

                print(f"    ✓ Results extracted for {model_name}")

    os.makedirs("./figures/csvs/", exist_ok=True)

    # Save detailed results
    if detailed_results:
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv("figures/csvs/multiframe_detailed_all.csv", index=False)
        print(f"Detailed results saved for {len(detailed_results)} model pairs")

    if all_data:
        all_results = pd.DataFrame(all_data)
        all_results.to_csv(
            "figures/csvs/multiframe_detailed_question_named.csv", index=False
        )

    return pd.DataFrame(comparison_results) if comparison_results else None


def save_multiframe_comparison_table(comparison_df: pd.DataFrame, save_path: str):
    """Generate LaTeX table for multiframe comparison"""

    if comparison_df is None or comparison_df.empty:
        print("No multiframe comparison data to save")
        return

    # Calculate best and second best values for highlighting
    # precision_vals = list(comparison_df["Single_Precision"].dropna()) + list(
    #     comparison_df["Multi_Precision"].dropna()
    # )
    # recall_vals = list(comparison_df["Single_Recall"].dropna()) + list(
    #     comparison_df["Multi_Recall"].dropna()
    # )
    single_f1_vals = list(comparison_df["Single_F1"].dropna())
    multi_f1_vals = list(comparison_df["Multi_F1"].dropna())

    # sorted_precision = sorted(set(precision_vals), reverse=True)
    # sorted_recall = sorted(set(recall_vals), reverse=True)
    sorted_single_f1 = sorted(set(single_f1_vals), reverse=True)
    sorted_multi_f1 = sorted(set(multi_f1_vals), reverse=True)

    # best_precision = sorted_precision[0] if sorted_precision else None
    # second_best_precision = sorted_precision[1] if len(sorted_precision) > 1 else None
    # best_recall = sorted_recall[0] if sorted_recall else None
    # second_best_recall = sorted_recall[1] if len(sorted_recall) > 1 else None

    best_single_f1 = sorted_single_f1[0] if sorted_single_f1 else None
    second_best_single_f1 = sorted_single_f1[1] if len(sorted_single_f1) > 1 else None
    best_multi_f1 = sorted_multi_f1[0] if sorted_multi_f1 else None
    second_best_multi_f1 = sorted_multi_f1[1] if len(sorted_multi_f1) > 1 else None

    # \caption{Single-frame vs two-frame input comparison on motion questions. Despite finetuning on temporal data (‡), single-frame input consistently outperforms two-frame input. † indicates single-frame finetuning.}

    out = "\n% Multiframe vs Single-frame Comparison\n"
    out += "\\begin{table}[H]\n"
    out += "\\centering\n"
    # Senna$''$
    out += "\\caption{Single-frame vs two-frame input comparison on motion questions. \\textmd{Senna$''$ compares single-frame finetuned Senna to two-frame finetuned Senna. Despite finetuning on temporal data, single-frame input consistently outperforms two-frame input. \\textbf{Bold} indicates best performance, \\underline{underline} indicates second-best.}}\n"
    # out += "\\caption{Single-frame vs two-frame input comparison on motion questions. \\textmd{Despite finetuning on temporal data (\\textdaggerdbl), single-frame input consistently outperforms two-frame input. \\textdagger indicates single-frame finetuning. \textbf{Bold} indicates best performance, \\underline{underline} indicates second-best.}}\n"
    out += "\\label{tab:multiframe_comparison}\n"
    out += "\\adjustbox{width=\\columnwidth,center}{% \n"
    out += "\\begin{tabular}{lccc}\n"
    out += "\\toprule\n"
    # out += "Model & \\multicolumn{2}{c}{Static Input} & \\multicolumn{2}{c}{Temporal Input} \\\\\n"
    out += "\\multirow{2}{*}{Model} & Single Frame & Two Frames \\\\\n"
    # out += "\\cmidrule(lr){1-2}\n"
    out += " & F1 (\\%) & F1 (\\%) & Improvement (\\%) \\\\\n"
    out += "\\midrule\n"

    # Sort by model name
    comparison_df_sorted = comparison_df.sort_values("Model")

    for _, row in comparison_df_sorted.iterrows():
        model_name = row["Model"]

        # single_precision_str = format_value(
        #     row["Single_Precision"], best_precision, second_best_precision
        # )
        # single_recall_str = format_value(
        #     row["Single_Recall"], best_recall, second_best_recall
        # )
        # multi_precision_str = format_value(
        #     row["Multi_Precision"], best_precision, second_best_precision
        # )
        # multi_recall_str = format_value(
        #     row["Multi_Recall"], best_recall, second_best_recall
        # )
        single_f1_str = format_value(
            row["Single_F1"], best_single_f1, second_best_single_f1
        )
        multi_recall_str = format_value(
            row["Multi_F1"], best_multi_f1, second_best_multi_f1
        )
        improvement_str = format_value(
            row["Multi_F1_Improvement"], 100.0, 100.0  # no underlining..
        )

        # out += f"{model_name} & {single_precision_str} & {single_recall_str} & {multi_precision_str} & {multi_recall_str} \\\\\n"
        out += f"{model_name} & {single_f1_str} & {multi_recall_str} & {improvement_str} \\\\\n"

    out += "\\bottomrule\n"
    out += "\\end{tabular}\n"
    out += "}\n"
    out += "\\end{table}\n"

    # Append to existing file or create new one
    multiframe_save_path = save_path.replace(".tex", "_multiframe.tex")
    with open(multiframe_save_path, "w") as f:
        f.write(out)

    print(f"Multiframe comparison table saved to: {multiframe_save_path}")

    # Also save as CSV
    csv_path = multiframe_save_path.replace(".tex", ".csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"Multiframe comparison data saved to: {csv_path}")


def main():
    """Main function to generate and save a VQA dataset."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate VQA dataset from processed data"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to processed waymo dataset",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)

    assert dataset_path.exists()

    generated_samples_path = dataset_path / "generated_vqa_samples"
    model_output_path = dataset_path / "model_outputs"

    model_suffix_map = {
        # "llava-v1.5-7b": "LLaVA",
        # "llava-v1.5-7b_nobbox": "LLaVA - No Drawn Box",
        "llava-v1.5-7b_raw_wchoices": "LLaVA",

        # for ablation
        # "llava-v1.5-7b_raw_wchoices_nodrawnbbox": "LLaVA - No Drawn Box",
        # "qwen-vlnobbox_wchoices": "Qwen-VL - No Drawn Box",
        
        "qwen-vlwchoices": "Qwen-VL",
        "llava-v1.5-7b-lorafinetuned_wchoices": "LLaVA\\textsuperscript{\\textdagger}",
        # "Senna_results_0527": "SENNA - No Choices",
        # "Senna_results_0528__FT_senna_pred": "Senna Finetuned_",
        # "Senna_results_0528__senna_pred": "Senna",
        # "senna_multi_frame_results_0529": "Senna",
        "single_frame_results_0528__senna_pred": "Senna",
        "single_frame_results_0528__FT_senna_pred": "Senna\\textsuperscript{\\textdagger}",
        # "single_frame_results_0528_multiframe__senna_pred": "Senna_multi",
        # "multi_frame_results_0529__multiframeconverted_senna_pred": "Senna_multi",
        # "llava-multiframe-wbbox-wchoices": "LLaVA_multi",
    }

    save_prefixes = ["26_05_2025_export"]

    multiframe_df = create_multiframe_comparison_table(
        generated_samples_path, model_output_path, model_suffix_map
    )

    os.makedirs("./figures/tables", exist_ok=True)
    if multiframe_df is not None:
        save_multiframe_comparison_table(
            multiframe_df, f"./figures/tables/multiframe_table.tex"
        )
        print(f"Multiframe comparison completed with {len(multiframe_df)} models")
    else:
        print("No multiframe comparison data available")
        exit()

    for save_prefix in save_prefixes:
        print("Step 1: Creating category datasets...")
        category_datasets = create_category_datasets(
            generated_samples_path, save_prefix
        )

        print(f"\nStep 2: Evaluating models...")
        all_model_results = {}

        for model_suffix, model_name in model_suffix_map.items():
            print(f"\nEvaluating {model_name}...")
            model_results = evaluate_model_on_categories(
                model_suffix,
                model_name,
                category_datasets,
                model_output_path,
                save_prefix,
            )
            if model_name not in all_model_results:
                all_model_results[model_name] = model_results
            else:
                raise ValueError(f"{model_name} already in all_model_results")
            

        print(f"\nStep 3: Creating results DataFrame...")
        results_df = create_results_dataframe(all_model_results, category_datasets)

        print(f"\nStep 4: Exporting results...")

        save_prefix_new = save_prefix + "_wacv"

        save_path = f"./figures/tables/{save_prefix_new}_all_models_tables.tex"
        weighted_save_path = f"./figures/tables/{save_prefix_new}_all_models_tables_weighted.tex"
        radar_save_path = f"./figures/tables/radar_{save_prefix_new}.png"
        radar_save_path_single = f"./figures/tables/radar_single_{save_prefix_new}.png"
        bar_save_path = f"./figures/tables/bar_{save_prefix_new}.png"
        bar_minimal_save_path = f"./figures/tables/bar_minimal_{save_prefix_new}.png"
        export_results_to_csv(results_df, save_path)
        save_latex_tables(results_df, save_path)
        save_latex_tables_with_weighted_averages(results_df, weighted_save_path)
        save_radar_charts(results_df, radar_save_path)
        save_bar_chart(results_df, bar_save_path)
        save_minimal_bar_chart(results_df, bar_minimal_save_path)
        save_radar_chart_single(results_df, radar_save_path_single)

        print(f"\nSummary:")
        print(f"Models evaluated: {len(all_model_results)}")
        print(f"Categories: {list(category_datasets.keys())}")
        for category, dataset in category_datasets.items():
            print(f"  {category}: {len(dataset.samples)} samples")

        # Print some diagnostics
        print(f"\nResults overview:")
        for model_name in all_model_results.keys():
            print(f"  {model_name}:")
            for category in category_datasets.keys():
                has_results = (
                    category in all_model_results[model_name]
                    and len(all_model_results[model_name][category]) > 0
                )
                print(f"    {category}: {'✓' if has_results else '✗'}")


if __name__ == "__main__":
    main()
