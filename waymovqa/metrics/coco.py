from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from PIL import Image
import cv2

from torchvision.ops import box_iou

from waymovqa.questions.single_image_single_object import (
    SingleImageSingleObjectQuestion,
)

from .base import BaseMetric
from waymovqa.answers.object_2d import Object2DAnswer
from waymovqa.answers.multi_object_2d import MultiObject2DAnswer


class COCOMetric(BaseMetric[Object2DAnswer]):
    """Evaluates object 2D grounding answers using IoU."""

    def __init__(
        self,
        dataset_path: Path,
        iou_threshold: float = 0.5,
        top_k_values: List[int] = [1, 3, 5],
    ):
        super().__init__(Object2DAnswer)
        self.dataset_path = dataset_path
        self.iou_threshold = iou_threshold
        self.top_k_values = top_k_values
        self.prompts = set()
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.predictions = []
        self.ground_truths = []
        self.scores = []
        self.matches = []
        self.prompt_dict = defaultdict(list)
        self.all_image_results = []  # Store per-image metrics

        # COCO format data structures
        self.coco_gt = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}],
        }
        self.coco_dt = []
        self.img_id = 0
        self.ann_id = 0
        self.pred_id = 0

    def summarise(self, metric_results: List[Dict]) -> Dict:
        """
        Summarize evaluation results using COCO-style metrics and top-K accuracy.

        Args:
            metric_results: List of dictionaries containing evaluation results
                            for each prediction-ground truth pair.

        Returns:
            A dictionary with summary metrics including mAP and top-K metrics.
        """
        summary = {"IoU_threshold": self.iou_threshold}

        # Create COCO objects
        coco_gt = COCO()
        coco_gt.dataset = self.coco_gt
        coco_gt.createIndex()

        # If we have no detections or ground truths, return zeros but don't stop evaluation
        if not self.coco_dt or not self.coco_gt["annotations"]:
            print("Warning: No detections or ground truths found for COCO evaluation")
            summary["mAP"] = 0.0
        else:
            try:
                # Create COCO eval object
                coco_dt = coco_gt.loadRes(self.coco_dt)
                coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
                coco_eval.params.iouThrs = [self.iou_threshold]  # Set IoU threshold
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                # Extract the AP value at the specified IoU threshold
                ap = coco_eval.stats[0]  # AP at specific IoU threshold
                summary["mAP"] = ap
            except Exception as e:
                print(f"COCO evaluation error: {e}")
                summary["mAP"] = 0.0
                summary["error"] = str(e)

        # Calculate per-prompt AP if we have prompt data
        prompt_ap = {}
        for prompt in self.prompts:
            # Filter predictions for this prompt
            prompt_dt_list = [
                det for det in self.coco_dt if det.get("prompt") == prompt
            ]

            if not prompt_dt_list:
                prompt_ap[prompt] = 0.0
                continue

            try:
                # Create COCO eval object for this prompt
                prompt_dt = coco_gt.loadRes(prompt_dt_list)
                prompt_eval = COCOeval(coco_gt, prompt_dt, "bbox")
                prompt_eval.params.iouThrs = [self.iou_threshold]
                prompt_eval.evaluate()
                prompt_eval.accumulate()
                prompt_eval.summarize()
                prompt_ap[prompt] = prompt_eval.stats[0]
            except Exception as e:
                print(f"Per-prompt evaluation error for '{prompt}': {e}")
                prompt_ap[prompt] = 0.0

        # Only include per-prompt metrics if we have prompt data
        if prompt_ap:
            summary["per_prompt_AP"] = prompt_ap
            # Calculate mean of per-prompt APs (this is different from overall mAP)
            if prompt_ap:
                summary["mean_prompt_AP"] = sum(prompt_ap.values()) / len(prompt_ap)

        # Calculate top-K metrics from accumulated results
        top_k_metrics = self._calculate_top_k_metrics(self.all_image_results)
        summary.update(top_k_metrics)

        return summary

    def _calculate_top_k_metrics(self, image_results: List[Dict]) -> Dict:
        """
        Calculate top-K accuracy metrics.

        Args:
            image_results: List of per-image evaluation results.

        Returns:
            Dictionary with top-K metrics.
        """
        metrics = {}

        # Initialize counters for each K value
        for k in self.top_k_values:
            metrics[f"AP@top{k}"] = 0.0

        # Count per-prompt top-K metrics
        prompt_topk = {
            prompt: {f"top{k}": 0 for k in self.top_k_values} for prompt in self.prompts
        }
        prompt_counts = defaultdict(int)

        # Calculate overall top-K metrics
        total_images = len(image_results)
        if total_images == 0:
            return metrics

        for result in image_results:
            matches = result.get("matches", [])
            scores = result.get("scores", [])
            prompt = result.get("prompt", "__noprompt__")

            if not matches or not scores:
                continue

            # Sort matches by scores (descending)
            sorted_pairs = sorted(
                zip(scores, matches), key=lambda x: x[0], reverse=True
            )

            # Check if any of the top-K predictions match
            for k in self.top_k_values:
                top_k_pairs = sorted_pairs[:k]
                if any(match > 0 for _, match in top_k_pairs):
                    metrics[f"AP@top{k}"] += 1

                    # Per-prompt top-K
                    if prompt in prompt_topk:
                        prompt_topk[prompt][f"top{k}"] += 1

            # Count this image for the prompt
            prompt_counts[prompt] += 1

        # Normalize by total images
        for k in self.top_k_values:
            metrics[f"AP@top{k}"] /= total_images if total_images > 0 else 1

        # Per-prompt top-K metrics
        for prompt in self.prompts:
            count = prompt_counts[prompt]
            if count > 0:
                for k in self.top_k_values:
                    prompt_topk[prompt][f"top{k}"] /= count

        metrics["per_prompt_topk"] = prompt_topk

        return metrics

    def evaluate(
        self,
        prediction: Object2DAnswer,
        ground_truth: Object2DAnswer,
        question: SingleImageSingleObjectQuestion,
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction against a single ground truth.

        Args:
            prediction: Object2DAnswer prediction
            ground_truth: Object2DAnswer ground truth
            question: SingleImageSingleObjectQuestion

        Returns:
            Dictionary with IoU and match indicator
        """
        # Handle the case of single box comparison
        if not isinstance(prediction, MultiObject2DAnswer) and not isinstance(
            ground_truth, MultiObject2DAnswer
        ):
            # Add to COCO format
            img_id = self.img_id
            self.img_id += 1

            # Get image dimensions
            try:
                with Image.open(self.dataset_path / question.image_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Error opening image {question.image_path}: {e}")
                width, height = 1000, 1000  # Default if image can't be opened

            # Add image
            self.coco_gt["images"].append(
                {"id": img_id, "width": width, "height": height}
            )

            # Add ground truth
            gt_box = np.array(ground_truth.box)
            x1, y1, x2, y2 = gt_box
            width_gt = x2 - x1
            height_gt = y2 - y1

            self.coco_gt["annotations"].append(
                {
                    "id": self.ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(width_gt), float(height_gt)],
                    "area": float(width_gt * height_gt),
                    "iscrowd": 0,
                }
            )
            self.ann_id += 1

            # Add prediction
            pred_box = np.array(prediction.box)
            x1, y1, x2, y2 = pred_box
            width_pred = x2 - x1
            height_pred = y2 - y1

            dt = {
                "id": self.pred_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(width_pred), float(height_pred)],
                "score": float(prediction.score),
                "area": float(width_pred * height_pred),
            }
            self.pred_id += 1

            # Add prompt if available
            if hasattr(prediction, "prompt"):
                dt["prompt"] = prediction.prompt
                self.prompts.add(prediction.prompt)

            self.coco_dt.append(dt)

            # Compute IoU using torch
            torch_iou = box_iou(
                torch.tensor(gt_box).reshape(1, 4), torch.tensor(pred_box).reshape(1, 4)
            ).item()

            match = float(torch_iou >= self.iou_threshold)

            result = {
                "scores": [prediction.score],
                "matches": [match],
                "prompt": (
                    prediction.prompt
                    if hasattr(prediction, "prompt")
                    else "__noprompt__"
                ),
            }

            # Store result for top-K calculation
            self.all_image_results.append(result)

            return result

        # Handle MultiObject2DAnswer or convert single to multi
        pred_multi = (
            prediction
            if isinstance(prediction, MultiObject2DAnswer)
            else MultiObject2DAnswer.from_single_object(prediction)
        )
        gt_multi = (
            ground_truth
            if isinstance(ground_truth, MultiObject2DAnswer)
            else MultiObject2DAnswer.from_single_object(ground_truth)
        )

        # Optional: visualize the multi-object predictions
        self.vis_multi(pred_multi, gt_multi, question)

        return self.evaluate_multi(pred_multi, gt_multi, question)

    def vis_multi(
        self,
        predictions: MultiObject2DAnswer,
        ground_truths: MultiObject2DAnswer,
        question: SingleImageSingleObjectQuestion,
    ) -> None:
        """
        Visualize multiple predictions and ground truths.

        Args:
            predictions: MultiObject2DAnswer with multiple predicted boxes
            ground_truths: MultiObject2DAnswer with multiple ground truth boxes
            question: SingleImageSingleObjectQuestion
        """
        try:
            img_path = self.dataset_path / question.image_path
            img_vis = cv2.imread(str(img_path))

            if img_vis is None:
                print(f"Warning: Could not load image {img_path} for visualization")
                return

            # Add ground truth boxes
            gt_boxes = np.array(ground_truths.boxes)
            for i, gt_box in enumerate(gt_boxes):
                x1, y1, x2, y2 = map(int, gt_box)
                color = (0, 0, 255)  # Red in BGR
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 3)

                # Add text
                text = f"GT"
                cv2.putText(
                    img_vis,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Add predictions
            pred_boxes = np.array(predictions.boxes)
            for i, pred_box in enumerate(pred_boxes):
                x1, y1, x2, y2 = map(int, pred_box)
                score = predictions.scores[i]

                # Use color based on confidence
                if score > 0.7:
                    color = (0, 255, 0)  # Green in BGR
                elif score > 0.4:
                    color = (0, 255, 255)  # Yellow in BGR
                else:
                    color = (0, 165, 255)  # Orange in BGR

                cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)

                # Add confidence text
                text = f"{score:.2f}"
                cv2.putText(
                    img_vis,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Add prompt to image
            prompt = (
                predictions.prompt if hasattr(predictions, "prompt") else "No prompt"
            )
            prompt_text = f"Q: {prompt}"
            cv2.putText(
                img_vis,
                prompt_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imwrite(f"coco_vis_{self.img_id}.jpg", img_vis)
        except Exception as e:
            print(f"Visualization error: {e}")

    def evaluate_multi(
        self,
        predictions: MultiObject2DAnswer,
        ground_truths: MultiObject2DAnswer,
        question: SingleImageSingleObjectQuestion,
    ) -> Dict:
        """
        Evaluate multiple predictions against multiple ground truths.

        Args:
            predictions: MultiObject2DAnswer with multiple predicted boxes
            ground_truths: MultiObject2DAnswer with multiple ground truth boxes
            question: SingleImageSingleObjectQuestion

        Returns:
            Dictionary with evaluation metrics
        """
        img_id = self.img_id
        self.img_id += 1

        try:
            with Image.open(self.dataset_path / question.image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error opening image {question.image_path}: {e}")
            width, height = 1000, 1000  # Default if image can't be opened

        # Add image to COCO GT
        self.coco_gt["images"].append({"id": img_id, "width": width, "height": height})

        # Add ground truth boxes
        gt_boxes = np.array(ground_truths.boxes)
        for i, gt_box in enumerate(gt_boxes):
            x1, y1, x2, y2 = gt_box
            width_gt = x2 - x1
            height_gt = y2 - y1

            self.coco_gt["annotations"].append(
                {
                    "id": self.ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(width_gt), float(height_gt)],
                    "area": float(width_gt * height_gt),
                    "iscrowd": 0,
                }
            )
            self.ann_id += 1

        # Add predictions and calculate matches against all ground truths
        pred_boxes = np.array(predictions.boxes)
        scores = np.array(predictions.scores)
        matches = np.zeros(len(pred_boxes))

        # Calculate IoU for each prediction against each ground truth
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            ious = box_iou(torch.tensor(gt_boxes), torch.tensor(pred_boxes))

            # For each prediction, check if it matches any ground truth
            for i in range(len(pred_boxes)):
                max_iou = torch.max(ious[:, i]).item() if ious.numel() > 0 else 0
                matches[i] = float(max_iou >= self.iou_threshold)

        # Add predictions to COCO format
        for i, (pred_box, score) in enumerate(zip(pred_boxes, scores)):
            x1, y1, x2, y2 = pred_box
            width_pred = x2 - x1
            height_pred = y2 - y1

            dt = {
                "id": self.pred_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [float(x1), float(y1), float(width_pred), float(height_pred)],
                "score": float(score),
                "area": float(width_pred * height_pred),
            }
            self.pred_id += 1

            # Add prompt if available
            prompt = predictions.prompt if hasattr(predictions, "prompt") else None
            if prompt:
                dt["prompt"] = prompt
                self.prompts.add(prompt)

            self.coco_dt.append(dt)

        result = {
            "scores": scores.tolist(),
            "matches": matches.tolist(),
            "prompt": prompt if prompt else "__noprompt__",
        }

        # Store result for top-K calculation
        self.all_image_results.append(result)

        return result


class COCOEvaluator:
    """Class to handle evaluation of a dataset using COCO metrics."""

    def __init__(self, dataset_path: Path, iou_thresholds=None, top_k_values=None):
        """
        Initialize the evaluator with IoU thresholds.

        Args:
            dataset_path: Path to Waymo extracted dataset.
            iou_thresholds: List of IoU thresholds for mAP calculation
                           If None, uses [0.5, 0.55, 0.6, ..., 0.95] (COCO standard)
            top_k_values: List of K values for top-K metrics
                         If None, uses [1, 3, 5]
        """
        self.dataset_path = dataset_path
        if iou_thresholds is None:
            # Standard COCO thresholds
            self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        else:
            self.iou_thresholds = iou_thresholds

        if top_k_values is None:
            self.top_k_values = [1, 3, 5]
        else:
            self.top_k_values = top_k_values

        self.metrics = {
            iou: COCOMetric(
                dataset_path, iou_threshold=iou, top_k_values=self.top_k_values
            )
            for iou in self.iou_thresholds
        }

    def evaluate(self, predictions, ground_truths, questions):
        """
        Evaluate a set of predictions against ground truths.

        Args:
            predictions: List of Object2DAnswer or MultiObject2DAnswer predictions
            ground_truths: List of Object2DAnswer or MultiObject2DAnswer ground truths
            questions: List of questions corresponding to each prediction/gt pair

        Returns:
            Dictionary with COCO-style evaluation metrics
        """
        results = {}

        # Reset all metrics
        for metric in self.metrics.values():
            metric.reset()

        # Evaluate at each IoU threshold
        for iou_threshold, metric in self.metrics.items():
            metric_results = []

            for i, (pred, gt, question) in enumerate(
                zip(predictions, ground_truths, questions)
            ):
                try:
                    # Add prompt info if available
                    if not hasattr(pred, "prompt") and hasattr(question, "question"):
                        pred.prompt = question.question
                    if not hasattr(gt, "prompt") and hasattr(question, "question"):
                        gt.prompt = question.question

                    # Evaluate the prediction
                    result = metric.evaluate(pred, gt, question)
                    metric_results.append(result)
                except Exception as e:
                    print(f"Error evaluating prediction {i}: {e}")
                    # Add empty result to maintain count
                    metric_results.append(
                        {
                            "scores": [],
                            "matches": [],
                            "prompt": (
                                question.question
                                if hasattr(question, "question")
                                else "__noprompt__"
                            ),
                        }
                    )

            # Summarize results for this threshold
            summary = metric.summarise(metric_results)
            results[f"mAP@{iou_threshold:.2f}"] = summary

        # Calculate mAP across IoU thresholds (standard COCO metric)
        map_values = [results[f"mAP@{iou:.2f}"]["mAP"] for iou in self.iou_thresholds]
        results["mAP"] = sum(map_values) / len(map_values)

        # Extract top-K metrics across IoU thresholds
        topk_metrics = {}
        for k in self.top_k_values:
            topk_values = [
                results[f"mAP@{iou:.2f}"].get(f"AP@top{k}", 0.0)
                for iou in self.iou_thresholds
            ]
            topk_metrics[f"AP@top{k}"] = sum(topk_values) / len(topk_values)

        results.update(topk_metrics)

        # Calculate per-prompt mAP if available
        prompt_set = set()
        for question in questions:
            if hasattr(question, "question"):
                prompt_set.add(question.question)

        per_prompt_maps = {prompt: [] for prompt in prompt_set}
        per_prompt_topk = {
            prompt: {f"top{k}": [] for k in self.top_k_values} for prompt in prompt_set
        }

        for iou_threshold in self.iou_thresholds:
            summary = results[f"mAP@{iou_threshold:.2f}"]

            # Per-prompt AP
            if "per_prompt_AP" in summary:
                for prompt, ap in summary["per_prompt_AP"].items():
                    if prompt in per_prompt_maps:
                        per_prompt_maps[prompt].append(ap)

            # Per-prompt top-K
            if "per_prompt_topk" in summary:
                for prompt in prompt_set:
                    if prompt in summary["per_prompt_topk"]:
                        for k in self.top_k_values:
                            key = f"top{k}"
                            if key in summary["per_prompt_topk"][prompt]:
                                per_prompt_topk[prompt][key].append(
                                    summary["per_prompt_topk"][prompt][key]
                                )

        # Average across IoU thresholds for each prompt
        results["per_prompt_mAP"] = {
            prompt: sum(aps) / len(aps) if aps else 0.0
            for prompt, aps in per_prompt_maps.items()
        }

        # Average top-K metrics across IoU thresholds for each prompt
        results["per_prompt_topk"] = {}
        for prompt in prompt_set:
            results["per_prompt_topk"][prompt] = {}
            for k in self.top_k_values:
                key = f"top{k}"
                values = per_prompt_topk[prompt][key]
                results["per_prompt_topk"][prompt][key] = (
                    sum(values) / len(values) if values else 0.0
                )

        # Overall mean of per-prompt mAPs
        if per_prompt_maps:
            non_empty_prompts = [p for p, aps in per_prompt_maps.items() if aps]
            if non_empty_prompts:
                results["mean_prompt_mAP"] = sum(
                    results["per_prompt_mAP"][p] for p in non_empty_prompts
                ) / len(non_empty_prompts)

        return results
