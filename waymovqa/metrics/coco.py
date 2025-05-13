from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import json
from enum import Enum

from .base import BaseMetric
from waymovqa.answers.object_2d import Object2DAnswer
from waymovqa.answers.multi_object_2d import MultiObject2DAnswer


class COCOMetric(BaseMetric[Object2DAnswer]):
    """Evaluates object 2D grounding answers using IoU."""

    def __init__(self, iou_threshold: float = 0.5):
        super().__init__(Object2DAnswer)
        self.iou_threshold = iou_threshold
        self.prompts = set()
        self.reset()
        
    def reset(self):
        """Reset accumulated statistics."""
        self.predictions = []
        self.ground_truths = []
        self.scores = []
        self.matches = []
        self.prompt_dict = defaultdict(list)

    def summarise(self, metric_results: List[Dict]) -> Dict:
        """
        Summarize evaluation results using COCO-style metrics.
        
        Args:
            metric_results: List of dictionaries containing evaluation results
                            for each prediction-ground truth pair.
        
        Returns:
            A dictionary with summary metrics including mAP.
        """
        # Extract scores and matches from results
        scores = []
        matches = []
        prompt_results = defaultdict(list)
        
        for result in metric_results:
            # Each result might have multiple detection pairs
            scores.extend(result["scores"])
            matches.extend(result["matches"])
            
            # Group by prompts if available
            if "prompt" in result:
                prompt = result["prompt"]
                self.prompts.add(prompt)
                prompt_results[prompt].append({
                    "scores": result["scores"],
                    "matches": result["matches"]
                })

        # Calculate overall AP
        ap = self._calculate_average_precision(scores, matches)
        
        # Calculate per-prompt AP if we have prompt data
        prompt_ap = {}
        for prompt, results in prompt_results.items():
            prompt_scores = []
            prompt_matches = []
            for r in results:
                prompt_scores.extend(r["scores"])
                prompt_matches.extend(r["matches"])
            
            if prompt_scores:  # Only calculate if we have data
                prompt_ap[prompt] = self._calculate_average_precision(prompt_scores, prompt_matches)
        
        summary = {
            "mAP": ap,
            "IoU_threshold": self.iou_threshold
        }
        
        # Only include per-prompt metrics if we have prompt data
        if prompt_ap:
            summary["per_prompt_AP"] = prompt_ap
            # Calculate mean of per-prompt APs (this is different from overall mAP)
            if prompt_ap:
                summary["mean_prompt_AP"] = sum(prompt_ap.values()) / len(prompt_ap)
        
        return summary

    def evaluate(self, prediction: Object2DAnswer, ground_truth: Object2DAnswer) -> Dict[str, float]:
        """
        Evaluate a single prediction against a single ground truth.
        
        Args:
            prediction: Object2DAnswer prediction
            ground_truth: Object2DAnswer ground truth
            
        Returns:
            Dictionary with IoU and match indicator
        """
        # Handle the case of single box comparison
        if not hasattr(prediction, 'boxes') and not hasattr(ground_truth, 'boxes'):
            pred_box = np.array(prediction.box)
            gt_box = np.array(ground_truth.box)
            
            iou = self._compute_iou(pred_box, gt_box)
            match = float(iou >= self.iou_threshold)
            
            # Store for accumulation
            self.scores.append(prediction.score)
            self.matches.append(match)
            
            return {
                "iou": iou,
                "match": match,
                "score": prediction.score
            }
        
        # Handle MultiObject2DAnswer or convert single to multi
        pred_multi = prediction if hasattr(prediction, 'boxes') else MultiObject2DAnswer.from_single_object(prediction)
        gt_multi = ground_truth if hasattr(ground_truth, 'boxes') else MultiObject2DAnswer.from_single_object(ground_truth)
        
        return self.evaluate_multi(pred_multi, gt_multi)

    def evaluate_multi(self, predictions: MultiObject2DAnswer, ground_truths: MultiObject2DAnswer) -> Dict:
        """
        Evaluate multiple predictions against multiple ground truths using optimal assignment.
        
        Args:
            predictions: MultiObject2DAnswer with multiple predicted boxes
            ground_truths: MultiObject2DAnswer with multiple ground truth boxes
            
        Returns:
            Dictionary with evaluation metrics
        """
        pred_boxes = np.array(predictions.boxes)
        gt_boxes = np.array(ground_truths.boxes)
        scores = np.array(predictions.scores)
        
        # Create IoU matrix between all pred and gt boxes
        num_preds = len(pred_boxes)
        num_gts = len(gt_boxes)
        iou_matrix = np.zeros((num_preds, num_gts))
        
        for i in range(num_preds):
            for j in range(num_gts):
                iou_matrix[i, j] = self._compute_iou(pred_boxes[i], gt_boxes[j])
        
        # For each prediction, find the best matching ground truth
        matches = np.zeros(num_preds)
        matched_gt_indices = set()
        
        # Sort predictions by score (descending)
        sorted_indices = np.argsort(-scores)
        
        for i in sorted_indices:
            # Find the ground truth with highest IoU
            if num_gts > 0:
                best_gt_idx = np.argmax(iou_matrix[i])
                best_iou = iou_matrix[i, best_gt_idx]
                
                # Check if this GT hasn't been matched yet and the IoU exceeds threshold
                if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt_indices:
                    matches[i] = 1.0
                    matched_gt_indices.add(best_gt_idx)
        
        # Store results for accumulation
        self.scores.extend(scores.tolist())
        self.matches.extend(matches.tolist())
        
        # Store by prompt if available
        prompt = predictions.prompt if hasattr(predictions, 'prompt') else None
        if prompt:
            self.prompt_dict[prompt].append({
                "scores": scores.tolist(),
                "matches": matches.tolist()
            })
        
        return {
            "scores": scores.tolist(),
            "matches": matches.tolist(),
            "prompt": prompt
        }

    @staticmethod
    def _compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """
        Computes IoU between two boxes in [x1, y1, x2, y2] format.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        unionArea = boxAArea + boxBArea - interArea

        if unionArea == 0:
            return 0.0
        return interArea / unionArea
    
    @staticmethod
    def _calculate_average_precision(scores: List[float], matches: List[float]) -> float:
        """
        Calculate Average Precision given scores and match indicators.
        
        Args:
            scores: List of confidence scores for predictions
            matches: List of binary indicators (1.0 if match, 0.0 if no match)
            
        Returns:
            Average Precision value
        """
        if not scores or not matches:
            return 0.0
            
        # Sort by score in descending order
        score_match_pairs = sorted(zip(scores, matches), key=lambda x: x[0], reverse=True)
        sorted_matches = [match for _, match in score_match_pairs]
        
        # Calculate precision and recall at each threshold
        precisions = []
        recalls = []
        tp_cumsum = 0
        fp_cumsum = 0
        total_positives = sum(matches)
        
        if total_positives == 0:
            return 0.0  # No ground truth positives
        
        for i, match in enumerate(sorted_matches):
            if match == 1.0:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
                
            precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            recalls.append(tp_cumsum / total_positives)
        
        # Compute AP using all points
        ap = 0.0
        for i in range(len(precisions)):
            if i == 0:
                ap += precisions[i] * recalls[i]
            else:
                ap += precisions[i] * (recalls[i] - recalls[i-1])
                
        return ap


class COCOEvaluator:
    """Class to handle evaluation of a dataset using COCO metrics."""
    
    def __init__(self, iou_thresholds=None):
        """
        Initialize the evaluator with IoU thresholds.
        
        Args:
            iou_thresholds: List of IoU thresholds for mAP calculation
                           If None, uses [0.5, 0.55, 0.6, ..., 0.95] (COCO standard)
        """
        if iou_thresholds is None:
            # Standard COCO thresholds
            self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        else:
            self.iou_thresholds = iou_thresholds
            
        self.metrics = {
            iou: COCOMetric(iou_threshold=iou) for iou in self.iou_thresholds
        }
    
    def evaluate(self, predictions, ground_truths, prompts=None):
        """
        Evaluate a set of predictions against ground truths.
        
        Args:
            predictions: List of Object2DAnswer or MultiObject2DAnswer predictions
            ground_truths: List of Object2DAnswer or MultiObject2DAnswer ground truths
            prompts: Optional list of prompts corresponding to each prediction/gt pair
            
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
            
            for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                # Add prompt info if available
                if prompts is not None:
                    if hasattr(pred, 'prompt'):
                        pred.prompt = prompts[i]
                    if hasattr(gt, 'prompt'):
                        gt.prompt = prompts[i]
                
                # Evaluate the prediction
                result = metric.evaluate(pred, gt)
                metric_results.append(result)
            
            # Summarize results for this threshold
            summary = metric.summarise(metric_results)
            results[f"mAP@{iou_threshold:.2f}"] = summary
        
        # Calculate mAP across IoU thresholds (standard COCO metric)
        map_values = [results[f"mAP@{iou:.2f}"]["mAP"] for iou in self.iou_thresholds]
        results["mAP"] = sum(map_values) / len(map_values)
        
        # Calculate per-prompt mAP if available
        if prompts is not None and len(prompts) > 0:
            prompt_set = set(prompts)
            per_prompt_maps = {prompt: [] for prompt in prompt_set}
            
            for iou_threshold in self.iou_thresholds:
                summary = results[f"mAP@{iou_threshold:.2f}"]
                if "per_prompt_AP" in summary:
                    for prompt, ap in summary["per_prompt_AP"].items():
                        per_prompt_maps[prompt].append(ap)
            
            # Average across IoU thresholds for each prompt
            results["per_prompt_mAP"] = {
                prompt: sum(aps) / len(aps) if aps else 0.0
                for prompt, aps in per_prompt_maps.items()
            }
            
            # Overall mean of per-prompt mAPs
            if per_prompt_maps:
                non_empty_prompts = [p for p, aps in per_prompt_maps.items() if aps]
                if non_empty_prompts:
                    results["mean_prompt_mAP"] = sum(results["per_prompt_mAP"][p] for p in non_empty_prompts) / len(non_empty_prompts)
        
        return results