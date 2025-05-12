from abc import ABC, abstractmethod
from typing import Dict, Any, List

from .base import BaseMetric
from waymovqa.eval.answer import *



# Example of an updated metric class
class ObjectRelationMetric(BaseMetric[ObjectRelationAnswer]):
    """Evaluates object relationship answers."""
    
    def __init__(self):
        super().__init__(ObjectRelationAnswer)
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate predicted object relationship.
        """
        scores = {
            "target_object_accuracy": 0.0,
            "reference_object_accuracy": 0.0,
            "object_relation_accuracy": 0.0,
            "overall": 0.0
        }
        
        # Parse and validate
        pred_struct, gt_struct = self.validate_formats(prediction, ground_truth)
        
        if not pred_struct or not gt_struct:
            # Return zero scores if parsing failed
            return scores
        
        # Score target object identification
        if pred_struct.target_object.lower() == gt_struct.target_object.lower():
            scores["target_object_accuracy"] = 1.0
        
        # Score reference object identification
        if pred_struct.reference_object.lower() == gt_struct.reference_object.lower():
            scores["reference_object_accuracy"] = 1.0
        
        # Score object relation accuracy
        if pred_struct.object_relation == gt_struct.object_relation:
            scores["object_relation_accuracy"] = 1.0
        
        # Calculate overall score
        weights = {
            "target_object_accuracy": 0.3,
            "reference_object_accuracy": 0.3,
            "object_relation_accuracy": 0.4
        }
        
        scores["overall"] = sum(scores[k] * weights[k] for k in weights)
        
        return scores