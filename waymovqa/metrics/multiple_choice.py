from abc import ABC, abstractmethod
from typing import Dict, Any, List

from .base import BaseMetric
from waymovqa.eval.answer import *

# Example of an updated metric class
class MultipleChoiceMetric(BaseMetric[MultipleChoiceAnswer]):
    """Evaluates multiple choice answers."""
    
    def __init__(self):
        super().__init__(MultipleChoiceAnswer)
    
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate predicted object relationship.
        """
        scores = {
            "valid_accuracy": 0.0,
            "parse_accuracy": 0.0,
            "overall": 0.0
        }
        
        # Parse and validate
        pred_struct, gt_struct = self.validate_formats(prediction, ground_truth)
        
        if not pred_struct or not gt_struct:
            # Return zero scores if parsing failed
            return scores
        

        
        return scores