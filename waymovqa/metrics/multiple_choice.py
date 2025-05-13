from abc import ABC, abstractmethod
from typing import Dict, Any, List

from waymovqa.answers.multiple_choice import MultipleChoiceAnswer

from .base import BaseMetric

class MultipleChoiceMetric(BaseMetric[MultipleChoiceAnswer]):
    """Evaluates multiple choice answers."""
    
    def __init__(self):
        super().__init__(MultipleChoiceAnswer)
    
    def evaluate(self, prediction: MultipleChoiceAnswer, ground_truth: MultipleChoiceAnswer) -> Dict[str, float]:
        """
        Evaluate predicted object relationship.
        """
        raise NotImplementedError()