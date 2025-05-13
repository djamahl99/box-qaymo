from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional, Tuple
from waymovqa.answers.base import BaseAnswer  
from waymovqa.questions.base import BaseQuestion

# Define type variable for answer types
T = TypeVar('T', bound=BaseAnswer)

# Updated metric base class with type checking
class BaseMetric(Generic[T], ABC):
    """Base class for all evaluation metrics with typed answers."""
   
    def __init__(self, answer_type: Type[T]):
        """
        Initialize metric with expected answer type.
        
        Args:
            answer_type: The type of answer this metric evaluates
        """
        self.answer_type = answer_type
   
    @abstractmethod
    def summarise(self, metric_results: List[Dict]) -> Dict[str, Any]:
        """
        Summarize evaluation results across multiple predictions.
        
        Args:
            metric_results: List of dictionaries containing evaluation results
                           for each prediction-ground truth pair.
        
        Returns:
            A dictionary with summary metrics.
        """
        pass
    
    @abstractmethod
    def evaluate(self, prediction: T, ground_truth: T) -> Dict[str, Any]:
        """
        Evaluate a single prediction against ground truth.
        
        Args:
            prediction: The prediction to evaluate
            ground_truth: The ground truth to compare against
            
        Returns:
            A dictionary with evaluation metrics.
        """
        pass
    
    def validate_types(self, prediction: BaseAnswer, ground_truth: BaseAnswer) -> Tuple[T, T]:
        """
        Validate that the prediction and ground truth are of the expected type.
        
        Args:
            prediction: The prediction to validate
            ground_truth: The ground truth to validate
            
        Returns:
            Tuple of validated prediction and ground truth
            
        Raises:
            TypeError: If either prediction or ground truth is not of the expected type
        """
        if not isinstance(prediction, self.answer_type):
            raise TypeError(f"Prediction must be of type {self.answer_type.__name__}, got {type(prediction).__name__}")
        if not isinstance(ground_truth, self.answer_type):
            raise TypeError(f"Ground truth must be of type {self.answer_type.__name__}, got {type(ground_truth).__name__}")
        return prediction, ground_truth
    
    def batch_evaluate(self, predictions: List[T], ground_truths: List[T]) -> Dict[str, Any]:
        """
        Evaluate a batch of predictions against ground truths.
        
        Args:
            predictions: List of predictions to evaluate
            ground_truths: List of ground truths to compare against
            
        Returns:
            A dictionary with summarized evaluation metrics.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError(f"Number of predictions ({len(predictions)}) does not match number of ground truths ({len(ground_truths)})")
        
        metric_results = []
        for pred, gt in zip(predictions, ground_truths):
            validated_pred, validated_gt = self.validate_types(pred, gt)
            result = self.evaluate(validated_pred, validated_gt)
            metric_results.append(result)
            
        return self.summarise(metric_results)