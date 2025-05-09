from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional

from waymovqa.eval.answer import BaseAnswer 

# Update BasePromptGenerator with typing information
T = TypeVar('T', bound=BaseAnswer)

# Updated metric base class with type checking
class BaseMetric(Generic[T], ABC):
    """Base class for all evaluation metrics with typed answers."""
    
    def __init__(self, answer_type: Type[T]):
        """Initialize metric with expected answer type."""
        self.answer_type = answer_type
    
    @abstractmethod
    def evaluate(self, prediction: str, ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a prediction against ground truth.
        """
        pass
    
    def parse_prediction(self, prediction: str) -> T:
        """Parse prediction text into structured format."""
        try:
            return self.answer_type.from_text(prediction)
        except ValueError as e:
            print(f"Warning: Failed to parse prediction: {e}")
            # Return a default/empty instance as fallback
            return self.answer_type()
            
    
    def validate_formats(self, prediction: str, ground_truth: Dict[str, Any]) -> Tuple[Optional[T], Optional[T]]:
        """
        Validate and convert prediction and ground truth to structured formats.
        Returns (structured_prediction, structured_ground_truth)
        """
        # Parse prediction
        try:
            structured_prediction = self.parse_prediction(prediction)
        except Exception as e:
            print(f"Error parsing prediction: {e}")
            return None, None
        
        # Get ground truth
        if "structured_answer" in ground_truth:
            try:
                # Convert dict back to object
                structured_ground_truth = self.answer_type(**ground_truth["structured_answer"])
            except Exception as e:
                print(f"Error parsing ground truth: {e}")
                return structured_prediction, None
        else:
            # Try to parse from text
            try:
                structured_ground_truth = self.answer_type.from_text(ground_truth["answer"])
            except Exception as e:
                print(f"Error parsing ground truth from text: {e}")
                return structured_prediction, None
                
        return structured_prediction, structured_ground_truth