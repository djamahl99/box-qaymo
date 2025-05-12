from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional, Tuple, Protocol
from pathlib import Path
import json

from waymovqa.eval.answer import BaseAnswer
from waymovqa.questions.base import BaseQuestion

TQuestion = TypeVar('TQuestion', bound=BaseQuestion)
TAnswer = TypeVar('TAnswer', bound=BaseAnswer)

class Model(Generic[TQuestion, TAnswer], ABC):
    """Base model interface with typed questions and answers."""
    
    def __init__(self, 
                 question_type: Type[TQuestion], 
                 answer_type: Type[TAnswer],
                ):
        """
        Initialize model with types and capabilities.
        
        Args:
            question_type: Type of questions this model handles
            answer_type: Type of answers this model produces
            prompt_generator: Generator for model-specific prompts
            capabilities: Model capabilities specification
        """
        self.question_type = question_type
        self.answer_type = answer_type
        
    @abstractmethod
    def predict(self, question: TQuestion) -> str:
        """
        Generate a raw text prediction for a single question.
        
        Args:
            question: Question instance to answer
            
        Returns:
            Raw text prediction from the model
        """
        pass
    
    def validate_question(self, question: BaseQuestion) -> Optional[str]:
        """
        Validate if a question can be handled by this model.
        
        Args:
            question: Question to validate
            
        Returns:
            Error message if question can't be handled, None otherwise
        """
        if not isinstance(question, self.question_type):
            return f"Question type mismatch: expected {self.question_type.__name__}, got {question.__class__.__name__}"
            
        return None
    
    def parse_prediction(self, raw_prediction: str) -> TAnswer:
        """
        Parse raw prediction into structured format.
        
        Args:
            raw_prediction: Raw text from model
            
        Returns:
            Structured answer object
        """
        try:
            return self.answer_type.from_text(raw_prediction)
        except ValueError as e:
            print(f"Warning: Failed to parse prediction: {e}")
            # Create an empty instance with low confidence
            empty = self.answer_type()
            if hasattr(empty, 'confidence'):
                empty.confidence = 0.0
            return empty
    
    def batch_predict(self, questions: List[BaseQuestion]) -> Dict[str, Dict[str, Any]]:
        """
        Run predictions on a batch of questions.
        
        Args:
            questions: List of questions to answer
            
        Returns:
            Dict mapping question IDs to prediction results
        """
        results = {}
        for question in questions:
            # Validate question
            error = self.validate_question(question)
            if error:
                print(f"Skipping question {question.id}: {error}")
                results[question.id] = {
                    'error': error,
                    'raw_prediction': None,
                    'structured_prediction': None,
                    'question': str(question)
                }
                continue
                
            # Cast to expected type (we've validated it's the right type)
            typed_question = question  # type: TQuestion
            
            # Generate prediction
            raw_prediction = self.predict(typed_question)
            structured_prediction = self.parse_prediction(raw_prediction)
            
            results[question.id] = {
                'raw_prediction': raw_prediction,
                'structured_prediction': structured_prediction.dict() if hasattr(structured_prediction, 'dict') else vars(structured_prediction),
                'question': str(question)
            }
            
        return results
    
    def export_predictions(self, predictions: Dict[str, Dict[str, Any]], output_path: Path) -> None:
        """
        Export predictions to a JSON file.
        
        Args:
            predictions: Dict of prediction results
            output_path: Path to output file
        """
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)