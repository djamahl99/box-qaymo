from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional
from enum import Enum
from dataclasses import dataclass
import re
from pydantic import BaseModel, Field, field_validator
import json
import numpy as np
# TODO: tests

# Define prediction and ground truth format types
class AnswerType(str, Enum):
    OBJECT_RELATION = "object_relation"
    YES_NO = "yes_no"
    COUNTING = "counting"
    # DESCRIPTIVE = "descriptive" # probably not
    MULTIPLE_CHOICE = "multiple_choice"
    GROUNDING_2D = "grounding_2d"
    # Add other answer types as needed

# Base models for different answer types
class BaseAnswer(BaseModel):
    """Base class for all answer formats."""
    answer_type: AnswerType

class ObjectRelationAnswer(BaseAnswer):
    """Typed format for object relationship answers."""
    answer_type: AnswerType = AnswerType.OBJECT_RELATION
    target_object: str
    reference_object: str
    object_relation: str  # e.g., "left", "right", "front", "behind"
    
    @field_validator('object_relation')  # Changed from 'spatial_relation'
    @classmethod
    def validate_object_relation(cls, v):
        valid_relations = ["left", "right", "front", "behind"]
        if v.lower() not in valid_relations:
            raise ValueError(f"Object relation must be one of {valid_relations}")
        return v.lower()

    @classmethod
    def from_text(cls, text: str) -> "ObjectRelationAnswer":
        """Parse a textual answer into a structured format."""
        # Example: "The car is to the left of the tree."
        pattern = r"The (\w+) is (to the \w+ of|in front of|behind) the (\w+)"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if not match:
            raise ValueError(f"Could not parse object relationship from: {text}")
        
        target_object = match.group(1)
        relation_text = match.group(2).lower()
        reference_object = match.group(3)
        
        # Convert text relation to standard format
        if "right" in relation_text:
            relation = "right"
        elif "left" in relation_text:
            relation = "left"
        elif "front" in relation_text:
            relation = "front"
        elif "behind" in relation_text:
            relation = "behind"
        else:
            raise ValueError(f"Unknown object relation: {relation_text}")
        
        return cls(
            target_object=target_object,
            reference_object=reference_object,
            object_relation=relation
        )


class YesNoAnswer(BaseAnswer):
    """Typed format for yes/no answers."""
    answer_type: AnswerType = AnswerType.YES_NO
    is_affirmative: bool
    confidence: float = 1.0  # How confident is the answer (0-1)
    
    @classmethod
    def from_text(cls, text: str) -> "YesNoAnswer":
        """Parse a textual yes/no answer."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["yes", "yeah", "correct", "true"]):
            return cls(is_affirmative=True)
        elif any(word in text_lower for word in ["no", "nope", "incorrect", "false"]):
            return cls(is_affirmative=False)
        else:
            raise ValueError(f"Could not determine yes/no from: {text}")


class CountingAnswer(BaseAnswer):
    """Typed format for counting answers."""
    answer_type: AnswerType = AnswerType.COUNTING
    count: int
    object_type: str
    
    @classmethod
    def from_text(cls, text: str) -> "CountingAnswer":
        """Parse a counting answer from text."""
        # Example: "There are 3 cars."
        pattern = r"(?:There (?:is|are)|I can see) (\d+) (\w+)"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if not match:
            # Try another pattern that might extract just the number
            number_pattern = r"(\d+)"
            match = re.search(number_pattern, text)
            if match:
                # We got a number but no object type
                count = int(match.group(1))
                # Try to extract object type separately
                object_pattern = r"(\w+)s?"
                object_match = re.search(object_pattern, text)
                object_type = object_match.group(1) if object_match else "objects"
                return cls(count=count, object_type=object_type)
            else:
                raise ValueError(f"Could not extract count from: {text}")
        
        count = int(match.group(1))
        object_type = match.group(2)
        
        return cls(count=count, object_type=object_type)
    
class MultipleChoiceAnswer(BaseAnswer):
    """Typed format for multiple choice answers."""
    answer_type: AnswerType = AnswerType.MULTIPLE_CHOICE
    choices: List[str]
    choice_idx: int  # index of the selected choice, -1 if ambiguous or not found

    @classmethod
    def from_text(cls, text: str, choices: List[str]) -> "MultipleChoiceAnswer":
        """
        Parse a multiple choice answer from text.
        
        Only one choice should match. If zero or multiple match, return an invalid index (-1).
        """
        matched_indices = [
            idx for idx, choice in enumerate(choices)
            if re.search(rf'\b{re.escape(choice)}\b', text, flags=re.IGNORECASE)
        ]

        choice_idx = matched_indices[0] if len(matched_indices) == 1 else -1
        return cls(choices=choices, choice_idx=choice_idx)
    
class GroundingAnswer(BaseAnswer):
    """Typed format for 2D Grounding answers."""
    answer_type: AnswerType = AnswerType.GROUNDING_2D
    box: np.array
    
    @classmethod
    def from_json(cls, text: str):
        """
        Parse an object detection result from text.
        """
        data = json.loads(text)
        
        box = data['box']
        confidence = data['score']
        
        return cls(box=np.array(box), confidence=confidence)