from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional
from enum import Enum
from dataclasses import dataclass
import re
from pydantic import BaseModel, Field, field_validator
import json
import numpy as np

# Define prediction and ground truth format types
class AnswerType(str, Enum):
    OBJECT_2D = "object_2d"
    MULTIPLE_OBJECT_2D = "multi_object_2d"
    RAW_TEXT = "raw_text"
    MULTIPLE_CHOICE = "multiple_choice"

# Base models for different answer types
class BaseAnswer(BaseModel):
    """Base class for all answer formats."""

    answer_type: AnswerType

    @classmethod
    def from_json(cls, text: str):
        raise NotImplementedError()
    
    @classmethod
    def from_text(cls, text: str):
        raise NotImplementedError()
    
    @classmethod
    def from_dict(cls, data: Dict):
        raise NotImplementedError()