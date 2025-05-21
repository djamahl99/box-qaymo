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
class QuestionType(str, Enum):
    SINGLE_IMAGE = "single_image"
    SINGLE_IMAGE_MULTI_PROMPT = "single_image_multi_prompt"
    SINGLE_IMAGE_MULTI_CHOICE = "single_image_multi_choice"
    SINGLE_BASE64_IMAGE_MULTI_CHOICE = "single_base64_image_multi_choice"
    MULTI_IMAGE_LIDAR = "multi_image_lidar"
    MULTI_IMAGE = "multi_image"
    MULTI_IMAGE_SINGLE_OBJECT = "multi_image_single_object" 


# Base models for different question types
class BaseQuestion(BaseModel):
    """Base class for all answer formats."""

    generator_name: str
    question_type: QuestionType

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

    @classmethod
    def from_json(cls, data: str):
        data = json.loads(data)

        return cls(**data)