from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional
from enum import Enum
from dataclasses import dataclass
import re
from pydantic import BaseModel, Field, field_validator
import json
import numpy as np
import uuid

# TODO: tests


# Define prediction and ground truth format types
class QuestionType(str, Enum):
    SINGLE_IMAGE = "single_image"
    SINGLE_IMAGE_SINGLE_OBJECT = "single_image_single_object"
    SINGLE_IMAGE_MULTI_PROMPT = "single_image_multi_prompt"
    SINGLE_IMAGE_MULTI_CHOICE = "single_image_multi_choice"
    SINGLE_BASE64_IMAGE_MULTI_CHOICE = "single_base64_image_multi_choice"
    MULTI_IMAGE_LIDAR = "multi_image_lidar"
    MULTI_IMAGE = "multi_image"
    MULTI_FRAME_MULTI_CHOICE = "multi_frame_multi_choice"
    MULTI_IMAGE_SINGLE_OBJECT = "multi_image_single_object"


# Base models for different question types
class BaseQuestion(BaseModel):
    """Base class for all answer formats."""

    generator_name: str
    question_type: QuestionType
    data: Optional[Dict[str, Any]] = (
        None  # Custom data field for generator-specific info
    )
    question_name: Optional[str] = None
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

    @classmethod
    def from_json(cls, data: str):
        json_dict = json.loads(data)

        return cls(**json_dict)
