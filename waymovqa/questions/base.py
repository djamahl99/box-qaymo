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
    MULTI_IMAGE = "multi_image"
    MULTI_IMAGE_LIDAR = "multi_image_lidar"


# Base models for different question types
class BaseQuestion(BaseModel):
    """Base class for all answer formats."""

    question_type: QuestionType
