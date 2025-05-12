from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
import cv2

class DataObject:
    """Base class for all data objects."""

    def __init__(self, scene_id: str):
        self.scene_id = scene_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create object from dictionary."""
        raise NotImplementedError

    def save(self, path: Path) -> Path:
        """Save object to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path):
        """Load object from file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
