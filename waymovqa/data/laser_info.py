from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
import cv2

from .base import DataObject


class LaserInfo(DataObject):
    """Laser calibration and point cloud data."""

    def __init__(
        self,
        scene_id: str,
        name: str,
        beam_inclinations: List[float] = None,
        extrinsic: List[List[float]] = None,
    ):
        super().__init__(scene_id)
        self.name = name
        self.beam_inclinations = beam_inclinations
        self.extrinsic = extrinsic
        self.point_cloud_path = None
        self.num_points = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert laser info to dictionary."""
        data = {
            "scene_id": self.scene_id,
            "name": self.name,
        }

        # Add calibration data if available
        if self.beam_inclinations is not None:
            data["beam_inclinations"] = self.beam_inclinations
        if self.extrinsic is not None:
            data["extrinsic"] = self.extrinsic

        # Add point cloud data if available
        if self.point_cloud_path is not None:
            data["path"] = self.point_cloud_path
        if self.num_points is not None:
            data["num_points"] = self.num_points

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        laser = cls(
            scene_id=data.get("scene_id", ""),
            name=data["name"],
            beam_inclinations=data.get("beam_inclinations"),
            extrinsic=data.get("extrinsic"),
        )

        if "path" in data:
            laser.point_cloud_path = data["path"]
            laser.num_points = data.get("num_points")

        return laser
