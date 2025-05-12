from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
import cv2

from .base import DataObject

class CameraInfo(DataObject):
    """Camera calibration and image data."""

    def __init__(
        self,
        scene_id: str,
        name: str,
        intrinsic: List[float] = None,
        extrinsic: List[List[float]] = None,
        width: int = None,
        height: int = None,
        rolling_shutter_direction: int = None,
    ):
        super().__init__(scene_id)
        self.name = name
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.width = width
        self.height = height
        self.rolling_shutter_direction = rolling_shutter_direction
        self.image_path = None
        self.velocity = None
        self.pose = None
        self.pose_timestamp = None
        self.shutter = None
        self.camera_trigger_time = None
        self.camera_readout_done_time = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert camera info to dictionary."""
        data = {
            "scene_id": self.scene_id,
            "name": self.name,
        }

        # Add calibration data if available
        if self.intrinsic is not None:
            data["intrinsic"] = self.intrinsic
        if self.extrinsic is not None:
            data["extrinsic"] = self.extrinsic
        if self.width is not None:
            data["width"] = self.width
        if self.height is not None:
            data["height"] = self.height
        if self.rolling_shutter_direction is not None:
            data["rolling_shutter_direction"] = self.rolling_shutter_direction

        # Add image data if available
        if self.image_path is not None:
            data["path"] = self.image_path
        if self.velocity is not None:
            data["velocity"] = self.velocity
        if self.pose is not None:
            data["pose"] = self.pose
        if self.pose_timestamp is not None:
            data["pose_timestamp"] = self.pose_timestamp
        if self.shutter is not None:
            data["shutter"] = self.shutter
        if self.camera_trigger_time is not None:
            data["camera_trigger_time"] = self.camera_trigger_time
        if self.camera_readout_done_time is not None:
            data["camera_readout_done_time"] = self.camera_readout_done_time

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        camera = cls(
            scene_id=data.get("scene_id", ""),
            name=data["name"],
            intrinsic=data.get("intrinsic"),
            extrinsic=data.get("extrinsic"),
            width=data.get("width"),
            height=data.get("height"),
            rolling_shutter_direction=data.get("rolling_shutter_direction"),
        )

        if "path" in data:
            camera.image_path = data["path"]
            camera.velocity = data.get("velocity")
            camera.pose = data.get("pose")
            camera.pose_timestamp = data.get("pose_timestamp")
            camera.shutter = data.get("shutter")
            camera.camera_trigger_time = data.get("camera_trigger_time")
            camera.camera_readout_done_time = data.get("camera_readout_done_time")

        return camera