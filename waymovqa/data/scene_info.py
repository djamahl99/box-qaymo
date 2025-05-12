from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
import cv2

from .base import DataObject
from .camera_info import CameraInfo
from .frame_info import FrameInfo
from .laser_info import LaserInfo
from .object_info import ObjectInfo

class SceneInfo(DataObject):
    """Scene information."""

    def __init__(self, scene_id: str):
        super().__init__(scene_id)
        self.camera_calibrations = []
        self.laser_calibrations = []
        self.frames = []
        self.tfrecord_name = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert scene info to dictionary."""
        data = {
            "scene_id": self.scene_id,
            "camera_calibrations": [cal.to_dict() for cal in self.camera_calibrations],
            "laser_calibrations": [cal.to_dict() for cal in self.laser_calibrations],
        }

        if self.tfrecord_name is not None:
            data["tfrecord_name"] = self.tfrecord_name

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        scene = cls(data["scene_id"])
        scene.tfrecord_name = data.get("tfrecord_name")

        # Load camera calibrations
        for cal_data in data.get("camera_calibrations", []):
            cal = CameraInfo.from_dict(cal_data)
            scene.add_camera_calibration(cal)

        # Load laser calibrations
        for cal_data in data.get("laser_calibrations", []):
            cal = LaserInfo.from_dict(cal_data)
            scene.add_laser_calibration(cal)

        return scene

    def add_camera_calibration(self, calibration: CameraInfo):
        """Add camera calibration to scene."""
        self.camera_calibrations.append(calibration)

    def add_laser_calibration(self, calibration: LaserInfo):
        """Add laser calibration to scene."""
        self.laser_calibrations.append(calibration)

    def add_frame(self, frame: FrameInfo):
        """Add frame to scene."""
        self.frames.append(frame)
        # Sort frames by timestamp
        self.frames.sort(key=lambda f: f.timestamp)

    def get_frame_by_timestamp(self, timestamp: int) -> Optional[FrameInfo]:
        """Get frame by timestamp."""
        for frame in self.frames:
            if frame.timestamp == timestamp:
                return frame
        return None