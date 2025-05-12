from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
import cv2

from .base import DataObject
from .camera_info import CameraInfo

class ObjectInfo(DataObject):
    """3D object information."""

    def __init__(self, scene_id: str, object_id: str, timestamp: int = None):
        super().__init__(scene_id)
        self.id = object_id
        self.timestamp = timestamp  # Initial timestamp
        self.in_cvat = False
        self.cvat_label = None
        self.cvat_color = None
        self.box = None
        self.camera_synced_box = None
        self.type = None
        self.detection_difficulty_level = None
        self.tracking_difficulty_level = None
        self.num_lidar_points_in_box = None
        self.num_top_lidar_points_in_box = None
        self.most_visible_camera_name = None
        self.metadata = None
        # New fields for multi-frame tracking
        self.frames = []  # List of timestamps where this object appears
        self.visible_cameras = []  # List of cameras where this object is visible

    def to_dict(self) -> Dict[str, Any]:
        """Convert object info to dictionary."""
        data = {
            "scene_id": self.scene_id,
            "id": self.id,
            "timestamp": self.timestamp,
            "in_cvat": self.in_cvat,
            "cvat_label": self.cvat_label,
            "cvat_color": self.cvat_color,
            "frames": self.frames,
            "visible_cameras": self.visible_cameras,
            "box": self.box,
            "type": self.type,
            "detection_difficulty_level": self.detection_difficulty_level,
            "tracking_difficulty_level": self.tracking_difficulty_level,
            "num_lidar_points_in_box": self.num_lidar_points_in_box,
        }

        # Add optional fields if they exist
        if self.camera_synced_box is not None:
            data["camera_synced_box"] = self.camera_synced_box
        if self.most_visible_camera_name is not None:
            data["most_visible_camera_name"] = self.most_visible_camera_name
        if self.num_top_lidar_points_in_box is not None:
            data["num_top_lidar_points_in_box"] = self.num_top_lidar_points_in_box
        if self.metadata is not None:
            data["metadata"] = self.metadata

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        obj = cls(
            scene_id=data["scene_id"], object_id=data["id"], timestamp=data["timestamp"]
        )

        obj.in_cvat = data["in_cvat"]
        obj.cvat_label = data["cvat_label"]
        obj.cvat_color = data["cvat_color"]
        obj.box = data["box"]
        obj.type = data["type"]
        obj.detection_difficulty_level = data["detection_difficulty_level"]
        obj.tracking_difficulty_level = data["tracking_difficulty_level"]
        obj.num_lidar_points_in_box = data["num_lidar_points_in_box"]

        # Add optional fields if they exist
        if "camera_synced_box" in data:
            obj.camera_synced_box = data["camera_synced_box"]

        if "most_visible_camera_name" in data:
            obj.most_visible_camera_name = data["most_visible_camera_name"]

        if "num_top_lidar_points_in_box" in data:
            obj.num_top_lidar_points_in_box = data["num_top_lidar_points_in_box"]

        if "metadata" in data:
            obj.metadata = data["metadata"]

        # Add multi-frame tracking fields
        if "frames" in data:
            obj.frames = data["frames"]
        elif "timestamps" in data:  # For compatibility with older format
            obj.frames = data["timestamps"]
        else:
            # Initialize with single timestamp if frames not provided
            obj.frames = [obj.timestamp] if obj.timestamp is not None else []

        if "visible_cameras" in data:
            obj.visible_cameras = data["visible_cameras"]
        else:
            # Initialize with most_visible_camera if available
            obj.visible_cameras = []
            if obj.most_visible_camera_name:
                obj.visible_cameras.append(obj.most_visible_camera_name)

        return obj

    def project_to_image(self, camera_info: CameraInfo) -> Optional[Dict[str, Any]]:
        """Project 3D object to 2D image."""
        # This would be implemented later
        # Example placeholder for the implementation
        pass

    def add_frame(self, timestamp: int):
        """Add a timestamp to the list of frames where this object appears."""
        if timestamp not in self.frames:
            self.frames.append(timestamp)

    def add_visible_camera(self, camera_name: str):
        """Add a camera to the list of cameras where this object is visible."""
        if camera_name and camera_name not in self.visible_cameras:
            self.visible_cameras.append(camera_name)
