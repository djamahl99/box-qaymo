from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
import cv2

from waymovqa.data.base import DataObject
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo

class FrameInfo(DataObject):
    """Frame information."""

    def __init__(self, scene_id: str, timestamp: int):
        super().__init__(scene_id)
        self.timestamp = timestamp
        self.pose = None
        self.cameras = []
        self.point_clouds = []
        self.objects = []
        self.range_image_top_pose = None
        self.camera_projections = {}
        self.time_of_day = None
        self.weather = None
        self.location = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame info to dictionary."""
        data = {
            "scene_id": self.scene_id,
            "timestamp": self.timestamp,
            "pose": self.pose,
            "images": [camera.to_dict() for camera in self.cameras],
            "point_clouds": [pc.to_dict() for pc in self.point_clouds],
            "objects": [
                {
                    "id": obj.id,
                    "path": f"object_infos/object_{obj.id}_{self.scene_id}.json",
                }
                for obj in self.objects
            ],
        }

        if self.range_image_top_pose is not None:
            data["range_image_top_pose"] = self.range_image_top_pose
        if self.camera_projections:
            data["camera_projections"] = self.camera_projections

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        frame = cls(data["scene_id"], data["timestamp"])
        frame.pose = data["pose"]

        # Load camera data
        for img_data in data.get("images", []):
            camera = CameraInfo.from_dict(img_data)
            frame.cameras.append(camera)

        # Load point cloud data
        for pc_data in data.get("point_clouds", []):
            pc = LaserInfo.from_dict(pc_data)
            frame.point_clouds.append(pc)

        for obj in data.get("objects", []):
            if 'path' in obj:
                if 'dataset_path' in data:
                    path = Path(data['dataset_path']) / obj['path']
                    frame.add_object(ObjectInfo.load(path))
            else:
                frame.add_object(ObjectInfo.from_dict(obj))

        frame.range_image_top_pose = data.get("range_image_top_pose")
        frame.camera_projections = data.get("camera_projections", {})

        frame.time_of_day = data.get("time_of_day", None)
        frame.location = data.get("location", None)
        frame.weather = data.get("weather", None)

        return frame

    def get_object_by_id(self, object_id: str) -> Optional[ObjectInfo]:
        """Get object by ID."""
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None

    def add_object(self, obj: ObjectInfo):
        """Add an object to the frame."""
        self.objects.append(obj)
        # Update the object's frames list
        obj.add_frame(self.timestamp)

    @classmethod
    def load(cls, path: Path):
        """Load object from file."""
        with open(path, "r") as f:
            data = json.load(f)

        data['dataset_path'] = path.parent.parent

        return cls.from_dict(data)
