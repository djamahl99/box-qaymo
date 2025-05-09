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


class ObjectInfo(DataObject):
    """3D object information."""

    def __init__(self, scene_id: str, object_id: str, timestamp: int):
        super().__init__(scene_id)
        self.id = object_id
        self.timestamp = timestamp
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
        self.visible_cameras = None
        self.timestamps = None

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
            
        if "visible_cameras" in data:
            obj.visible_cameras = data['visible_cameras']
            
        if "timestamps" in data:
            obj.timestamps = data['timestamps']

        return obj

    def project_to_image(self, camera_info: CameraInfo) -> Optional[Dict[str, Any]]:
        """Project 3D object to 2D image."""
        # This would be implemented later
        # Example placeholder for the implementation
        pass


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

        # Object references are loaded separately

        frame.range_image_top_pose = data.get("range_image_top_pose")
        frame.camera_projections = data.get("camera_projections", {})

        return frame

    def get_object_by_id(self, object_id: str) -> Optional[ObjectInfo]:
        """Get object by ID."""
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        return None


class SceneInfo(DataObject):
    """Scene information."""

    def __init__(self, scene_id: str):
        super().__init__(scene_id)
        self.camera_calibrations = []
        self.laser_calibrations = []
        self.frames = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        scene = cls(data["scene_id"])

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

    def get_frame_by_timestamp(self, timestamp: int) -> Optional[FrameInfo]:
        """Get frame by timestamp."""
        for frame in self.frames:
            if frame.timestamp == timestamp:
                return frame
        return None
