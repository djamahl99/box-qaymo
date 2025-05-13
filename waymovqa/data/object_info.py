from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
import cv2

from waymovqa.data.base import DataObject

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from waymovqa.data.frame_info import FrameInfo  # No circular import at runtime
    from waymovqa.data.camera_info import CameraInfo
    from waymovqa.data.scene_info import SceneInfo

from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset.utils import box_utils


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

    def project_to_image(
        self,
        frame_info: "FrameInfo",
        camera_info: "CameraInfo",
        scene_info: "SceneInfo" = None,
        return_depth: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Project 3D object to 2D image."""
        pose_matrix = np.array(frame_info.pose)

        box_coords = np.array(
            [
                [
                    self.box["center_x"],
                    self.box["center_y"],
                    self.box["center_z"],
                    self.box["length"],
                    self.box["width"],
                    self.box["height"],
                    self.box["heading"],
                ]
            ]
        )
        corners = box_utils.get_upright_3d_box_corners(box_coords)[0].numpy()

        homogeneous_points = np.hstack([corners, np.ones((corners.shape[0], 1))])
        # Matrix multiplication
        world_points_homogeneous = np.matmul(homogeneous_points, pose_matrix.T)
        # Extract 3D coordinates
        world_points = world_points_homogeneous[:, :3]

        image_metadata = np.array(frame_info.pose).reshape(-1).tolist()

        # Find camera info from scene if needed
        # TODO: fix waymo_extract.py to have this information under a single source
        camera_info_scene = None
        if scene_info:
            for cam in scene_info.camera_calibrations:
                if cam.name == camera_info.name:
                    camera_info_scene = cam
                    break

        camera_info_frame = None
        if frame_info:
            for cam in frame_info.cameras:
                if cam.name == camera_info.name:
                    camera_info_frame = cam
                    break

        # for name, cam_info in [('cam_info', camera_info), ('camera_info_scene', camera_info_scene), ('camera_info_frame', camera_info_frame)]:
            # print(f'{name}.width, {name}.height, {name}.rolling_shutter_direction', cam_info.width, cam_info.height, cam_info.rolling_shutter_direction)

        metadata = list(
            [
                camera_info_scene.width,
                camera_info_scene.height,
                camera_info_scene.rolling_shutter_direction,
            ]
        )

        velocity = camera_info_frame.velocity
        image_metadata.append(velocity.get("v_x", 0))
        image_metadata.append(velocity.get("v_y", 0))
        image_metadata.append(velocity.get("v_z", 0))
        image_metadata.append(velocity.get("w_x", 0))
        image_metadata.append(velocity.get("w_y", 0))
        image_metadata.append(velocity.get("w_z", 0))

        # Get timing attributes with fallbacks
        image_metadata.append(camera_info_frame.pose_timestamp)
        image_metadata.append(camera_info_frame.shutter)
        image_metadata.append(camera_info_frame.camera_trigger_time)
        image_metadata.append(camera_info_frame.camera_readout_done_time)

        extrinsic, intrinsic = camera_info_frame.extrinsic, camera_info_scene.intrinsic
        extrinsic = camera_info_scene.extrinsic if extrinsic is None else extrinsic
        intrinsic = camera_info_scene.intrinsic if intrinsic is None else intrinsic

        assert intrinsic is not None and extrinsic is not None

        extrinsic = np.array(extrinsic, dtype=np.float32).reshape(4, 4)
        intrinsic = np.array(intrinsic, dtype=np.float32)

        try:
            return py_camera_model_ops.world_to_image(
                extrinsic,
                intrinsic,
                metadata,
                image_metadata,
                world_points,
                return_depth=return_depth,
            ).numpy()
        except ValueError as e:
            print(
                dict(
                    extrinsic=extrinsic,
                    intrinsic=intrinsic,
                    metadata=metadata,
                    image_metadata=image_metadata,
                    world_points=world_points,
                )
            )

            raise e

    def add_frame(self, timestamp: int):
        """Add a timestamp to the list of frames where this object appears."""
        if timestamp not in self.frames:
            self.frames.append(timestamp)

    def add_visible_camera(self, camera_name: str):
        """Add a camera to the list of cameras where this object is visible."""
        if camera_name and camera_name not in self.visible_cameras:
            self.visible_cameras.append(camera_name)
