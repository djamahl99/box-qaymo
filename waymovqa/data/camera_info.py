from pathlib import Path
import pprint
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

    def get_camera_name(self):
        return self.name.replace("_", " ").lower()

    def project_to_camera_xyz(self, points: np.ndarray) -> np.ndarray:
        # vehicle frame to camera sensor frame.
        extrinsic = np.array(self.extrinsic).reshape((4, 4))
        vehicle_to_sensor = extrinsic
        points_hom = np.concatenate((points, np.ones_like(points[:, [0]])), axis=1)
        points_camera_frame = points_hom @ vehicle_to_sensor

        return points_camera_frame[:, :3]

    def project_to_image(
        self,
        points: np.ndarray,
        frame_info: "FrameInfo",
        return_depth: bool = True,
    ) -> np.ndarray:
        from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
        from waymo_open_dataset.utils import box_utils

        pose_matrix = np.array(frame_info.pose)

        homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
        # Matrix multiplication
        world_points_homogeneous = np.matmul(homogeneous_points, pose_matrix.T)
        # Extract 3D coordinates
        world_points = world_points_homogeneous[:, :3]

        image_metadata = np.array(frame_info.pose).reshape(-1).tolist()

        # Find camera info from scene if needed
        camera_info_frame = None
        if frame_info:
            for cam in frame_info.cameras:
                if cam.name == self.name:
                    camera_info_frame = cam
                    break

        metadata = list(
            [
                camera_info_frame.width,
                camera_info_frame.height,
                camera_info_frame.rolling_shutter_direction,
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

        extrinsic, intrinsic = camera_info_frame.extrinsic, camera_info_frame.intrinsic

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

        if "metadata" in data:
            metadata = data["metadata"]
            camera.width = metadata.get("width", camera.width)
            camera.height = metadata.get("height", camera.height)
            camera.rolling_shutter_direction = metadata.get(
                "rolling_shutter_direction", camera.rolling_shutter_direction
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
