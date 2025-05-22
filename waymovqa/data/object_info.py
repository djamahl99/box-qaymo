from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
import cv2
from enum import Enum

from waymovqa.data.base import DataObject

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from waymovqa.data.frame_info import FrameInfo  # No circular import at runtime
    from waymovqa.data.camera_info import CameraInfo
    from waymovqa.data.scene_info import SceneInfo

WAYMO_TYPE_MAPPING = {
    "TYPE_UNKNOWN": None,
    "TYPE_VEHICLE": "Vehicle",
    "TYPE_PEDESTRIAN": "Pedestrian",
    "TYPE_SIGN": "Sign",
    "TYPE_CYCLIST": "Cyclist",
}


class HeadingType(str, Enum):
    TOWARDS = "towards"
    AWAY = "away"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class ObjectInfo(DataObject):
    """3D object information."""

    def __init__(self, scene_id: str, object_id: str, timestamp: int):
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

        if "visible_cameras" in data:
            obj.visible_cameras = data["visible_cameras"]
        else:
            # Initialize with most_visible_camera if available
            obj.visible_cameras = []
            if obj.most_visible_camera_name:
                obj.visible_cameras.append(obj.most_visible_camera_name)

        return obj

    def get_centre(self) -> np.ndarray:
        """Get object centre in np as (3,) shape"""
        if self.camera_synced_box:
            return np.array(
                [
                    self.camera_synced_box["center_x"],
                    self.camera_synced_box["center_y"],
                    self.camera_synced_box["center_z"],
                ]
            )
        return np.array(
            [self.box["center_x"], self.box["center_y"], self.box["center_z"]]
        )

    def project_to_image(
        self,
        frame_info: "FrameInfo",
        camera_info: "CameraInfo",
        return_depth: bool = True,
    ) -> np.ndarray:
        """Project 3D object to 2D image.

        Note: waymo_open_dataset is imported inside this function to allow
        partial functionality when running in environments where the package
        is not available.
        """
        from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
        from waymo_open_dataset.utils import box_utils
        import tensorflow as tf

        with tf.device("/CPU:0"):

            pose_matrix = np.array(frame_info.pose)

            box = (
                self.camera_synced_box
                if self.camera_synced_box is not None
                else self.box
            )

            box_coords = np.array(
                [
                    [
                        box["center_x"],
                        box["center_y"],
                        box["center_z"],
                        box["length"],
                        box["width"],
                        box["height"],
                        box["heading"],
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
            camera_info_frame = None
            if frame_info:
                for cam in frame_info.cameras:
                    if cam.name == camera_info.name:
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

            extrinsic, intrinsic = (
                camera_info_frame.extrinsic,
                camera_info_frame.intrinsic,
            )

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

    def add_visible_camera(self, camera_name: str):
        """Add a camera to the list of cameras where this object is visible."""
        if camera_name and camera_name not in self.visible_cameras:
            self.visible_cameras.append(camera_name)

    def get_simple_type(self) -> Optional[str]:
        """Return type from Waymo in readable format."""
        return self.type if self.type is None else WAYMO_TYPE_MAPPING[self.type]

    def get_object_cvat_description(self) -> Optional[str]:
        """Generates basic object description for question."""
        prompt = None
        if self.cvat_color and self.cvat_label:
            prompt = f"{self.cvat_color} {self.cvat_label.lower()}"
        elif self.cvat_label:
            prompt = self.cvat_label.lower()

        return prompt

    def get_object_description(self) -> str:
        """Generates basic object description for question."""
        prompt = None
        if self.cvat_color and self.cvat_label:
            prompt = f"{self.cvat_color} {self.cvat_label.lower()}"
        elif self.cvat_label:
            prompt = self.cvat_label.lower()
        elif self.get_simple_type() is not None:
            prompt = self.get_simple_type()

        prompt = (
            "object" if prompt is None else prompt
        )  # very vague... hope it doesnt happen

        return prompt

    def get_ego_relative_location_description(self, frame: "FrameInfo") -> str:
        """
        Returns a string like 'in front of the ego vehicle' based on object position in ego frame.
        Assumes Waymo-like ego frame:
            - X axis: left
            - Y axis: forward
            - Z axis: up
        """
        # Get 4x4 transformation matrix from world to ego
        pose_matrix = np.array(frame.pose).reshape(4, 4)

        # Transform object centre to ego frame
        obj_center_world = np.concatenate(
            [self.get_centre(), [1.0]]
        )  # homogeneous coords
        obj_center_ego = pose_matrix @ obj_center_world  # shape (4,)
        obj_center_ego = obj_center_ego[:3]  # drop homogeneous component

        # Use the X (left/right) and Y (forward/backward) coordinates
        x, y = obj_center_ego[0], obj_center_ego[1]
        angle = np.arctan2(x, y) * 180 / np.pi  # angle from front (Y-axis)

        if -45 <= angle <= 45:
            return "in front of the ego vehicle"
        elif 45 < angle <= 135:
            return "to the left of the ego vehicle"
        elif -135 <= angle < -45:
            return "to the right of the ego vehicle"
        else:
            return "behind the ego vehicle"

    def is_visible_on_camera(self, frame: "FrameInfo", camera: "CameraInfo") -> bool:
        """Generates a description for a specific object on a specific camera."""
        uvdok = self.project_to_image(frame, camera)

        # Extract projected coordinates
        u, v, depth, ok = (
            uvdok[..., 0],
            uvdok[..., 1],
            uvdok[..., 2],
            uvdok[..., -1].astype(bool),
        )

        # Short-circuit if projection is invalid
        return (
            np.all(ok)
            and np.all(depth > 0)
            and all(u >= 0)
            and all(v >= 0)
            and all(u <= camera.width)
            and all(v <= camera.height)
        )

    def is_visible_and_has_lidar_pts(
        self, frame: "FrameInfo", camera: "CameraInfo"
    ) -> bool:
        """Generates a description for a specific object on a specific camera."""
        return self.is_visible_on_camera(frame, camera) & (
            self.num_lidar_points_in_box > 5
        )

    def get_detailed_object_description(
        self, frame: "FrameInfo", camera: "CameraInfo"
    ) -> str:
        """Generates a description for a specific object on a specific camera."""
        uvdok = self.project_to_image(frame, camera)

        # Extract projected coordinates
        u, v, depth, ok = (
            uvdok[..., 0],
            uvdok[..., 1],
            uvdok[..., 2],
            uvdok[..., -1].astype(bool),
        )

        # Short-circuit if projection is invalid
        if not np.all(ok) or not np.all(depth > 0):
            return self._basic_description(frame, camera)

        # Ensure coordinates fall within image bounds
        if not (
            np.all((0 <= u) & (u <= camera.width))
            and np.all((0 <= v) & (v <= camera.height))
        ):
            return self._basic_description(frame, camera)

        # Calculate mean image position
        u_mean, v_mean = u.mean(), v.mean()

        # Avoid NaNs if projection is empty
        if np.isnan(u_mean) or np.isnan(v_mean):
            return self._basic_description(frame, camera)

        # Determine image region (rule of thirds)
        horiz = (
            "left"
            if u_mean < camera.width / 3
            else "center" if u_mean < 2 * camera.width / 3 else "right"
        )
        vert = (
            "top"
            if v_mean < camera.height / 3
            else "middle" if v_mean < 2 * camera.height / 3 else "bottom"
        )
        location_description = f"{vert}-{horiz}"

        # Combine all parts
        return f"{self.get_object_description()} in the {location_description} of the {camera.get_camera_name()} camera, {self._heading_text(frame, camera)}"

    def _basic_description(self, frame: "FrameInfo", camera: "CameraInfo") -> str:
        return f"{self.get_object_description()}, {self._heading_text(frame, camera)}"

    def _heading_text(self, frame: "FrameInfo", camera: "CameraInfo") -> str:
        ego_relative = self.get_ego_relative_location_description(frame)
        camera_heading = self.get_camera_heading_text(camera, frame)
        if camera_heading is not None:
            return f"heading {camera_heading} in the {camera.get_camera_name()} camera, {ego_relative}"
        else:
            return f"in the {camera.get_camera_name()} camera, {ego_relative}"

    def get_camera_heading_text(
        self, camera: "CameraInfo", frame: "FrameInfo", include_z: bool = False
    ) -> str:
        """Returns heading choice -> returns one of 'towards', 'away', 'left', 'right'"""
        if self.type == "TYPE_SIGN":
            return None

        obj_centre = self.get_centre().reshape(1, 3)

        box = self.camera_synced_box if self.camera_synced_box is not None else self.box
        # Create a point slightly ahead of the object in its heading direction
        heading_angle = box["heading"]

        heading_vector = np.array(
            [np.cos(heading_angle), np.sin(heading_angle), 0.0]
        ).reshape(1, 3)

        # Scale the heading vector to a reasonable length
        heading_vector = heading_vector * box["length"] * 0.5

        # Create a new point by adding the heading vector to the object's position
        ahead_point = obj_centre + heading_vector

        # Project the points to camera coordinates
        obj_centre_cam = camera.project_to_camera_xyz(obj_centre).reshape(3)
        ahead_point_cam = camera.project_to_camera_xyz(ahead_point).reshape(3)

        # Calculate movement vector in camera coordinates
        vector = ahead_point_cam - obj_centre_cam

        # Normalize the vector for direction determination
        normalized_vector = vector.reshape(3) / np.linalg.norm(vector)

        if not include_z:
            # only left/right or towards/away (remove z axis)
            normalized_vector = normalized_vector[:2]

        # Find the dominant direction
        max_axis = np.argmax(np.abs(normalized_vector))

        if max_axis == 0:  # X-axis dominates (left-right)
            if normalized_vector[max_axis] > 0:
                return "away"
            else:
                return "towards"
        elif max_axis == 1:  # Y-axis dominates (up-down)
            if normalized_vector[max_axis] > 0:
                return "left"
            else:
                return "right"

        else:  # Z-axis dominates (towards-away)
            if normalized_vector[max_axis] > 0:
                return "up"
            else:
                return "down"

    def __repr__(self) -> str:
        text = f"Object #{self.id} timestamp={self.timestamp}"

        if self.cvat_label is not None:
            text += f" {self.cvat_label}\n"

        text += "Visible Cameras: " + ",".join(self.visible_cameras)

        return text
