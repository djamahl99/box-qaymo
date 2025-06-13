from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import numpy as np
from enum import Enum

from box_qaymo.data.base import DataObject

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from box_qaymo.data.frame_info import FrameInfo  # No circular import at runtime
    from box_qaymo.data.camera_info import CameraInfo
    from box_qaymo.data.scene_info import SceneInfo

WAYMO_TYPE_MAPPING = {
    "TYPE_UNKNOWN": None,
    "TYPE_VEHICLE": "Vehicle",
    "TYPE_PEDESTRIAN": "Pedestrian",
    "TYPE_SIGN": "Sign",
    "TYPE_CYCLIST": "Cyclist",
}

OBJECT_SPEED_THRESH = {
    "TYPE_UNKNOWN": float("inf"),
    "TYPE_VEHICLE": 2.8,  # metres/second
    "TYPE_PEDESTRIAN": 1.1,  # walking speed
    "TYPE_SIGN": float("inf"),  # shouldn't move
    "TYPE_CYCLIST": 1.5,  # a bit faster than say walking
}

# all in m/s
OBJECT_SPEED_CATS = {
    "TYPE_UNKNOWN": [],
    "TYPE_VEHICLE": [
        ("stationary", 0.0, 2.8),  # up to 10km/h
        ("slow speed", 2.8, 11.11),  # up to 40 km/h
        ("medium speed", 11.11, 19.44),  # up to 70 km/h
        ("highway speed", 19.44, float("inf")),
    ],
    "TYPE_PEDESTRIAN": [
        ("stationary", 0.0, 1.1),
        ("walking", 1.1, 2.3),
        ("jogging", 2.3, 3.0),
        ("running", 3.0, float("inf")),
    ],  # jogging about 7mins/km and running faster than 3.0 m/s
    "TYPE_SIGN": [("stationary", 0.0, float("inf"))],
    "TYPE_CYCLIST": [
        ("stationary", 0.0, 1.5),  # up to a bit faster than walking speed
        ("slow moving", 1.5, 8.33),  # up to 30 km/h
        ("fast", 8.33, float("inf")),  # any faster than 30km/h
    ],
}

# DifficultyLevelType from https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/box.py
class DifficultyLevelType(Enum):
    """The difficulty level types. The higher, the harder."""
    UNKNOWN = 0
    LEVEL_1 = 1
    LEVEL_2 = 2

class HeadingType(str, Enum):
    TOWARDS = "towards"
    AWAY = "away"
    LEFT = "left"
    RIGHT = "right"


class MovementType(str, Enum):
    TOWARDS = "towards"
    AWAY = "away"
    LEFT = "left"
    RIGHT = "right"
    NOT_MOVING = "not moving"


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

    def get_world_centre(self, frame: "FrameInfo") -> np.ndarray:
        """Get the object centre position in world coordinates."""
        world_pose = np.array(frame.pose)

        centre = self.get_centre()
        centre_h = np.append(centre, 1.0).reshape(1, 4)
        world_centre = np.matmul(centre_h, world_pose.T).reshape(-1)
        world_centre = world_centre[:3]

        return world_centre

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

    def get_object_bbox_2d(
        self, frame: "FrameInfo", camera: "CameraInfo"
    ) -> List[float]:
        """Get the object's 2d bbox."""
        uvdok = self.project_to_image(
            frame_info=frame, camera_info=camera, return_depth=True
        )
        u, v, depth, ok = uvdok.transpose()
        ok = ok.astype(bool)

        mask = ok & (depth > 0)
        u = u[mask]
        v = v[mask]

        if len(u) < 3:
            return [0.0, 0.0, 0.0, 0.0]

        # Calculate projected area
        x_min, x_max = int(min(u)), int(max(u))
        y_min, y_max = int(min(v)), int(max(v))

        x_min, x_max = [max(min(x, camera.width), 0) for x in [x_min, x_max]]
        y_min, y_max = [max(min(y, camera.height), 0) for y in [y_min, y_max]]

        return [x_min, y_min, x_max, y_max]

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
        """Check if all corners of the projected object lie on the camera."""
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
            return self.basic_description(frame, camera)

        # Ensure coordinates fall within image bounds
        if not (
            np.all((0 <= u) & (u <= camera.width))
            and np.all((0 <= v) & (v <= camera.height))
        ):
            return self.basic_description(frame, camera)

        # Calculate mean image position
        u_mean, v_mean = u.mean(), v.mean()

        # Avoid NaNs if projection is empty
        if np.isnan(u_mean) or np.isnan(v_mean):
            return self.basic_description(frame, camera)

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

    def basic_description(self, frame: "FrameInfo", camera: "CameraInfo") -> str:
        return f"{self.get_object_description()}, {self._heading_text(frame, camera)}"

    def _heading_text(self, frame: "FrameInfo", camera: "CameraInfo") -> str:
        ego_relative = self.get_ego_relative_location_description(frame)
        camera_heading = self.get_camera_heading_direction(frame, camera)
        if camera_heading is not None:
            return f"heading {camera_heading} in the {camera.get_camera_name()} camera, {ego_relative}"
        else:
            return f"in the {camera.get_camera_name()} camera, {ego_relative}"

    def get_heading_vector(self) -> np.ndarray:
        """Get the objects heading vector."""
        obj_centre = self.get_centre().reshape(1, 3)

        box = self.camera_synced_box if self.camera_synced_box is not None else self.box
        # Create a point slightly ahead of the object in its heading direction
        heading_angle = box["heading"]

        heading_vector = np.array(
            [np.cos(heading_angle), np.sin(heading_angle), 0.0]
        ).reshape(1, 3)

        # Scale the heading vector to a reasonable length
        heading_vector = heading_vector * box["length"] * 0.5

        return heading_vector

    def get_camera_heading_direction(
        self,
        frame: "FrameInfo",
        camera: "CameraInfo",
    ) -> HeadingType:
        """Returns heading choice -> returns one of 'towards', 'away', 'left', 'right'"""
        obj_centre = self.get_centre().reshape(1, 3)

        box = self.camera_synced_box if self.camera_synced_box is not None else self.box

        heading_vector = self.get_heading_vector()

        # Create a new point by adding the heading vector to the object's position
        ahead_point = obj_centre + heading_vector

        # Project the points to camera coordinates
        obj_centre_cam = camera.project_to_camera_xyz(obj_centre).reshape(3)
        ahead_point_cam = camera.project_to_camera_xyz(ahead_point).reshape(3)

        # Calculate movement vector in camera coordinates
        vector = ahead_point_cam - obj_centre_cam

        # Normalize the vector for direction determination
        normalized_vector = vector.reshape(3) / np.linalg.norm(vector)

        # only left/right or towards/away (remove z axis)
        normalized_vector = normalized_vector[:2]

        # Find the dominant direction
        max_axis = np.argmax(np.abs(normalized_vector))

        if max_axis == 0:  # X-axis dominates (left-right)
            if normalized_vector[max_axis] > 0:
                return HeadingType.AWAY
            else:
                return HeadingType.TOWARDS
        elif max_axis == 1:  # Y-axis dominates (up-down)
            if normalized_vector[max_axis] > 0:
                return HeadingType.LEFT
            else:
                return HeadingType.RIGHT

    def get_camera_movement_direction(
        self,
        camera: "CameraInfo",
        frame: "FrameInfo",
    ) -> MovementType:
        """Returns heading choice -> returns one of 'towards', 'away', 'left', 'right'"""

        if not self.is_object_moving():
            return MovementType.NOT_MOVING

        obj_centre = self.get_centre().reshape(1, 3)

        # Create a new point by adding the movement vector to the object's position
        ahead_point = obj_centre + self.get_vector()

        # Project the points to camera coordinates
        obj_centre_cam = camera.project_to_camera_xyz(obj_centre).reshape(3)
        ahead_point_cam = camera.project_to_camera_xyz(ahead_point).reshape(3)

        # Calculate movement vector in camera coordinates
        vector = ahead_point_cam - obj_centre_cam

        # Normalize the vector for direction determination
        normalized_vector = vector.reshape(3) / (np.linalg.norm(vector) + 1e-6)

        # only left/right or towards/away (remove z axis)
        normalized_vector = normalized_vector[:2]

        # Find the dominant direction
        max_axis = np.argmax(np.abs(normalized_vector))
        assert (max_axis == 0) or (max_axis == 1)

        if max_axis == 0:  # X-axis dominates (left-right)
            if normalized_vector[max_axis] > 0:
                return MovementType.AWAY
            else:
                return MovementType.TOWARDS
        else:  # Y-axis dominates (up-down)
            if normalized_vector[max_axis] > 0:
                return MovementType.LEFT
            else:
                return MovementType.RIGHT

    def get_vector(self) -> float:
        """Get the object vector in m/s."""
        if self.metadata is None:
            return np.zeros((3,), dtype=float)

        speed_x = float(self.metadata.get("speed_x", 0.0))
        speed_y = float(self.metadata.get("speed_y", 0.0))
        speed_z = float(self.metadata.get("speed_z", 0.0))

        return np.array([speed_x, speed_y, speed_z], dtype=float)

    def get_speed(self) -> float:
        """Get the object scalar speed in m/s."""
        if (
            self.metadata is not None
            and "speed_x" in self.metadata
            and "speed_y" in self.metadata
        ):
            speed_x = float(self.metadata["speed_x"])
            speed_y = float(self.metadata["speed_y"])
            return np.sqrt(speed_x**2 + speed_y**2)

        print(
            f"Object has no speed attributes",
            self.metadata,
            self.metadata.keys() if self.metadata is not None else [],
        )
        return 0.0

    def get_accel(self) -> np.ndarray:
        """Get the object acceleration in m/s**2."""
        if (
            self.metadata is not None
            and "accel_x" in self.metadata
            and "accel_y" in self.metadata
        ):
            return np.array(
                [float(self.metadata["accel_x"]), float(self.metadata["accel_y"])],
                dtype=float,
            )

        print(
            f"Object has no accel attributes",
            self.metadata,
            self.metadata.keys() if self.metadata is not None else [],
        )
        return np.zeros((2,), dtype=float)

    def is_object_moving(self) -> bool:
        """Checks if the object is moving"""
        if self.type is None:
            return False

        return self.get_speed() > OBJECT_SPEED_THRESH[self.type]

    def get_speed_category(self) -> str:
        """Get speed category string."""
        speed = self.get_speed()
        if self.type is None:
            return False

        for speed_cat, lwbnd, upbnd in OBJECT_SPEED_CATS[self.type]:
            if (speed >= lwbnd) and (speed <= upbnd):
                return speed_cat

        return "stationary"

    def __repr__(self) -> str:
        text = f"Object #{self.id} timestamp={self.timestamp}"

        if self.cvat_label is not None:
            text += f" {self.cvat_label}\n"

        text += "Visible Cameras: " + ",".join(self.visible_cameras)

        return text
