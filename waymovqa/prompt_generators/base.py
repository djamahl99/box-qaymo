from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Type, TypeVar, Generic, Optional, Tuple

import numpy as np

from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.object_info import ObjectInfo
from waymovqa.data.frame_info import FrameInfo
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.laser_info import LaserInfo
from waymovqa.answers import BaseAnswer
from waymovqa.questions import BaseQuestion

# Update BasePromptGenerator with typing information
TQ = TypeVar("TQ", bound=BaseAnswer)
TA = TypeVar("TA", bound=BaseQuestion)

CONFIG = {
    "max_object_distance": 25.0,  # metres
    "min_lidar_pts": 25.0,  # metres
}


class BasePromptGenerator(Generic[TQ], ABC):
    """Base class for all prompt generators with typed answers."""

    @abstractmethod
    def generate(
        self,
        frames: List[FrameInfo],
    ) -> List[Tuple[TQ, TA]]:
        """Generate VQA samples from scene, objects, and optionally a specific frame."""
        pass

    @abstractmethod
    def get_metric_class(self) -> type:
        """Return the name of the metric class to use for evaluation."""
        pass

    @abstractmethod
    def get_question_type(self) -> Type[TQ]:
        """Return the question type class used by this generator."""
        pass

    @abstractmethod
    def get_answer_type(self) -> Type[TA]:
        """Return the answer type class used by this generator."""
        pass

    def _find_objects_visible_in_same_frame(
        self, frames: List[FrameInfo]
    ) -> List[List[str]]:
        frame_seen = []

        for frame in frames:
            seen_this_frame = []

            for obj in frame.objects:
                if (
                    obj.num_lidar_points_in_box >= CONFIG["min_lidar_pts"]
                    and np.linalg.norm(obj.get_centre())
                    <= CONFIG["max_object_distance"]
                    and obj.most_visible_camera_name
                ):
                    seen_this_frame.append(obj.id)

            frame_seen.append(seen_this_frame)

        return frame_seen

    def _find_best_object_views(
        self, frames: List[FrameInfo]
    ) -> Dict[str, Tuple[FrameInfo, CameraInfo, ObjectInfo]]:
        """Find the best frame and camera view for each unique object."""
        object_views = {}

        for frame in frames:
            for obj in frame.objects:
                # Skip objects with no visibility info
                if (
                    not hasattr(obj, "most_visible_camera_name")
                    or not obj.most_visible_camera_name
                ):
                    continue

                # Get the most visible camera for this object
                camera = next(
                    (
                        cam
                        for cam in frame.cameras
                        if cam.name == obj.most_visible_camera_name
                    ),
                    None,
                )
                if not camera:
                    continue

                # Calculate visibility score
                visibility_score = self._calculate_visibility(frame, camera, obj)

                # Skip objects that aren't visible enough
                if visibility_score <= 0:
                    continue

                # Update if this is the best view of this object so far
                if (
                    obj.id not in object_views
                    or visibility_score > object_views[obj.id][3]
                ):
                    object_views[obj.id] = (frame, camera, obj, visibility_score)

        # Remove the visibility score from the returned dictionary
        return {
            obj_id: (frame, camera, obj)
            for obj_id, (frame, camera, obj, _) in object_views.items()
        }

    def _calculate_visibility(
        self, frame: FrameInfo, camera: CameraInfo, obj: ObjectInfo
    ) -> float:
        """Calculate visibility score for an object."""
        # Project object to image
        import tensorflow as tf  # only import when running this functin

        with tf.device("/CPU:0"):
            uvdok = obj.project_to_image(
                frame_info=frame, camera_info=camera, return_depth=True
            )
            u, v, depth, ok = uvdok.transpose()
            ok = ok.astype(bool)

        # Check basic visibility
        if sum(ok) < 6 or min(depth) < 0:
            return 0

        # Calculate projected area
        x_min, x_max = int(min(u)), int(max(u))
        y_min, y_max = int(min(v)), int(max(v))

        x_min, x_max = [max(min(x, camera.width), 0) for x in [x_min, x_max]]
        y_min, y_max = [max(min(y, camera.height), 0) for y in [y_min, y_max]]

        width = x_max - x_min
        height = y_max - y_min

        if width < 32 or height < 32:
            return 0

        if obj.num_lidar_points_in_box < 5:
            return 0

        # Return a visibility score (combination of size and LiDAR points)
        return width * height * obj.num_lidar_points_in_box
