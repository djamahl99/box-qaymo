import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime
import pytz
import numpy as np
from collections import defaultdict
import logging

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


from waymovqa.data import *
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.frame_info import FrameInfo, TimeOfDayType, WeatherType, LocationType
from waymovqa.data.object_info import ObjectInfo
from waymovqa.prompt_generators import BasePromptGenerator, register_prompt_generator

from waymovqa.questions.multi_image_multi_choice import (
    MultipleImageMultipleChoiceQuestion,
)
from waymovqa.questions.single_image_multi_choice import (
    SingleImageMultipleChoiceQuestion,
)
from waymovqa.questions.base import BaseQuestion
from waymovqa.answers.multiple_choice import MultipleChoiceAnswer
from waymovqa.metrics.multiple_choice import MultipleChoiceMetric
from waymovqa.metrics.base import BaseMetric

# Set up logging
logger = logging.getLogger(__name__)

# Default config values that can be overridden
DEFAULT_CONFIG = {
    "distance_threshold": 10.0,  # Threshold for similar trajectories
    "min_variance": 0.5,  # Minimum velocity for moving objects of 0.5 metres/second
    "min_trajectory_points": 3,  # Minimum number of points needed for a valid trajectory
    "max_objects_per_scene": 10,  # Maximum objects to consider per scene
    "consider_stationary_objects": True,  # Whether to include stationary objects in some questions
    "max_camera_distance": 800,  # pixels between query objects
}


@register_prompt_generator
class ObjectTrajectoryPromptGenerator(BasePromptGenerator):
    """
    Generates questions about object trajectories and their relationships in a driving scene.

    This generator analyzes movement patterns of objects in the scene and creates questions
    about objects with similar or intersecting paths, helping evaluate perception and prediction
    capabilities for autonomous driving systems.
    """

    def __init__(self, config=None):
        """
        Initialize the prompt generator with optional configuration.

        Args:
            config: Dictionary with configuration parameters that override defaults
        """
        super().__init__()
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")

    def _get_object_trajectories(self, frames) -> Dict[str, Dict[int, List[float]]]:
        object_positions = defaultdict(dict)
        timestamps = np.array(sorted(frame.timestamp for frame in frames), dtype=float)

        # Build object position lookup per timestamp
        for frame in frames:
            world_pose = np.array(frame.pose)
            for obj in frame.objects:
                if obj and obj.id:  # Ensure object exists and has ID
                    try:
                        centre = obj.get_centre()
                        centre_h = np.append(centre, 1.0).reshape(1, 4)
                        world_centre = np.matmul(centre_h, world_pose.T).reshape(-1)
                        world_centre = world_centre[:3]
                        position = world_centre.tolist()
                        object_positions[obj.id][frame.timestamp] = position
                    except (AttributeError, ValueError) as e:
                        pass

        # Fill in trajectories, holding the last known position
        object_trajectories = {}
        for obj_id, position_by_time in object_positions.items():

            trajectory = []
            # Use first available position as starting point
            # last_position = next(iter(position_by_time.values()))

            object_ts = np.array(
                [float(x) for x in position_by_time.keys()], dtype=float
            )

            object_positions = np.array(
                [x for x in position_by_time.values()], dtype=float
            )

            trajectory = []
            for i in range(3):
                vals = np.interp(timestamps, object_ts, object_positions[:, i])
                trajectory.append(vals)

            trajectory = np.stack(trajectory, axis=1)
            assert trajectory.shape[0] == len(timestamps)

            object_trajectories[obj_id] = trajectory

        return object_trajectories

    def _trajectory_variance(self, traj: np.ndarray) -> float:
        """
        Calculate the variance of a trajectory to determine object movement.

        Args:
            traj: Numpy array of trajectory points

        Returns:
            Float representing the variance of the trajectory
        """
        if traj is None or len(traj) < 2:
            return 0.0

        try:
            return float(np.sum(np.var(traj, axis=0)))
        except (ValueError, TypeError) as e:
            return 0.0

    def _calculate_distance_matrix(
        self, object_ids, trajectories, object_most_visible_cameras, timestamps
    ):
        """
        Calculate distance matrix between all object trajectories.

        Args:
            object_ids: List of object IDs
            trajectories: Dictionary mapping object IDs to trajectory arrays

        Returns:
            Numpy array of distances between trajectories
        """
        n_objects = len(object_ids)
        distance_matrix = np.full((n_objects, n_objects), np.inf, dtype=float)
        frame_matrix = np.full((n_objects, n_objects), 0, dtype=int)

        # Use local variables to avoid repeated dict lookups
        min_variance = self.config["min_variance"]

        # Pre-calculate motion flags to avoid redundant variance calculations
        motion_flags = {
            obj_id: self._trajectory_variance(trajectories[obj_id]) >= min_variance
            for obj_id in object_ids
        }

        # Calculate distances between trajectories
        for i in range(n_objects):
            obj_id_i = object_ids[i]

            # Skip if object doesn't have enough motion, unless we're considering stationary objects
            if (
                not motion_flags[obj_id_i]
                and not self.config["consider_stationary_objects"]
            ):
                continue

            for j in range(i + 1, n_objects):
                obj_id_j = object_ids[j]

                if (
                    object_most_visible_cameras[obj_id_j]
                    != object_most_visible_cameras[obj_id_i]
                ):
                    continue

                try:
                    min_dist = float("inf")
                    min_time = timestamps[0]
                    for pos1, pos2, timestamp in zip(
                        trajectories[obj_id_i], trajectories[obj_id_j], timestamps
                    ):
                        d = np.linalg.norm(pos1 - pos2)

                        if d < min_dist:
                            min_dist = d
                            min_time = timestamp

                    distance = min_dist

                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
                    frame_matrix[i, j] = min_time
                    frame_matrix[j, i] = min_time
                except (ValueError, TypeError) as e:
                    # Keep as infinity
                    logger.info(f"Error when creating distance matrix {e}")

        return distance_matrix, motion_flags, frame_matrix

    def _calculate_trajectory_matrix_and_distance_matrix(self, frames, timestamps):
        object_positions = defaultdict(dict)
        # Build object position lookup per timestamp
        for frame in frames:
            world_pose = np.array(frame.pose)
            for obj in frame.objects:
                if obj and obj.id:  # Ensure object exists and has ID
                    try:
                        centre = obj.get_centre()
                        centre_h = np.append(centre, 1.0).reshape(1, 4)
                        world_centre = np.matmul(centre_h, world_pose.T).reshape(-1)
                        world_centre = world_centre[:3]
                        position = world_centre
                        object_positions[obj.id][frame.timestamp] = position
                    except (AttributeError, ValueError) as e:
                        pass

        # Fill in trajectories, holding the last known position
        full_object_trajectories = {}
        for obj_id, position_by_time in object_positions.items():

            trajectory = []

            object_ts = np.array(
                [float(x) for x in position_by_time.keys()], dtype=float
            )

            object_positions_array = np.array(
                [x for x in position_by_time.values()], dtype=float
            )

            trajectory = []
            for i in range(3):
                vals = np.interp(timestamps, object_ts, object_positions_array[:, i])
                trajectory.append(vals)

            trajectory = np.stack(trajectory, axis=1)
            assert trajectory.shape[0] == len(timestamps)

            full_object_trajectories[obj_id] = trajectory

        object_ids = list(object_positions.keys())
        n_objects = len(object_ids)
        distance_matrix = np.full((n_objects, n_objects), np.inf, dtype=float)
        frame_matrix = np.full((n_objects, n_objects), 0, dtype=int)

        # Use local variables to avoid repeated dict lookups
        min_variance = self.config["min_variance"]

        # Pre-calculate motion flags to avoid redundant variance calculations
        motion_flags = {
            obj_id: self._trajectory_variance(full_object_trajectories[obj_id])
            >= min_variance
            for obj_id in object_ids
        }

        # Calculate distances between trajectories, only for those frames where the positions exist.
        for i in range(n_objects):
            obj_id_i = object_ids[i]

            # Skip if object doesn't have enough motion, unless we're considering stationary objects
            if (
                not motion_flags[obj_id_i]
                and not self.config["consider_stationary_objects"]
            ):
                continue

            obj_i_position_by_time = object_positions[obj_id_i]

            for j in range(i + 1, n_objects):
                obj_id_j = object_ids[j]
                obj_j_position_by_time = object_positions[obj_id_j]

                try:
                    min_dist = float("inf")
                    min_time = timestamps[0]
                    for timestamp in timestamps:
                        if (
                            timestamp in obj_i_position_by_time
                            and timestamp in obj_j_position_by_time
                        ):
                            pos1 = obj_i_position_by_time[timestamp]
                            pos2 = obj_j_position_by_time[timestamp]

                            d = np.linalg.norm(pos1 - pos2)

                            if d < min_dist:
                                min_dist = d
                                min_time = timestamp

                    distance = min_dist

                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
                    frame_matrix[i, j] = min_time
                    frame_matrix[j, i] = min_time
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Error calculating distance between trajectories {obj_id_i} and {obj_id_j}: {e}"
                    )
                    # Keep as infinity

        return full_object_trajectories, distance_matrix, motion_flags, frame_matrix

    def _calculate_velocity(self, trajectory):
        """Calculate velocity vector from trajectory points."""
        velocities = []
        valid_points = trajectory

        if len(valid_points) < 2:
            return np.zeros(3)

        for i in range(1, len(valid_points)):
            velocity = np.array(valid_points[i]) - np.array(valid_points[i - 1])
            velocities.append(velocity)

        # Return average velocity if we have velocities
        if velocities:
            return np.mean(velocities, axis=0)
        return np.zeros(3)

    def _is_same_direction(self, traj1, traj2, threshold=0.7):
        """
        Determine if two objects are moving in the same direction.
        """
        vel1 = self._calculate_velocity(traj1)
        vel2 = self._calculate_velocity(traj2)

        # # If either velocity is approximately zero, they're not moving in same direction
        # if np.linalg.norm(vel1) < 0.1 or np.linalg.norm(vel2) < 0.1:
        #     return False

        # Calculate cosine similarity
        cos_sim = np.dot(vel1, vel2) / (np.linalg.norm(vel1) * np.linalg.norm(vel2))
        return cos_sim > threshold

    def _calculate_speed(self, trajectory):
        """
        Calculate scalar speed from trajectory points.

        Args:
            trajectory: Numpy array of trajectory points

        Returns:
            Float representing average speed
        """
        velocity = self._calculate_velocity(trajectory)
        return np.linalg.norm(velocity)

    def _is_faster(self, traj1, traj2):
        """
        Determine if first object is moving faster than second.

        Args:
            traj1: First trajectory
            traj2: Second trajectory

        Returns:
            Boolean indicating if first object is faster and the speed difference
        """
        speed1 = self._calculate_speed(traj1)
        speed2 = self._calculate_speed(traj2)

        difference = speed1 - speed2
        return speed1 > speed2, abs(difference)

    def _is_following(self, traj1, traj2, threshold=0.7):
        """
        Determine if first object is following behind second object.

        Args:
            traj1: Trajectory of potential follower
            traj2: Trajectory of potential leader
            threshold: Cosine similarity threshold

        Returns:
            Boolean indicating if first object is following second and confidence level
        """
        valid_points1 = traj1
        valid_points2 = traj2

        if len(valid_points1) < 2 or len(valid_points2) < 2:
            return False

        # Get velocity vectors
        vel1 = self._calculate_velocity(traj1)
        vel2 = self._calculate_velocity(traj2)

        # Get last positions
        last_pos1 = valid_points1[-1]
        last_pos2 = valid_points2[-1]

        # Calculate vector from follower to leader
        follow_vector = last_pos2 - last_pos1

        # Check if directions are similar
        is_same_dir = self._is_same_direction(traj1, traj2)

        if is_same_dir:
            # Calculate if follower is behind leader (dot product of velocity and follow vector)
            dot_product = np.dot(vel1, follow_vector)

            # If dot product is positive, follower is behind leader
            is_behind = dot_product > 0

            return is_behind

        return False

    def generate(self, frames):
        """
        Generate questions about object trajectories.

        Args:
            frames: List of FrameInfo objects

        Returns:
            List of (question, answer) tuples
        """
        if not frames or len(frames) < self.config["min_trajectory_points"]:
            return []

        samples = []

        timestamps = np.array([frame.timestamp for frame in frames], dtype=int)

        trajectories, distance_matrix, motion_flags, frame_matrix = (
            self._calculate_trajectory_matrix_and_distance_matrix(frames, timestamps)
        )

        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

        # Get object IDs from trajectories
        object_ids = list(trajectories.keys())
        if not object_ids:
            logger.warning("No object IDs found")
            return []

        # Generate questions
        for i, object_id in enumerate(object_ids):

            # Get object's motion type (moving or stationary)
            is_moving = motion_flags[object_id]

            # Find similar trajectories based on distance
            distances = distance_matrix[i].copy()
            matched_times = frame_matrix[i].copy()
            distances[i] = np.inf  # Avoid self-match

            # Find closest trajectory
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            min_timestamp = matched_times[min_idx]

            # Skip if no close trajectories found
            if min_distance >= self.config["distance_threshold"]:
                continue

            # get the frame
            frame = None
            frame_idx = 0
            for idx, f in enumerate(frames):
                if f.timestamp == min_timestamp:
                    frame = f
                    frame_idx = idx
                    break

            # make sure frame is not none
            if frame is None:
                continue

            # Get other object ID
            other_object_id = object_ids[min_idx]

            try:
                obj = next(obj for obj in frame.objects if obj.id == object_id)
                other_obj = next(
                    obj for obj in frame.objects if obj.id == other_object_id
                )
            except StopIteration:
                continue

            # only one can be a sign
            if any(o.type == "TYPE_SIGN" for o in [obj, other_obj]):
                continue

            # don't use objects with no cvat label (as will use vague waymo label)
            # allow for one to have no cvat label
            if all(x.cvat_label is None for x in [obj, other_obj]):
                continue

            timestamp_idx = timestamps[frame_idx]
            assert frame.timestamp == timestamps[frame_idx]

            obj1_centre = obj.get_centre().reshape(1, 3)
            obj2_centre = other_obj.get_centre().reshape(1, 3)
            centres = np.concatenate((obj1_centre, obj2_centre), axis=0)
            cam_distances = np.full((len(frame.cameras),), fill_value=np.inf)

            for cam_idx, camera in enumerate(frame.cameras):
                uvdok = camera.project_to_image(centres, frame)
                ok = uvdok[..., -1].astype(bool)
                depth = uvdok[..., 2]

                mask = ok & (depth > 0)
                uv = uvdok[mask, :2]

                if uv.shape[0] != 2:
                    continue  # not all on this image

                if (
                    any(uv[:, 0] > camera.width)
                    or any(uv[:, 0] < 0)
                    or any(uv[:, 1] > camera.height)
                    or any(uv[:, 1] < 0)
                ):
                    continue

                cam_centre = np.array([camera.width / 2, camera.height / 2]).reshape(
                    1, 2
                )
                distances = np.linalg.norm(uv - cam_centre, axis=1)

                cam_distances[cam_idx] = distances.mean()

            # for single questions / for object description
            print("cam_distances", cam_distances)
            best_cam_idx = np.argmin(cam_distances)
            best_camera = frame.cameras[best_cam_idx]

            object_description = obj.get_detailed_object_description(frame, best_camera)
            other_object_description = obj.get_detailed_object_description(
                frame, best_camera
            )

            # too far away on the image
            if cam_distances[best_cam_idx] > self.config["max_camera_distance"]:
                continue

            questions_list = []
            answers_list = []
            choices_list = []

            # Determine which window to use based on frame_idx
            num_frames = len(frames)

            # Case 1: Near the beginning of the sequence - use future window
            if frame_idx < 10:
                window_direction = "future"
                object1_windowed_traj = trajectories[object_id][
                    frame_idx : (frame_idx + 10)
                ]
                object2_windowed_traj = trajectories[other_object_id][
                    frame_idx : (frame_idx + 10)
                ]
                time_text = "in the next second"
            # Case 2: Near the end of the sequence - use past window
            elif frame_idx >= num_frames - 10:
                window_direction = "past"
                object1_windowed_traj = trajectories[object_id][
                    (frame_idx - 10) : frame_idx
                ]
                object2_windowed_traj = trajectories[other_object_id][
                    (frame_idx - 10) : frame_idx
                ]
                time_text = "in the past second"
            # Case 3: Middle of the sequence - randomly choose past or future
            else:
                # Use a deterministic approach based on frame_idx to decide
                window_direction = "future" if frame_idx % 2 == 0 else "past"

                if window_direction == "future":
                    object1_windowed_traj = trajectories[object_id][
                        frame_idx : (frame_idx + 10)
                    ]
                    object2_windowed_traj = trajectories[other_object_id][
                        frame_idx : (frame_idx + 10)
                    ]
                    time_text = "in the next second"
                else:
                    object1_windowed_traj = trajectories[object_id][
                        (frame_idx - 10) : frame_idx
                    ]
                    object2_windowed_traj = trajectories[other_object_id][
                        (frame_idx - 10) : frame_idx
                    ]
                    time_text = "in the past second"

            # Adjust verb tense based on window direction
            if window_direction == "future":
                verb = "Will"
                verb_be = "be"
            else:
                verb = "Was"
                verb_be = ""

            # Generate questions with appropriate tense
            faster = self._is_faster(object1_windowed_traj, object2_windowed_traj)
            questions_list.append(  # TODO: add trajectory color and draw on image?
                f"{verb} {object_description} {verb_be} moving faster than {other_object_description} {time_text}?",
            )
            answers_list.append("Yes" if faster else "No")
            choices_list.append(["Yes", "No"])

            same_dir = self._is_same_direction(
                object1_windowed_traj, object2_windowed_traj
            )
            questions_list.append(
                f"{verb} {object_description} {verb_be} moving in the same direction as {other_object_description} {time_text}?",
            )
            answers_list.append("Yes" if same_dir else "No")
            choices_list.append(["Yes", "No"])

            following = self._is_following(object1_windowed_traj, object2_windowed_traj)
            questions_list.append(
                f"{verb} {object_description} {verb_be} following {other_object_description} {time_text}?",
            )
            answers_list.append("Yes" if following else "No")
            choices_list.append(["Yes", "No"])

            # Prepare image paths and camera names
            image_paths = [camera.image_path for camera in frame.cameras]
            camera_names = [camera.name for camera in frame.cameras]

            for question_text, answer_txt, choices in zip(
                questions_list, answers_list, choices_list
            ):
                # Create question object
                question = MultipleImageMultipleChoiceQuestion(
                    image_paths=image_paths,
                    question=question_text,
                    scene_id=obj.scene_id,
                    timestamp=frame.timestamp,
                    camera_names=camera_names,
                    choices=choices,
                    generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                    data={
                        "object_id": object_id,
                        "other_object_id": other_object_id,
                        "best_camera_name": best_camera.name,
                    },
                )

                # Create answer object
                answer = MultipleChoiceAnswer(
                    choices=choices,
                    answer=answer_txt,
                )

                # Add to samples
                samples.append((question, answer))

        return samples

    def visualize_trajectories(
        self,
        object_id,
        other_object_id,
        trajectories,
        frame,
        camera,
        timestamps: np.ndarray,
        save_path=None,
        text="",
    ):
        """
        Visualize object trajectories for debugging purposes.

        Creates a figure with two subplots:
        1. 3D plot showing the trajectories in world coordinates
        2. 2D plot showing the trajectories projected onto a camera image

        Args:
            object_id: ID of the main object
            other_object_id: ID of the object with similar trajectory
            trajectories: Dictionary mapping object IDs to trajectory arrays
            frame: frameinfo chosen
            timestamps: timestamps
            save_path: Optional path to save the figure, if None the figure is displayed

        Returns:
            matplotlib.figure.Figure object
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.image as mpimg
        import os
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np

        # frame, camera, obj = best_object_views[object_id]
        obj = next(obj for obj in frame.objects if obj.id == object_id)
        other_obj = next(obj for obj in frame.objects if obj.id == other_object_id)

        timestamp_idx = next(
            idx for idx, time in enumerate(timestamps) if time == frame.timestamp
        )

        # Get frame and camera for visualization
        # Create figure with two subplots
        fig = plt.figure(figsize=(16, 8))

        # Get trajectories
        traj1 = trajectories[object_id]
        traj2 = trajectories[other_object_id]

        # Get object descriptions for labels
        obj1_desc = obj.get_object_description()
        obj2_desc = other_obj.get_object_description()

        # 1. 3D plot in world coordinates
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")

        # Create custom colormaps for fading effect
        n_points = len(traj1)
        colors1 = plt.cm.viridis(np.linspace(0, 1, n_points))
        colors2 = plt.cm.plasma(np.linspace(0, 1, n_points))

        # Plot trajectories with fading colors to show time progression
        for i in range(len(traj1) - 1):
            # Object 1 trajectory
            ax1.plot(
                traj1[i : i + 2, 0],
                traj1[i : i + 2, 1],
                traj1[i : i + 2, 2] if traj1.shape[1] > 2 else [0, 0],
                color=colors1[i],
                linewidth=2,
            )

            # Object 2 trajectory
            ax1.plot(
                traj2[i : i + 2, 0],
                traj2[i : i + 2, 1],
                traj2[i : i + 2, 2] if traj2.shape[1] > 2 else [0, 0],
                color=colors2[i],
                linewidth=2,
            )

        # Add start and end markers
        ax1.scatter(
            traj1[0, 0],
            traj1[0, 1],
            traj1[0, 2] if traj1.shape[1] > 2 else 0,
            color="green",
            s=100,
            marker="o",
            label=f"{obj1_desc} (start)",
        )
        ax1.scatter(
            traj1[timestamp_idx, 0],
            traj1[timestamp_idx, 1],
            traj1[timestamp_idx, 2] if traj1.shape[1] > 2 else 0,
            color="blue",
            s=100,
            marker="v",
            label=f"{obj1_desc} (current)",
        )
        ax1.scatter(
            traj1[-1, 0],
            traj1[-1, 1],
            traj1[-1, 2] if traj1.shape[1] > 2 else 0,
            color="darkgreen",
            s=100,
            marker="s",
            label=f"{obj1_desc} (end)",
        )

        ax1.scatter(
            traj2[0, 0],
            traj2[0, 1],
            traj2[0, 2] if traj2.shape[1] > 2 else 0,
            color="red",
            s=100,
            marker="o",
            label=f"{obj2_desc} (start)",
        )
        ax1.scatter(
            traj2[timestamp_idx, 0],
            traj2[timestamp_idx, 1],
            traj2[timestamp_idx, 2] if traj1.shape[1] > 2 else 0,
            color="red",
            s=100,
            marker="v",
            label=f"{obj1_desc} (current)",
        )
        ax1.scatter(
            traj2[-1, 0],
            traj2[-1, 1],
            traj2[-1, 2] if traj2.shape[1] > 2 else 0,
            color="darkred",
            s=100,
            marker="s",
            label=f"{obj2_desc} (end)",
        )

        # Set labels and title
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Object Trajectories in 3D Space")
        ax1.legend()

        # Equal aspect ratio for all axes
        max_range = max(
            [
                np.max([np.ptp(traj1[:, 0]), np.ptp(traj2[:, 0])]),
                np.max([np.ptp(traj1[:, 1]), np.ptp(traj2[:, 1])]),
                np.max(
                    [
                        np.ptp(
                            traj1[:, 2] if traj1.shape[1] > 2 else np.zeros(len(traj1))
                        ),
                        np.ptp(
                            traj2[:, 2] if traj2.shape[1] > 2 else np.zeros(len(traj2))
                        ),
                    ]
                ),
            ]
        )

        mid_x = (np.mean(traj1[:, 0]) + np.mean(traj2[:, 0])) / 2
        mid_y = (np.mean(traj1[:, 1]) + np.mean(traj2[:, 1])) / 2
        mid_z = (
            np.mean(traj1[:, 2] if traj1.shape[1] > 2 else np.zeros(len(traj1)))
            + np.mean(traj2[:, 2] if traj2.shape[1] > 2 else np.zeros(len(traj2)))
        ) / 2

        ax1.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax1.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax1.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        # 2. Image projection plot
        ax2 = fig.add_subplot(1, 2, 2)

        # Load the camera image
        image_path = camera.image_path
        try:
            if os.path.exists(image_path):
                img = mpimg.imread(image_path)
                ax2.imshow(img)
            else:
                # If image doesn't exist, create a blank image
                ax2.set_facecolor("black")
                print(f"Warning: Image file not found at {image_path}")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            ax2.set_facecolor("black")

        # Project 3D trajectories onto the image
        try:
            pose_matrix_inv = np.linalg.inv(np.array(frame.pose))

            traj1_ego = traj1.copy().reshape(-1, 3)
            traj1_ego = np.hstack([traj1_ego, np.ones((traj1_ego.shape[0], 1))])
            traj1_ego = np.matmul(traj1_ego, pose_matrix_inv.T)
            traj1_ego = traj1_ego[:, :3]

            traj2_ego = traj2.copy().reshape(-1, 3)
            traj2_ego = np.hstack([traj2_ego, np.ones((traj2_ego.shape[0], 1))])
            traj2_ego = np.matmul(traj2_ego, pose_matrix_inv.T)
            traj2_ego = traj2_ego[:, :3]

            # Project to image
            projected_points1 = camera.project_to_image(traj1_ego, frame)
            projected_points2 = camera.project_to_image(traj2_ego, frame)

            # Extract x, y coordinates (and depth if available)
            if projected_points1.shape[1] > 2:  # If depth is returned
                img_points1 = projected_points1[:, :2]
                valid_mask1 = projected_points1[:, -1].astype(bool)
                img_points2 = projected_points2[:, :2]
                valid_mask2 = projected_points2[:, -1].astype(bool)

                # img_points1 = img_points1[valid_mask1]
                # img_points2 = img_points2[valid_mask2]

                # Adjust colors based on depth
                colors1 = plt.cm.viridis(np.linspace(0, 1, len(img_points1)))
                colors2 = plt.cm.plasma(np.linspace(0, 1, len(img_points2)))
            else:
                img_points1 = projected_points1
                img_points2 = projected_points2

            # Plot trajectory points on image
            for i in range(len(img_points1) - 1):
                ax2.plot(
                    img_points1[i : i + 2, 0],
                    img_points1[i : i + 2, 1],
                    color=colors1[i],
                    linewidth=2,
                    alpha=0.8,
                )

            for i in range(len(img_points2) - 1):
                ax2.plot(
                    img_points2[i : i + 2, 0],
                    img_points2[i : i + 2, 1],
                    color=colors2[i],
                    linewidth=2,
                    alpha=0.8,
                )

            # Add start and end markers
            if len(img_points1) > 0:
                ax2.scatter(
                    img_points1[0, 0],
                    img_points1[0, 1],
                    color="green",
                    s=100,
                    marker="o",
                    label=f"{obj1_desc} (start)",
                )
                ax2.scatter(
                    img_points1[timestamp_idx, 0],
                    img_points1[timestamp_idx, 1],
                    color="green",
                    s=100,
                    marker="v",
                    label=f"{obj1_desc} (current)",
                )
                ax2.scatter(
                    img_points1[-1, 0],
                    img_points1[-1, 1],
                    color="darkgreen",
                    s=100,
                    marker="s",
                    label=f"{obj1_desc} (end)",
                )

            if len(img_points2) > 0:
                ax2.scatter(
                    img_points2[0, 0],
                    img_points2[0, 1],
                    color="red",
                    s=100,
                    marker="o",
                    label=f"{obj2_desc} (start)",
                )
                ax2.scatter(
                    img_points2[timestamp_idx, 0],
                    img_points2[timestamp_idx, 1],
                    color="red",
                    s=100,
                    marker="v",
                    label=f"{obj2_desc} (current)",
                )
                ax2.scatter(
                    img_points2[-1, 0],
                    img_points2[-1, 1],
                    color="darkred",
                    s=100,
                    marker="s",
                    label=f"{obj2_desc} (end)",
                )

        except Exception as e:
            print(f"Error projecting points to image: {e}")

        # Set title and legend
        ax2.set_title(f"Trajectories Projected on {camera.name} Camera")
        ax2.legend(loc="upper right")

        # Set axes limits to match image dimensions if available
        if hasattr(camera, "width") and hasattr(camera, "height"):
            ax2.set_xlim(0, camera.width)
            ax2.set_ylim(camera.height, 0)  # Inverted y-axis for image coordinates

        try:
            distance, _ = fastdtw(traj1, traj2, dist=euclidean)
            plt.figtext(
                0.5,
                0.01,
                f"DTW Distance: {distance:.2f}",
                ha="center",
                fontsize=12,
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
            )
        except Exception as e:
            print(f"Error calculating DTW distance: {e}")

        # Add metadata as text
        info_text = (
            f"Scene ID: {obj.scene_id}\n"
            f"Timestamp: {frame.timestamp}\n"
            f"Object 1: {obj1_desc}\n"
            f"Object 2: {obj2_desc}\n"
            f"Camera: {camera.name}\n"
            f"Question: {text}"
        )

        plt.figtext(
            0.01,
            0.01,
            info_text,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 5},
        )

        # Adjust layout
        plt.tight_layout()

        # Save or show figure
        if save_path:
            plt.savefig(
                # save_path.replace(".png", f"_{view_type}.png"),
                # save_path.replace(".png", f"_{camera.name}.png"),
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()

    def visualise_sample(
        self,
        question_obj: MultipleImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
        figsize=(12, 8),
        box_color="green",
        text_fontsize=12,
        title_fontsize=14,
        dpi=150,
    ):
        """
        Visualises trajectories etc.

        Args:
            question_obj: Question object with image_path and question text
            answer_obj: MultipleChoiceAnswer
            save_path: Path to save the visualization
            frames:
            figsize: Figure size (width, height)
            box_color: Color for bounding boxes
            text_fontsize: Font size for question/answer text
            title_fontsize: Font size for the title
            dpi: DPI for saved image
        """
        if question_obj.data is None:
            raise ValueError("question_obj.data should not be none")

        object_id = question_obj.data["object_id"]
        other_object_id = question_obj.data["other_object_id"]
        best_camera_name = question_obj.data["best_camera_name"]

        timestamps = np.array([frame.timestamp for frame in frames], dtype=int)

        trajectories, distance_matrix, motion_flags, frame_matrix = (
            self._calculate_trajectory_matrix_and_distance_matrix(frames, timestamps)
        )

        frame = next(
            frame for frame in frames if frame.timestamp == question_obj.timestamp
        )

        best_camera = next(
            camera for camera in frame.cameras if camera.name == best_camera_name
        )

        question_text = question_obj.question
        answer_txt = answer_obj.answer

        self.visualize_trajectories(
            object_id,
            other_object_id,
            trajectories,
            frame,
            best_camera,
            timestamps,
            save_path=save_path,
            text=question_text + answer_txt,
        )

    def get_question_type(self):
        """Return the type of questions generated by this generator."""
        return MultipleImageMultipleChoiceQuestion

    def get_answer_type(self):
        """Return the type of answers generated by this generator."""
        return MultipleChoiceAnswer

    def get_metric_class(self):
        """Return the metric class used to evaluate answers to these questions."""
        return MultipleChoiceMetric


class TrajectoryQuestionStrategy:
    """Base class for different trajectory question strategies"""

    def generate_question(self, obj1, obj2, context):
        """Generate a question based on two objects"""
        raise NotImplementedError

    def can_apply(self, obj1, obj2, trajectories, motion_flags):
        """Check if this strategy can be applied to these objects"""
        raise NotImplementedError
