import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple
from datetime import datetime
import cv2
import pytz
import numpy as np
from collections import defaultdict
import logging

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist


from waymovqa.data import *
from waymovqa.data.camera_info import CameraInfo
from waymovqa.data.scene_info import SceneInfo
from waymovqa.data.frame_info import FrameInfo, TimeOfDayType, WeatherType, LocationType
from waymovqa.data.object_info import (
    OBJECT_SPEED_CATS,
    OBJECT_SPEED_THRESH,
    WAYMO_TYPE_MAPPING,
    HeadingType,
    MovementType,
    ObjectInfo,
)
from waymovqa.prompt_generators import BasePromptGenerator, register_prompt_generator

from waymovqa.questions.multi_frame_multi_choice import (
    MultipleFrameMultipleChoiceQuestion,
)
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
from waymovqa.primitives import CHOICES_OPTIONS


# Default config values that can be overridden
DEFAULT_CONFIG = {
    "distance_threshold": 30.0,  # Threshold for similar trajectories
    "min_variance": 0.5,  # Minimum velocity for moving objects of 0.5 metres/second
    "min_trajectory_points": 3,  # Minimum number of points needed for a valid trajectory
    "max_objects_per_scene": 10,  # Maximum objects to consider per scene
    "consider_stationary_objects": True,  # Whether to include stationary objects in some questions
    "max_camera_distance": 800,  # pixels between query objects
}
# Constants for trajectory analysis
MOVEMENT_THRESHOLDS = {
    "min_object_speed": 1.0,
    "speed_multiplier": 0.8,
    "convergence_threshold": 0.3,  # Lowered - more lenient for approaching
    "divergence_threshold": -0.3,  # Separate threshold for diverging
    "heading_threshold": 0.3,
    "divergence_heading_threshold": -0.1,
    "min_lidar_points": 10,
    "consistency_frames": 3,
    "min_distance": 2.0,
}

SCENARIO_THRESHOLDS = {
    "head_on_convergence": -0.95,
    "head_on_heading": 0.95,
    "strong_heading": 0.6,
    "weak_heading": 0.3,
    "parallel_threshold": 0.7,
    "divergence_threshold": -0.5,
}

DISTANCE_BOUNDS = {
    "Pedestrian": (2.0, 10.0),
    "Cyclist": (2.0, 15.0),
    "Vehicle": (2.0, 25.0),
}

SCENARIO_TEMPLATES = {
    "head_on": [
        "Are the ego vehicle and the {} traveling directly towards each other?",
        "Is the {} on a direct collision course with the ego vehicle?",
        "Are the ego vehicle and the {} moving head-on towards each other?",
    ],
    "object_approaching": [
        "Is the {} moving towards the ego vehicle?",
        "Is the {} approaching the ego vehicle?",
        "Is the {} heading towards the ego vehicle?",
    ],
    "ego_approaching": [
        "Is the ego vehicle moving towards the {}?",
        "Is the ego vehicle approaching the {}?",
        "Is the ego vehicle heading towards the {}?",
    ],
    "general_convergence": [
        "Are the ego vehicle and the {} getting closer to each other?",
        "Is the distance between the ego vehicle and the {} decreasing?",
        "Are the ego vehicle and the {} converging?",
    ],
    "parallel_diverging": [
        "Are the ego vehicle and the {} moving in the same direction away from each other?",
        "Is the {} moving away from the ego vehicle in the same direction?",
    ],
    "object_diverging": [
        "Is the {} moving away from the ego vehicle?",
        "Is the {} heading away from the ego vehicle?",
    ],
    "ego_diverging": [
        "Is the ego vehicle moving away from the {}?",
        "Is the ego vehicle heading away from the {}?",
    ],
    "general_divergence": [
        "Are the ego vehicle and the {} moving apart?",
        "Is the distance between the ego vehicle and the {} increasing?",
    ],
}


HZ = 10.0
FRAME_DELTA = 1.0 / HZ


@register_prompt_generator
class EgoRelativeObjectTrajectoryPromptGenerator(BasePromptGenerator):
    """
    Generates questions about object trajectories and their relationships to the ego vehicle.
    """

    negative_sample_ratio: float = 0.5

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

        random.seed(42)

        self.prompt_functions = [
            self._prompt_faster_than_ego,
            self._prompt_moving_towards_ego,
            self._prompt_parallel_motion,
            self._prompt_approaching_stop_sign,
            self._prompt_vehicle_future_path,
            # self._prompt_ego_following,
        ]

    def _balance_neg_pos(
        self, negative_samples: List[Dict], positive_samples: List[Dict]
    ):
        """Balance the negative and positive samples according to the target ratio."""
        num_neg = len(negative_samples)
        num_pos = len(positive_samples)

        if num_neg == 0 and num_pos == 0:
            return []

        if num_neg == 0:
            return random.choice(positive_samples)  # Can't balance without negatives

        if num_pos == 0:
            return random.choice(negative_samples)  # Can't balance without positives

        # Calculate the maximum total samples we can have while maintaining the ratio
        # Given the constraints of available samples

        # If we use all negatives, how many positives do we need?
        pos_needed_for_all_neg = int(
            num_neg * (1.0 - self.negative_sample_ratio) / self.negative_sample_ratio
        )

        # If we use all positives, how many negatives do we need?
        neg_needed_for_all_pos = int(
            num_pos * self.negative_sample_ratio / (1.0 - self.negative_sample_ratio)
        )

        # Choose the scenario that maximizes total samples while respecting constraints
        if pos_needed_for_all_neg <= num_pos:
            # We can use all negatives and sample the required positives
            final_negatives = negative_samples
            final_positives = random.sample(positive_samples, pos_needed_for_all_neg)
            print(
                f"Using all {num_neg} negatives, sampling {pos_needed_for_all_neg} positives from {num_pos} available"
            )

        elif neg_needed_for_all_pos <= num_neg:
            # We can use all positives and sample the required negatives
            final_positives = positive_samples
            final_negatives = random.sample(negative_samples, neg_needed_for_all_pos)
            print(
                f"Using all {num_pos} positives, sampling {neg_needed_for_all_pos} negatives from {num_neg} available"
            )

        else:
            # Neither scenario works perfectly, so we need to find the best balance
            # Calculate how many of each we can use while maintaining the ratio

            # Try to maximize total samples
            max_total_with_neg_constraint = num_neg / self.negative_sample_ratio
            max_total_with_pos_constraint = num_pos / (1.0 - self.negative_sample_ratio)

            # Use the smaller total (the limiting constraint)
            target_total = min(
                max_total_with_neg_constraint, max_total_with_pos_constraint
            )

            target_neg = int(target_total * self.negative_sample_ratio)
            target_pos = int(target_total * (1.0 - self.negative_sample_ratio))

            # Ensure we don't exceed available samples
            target_neg = min(target_neg, num_neg)
            target_pos = min(target_pos, num_pos)

            final_negatives = random.sample(negative_samples, target_neg)
            final_positives = random.sample(positive_samples, target_pos)
            print(
                f"Balanced sampling: {target_neg} negatives from {num_neg}, {target_pos} positives from {num_pos}"
            )

        # Combine and shuffle the results
        result = final_negatives + final_positives
        random.shuffle(result)

        actual_ratio = len(final_negatives) / len(result) if result else 0
        print(
            f"Final ratio: {actual_ratio:.3f} (target: {self.negative_sample_ratio:.3f})"
        )

        return result

    def _prompt_parallel_motion(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
        angle_threshold=20,  # degrees for parallel/perpendicular classification
        min_speed=2.0,  # minimum speed to consider motion meaningful
    ):
        results = []
        choices = ["parallel", "perpendicular"]

        templates = [
            "Is the {} moving parallel or perpendicular to the ego vehicle?",
            "What is the relative motion direction of the {} compared to ego?",
        ]

        for obj_id in trajectories.keys():
            if obj_types[obj_id] not in ["Vehicle", "Pedestrian", "Cyclist"]:
                continue
            obj_trajectory = trajectories[obj_id]
            obj_speeds = speeds[obj_id]

            # Calculate relative distances over time
            relative_positions = obj_trajectory - ego_positions
            distances = np.linalg.norm(relative_positions, axis=1)

            template = random.choice(templates)

            # Find valid timestamps (close enough, both moving)
            valid_mask = (
                (distances < DEFAULT_CONFIG["distance_threshold"])
                & (obj_speeds > min_speed)
                & (ego_speeds > min_speed)
            )

            valid_indices = np.where(valid_mask)[0]
            # Need at least 2 points to calculate direction
            valid_indices = valid_indices[valid_indices < len(timestamps) - 1]

            if len(valid_indices) == 0:
                continue

            best_results = {"parallel": [], "perpendicular": []}

            for i in valid_indices:
                # Calculate ego motion vector
                if i + 1 < len(ego_positions):
                    ego_motion = ego_positions[i + 1] - ego_positions[i]
                else:
                    ego_motion = ego_vectors[i]  # Use velocity vector if at end

                # Calculate object motion vector
                if i + 1 < len(obj_trajectory):
                    obj_motion = obj_trajectory[i + 1] - obj_trajectory[i]
                else:
                    continue

                # Skip if either vector is too small
                if (
                    np.linalg.norm(ego_motion) < 0.01
                    or np.linalg.norm(obj_motion) < 0.01
                ):
                    continue

                # Calculate angle between motion vectors
                cos_angle = np.dot(ego_motion, obj_motion) / (
                    np.linalg.norm(ego_motion) * np.linalg.norm(obj_motion)
                )
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                # Handle obtuse angles (consider absolute angle for parallel/perpendicular)
                angle_deg = min(angle_deg, 180 - angle_deg)

                obj = next((obj for obj in frames[i].objects if obj.id == obj_id), None)
                if obj is None or obj.cvat_label is None:
                    continue

                if angle_deg < angle_threshold:  # Parallel motion
                    best_results["parallel"].append(
                        {
                            "object_id": obj_id,
                            "timestamp": int(timestamps[i]),
                            "question": template.format(obj.get_object_description()),
                            "answer_description": f"parallel motion at {angle_deg:.1f}° angle",
                            "answer": "parallel",
                            "confidence": float(
                                (angle_threshold - angle_deg) / angle_threshold
                            ),
                        }
                    )
                elif angle_deg > (90 - angle_threshold):  # Perpendicular motion
                    best_results["perpendicular"].append(
                        {
                            "object_id": obj_id,
                            "timestamp": int(timestamps[i]),
                            "question": template.format(obj.get_object_description()),
                            "answer_description": f"perpendicular motion at {angle_deg:.1f}° angle",
                            "answer": "perpendicular",
                            "confidence": float(
                                (angle_deg - (90 - angle_threshold)) / angle_threshold
                            ),
                        }
                    )

            # Add best result for each category if available
            for motion_type, motion_results in best_results.items():
                if len(motion_results) > 0:
                    best_result = sorted(
                        motion_results, key=lambda x: x["confidence"], reverse=True
                    )[0]
                    results.append(best_result)

        return choices, results

    # def _prompt_ego_following(
    #     self,
    #     trajectories,
    #     speeds,
    #     accels,
    #     ego_positions,
    #     ego_speeds,
    #     ego_vectors,
    #     timestamps,
    #     frames,
    #     obj_types,
    #     obj_cvat_types,
    #     max_variance=5.0,
    #     min_trajectory_length=10,
    #     min_speed=5.0,
    #     distance_threshold=15.0,
    #     velocity_alignment_thresh=0.7
    # ):
    #     results = []
    #     choices = ["yes", "no"]

    #     templates = [
    #         "Is the ego vehicle following the {}?",
    #         "Is the ego vehicle behind and following the {}?",
    #     ]

    #     ego_velocity = np.gradient(ego_positions, axis=0) / FRAME_DELTA
    #     ego_direction = ego_velocity / (
    #         np.linalg.norm(ego_velocity, axis=1, keepdims=True) + 1e-8
    #     )

    #     obj_samples = defaultdict(list)
    #     objects_followed_per_frame = [[] for _ in timestamps]
    #     all_descriptions = set()

    #     for obj_id in trajectories.keys():
    #         if obj_types[obj_id] not in ["Vehicle", "Pedestrian", "Cyclist"]:
    #             continue
    #         obj_trajectory = trajectories[obj_id]
    #         obj_speeds = speeds[obj_id]

    #         # Calculate relative distances over time
    #         relative_positions = obj_trajectory - ego_positions
    #         distances = np.linalg.norm(relative_positions, axis=1)

    #         # if np.var(distances) > max_variance:
    #         #     continue

    #         obj_velocity = np.gradient(obj_trajectory, axis=0) / FRAME_DELTA

    #         # Calculate direction vectors
    #         obj_direction = obj_velocity / (
    #             np.linalg.norm(obj_velocity, axis=1, keepdims=True) + 1e-8
    #         )

    #         # must not be too far away
    #         distance_mask = distances < distance_threshold

    #         # velocity alignment
    #         velocity_alignment = np.sum(obj_direction * ego_direction, axis=1)
    #         print('velocity_alignment', velocity_alignment.min(), velocity_alignment.max())

    #         velocity_aligned_mask = velocity_alignment > velocity_alignment_thresh

    #         template = random.choice(templates)

    #         # Filter for moving segments
    #         moving_mask = (obj_speeds > min_speed) & (ego_speeds > min_speed)

    #         positive_mask = distance_mask & moving_mask & velocity_aligned_mask

    #         positive_idx = np.where(positive_mask)[0]

    #         for frame_idx in positive_idx:
    #             obj = next(
    #                 (
    #                     obj
    #                     for obj in frames[frame_idx].objects
    #                     if obj.id == obj_id
    #                 ),
    #                 None,
    #             )

    #             if obj is None:
    #                 continue

    #             res = {
    #                 "object_id": obj_id,
    #                 "timestamp": int(timestamps[frame_idx]),
    #                 "question": template.format(obj.get_object_description()),
    #                 "answer_description": f"following similar path (velocity_alignment: {velocity_alignment[frame_idx]:.2f}, distance: {distances[frame_idx]:.2f}, dist variance={np.var(distances):.2f})",
    #                 "answer": "yes",
    #             }

    #             obj_samples[obj_id].append(res)
    #             objects_followed_per_frame[frame_idx].append(obj.get_object_description())
    #             all_descriptions.add(obj.get_object_description())

    #     neg_samples = []
    #     for frame_idx, timestamp in enumerate(timestamps):
    #         for obj in frames[frame_idx].objects:
    #             if obj_types[obj_id] not in ["Vehicle", "Pedestrian", "Cyclist"]:
    #                 continue
    #             description = obj.get_object_description()
    #             if description in objects_followed_per_frame[frame_idx]:
    #                 continue

    #             template = random.choice(templates)

    #             res = {
    #                 "object_id": obj.id,
    #                 "timestamp": int(timestamp),
    #                 "question": template.format(description),
    #                 "answer_description": f"mined negative sample",
    #                 "answer": "no",
    #             }
    #             neg_samples.append(res)

    #     pos_samples = []
    #     for obj_id, samples in obj_samples.items():
    #         print(f'{obj_id} -> {len(samples)} samples')
    #         pos_samples.append(random.choice(samples))

    #     print('neg_samples, pos_samples', len(neg_samples), len(pos_samples))

    #     if len(pos_samples) > 0:
    #         results = self._balance_neg_pos(neg_samples, pos_samples)
    #     else:
    #         results = [random.choice(neg_samples)]

    #     return choices, results

    def _prompt_approaching_stop_sign(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
    ):
        results = []
        choices = ["yes", "no"]

        # Constants for clarity
        APPROACH_THRESHOLD_MULTIPLIER = -1.0
        DEPARTURE_THRESHOLD_MULTIPLIER = 1.0
        MAX_CONFIDENCE_SPEED = (
            OBJECT_SPEED_THRESH["TYPE_VEHICLE"] * 2.0
        )  # Cap confidence calculation
        MIN_MEANINGFUL_SPEED = (
            0.5  # m/s - minimum speed to consider meaningful approach/departure
        )

        templates = [
            "Is the ego vehicle approaching a stop sign?",
            "Is there a stop sign ahead that the ego vehicle is approaching?",
            "Is the ego vehicle getting closer to a stop sign?",
            "Is there a stop sign that the ego vehicle is moving towards?",
            "Is the ego vehicle heading towards a stop sign?",
            "Is the ego vehicle nearing a stop sign?",
            "Can you see a stop sign that the ego vehicle is approaching?",
        ]

        def _is_heading_towards_sign(
            ego_pos: np.ndarray, ego_vector: np.ndarray, sign_pos: np.ndarray
        ) -> bool:
            """Check if ego vehicle is actually heading towards the sign (not just getting closer)."""
            if np.linalg.norm(ego_vector) < 0.1:  # Very slow or stationary
                return False

            # Vector from ego to sign
            to_sign = sign_pos - ego_pos
            to_sign_norm = to_sign / (np.linalg.norm(to_sign) + 1e-8)
            ego_vector_norm = ego_vector / (np.linalg.norm(ego_vector) + 1e-8)

            # Dot product to check alignment (> 0.5 means roughly same direction)
            alignment = np.dot(ego_vector_norm, to_sign_norm)
            return alignment > 0.5

        def _handle_sample(
            frame_idx: int,
            object_id: str,
            distance: float,
            relative_speed: float,
            positive: bool,
        ):
            """Handles generating a question for a positive/negative situation."""
            frame = frames[frame_idx]

            obj = next((obj for obj in frame.objects if obj.id == object_id), None)
            if obj is None:
                return None

            # Find any camera where the sign is visible (removed facing direction requirement)
            cam = next(
                (cam for cam in frame.cameras if obj.is_visible_on_camera(frame, cam)),
                None,
            )
            if cam is None:
                return None

            # Skip samples with very low speeds (not meaningful)
            if abs(relative_speed) < MIN_MEANINGFUL_SPEED:
                return None

            # For positive samples, verify ego is actually heading towards the sign
            if positive:
                ego_pos = ego_positions[frame_idx]
                ego_vector = (
                    ego_vectors[frame_idx]
                    if len(ego_vectors) > frame_idx
                    else np.zeros(3)
                )
                sign_pos = trajectories[object_id][frame_idx]

                if not _is_heading_towards_sign(ego_pos, ego_vector, sign_pos):
                    return None

                if obj.get_camera_heading_direction(frame, cam) != HeadingType.TOWARDS:
                    return None

            template = random.choice(templates)
            speed_magnitude = abs(relative_speed)

            if positive:
                answer_description = f"approaching stop sign at {speed_magnitude:.2f} m/s, currently {distance:.2f} metres away."
            else:
                answer_description = f"driving away from stop sign at {speed_magnitude:.2f} m/s, currently {distance:.2f} metres away."

            return {
                "object_id": object_id,
                "positive": positive,
                "timestamp": int(frame.timestamp),
                "question": template,
                "answer_description": answer_description,
                "answer": "yes" if positive else "no",
                "metadata": {
                    "distance": distance,
                    "relative_speed": relative_speed,
                    "speed_magnitude": speed_magnitude,
                    "camera_name": cam.name,
                },
            }

        positive_samples = []
        negative_samples = []

        # Find any stop signs
        stop_sign_object_ids = [
            object_id
            for object_id, cvat_label in obj_cvat_types.items()
            if cvat_label == "Stop Sign"
        ]

        if not stop_sign_object_ids:
            return choices, results  # No stop signs found

        for obj_id in stop_sign_object_ids:
            if obj_id not in trajectories:
                continue

            sign_positions = trajectories[obj_id]
            if len(sign_positions) != len(ego_positions):
                continue  # Skip if trajectory lengths don't match

            # vectors from ego -> sign
            relative_positions = (sign_positions - ego_positions).reshape(-1, 3)
            distances = np.linalg.norm(relative_positions, axis=1)
            distance_mask = distances < DEFAULT_CONFIG["distance_threshold"]

            # Check if approaching
            diff = np.concatenate(
                [np.zeros((1,), dtype=float), np.diff(distances)], axis=0
            )
            # relative speeds (speed between ego and sign)
            relative_speed = diff / FRAME_DELTA

            # is the difference in positions decreasing at the vehicle "moving" speed?
            is_approaching = relative_speed < (
                APPROACH_THRESHOLD_MULTIPLIER * OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
            )
            # is the ego moving away at a reasonable speed?
            is_moving_away = relative_speed > (
                DEPARTURE_THRESHOLD_MULTIPLIER * OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
            )

            # find where we are close to the sign and approaching
            valid_mask = distance_mask & is_approaching
            positive_idx = np.where(valid_mask)[0]

            # find negative samples where we are moving away
            negative_mask = is_moving_away | (~distance_mask)
            negative_idx = np.where(negative_mask)[0]

            # Process positive samples
            for frame_idx in positive_idx:
                sample = _handle_sample(
                    frame_idx,
                    obj_id,
                    distances[frame_idx],
                    relative_speed[frame_idx],
                    positive=True,
                )

                if sample is not None:
                    positive_samples.append(sample)

            # Process negative samples
            for frame_idx in negative_idx:
                sample = _handle_sample(
                    frame_idx,
                    obj_id,
                    distances[frame_idx],
                    relative_speed[frame_idx],
                    positive=False,
                )

                if sample is not None:
                    negative_samples.append(sample)

        # Generate negatives from frames with signs
        frames_with_signs = set()
        sign_object_ids = [
            object_id
            for object_id, label in obj_types.items()
            if "sign" in label.lower()
        ]
        for obj_id in sign_object_ids:
            if obj_id in trajectories:
                # Find frames where this sign is visible and close
                sign_positions = trajectories[obj_id]
                relative_positions = (sign_positions - ego_positions).reshape(-1, 3)
                distances = np.linalg.norm(relative_positions, axis=1)
                close_frames = np.where(
                    distances < DEFAULT_CONFIG["distance_threshold"]
                )[0]
                frames_with_signs.update(close_frames)

        # Sample frames without nearby stop signs as negatives
        all_frames = set(range(len(frames)))
        frames_without_signs = list(all_frames - frames_with_signs)

        # Randomly sample some frames without stop signs
        num_no_sign_negatives = min(
            len(frames_without_signs), len(positive_samples) * 2
        )
        if num_no_sign_negatives > 0:
            sampled_no_sign_frames = random.sample(
                frames_without_signs, num_no_sign_negatives
            )

            for frame_idx in sampled_no_sign_frames:
                frame = frames[frame_idx]
                # Pick any available camera for the frame
                cam = next((cam for cam in frame.cameras), None)
                if cam is None:
                    continue

                template = random.choice(templates)

                negative_samples.append(
                    {
                        "object_id": None,  # No specific object
                        "positive": False,
                        "timestamp": int(frame.timestamp),
                        "question": template,
                        "answer_description": "no signs visible or approaching.",
                        "answer": "no",
                        "metadata": {
                            "frame_type": "no_stop_signs",
                            "camera_name": cam.name,
                        },
                    }
                )

        # DON'T balance here - let VQAGenerator handle it across the full dataset
        results.extend(positive_samples)
        results.extend(negative_samples)

        return choices, results

    def _is_vehicle_in_same_lane(
        self,
        vehicle_pos: np.ndarray,  # n, 3 for n timestamps
        road_left: np.ndarray,  # n, 3 for n timestamps
        road_right: np.ndarray,  # n, 3 for n timestamps
    ) -> np.ndarray:  # Returns array of bools, not single bool
        """Check if vehicle is between the left and right road boundaries."""

        # Vector from right edge to left edge (road width vector)
        road_width_vector = road_left - road_right

        # Vector from right edge to vehicle
        right_to_vehicle = vehicle_pos - road_right

        # Project vehicle position onto the road width vector
        road_width_length_sq = np.sum(road_width_vector * road_width_vector, axis=1)

        # Handle division by zero for each timestamp
        valid_road_width = road_width_length_sq > 1e-8

        # Parameter t: 0 = right edge, 1 = left edge
        t_vehicle = np.zeros(len(vehicle_pos))  # Initialize with zeros
        t_vehicle[valid_road_width] = (
            np.sum(
                right_to_vehicle[valid_road_width]
                * road_width_vector[valid_road_width],
                axis=1,
            )
            / road_width_length_sq[valid_road_width]
        )

        # Vehicle is in lane if 0 <= t <= 1
        vehicle_in_bounds = (t_vehicle >= 0) & (t_vehicle <= 1) & valid_road_width

        return vehicle_in_bounds

    def _prompt_vehicle_future_path(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
        forward_thresh=0.3,
        trajectory_thresh=5.0,
        ego_speed_thresh=5.0,
    ):
        results = []
        choices = ["yes", "no"]

        templates = [
            "Is there a vehicle in the ego vehicle's future path?",
            "Is there a vehicle ahead in the ego vehicle's path?",
            "Can you see a vehicle in front of the ego vehicle?",
            "Is there a vehicle blocking the ego vehicle's path ahead?",
        ]

        def _handle_sample(
            frame_idx: int,
            object_id: str,
            trajectory_distance: float,
            positive: bool,
        ):
            """Handles generating a question for a positive/negative situation."""
            frame = frames[frame_idx]

            obj = (
                next((obj for obj in frame.objects if obj.id == object_id), None)
                if object_id
                else None
            )

            # Find best camera - prioritize one with the object if it exists
            if obj:
                cam = next(
                    (
                        cam
                        for cam in frame.cameras
                        if obj.is_visible_on_camera(frame, cam)
                    ),
                    None,
                )
            else:
                # For negative samples without specific objects, use front
                cam = next((cam for cam in frame.cameras if cam.name == "FRONT"), None)

            if cam is None:
                return None

            template = random.choice(templates)

            if positive:
                answer_description = f"vehicle in path ahead, {trajectory_distance:.2f} metres away from future ego path."
            else:
                answer_description = f"no vehicles in immediate path ahead."

            return {
                "object_id": object_id,
                "positive": positive,
                "timestamp": int(frame.timestamp),
                "question": template,
                "answer_description": answer_description,
                "answer": "yes" if positive else "no",
                "confidence": -1.0 * trajectory_distance,
                "metadata": {
                    "distance": trajectory_distance if positive else None,
                    "camera_name": cam.name,
                },
            }

        # Find any vehicles
        vehicle_object_ids = [
            object_id
            for object_id, label in obj_types.items()
            if "vehicle" in label.lower()
        ]

        # Calculate the "road" based on the ego trajectory.
        road_left = []
        road_right = []

        for frame_idx, frame in enumerate(frames):
            world_pose = np.array(frame.pose, dtype=float).reshape(4, 4)
            ego_y_axis = world_pose[:3, 1]

            left_edge = ego_positions[frame_idx] + 0.5 * ego_y_axis
            right_edge = ego_positions[frame_idx] - 0.5 * ego_y_axis

            road_left.append(left_edge)
            road_right.append(right_edge)

        road_left = np.stack(road_left, axis=0)
        road_right = np.stack(road_right, axis=0)

        valid_speed = ego_speeds > ego_speed_thresh  # Only when ego is moving

        # Normalize ego direction vectors
        ego_forward_normalized = ego_vectors / (
            np.linalg.norm(ego_vectors, axis=1, keepdims=True) + 1e-6
        )

        # Store frame-level information
        frame_samples = []  # Will store (frame_idx, obj_id, distance, is_positive)

        for obj_id in vehicle_object_ids:
            if obj_id not in trajectories:
                continue

            vehicle_positions = trajectories[obj_id]
            if len(vehicle_positions) != len(ego_positions):
                continue

            # vectors from ego -> vehicle
            relative_positions = (vehicle_positions - ego_positions).reshape(-1, 3)
            distances = np.linalg.norm(relative_positions, axis=1)
            distance_mask = distances < DEFAULT_CONFIG["distance_threshold"]

            # at some point they're within ~5 metres
            trajectory_distances = []
            for pos_i, veh_pos in enumerate(vehicle_positions):
                dists = np.linalg.norm(
                    ego_positions[pos_i:, :2] - veh_pos[np.newaxis, :2], axis=1
                )
                trajectory_distances.append(dists.min())
            trajectory_distances = np.array(trajectory_distances).reshape(-1)

            future_dist_mask = trajectory_distances < trajectory_thresh

            # Dot product to check if object is in front
            forward_component = np.sum(
                relative_positions * ego_forward_normalized, axis=1
            )

            # Object is in front if projection is positive AND ego is moving
            in_front_mask = (forward_component > forward_thresh) & valid_speed

            # Check which frames have vehicle in same lane (for more accurate "future path")
            same_lane_mask = self._is_vehicle_in_same_lane(
                vehicle_positions, road_left, road_right
            )

            # DECISION: What constitutes "future path"?
            # Option 1: Same lane ahead (more restrictive)
            future_path_mask = (
                distance_mask & in_front_mask & (same_lane_mask | future_dist_mask)
            )

            # Option 2: Any vehicle ahead (less restrictive)
            # future_path_mask = distance_mask & in_front_mask

            positive_idx = np.where(future_path_mask)[0]

            # Negatives: vehicles that are close but NOT in future path
            negative_mask = distance_mask & ~future_path_mask
            negative_idx = np.where(negative_mask)[0]

            # Store positive samples
            for frame_idx in positive_idx:
                frame_samples.append(
                    (frame_idx, obj_id, trajectory_distances[frame_idx], True)
                )

            # Store negative samples
            for frame_idx in negative_idx:
                frame_samples.append(
                    (frame_idx, obj_id, trajectory_distances[frame_idx], False)
                )

        # Process all samples
        num_neg, num_pos, num_no_obj = 0, 0, 0
        pos_per_obj = defaultdict(list)
        neg_per_obj = defaultdict(list)
        empty_samples = []
        total = 0
        for frame_idx, obj_id, trajectory_distance, is_positive in frame_samples:
            sample = _handle_sample(frame_idx, obj_id, trajectory_distance, is_positive)
            if sample is not None:
                total += 1
                if is_positive:
                    num_pos += 1
                    pos_per_obj[obj_id].append(sample)
                else:
                    num_neg += 1
                    neg_per_obj[obj_id].append(sample)

                if obj_id is None:
                    empty_samples.append(sample)

        total = max(1, total)
        num_empty = len(empty_samples)
        print(
            "num_neg, num_pos, num_no_obj, total",
            num_neg,
            num_pos,
            num_no_obj,
            num_empty,
            total,
        )
        print(
            "num_neg, num_pos, num_no_obj",
            num_neg / total,
            num_pos / total,
            num_no_obj / total,
            num_empty / total,
        )

        negative_samples, positive_samples = [], []
        neg_drwn = 0
        pos_drwn = 0
        obj_seen_set = set(list(pos_per_obj.keys()) + list(neg_per_obj.keys()))
        for obj_id in obj_seen_set:
            # randomly get a positive sample
            neg_sample, pos_sample = None, None
            if obj_id in pos_per_obj:
                pos_sample = random.choice(pos_per_obj[obj_id])
                positive_samples.append(pos_sample)
                pos_drwn += 1
            if obj_id in neg_per_obj:
                neg_sample = random.choice(neg_per_obj[obj_id])
                negative_samples.append(neg_sample)
                neg_drwn += 1

        if len(positive_samples) == 0 or len(negative_samples) == 0:
            return choices, []

        processed_samples = self._balance_neg_pos(negative_samples, positive_samples)
        print("neg_drwn, pos_drwn, total", neg_drwn, pos_drwn, neg_drwn + pos_drwn)

        return choices, processed_samples

    def _prompt_faster_than_ego(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
        threshold_factor=1.5,
    ):
        results = []
        choices = ["yes", "no"]

        templates = [
            "Is the {} moving faster than the ego vehicle?",
        ]

        for obj_id in trajectories.keys():
            if obj_types[obj_id] not in ["Vehicle", "Pedestrian", "Cyclist"]:
                continue
            obj_speeds = speeds[obj_id]  # Speed array over time
            obj_trajectory = trajectories[obj_id]

            # Calculate relative distances over time
            relative_positions = obj_trajectory - ego_positions
            distances = np.linalg.norm(relative_positions, axis=1)

            # Find timestamps where object is significantly faster
            speed_ratios = obj_speeds / (ego_speeds + 1e-6)
            faster_mask = speed_ratios > threshold_factor
            valid_mask = distances < DEFAULT_CONFIG["distance_threshold"]

            # moving relatively fast (e.g. vehicle speed since comparing with ego)
            moving_fast = obj_speeds > OBJECT_SPEED_THRESH["TYPE_VEHICLE"]

            all_condition = faster_mask & valid_mask & moving_fast

            template = random.choice(templates)

            if np.any(all_condition):  # Only if moving reasonably fast
                # Pick best timestamp - highest speed difference
                best_idx = np.argmax(speed_ratios * all_condition)
                timestamp = timestamps[best_idx]
                speed_diff = obj_speeds[best_idx] - ego_speeds[best_idx]

                obj = next(
                    (obj for obj in frames[best_idx].objects if obj.id == obj_id), None
                )

                if obj is None:
                    continue

                results.append(
                    {
                        "object_id": obj_id,
                        "timestamp": int(timestamp),
                        "question": template.format(obj.get_object_description()),
                        "answer_description": f"faster by {speed_diff:.1f} m/s",
                        "answer": "yes",
                    }
                )

            # negative samples
            if np.any((~faster_mask) & valid_mask):
                # Pick best timestamp - highest speed difference
                best_idx = np.argmax(
                    (ego_speeds / (obj_speeds + 1e-6)) * ((~faster_mask) & valid_mask)
                )
                timestamp = timestamps[best_idx]
                speed_diff = ego_speeds[best_idx] - obj_speeds[best_idx]

                obj = next(
                    (obj for obj in frames[best_idx].objects if obj.id == obj_id), None
                )

                if obj is None:
                    continue

                results.append(
                    {
                        "object_id": obj_id,
                        "timestamp": int(timestamp),
                        "question": template.format(obj.get_object_description()),
                        "answer_description": f"slower by {speed_diff:.1f} m/s",
                        "answer": "no",
                    }
                )

        return choices, results

    def _prompt_moving_towards_ego(
        self,
        trajectories,
        speeds,
        accels,
        ego_positions,
        ego_speeds,
        ego_vectors,
        timestamps,
        frames,
        obj_types,
        obj_cvat_types,
        window_size=5,
    ):
        """Generate questions about objects moving towards/away from ego vehicle."""
        results = []
        choices = ["yes", "no"]

        def _get_scenario_and_templates(
            convergence_factor, obj_heading_to_ego, ego_heading_to_obj
        ):
            """Determine scenario type and return appropriate question templates."""
            # Head-on collision scenario
            if (
                convergence_factor < SCENARIO_THRESHOLDS["head_on_convergence"]
                and obj_heading_to_ego > SCENARIO_THRESHOLDS["head_on_heading"]
                and ego_heading_to_obj > SCENARIO_THRESHOLDS["head_on_heading"]
            ):
                return "head_on", SCENARIO_TEMPLATES["head_on"]

            # Object primarily approaching ego
            elif (
                obj_heading_to_ego > SCENARIO_THRESHOLDS["strong_heading"]
                and ego_heading_to_obj < SCENARIO_THRESHOLDS["weak_heading"]
            ):
                return "object_approaching", SCENARIO_TEMPLATES["object_approaching"]

            # Ego primarily approaching object
            elif (
                ego_heading_to_obj > SCENARIO_THRESHOLDS["strong_heading"]
                and obj_heading_to_ego < SCENARIO_THRESHOLDS["weak_heading"]
            ):
                return "ego_approaching", SCENARIO_TEMPLATES["ego_approaching"]

            # General convergence
            else:
                return "general_convergence", SCENARIO_TEMPLATES["general_convergence"]

        def _get_divergence_templates(
            convergence_factor, obj_heading_to_ego, ego_heading_to_obj
        ):
            """Get templates for diverging scenarios."""
            # Parallel movement
            if convergence_factor > SCENARIO_THRESHOLDS["parallel_threshold"]:
                return "parallel_diverging", SCENARIO_TEMPLATES["parallel_diverging"]

            # Object moving away
            elif obj_heading_to_ego < SCENARIO_THRESHOLDS["divergence_threshold"]:
                return "object_diverging", SCENARIO_TEMPLATES["object_diverging"]

            # Ego moving away
            elif ego_heading_to_obj < SCENARIO_THRESHOLDS["divergence_threshold"]:
                return "ego_diverging", SCENARIO_TEMPLATES["ego_diverging"]

            # General divergence
            else:
                return "general_divergence", SCENARIO_TEMPLATES["general_divergence"]

        def _find_pivotal_frame(
            frame_indices,
            distances,
            relative_speeds,
            direction_dots,
            obj_headings,
            ego_headings,
        ):
            """Find the most representative frame based on convergence strength."""
            if len(frame_indices) == 0:
                return None

            convergence_scores = []
            for idx in frame_indices:
                distance_factor = max(0, (50.0 - distances[idx]) / 50.0)
                speed_factor = min(abs(relative_speeds[idx]) / 5.0, 1.0)
                direction_factor = max(0, -direction_dots[idx])
                heading_factor = max(obj_headings[idx], ego_headings[idx])

                total_score = (
                    distance_factor + speed_factor + direction_factor + heading_factor
                )
                convergence_scores.append(total_score)

            best_idx_pos = np.argmax(convergence_scores)
            return frame_indices[best_idx_pos]

        def _is_object_visible_and_significant(frame_idx, obj_id):
            """Check if object is visible and meets minimum requirements."""
            frame = frames[frame_idx]
            obj = next((obj for obj in frame.objects if obj.id == obj_id), None)

            if (
                obj is None
                or obj.num_lidar_points_in_box < MOVEMENT_THRESHOLDS["min_lidar_points"]
            ):
                return False

            visible_cameras = [
                cam for cam in frame.cameras if obj.is_visible_on_camera(frame, cam)
            ]
            return bool(visible_cameras)

        def _create_sample(
            frame_idx,
            object_id,
            distance,
            relative_speed,
            convergence_factor,
            obj_heading_to_ego,
            ego_heading_to_obj,
            positive,
        ):
            """Create a training sample with question and answer."""
            frame = frames[frame_idx]
            obj = next((obj for obj in frame.objects if obj.id == object_id), None)

            if obj is None:
                return None

            cam = next(
                (cam for cam in frame.cameras if obj.is_visible_on_camera(frame, cam)),
                None,
            )
            if cam is None:
                return None

            if positive:
                scenario, templates = _get_scenario_and_templates(
                    convergence_factor, obj_heading_to_ego, ego_heading_to_obj
                )
                answer = "yes"
                answer_description = f"approaching {obj.get_object_description()} at {abs(relative_speed):.2f} m/s, currently {distance:.2f} metres away. Scenario: {scenario}"
            else:
                scenario, templates = _get_divergence_templates(
                    convergence_factor, obj_heading_to_ego, ego_heading_to_obj
                )
                answer = "no"
                answer_description = f"moving away from {obj.get_object_description()} at {abs(relative_speed):.2f} m/s, currently {distance:.2f} metres away. Scenario: {scenario}"

            template = random.choice(templates)

            return {
                "object_id": object_id,
                "positive": positive,
                "scenario": scenario,
                "timestamp": int(frame.timestamp),
                "question": template.format(obj.get_object_description()),
                "answer_description": answer_description,
                "answer": answer,
                "confidence": float(
                    abs(relative_speed) / OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
                ),
                "convergence_factor": float(convergence_factor),
                "obj_heading_to_ego": float(obj_heading_to_ego),
                "ego_heading_to_obj": float(ego_heading_to_obj),
                "distance": float(distance),
                "relative_speed": float(relative_speed),
            }

        def _analyze_object_trajectory(obj_id, obj_type, obj_trajectory):
            """Analyze a single object's trajectory for approach/divergence patterns."""
            # Calculate relative positions and distances
            relative_positions = obj_trajectory - ego_positions
            distances = np.linalg.norm(relative_positions, axis=1)

            # Calculate relative speed and movement directions
            diff = np.concatenate(
                [np.zeros((1,), dtype=float), np.diff(distances)], axis=0
            )
            relative_speed = diff / FRAME_DELTA

            obj_velocity = np.gradient(obj_trajectory, axis=0) / FRAME_DELTA
            ego_velocity = np.gradient(ego_positions, axis=0) / FRAME_DELTA
            relative_velocity = obj_velocity - ego_velocity

            # Calculate direction vectors
            obj_direction = obj_velocity / (
                np.linalg.norm(obj_velocity, axis=1, keepdims=True) + 1e-8
            )
            ego_direction = ego_velocity / (
                np.linalg.norm(ego_velocity, axis=1, keepdims=True) + 1e-8
            )
            relative_direction = relative_velocity / (
                np.linalg.norm(relative_velocity, axis=1, keepdims=True) + 1e-8
            )

            # Calculate convergence metrics
            obj_to_ego_vector = ego_positions - obj_trajectory
            obj_to_ego_normalized = obj_to_ego_vector / (
                np.linalg.norm(obj_to_ego_vector, axis=1, keepdims=True) + 1e-8
            )
            convergence_factor = np.sum(
                relative_direction * obj_to_ego_normalized, axis=1
            )

            obj_heading_to_ego = np.sum(obj_direction * obj_to_ego_normalized, axis=1)
            ego_to_obj_normalized = -obj_to_ego_normalized
            ego_heading_to_obj = np.sum(ego_direction * ego_to_obj_normalized, axis=1)

            return {
                "distances": distances,
                "relative_speed": relative_speed,
                "convergence_factor": convergence_factor,
                "obj_heading_to_ego": obj_heading_to_ego,
                "ego_heading_to_obj": ego_heading_to_obj,
            }

        def _find_approach_samples(metrics, obj_type):
            """Find frames where object is approaching ego."""
            speed_threshold = OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
            distance_threshold = DEFAULT_CONFIG["distance_threshold"]

            # Movement conditions - FIXED LOGIC
            distance_mask = metrics["distances"] < distance_threshold
            distance_approaching = metrics["relative_speed"] <= (-speed_threshold)

            # Objects are converging if moving toward each other OR one is heading toward the other
            paths_converging = (
                (
                    metrics["convergence_factor"]
                    > MOVEMENT_THRESHOLDS["convergence_threshold"]
                )
                | (
                    metrics["obj_heading_to_ego"]
                    > MOVEMENT_THRESHOLDS["heading_threshold"]
                )
                | (
                    metrics["ego_heading_to_obj"]
                    > MOVEMENT_THRESHOLDS["heading_threshold"]
                )
            )

            is_approaching = distance_approaching & paths_converging
            approach_consistency = (
                np.cumsum(is_approaching.astype(int))
                >= MOVEMENT_THRESHOLDS["consistency_frames"]
            )
            significant_speed = np.abs(metrics["relative_speed"]) > (
                speed_threshold * MOVEMENT_THRESHOLDS["speed_multiplier"]
            )

            min_dist, max_dist = DISTANCE_BOUNDS.get(
                obj_type, DISTANCE_BOUNDS["Vehicle"]
            )
            reasonable_distance = (metrics["distances"] >= min_dist) & (
                metrics["distances"] <= max_dist
            )

            positive_mask = (
                distance_mask
                & is_approaching
                & approach_consistency
                & significant_speed
                & reasonable_distance
            )
            return np.where(positive_mask)[0]

        def _find_divergence_samples(metrics, obj_type):
            """Find frames where object is moving away from ego."""
            speed_threshold = OBJECT_SPEED_THRESH["TYPE_VEHICLE"]
            distance_threshold = DEFAULT_CONFIG["distance_threshold"]

            # Movement conditions - FIXED LOGIC
            distance_moving_away = metrics["relative_speed"] >= speed_threshold

            # Objects are diverging if moving away from each other AND both heading away
            paths_diverging = (
                metrics["convergence_factor"]
                < MOVEMENT_THRESHOLDS["divergence_threshold"]
            ) | (  # Actually diverging
                (
                    metrics["obj_heading_to_ego"]
                    < MOVEMENT_THRESHOLDS["divergence_heading_threshold"]
                )
                & (
                    metrics["ego_heading_to_obj"]
                    < MOVEMENT_THRESHOLDS["divergence_heading_threshold"]
                )
            )

            is_moving_away = distance_moving_away & paths_diverging
            away_consistency = (
                np.cumsum(is_moving_away.astype(int))
                >= MOVEMENT_THRESHOLDS["consistency_frames"]
            )
            significant_speed = np.abs(metrics["relative_speed"]) > (
                speed_threshold * MOVEMENT_THRESHOLDS["speed_multiplier"]
            )
            reasonable_distance = (
                metrics["distances"] > MOVEMENT_THRESHOLDS["min_distance"]
            ) & (metrics["distances"] < distance_threshold * 2)

            negative_mask = (
                is_moving_away
                & away_consistency
                & significant_speed
                & reasonable_distance
            )
            return np.where(negative_mask)[0]

        # Main processing loop
        for obj_id in trajectories.keys():
            obj_type = obj_types[obj_id]
            if obj_type not in ["Vehicle", "Pedestrian", "Cyclist"]:
                continue

            if (
                not np.any(speeds[obj_id] > MOVEMENT_THRESHOLDS["min_object_speed"])
                or obj_cvat_types[obj_id] is None
            ):
                continue

            obj_trajectory = trajectories[obj_id]
            metrics = _analyze_object_trajectory(obj_id, obj_type, obj_trajectory)

            # Find positive and negative samples
            positive_idx = _find_approach_samples(metrics, obj_type)
            negative_idx = _find_divergence_samples(metrics, obj_type)

            # Find pivotal frames
            pivotal_positive_idx = _find_pivotal_frame(
                positive_idx,
                metrics["distances"],
                metrics["relative_speed"],
                metrics["convergence_factor"],
                metrics["obj_heading_to_ego"],
                metrics["ego_heading_to_obj"],
            )

            pivotal_negative_idx = _find_pivotal_frame(
                negative_idx,
                metrics["distances"],
                metrics["relative_speed"],
                metrics["convergence_factor"],
                metrics["obj_heading_to_ego"],
                metrics["ego_heading_to_obj"],
            )

            # Process samples
            for frame_idx, positive in [
                (pivotal_positive_idx, True),
                (pivotal_negative_idx, False),
            ]:
                if frame_idx is None or not _is_object_visible_and_significant(
                    frame_idx, obj_id
                ):
                    continue

                sample = _create_sample(
                    frame_idx,
                    obj_id,
                    metrics["distances"][frame_idx],
                    metrics["relative_speed"][frame_idx],
                    metrics["convergence_factor"][frame_idx],
                    metrics["obj_heading_to_ego"][frame_idx],
                    metrics["ego_heading_to_obj"][frame_idx],
                    positive,
                )

                if sample is not None:
                    results.append(sample)

        return choices, results

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

        (trajectories, speeds, accels, ego_positions, ego_speeds, ego_vectors) = (
            self._calculate_trajectories(frames, timestamps)
        )

        obj_types = {
            obj.id: obj.get_simple_type() for frame in frames for obj in frame.objects
        }

        obj_cvat_types = {
            obj.id: obj.cvat_label for frame in frames for obj in frame.objects
        }

        for prompt_func in self.prompt_functions:
            prompt_out = prompt_func(
                trajectories,
                speeds,
                accels,
                ego_positions,
                ego_speeds,
                ego_vectors,
                timestamps,
                frames,
                obj_types,
                obj_cvat_types,
            )

            #
            choices, questions = prompt_out

            for question_dict in questions:
                try:
                    timestamp = question_dict["timestamp"]
                    object_id = question_dict["object_id"]
                    question_txt = question_dict["question"]
                    answer_txt = question_dict["answer"]
                except TypeError as e:
                    print("Typeerror:", e)
                    print("questions", type(questions))
                    print("questions", questions)
                    print("question_dict", type(question_dict))
                    print("prompt_func", prompt_func)
                    print("prompt_func.__name__", prompt_func.__name__)
                    print("question_dict", question_dict)
                    raise e

                frame_idx = next(
                    (idx for idx, time in enumerate(timestamps) if time == timestamp),
                    None,
                )

                if frame_idx is None:
                    continue

                frame = frames[frame_idx]
                obj = next((obj for obj in frame.objects if obj.id == object_id), None)

                if obj is None:
                    continue

                assert obj.id == object_id

                camera = next(
                    (
                        cam
                        for cam in frame.cameras
                        if obj.most_visible_camera_name == cam.name
                    ),
                    None,
                )

                if camera is None:
                    continue

                if not obj.is_visible_on_camera(frame, camera):
                    continue

                if len(choices) > CHOICES_OPTIONS:
                    # Remove correct answer from choices
                    distractors = [x for x in choices if x != answer_txt]

                    # Sample (CHOICES_OPTIONS - 1) distractors
                    sampled = np.random.choice(
                        distractors, CHOICES_OPTIONS - 1, replace=False
                    ).tolist()

                    # Add the correct answer back in
                    choices = sampled + [answer_txt]

                    # (Optional) Shuffle the final choices
                    np.random.shuffle(choices)

                if len(choices) < 2:
                    print(f"Received only {len(choices)} choices from {prompt_func}")
                    continue

                if answer_txt is None or answer_txt not in choices:
                    print(
                        f"answer_txt={answer_txt} is none or not in choices={choices}"
                    )
                    continue

                question = MultipleImageMultipleChoiceQuestion(
                    image_paths=[camera.image_path for camera in frame.cameras],
                    question=question_txt,
                    choices=choices,
                    scene_id=obj.scene_id,
                    timestamp=frame.timestamp,
                    camera_names=[camera.name for camera in frame.cameras],
                    generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                    question_name=prompt_func.__name__,
                    data=dict(
                        question_dict=question_dict,
                        best_camera_name=camera.name,
                        object_id=object_id,
                        bbox=obj.get_object_bbox_2d(frame, camera),
                    ),
                )

                answer = MultipleChoiceAnswer(
                    choices=choices,
                    answer=answer_txt,
                )

                samples.append((question, answer))

        random.shuffle(samples)

        return samples

    def _calculate_trajectories(self, frames, timestamps):
        object_positions = defaultdict(dict)
        object_speeds = defaultdict(dict)
        object_accelerations = defaultdict(dict)
        ego_positions = []
        # Build object position lookup per timestamp
        for frame in frames:
            world_pose = np.array(frame.pose)

            ego_position = world_pose[:3, -1]
            ego_positions.append(ego_position)

            for obj in frame.objects:
                if (
                    obj
                    and obj.id
                    and (
                        obj.get_simple_type()
                        in ["Vehicle", "Pedestrian", "Cyclist", "Sign"]
                    )  # include sign for stop sign question
                ):  # Ensure object exists and has ID
                    try:
                        object_positions[obj.id][frame.timestamp] = (
                            obj.get_world_centre(frame)
                        )
                        object_speeds[obj.id][frame.timestamp] = obj.get_speed()
                        object_accelerations[obj.id][frame.timestamp] = obj.get_accel()
                    except (AttributeError, ValueError) as e:
                        pass

        ego_positions = np.stack(ego_positions, axis=0)
        ego_speeds = [0.0]
        ego_vectors = [np.zeros((3,), dtype=float)]
        last_pos = ego_positions[0]

        for pos in ego_positions[1:]:
            displacement = pos - last_pos  # Displacement vector (meters)
            distance = np.linalg.norm(displacement)  # Distance (meters)
            speed = float(distance / FRAME_DELTA)  # Speed (m/s)

            # Vector should be normalized displacement, not divided by time
            if distance > 1e-6:  # Avoid division by zero
                direction_vector = displacement / distance  # Unit direction vector
            else:
                direction_vector = np.zeros((3,), dtype=float)

            ego_speeds.append(speed)
            ego_vectors.append(
                direction_vector
            )  # Or just use displacement if you want magnitude
            last_pos = pos

        ego_speeds = np.array(ego_speeds).reshape(-1)
        ego_vectors = np.stack(ego_vectors, axis=0)

        # Fill in trajectories, holding the last known position
        full_object_trajectories = {}
        full_object_speeds = {}
        full_object_accels = {}
        for obj_id, position_by_time in object_positions.items():
            object_ts = np.array(
                [float(x) for x in position_by_time.keys()], dtype=float
            )

            object_positions_array = np.array(
                [x for x in position_by_time.values()], dtype=float
            )

            object_speed_array = np.array(
                [x for x in object_speeds[obj_id].values()], dtype=float
            )

            object_accel_array = np.array(
                [x for x in object_accelerations[obj_id].values()], dtype=float
            )

            trajectory = []
            accel = []
            for i in range(3):
                vals = np.interp(timestamps, object_ts, object_positions_array[:, i])
                trajectory.append(vals)

                if i < object_accel_array.shape[1]:
                    vals = np.interp(timestamps, object_ts, object_accel_array[:, i])
                    accel.append(vals)

            trajectory = np.stack(trajectory, axis=1)
            assert trajectory.shape[0] == len(timestamps)

            full_object_trajectories[obj_id] = trajectory
            full_object_speeds[obj_id] = np.interp(
                timestamps, object_ts, object_speed_array.reshape(-1)
            )
            full_object_accels[obj_id] = np.stack(accel, axis=1)

        return (
            full_object_trajectories,
            full_object_speeds,
            full_object_accels,
            ego_positions,
            ego_speeds,
            ego_vectors,
        )

    def visualize_trajectories(
        self,
        object_id,
        trajectories,
        ego_positions,
        frame,
        camera,
        timestamps: np.ndarray,
        save_path=None,
        question_txt="",
        answer_txt="",
        answer_description="",
    ):
        """
        Visualize object trajectories for debugging purposes.

        Creates a figure with two subplots:
        1. 3D plot showing the trajectories in world coordinates
        2. 2D plot showing the trajectories projected onto a camera image

        Args:
            object_id: ID of the main object
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

        timestamp_idx = next(
            idx for idx, time in enumerate(timestamps) if time == frame.timestamp
        )

        # Get frame and camera for visualization
        # Create figure with two subplots
        fig = plt.figure(figsize=(16, 8))

        # Get trajectories
        traj1 = trajectories[object_id]
        traj2 = ego_positions

        # Get object descriptions for labels
        obj1_desc = obj.get_object_description()
        obj2_desc = "Ego"

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
                img_points2 = projected_points2[:, :2]

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

        # Add metadata as text
        info_text = (
            f"Scene ID: {obj.scene_id}\n"
            f"Timestamp: {frame.timestamp}\n"
            f"Object: {obj1_desc}\n"
            f"Camera: {camera.name}\n"
            f"Question: {question_txt}\n"
            f"Answer: {answer_txt}\n"
            f"answer_description: {answer_description}\n"
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
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()

    def visualize_trajectories_plotly(
        self,
        object_id,
        trajectories,
        ego_positions,
        frame,
        camera,
        timestamps: np.ndarray,
        save_path=None,
        question_txt="",
        answer_txt="",
        answer_description="",
    ):
        """
        Simple but modern Plotly trajectory visualization that avoids subplot complexities.
        Creates separate figures for different views that can be displayed together.
        """
        import plotly.graph_objects as go
        import plotly.express as px
        import numpy as np
        from PIL import Image
        import base64
        from io import BytesIO

        # Get object and timestamp info
        obj = next(obj for obj in frame.objects if obj.id == object_id)
        timestamp_idx = next(
            idx for idx, time in enumerate(timestamps) if time == frame.timestamp
        )

        # Prepare trajectory data
        traj1 = (
            trajectories[object_id]
            if trajectories is not None
            else np.zeros((0, 3), dtype=float)
        )
        traj2 = (
            ego_positions
            if ego_positions is not None
            else np.zeros((0, 3), dtype=float)
        )

        obj1_desc = obj.get_object_description()
        obj2_desc = "Ego Vehicle"

        # === MAIN 3D TRAJECTORY PLOT ===

        fig_3d = go.Figure()

        # Create color scales for time progression
        n_points = len(traj1)
        time_colors = np.linspace(0, 1, n_points)

        # Object trajectory
        fig_3d.add_trace(
            go.Scatter3d(
                x=traj1[:, 0],
                y=traj1[:, 1],
                z=traj1[:, 2] if traj1.shape[1] > 2 else np.zeros(len(traj1)),
                mode="lines+markers",
                line=dict(
                    color=time_colors,
                    colorscale="Viridis",
                    width=6,
                    # colorbar=dict(title="Time Progress"),
                ),
                marker=dict(size=4, opacity=0.8),
                name=obj1_desc,
                hovertemplate=(
                    f"<b>{obj1_desc}</b><br>"
                    "Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>"
                    "Time: %{text}<br>"
                    "<extra></extra>"
                ),
                text=[f"t={t:.2f}s" for t in timestamps],
            )
        )

        # Ego trajectory
        fig_3d.add_trace(
            go.Scatter3d(
                x=traj2[:, 0],
                y=traj2[:, 1],
                z=traj2[:, 2] if traj2.shape[1] > 2 else np.zeros(len(traj2)),
                mode="lines+markers",
                line=dict(color=time_colors, colorscale="Plasma", width=6),
                marker=dict(size=4, opacity=0.8),
                name=obj2_desc,
                hovertemplate=(
                    f"<b>{obj2_desc}</b><br>"
                    "Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>"
                    "Time: %{text}<br>"
                    "<extra></extra>"
                ),
                text=[f"t={t:.2f}s" for t in timestamps],
            )
        )

        # Add waypoints
        waypoints = [
            (0, "start", "circle", ["green", "red"]),
            (timestamp_idx, "current", "diamond", ["blue", "orange"]),
            (-1, "end", "square", ["darkgreen", "darkred"]),
        ]

        for idx, label, symbol, colors in waypoints:
            # Object waypoint
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[traj1[idx, 0]],
                    y=[traj1[idx, 1]],
                    z=[traj1[idx, 2] if traj1.shape[1] > 2 else 0],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=colors[0],
                        symbol=symbol,
                        line=dict(width=2, color="white"),
                    ),
                    name=f"{obj1_desc} ({label})",
                    showlegend=True,
                    hovertemplate=f"<b>{obj1_desc} ({label})</b><br>Time: {timestamps[idx]:.2f}s<extra></extra>",
                )
            )

            # Ego waypoint
            fig_3d.add_trace(
                go.Scatter3d(
                    x=[traj2[idx, 0]],
                    y=[traj2[idx, 1]],
                    z=[traj2[idx, 2] if traj2.shape[1] > 2 else 0],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=colors[1],
                        symbol=symbol,
                        line=dict(width=2, color="white"),
                    ),
                    name=f"{obj2_desc} ({label})",
                    showlegend=True,
                    hovertemplate=f"<b>{obj2_desc} ({label})</b><br>Time: {timestamps[idx]:.2f}s<extra></extra>",
                )
            )


        # Get the current position of traj1 at the specified timestamp
        current_pos = traj1[timestamp_idx]
        center_x = current_pos[0]
        center_y = current_pos[1]
        center_z = current_pos[2] if traj1.shape[1] > 2 else 0
        
        traj1_mins = traj1.min(axis=0)
        traj1_maxes = traj1.max(axis=0)
        traj2_mins = traj2.min(axis=0)
        traj2_maxes = traj2.max(axis=0)
        print('traj1_mins', traj1_mins, traj2_mins)

        # Update 3D layout with centered view
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(
                    title="X (m)",
                    range=[center_x - 5, center_x + 5]
                ),
                yaxis=dict(
                    title="Y (m)", 
                    range=[center_y - 5, center_y + 5]
                ),
                zaxis=dict(
                    title="Z (m)",
                    range=[center_z - 5, center_z + 5]
                ),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.2)  # This maintains the 50:50:10 ratio visually
            ),
            title=dict(
                text=f"3D Trajectory Analysis - {obj1_desc}", x=0.5, font=dict(size=20)
            ),
            height=600,
            showlegend=True,
            annotations=[
                dict(
                    text=(
                        f"<b>Scene Metadata</b><br>"
                        f"Scene ID: {obj.scene_id}<br>"
                        f"Timestamp: {frame.timestamp:.2f}s<br>"
                        f"Camera: {camera.name}<br>"
                        f"Question: {question_txt}<br>"
                        f"Answer: {answer_txt}"
                    ),
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray",
                    borderwidth=1,
                    font=dict(size=10),
                )
            ],
        )

        fig_bev = go.Figure()

        # Create color scales for time progression
        n_points = len(traj1)
        time_colors = np.linspace(0, 1, n_points)

        # Object trajectory - using 2D Scatter for BEV
        fig_bev.add_trace(
            go.Scatter(
                x=traj1[:, 0],
                y=traj1[:, 1],
                mode="lines+markers",
                line=dict(
                    color='green',
                    width=6,
                ),
                marker=dict(
                    size=4, 
                    opacity=0.8,
                    color='green',
                ),
                name=obj1_desc,
                hovertemplate=(
                    f"<b>{obj1_desc}</b><br>"
                    "Position: (%{x:.2f}, %{y:.2f})<br>"
                    "Time: %{text}<br>"
                    "<extra></extra>"
                ),
                text=[f"t={t:.2f}s" for t in timestamps],
            )
        )

        # Ego trajectory - using 2D Scatter for BEV
        fig_bev.add_trace(
            go.Scatter(
                x=traj2[:, 0],
                y=traj2[:, 1],
                mode="lines+markers",
                line=dict(
                    color='red', 
                    width=6
                ),
                marker=dict(
                    size=4, 
                    opacity=0.8,
                    color='red',
                ),
                name=obj2_desc,
                hovertemplate=(
                    f"<b>{obj2_desc}</b><br>"
                    "Position: (%{x:.2f}, %{y:.2f})<br>"
                    "Time: %{text}<br>"
                    "<extra></extra>"
                ),
                text=[f"t={t:.2f}s" for t in timestamps],
            )
        )

        # Add waypoints
        waypoints = [
            (0, "start", "circle", ["green", "red"]),
            (timestamp_idx, "current", "diamond", ["blue", "orange"]),
            (-1, "end", "square", ["darkgreen", "darkred"]),
        ]

        for idx, label, symbol, colors in waypoints:
            # Object waypoint
            fig_bev.add_trace(
                go.Scatter(
                    x=[traj1[idx, 0]],
                    y=[traj1[idx, 1]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=colors[0],
                        symbol=symbol,
                        line=dict(width=2, color="white"),
                    ),
                    name=f"{obj1_desc} ({label})",
                    showlegend=True,
                    hovertemplate=f"<b>{obj1_desc} ({label})</b><br>Time: {timestamps[idx]:.2f}s<extra></extra>",
                )
            )

            # Ego waypoint
            fig_bev.add_trace(
                go.Scatter(
                    x=[traj2[idx, 0]],
                    y=[traj2[idx, 1]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=colors[1],
                        symbol=symbol,
                        line=dict(width=2, color="white"),
                    ),
                    name=f"{obj2_desc} ({label})",
                    showlegend=True,
                    hovertemplate=f"<b>{obj2_desc} ({label})</b><br>Time: {timestamps[idx]:.2f}s<extra></extra>",
                )
            )

        # Get the current position of traj1 at the specified timestamp
        current_pos = traj1[timestamp_idx]
        center_x = current_pos[0]
        center_y = current_pos[1]

        # Update layout for Bird's Eye View (2D plot)
        fig_bev.update_layout(
            xaxis=dict(
                title="X (m)",
                range=[center_x - 25, center_x + 25],
                scaleanchor="y",  # This ensures equal scaling between x and y axes
                scaleratio=1,
            ),
            yaxis=dict(
                title="Y (m)", 
                range=[center_y - 25, center_y + 25],
            ),
            title=dict(
                text=f"Bird's Eye View - {obj1_desc}", x=0.5, font=dict(size=20)
            ),
            height=600,
            width=600,  # Square aspect ratio for BEV
            showlegend=True,
            annotations=[
                dict(
                    text=(
                        f"<b>Scene Metadata</b><br>"
                        f"Scene ID: {obj.scene_id}<br>"
                        f"Timestamp: {frame.timestamp:.2f}s<br>"
                        f"Camera: {camera.name}<br>"
                        f"Question: {question_txt}<br>"
                        f"Answer: {answer_txt}"
                    ),
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.98,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray",
                    borderwidth=1,
                    font=dict(size=10),
                )
            ],
            # Add grid for better spatial reference
            plot_bgcolor="white",
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            xaxis_gridcolor="lightgray",
            yaxis_gridcolor="lightgray",
        )
        # === TIMELINE PLOT ===

        fig_timeline = go.Figure()

        # Calculate distance between trajectories over time
        distances = np.linalg.norm(traj1 - traj2, axis=1)

        fig_timeline.add_trace(
            go.Scatter(
                x=timestamps,
                y=distances,
                mode="lines+markers",
                line=dict(color="orange", width=3),
                marker=dict(size=8, color="orange"),
                name="Distance",
                hovertemplate="<b>Distance</b><br>Time: %{x:.2f}s<br>Distance: %{y:.2f}m<extra></extra>",
            )
        )

        # Add current time marker
        fig_timeline.add_trace(
            go.Scatter(
                x=[timestamps[timestamp_idx]],
                y=[distances[timestamp_idx]],
                mode="markers",
                marker=dict(size=15, color="red", symbol="diamond"),
                name="Current Time",
                hovertemplate=f"<b>Current Time</b><br>Time: {timestamps[timestamp_idx]:.2f}s<br>Distance: {distances[timestamp_idx]:.2f}m<extra></extra>",
            )
        )

        # Add vertical line for current time
        fig_timeline.add_shape(
            type="line",
            x0=timestamps[timestamp_idx],
            y0=0,
            x1=timestamps[timestamp_idx],
            y1=max(distances),
            line=dict(color="red", width=2, dash="dash"),
        )

        fig_timeline.update_layout(
            title="Distance Between Objects Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Distance (m)",
            height=400,
            showlegend=True,
        )

        # === CAMERA VIEW PLOT ===

        fig_camera = go.Figure()

        try:
            # Load and process camera image
            if hasattr(camera, "image_path") and camera.image_path:
                img = Image.open(camera.image_path)

                # Project trajectories to image coordinates
                pose_matrix_inv = np.linalg.inv(np.array(frame.pose))

                # Transform trajectories to ego coordinates
                traj1_ego = np.hstack([traj1.reshape(-1, 3), np.ones((len(traj1), 1))])
                traj1_ego = np.matmul(traj1_ego, pose_matrix_inv.T)[:, :3]

                traj2_ego = np.hstack([traj2.reshape(-1, 3), np.ones((len(traj2), 1))])
                traj2_ego = np.matmul(traj2_ego, pose_matrix_inv.T)[:, :3]

                # Project to image coordinates
                proj_points1 = camera.project_to_image(traj1_ego, frame)
                proj_points1_ok = proj_points1[:, -1].astype(bool) & (
                    proj_points1[:, 2] > 0
                )
                proj_points1 = proj_points1[proj_points1_ok]

                proj_points2 = camera.project_to_image(traj2_ego, frame)
                proj_points2_ok = proj_points2[:, -1].astype(bool) & (
                    proj_points2[:, 2] > 0
                )
                proj_points2 = proj_points2[proj_points1_ok]

                # Add background image
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                fig_camera.add_layout_image(
                    dict(
                        source=f"data:image/png;base64,{img_str}",
                        xref="x",
                        yref="y",
                        x=0,
                        y=0,
                        sizex=img.width,
                        sizey=img.height,
                        sizing="stretch",
                        opacity=1.0,
                        layer="below",
                    )
                )

                # Add projected trajectories
                fig_camera.add_trace(
                    go.Scatter(
                        x=proj_points1[:, 0],
                        y=proj_points1[:, 1],
                        mode="lines+markers",
                        line=dict(color="cyan", width=4),
                        marker=dict(size=6, color="cyan"),
                        name=f"{obj1_desc} (projected)",
                        hovertemplate="<b>Image Coordinates</b><br>x: %{x}<br>y: %{y}<extra></extra>",
                    )
                )

                fig_camera.add_trace(
                    go.Scatter(
                        x=proj_points2[:, 0],
                        y=proj_points2[:, 1],
                        mode="lines+markers",
                        line=dict(color="magenta", width=4),
                        marker=dict(size=6, color="magenta"),
                        name=f"{obj2_desc} (projected)",
                        hovertemplate="<b>Image Coordinates</b><br>x: %{x}<br>y: %{y}<extra></extra>",
                    )
                )

                # Add waypoint markers
                waypoint_indices = [0, timestamp_idx, -1]
                waypoint_colors = ["green", "blue", "darkgreen"]
                waypoint_symbols = ["circle", "diamond", "square"]

                for i, (idx, color, symbol) in enumerate(
                    zip(waypoint_indices, waypoint_colors, waypoint_symbols)
                ):
                    if len(proj_points1) > abs(idx):
                        fig_camera.add_trace(
                            go.Scatter(
                                x=[proj_points1[idx, 0]],
                                y=[proj_points1[idx, 1]],
                                mode="markers",
                                marker=dict(
                                    size=15,
                                    color=color,
                                    symbol=symbol,
                                    line=dict(width=2, color="white"),
                                ),
                                name=f"{obj1_desc} ({'start' if i==0 else 'current' if i==1 else 'end'})",
                                showlegend=True,
                                hovertemplate=f"<b>{obj1_desc}</b><br>x: %{{x}}<br>y: %{{y}}<extra></extra>",
                            )
                        )

                    if len(proj_points2) > abs(idx):
                        ego_color = (
                            "red"
                            if color == "green"
                            else "orange" if color == "blue" else "darkred"
                        )
                        fig_camera.add_trace(
                            go.Scatter(
                                x=[proj_points2[idx, 0]],
                                y=[proj_points2[idx, 1]],
                                mode="markers",
                                marker=dict(
                                    size=15,
                                    color=ego_color,
                                    symbol=symbol,
                                    line=dict(width=2, color="white"),
                                ),
                                name=f"{obj2_desc} ({'start' if i==0 else 'current' if i==1 else 'end'})",
                                showlegend=True,
                                hovertemplate=f"<b>{obj2_desc}</b><br>x: %{{x}}<br>y: %{{y}}<extra></extra>",
                            )
                        )

                # Update camera view layout
                fig_camera.update_layout(
                    title=f"Camera View - {camera.name}",
                    xaxis=dict(
                        range=[0, img.width], title="X (pixels)", showgrid=False
                    ),
                    yaxis=dict(
                        range=[img.height, 0], title="Y (pixels)", showgrid=False
                    ),  # Flip Y axis
                    height=600,
                    showlegend=True,
                    plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                    annotations=[
                        dict(
                            text=(
                                f"<b>Scene Metadata</b><br>"
                                f"Scene ID: {obj.scene_id}<br>"
                                f"Timestamp: {frame.timestamp:.2f}s<br>"
                                f"Camera: {camera.name}<br>"
                                f"Question: {question_txt}<br>"
                                f"Answer: {answer_txt}"
                            ),
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.02,
                            y=0.98,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="gray",
                            borderwidth=1,
                            font=dict(size=10),
                        )
                    ],
                )
        except Exception as e:
            print(f"Error loading camera image: {e}")
            fig_camera.add_annotation(
                text="Camera image unavailable",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="red"),
            )
            fig_camera.update_layout(
                title="Camera View (Image Unavailable)", height=400
            )

        # === SAVE OR DISPLAY ===

        figures = {
            "3d_trajectory": fig_3d,
            "bev": fig_bev,
            "timeline": fig_timeline,
            "camera_view": fig_camera,
        }

        if save_path:
            base_path = save_path.replace(".html", "").replace(".png", "")

            if save_path.endswith(".html"):
                # Save as combined HTML with all three plots
                combined_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Trajectory Analysis - {obj1_desc}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .plot-container {{ margin: 20px 0; }}
                        h1, h2 {{ color: #333; }}
                    </style>
                </head>
                <body>
                    <h1>Trajectory Analysis: {obj1_desc}</h1>
                    <p><strong>Scene:</strong> {obj.scene_id} | <strong>Time:</strong> {frame.timestamp:.2f}s | <strong>Camera:</strong> {camera.name}</p>
                    
                    <div class="plot-container">
                        <h2>3D Trajectory View</h2>
                        {fig_3d.to_html(include_plotlyjs=False, div_id="3d-plot")}
                    </div>
                    
                    <div class="plot-container">
                        <h2>Timeline Analysis</h2>
                        {fig_timeline.to_html(include_plotlyjs=False, div_id="timeline-plot")}
                    </div>
                    
                    <div class="plot-container">
                        <h2>Camera View</h2>
                        {fig_camera.to_html(include_plotlyjs=False, div_id="camera-plot")}
                    </div>
                    
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                </body>
                </html>
                """

                with open(save_path, "w") as f:
                    f.write(combined_html)

            else:
                # Save individual plots as images
                fig_3d.write_image(
                    f"{base_path}_3d.png", width=1200, height=800, scale=2
                )
                fig_bev.write_image(
                    f"{base_path}_bev.png", width=1200, height=1200, scale=2
                )
                fig_timeline.write_image(
                    f"{base_path}_timeline.png", width=1200, height=600, scale=2
                )
                fig_camera.write_image(
                    f"{base_path}_camera.png", width=1200, height=800, scale=2
                )

            print(f"Plots saved with base path: {base_path}")
        else:
            # Display all figures
            fig_3d.show()
            fig_timeline.show()
            fig_camera.show()

        return figures

    def visualise_sample(
        self,
        question_obj: MultipleImageMultipleChoiceQuestion,
        answer_obj: MultipleChoiceAnswer,
        save_path,
        frames,
        pred_answer_obj: Optional[MultipleChoiceAnswer] = None,
        extra_text="",
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
        best_camera_name = question_obj.data["best_camera_name"]
        answer_description = question_obj.data["question_dict"]["answer_description"]

        if len(frames) > 0:
            timestamps = np.array([frame.timestamp for frame in frames], dtype=int)

            (trajectories, speeds, accels, ego_positions, ego_speeds, ego_vectors) = (
                self._calculate_trajectories(frames, timestamps)
            )

            frame = next(
                frame for frame in frames if frame.timestamp == question_obj.timestamp
            )

            best_camera = next(
                camera for camera in frame.cameras if camera.name == best_camera_name
            )

            question_txt = question_obj.question
            answer_txt = answer_obj.answer

            if pred_answer_obj is not None:
                extra_text = (
                    f"<br>Prediction: {pred_answer_obj.answer}"
                    if isinstance(pred_answer_obj, MultipleChoiceAnswer)
                    else f"<br>Raw Prediction: {pred_answer_obj.get_answer_text()}"
                )

            self.visualize_trajectories_plotly(
                object_id,
                trajectories,
                ego_positions,
                frame,
                best_camera,
                timestamps,
                save_path=str(save_path),
                question_txt=question_txt,
                answer_txt=answer_txt + extra_text,
                answer_description=answer_description,
            )
        else:
            from waymovqa.prompt_generators.object_prompts.object_drawn_box_prompt import (
                ObjectDrawnBoxPromptGenerator,
            )

            single_image_question = SingleImageMultipleChoiceQuestion(
                generator_name=question_obj.generator_name,
                data=question_obj.data,
                question_id=question_obj.question_id,
                image_path=question_obj.image_paths[
                    question_obj.camera_names.index(
                        question_obj.data["best_camera_name"]
                    )
                ],
                question=question_obj.question,
                choices=question_obj.choices,
                scene_id=question_obj.scene_id,
                timestamp=question_obj.timestamp,
                camera_name=question_obj.data["best_camera_name"],
            )

            box_prompt_generator = ObjectDrawnBoxPromptGenerator()
            box_prompt_generator.visualise_sample(
                single_image_question,
                answer_obj,
                save_path,
                frames,
                pred_answer_obj,
                extra_text,
                figsize,
                box_color,
                text_fontsize,
                title_fontsize,
                dpi,
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


@register_prompt_generator
class EgoRelativeObjectTrajectoryMultiFramePromptGenerator(
    EgoRelativeObjectTrajectoryPromptGenerator
):
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

        (trajectories, speeds, accels, ego_positions, ego_speeds, ego_vectors) = (
            self._calculate_trajectories(frames, timestamps)
        )

        obj_types = {
            obj.id: obj.get_simple_type() for frame in frames for obj in frame.objects
        }

        obj_cvat_types = {
            obj.id: obj.cvat_label for frame in frames for obj in frame.objects
        }

        for prompt_func in self.prompt_functions:
            prompt_out = prompt_func(
                trajectories,
                speeds,
                accels,
                ego_positions,
                ego_speeds,
                ego_vectors,
                timestamps,
                frames,
                obj_types,
                obj_cvat_types,
            )

            #
            choices, questions = prompt_out

            for question_dict in questions:
                try:
                    timestamp = question_dict["timestamp"]
                    object_id = question_dict["object_id"]
                    question_txt = question_dict["question"]
                    answer_txt = question_dict["answer"]
                except TypeError as e:
                    print("Typeerror:", e)
                    print("questions", type(questions))
                    print("questions", questions)
                    print("question_dict", type(question_dict))
                    print("prompt_func", prompt_func)
                    print("prompt_func.__name__", prompt_func.__name__)
                    print("question_dict", question_dict)
                    raise e

                frame_idx = next(
                    (idx for idx, time in enumerate(timestamps) if time == timestamp),
                    None,
                )

                if frame_idx is None:
                    continue

                if frame_idx == 0:
                    frame_idx = (
                        10  # make it the second frame if its currently the first
                    )

                previous_frame = frames[frame_idx - 1]
                current_frame = frames[frame_idx]

                current_obj = next(
                    (obj for obj in current_frame.objects if obj.id == object_id), None
                )
                previous_obj = next(
                    (obj for obj in previous_frame.objects if obj.id == object_id), None
                )

                if current_obj is None or previous_obj is None:
                    continue

                assert current_obj.id == object_id and previous_obj.id == object_id
                best_camera_name = previous_obj.most_visible_camera_name

                current_camera = next(
                    (
                        cam
                        for cam in current_frame.cameras
                        if best_camera_name == cam.name
                    ),
                    None,
                )

                previous_camera = next(
                    (
                        cam
                        for cam in previous_frame.cameras
                        if best_camera_name == cam.name
                    ),
                    None,
                )

                if current_camera is None or previous_camera is None:
                    continue

                if not current_obj.is_visible_on_camera(
                    current_frame, current_camera
                ) or not previous_obj.is_visible_on_camera(
                    previous_frame, previous_camera
                ):
                    continue

                if len(choices) > CHOICES_OPTIONS:
                    # Remove correct answer from choices
                    distractors = [x for x in choices if x != answer_txt]

                    # Sample (CHOICES_OPTIONS - 1) distractors
                    sampled = np.random.choice(
                        distractors, CHOICES_OPTIONS - 1, replace=False
                    ).tolist()

                    # Add the correct answer back in
                    choices = sampled + [answer_txt]

                    # (Optional) Shuffle the final choices
                    np.random.shuffle(choices)

                if len(choices) < 2:
                    print(f"Received only {len(choices)} choices from {prompt_func}")
                    continue

                if answer_txt is None or answer_txt not in choices:
                    print(
                        f"answer_txt={answer_txt} is none or not in choices={choices}"
                    )
                    continue

                question = MultipleFrameMultipleChoiceQuestion(
                    current_image_path=current_camera.image_path,
                    previous_image_path=previous_camera.image_path,
                    question=question_txt,
                    choices=choices,
                    scene_id=current_frame.scene_id,
                    current_timestamp=current_frame.timestamp,
                    previous_timestamp=previous_frame.timestamp,
                    camera_name=best_camera_name,
                    generator_name=f"{self.__class__.__module__}.{self.__class__.__name__}",
                    question_name=prompt_func.__name__,
                    data=dict(
                        question_dict=question_dict,
                    ),
                    current_bbox=current_obj.get_object_bbox_2d(
                        current_frame, current_camera
                    ),
                    previous_bbox=previous_obj.get_object_bbox_2d(
                        previous_frame, previous_camera
                    ),
                )

                answer = MultipleChoiceAnswer(
                    choices=choices,
                    answer=answer_txt,
                )

                samples.append((question, answer))

        random.shuffle(samples)

        return samples

    def get_question_type(self):
        return MultipleFrameMultipleChoiceQuestion
