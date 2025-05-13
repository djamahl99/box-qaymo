# WaymoVQA Code Structure

## Overview
WaymoVQA is a comprehensive framework for generating, processing, and evaluating visual question answering (VQA) tasks on the Waymo dataset. The codebase facilitates creating diverse question types about scenes, evaluating model responses using various metrics, and supporting different answer formats.

## Setup

### Data
1. Downlod ```waymo_open_dataset_v_1_4_3``` validation tfrecords
2. Setup waymo_env following ```requirements_extraction.txt```, this environment will be used for extracting Waymo only.
3. Extract Waymo tfrecord data using ```waymo_extract.py```
    - we recommend just having an environment just for this that has an environment with ```waymo-open-dataset-tf-2-12-0==1.6.4``` to handle 
4. Download ```objectid_to_label.json``` and ```objectid_to_color.json``` and place in ```./data```

### Waymo Dataset Preprocessing 

```waymo_extract.py```

Exports Waymo tfrecord files to json, jpg, npy files for metainfo, camera images and LiDAR respectively.

- Need to export object information separately to allow for us to generate prompts etc.
- Probably necessary to have per frame object information, as the prompt might be possible for one timestamp but not the next.
- Probably save scene information separately?

Structure
```
scene_infos
    - {scene_id}.json

object_infos
    - object_{object_id}

camera_images
    - {scene_id}_{timestamp}_{camera_name}.jpg

point_clouds
    - {scene_id}_{timestamp}_{lidar_name}.npy

```

### VQA Dataset Generation

```
python vqa_generator.py --dataset_path /media/local-data/uqdetche/waymo_vqa --generators Grounding2DPromptGenerator --method object --total_samples 20 --model 
```

### Models to Evaluate

1. SENNA
2. Language Prompt for Autonomous Driving https://github.com/wudongming97/Prompt4Driving
3. https://github.com/ThierryDeruyttere/vilbert-Talk2car
4. 

VLMs?
1. OWLViT
2. Grounding DINO

 - we can evaluate these as 2d grounding models


## Core Components

### `waymovqa.data`
Data representation and loading utilities for Waymo dataset.

- **`DataObject`**: Base class for all data objects
- **`CameraInfo`**: Camera parameters and metadata
- **`FrameInfo`**: Information about a single frame from a scene
- **`LaserInfo`**: LiDAR data representation
- **`ObjectInfo`**: Object information
- **`SceneInfo`**: Complete scene representation with frames, objects, cameras, lidar etc
- **`VQADataset`**: Dataset container for VQA tasks
- **`WaymoDatasetLoader`**: Handles extraction and loading of Waymo data

### `waymovqa.metrics`
Evaluation metrics for different answer types.

- **`BaseMetric<T>`**: Generic base class for all metrics
- **`COCOMetric`**: Object detection evaluation using COCO metrics (IoU, mAP)
- **`COCOEvaluator`**: Higher-level evaluator for COCO metrics across IoU thresholds
- **`MultipleChoiceMetric`**: Accuracy metrics for multiple-choice answers
- **`TextSimilarityMetric`**: Advanced NLP-based text similarity metrics
- **`RougeMetric`**: ROUGE metrics for text comparison
- **`BERTScoreMetric`**: BERT-based semantic similarity metrics
- **`SimpleTextSimilarityMetric`**: Lightweight text similarity without dependencies

### `waymovqa.models`
Model interfaces and implementations.

- **`BaseModel`**: Abstract interface for all VQA models
- **`Grounding2DModel`**: Model for 2D object localization tasks

### `waymovqa.prompts`
Prompt generators for different question types.

- **`BasePromptGenerator`**: Abstract base for all prompt generators
- **Scene Prompts**:
  - **`SceneDescriptionPromptGenerator`**: Generates prompts for scene descriptions
- **Object Prompts**:
  - **`Grounding2DPromptGenerator`**: Prompts for 2D object localization
  - **`ObjectColorPromptGenerator`**: Prompts about object colors
  - **`ObjectLocationPromptGenerator`**: Prompts about object positions
  - **`ObjectRelationPromptGenerator`**: Prompts about relationships between objects

### `waymovqa.questions`
Question types and representations.

- **`QuestionType`**: Enumeration of supported question types
- **`BaseQuestion`**: Abstract base for all question types
- **`SingleImageQuestion`**: Questions about a single image
- **`MultipleImageQuestion`**: Questions involving multiple images
- **`MultiChoiceSingleImageQuestion`**: Multiple-choice questions about a single image

### `waymovqa.answers`
Answer representations for different question types.

- **`AnswerType`**: Enumeration of supported answer types
- **`BaseAnswer`**: Abstract base for all answer types
- **`Object2DAnswer`**: 2D bounding box answers
- **`MultiObject2DAnswer`**: Multiple 2D bounding box answers
- **`MultipleChoiceAnswer`**: Multiple-choice answer representation
- **`RawTextAnswer`**: Free-form text answers

## Utility Scripts

- **`validate.py`**: Validates model predictions and gt vqa dataset
- **`vqa_generator.py`**: Generates VQA samples from Waymo data for a given model
- **`waymo_extract.py`**: Extracts and processes Waymo dataset (preprocessing)

### Scripts Directory
- **`gdino_predict.py`**: Prediction script using Grounding DINO model
- **`owlvit_predict.py`**: Prediction script using OWL-ViT model
- **`senna_predict.py`**: Prediction script using SENNA model

## Data Flow

1. **Data Loading**: `WaymoDatasetLoader` extracts and processes Waymo data into `SceneInfo` objects
2. **Question Generation**: Prompt generators create questions based on scene information
3. **Model Execution**: Models process questions and generate answers
4. **Evaluation**: Metrics evaluate model answers against ground truth

## Extension Points

The framework is designed for extensibility through:

1. **New question types**: Add new classes inheriting from `BaseQuestion`
2. **New prompt generators**: Create specialized generators for specific question types
3. **Additional metrics**: Implement custom metrics by extending `BaseMetric`
4. **Model integration**: Support new models by implementing the `BaseModel` interface

## Configuration
Configuration files in the `configs/` directory control:
- Dataset paths and processing parameters
- Model settings and hyperparameters
- Evaluation metrics and thresholds