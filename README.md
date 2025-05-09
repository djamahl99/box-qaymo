# Waymo VQA

This code creates our dataset of VQA pairs from Waymo.

## Setup

### Data
1. Downlod ```waymo_open_dataset_v_1_4_3``` validation tfrecords
2. Setup waymo_env following ```requirements_extraction.txt```, this environment will be used for extracting Waymo only.
3. Extract Waymo tfrecord data using ```waymo_extract.py```
    - we recommend just having an environment just for this that has an environment with ```waymo-open-dataset-tf-2-12-0==1.6.4``` to handle 
4. Download ```objectid_to_label.json``` and ```objectid_to_color.json``` and place in ```./data```

<!-- table with download links for objectid_to_label and objectid_to_color -->

### Generate the dataset

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


### Types of Prompts
1. Number of {x} object
2. 