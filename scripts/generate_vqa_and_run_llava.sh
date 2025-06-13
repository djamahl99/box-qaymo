if [[ "$CONDA_DEFAULT_ENV" != "llava" ]]; then
    conda activate llava
fi

# TODO: add installation lines from llava: add cv2, protobuf

# generate toy dataset
# python vqa_generator.py --dataset_path /media/local-data/uqdetche/waymo_vqa_dataset/ --generators SceneSingleImageMultiChoicePromptGenerator --total_samples 20 --save_path SceneSingleImageMultiChoicePromptGenerator.json

# run on llava
python scripts/llava_predict.py --dataset_path /media/local-data/uqdetche/waymo_vqa_dataset/ --vqa_path ./SceneSingleImageMultiChoicePromptGenerator.json --save_path ./SceneSingleImageMultiChoicePromptGenerator_llava_preds.json --batch_size 1