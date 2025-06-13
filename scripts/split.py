from box_qaymo.waymo_loader import WaymoDatasetLoader
import random
from pathlib import Path


def main():
    dataset_path = Path("/media/local-data/uqdetche/waymo_vqa_dataset")
    loader = WaymoDatasetLoader(dataset_path)

    scene_ids = loader.get_scene_ids()

    random.seed(42)
    
    train_size = 101
    # train_size = 202 - 42 
    # val_size = 42

    random.shuffle(scene_ids)

    train_split = set(scene_ids[:train_size])
    val_split = set(scene_ids[train_size:])

    print("train_split", len(train_split))
    print("val_split", len(val_split))

    print("intersection", train_split.intersection(val_split))
    # print("diff", train_split.difference(val_split))

    split_dir = dataset_path / "splits"
    split_dir.mkdir(exist_ok=True)

    with open(split_dir / "train.txt", "w") as f:
        f.writelines([f"{x}\n" for x in train_split])

    with open(split_dir / "val.txt", "w") as f:
        f.writelines([f"{x}\n" for x in val_split])


if __name__ == "__main__":
    main()
