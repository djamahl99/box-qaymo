import argparse

from waymovqa.data.vqa_dataset import VQADataset


def main():
    """Main function to generate and save a VQA dataset."""

    parser = argparse.ArgumentParser(
        description="Generate VQA dataset from processed data"
    )
    parser.add_argument(
        "--gt_dataset_path",
        type=str,
        required=True,
        help="Path to ground truth vqa dataset",
    )
    parser.add_argument(
        "--pred_dataset_path",
        type=str,
        required=True,
        help="Path to prediction vqa dataset",
    )
    parser.add_argument("--model", type=str, required=True, help="Type of model")

    args = parser.parse_args()

    pred_dataset = VQADataset.load_dataset(args.pred_dataset_path)
    gt_dataset = VQADataset.load_dataset(args.gt_dataset_path)


if __name__ == "__main__":
    main()
