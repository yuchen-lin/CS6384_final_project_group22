import os
import pandas as pd
import numpy as np
from PIL import Image
import io
import yaml


# Convert bounding box coordinates from (y_min, x_min, y_max, x_max) to YOLO format
def convert_bbox_coordinates(bbox, width, height):
    y_min, x_min, y_max, x_max = bbox

    # Calculate center points and dimensions
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    return [x_center, y_center, bbox_width, bbox_height]


# Converts parquet files to YOLO format
def process_dataset(parquet_files, dataset_type="train"):
    processed_count = 0

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)

        for _, row in df.iterrows():
            image_id = row["image_id"]
            image_path = f"datasets/images/{dataset_type}/{image_id}.jpg"
            label_path = f"datasets/labels/{dataset_type}/{image_id}.txt"

            # Skip if image already exists
            if os.path.exists(image_path):
                print(f"Skipping existing image: {image_id}")
                continue

            # Save image
            image_bytes = row["image"]["bytes"]
            image = Image.open(io.BytesIO(image_bytes))
            image.save(image_path)

            # Process bounding boxes
            bboxes = row["objects"]["bbox"]
            with open(label_path, "w") as f:
                for bbox in bboxes:
                    yolo_bbox = convert_bbox_coordinates(
                        bbox, row["width"], row["height"]
                    )
                    f.write(f"0 {' '.join(map(str, yolo_bbox))}\n")

            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images for {dataset_type} dataset")


def create_dataset_yaml():
    dataset_config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "nutrition-table"},
    }

    with open("dataset.yaml", "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)


def main():
    dirs = [
        "datasets/images/train",
        "datasets/images/val",
        "datasets/labels/train",
        "datasets/labels/val",
    ]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

    # Process training data
    train_files = [
        "datasets/train-00000-of-00002.parquet",
        "datasets/train-00001-of-00002.parquet",
    ]
    process_dataset(train_files, "train")

    # Process validation data
    val_files = ["datasets/val-00000-of-00001.parquet"]
    process_dataset(val_files, "val")

    # Create dataset.yaml
    create_dataset_yaml()


if __name__ == "__main__":
    main()
