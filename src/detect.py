import torch
from yolov5 import train, val, detect
import os

def train_yolov5(data_yaml, weights="yolov5s.pt", epochs=50, batch_size=16):
    """Train YOLOv5 model."""
    train.run(
        data=data_yaml,
        weights=weights,
        epochs=epochs,
        batch_size=batch_size,
        project="output",
        name="yolov5_train",
        device=0 if torch.cuda.is_available() else "cpu"
    )

def evaluate_yolov5(data_yaml, weights):
    """Evaluate YOLOv5 model on test set."""
    val.run(
        data=data_yaml,
        weights=weights,
        project="output",
        name="yolov5_eval",
        device=0 if torch.cuda.is_available() else "cpu"
    )

def detect_nid(image_path, weights, output_dir):
    """Detect NID in an image."""
    detect.run(
        source=image_path,
        weights=weights,
        project=output_dir,
        name="detections",
        save_crop=True,
        device=0 if torch.cuda.is_available() else "cpu"
    )
    # Return path to cropped NID image
    return os.path.join(output_dir, "detections", "crops", "NID", os.path.basename(image_path))

# Example usage
if __name__ == "__main__":
    data_yaml = "data/dataset.yaml"
    weights = "models/yolov5s.pt"
    train_yolov5(data_yaml, weights, epochs=50, batch_size=16)
    evaluate_yolov5(data_yaml, "output/yolov5_train/weights/best.pt")
    # Detect example image
    detect_nid("data/images/sample.jpg", "output/yolov5_train/weights/best.pt", "output")