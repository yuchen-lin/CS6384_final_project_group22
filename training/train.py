from ultralytics import YOLO
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(
    data_yaml: str = "dataset.yaml",
    model_name: str = "yolo11s.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",
) -> None:
    try:
        model = YOLO(model_name)

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            verbose=True,
            patience=20,
            save=True,
            project="runs/train",
            name="nutrition_label_detector",
        )

        logger.info(f"Training completed. Results saved in: {results}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    train_model()
