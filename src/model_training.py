import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_preprocessing import DataPreprocessor
from config.paths_config import MODEL_OUTPUT_PATH, MODEL_DIR, CLASS_NAMES_PATH, CONFIG_PATH
from config.model_params import VGG16_PARAMS
from utils.common import read_yaml

import dagshub
import mlflow
import mlflow.pytorch
from dotenv import load_dotenv

load_dotenv()
dagshub.init(
    repo_owner=os.environ["DAGSHUB_USERNAME"],
    repo_name=os.environ["DAGSHUB_REPO"],
    mlflow=True
)

logger = get_logger(__name__)


class ModelTraining:

    def __init__(self, config):
        self.config = config
        self.params = VGG16_PARAMS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def build_model(self):
        """
        Load pretrained VGG16 and replace the final classifier layer to match num_classes.
        freeze_backbone=True trains only the new classifier head — faster, less overfitting on small datasets.
        """
        try:
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

            if self.params["freeze_backbone"]:
                for param in model.features.parameters():
                    param.requires_grad = False
                logger.info("VGG16 backbone frozen — only classifier head will be trained")

            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, self.params["num_classes"])

            model = model.to(self.device)
            logger.info(f"VGG16 model built — output classes: {self.params['num_classes']}")
            return model

        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise CustomException("Failed to build VGG16 model", e)

    def train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / total, correct / total

    def eval_epoch(self, model, loader, criterion):
        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return running_loss / total, correct / total

    def collect_predictions(self, model, loader):
        """Run inference on a loader and return all true labels + predicted labels as numpy arrays."""
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_labels), np.array(all_preds)

    def save_model(self, model):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
            logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise CustomException("Failed to save model", e)

    def run(self):
        try:
            mlflow.set_experiment("Vegetable Classification VGG16")
            with mlflow.start_run():
                mlflow.set_tag("model_type", "VGG16")
                mlflow.set_tag("dataset", "Vegetable_7_Classes")
                mlflow.log_params(self.params)

                logger.info("Starting model training pipeline")

                # Load class names for evaluation reporting
                with open(CLASS_NAMES_PATH, "r") as f:
                    class_names = json.load(f)

                # Build dataloaders
                preprocessor = DataPreprocessor(self.config)
                train_loader, val_loader, test_loader, _ = preprocessor.get_dataloaders()

                model = self.build_model()
                criterion = nn.CrossEntropyLoss()

                # Only optimize parameters that require gradients (classifier head when backbone frozen)
                optimizer = optim.SGD(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=self.params["learning_rate"],
                    momentum=self.params["momentum"],
                    weight_decay=self.params["weight_decay"],
                )
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.params["step_size"],
                    gamma=self.params["gamma"],
                )

                best_val_acc = 0.0

                for epoch in range(1, self.params["epochs"] + 1):
                    train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
                    val_loss, val_acc = self.eval_epoch(model, val_loader, criterion)
                    scheduler.step()

                    logger.info(
                        f"Epoch {epoch}/{self.params['epochs']} — "
                        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                    )
                    mlflow.log_metrics({
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }, step=epoch)

                    # Save best model based on validation accuracy
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        self.save_model(model)
                        logger.info(f"New best model saved (val_acc={val_acc:.4f})")

                # ── Test evaluation ───────────────────────────────────────────
                model.load_state_dict(torch.load(MODEL_OUTPUT_PATH, map_location=self.device))
                test_loss, test_acc = self.eval_epoch(model, test_loader, criterion)
                logger.info(f"Test — loss={test_loss:.4f}, acc={test_acc:.4f}")

                # Per-class metrics + confusion matrix
                y_true, y_pred = self.collect_predictions(model, test_loader)

                report_str = classification_report(y_true, y_pred, target_names=class_names)
                logger.info(f"Classification Report:\n{report_str}")

                cm = confusion_matrix(y_true, y_pred)
                logger.info(f"Confusion Matrix:\n{cm}")

                with open("classification_report.txt", "w") as f:
                    f.write(report_str)
                mlflow.log_artifact("classification_report.txt")

                cm_lines = ["Confusion Matrix (rows=true, cols=predicted):", ""]
                header = "\t".join([""] + class_names)
                cm_lines.append(header)
                for i, row in enumerate(cm):
                    cm_lines.append(class_names[i] + "\t" + "\t".join(map(str, row)))
                with open("confusion_matrix.txt", "w") as f:
                    f.write("\n".join(cm_lines))
                mlflow.log_artifact("confusion_matrix.txt")

                # Log per-class F1 to MLflow
                report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
                per_class_f1 = {f"f1_{cls.replace(' ', '_')}": report_dict[cls]["f1-score"] for cls in class_names}
                mlflow.log_metrics(per_class_f1)

                metrics = {
                    "best_val_acc": best_val_acc,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "macro_f1": report_dict["macro avg"]["f1-score"],
                    "weighted_f1": report_dict["weighted avg"]["f1-score"],
                }
                mlflow.log_metrics(metrics)

                with open("metrics.json", "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.info("metrics.json saved")

                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="VegetableClassificationVGG16",
                )

                logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            raise CustomException("Failed during model training pipeline", e)


if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    trainer = ModelTraining(config)
    trainer.run()
