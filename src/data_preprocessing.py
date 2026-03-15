import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import TRAIN_DIR, TEST_DIR, VALIDATION_DIR, CLASS_NAMES_PATH, CONFIG_PATH
from utils.common import read_yaml

logger = get_logger(__name__)

# ImageNet mean and std — used because VGG16 was pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DataPreprocessor:

    def __init__(self, config):
        self.config = config["data_processing"]
        self.image_size = self.config["image_size"]
        self.batch_size = self.config["batch_size"]
        self.num_workers = self.config["num_workers"]

        logger.info(f"DataPreprocessor initialized — image_size={self.image_size}, batch_size={self.batch_size}")

    def get_transforms(self):
        """
        Train: augmentation (random flip, rotation, color jitter) to improve generalization.
        Val/Test: only resize + normalize — no augmentation to get unbiased evaluation.
        """
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        eval_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        return train_transform, eval_transform

    def get_dataloaders(self):
        """Build PyTorch DataLoaders for train, validation, and test splits."""
        try:
            train_transform, eval_transform = self.get_transforms()

            train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
            val_dataset = datasets.ImageFolder(root=VALIDATION_DIR, transform=eval_transform)
            test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=eval_transform)

            # Verify class alignment across splits — folder order must match
            assert train_dataset.classes == val_dataset.classes == test_dataset.classes, (
                "Class mismatch between dataset splits"
            )

            class_names = train_dataset.classes
            logger.info(f"Classes: {class_names}")
            logger.info(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            return train_loader, val_loader, test_loader, class_names

        except Exception as e:
            logger.error(f"Error building dataloaders: {e}")
            raise CustomException("Failed to build dataloaders", e)


if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    preprocessor = DataPreprocessor(config)
    train_loader, val_loader, test_loader, class_names = preprocessor.get_dataloaders()
    logger.info(f"Dataloaders ready. Classes: {class_names}")
