import os
import json
import hashlib
from PIL import Image
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import TRAIN_DIR, TEST_DIR, VALIDATION_DIR, MODEL_DIR, CLASS_NAMES_PATH, CONFIG_PATH
from utils.common import read_yaml

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.train_dir = self.config["train_dir"]
        self.test_dir = self.config["test_dir"]
        self.validation_dir = self.config["validation_dir"]

        logger.info(f"DataIngestion initialized — train: {self.train_dir}, test: {self.test_dir}, val: {self.validation_dir}")

    def validate_dataset(self):
        """Validate that all three splits exist and have the same class folders."""
        try:
            for split_dir in [self.train_dir, self.test_dir, self.validation_dir]:
                if not os.path.isdir(split_dir):
                    raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

            train_classes = sorted(os.listdir(self.train_dir))
            test_classes = sorted(os.listdir(self.test_dir))
            val_classes = sorted(os.listdir(self.validation_dir))

            if train_classes != test_classes or train_classes != val_classes:
                raise ValueError(
                    f"Class mismatch across splits.\n"
                    f"Train: {train_classes}\nTest: {test_classes}\nVal: {val_classes}"
                )

            logger.info(f"Dataset validated — {len(train_classes)} classes found: {train_classes}")
            return train_classes

        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            raise CustomException("Failed to validate dataset", e)

    def count_images(self):
        """Log image counts per split for visibility."""
        try:
            for split_name, split_dir in [("train", self.train_dir), ("test", self.test_dir), ("validation", self.validation_dir)]:
                total = sum(
                    len(files)
                    for _, _, files in os.walk(split_dir)
                )
                logger.info(f"{split_name}: {total} images")
        except Exception as e:
            logger.error(f"Error counting images: {e}")
            raise CustomException("Failed to count images", e)

    def check_corrupted_images(self):
        """Try opening every image — log and collect any files that cannot be read."""
        try:
            corrupted = []
            for split_name, split_dir in [("train", self.train_dir), ("test", self.test_dir), ("validation", self.validation_dir)]:
                for root, _, files in os.walk(split_dir):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            with Image.open(fpath) as img:
                                img.verify()   # verify header without fully decoding
                        except Exception:
                            corrupted.append(fpath)
                            logger.warning(f"Corrupted image: {fpath}")

            if corrupted:
                logger.warning(f"Total corrupted images found: {len(corrupted)}")
            else:
                logger.info("No corrupted images found")

            return corrupted
        except Exception as e:
            logger.error(f"Error during corruption check: {e}")
            raise CustomException("Failed to check corrupted images", e)

    def check_duplicate_images(self):
        """Hash every image — flag files with identical content across the full dataset."""
        try:
            seen_hashes = {}   # hash -> first file path
            duplicates = []

            for split_name, split_dir in [("train", self.train_dir), ("test", self.test_dir), ("validation", self.validation_dir)]:
                for root, _, files in os.walk(split_dir):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        try:
                            with open(fpath, "rb") as f:
                                file_hash = hashlib.md5(f.read()).hexdigest()
                            if file_hash in seen_hashes:
                                duplicates.append((fpath, seen_hashes[file_hash]))
                                logger.warning(f"Duplicate: {fpath} == {seen_hashes[file_hash]}")
                            else:
                                seen_hashes[file_hash] = fpath
                        except Exception:
                            pass   # corrupted files already caught in check_corrupted_images

            if duplicates:
                logger.warning(f"Total duplicate image pairs found: {len(duplicates)}")
            else:
                logger.info("No duplicate images found")

            return duplicates
        except Exception as e:
            logger.error(f"Error during duplicate check: {e}")
            raise CustomException("Failed to check duplicate images", e)

    def save_class_names(self, class_names):
        """Persist class names to JSON so inference can map indices to labels without reloading the dataset."""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(CLASS_NAMES_PATH, "w") as f:
                json.dump(class_names, f, indent=4)
            logger.info(f"Class names saved to {CLASS_NAMES_PATH}")
        except Exception as e:
            logger.error(f"Error saving class names: {e}")
            raise CustomException("Failed to save class names", e)

    def run(self):
        try:
            logger.info("Starting data ingestion")
            class_names = self.validate_dataset()
            self.count_images()
            self.check_corrupted_images()
            self.check_duplicate_images()
            self.save_class_names(class_names)
            logger.info("Data ingestion completed successfully")
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
            raise
        finally:
            logger.info("Data ingestion finished")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
