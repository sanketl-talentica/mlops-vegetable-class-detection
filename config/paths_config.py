import os

CONFIG_PATH = "config/config.yaml"

########################### DATA INGESTION #########################

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
VALIDATION_DIR = "dataset/validation"

######################## ARTIFACTS ########################

ARTIFACTS_DIR = "artifacts"
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "vgg16_model.pth")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
