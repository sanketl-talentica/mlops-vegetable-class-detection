from src.data_ingestion import DataIngestion
from src.model_training import ModelTraining
from utils.common import read_yaml
from config.paths_config import CONFIG_PATH


if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)

    ### 1. Data Ingestion — validate dataset splits and save class names
    data_ingestion = DataIngestion(config)
    data_ingestion.run()

    ### 2. Model Training — builds dataloaders, trains VGG16, logs to MLflow
    trainer = ModelTraining(config)
    trainer.run()
