import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(
            f"Data ingestion started with {self.bucket_name} and file is {self.file_name}"
        )

    def download_csv_from_gcp(self):
        try:
            client = storage.Client(
                project=self.config["project_id"]
            )  # Explicitly pass project_id
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"CSV file downloaded from GCP bucket to: {RAW_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error downloading CSV from GCP: {e}")
            raise CustomException("Failed to download CSV from GCP", e)

    def split_data(self):
        try:
            logger.info("Splitting data into train and test sets")
            data = pd.read_csv(RAW_FILE_PATH)

            train_data, test_data = train_test_split(
                data, test_size=1 - self.train_test_ratio, random_state=42
            )

            train_data.to_csv(TRAIN_FILE_PATH)
            test_data.to_csv(TEST_FILE_PATH)
            logger.info(f"Train data saved to: {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to: {TEST_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error while splitting data: {e}")
            raise CustomException("Failed to split data!", e)

    def run(self):
        try:
            logger.info("Starting Data ingestion")

            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully")
        except CustomException as e:
            logger.error(f"Error in data ingestion: {str(e)}")

        finally:
            logger.info("Data ingestion process finished")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
