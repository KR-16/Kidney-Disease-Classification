import os
import zipfile

import gdown

from KidneyDiseaseClassifier import logger
from KidneyDiseaseClassifier.entity import DataIngestionConfig
from KidneyDiseaseClassifier.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """Fetch the dataset zip from a Google Drive share link."""
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, str(zip_download_dir), quiet=False)

            logger.info(f"Downloaded data into {zip_download_dir}")
            return str(zip_download_dir)
        except Exception as e:
            logger.exception("Failed to download the dataset")
            raise e

    def extract_zip_file(self):
        """Extract the downloaded zip into the unzip directory."""
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(
            f"Extracted {self.config.local_data_file} "
            f"({get_size(self.config.local_data_file)}) into {unzip_path}"
        )
