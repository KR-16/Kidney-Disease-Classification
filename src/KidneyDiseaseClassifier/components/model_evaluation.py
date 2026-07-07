import os
from pathlib import Path
from urllib.parse import urlparse

import tensorflow as tf
import mlflow
import mlflow.keras

from KidneyDiseaseClassifier import logger
from KidneyDiseaseClassifier.entity import EvaluationConfig
from KidneyDiseaseClassifier.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None

    def _valid_generator(self):
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.30)
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=self.config.scores_path, data=scores)
        logger.info(f"Evaluation scores: {scores}")

    def log_into_mlflow(self):
        # Prefer the config value; otherwise fall back to the standard
        # MLFLOW_TRACKING_URI env var (how DagsHub credentials are wired in).
        tracking_uri = self.config.mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "")
        if not tracking_uri:
            logger.info(
                "No MLflow tracking URI configured (config.evaluation.mlflow_uri "
                "or MLFLOW_TRACKING_URI); skipping MLflow logging."
            )
            return

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model, "model", registered_model_name="VGG16Model"
                )
            else:
                mlflow.keras.log_model(self.model, "model")
        logger.info("Logged run into MLflow")
