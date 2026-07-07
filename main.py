from src.KidneyDiseaseClassifier import logger
from src.KidneyDiseaseClassifier.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.KidneyDiseaseClassifier.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from src.KidneyDiseaseClassifier.pipeline.stage_03_model_training import (
    ModelTrainingPipeline,
)
from src.KidneyDiseaseClassifier.pipeline.stage_04_model_evaluation import (
    ModelEvaluationPipeline,
)

STAGES = [
    ("Data Ingestion stage", DataIngestionTrainingPipeline),
    ("Prepare Base Model stage", PrepareBaseModelTrainingPipeline),
    ("Model Training stage", ModelTrainingPipeline),
    ("Model Evaluation stage", ModelEvaluationPipeline),
]


def main():
    for stage_name, pipeline_cls in STAGES:
        try:
            logger.info(f">>>>>> stage {stage_name} started <<<<<<")
            pipeline_cls().main()
            logger.info(
                f">>>>>> stage {stage_name} completed <<<<<<\n\nx==========x"
            )
        except Exception as e:
            logger.exception(e)
            raise e


if __name__ == "__main__":
    main()
