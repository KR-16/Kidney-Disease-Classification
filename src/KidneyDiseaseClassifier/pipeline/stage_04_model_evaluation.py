from KidneyDiseaseClassifier import logger
from KidneyDiseaseClassifier.config.configuration import ConfigurationManager
from KidneyDiseaseClassifier.components.model_evaluation import Evaluation

STAGE_NAME = "Model Evaluation stage"


class ModelEvaluationPipeline:
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        ModelEvaluationPipeline().main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
