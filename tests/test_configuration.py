from KidneyDiseaseClassifier.config.configuration import ConfigurationManager
from KidneyDiseaseClassifier.entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)


def test_configuration_manager_builds_all_configs():
    manager = ConfigurationManager()

    assert isinstance(manager.get_data_ingestion_config(), DataIngestionConfig)
    assert isinstance(manager.get_prepare_base_model_config(), PrepareBaseModelConfig)
    assert isinstance(manager.get_training_config(), TrainingConfig)
    assert isinstance(manager.get_evaluation_config(), EvaluationConfig)


def test_params_are_wired_through():
    manager = ConfigurationManager()
    base = manager.get_prepare_base_model_config()
    assert base.params_classes == manager.params.CLASSES
    assert list(base.params_image_size) == list(manager.params.IMAGE_SIZE)
