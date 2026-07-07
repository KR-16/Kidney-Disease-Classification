# Kidney Disease Classification

An end-to-end **deep-learning** pipeline that classifies kidney **CT-scan images**
(e.g. *Normal* vs *Tumor*) using **VGG16 transfer learning**. The project is built as a
reproducible MLOps workflow with **DVC** for pipeline orchestration, **MLflow** for
experiment tracking, and a **Flask** web app for inference.

> **Note:** this repository expects an image dataset (CT scans organized into one
> subfolder per class), *not* the tabular UCI Chronic Kidney Disease dataset.

## Pipeline

```
Data Ingestion  ->  Prepare Base Model  ->  Training  ->  Evaluation (MLflow)
   (gdown)          (VGG16 + head)         (fit)         (scores.json + MLflow)
```

Each stage is implemented as a component (`src/KidneyDiseaseClassifier/components`)
driven by a thin pipeline wrapper (`.../pipeline`) and configured centrally through
`config/config.yaml` + `params.yaml` via the `ConfigurationManager`.

## Project structure

```
Kidney-Disease-Classification/
├── config/
│   └── config.yaml                 # paths & data source per stage
├── params.yaml                     # model / training hyperparameters
├── dvc.yaml                        # DVC pipeline (4 stages)
├── main.py                         # runs all stages sequentially
├── app.py                          # Flask prediction web app
├── templates/index.html            # upload + predict UI
├── src/KidneyDiseaseClassifier/
│   ├── components/                 # data_ingestion, prepare_base_model,
│   │                               # model_training, model_evaluation
│   ├── config/configuration.py     # ConfigurationManager
│   ├── constants/                  # CONFIG/PARAMS file paths
│   ├── entity/                     # config dataclasses
│   ├── pipeline/                   # stage_01..04 + prediction
│   └── utils/common.py             # yaml/json/bin/image helpers
├── tests/                          # pytest suite
└── requirements.txt / pyproject.toml
```

## Requirements

Runtime dependencies are declared in `pyproject.toml`. Install with either tool:

```bash
# with uv (recommended)
uv sync

# or with pip
pip install -r requirements.txt
```

Python **3.12+** is required.

## Configuration

1. Put your dataset zip on Google Drive and set its share link in
   `config/config.yaml` under `data_ingestion.source_URL`.
   The archive should extract to `artifacts/data_ingestion/kidney-ct-scan-image/`
   with one subfolder per class.
2. Adjust hyperparameters in `params.yaml` (image size, batch size, epochs,
   learning rate, number of classes, augmentation).
3. (Optional) Set `evaluation.mlflow_uri` in `config/config.yaml` to log runs to a
   remote MLflow / DagsHub tracking server.

## Usage

Run the whole training pipeline:

```bash
python main.py
```

Or run it reproducibly with DVC (only re-runs stages whose inputs changed):

```bash
dvc repro
```

Serve the prediction web app:

```bash
python app.py        # http://localhost:8080
```

Run the tests:

```bash
pytest
```

## Workflow (for contributors)

1. Update `config/config.yaml`
2. Update `params.yaml`
3. Update the entity (`entity/__init__.py`)
4. Update the `ConfigurationManager` (`config/configuration.py`)
5. Update the components
6. Update the pipeline
7. Update `main.py`
8. Update `dvc.yaml`
9. Update `app.py`

## License

Licensed under the MIT License — see [LICENSE](LICENSE).

## Acknowledgments

- Built with TensorFlow/Keras, DVC, MLflow, and Flask.

## Contact

Questions or suggestions? Open an issue or reach out at
[keerthirajkr16@gmail.com](mailto:keerthirajkr16@gmail.com).
