# Kidney Disease Classification
This repository contains code for classifying kidney disease using machine learning techniques. The dataset used is the "Chronic Kidney Disease" dataset from the UCI Machine Learning Repository.
# Dataset
The dataset is available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/chronic+kidney+disease). It contains various features related to kidney health and a target variable indicating the presence of chronic kidney disease.
# Requirements
To run the code, you need to have the following Python packages installed:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imblearn
You can install these packages using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imblearn
```


# Project Structure
```kidney-disease-classification/
├── data/             # Directory to store the dataset
├── src/              # Source code directory
│   ├── KidneyDiseaseClassifier/  # Main package for kidney disease classification
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration file
│   │   ├── utils/             # Utility functions 
│   │   │   ├── __init__.py
│   │   │   ├── common.py       # Common utility functions
│   │   ├── models/            # Machine learning models
│   │   │   ├── __init__.py
│   │   │   ├── model.py        # Model training and evaluation
│   ├── notebooks/          # Jupyter notebooks for exploration and analysis
│   │   ├── kidney_disease_classification.ipynb  # Notebook for kidney disease classification
├── requirements.txt      # List of required packages
├── README.md             # Project documentation
└── LICENSE               # License file
```
# workflow
1. Update the config.yaml
2. Update the secrets.yaml [Optional]
3. Update the params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml
10. app.py

# Usage
1. Clone the repository:
   ```bash
   git clone https://www.github.com/KR-16/kidney-disease-classification.git
   cd kidney-disease-classification
   ```
2. Download the dataset from the UCI Machine Learning Repository and place it in the `data` directory.
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook kidney_disease_classification.ipynb
   ```
# License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details
# Acknowledgments
- The dataset is provided by the UCI Machine Learning Repository.
- Thanks to the contributors of the scikit-learn, pandas, and seaborn libraries for their valuable tools and resources.
# Contact
For any questions or suggestions, feel free to open an issue or contact me at [keerthirajkv2@gmail.com](mailto:)