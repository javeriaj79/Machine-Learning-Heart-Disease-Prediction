# Machine-Learning-Heart-Disease-Prediction 

## Table of Contents
1. [Introduction](#introduction)
2. [Libraries](#libraries)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset Description](#dataset-description)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Training and Evaluation](#model-traing-and-evaluation)
8. [Results](#results)
9. [Contributing](#Contributing)

## Introduction :
This project aims to predict the occurrence of coronary heart disease (CHD) using various machine learning models. The dataset used for this project contains several health-related features and whether the individual has CHD over a ten-year period. Multiple models, including Logistic Regression, Naive Bayes, K-Nearest Neighbors, and Random Forest, are evaluated for their accuracy in predicting CHD.

## Libraries :
The following libraries are used in this code:
- `pandas` - for data manipulation and analysis
- `numpy` - for numerical computations
- `scikit-learn` - for implementing machine learning models and preprocessing
- `matplotlib` - for data visualization
- `seaborn` - for enhanced visualizations

## Installation :
To run this project, ensure you have Python installed along with the following libraries. You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage :
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Place the `heartdataset.csv` file in the appropriate directory.
4. Run the script using a Python environment or a Jupyter Notebook.
   
## Dataset Description :
The dataset used in this project is the Heart Disease dataset. It contains information on various health metrics, such as age, sex, cholesterol levels, blood pressure, and glucose levels, to predict the likelihood of developing coronary heart disease.

## Data Preprocessing :
- The dataset is read using `pandas` and unnecessary columns (like 'education') are dropped.
- Missing values are removed to ensure a clean dataset.
- Features for prediction are selected, and the target variable is defined.
- Feature scaling is performed using `StandardScaler` for normalization.

## Model Training and Evaluation :
A function `train_evaluate_and_plot` is defined to train each model, calculate training and testing accuracy, and plot the confusion matrix. The following models are evaluated:
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Random Forest

## Results :
The accuracies for each model are displayed after evaluation. The confusion matrix is plotted for visual representation of the model performance.

## Contributing :
Contributions are welcome! To contribute:
- **Fork the repository**: Create your own copy of the project to make changes.
- **Open a pull request**: Submit your changes and provide a clear description of what you've done.




