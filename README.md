# Spam Detection using XGBoost

This project demonstrates a spam detection system using XGBoost. The code processes and analyzes text data from a dataset to classify messages as spam or not spam. The approach includes data preprocessing, feature extraction using TF-IDF, and applying the XGBoost classifier.

## Project Overview

- **Data Loading**: The dataset is loaded and preprocessed, including text cleaning and transformation.
- **Text Processing**: Removal of unwanted characters, emojis, and punctuation.
- **Feature Extraction**: TF-IDF Vectorization of text data.
- **Model Training**: Training an XGBoost model with SMOTE for handling class imbalance.
- **Model Evaluation**: Evaluating the model's performance on the test set.

## Dataset

The dataset used is `spam.csv`, which contains labeled messages categorized as spam or not spam.

## Installation

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy nltk scikit-learn imbalanced-learn xgboost
```
# Usage
- **Load and Preprocess Data**: The dataset is loaded and processed to remove unwanted characters and standardize the text.

- **Feature Extraction**: TF-IDF vectorization is applied to convert text data into numerical features.

- **Train-Test Split**: The data is split into training and testing sets.

- **Handle Imbalance**: SMOTE is used to balance the dataset by oversampling the minority class.

- **Model Training**: An XGBoost model is trained with specific hyperparameters.

- **Evaluation**: The model is evaluated using accuracy and other classification metrics.
```bash
import pandas as pd
import re
import warnings
import nltk
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess the data
data = pd.read_csv(r'D:\Externs Club\Major project\spam.csv', encoding='latin-1')
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.replace(['ham', 'spam'], [0, 1])
data = data.rename(columns={"v2": "text", "v1": "label"})

# Define processing functions and apply
# ...

# Feature extraction
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(data['text'])
features = vectors

# Split and balance data
X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2, random_state=111)
smote = SMOTE(sampling_strategy={1: 4000}, random_state=42)
X_train, y_train = smote.fit_resample(X_train.astype('float'), y_train)

# Train model
params = {'objective': 'binary:logistic', 'eval_metric': 'error', 'eta': 0.02, 'max_depth': 10}
d_train = xgb.DMatrix(X_train, label=y_train)
watchlist = [(d_train, 'train')]
bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=400, verbose_eval=10)

# Evaluate model
d_test = xgb.DMatrix(X_test)
p_test = bst.predict(d_test)
p_test_ints = np.vectorize(round)(p_test)
accuracy = accuracy_score(y_test, p_test_ints)
print("Test Accuracy: ", accuracy)
print(classification_report(y_test, p_test_ints))
```
