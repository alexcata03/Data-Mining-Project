Stress Level Prediction

This project predicts stress levels using regression models and tests the impact of data augmentation on model performance. It compares several algorithms — Random Forest, Extra Trees, Gradient Boosting, SVR, and KNN — using R² scores and visual plots.

Dataset: https://www.kaggle.com/datasets/sacramentotechnology/sleep-deprivation-and-cognitive-performance

The pipeline:

Loads and summarizes an ARFF dataset

Generates synthetic data using Gaussian noise and bootstrap sampling

Scales numeric features and encodes categoricals

Trains and evaluates multiple regressors

Visualizes results with histograms, correlation plots, R² bar charts, and scatter plots

Tech stack: Python, pandas, NumPy, scikit-learn, matplotlib, liac-arff
