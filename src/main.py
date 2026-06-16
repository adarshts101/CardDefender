import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("dataset/creditcard.csv")
print("Data loaded ✅")

# Data shape and basic info
print(data.shape)
print(data["Class"].value_counts())  # 0: normal, 1: fraud

# Remove Time column (optional)
data.drop(['Time'], axis=1, inplace=True)

# Separate features and labels
X = data.drop("Class", axis=1)
y = data["Class"]

# Define classifiers
classifiers = {
    "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.001, random_state=42),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=0.001),
    "One-Class SVM": OneClassSVM(nu=0.001, kernel="rbf", gamma=0.1)
}

for name, model in classifiers.items():
    print(f"\n{name} running...")

    if name == "Local Outlier Factor":
        y_pred = model.fit_predict(X)
        scores = model.negative_outlier_factor_
    else:
        model.fit(X)
        y_pred = model.predict(X)

    # Map -1, 1 to 1 (fraud), 0 (normal)
    y_pred = [1 if x == -1 else 0 for x in y_pred]

    print(f"{name} Results:")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
