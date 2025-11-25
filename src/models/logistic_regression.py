"""
logistic_regression.py

Wrapper for Logistic Regression training and prediction
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LogisticWrapper:
    def __init__(self, max_iter=1000, scaler=True):
        self.clf = LogisticRegression(max_iter=max_iter)
        self.scaler = StandardScaler() if scaler else None

    def train(self, X_train, y_train):
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
        self.clf.fit(X_train, y_train)
        return self

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.clf.predict_proba(X)[:,1]

    def predict_label(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.clf.predict(X)
