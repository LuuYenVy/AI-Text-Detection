"""
xgboost_model.py

Wrapper for XGBoost with standard configuration
"""

import xgboost as xgb
import numpy as np

class XGBWrapper:
    def __init__(self, n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42):
        self.clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric='auc',
            random_state=random_state
        )

    def train(self, X_train, y_train, eval_set=None):
        if eval_set:
            self.clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.clf.fit(X_train, y_train, verbose=False)
        return self

    def predict(self, X):
        return self.clf.predict_proba(X)[:,1]

    def predict_label(self, X):
        return self.clf.predict(X)
