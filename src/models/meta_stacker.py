"""
meta_stacker.py

Train meta model (Logistic Regression) and isotonic calibration
"""

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np

class MetaStacker:
    def __init__(self, max_iter=2000):
        self.meta_clf = LogisticRegression(max_iter=max_iter)
        self.iso = IsotonicRegression(out_of_bounds='clip')

    def train_meta(self, oof_preds, y_true):
        """
        Train meta logistic regression on out-of-fold base predictions
        oof_preds: np.array (n_samples, n_base_models)
        """
        self.meta_clf.fit(oof_preds, y_true)
        meta_pred = self.meta_clf.predict_proba(oof_preds)[:,1]
        self.iso.fit(meta_pred, y_true)
        return self

    def predict(self, base_preds):
        """
        Predict calibrated probabilities for test set
        base_preds: np.array (n_samples, n_base_models)
        """
        meta_pred = self.meta_clf.predict_proba(base_preds)[:,1]
        cal_pred = self.iso.predict(meta_pred)
        return cal_pred
