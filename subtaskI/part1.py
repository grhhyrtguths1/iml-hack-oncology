import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

from src.evaluate_part_0 import Encode_Multi_Hot

class BaselineModel:
    def __init__(self):
        self.label_encoder = Encode_Multi_Hot()
        self.model = None

    def load_data(self, x_path, y_path):
        X = pd.read_csv(x_path)
        y_raw = pd.read_csv(y_path).iloc[:, 0].apply(eval).tolist()
        self.label_encoder.fit(y_raw)
        Y = np.array([self.label_encoder.enc(lbls) for lbls in y_raw])
        return X, Y

    def train(self, X, Y):
        # Use Logistic Regression as baseline (can replace with XGBClassifier)
        base_clf = LogisticRegression(max_iter=1000)
        self.model = MultiOutputClassifier(base_clf)
        self.model.fit(X, Y)

    def predict(self, X):
        Y_pred = self.model.predict(X)
        return Y_pred

    def save_predictions(self, Y_pred, out_path="baseline_submission.csv"):
        decoded = [self.label_encoder.decode(row) for row in Y_pred]
        pd.DataFrame({"Metastasis": [str(row) for row in decoded]}).to_csv(out_path, index=False)

    def evaluate(self, Y_true, Y_pred):
        return {
            "micro_f1": f1_score(Y_true, Y_pred, average='micro'),
            "macro_f1": f1_score(Y_true, Y_pred, average='macro')
        }