import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import TensorDataset

from subtaskI.evaluate_part_0 import Encode_Multi_Hot


class DataLoader:
    def __init__(self, X_path, y_path):
        self.X_path = X_path
        self.y_path = y_path
        self.label_encoder = Encode_Multi_Hot()
        self.X = None
        self.Y = None

    def load_data(self, x_path, y_path):
        X = pd.read_csv(x_path)
        y_raw = pd.read_csv(y_path).iloc[:, 0].apply(eval).tolist()
        self.label_encoder.fit(y_raw)
        Y = np.array([self.label_encoder.enc(lbls) for lbls in y_raw])
        self.X = X
        self.Y = Y
        return X, Y
    def get_train_test_split(self, test_size):
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=test_size, random_state=42)
        return X_train, X_test, Y_train, Y_test


class BaselineModel:
    def __init__(self):
        self.label_encoder = Encode_Multi_Hot()
        self.model = None

    def set_label_encoder(self, encoder):
        self.label_encoder = encoder

    def train(self, X, Y):
        # Use Logistic Regression as baseline (can replace with XGBClassifier)
        base_clf = LogisticRegression(max_iter=1000)
        self.model = MultiOutputClassifier(base_clf)
        self.model.fit(X, Y)

    def predict(self, X):
        Y_pred = self.model.predict(X)
        return Y_pred

    def decodeData(self, Y_pred):
        decoded = [self.label_encoder.decode(row) for row in Y_pred]
        return np.array(decoded)

    def save_predictions(self, Y_pred, out_path="baseline_submission.csv"):
        decoded = [self.label_encoder.decode(row) for row in Y_pred]
        pd.DataFrame({"Metastasis": [str(row) for row in decoded]}).to_csv(out_path, index=False)

    def evaluate(self, Y_true, Y_pred):
        return {
            "micro_f1": f1_score(Y_true, Y_pred, average='micro'),
            "macro_f1": f1_score(Y_true, Y_pred, average='macro')
        }


class XGBoostModel:
    def __init__(self):
        from xgboost import XGBClassifier
        from sklearn.multioutput import MultiOutputClassifier
        self.label_encoder = Encode_Multi_Hot()
        self.base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model = MultiOutputClassifier(self.base_model)

    def set_label_encoder(self, encoder):
        self.label_encoder = encoder

    def train(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def decodeData(self, Y_pred):
        decoded = [self.label_encoder.decode(row) for row in Y_pred]
        return np.array(decoded)

    def save_predictions(self, Y_pred, out_path="xgboost_submission.csv"):
        decoded = [self.label_encoder.decode(row) for row in Y_pred]
        pd.DataFrame({"Metastasis": [str(row) for row in decoded]}).to_csv(out_path, index=False)

    def evaluate(self, Y_true, Y_pred):
        return {
            "micro_f1": f1_score(Y_true, Y_pred, average='micro'),
            "macro_f1": f1_score(Y_true, Y_pred, average='macro')
        }


class NeuralNetModel:
    def __init__(self, input_dim=None, output_dim=None, hidden_dim=100, epochs=20, batch_size=32, lr=1e-3):
        self.label_encoder = None
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def set_label_encoder(self, encoder):
        self.label_encoder = encoder

    def build_model(self):
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def train(self, X, Y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.build_model()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor)
            preds = torch.sigmoid(logits).numpy()
            return (preds > 0.5).astype(int)

    def decodeData(self, Y_pred):
        return np.array([self.label_encoder.decode(row) for row in Y_pred])

    def save_predictions(self, Y_pred, out_path="nn_submission.csv"):
        decoded = [self.label_encoder.decode(row) for row in Y_pred]
        pd.DataFrame({"Metastasis": [str(row) for row in decoded]}).to_csv(out_path, index=False)

    def evaluate(self, Y_true, Y_pred):
        return {
            "micro_f1": f1_score(Y_true, Y_pred, average='micro'),
            "macro_f1": f1_score(Y_true, Y_pred, average='macro')
        }

