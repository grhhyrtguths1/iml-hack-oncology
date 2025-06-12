import pandas as pd
from part1 import DataLoader, BaselineModel
import evaluate_part_0
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import torch
# play with hyper parametrs
def trainBaselineModel(X_train, y_train):
    model = BaselineModel()
    model.train(X_train, y_train)
    return model


def trainXGBModel(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model


def trainNN(X_train, y_train):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X_train, y_train)
    return model


def predictWithModel(model, X_test, output_path="predictions.csv"):
    Y_pred = model.predict(X_test)
    model.save_predictions(Y_pred, out_path=output_path)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    X_path = "X_test_preprocessed.csv"
    y_path: str = "y_train_processed.csv"
    X_final_test_path = "X_test_processed.csv"
    output_path = "predictions.csv"
    dataLoader = DataLoader(X_path, y_path)
    X, y = dataLoader.load_data(X_path, y_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = trainBaselineModel(X_train, y_train)
    predictWithModel(model, X_test, output_path=output_path)

    # Uncomment below to train XGBoost model
    # xgb_model = trainXGBModel(X_train, y_train)
    # predictWithModel(xgb_model, X_test, output_path="xgb_predictions.csv")

    # Uncomment below to train Neural Network model
    # nn_model = trainNN(X_train, y_train)
    # predictWithModel(nn_model, X_test, output_path="nn_predictions.csv")
