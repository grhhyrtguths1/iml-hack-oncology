import pandas as pd
from part1 import DataLoader, BaselineModel, XGBoostModel
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from docopt import docopt
import torch
# play with hyper parametrs


def trainBaselineModel(X_train, y_train):
    model = BaselineModel()
    model.train(X_train, y_train)
    return model


def trainXGBModel(X_train, y_train):
    model = XGBoostModel()
    model.train(X_train, y_train)
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
    X_path = "X_train_processed.csv"
    y_path: str = "y_train_processed.csv"
    X_final_test_path = "X_test_processed.csv"
    output_path = "predictions.csv"
    dataLoader = DataLoader(X_path, y_path)
    X, y = dataLoader.load_data(X_path, y_path)
    X_test = pd.read_csv(X_final_test_path)
    X = pd.DataFrame(X)  # ensure it's a DataFrame
    X = X.fillna(X.median(numeric_only=True))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model = trainBaselineModel(X, y)
    # predictWithModel(model, X, output_path=output_path)

    # Uncomment below to train XGBoost model
    xgb_model = trainXGBModel(X, y)
    xgb_model.set_label_encoder(dataLoader.label_encoder)

    predictWithModel(xgb_model, X_test, output_path="predictions.csv")
    # s=xgb_model.evaluate(y_test, xgb_model.predict(X_test))
    # print(s)

    # Uncomment below to train Neural Network model
    # nn_model = trainNN(X_train, y_train)
    # predictWithModel(nn_model, X_test, output_path="nn_predictions.csv")
