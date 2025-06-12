from part1 import BaselineModel  # Import your class from wherever it's defined
import pandas as pd
import ast
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

# Load processed training data
X, Y = BaselineModel().load_data('Processed_First_500_Rows.csv', 'Y_train_processed500.csv')

# Split into train/dev (optional)
from sklearn.model_selection import train_test_split
X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.5, random_state=42)

# Train model
base_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model = MultiOutputClassifier(base_clf)
model.fit(X_train, Y_train)

# Predict
Y_pred_train = model.predict(X_train)
Y_pred_dev = model.predict(X_dev)

# Evaluate
BaselineModel().evaluate(Y_train, Y_pred_train)
BaselineModel().evaluate(Y_dev, Y_pred_dev)