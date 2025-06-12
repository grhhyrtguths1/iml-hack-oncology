import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
train_features = pd.read_csv("../subtaskI/y_train_processed500.csv", low_memory=False)
train_labels = pd.read_csv("../train_test_splits/train.labels.1.csv")
test_features = pd.read_csv("../train_test_splits/test.feats.csv")

# train_features = train_features.iloc[:500]
train_labels = train_labels.iloc[:500]
test_features = test_features.iloc[:500]
test_features = test_features.apply(pd.to_numeric, errors='coerce')
train_features = train_features.apply(pd.to_numeric, errors='coerce')

# מחיקת שורות עם NaN ב-train_features ובתוויות המתאימות
not_null_mask = train_features.notnull().all(axis=1) & train_labels.notnull()
train_features = train_features.loc[not_null_mask]
train_labels = train_labels.loc[not_null_mask]



# # Optional: check that train and test have the same features
# assert list(train_features.columns) == list(test_features.columns)

# Split train set for internal validation (optional, to estimate performance)
X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=42
)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on validation set (optional, for internal evaluation)
val_preds = model.predict(X_val)
mse = mean_squared_error(y_val, val_preds)
print(f"Validation MSE: {mse:.4f}")

# Predict on test set
test_preds = model.predict(test_features)

# Ensure output is in the correct format
output = pd.DataFrame(test_preds, columns=["tumor_size"])
output.to_csv("predictions.csv", index=False)



# if __name__ == '__main__':
#     train_feats = pd.read_csv('../train_test_splits/train.feats.csv', dtype=dtype_override)
#     test_feats = pd.read_csv('../train_test_splits/test.feats.csv', dtype=dtype_override)
#     train_labels = pd.read_csv('../train_test_splits/train.labels.0.csv')