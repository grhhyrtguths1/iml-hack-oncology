import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def remove_abchana_prefix(df):
    df = df.copy()
    df.columns = [col.replace('אבחנה-', '') for col in df.columns]
    return df

def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_abchana_prefix(df)
    # feature_names = list(df.columns)
    # print(feature_names)
    # exit()
    # Convert date columns to datetime
    for col in ['Diagnosis date', 'Surgery date1', 'Surgery date2', 'Surgery date3', 'surgery before or after-Activity date']:
        # Remove known bad values and extra time information
        df[col] = df[col].astype(str).str.slice(0, 10)  # keep only the date part
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')  # parse date

    # Calculate date differences in days
    df['days_to_surgery1'] = (df['Surgery date1'] - df['Diagnosis date']).dt.days
    df['days_to_surgery2'] = (df['Surgery date2'] - df['Diagnosis date']).dt.days
    df['days_to_surgery3'] = (df['Surgery date3'] - df['Diagnosis date']).dt.days
    df['days_to_activity'] = (df['surgery before or after-Activity date'] - df['Diagnosis date']).dt.days

    # Age binning
    df['age_group'] = pd.cut(df['Age'], bins=[0, 40, 60, 80, 120], labels=['young', 'mid', 'old', 'very_old'])

    # Numerical columns - fill missing values with median
    num_cols = ['Tumor width', 'Tumor depth', 'Positive nodes', 'Nodes exam', 'KI67 protein',
                'days_to_surgery1', 'days_to_surgery2', 'days_to_surgery3', 'days_to_activity']

    for col in num_cols:
        df[col] = df[col].astype(str).str.replace('%', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns
    cat_cols = ['Her2', 'Histological diagnosis', 'Histopatological degree', 'Ivi -Lymphovascular invasion',
                'Lymphatic penetration', 'M -metastases mark (TNM)', 'Margin Type', 'N -lymph nodes mark (TNM)',
                'T -Tumor mark (TNM)', 'er', 'pr',
                'Stage', 'Side', 'Surgery name1', 'Surgery name2', 'Surgery name3', 'age_group']
    for col in cat_cols:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            if 'unknown' not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories(['unknown'])
        df[col] = df[col].fillna('unknown')


    # Drop identifier and raw date columns
    drop_cols = ['Form Name', 'Hospital', 'User Name', 'id-hushed_internalpatientid',
                 'Diagnosis date', 'Surgery date1', 'Surgery date2', 'Surgery date3', 'surgery before or after-Activity date']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    return df

def preprocess_train(X: pd.DataFrame, y: pd.DataFrame):
    X_processed = basic_feature_engineering(X)
    y_processed = y.copy()
    return X_processed, y_processed

def preprocess_test(X: pd.DataFrame, train_columns: list):
    X_processed = basic_feature_engineering(X)
    for col in train_columns:
        if col not in X_processed.columns:
            X_processed[col] = 0
    X_processed = X_processed[train_columns]
    return X_processed

if __name__ == '__main__':
    dtype_override = {
        'אבחנה-Ivi -Lymphovascular invasion': 'str',
        'אבחנה-Surgery date2': 'str',
        'אבחנה-Surgery date3': 'str',
        'אבחנה-Surgery name2': 'str',
        'אבחנה-Surgery name3': 'str'
    }

    train_feats = pd.read_csv('../train_test_splits/train.feats.csv', dtype=dtype_override)
    test_feats = pd.read_csv('../train_test_splits/test.feats.csv', dtype=dtype_override)
    train_labels = pd.read_csv('../train_test_splits/train.labels.0.csv')

    X_train, y_train = preprocess_train(train_feats, train_labels)
    X_test = preprocess_test(test_feats, X_train.columns)

    X_train.to_csv("X_train_processed.csv", index=False)
    X_test.to_csv("X_test_processed.csv", index=False)
    y_train.to_csv("y_train_processed.csv", index=False)
