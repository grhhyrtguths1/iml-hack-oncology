import pandas as pd
import numpy as np
import re

def remove_abchana_prefix(df):
    df = df.copy()
    df.columns = [col.replace('אבחנה-', '').strip() for col in df.columns]
    return df

def normalize_er_pr(value):
    if pd.isna(value):
        return 'unknown'

    raw_val = str(value).strip().lower()
    had_percent = '%' in raw_val
    val = re.sub(r'[^\w.%+-]', ' ', raw_val)
    val = val.replace(',', '.').replace('%', '')

    positive_terms = {'+', 'positive', 'pos', 'yes', 'strongly pos', 'strongly positive', 'strong pos'}
    negative_terms = {'-', 'negative', 'neg', 'no', 'none', 'not', 'weak neg', 'שלילי'}

    pos_found = any(term in val for term in positive_terms)
    neg_found = any(term in val for term in negative_terms)

    if pos_found and neg_found:
        return 'unknown'
    elif pos_found:
        return 'positive'
    elif neg_found:
        return 'negative'

    matches = re.findall(r'[-+]?\d*\.?\d+', val)
    if matches:
        try:
            num = float(matches[0])
            if not had_percent and num < 1:
                num *= 100
            return 'positive' if num >= 1.0 else 'negative'
        except:
            return 'unknown'

    return 'unknown'

def clean_and_normalize_features(df):
    df['er'] = df['er'].apply(normalize_er_pr)
    df['pr'] = df['pr'].apply(normalize_er_pr)
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').round().astype('Int64')
    df['Basic stage'] = df['Basic stage'].astype(str).str.lower().str.extract(r'\b(c|p|r)\b', expand=False)

    valid_grades = ['G1', 'G2', 'G3', 'G4', 'GX']
    df['Histopatological degree'] = df['Histopatological degree'].str.upper().where(
        df['Histopatological degree'].str.upper().isin(valid_grades), 'Null')

    df['Ivi -Lymphovascular invasion'] = df['Ivi -Lymphovascular invasion'].str.lower().replace({
        '+': 'positive', '(+)': 'positive', 'pos': 'positive', 'yes': 'positive',
        '-': 'negative', '(-)': 'negative', 'neg': 'negative', 'no': 'negative', 'none': 'negative', 'not': 'negative',
    })

    df['Lymphatic penetration'] = df['Lymphatic penetration'].str.upper().where(
        df['Lymphatic penetration'].str.match(r'^L\d+$'), 'Null')

    df['M -metastases mark (TNM)'] = df['M -metastases mark (TNM)'].str.upper().where(
        df['M -metastases mark (TNM)'].str.match(r'^M\d+$'), 'Null')
    df['N -lymph nodes mark (TNM)'] = df['N -lymph nodes mark (TNM)'].str.upper().where(
        df['N -lymph nodes mark (TNM)'].str.match(r'^N\d+$'), 'Null')
    df['T -Tumor mark (TNM)'] = df['T -Tumor mark (TNM)'].str.upper().where(
        df['T -Tumor mark (TNM)'].str.match(r'^T\d+[a-zA-Z]*$'), 'Null')

    df['Stage'] = df['Stage'].str.replace(r'^Stage', '', regex=True)
    return df

def drop_mostly_missing_or_unknown_columns(df, threshold=0.9):
    cols_to_drop = []
    for col in df.columns:
        total = len(df)
        missing_or_unknown = df[col].isna().sum()
        if df[col].dtype == object:
            missing_or_unknown += (df[col].astype(str).str.lower() == 'unknown').sum()
        if missing_or_unknown / total > threshold:
            cols_to_drop.append(col)
    if cols_to_drop:
        # print("Dropping columns with >90% missing or unknown values:", cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    return df

def process_tumor_size(df):
    df['Tumor width'] = pd.to_numeric(df['Tumor width'], errors='coerce')
    df['Tumor depth'] = pd.to_numeric(df['Tumor depth'], errors='coerce')
    df['tumor size'] = df['Tumor width'] * df['Tumor depth']
    return df.drop(columns=['Tumor width', 'Tumor depth'])

def process_dates_and_durations(df):
    date_cols = ['Diagnosis date', 'Surgery date1', 'Surgery date2', 'Surgery date3', 'surgery before or after-Activity date']
    for col in date_cols:
        df[col] = df[col].astype(str).str.slice(0, 10)
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')
    df['days_to_surgery1'] = (df['Surgery date1'] - df['Diagnosis date']).dt.days
    df['days_to_surgery2'] = (df['Surgery date2'] - df['Diagnosis date']).dt.days
    df['days_to_surgery3'] = (df['Surgery date3'] - df['Diagnosis date']).dt.days
    df['days_to_activity'] = (df['surgery before or after-Activity date'] - df['Diagnosis date']).dt.days
    return df

def process_age_binning(df):
    df['age_group'] = pd.cut(df['Age'], bins=[0, 40, 60, 80, 120], labels=['young', 'mid', 'old', 'very_old'])
    return df

def process_numerical_columns(df):
    num_cols = ['Positive nodes', 'Nodes exam', 'KI67 protein',
                'days_to_surgery1', 'days_to_surgery2', 'days_to_surgery3', 'days_to_activity', 'tumor size']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    return df


def process_categorical_columns(df):
    cat_cols = ['Her2', 'Histological diagnosis', 'Histopatological degree',
                'Ivi -Lymphovascular invasion', 'Lymphatic penetration',
                'M -metastases mark (TNM)', 'Margin Type', 'N -lymph nodes mark (TNM)',
                'T -Tumor mark (TNM)', 'er', 'pr',
                'Stage', 'Side', 'Surgery name1', 'Surgery name2', 'Surgery name3', 'age_group']

    for col in cat_cols:
        if col in df.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                if 'unknown' not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(['unknown'])
            df[col] = df[col].fillna('unknown')

    return df

def drop_unneeded_columns(df):
    drop_cols = ['Form Name', 'Hospital', 'User Name', 'id-hushed_internalpatientid',
                 'Diagnosis date', 'Surgery date1', 'Surgery date2', 'Surgery date3', 'surgery before or after-Activity date']
    return df.drop(columns=[col for col in drop_cols if col in df.columns])

def basic_feature_engineering(df):
    df = remove_abchana_prefix(df)
    df = clean_and_normalize_features(df)
    df = process_tumor_size(df)
    df = process_dates_and_durations(df)
    df = drop_mostly_missing_or_unknown_columns(df)
    df = process_age_binning(df)
    df = process_numerical_columns(df)
    df = process_categorical_columns(df)
    df = drop_unneeded_columns(df)
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

    X_train.to_csv("X_train_processed_second.csv", index=False)
    X_test.to_csv("X_test_processed_second.csv", index=False)
    y_train.to_csv("y_train_processed_second.csv", index=False)
