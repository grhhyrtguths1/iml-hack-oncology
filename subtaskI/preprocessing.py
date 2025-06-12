import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder




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

def clean_her2_text(val):
    import re
    if pd.isna(val):
        return ""
    val = str(val).lower()
    val = val.strip()
    val = re.sub(r'[\"\',()\[\]%]', '', val)
    val = val.replace('חיובי', 'positive')
    val = val.replace('שלילי', 'negative')
    val = val.replace('לא', 'not')
    val = val.replace('נבדק', '')
    val = val.replace('נבדקה', '')
    val = val.replace('nd', '')
    val = re.sub(r'\s+', ' ', val)
    return val

def extract_ihc(val):
    if re.search(r'3\+|\+3|pos\s*\+3', val):
        return "3+"
    elif re.search(r'2\+|\+2', val):
        return "2+"
    elif re.search(r'1\+|\+1', val):
        return "1+"
    elif re.search(r'\b0\b|negative|neg\b|0-1|--|-', val):
        return "0"
    else:
        return "uncertain"


def extract_fish(val):
    if "fish" not in val:
        return "not_tested"

    if re.search(r'amplified|fish\+|positive|pos\b|amp\b', val):
        return "positive"
    elif re.search(
            r'notamplified|nonamplified|negative|fish-|fish\s*-|noamplificated|not amplified|fish neg|fish negative',
            val):
        return "negative"
    elif re.search(r'equivocal|borderline|pending|indeterminate|intermediate|indet', val):
        return "equivocal"
    else:
        return "uncertain"


def classify_final(row):
    ihc = row["ihc_score"]
    fish = row["fish_result"]

    if ihc == "3+":
        return "positive"
    if ihc == "2+":
        if fish == "positive":
            return "positive"
        elif fish == "negative":
            return "negative"
        elif fish == "equivocal":
            return "equivocal"
        else:
            return "uncertain"
    if ihc in ["1+", "0"]:
        return "negative"
    return "uncertain"

def normalize_her2(df):
    df["her2_clean"] = df["Her2"].apply(clean_her2_text)
    df["ihc_score"] = df["her2_clean"].apply(extract_ihc)
    df["fish_result"] = df["her2_clean"].apply(extract_fish)
    df["her2_final"] = df.apply(classify_final, axis=1)
    df = df.drop(columns=["Her2", "her2_clean", "ihc_score", "fish_result"])
    return df


def parse_ki67(val):
    if pd.isnull(val):
        return np.nan

    val = str(val).strip().lower()
    val = re.sub(r'[^\d<>\-%\.]', '', val)
    match = re.search(r'(\d{1,3})(?:\.?\d*)', val)
    if match:
        num = float(match.group(1))
        if num > 100 and num < 1000:
            num = num / 10
        elif num >= 1000:
            num = num / 100
        return num

    if 'low' in val:
        return 5
    elif 'inter' in val:
        return 15
    elif 'high' in val:
        return 30

    return np.nan


def categorize_ki67(value):
    if pd.isnull(value):
        return 'unknown'
    elif value < 10:
        return 'low'
    elif value < 20:
        return 'intermediate'
    else:
        return 'high'



def clean_and_normalize_features(df):
    df['er'] = df['er'].apply(normalize_er_pr)
    df['pr'] = df['pr'].apply(normalize_er_pr)

    if 'Her2' in df.columns:
        df = normalize_her2(df)


    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').round().astype('Int64')
    df['Basic stage'] = df['Basic stage'].astype(str).str.lower().str.extract(r'\b(c|p|r)\b', expand=False)

    # valid_grades = ['G1', 'G2', 'G3', 'G4', 'GX']
    # df['Histopatological degree'] = df['Histopatological degree'].str.upper().where(
    #     df['Histopatological degree'].str.upper().isin(valid_grades), 'Null')

    df['Ivi -Lymphovascular invasion'] = df['Ivi -Lymphovascular invasion'].str.lower().replace({
        '+': 'positive', '(+)': 'positive', 'pos': 'positive', 'yes': 'positive',
        '-': 'negative', '(-)': 'negative', 'neg': 'negative', 'no': 'negative', 'none': 'negative', 'not': 'negative',
    })

    # df['Lymphatic penetration'] = df['Lymphatic penetration'].str.upper().where(
    #     df['Lymphatic penetration'].str.match(r'^L\d+$'), 'Null')

    df['M -metastases mark (TNM)'] = df['M -metastases mark (TNM)'].str.upper().where(
        df['M -metastases mark (TNM)'].str.match(r'^M\d+$'), 'Null')
    df['N -lymph nodes mark (TNM)'] = df['N -lymph nodes mark (TNM)'].str.upper().where(
        df['N -lymph nodes mark (TNM)'].str.match(r'^N\d+$'), 'Null')
    df['T -Tumor mark (TNM)'] = df['T -Tumor mark (TNM)'].str.upper().where(
        df['T -Tumor mark (TNM)'].str.match(r'^T\d+[a-zA-Z]*$'), 'Null')

    df['Stage'] = df['Stage'].str.replace(r'^Stage', '', regex=True)

    if 'KI67 protein' in df.columns:
        df['KI67_clean'] = df['KI67 protein'].apply(parse_ki67)
        df['KI67_category'] = df['KI67_clean'].apply(categorize_ki67)

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
                 'Diagnosis date', 'Surgery date1', 'Surgery date2', 'Surgery date3', 'surgery before or after-Activity date',
                 'age_group','KI67_category','her2_final','Side','Lymphatic penetration','Histological diagnosis']
    return df.drop(columns=[col for col in drop_cols if col in df.columns])




surgery_translation_map = {
    'כיר-לאפ-הוצ טבעת/שנוי מי': 'Laparotomy - ring removal or gender reassignment',
    'כירו-שד-למפקטומי+בלוטות': 'Breast surgery - lumpectomy + lymph nodes',
    'כירו-שד-מסטקטומי+בלוטות': 'Breast surgery - mastectomy + lymph nodes',
    'כירורגיה-שד למפקטומי': 'Breast surgery - lumpectomy',
    'שד-כריתה בגישה זעירה+בלוטות': 'Minimally invasive breast removal + lymph nodes',
    'כירו-שד-למפקטומי+בלוטות+קרינה תוך ניתוחית (intrabeam)': 'Lumpectomy + lymph nodes + intraoperative radiation (intrabeam)',
    'שד-כריתה בגישה זעירה דרך העטרה': 'Minimally invasive breast removal via areola',
    'כירור-הוצאת בלוטות לימפה': 'Lymph node removal surgery',
    'כיר-שד-הוצ.בלוטות בית שח': 'Breast surgery - axillary lymph node removal',
    'כירורגיה-שד מסטקטומי': 'Breast surgery - mastectomy',
    np.nan: 'unknown'
}

side_translation_map = {
    'שמאל': 'left',
    'ימין': 'right',
    'דו צדדי': 'bilateral',
    'unknown': 'unknown'
}

margin_translation_map = {
    'נקיים': 'clear',
    'ללא': 'none',
    'נגועים': 'involved',
    'unknown': 'unknown'
}


def translate_columns(df):
    # עמודות ניתוח
    surgery_columns = ['Surgery name1', 'Surgery name2', 'Surgery name3', 'surgery before or after-Actual activity']
    for col in surgery_columns:
        if col in df.columns:
            df[col] = df[col].map(surgery_translation_map).fillna('unknown')

    # עמודת צד
    if 'Side' in df.columns:
        df['Side'] = df['Side'].map(side_translation_map).fillna('unknown')

    # עמודת שולי ניתוח
    if 'Margin Type' in df.columns:
        df['Margin Type'] = df['Margin Type'].map(margin_translation_map).fillna('unknown')

    return df

def bucket_histological_diagnosis(df):
    def histology_bucket(hist):
        if pd.isna(hist):
            return 'Other/NOS'
        hist = hist.upper().strip()

        malignant_invasive = [
            'INFILTRATING DUCT CARCINOMA',
            'LOBULAR INFILTRATING CARCINOMA',
            'INFILTRATING DUCTULAR CARCINOMA WITH DCIS',
            'DUCTAL AND LOBULAR CARCINOMA',
            'NEUROENDOCRINE CARCINOMA',
            'INFLAMMATORY CARCINOMA',
            'MEDULLARY CARCINOMA',
            'MUCINOUS ADENOCARCINOMA',
            'APOCRINE ADENOCARCINOMA',
            'MUCIN PRODUCING ADENOCARCINOMA',
            'PAGET`S AND INTRADUCTAL CARCINOMA OF BREAST',
            'INTRACYSTIC CARCINOMA',
            'ADENOCARCINOMA',
            'PAPILLARY ADENOCARCINOMA',
            'TUBULAR CARCINOMA',
            'PHYLLODES TUMOR MALIGNANT',
            'COMEDOCARCINOMA',
            'INTRADUCTAL PAP CARCINOMA WITH INVASION'
        ]

        in_situ = [
            'DUCTAL CARCINOMA IN SITU',
            'LOBULAR CARCINOMA IN SITU',
            'COMEDOCARCINOMA IN SITU',
            'INTRADUCTAL CARCINOMA',
            'INTRADUCT AND LOBULAR CARCINOMA IN SITU'
        ]

        benign = [
            'BENIGN TUMOR, NOS',
            'FIBROADENOMA, NOS',
            'ADENOMA OF NIPPLE',
            'PHYLLODES TUMOR BENIGN',
            'INTRADUCTAL PAPILLOMA',
            'INTRACYSTIC PAP ADENOMA',
            'INTRADUCTAL PAPILLOMATOSIS, NOS'
        ]

        other_or_nos = [
            'CARCINOMA, NOS',
            'TUMOR  MALIGNANT, NOS',
            'PAGET`S DISEASE OF BREAST',
            'VERRUCOUS CARCINOMA, VERRUCOUS SQUAMOUS CELL CARC',
            'PHYLLODES TUMOR NOS',
            'INTRADUCTAL PAPILLARY CARCINOMA',
            'ADENOID CYSTIC CA,ADENOCYSTIC CA',
            'PAPILLARY CARCINOMA'
        ]

        if hist in malignant_invasive:
            return 'Malignant Invasive'
        elif hist in in_situ:
            return 'Carcinoma In Situ'
        elif hist in benign:
            return 'Benign/Borderline'
        elif hist in other_or_nos:
            return 'Other/NOS'
        else:
            return 'Other/NOS'

    if 'Histological diagnosis' in df.columns:
        df['Histological diagnosis'] = df['Histological diagnosis'].apply(histology_bucket)
    else:
        df['Histological diagnosis'] = 'Other/NOS'


    return df

def label_encode_columns(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df
def basic_feature_engineering(df):
    df = remove_abchana_prefix(df)
    df = translate_columns(df)
    df = clean_and_normalize_features(df)
    df = bucket_histological_diagnosis(df)
    df = process_tumor_size(df)
    df = process_dates_and_durations(df)
    df = drop_mostly_missing_or_unknown_columns(df)
    df = process_age_binning(df)
    df = process_numerical_columns(df)
    df = process_categorical_columns(df)
    df = drop_unneeded_columns(df)
    label_cols = [
        'Histopatological degree', 'M -metastases mark (TNM)', 'Margin Type',
        'Basic stage', 'N -lymph nodes mark (TNM)', 'T -Tumor mark (TNM)',
        'Stage', 'er', 'pr', 'surgery before or after-Actual activity'
    ]
    df = label_encode_columns(df, label_cols)
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
