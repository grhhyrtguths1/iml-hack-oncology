import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("train.feats.csv")

# Convert all date columns
date_cols = [col for col in df.columns if "date" in col.lower()]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

# Compute last treatment date
df['last_treatment_date'] = df[date_cols].max(axis=1)

# Compute 'days since last treatment'
for col in date_cols:
    df[f"{col}_since_last"] = (df['last_treatment_date'] - df[col]).dt.days

# Drop original dates
df = df.drop(columns=date_cols + ['last_treatment_date'])

# Select timeline features
timeline_cols = [col for col in df.columns if col.endswith('_since_last')]
timeline_df = df[timeline_cols].fillna(df[timeline_cols].median())

# Normalize
scaler = StandardScaler()
timeline_scaled = scaler.fit_transform(timeline_df)

# DBSCAN clustering
db = DBSCAN(eps=300, min_samples=5)
clusters = db.fit_predict(timeline_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(timeline_scaled)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set2')
plt.title("Timeline-Based Clustering with DBSCAN")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()