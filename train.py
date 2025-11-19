
"""train.py for Diabetes Prediction"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------------------
# Parameters
# ---------------------------
C = 1.0
n_splits = 5
output_file = f'model_C={C}_diabetes.bin'

# ---------------------------
# Data Preparation
# ---------------------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Standardize column names
df.columns = (
    df.columns
    .str.replace('([a-z])([A-Z])', r'\1_\2', regex=True)
    .str.replace(' ', '_')
    .str.lower()
)

# Split full training and test
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.outcome.values
y_test = df_test.outcome.values

df_full_train = df_full_train.drop('outcome', axis=1)
df_test = df_test.drop('outcome', axis=1)

# ---------------------------
# Features
# ---------------------------
numerical = df_full_train.columns.tolist()  # all numeric
categorical = []  # no categorical columns

# ---------------------------
# Train / Predict functions
# ---------------------------
def train(df_train, y_train, C=1.0):
    dicts = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(C=C, max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)
    return model.predict_proba(X)[:, 1]

# ---------------------------
# Cross-validation
# ---------------------------
print(f"Running {n_splits}-fold CV with C={C}")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
scores = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = y_full_train[train_idx]
    y_val = y_full_train[val_idx]

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f"Fold {fold} ROC AUC: {auc:.3f}")

print(f"CV results: C={C} -> ROC AUC: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

# ---------------------------
# Train final model on full training set
# ---------------------------
print("Training final model on full training set...")
dv, model = train(df_full_train, y_full_train, C=C)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)
print(f"Test set ROC AUC: {auc:.3f}")

# ---------------------------
# Save final model
# ---------------------------
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f"Model saved to {output_file}")
