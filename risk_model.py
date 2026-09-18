# ============================================================
# DEFAULT PREDICTION & CREDIT RISK ANALYSIS
# Dataset: Home Credit Default Risk
# Model: XGBoost
# Evaluation: ROC-AUC
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)

from xgboost import XGBClassifier


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "data/archive/application_train.csv"
OUTPUT_PATH = "outputs"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("DEFAULT PREDICTION & CREDIT RISK ANALYSIS")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ============================================================
# 3. BASIC DATA EXPLORATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nTarget distribution:")
print(df["TARGET"].value_counts())

print("\nTarget percentage:")
print(
    (df["TARGET"].value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 4. MISSING VALUE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()

missing_percentage = df.isnull().mean() * 100

missing_table = pd.DataFrame(
    {
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_percentage
    }
)

missing_table = missing_table.sort_values(
    by="Missing_Percentage",
    ascending=False
)

print("\nTop 20 columns with missing values:")
print(missing_table.head(20))


# ============================================================
# 5. REMOVE COLUMNS WITH MORE THAN 60% MISSING VALUES
# ============================================================

print("\nRemoving columns with more than 60% missing values...")

missing_ratio = df.isnull().mean()

columns_to_drop = missing_ratio[
    missing_ratio > 0.60
].index.tolist()

print("Number of columns removed:", len(columns_to_drop))

df = df.drop(columns=columns_to_drop)


# ============================================================
# 6. REMOVE ID COLUMN
# ============================================================

if "SK_ID_CURR" in df.columns:
    df = df.drop(columns=["SK_ID_CURR"])
    print("SK_ID_CURR removed.")


# ============================================================
# 7. DEFAULT RATE
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS ANALYSIS")
print("=" * 70)

default_rate = df["TARGET"].mean() * 100

print(f"\nOverall Default Rate: {default_rate:.2f}%")


# ============================================================
# 8. DEFAULT DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="TARGET"
)

plt.title("Loan Default Distribution")
plt.xlabel("Default Status")
plt.ylabel("Number of Applicants")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "default_distribution.png"
    )
)

plt.close()


# ============================================================
# 9. DEFAULT RATE BY EDUCATION
# ============================================================

if "NAME_EDUCATION_TYPE" in df.columns:

    education_default = (
        df.groupby("NAME_EDUCATION_TYPE")["TARGET"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print("\nDefault Rate by Education:")
    print(education_default)

    plt.figure(figsize=(10, 6))

    education_default.plot(kind="bar")

    plt.title("Default Rate by Education")
    plt.xlabel("Education")
    plt.ylabel("Default Rate (%)")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "default_by_education.png"
        )
    )

    plt.close()


# ============================================================
# 10. DEFAULT RATE BY INCOME TYPE
# ============================================================

if "NAME_INCOME_TYPE" in df.columns:

    income_default = (
        df.groupby("NAME_INCOME_TYPE")["TARGET"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print("\nDefault Rate by Income Type:")
    print(income_default)

    plt.figure(figsize=(10, 6))

    income_default.plot(kind="bar")

    plt.title("Default Rate by Income Type")
    plt.xlabel("Income Type")
    plt.ylabel("Default Rate (%)")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_PATH,
            "default_by_income_type.png"
        )
    )

    plt.close()


# ============================================================
# 11. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# AGE

if "DAYS_BIRTH" in df.columns:

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    df = df.drop(
        columns=["DAYS_BIRTH"]
    )

    print("Created: AGE_YEARS")


# EMPLOYMENT YEARS

if "DAYS_EMPLOYED" in df.columns:

    df["EMPLOYMENT_YEARS"] = (
        df["DAYS_EMPLOYED"] / 365
    )

    # Treat unrealistic values as missing
    df.loc[
        df["EMPLOYMENT_YEARS"] > 50,
        "EMPLOYMENT_YEARS"
    ] = np.nan

    df = df.drop(
        columns=["DAYS_EMPLOYED"]
    )

    print("Created: EMPLOYMENT_YEARS")


# CREDIT / INCOME RATIO

if (
    "AMT_CREDIT" in df.columns
    and
    "AMT_INCOME_TOTAL" in df.columns
):

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"]
        /
        df["AMT_INCOME_TOTAL"].replace(
            0,
            np.nan
        )
    )

    print("Created: CREDIT_INCOME_RATIO")


# ANNUITY / INCOME RATIO

if (
    "AMT_ANNUITY" in df.columns
    and
    "AMT_INCOME_TOTAL" in df.columns
):

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"]
        /
        df["AMT_INCOME_TOTAL"].replace(
            0,
            np.nan
        )
    )

    print("Created: ANNUITY_INCOME_RATIO")


# ============================================================
# 12. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["TARGET"]
)

y = df["TARGET"]


# ============================================================
# 13. IDENTIFY DATA TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumber of numerical features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))


# ============================================================
# 14. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 15. HANDLE CLASS IMBALANCE
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = (
    negative_count / positive_count
)

print(
    "\nScale positive weight:",
    round(scale_pos_weight, 2)
)


# ============================================================
# 16. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 17. XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 18. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# 19. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST MODEL")
print("=" * 70)

print("\nTraining started...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 20. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]

y_prediction = (
    y_probability >= 0.50
).astype(int)


# ============================================================
# 21. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nROC-AUC Score: {roc_auc:.4f}"
)


# ============================================================
# 22. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_prediction
    )
)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_prediction
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "confusion_matrix.png"
    )
)

plt.close()


# ============================================================
# 24. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title(
    f"ROC Curve - XGBoost\nROC-AUC = {roc_auc:.4f}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "roc_curve.png"
    )
)

plt.close()


# ============================================================
# 25. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

fitted_preprocessor = pipeline.named_steps[
    "preprocessor"
]

xgb_model = pipeline.named_steps[
    "model"
]

feature_names = fitted_preprocessor.get_feature_names_out()

feature_importance = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": xgb_model.feature_importances_
    }
)

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features:")

print(
    feature_importance.head(20).to_string(
        index=False
    )
)


plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 15 Credit Risk Features"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_PATH,
        "feature_importance.png"
    )
)

plt.close()


# ============================================================
# 26. SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame(
    {
        "Actual_Default": y_test.values,
        "Predicted_Default": y_prediction,
        "Default_Probability": y_probability
    }
)

predictions.to_csv(
    os.path.join(
        OUTPUT_PATH,
        "default_predictions.csv"
    ),
    index=False
)

print(
    "\nPredictions saved successfully."
)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)

print(
    f"\nDataset rows: {len(df):,}"
)

print(
    f"Columns after cleaning: {df.shape[1]}"
)

print(
    f"Default Rate: {default_rate:.2f}%"
)

print(
    f"ROC-AUC Score: {roc_auc:.4f}"
)

print("\nTop 10 Risk Features:")

print(
    feature_importance.head(10).to_string(
        index=False
    )
)

print(
    "\nOutput files saved inside:",
    OUTPUT_PATH
)

print(
    "\nPROJECT COMPLETED SUCCESSFULLY!"
)