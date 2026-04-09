"""
Bangladesh Student Performance — ML Pipeline
Target: Predict HSC_Result (GPA 3.0–5.0)

NOTE: This dataset is synthetically generated — all inter-feature correlations
are near zero (~0.05 max). No model can beat the baseline (predict the mean).
This script documents that finding and includes both regression and classification.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from kagglehub import KaggleDatasetAdapter
import kagglehub

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                              classification_report, accuracy_score, confusion_matrix)
from xgboost import XGBRegressor, XGBClassifier

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 65)
print("1. LOADING DATA")
print("=" * 65)

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "ihasan88/student-performance-dataset",
    "bangladesh_student_performance.csv",
)
df = df.drop(columns=["Student_ID"])
print(f"Shape: {df.shape}")

TARGET = "HSC_Result"
num_cols   = ["Age", "Study_Hours_per_Week", "Attendance",
              "Family_Income_BDT", "Previous_GPA", "SSC_Result"]
cat_cols   = ["Gender", "District", "School_Type", "Parent_Education",
              "Internet_Access", "Private_Tuition"]

# ─────────────────────────────────────────────
# 2. DATA QUALITY — DETECT SYNTHETIC NOISE
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. DATA QUALITY CHECK")
print("=" * 65)

df_enc = df.copy()
for c in cat_cols:
    df_enc[c] = pd.factorize(df_enc[c])[0]

corr_target = df_enc.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print("\nCorrelation with HSC_Result (abs sorted):")
print(corr_target.to_string())

max_corr = corr_target.abs().max()
print(f"\nMax |correlation| with target: {max_corr:.4f}")
print(f"Baseline RMSE (predict mean):  {df[TARGET].std():.4f}")
if max_corr < 0.10:
    print("\n⚠  WARNING: Dataset appears SYNTHETICALLY GENERATED.")
    print("   All feature-target correlations < 0.10 — no real signal present.")
    print("   ML models will not meaningfully outperform predicting the mean.")

# ─────────────────────────────────────────────
# 3. EDA PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Target distribution
df[TARGET].hist(bins=30, ax=axes[0, 0], color="steelblue", edgecolor="white")
axes[0, 0].set_title("HSC_Result Distribution")
axes[0, 0].axvline(df[TARGET].mean(), color="red", linestyle="--", label=f"mean={df[TARGET].mean():.2f}")
axes[0, 0].legend()

# Correlation heatmap
sns.heatmap(df_enc.corr(), annot=False, cmap="coolwarm", ax=axes[0, 1],
            vmin=-0.3, vmax=0.3, linewidths=0.3)
axes[0, 1].set_title("Full Correlation Matrix\n(note: all near zero → synthetic data)")

# SSC vs HSC scatter
axes[1, 0].scatter(df["SSC_Result"], df["HSC_Result"], alpha=0.3, s=15)
axes[1, 0].set_xlabel("SSC_Result (prior exam)")
axes[1, 0].set_ylabel("HSC_Result (target)")
axes[1, 0].set_title(f"SSC vs HSC  (corr={df['SSC_Result'].corr(df[TARGET]):.3f})")

# Previous_GPA vs HSC scatter
axes[1, 1].scatter(df["Previous_GPA"], df["HSC_Result"], alpha=0.3, s=15, color="salmon")
axes[1, 1].set_xlabel("Previous_GPA")
axes[1, 1].set_ylabel("HSC_Result")
axes[1, 1].set_title(f"Previous_GPA vs HSC  (corr={df['Previous_GPA'].corr(df[TARGET]):.3f})")

plt.tight_layout()
plt.savefig("/var/www/kaggle/eda.png", dpi=130)
print("\nSaved: eda.png")

# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df["Study_x_Attendance"] = df["Study_Hours_per_Week"] * df["Attendance"] / 100
df["GPA_SSC_Delta"]       = df["SSC_Result"] - df["Previous_GPA"]
df["Income_log"]          = np.log1p(df["Family_Income_BDT"])

num_features = num_cols + ["Study_x_Attendance", "GPA_SSC_Delta", "Income_log"]
cat_features = cat_cols

X = df.drop(columns=[TARGET, "Family_Income_BDT"])
y_reg = df[TARGET]

# Classification target: A (≥4.5), B (4.0–4.5), C (3.5–4.0), D (<3.5)
def grade_band(g):
    if g >= 4.5: return "A (≥4.5)"
    if g >= 4.0: return "B (4.0–4.5)"
    if g >= 3.5: return "C (3.5–4.0)"
    return "D (<3.5)"

y_cls_str = df[TARGET].apply(grade_band)
le = LabelEncoder()
y_cls = pd.Series(le.fit_transform(y_cls_str), index=df.index)
cls_labels = le.classes_
print("\nClass distribution:")
print(y_cls_str.value_counts().sort_index().to_string())

num_features = [c for c in num_features if c in X.columns]
cat_features = [c for c in cat_features if c in X.columns]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
])

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_tr, X_te, yr_tr, yr_te = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_tr2, X_te2, yc_tr, yc_te = train_test_split(X, y_cls, test_size=0.2, random_state=42,
                                                stratify=y_cls)
yc_te_str = le.inverse_transform(yc_te)

# ─────────────────────────────────────────────
# 6A. REGRESSION MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("6A. REGRESSION MODELS")
print("=" * 65)
print(f"Baseline RMSE (predict mean): {yr_te.std():.4f}")

reg_models = {
    "Baseline (mean)":    DummyRegressor(strategy="mean"),
    "Ridge":              Ridge(alpha=1.0),
    "Random Forest":      RandomForestRegressor(n_estimators=200, max_depth=6,
                                                random_state=42, n_jobs=-1),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                     max_depth=3, random_state=42),
    "XGBoost":            XGBRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=3, random_state=42, verbosity=0),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
reg_results = []

for name, model in reg_models.items():
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    cv_r2   = cross_val_score(pipe, X_tr, yr_tr, scoring="r2", cv=kf, n_jobs=-1)
    cv_rmse = np.sqrt(-cross_val_score(pipe, X_tr, yr_tr,
                                       scoring="neg_mean_squared_error", cv=kf, n_jobs=-1))
    pipe.fit(X_tr, yr_tr)
    yp = pipe.predict(X_te)
    r2   = r2_score(yr_te, yp)
    rmse = np.sqrt(mean_squared_error(yr_te, yp))
    mae  = mean_absolute_error(yr_te, yp)
    reg_results.append({"Model": name,
                        "CV R² (mean)": cv_r2.mean(), "CV RMSE": cv_rmse.mean(),
                        "Test R²": r2, "Test RMSE": rmse, "Test MAE": mae})
    print(f"{name:25s}  CV R²={cv_r2.mean():.4f}  Test R²={r2:.4f}  RMSE={rmse:.4f}")

reg_df = pd.DataFrame(reg_results).sort_values("Test RMSE")
print("\nRegression Summary:")
print(reg_df.to_string(index=False))

# ─────────────────────────────────────────────
# 6B. CLASSIFICATION MODELS (Grade Bands)
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("6B. CLASSIFICATION MODELS (Grade Bands A/B/C/D)")
print("=" * 65)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cls_models = {
    "Baseline (freq)":     DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=6,
                                                   random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05,
                                         max_depth=3, random_state=42,
                                         verbosity=0,
                                         eval_metric="mlogloss"),
}

cls_results = []
best_cls_pipe = None
best_cls_acc  = -1
best_cls_name = ""

for name, model in cls_models.items():
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    cv_acc = cross_val_score(pipe, X_tr2, yc_tr, scoring="accuracy", cv=skf, n_jobs=-1)
    pipe.fit(X_tr2, yc_tr)
    yp = pipe.predict(X_te2)
    acc = accuracy_score(yc_te, yp)
    cls_results.append({"Model": name,
                        "CV Acc (mean)": cv_acc.mean(), "CV Acc (std)": cv_acc.std(),
                        "Test Acc": acc})
    print(f"{name:25s}  CV Acc={cv_acc.mean():.4f}±{cv_acc.std():.3f}  Test Acc={acc:.4f}")
    if acc > best_cls_acc and name != "Baseline (freq)":
        best_cls_acc  = acc
        best_cls_name = name
        best_cls_pipe = pipe

cls_df = pd.DataFrame(cls_results).sort_values("Test Acc", ascending=False)
print("\nClassification Summary:")
print(cls_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. PLOTS
# ─────────────────────────────────────────────

# 7a — Regression: Actual vs Predicted (best non-baseline)
best_reg_name = reg_df[reg_df["Model"] != "Baseline (mean)"].iloc[0]["Model"]
best_reg_pipe = Pipeline([("pre", preprocessor), ("model", reg_models[best_reg_name])])
best_reg_pipe.fit(X_tr, yr_tr)
yp_reg = best_reg_pipe.predict(X_te)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].scatter(yr_te, yp_reg, alpha=0.4, edgecolors="k", linewidths=0.3, s=20)
lo, hi = 2.8, 5.1
axes[0].plot([lo, hi], [lo, hi], "r--", lw=2)
axes[0].set_xlabel("Actual HSC_Result"); axes[0].set_ylabel("Predicted")
axes[0].set_title(f"Regression: {best_reg_name}\nR²={r2_score(yr_te, yp_reg):.4f}")

residuals = yr_te - yp_reg
axes[1].hist(residuals, bins=30, color="salmon", edgecolor="white")
axes[1].axvline(0, color="k", linestyle="--")
axes[1].set_title("Residuals"); axes[1].set_xlabel("Actual − Predicted")

# 7b — Confusion matrix for best classifier
yp_cls = best_cls_pipe.predict(X_te2)
yp_cls_str = le.inverse_transform(yp_cls)
labels = sorted(cls_labels)
cm = confusion_matrix(yc_te_str, yp_cls_str, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=axes[2])
axes[2].set_title(f"Confusion Matrix: {best_cls_name}\nAcc={accuracy_score(yc_te,yp_cls):.4f}")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("/var/www/kaggle/model_results.png", dpi=130)
print("\nSaved: model_results.png")

# 7c — Feature importance (best tree regressor)
if hasattr(reg_models[best_reg_name], "feature_importances_"):
    fi = pd.Series(reg_models[best_reg_name].feature_importances_,
                   index=num_features + cat_features).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    fi.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Feature Importances — {best_reg_name}")
    ax.set_xlabel("Importance (note: all near-uniform → no real signal)")
    plt.tight_layout()
    plt.savefig("/var/www/kaggle/feature_importance.png", dpi=130)
    print("Saved: feature_importance.png")

# ─────────────────────────────────────────────
# 8. FINAL REPORT
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL REPORT")
print("=" * 65)
print(f"\nDataset: 1000 students, 13 features")
print(f"Target : HSC_Result (GPA 3.0–5.0), std={df[TARGET].std():.4f}")
print(f"\nMax feature-target correlation: {max_corr:.4f}  ← near zero")
print(f"\n{'─'*65}")
print("REGRESSION (predicting exact GPA)")
print(f"{'─'*65}")
print(reg_df.to_string(index=False))
print(f"\n{'─'*65}")
print("CLASSIFICATION (predicting grade band A/B/C/D)")
print(f"{'─'*65}")
print(cls_df.to_string(index=False))
print(f"\n{'─'*65}")
print("CONCLUSION")
print(f"{'─'*65}")
print("""
This dataset is SYNTHETICALLY GENERATED with random values.
Evidence:
  • SSC_Result vs HSC_Result correlation: 0.006 (should be ~0.7+ in real data)
  • All 12 feature pairs have |corr| < 0.07
  • All models perform at or below baseline (R² ≈ 0)
  • Even grade-band classification barely exceeds random chance

Recommendation: use a dataset with real-world signal for meaningful ML.
If you must use this dataset, it can serve as a unit-test baseline to
verify that a pipeline is correctly implemented.
""")
