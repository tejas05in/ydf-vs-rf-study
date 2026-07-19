"""
Matched-Condition Comparison: scikit-learn RandomForestClassifier vs. YDF RandomForestLearner
Dataset: BRFSS 2015 Diabetes Health Indicators (UCI ML Repository)

This script implements a corrected experimental protocol addressing the six
methodological issues identified in the original notebook review:
  1. RF-vs-RF (not RF-vs-GBT)
  2. Principled outlier handling (not median-only IQR retention)
  3. Consistent class-imbalance handling across both frameworks
  4. Symmetric metric definitions (positive class = 1 = diabetic, for both)
  5. Comparable model-size measurement (on-disk serialized bytes, both frameworks)
  6. Paired / variance-corrected significance testing (not independent t-test)

Run in an environment with: pandas, numpy, scikit-learn, scipy, ydf installed.
"""

import time
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score
)
import ydf

RANDOM_SEED = 42
N_REPEATS = 10
TARGET = "Diabetes_binary"

# Note on matched capacity: sklearn's default n_estimators=100 is matched
# explicitly against YDF's num_trees=100 below (YDF's own RF default is 300).
# Leaving either at its library default would confound "framework" with
# "forest size" as a hidden variable, so both are pinned to the same value.

# ---------------------------------------------------------------------------
# BLOCK 1 — Load data and report class balance (no filtering yet)
# ---------------------------------------------------------------------------
df = pd.read_csv("data.csv")

# Ensure the target is a clean integer 0/1 column. YDF infers label_classes()
# in sorted order for binary classification; casting to int guarantees
# label_classes() == [0, 1], so predict() returns P(class == 1) as intended.
df[TARGET] = df[TARGET].astype(int)

print("Full dataset shape:", df.shape)
print(df[TARGET].value_counts(normalize=True).rename("proportion"))

# ---------------------------------------------------------------------------
# BLOCK 2 — Principled outlier handling (correction #2)
# Standard 1.5x IQR rule applied to BMI only; retains the vast majority of
# legitimate data rather than collapsing to the interquartile range alone.
# This step is reported transparently and its effect on N is logged.
# ---------------------------------------------------------------------------
q1 = df["BMI"].quantile(0.25)
q3 = df["BMI"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

n_before = df.shape[0]
df = df[(df["BMI"] >= lower_bound) & (df["BMI"] <= upper_bound)].reset_index(drop=True)
n_after = df.shape[0]
print(f"Outlier removal (1.5x IQR on BMI): {n_before} -> {n_after} "
      f"({100 * (n_before - n_after) / n_before:.2f}% removed)")

# Report class balance post-filtering — verify filtering did not itself
# distort the target distribution in a way that needs disclosure.
print(df[TARGET].value_counts(normalize=True).rename("proportion (post-filter)"))

# ---------------------------------------------------------------------------
# BLOCK 3 — Fixed, shared data splits (correction: matched protocol)
# Generate N_REPEATS stratified splits ONCE. Both frameworks are evaluated
# on the *same* splits, not independently regenerated ones. Stratification
# is used given the ~84:16 class imbalance confirmed above.
# ---------------------------------------------------------------------------
splits = []
for i in range(N_REPEATS):
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df[TARGET], random_state=i
    )
    splits.append((train_idx, test_idx))

# ---------------------------------------------------------------------------
# BLOCK 4 — Shared preprocessing decision (correction #2 in prior review)
# RF splits are scale-invariant; StandardScaler is dropped for both arms
# rather than applied asymmetrically. This is stated explicitly as a
# methodological decision, not a silent omission.
# ---------------------------------------------------------------------------
feature_cols = [c for c in df.columns if c != TARGET]

# ---------------------------------------------------------------------------
# BLOCK 5 — Consistent class-imbalance handling (correction #3)
# Both frameworks use "balanced" class weighting derived identically:
# weight_i = n_samples / (n_classes * count(class_i))
# sklearn: pass via class_weight="balanced" (computed internally, identical formula)
# YDF: compute the same weights explicitly and pass as a sample_weight column
# ---------------------------------------------------------------------------
def add_balanced_sample_weight(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    counts = frame[target].value_counts()
    n = len(frame)
    k = counts.shape[0]
    weight_map = {cls: n / (k * cnt) for cls, cnt in counts.items()}
    frame = frame.copy()
    frame["sample_weight"] = frame[target].map(weight_map)
    return frame

# ---------------------------------------------------------------------------
# BLOCK 6 — Comparable model-size measurement (correction #5)
# Both models are serialized to disk and compared by on-disk byte size,
# not by sys.getsizeof() on an in-memory Python object handle.
# ---------------------------------------------------------------------------
def sklearn_model_size_bytes(clf) -> int:
    return len(pickle.dumps(clf))

def ydf_model_size_bytes(model, tmp_dir: str) -> int:
    import os
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    model.save(tmp_dir)
    total = 0
    for root, _, files in os.walk(tmp_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total

# ---------------------------------------------------------------------------
# BLOCK 7 — Symmetric metric computation (correction #4)
# Positive class is explicitly defined as 1 (diabetic) for BOTH frameworks.
# AUC-ROC and AUC-PR are included as primary metrics given class imbalance
# (Saito & Rehmsmeier, 2015, PLOS ONE, on PR-AUC's superiority under
# imbalance; PMID: 25781975).
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1),
        "Recall": recall_score(y_true, y_pred, pos_label=1),
        "F1": f1_score(y_true, y_pred, pos_label=1),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
        "Brier": brier_score_loss(y_true, y_proba),
    }

# ---------------------------------------------------------------------------
# BLOCK 8 — sklearn RandomForest evaluation loop (matched splits, weights)
# ---------------------------------------------------------------------------
sklearn_rows = []
for i, (train_idx, test_idx) in enumerate(splits):
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET].values

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    start = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    row = compute_metrics(y_test, y_pred, y_proba)
    row["Model_Size_Bytes"] = sklearn_model_size_bytes(clf)
    row["Train_Time_s"] = train_time
    sklearn_rows.append(row)

sklearn_results = pd.DataFrame(sklearn_rows)
print("\n=== scikit-learn RandomForest — per-split results ===")
print(sklearn_results.round(5))

# ---------------------------------------------------------------------------
# BLOCK 9 — YDF RandomForest evaluation loop (SAME matched splits, weights)
# ---------------------------------------------------------------------------
ydf_rows = []
for i, (train_idx, test_idx) in enumerate(splits):
    train_df = add_balanced_sample_weight(df.loc[train_idx], TARGET)
    test_df = df.loc[test_idx].copy()
    # test set does not need weighting for evaluation of unweighted metrics;
    # weights are a training-time correction only.

    learner = ydf.RandomForestLearner(
        label=TARGET,
        weights="sample_weight",
        num_trees=100,
    )

    start = time.perf_counter()
    model = learner.train(train_df[feature_cols + [TARGET, "sample_weight"]])
    train_time = time.perf_counter() - start

    y_test = test_df[TARGET].values
    y_proba_df = model.predict(test_df[feature_cols + [TARGET]])
    # ydf.predict returns probability of the positive class for binary tasks
    y_proba = np.asarray(y_proba_df)
    y_pred = (y_proba >= 0.5).astype(int)

    row = compute_metrics(y_test, y_pred, y_proba)
    row["Model_Size_Bytes"] = ydf_model_size_bytes(model, f"/tmp/ydf_model_{i}")
    row["Train_Time_s"] = train_time
    ydf_rows.append(row)

ydf_results = pd.DataFrame(ydf_rows)
print("\n=== YDF RandomForest — per-split results ===")
print(ydf_results.round(5))

# ---------------------------------------------------------------------------
# BLOCK 10 — Descriptive summary
# ---------------------------------------------------------------------------
print("\n=== scikit-learn summary (mean ± SD across", N_REPEATS, "splits) ===")
print(sklearn_results.agg(["mean", "std"]).round(5).T)

print("\n=== YDF summary (mean ± SD across", N_REPEATS, "splits) ===")
print(ydf_results.agg(["mean", "std"]).round(5).T)

# ---------------------------------------------------------------------------
# BLOCK 11 — Paired, variance-corrected significance testing (correction #6)
# Because both frameworks were evaluated on IDENTICAL splits, a paired test
# is now valid (ttest_rel), unlike the original independent-samples test.
# Additionally report the Nadeau-Bengio corrected variance estimate, since
# splits share overlapping training data and violate strict independence
# even when paired (Nadeau & Bengio, 2003, Machine Learning, 52(3), 239-281).
# ---------------------------------------------------------------------------
def nadeau_bengio_corrected_ttest(diffs: np.ndarray, n_train: int, n_test: int):
    """
    Corrected resampled paired t-test.
    diffs: array of per-split metric differences (sklearn - ydf)
    n_train, n_test: sizes of train/test partitions used in each split
    """
    n = len(diffs)
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)
    correction = (1 / n) + (n_test / n_train)
    corrected_var = var_diff * correction
    if corrected_var <= 0:
        return mean_diff, np.nan, np.nan
    t_stat = mean_diff / np.sqrt(corrected_var)
    df_ = n - 1
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df_))
    return mean_diff, t_stat, p_val

n_train_size = len(splits[0][0])
n_test_size = len(splits[0][1])

print("\n=== Paired significance testing (sklearn vs. YDF, same splits) ===")
for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Brier"]:
    diffs = sklearn_results[metric].values - ydf_results[metric].values

    paired_t = stats.ttest_rel(sklearn_results[metric], ydf_results[metric])
    mean_diff, nb_t, nb_p = nadeau_bengio_corrected_ttest(diffs, n_train_size, n_test_size)
    cohens_d = mean_diff / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else np.nan

    print(f"\n{metric}")
    print(f"  Standard paired t-test:      t={paired_t.statistic:.4f}, p={paired_t.pvalue:.6f}")
    print(f"  Nadeau-Bengio corrected:     t={nb_t:.4f}, p={nb_p:.6f}")
    print(f"  Mean difference (skl-ydf):   {mean_diff:.5f}")
    print(f"  Cohen's d:                   {cohens_d:.4f}")

# ---------------------------------------------------------------------------
# BLOCK 12 — Save all results for reproducibility / supplementary material
# ---------------------------------------------------------------------------
sklearn_results.to_csv("sklearn_rf_results.csv", index=False)
ydf_results.to_csv("ydf_rf_results.csv", index=False)
print("\nResults saved: sklearn_rf_results.csv, ydf_rf_results.csv")
