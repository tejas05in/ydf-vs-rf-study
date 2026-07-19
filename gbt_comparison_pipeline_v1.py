"""
Matched-Condition Comparison: scikit-learn HistGradientBoostingClassifier
vs. YDF GradientBoostedTreesLearner
Dataset: BRFSS 2015 Diabetes Health Indicators (UCI ML Repository)

Design decisions (explicitly disclosed here for the Methods section):
  - sklearn's HistGradientBoostingClassifier was selected over the classic
    GradientBoostingClassifier because both HGBC and YDF's GBT use
    histogram-binned split-finding, making this the correct algorithmic
    counterpart rather than an apples-to-oranges comparison.
  - Both learners are run at their OWN LIBRARY DEFAULT hyperparameters
    (per the user's explicit choice), not matched capacity. This means
    the comparison answers "how do the two libraries perform out-of-the-box,"
    NOT "how does the algorithm perform under equalized capacity." This
    must be stated as a limitation, since:
        sklearn HistGradientBoostingClassifier defaults:
            learning_rate=0.1, max_iter=100, max_leaf_nodes=31,
            max_depth=None (unconstrained), min_samples_leaf=20,
            l2_regularization=0.0, max_bins=255, early_stopping='auto'
        YDF GradientBoostedTreesLearner defaults:
            num_trees=300, shrinkage=0.1, max_depth=6,
            l1/l2_regularization=0, growing_strategy='LOCAL',
            early_stopping='LOSS_INCREASE' (enabled by default)
    YDF trains 3x more boosting rounds by default and both use early
    stopping, so the "final" number of trees actually used by each model
    is itself a result to report, not assumed equal.

All other corrections carried over from the prior RF study apply
identically here:
  1. Same 10 matched stratified train/test splits reused for both learners.
  2. Same principled 1.5x IQR BMI outlier filter.
  3. Consistent class-imbalance handling: HGBC's native class_weight="balanced"
     paired with an equivalent explicit sample_weight column for YDF.
  4. Symmetric metric definitions (positive class = 1 = diabetic, both).
  5. Comparable model-size measurement (on-disk serialized bytes, both).
  6. Paired / Nadeau-Bengio corrected significance testing.

Run in an environment with: pandas, numpy, scikit-learn, scipy, ydf installed.
"""

import time
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score, accuracy_score
)
import ydf

RANDOM_SEED = 42
N_REPEATS = 10
TARGET = "Diabetes_binary"

# ---------------------------------------------------------------------------
# BLOCK 1 — Load data and report class balance (no filtering yet)
# ---------------------------------------------------------------------------
df = pd.read_csv("data.csv")
df[TARGET] = df[TARGET].astype(int)

print("Full dataset shape:", df.shape)
print(df[TARGET].value_counts(normalize=True).rename("proportion"))

# ---------------------------------------------------------------------------
# BLOCK 2 — Principled outlier handling (same 1.5x IQR rule as the RF study,
# for direct comparability across your two experiments)
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
print(df[TARGET].value_counts(normalize=True).rename("proportion (post-filter)"))

# ---------------------------------------------------------------------------
# BLOCK 3 — Fixed, shared data splits (identical protocol to the RF study)
# ---------------------------------------------------------------------------
splits = []
for i in range(N_REPEATS):
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df[TARGET], random_state=i
    )
    splits.append((train_idx, test_idx))

feature_cols = [c for c in df.columns if c != TARGET]

# ---------------------------------------------------------------------------
# BLOCK 4 — Consistent class-imbalance handling
# HGBC has a native class_weight="balanced" option (added in sklearn 1.2+),
# computed as: weight_i = n_samples / (n_classes * count(class_i))
# The identical formula is applied explicitly for YDF's sample_weight.
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
# BLOCK 5 — Comparable model-size measurement (on-disk bytes, both frameworks)
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
# BLOCK 6 — Symmetric metric computation (positive class = 1, both frameworks)
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
# BLOCK 7 — sklearn HistGradientBoostingClassifier evaluation loop
# Library defaults used throughout, EXCEPT class_weight and random_state,
# which are required for imbalance-handling consistency and reproducibility
# respectively (neither changes model capacity).
# ---------------------------------------------------------------------------
sklearn_rows = []
sklearn_n_iter_actual = []  # tracks early-stopping effect on tree count
for i, (train_idx, test_idx) in enumerate(splits):
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET].values
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET].values

    clf = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=RANDOM_SEED,
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
    sklearn_n_iter_actual.append(clf.n_iter_)  # trees actually used post early-stopping

sklearn_results = pd.DataFrame(sklearn_rows)
print("\n=== scikit-learn HistGradientBoostingClassifier — per-split results ===")
print(sklearn_results.round(5))
print("Actual n_iter_ per split (post early-stopping):", sklearn_n_iter_actual)

# ---------------------------------------------------------------------------
# BLOCK 8 — YDF GradientBoostedTreesLearner evaluation loop
# Library defaults used throughout (num_trees=300, shrinkage=0.1, max_depth=6,
# early_stopping='LOSS_INCREASE' by default), plus sample_weight for parity
# with sklearn's class_weight="balanced".
# ---------------------------------------------------------------------------
ydf_rows = []
ydf_num_trees_actual = []  # tracks early-stopping effect on tree count
for i, (train_idx, test_idx) in enumerate(splits):
    train_df = add_balanced_sample_weight(df.loc[train_idx], TARGET)
    test_df = df.loc[test_idx].copy()

    learner = ydf.GradientBoostedTreesLearner(
        label=TARGET,
        weights="sample_weight",
        # all other hyperparameters left at library default, per design choice
    )

    start = time.perf_counter()
    model = learner.train(train_df[feature_cols + [TARGET, "sample_weight"]])
    train_time = time.perf_counter() - start

    y_test = test_df[TARGET].values
    y_proba = np.asarray(model.predict(test_df[feature_cols + [TARGET]]))
    y_pred = (y_proba >= 0.5).astype(int)

    row = compute_metrics(y_test, y_pred, y_proba)
    row["Model_Size_Bytes"] = ydf_model_size_bytes(model, f"/tmp/ydf_gbt_model_{i}")
    row["Train_Time_s"] = train_time
    ydf_rows.append(row)

    # Extract actual number of trees used (reflects early stopping behavior)
    try:
        ydf_num_trees_actual.append(model.num_trees())
    except AttributeError:
        ydf_num_trees_actual.append(None)  # fallback if API differs by version

ydf_results = pd.DataFrame(ydf_rows)
print("\n=== YDF GradientBoostedTreesLearner — per-split results ===")
print(ydf_results.round(5))
print("Actual num_trees per split (post early-stopping):", ydf_num_trees_actual)

# ---------------------------------------------------------------------------
# BLOCK 9 — Descriptive summary
# ---------------------------------------------------------------------------
print("\n=== scikit-learn HGBC summary (mean ± SD across", N_REPEATS, "splits) ===")
print(sklearn_results.agg(["mean", "std"]).round(5).T)

print("\n=== YDF GBT summary (mean ± SD across", N_REPEATS, "splits) ===")
print(ydf_results.agg(["mean", "std"]).round(5).T)

# ---------------------------------------------------------------------------
# BLOCK 10 — Paired, variance-corrected significance testing
# Valid because both frameworks were evaluated on IDENTICAL splits.
# ---------------------------------------------------------------------------
def nadeau_bengio_corrected_ttest(diffs: np.ndarray, n_train: int, n_test: int):
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

# IMPORTANT: replace these with your ACTUAL post-filter train/test sizes,
# read directly from len(splits[0][0]) and len(splits[0][1]) at runtime.
n_train_size = len(splits[0][0])
n_test_size = len(splits[0][1])

print("\n=== Paired significance testing (sklearn HGBC vs. YDF GBT, same splits) ===")
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
# BLOCK 11 — Save all results for reproducibility / supplementary material
# ---------------------------------------------------------------------------
sklearn_results.to_csv("sklearn_hgbc_results.csv", index=False)
ydf_results.to_csv("ydf_gbt_results.csv", index=False)

pd.DataFrame({
    "split": list(range(N_REPEATS)),
    "sklearn_n_iter": sklearn_n_iter_actual,
    "ydf_num_trees": ydf_num_trees_actual,
}).to_csv("gbt_actual_tree_counts.csv", index=False)

print("\nResults saved: sklearn_hgbc_results.csv, ydf_gbt_results.csv, "
      "gbt_actual_tree_counts.csv")
