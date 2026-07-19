"""
Matched-Condition Comparison (Enhanced): scikit-learn HistGradientBoostingClassifier
vs. YDF GradientBoostedTreesLearner
Dataset: BRFSS 2015 Diabetes Health Indicators (UCI ML Repository)

CHANGES FROM THE PRIOR VERSION (two priority enhancements implemented):

1. REPEATED STRATIFIED K-FOLD CROSS-VALIDATION replaces the 10 independent
   80/20 hold-out splits. With RepeatedStratifiedKFold(n_splits=5, n_repeats=5),
   every one of the 25 runs uses a disjoint 20% test fold within each repeat,
   so every observation is evaluated exactly once per repeat rather than only
   whichever 20% happened to land in a single hold-out draw. This is the
   design standard expected for tabular ML comparisons (Kohavi, 1995) and
   gives 25 paired observations instead of 10, improving the power of the
   downstream significance/equivalence tests.

2. TOST EQUIVALENCE TESTING (Lakens, 2017, Soc. Psychol. Personal. Sci.)
   is added alongside the existing paired t-test / Nadeau-Bengio correction.
   A non-significant paired t-test only means "we failed to detect a
   difference" -- it does NOT positively demonstrate equivalence. TOST
   (two one-sided tests) tests the complementary, positively-stated
   hypothesis: "the difference lies within a pre-specified, practically
   negligible equivalence margin." This is the statistically correct tool
   for making a "the two frameworks perform equivalently" claim, rather
   than relying on the absence of significance as implicit evidence of
   equivalence (a common and reviewer-flagged error).

   Equivalence margins are set per metric based on customary "practically
   negligible" thresholds in clinical ML literature (+/-1 percentage point
   for proportions/rates such as Accuracy, Precision, Recall, F1, ROC-AUC,
   PR-AUC; +/-0.01 for Brier score). These margins are a judgment call and
   MUST be justified and cited in your Methods section BEFORE seeing the
   results, not chosen post hoc to produce a desired equivalence
   conclusion -- doing so after inspecting results would itself be a
   methodological violation (equivalence margins are pre-registered by
   convention in equivalence-testing literature).

All prior corrections (matched splits shared across both learners, 1.5x IQR
BMI filter, consistent class-imbalance handling, symmetric metric definitions,
comparable on-disk model-size measurement) are preserved unchanged.

Run in an environment with: pandas, numpy, scikit-learn, scipy, statsmodels,
ydf installed.
    pip install statsmodels --break-system-packages   (if not already present)
"""

import time
import os
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.weightstats import ttost_paired
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import ydf

RANDOM_SEED = 42
N_SPLITS = 5  # folds per repeat
N_REPEATS = 5  # repeats -> 25 total paired runs
TARGET = "Diabetes_binary"

# Equivalence margins (per-metric), set a priori and disclosed in Methods.
# These represent the maximum difference considered "practically negligible"
# for each metric, not derived from or tuned to the observed results.
EQUIVALENCE_MARGINS = {
    "Accuracy": 0.01,
    "Precision": 0.01,
    "Recall": 0.01,
    "F1": 0.01,
    "ROC_AUC": 0.01,
    "PR_AUC": 0.01,
    "Brier": 0.01,
}

# ---------------------------------------------------------------------------
# BLOCK 1 — Load data and report class balance
# ---------------------------------------------------------------------------
df = pd.read_csv("data.csv")
df[TARGET] = df[TARGET].astype(int)

print("Full dataset shape:", df.shape)
print(df[TARGET].value_counts(normalize=True).rename("proportion"))

# ---------------------------------------------------------------------------
# BLOCK 2 — Principled outlier handling (1.5x IQR on BMI, unchanged)
# ---------------------------------------------------------------------------
q1 = df["BMI"].quantile(0.25)
q3 = df["BMI"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

n_before = df.shape[0]
df = df[(df["BMI"] >= lower_bound) & (df["BMI"] <= upper_bound)].reset_index(drop=True)
n_after = df.shape[0]
print(
    f"Outlier removal (1.5x IQR on BMI): {n_before} -> {n_after} "
    f"({100 * (n_before - n_after) / n_before:.2f}% removed)"
)
print(df[TARGET].value_counts(normalize=True).rename("proportion (post-filter)"))

feature_cols = [c for c in df.columns if c != TARGET]

# ---------------------------------------------------------------------------
# BLOCK 3 — Repeated Stratified K-Fold splits (PRIORITY ENHANCEMENT #1)
# Generated ONCE and reused identically for both learners, exactly as the
# prior hold-out splits were shared -- this preserves the "matched
# conditions" property that makes the paired tests valid.
# ---------------------------------------------------------------------------
rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED
)
splits = list(rskf.split(df[feature_cols], df[TARGET]))
N_RUNS = len(splits)  # = N_SPLITS * N_REPEATS = 25
print(
    f"\nGenerated {N_RUNS} paired train/test runs via "
    f"RepeatedStratifiedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS})."
)


# ---------------------------------------------------------------------------
# BLOCK 4 — Consistent class-imbalance handling (unchanged)
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
# BLOCK 5 — Comparable model-size measurement (unchanged)
# ---------------------------------------------------------------------------
def sklearn_model_size_bytes(clf, tmp_path="/tmp/sklearn_model.joblib") -> int:
    joblib.dump(clf, tmp_path, compress=0)
    size = os.path.getsize(tmp_path)
    os.remove(tmp_path)
    return size


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
# BLOCK 6 — Symmetric metric computation (unchanged)
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
# BLOCK 7 — sklearn HistGradientBoostingClassifier: repeated k-fold loop
# ---------------------------------------------------------------------------
sklearn_rows = []
sklearn_n_iter_actual = []
for i, (train_idx, test_idx) in enumerate(splits):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

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
    sklearn_n_iter_actual.append(clf.n_iter_)

    if (i + 1) % 5 == 0:
        print(f"  [sklearn] completed run {i + 1}/{N_RUNS}")

sklearn_results = pd.DataFrame(sklearn_rows)
print("\n=== scikit-learn HistGradientBoostingClassifier — per-run results ===")
print(sklearn_results.round(5))

# ---------------------------------------------------------------------------
# BLOCK 8 — YDF GradientBoostedTreesLearner: SAME repeated k-fold loop
# ---------------------------------------------------------------------------
ydf_rows = []
ydf_num_trees_actual = []
for i, (train_idx, test_idx) in enumerate(splits):
    train_df = add_balanced_sample_weight(df.iloc[train_idx], TARGET)
    test_df = df.iloc[test_idx].copy()

    learner = ydf.GradientBoostedTreesLearner(
        label=TARGET,
        weights="sample_weight",
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

    try:
        ydf_num_trees_actual.append(model.num_trees())
    except AttributeError:
        ydf_num_trees_actual.append(None)

    if (i + 1) % 5 == 0:
        print(f"  [YDF] completed run {i + 1}/{N_RUNS}")

ydf_results = pd.DataFrame(ydf_rows)
print("\n=== YDF GradientBoostedTreesLearner — per-run results ===")
print(ydf_results.round(5))

# ---------------------------------------------------------------------------
# BLOCK 9 — Descriptive summary
# ---------------------------------------------------------------------------
print(f"\n=== scikit-learn HGBC summary (mean ± SD across {N_RUNS} runs) ===")
print(sklearn_results.agg(["mean", "std"]).round(5).T)

print(f"\n=== YDF GBT summary (mean ± SD across {N_RUNS} runs) ===")
print(ydf_results.agg(["mean", "std"]).round(5).T)

print("\n=== Tree counts (post early-stopping) ===")
print(
    f"sklearn n_iter: mean={np.mean(sklearn_n_iter_actual):.1f}, "
    f"std={np.std(sklearn_n_iter_actual, ddof=1):.1f}"
)
print(
    f"YDF num_trees:  mean={np.mean(ydf_num_trees_actual):.1f}, "
    f"std={np.std(ydf_num_trees_actual, ddof=1):.1f}"
)


# ---------------------------------------------------------------------------
# BLOCK 10 — Paired, variance-corrected significance testing (unchanged
# methodology, now run across N_RUNS=25 repeated-k-fold observations)
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


# n_train / n_test are now approximately fixed by the fold structure:
# with N_SPLITS=5, each test fold is ~1/5 of the data, train is the rest.
n_test_size = len(splits[0][1])
n_train_size = len(splits[0][0])

metrics_list = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "Brier"]

print("\n=== Paired significance testing (sklearn HGBC vs. YDF GBT) ===")
sig_results = {}
for metric in metrics_list:
    diffs = sklearn_results[metric].values - ydf_results[metric].values

    paired_t = stats.ttest_rel(sklearn_results[metric], ydf_results[metric])
    mean_diff, nb_t, nb_p = nadeau_bengio_corrected_ttest(
        diffs, n_train_size, n_test_size
    )
    cohens_d = (
        mean_diff / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else np.nan
    )

    sig_results[metric] = {
        "mean_diff": mean_diff,
        "paired_t_p": paired_t.pvalue,
        "nb_corrected_p": nb_p,
        "cohens_d": cohens_d,
    }

    print(f"\n{metric}")
    print(
        f"  Standard paired t-test:      t={paired_t.statistic:.4f}, p={paired_t.pvalue:.6f}"
    )
    print(f"  Nadeau-Bengio corrected:     t={nb_t:.4f}, p={nb_p:.6f}")
    print(f"  Mean difference (skl-ydf):   {mean_diff:.5f}")
    print(f"  Cohen's d:                   {cohens_d:.4f}")

# ---------------------------------------------------------------------------
# BLOCK 11 — TOST equivalence testing (PRIORITY ENHANCEMENT #2)
# Tests, for each metric, whether the mean difference lies within the
# pre-specified equivalence margin [-margin, +margin]. A significant TOST
# result (p < 0.05) supports a positive equivalence claim; it is a
# DIFFERENT question from the paired t-test above, and both should be
# reported together, not as substitutes for one another.
# ---------------------------------------------------------------------------
print("\n=== TOST Equivalence Testing ===")
print("(equivalence margins pre-specified per metric; see EQUIVALENCE_MARGINS)")
tost_results = {}
for metric in metrics_list:
    margin = EQUIVALENCE_MARGINS[metric]
    x1 = sklearn_results[metric].values
    x2 = ydf_results[metric].values

    overall_p, (t_lower, p_lower, df_lower), (t_upper, p_upper, df_upper) = (
        ttost_paired(x1, x2, -margin, margin)
    )

    equivalent = overall_p < 0.05
    tost_results[metric] = {
        "margin": margin,
        "overall_p": overall_p,
        "equivalent_at_0.05": equivalent,
    }

    print(f"\n{metric} (equivalence margin: ±{margin})")
    print(f"  TOST overall p-value:        {overall_p:.6f}")
    print(f"  Lower bound test:            t={t_lower:.4f}, p={p_lower:.6f}")
    print(f"  Upper bound test:            t={t_upper:.4f}, p={p_upper:.6f}")
    print(f"  Statistically equivalent (α=0.05)?  {'YES' if equivalent else 'NO'}")

# ---------------------------------------------------------------------------
# BLOCK 12 — Combined summary table (for direct inclusion in a manuscript
# Results table: difference test AND equivalence test side by side)
# ---------------------------------------------------------------------------
summary_rows = []
for metric in metrics_list:
    summary_rows.append(
        {
            "Metric": metric,
            "Mean_sklearn": sklearn_results[metric].mean(),
            "Mean_YDF": ydf_results[metric].mean(),
            "Mean_Diff": sig_results[metric]["mean_diff"],
            "Paired_t_p": sig_results[metric]["paired_t_p"],
            "NB_corrected_p": sig_results[metric]["nb_corrected_p"],
            "Cohens_d": sig_results[metric]["cohens_d"],
            "TOST_margin": tost_results[metric]["margin"],
            "TOST_p": tost_results[metric]["overall_p"],
            "Statistically_Equivalent": tost_results[metric]["equivalent_at_0.05"],
        }
    )

summary_table = pd.DataFrame(summary_rows)
print("\n=== Combined Results + Equivalence Summary Table ===")
print(summary_table.round(6).to_string(index=False))

# ---------------------------------------------------------------------------
# BLOCK 13 — Save all results for reproducibility / supplementary material
# ---------------------------------------------------------------------------
sklearn_results.to_csv("sklearn_hgbc_results_rskf.csv", index=False)
ydf_results.to_csv("ydf_gbt_results_rskf.csv", index=False)
summary_table.to_csv("gbt_comparison_summary_with_equivalence.csv", index=False)

pd.DataFrame(
    {
        "run": list(range(N_RUNS)),
        "sklearn_n_iter": sklearn_n_iter_actual,
        "ydf_num_trees": ydf_num_trees_actual,
    }
).to_csv("gbt_actual_tree_counts_rskf.csv", index=False)

print("\nResults saved:")
print("  sklearn_hgbc_results_rskf.csv")
print("  ydf_gbt_results_rskf.csv")
print("  gbt_comparison_summary_with_equivalence.csv")
print("  gbt_actual_tree_counts_rskf.csv")
