# RF vs. YDF-GBTL Benchmark — Setup & Run Instructions

This script reruns the benchmark with every issue from peer review fixed
(see the docstring at the top of `benchmark_rf_vs_ydf.py` for the full list
mapped to comment numbers).

## 1. Install dependencies

```bash
pip install pandas numpy scikit-learn ydf scipy joblib
```

`ydf` requires Python 3.9+ and a 64-bit OS (Linux, macOS, or Windows via WSL).
If you hit a `TypeError` mentioning an unexpected keyword argument when the
script calls `ydf.GradientBoostedTreesLearner(...)`, update to the latest
version first: `pip install -U ydf`. This script deliberately avoids
version-specific YDF parameters (see Section 6 below) so it should run on
any reasonably recent release, but updating is the first thing to try if
you see a YDF-related TypeError.

## 2. Get the dataset

Download the **full, imbalanced** release (253,680 rows) — not the
pre-balanced 50/50 version — from the UCI Machine Learning Repository:

- https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

or from the original Kaggle BRFSS2015 source you used before
(`diabetes_binary_health_indicators_BRFSS2015.csv`). Either source is fine
as long as it's the full imbalanced file with all 253,680 rows and the
`Diabetes_binary` label column.

## 3. Run the script

```bash
cd benchmark_rf_vs_ydf_script/
python benchmark_rf_vs_ydf.py --data /path/to/diabetes_binary_health_indicators_BRFSS2015.csv --trials 10
```

This will take a few minutes (YDF and RF are both trained 10 times each,
plus one extra fit each for variable importances). Expect roughly
5-15 minutes total depending on your machine.

Optional flags:
- `--trials N` — number of paired trials (default 10, matching the
  original manuscript's design)
- `--outdir DIR` — output directory (default `results/`)
- `--top-k K` — number of top predictor variables to report in Table 4
  (default 10)

## 4. What you'll get

A `results/` folder containing:

| File | Maps to |
|---|---|
| `table1_rf_trials.csv` | Manuscript Table 1 (RF per-trial metrics) |
| `table2_ydf_trials.csv` | Manuscript Table 2 (YDF-GBTL per-trial metrics) |
| `table3_paired_comparison.csv` | Manuscript Table 3 (mean±SD + paired-sample significance tests) |
| `table4_variable_importance.csv` | Manuscript Table 4 (top predictor variables for both models) |
| `run_summary.json` | Dataset size, prevalence, and configuration used — needed to correct the Abstract and Methodology sections |

Each per-trial table also includes the full confusion matrix (`tn`, `fp`,
`fn`, `tp`) and `auc_roc`, so you (or I, on the next pass) can immediately
sanity-check the numbers against the published baseline AUC (~0.80–0.83)
before they go back into the manuscript.

## 5. Class imbalance handling (now enabled by default)

`CLASS_IMBALANCE_STRATEGY` at the top of the script is set to
`"class_weight"`, so both models now train with balanced class weights:

- **RF** uses scikit-learn's `class_weight="balanced"`.
- **YDF-GBTL** uses a per-row sample-weight column, computed with the
  same formula (`n_samples / (n_classes * count_per_class)`) and passed
  via YDF's `weights="<column>"` parameter. An earlier version of this
  script used YDF's `class_weights=dict` parameter, but that parameter
  is only present in some YDF releases and raised
  `TypeError: unexpected keyword argument 'class_weights'` on others.
  The sample-weight-column approach works across YDF versions and
  produces the same effective weighting.

**Expect an asymmetric effect, and don't mistake it for a bug.** In
testing, YDF's recall rose sharply under weighting (it directly reweights
the boosting loss), while RF's recall moved only slightly (scikit-learn's
`class_weight` reweights the impurity criterion during tree growth, but
its effect on the final 0.5-threshold decision is documented to be
weaker than on boosting-based methods). This is a genuine, reportable
difference in how the two libraries implement class weighting — worth a
sentence in the Discussion — not a sign that the run failed. If you want
symmetric behavior instead, an alternative is to leave both models
unweighted and instead tune the decision threshold post hoc for each
model (not implemented here, but flag it and I can add it).

To turn weighting off and instead report raw/unweighted metrics, set
`CLASS_IMBALANCE_STRATEGY = None` near the top of the script before
running.

If you still hit a `TypeError` related to YDF's constructor after this
fix, it likely means your installed `ydf` version's `train()` doesn't
accept the sample-weight column the way expected here — run
`pip install -U ydf` to update to the latest release, then re-run.
