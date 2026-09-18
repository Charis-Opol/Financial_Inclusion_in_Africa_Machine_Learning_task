# EDA Summary — Phase 1

Source notebook: `notebooks/01_eda.ipynb`. Dataset: `data/raw/Train_v2.csv`
(23,524 rows), `data/raw/Test_v2.csv` (10,086 rows). Target: `bank_account`
(No 85.9% / Yes 14.1%).

Each finding below maps to its numbered task in `Implementation_plan.md`
Phase 1 and states the decision it produced.

---

### 1.1 — Per-country target distribution

| Country | No | Yes |
|---|---|---|
| Kenya | 74.9% | **25.1%** |
| Rwanda | 88.5% | 11.5% |
| Tanzania | 90.8% | 9.2% |
| Uganda | 91.4% | 8.6% |

**Decision:** confirms the ~3x spread across countries. Stratify CV by
country+target jointly (Phase 2.8); report metrics per-country throughout
(Phase 3.7, 4.9), since Uganda is the brief's primary country of interest.

### 1.2 — Per-country feature distributions

`cellphone_access` ranges from 59.8% (Tanzania) to 83.0% (**Rwanda**,
highest), with Kenya at 78.9%. Distributions of `education_level`,
`job_type`, and `location_type` also differ materially by country.

**Decision:** country is a real structural effect, not noise — keep it as
a pooled feature. Note Rwanda has *higher* phone access than Kenya but a
*much lower* account-ownership rate, so phone access alone doesn't explain
Kenya's gap; whatever does (plausibly M-Pesa-specific mobile-money
adoption) is only recoverable through the `country` feature itself, which
further justifies keeping it in the model rather than relying on
`cellphone_access` as a proxy.

### 1.3 — Missingness-in-disguise correlation check

| Placeholder | Column | n | bank_account='Yes' rate | Overall |
|---|---|---|---|---|
| "Dont know" | `marital_status` | 8 | 25.0% | 14.1% |
| "Other/Dont know/RTA" | `education_level` | 35 | 31.4% | 14.1% |
| "Dont Know/Refuse to answer" | `job_type` | 126 | 11.1% | 14.1% |

**Decision:** two of three placeholders show a rate well above baseline
(clear signal); `job_type`'s is close to baseline (weak signal on its
own). All three are kept as explicit categories rather than
imputed/dropped — treating them inconsistently based on a per-column
threshold isn't justified when 2 of 3 clearly aren't random noise.

### 1.4 — Bivariate: bank_account rate by key categoricals

`cellphone_access` and `education_level` show the largest spread between
levels. `relationship_with_head` and `marital_status` show flatter, more
overlapping rates.

**Decision:** `cellphone_access`, `education_level`, `country` are
expected to dominate SHAP importance in Phase 4.8 — this is the EDA
prediction that result should be checked against. The flatter
`relationship_with_head`/`marital_status` rates are a first hint toward
the redundancy tested directly in 1.9.

### 1.5 — household_size and age_of_respondent vs. target (binned)

Account ownership peaks in working-age bins (25–55) and falls off for
both the youngest and oldest respondents — non-monotonic.
`household_size` shows a weaker, noisier relationship with the target.

**Decision:** the non-linear age relationship is invisible to a linear
model on raw age. Confirms `LogisticRegressionModel` is deliberately
scoped as a sanity-check floor in Phase 3, not a competitive model — tree
and MLP models can capture this shape natively.

### 1.6 — household_size outlier sanity check (>15)

8 rows, spread across Kenya and Uganda, with plausible respondent ages
(24–70) — not obvious data-entry errors (e.g. no age=0 or duplicated
rows accompanying them).

**Decision:** flag rather than drop/cap (Phase 2.3) — extended households
are a real phenomenon here, and 8 rows out of 23,524 is too small a
population to justify assuming they're errors without more evidence.

### 1.7 — Cardinality audit

| Column | Unique levels |
|---|---|
| `job_type` | 10 |
| `relationship_with_head` | 6 |
| `education_level` | 6 |
| `marital_status` | 5 |
| `country` | 4 |
| `gender_of_respondent` | 2 |
| `location_type` | 2 |
| `cellphone_access` | 2 |

**Decision:** `job_type` and `relationship_with_head` are the highest-
cardinality columns — motivates the two-branch encoding in Phase 2.4
(one-hot for XGBoost, learned embeddings for PyTorch), since embeddings
are most interesting as an ablation axis precisely where cardinality is
non-trivial.

### 1.8 — Train/test category consistency check

Zero unseen categories in test relative to train, across all 8
categorical columns.

**Decision:** this particular train/test split doesn't exercise unseen-
category handling, but Phase 2.5's explicit unseen-category logic is still
required and tested — CV fold splits or future retrains can produce
genuinely unseen categories even when the full split happens not to.

### 1.9 — relationship_with_head × marital_status crosstab

"Head of Household" is 45.1% Married, 19.9% Widowed, 25.8% Single, 9.1%
Divorced. "Spouse" is 71.5% Married, 28.3% Single. Every
`relationship_with_head` level spans multiple `marital_status` values.

**Decision:** strong but *not total* overlap — doesn't clear the bar set
in Phase 2.7 for engineering a combined feature. Both columns kept
separately; collapsing them would lose real, non-redundant variation
(e.g. unmarried partners recorded as "Spouse").

### 1.10 — uniqueid uniqueness check (composite key confirmation)

`uniqueid` alone: 14,789 duplicate values (not a valid key — repeats
across country/year). `uniqueid + country + year`: 0 duplicates (valid
composite key).

**Decision:** enforced as a hard invariant in
`DataLoader._validate_composite_key` (Phase 2.1) so no downstream code can
silently join on `uniqueid` alone.

### 1.11 — Correlation matrix (numeric + binarized features)

Pearson correlation over `age_of_respondent`, `household_size`, `year`,
and binarized (0/1) `bank_account`, `cellphone_access`, `location_type`,
`gender_of_respondent`:

| | age | hh_size | year | bank_acct | cellphone | urban | male |
|---|---|---|---|---|---|---|---|
| age | 1.00 | -0.13 | -0.02 | 0.02 | -0.10 | -0.05 | 0.01 |
| household_size | -0.13 | 1.00 | -0.05 | -0.03 | 0.09 | -0.26 | 0.01 |
| year | -0.02 | -0.05 | 1.00 | 0.11 | -0.07 | 0.21 | 0.00 |
| bank_account | 0.02 | -0.03 | 0.11 | 1.00 | 0.21 | 0.09 | 0.12 |
| cellphone_access | -0.10 | 0.09 | -0.07 | 0.21 | 1.00 | -0.09 | 0.10 |
| location_urban | -0.05 | -0.26 | 0.21 | 0.09 | -0.09 | 1.00 | 0.01 |
| gender_male | 0.01 | 0.01 | 0.00 | 0.12 | 0.10 | 0.01 | 1.00 |

`job_type`, `education_level`, and `relationship_with_head` are excluded
— they're nominal with no natural 0/1 or ordinal encoding, and imposing
one just for this matrix would presuppose Phase 2's encoding decision.

**Decision:** `cellphone_access` (0.21) is the strongest linear correlate
of the target among these features, consistent with 1.4's bivariate
finding. `age_of_respondent`'s linear correlation is weak (0.02) despite
the real relationship found in 1.5 — confirms that finding was non-linear
(binned rates caught what a correlation coefficient misses), not spurious.
No feature pair is strongly collinear (largest off-diagonal magnitude:
-0.26, `household_size`/`location_urban`), so multicollinearity isn't a
concern for the `LogisticRegressionModel` baseline in Phase 3.

---

## Summary of decisions carried into Phase 2

| Phase 2 task | Driven by |
|---|---|
| 2.1 Composite-key enforcement in `DataLoader` | 1.10 |
| 2.2 Keep missingness-in-disguise as explicit category | 1.3 |
| 2.3 Flag (don't drop) `household_size` outliers | 1.6 |
| 2.4 One-hot (XGBoost) + embeddings (PyTorch) branches | 1.7 |
| 2.5 Unseen-category handling, fail loudly | 1.8 |
| 2.7 Do *not* engineer combined relationship/marital feature | 1.9 |
| 2.8 Country in pooled features + CV stratification key | 1.1, 1.2 |
