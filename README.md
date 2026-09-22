# hf-outpatient-risk-updating

Reproducible analysis code for a multicenter retrospective study evaluating whether updating discharge laboratory measurements with recent outpatient laboratory values improves prediction of subsequent 180-day heart-failure rehospitalization or all-cause death.

## Study design

- **Early cohort:** first outpatient visit on days 15-45 after discharge.
- **Later cohort:** first outpatient visit on days 75-105 after discharge.
- **Prediction time:** the actual selected outpatient visit date.
- Patients may contribute to both cohorts.
- **Current laboratory values:** latest value in the 14-day look-back window ending on the index outpatient visit.
- **Comparator:** discharge-value model fitted in the same patients at the same prediction time and over the same follow-up period.
- **Outcome:** first recorded emergency rehospitalization for heart failure or all-cause death during the subsequent 180 days.

## Files

| File | Purpose |
|---|---|
| `01_model_development.py` | Cohort construction, MICE, Cox model development, internal validation, IECV, subgroup/sensitivity analyses, model saving, source-table export |
| `02_tables_figures.py` | Manuscript-facing tables and figures from saved outputs of Script 01 |
| `MODEL_SPECIFICATION.md` | Model and validation specification |
| `DATA_SCHEMA.md` | Required source columns and timing conventions |
| `REPRODUCIBILITY.md` | Run order and computational settings |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes patient-level data and generated artifacts |
| `PUBLIC_RELEASE_NOTES.md` | Public-release-only changes |

## Data availability

Patient-level data are not included because they contain confidential clinical information. The scripts expect:

```text
df_LR.parquet
```

in the directory specified by `HF_STUDY_DATA_DIR`.

## Installation

Python 3.11.11 was used.

```bash
pip install -r requirements.txt
```

## Configure the data directory

### Windows PowerShell

```powershell
$env:HF_STUDY_DATA_DIR = "C:\path\to\study"
python 01_model_development.py
python 02_tables_figures.py
```

### macOS / Linux

```bash
export HF_STUDY_DATA_DIR="/path/to/study"
python 01_model_development.py
python 02_tables_figures.py
```

If `HF_STUDY_DATA_DIR` is not set, the current working directory is used.

## Output directory

```text
HF_visit_window_Cox_FULL_v9_composite/
```

This contains caches, model files, artifacts, tables, figures, and parameter exports. Some generated files contain patient-level identifiers, dates, predictions, or imputed values; do not commit them publicly.

## Main statistical methods

- Standard Cox proportional hazards regression
- No L2 penalization or automated predictor selection
- MICE with predictive mean matching
- Bootstrap optimism correction
- Leave-one-center-out internal-external cross-validation
- Time-dependent AUC at 180 days, Harrell's C-index, Brier score, calibration-in-the-large, and calibration slope
- Exploratory subgroup analyses without refitting
- Sensitivity analyses for longitudinal changes and alternative missing-data approaches

## Reproducibility

Master random seed:

```text
20260916
```

Primary development uses 20 imputed datasets with 20 chained-equation cycles per block. Bootstrap redevelopment uses a reduced 5-dataset / 5-cycle profile as an explicitly documented computational approximation.

## Suggested manuscript code-availability statement

> The analysis and prediction code, together with model specifications and accompanying documentation, are publicly available at [repository URL], version [release/tag]. Patient-level data are not publicly available because of patient confidentiality.

## AI-assisted coding disclosure

ChatGPT (OpenAI) assisted with drafting and revising portions of the analysis code. The authors reviewed and edited the code, executed the analyses, and verified the resulting outputs.

## License

No software license is included in this package. Add an institutional/preferred license before release if reuse rights are to be granted.
