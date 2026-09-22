# Reproducibility Guide

## Recorded core environment

- Python 3.11.11
- NumPy 1.26.4
- pandas 2.2.3
- SciPy 1.15.2
- scikit-learn 1.6.1
- statsmodels 0.14.4
- matplotlib 3.10.0

## Run order

1. Put `df_LR.parquet` in `HF_STUDY_DATA_DIR`.
2. Run `python 01_model_development.py`.
3. Confirm `artifacts/run_provenance.json` reports a completed run.
4. Run `python 02_tables_figures.py`.

## Analysis version

`v9_visit_window_first_outpatient_composite`

## Numerical settings

| Setting | Value |
|---|---:|
| Prediction horizon | 180 days |
| Lab look-back | 14 days |
| Primary MICE | 20 datasets x 20 cycles |
| PMM donors | 5 |
| Internal bootstrap | 200 |
| Bootstrap MICE | 5 datasets x 5 cycles |
| Primary conditional CI bootstrap | 1000 |
| IECV CI bootstrap | 500 |
| Subgroup bootstrap | 1000 |
| Sensitivity bootstrap | 1000 |
| Master seed | 20260916 |

## Checkpointing

Script 01 saves intermediate checkpoints and uses configuration/data signatures before reusing cached stages.

## Computational approximation

The primary analysis uses 20 imputed datasets and 20 cycles per block. For computational feasibility, bootstrap redevelopment uses 5 datasets and 5 cycles per block while repeating functional-form assessment by default. This approximation is explicitly documented in the analysis outputs and should remain disclosed.

## Sensitive generated outputs

Patient-level predictions, model-state pickle files, completed/imputed data, identifiers, and local workbooks must not be uploaded publicly. The supplied `.gitignore` excludes common sensitive output formats and directories.
