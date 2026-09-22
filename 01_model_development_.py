# %% [markdown]
# # HF outpatient-window Cox v9: model development and evaluation
# 
# ## Final design
# 
# - **Early:** first outpatient visit between days 15 and 45 after discharge.
# - **Later:** first outpatient visit between days 75 and 105 after discharge.
# - **Time zero:** the actual selected outpatient visit date.
# - **Early and Later cohorts are separately defined; patients may contribute to both.**
# - **Current laboratory values:** latest value in the 14-day look-back window ending on the index outpatient visit.
# - **Comparator:** the same patients, same time zero, and same follow-up, using discharge laboratory values.
# - **Common timing adjustment:** days from discharge to the index outpatient visit is included in both models as a prespecified linear term.
# - **Outcome:** first emergency readmission or all-cause death within 180 days after time zero.
# - **Same-day or earlier events:** excluded because temporal ordering relative to the outpatient assessment is not known from day-level data.
# - **Model:** standard Cox proportional hazards regression; no L2 penalty and no tuning CV.
# - **Missing data:** MICE with PMM; baseline/discharge variables are completed without outpatient variables and then frozen before current values are imputed.
# - **Validation:** bootstrap optimism correction, paired model comparison, internal-external validation by hospital, subgroup and sensitivity analyses.
# 
# Notebook 01 performs all model fitting, validation, confidence-interval calculation, and artifact saving.
# 

# %% [markdown]
# ## Settings

# %%
from __future__ import annotations

from pathlib import Path
import os
import json
import math
import pickle
import hashlib
import time as _clock
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy.integrate import trapezoid
from scipy.linalg import qr
from scipy.stats import chi2, t as student_t
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
from IPython.display import display

ANALYSIS_VERSION = "v9_visit_window_first_outpatient_composite"
MODEL_FAMILY = "Standard Cox proportional hazards"
COX_TIES = "breslow"
OUTCOME_MODE = "READMISSION_OR_DEATH"
DATA_DIR = Path(os.environ.get("HF_STUDY_DATA_DIR", ".")).expanduser().resolve()
PARQUET_PATH = DATA_DIR / "df_LR.parquet"
OUTPUT_DIR = DATA_DIR / "HF_visit_window_Cox_FULL_v9_composite"
CACHE_DIR = OUTPUT_DIR / "cache"
MODEL_DIR = OUTPUT_DIR / "models"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
PARAMETER_DIR = OUTPUT_DIR / "model_parameters"
RESULT_XLSX = OUTPUT_DIR / "HF_visit_window_Cox_FULL_v9_results.xlsx"

MASTER_SEED = 20260916
HORIZON = 180
LAB_LOOKBACK_DAYS = 14
EARLY_WINDOW = (15, 45)
LATER_WINDOW = (75, 105)
EVAL_TIMES = np.arange(1, HORIZON + 1, dtype=float)  # IBS: days 1-180, as in v6
LOOP_DIURETIC_COLUMN_MANUAL = "Loop_内服"

# Primary and leave-one-center-out development: no L2 and no tuning CV.
MICE_M = 20
MICE_ITERATIONS = 20
PMM_DONORS = 5
NONLINEARITY_MI_DATASETS = 20
NONLINEARITY_ALPHA = 0.05

# Practical bootstrap profile. Reduced MI is an explicit approximation.
# Set BOOTSTRAP_MICE_M/ITERATIONS to MICE_M/ITERATIONS for an exact numerical profile.
INTERNAL_BOOTSTRAP_N = 200
PAIRED_BOOTSTRAP_N = 200       # same redevelopment replicates; no second model-fitting loop
BOOTSTRAP_MICE_M = 5
BOOTSTRAP_MICE_ITERATIONS = 5
BOOTSTRAP_NONLINEARITY_MI_DATASETS = 5
RESELECT_FORMS_IN_BOOTSTRAP = True
BOOTSTRAP_N_JOBS = min(4, max(1, (os.cpu_count() or 1) // 2))
PHREG_THREAD_LIMIT = 1
BOOTSTRAP_WORKER_THREADS = 1
PRIMARY_FIXED_CI_BOOTSTRAP_N = 1000
IECV_CI_BOOTSTRAP_N = 500
SUBGROUP_BOOTSTRAP_N = 1000
SENSITIVITY_BOOTSTRAP_N = 1000
CI_MIN_VALID_FRACTION = 0.90

RUN_SUBGROUPS = True
RUN_LONGITUDINAL_SENSITIVITY = True
RUN_OTHER_SENSITIVITY = True
MIN_SUBGROUP_N = 50
MIN_SENSITIVITY_N = 100
MIN_SENSITIVITY_EVENTS = 20
RESUME_FROM_CHECKPOINT = True

# Explicit source assumptions.
# LM30_対象 and LM90_対象 define separate visit-window eligibility.
# Day30_外来受診日 is the first visit in days 15-45.
# Day90_外来受診日 is the first visit in days 75-105.
# d30_*/d90_* values are the latest laboratory values in the 14-day
# look-back window ending on the corresponding index visit.
LAB_TIMING_POLICY = "error"
KNOWN_FIRST_EVENT_EXTENDS_LASTDATE = True
IPCW_MIN_G = 1e-6
VERBOSE_FITS = True

FIXED_BINARY = ["male", "ref", "af", "dm", "dialysis", "sglt2",
                "beta_blocker", "ras_arni", "mra", "loop_diuretic"]
FIXED_CONTINUOUS = ["age", "bmi"]
ALWAYS_LINEAR_CONTINUOUS = ["visit_day"]
LAB_NAMES = ["albumin", "bun", "hemoglobin", "potassium", "sodium", "egfr", "ntprobnp_log2"]
CONTINUOUS_CONCEPTS = FIXED_CONTINUOUS + LAB_NAMES
SOURCE_LABS = {"albumin": "Alb", "bun": "BUN", "hemoglobin": "Hb", "potassium": "K",
               "sodium": "Na", "egfr": "eGFR", "ntprobnp": "NT-proBNP"}
LAB_LABELS = {"albumin": "Albumin", "bun": "Blood urea nitrogen", "hemoglobin": "Hemoglobin",
              "potassium": "Potassium", "sodium": "Sodium", "egfr": "eGFR",
              "ntprobnp_log2": "log2(NT-proBNP)"}
CENTER_DISPLAY = {"付属": "Nippon Medical School Hospital", "北総": "Chiba Hokusoh Hospital",
                  "小杉": "Musashi-Kosugi Hospital", "永山": "Tama Nagayama Hospital"}
METRICS = ["C-index", "AUC180", "Brier180", "IBS", "Calibration intercept", "Calibration slope"]
DELTA_METRICS = ["Delta C-index", "Delta AUC180", "Delta Brier180", "Delta IBS"]
SCHEMA_VERSION = "v9_tables_ci_schema1"
for p in (OUTPUT_DIR, CACHE_DIR, MODEL_DIR, ARTIFACT_DIR, PARAMETER_DIR):
    p.mkdir(parents=True, exist_ok=True)
print(f"{ANALYSIS_VERSION}\nInput: {PARQUET_PATH}\nOutput: {OUTPUT_DIR}")
print("time=0: first outpatient visit in days 15-45 / 75-105; cohorts are separately defined and may overlap.")
print(f"Primary MI={MICE_M} x {MICE_ITERATIONS}; bootstrap={INTERNAL_BOOTSTRAP_N}, "
      f"MI={BOOTSTRAP_MICE_M} x {BOOTSTRAP_MICE_ITERATIONS}; no Cox penalty / no CV.")

# %% [markdown]
# ## Data preparation, source audit and persistence

# %%
def log_message(msg: str) -> None:
    print(msg, flush=True)


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def _frame_digest(df):
    h = hashlib.sha256()
    h.update(json.dumps(list(df.columns), ensure_ascii=False).encode("utf-8"))
    h.update(pd.util.hash_pandas_object(df.reset_index(drop=True), index=True).to_numpy().tobytes())
    return h.hexdigest()


def _array_digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def _config_digest(config):
    return hashlib.sha256(json.dumps(_json_safe(config), sort_keys=True, default=str).encode()).hexdigest()


def _atomic_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _settings_signature():
    return {
        "version": ANALYSIS_VERSION,
        "schema": SCHEMA_VERSION,
        "outcome": OUTCOME_MODE,
        "horizon": HORIZON,
        "early_window": EARLY_WINDOW,
        "later_window": LATER_WINDOW,
        "lab_lookback_days": LAB_LOOKBACK_DAYS,
        "time_zero": "actual first outpatient visit within prespecified window",
        "visit_day_common_predictor": True,
        "MICE_M": MICE_M,
        "MICE_ITERATIONS": MICE_ITERATIONS,
        "PMM_DONORS": PMM_DONORS,
        "forms_m": NONLINEARITY_MI_DATASETS,
        "forms_alpha": NONLINEARITY_ALPHA,
        "ties": COX_TIES,
        "imputation": "baseline block excludes outpatient information and outcomes; visit_day common to both models",
        "longitudinal": RUN_LONGITUDINAL_SENSITIVITY,
        "seed": MASTER_SEED,
    }


def cached_stage(stage, landmark, raw, seed, compute, extra=None):
    signature = _config_digest({"settings": _settings_signature(), "stage": stage,
        "landmark": landmark, "raw": _frame_digest(raw), "seed": seed, "extra": extra})
    path = CACHE_DIR / f"{landmark}_{stage}_{signature[:16]}.pkl"
    if RESUME_FROM_CHECKPOINT and path.exists():
        with path.open("rb") as f:
            saved = pickle.load(f)
        if saved.get("signature") != signature:
            raise ValueError(f"Cache signature mismatch: {path}")
        log_message(f"[{landmark}] Reusing {stage}; no recalculation.")
        return saved["result"]
    with threadpool_limits(limits=PHREG_THREAD_LIMIT):
        result = compute()
    _atomic_pickle({"signature": signature, "result": result}, path)
    log_message(f"[{landmark}] Saved {stage}.")
    return result


def export_excel(tables, path):
    """User-environment export; numerical cells remain numerical."""
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.xlsx")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            if len(df.columns):
                ws.auto_filter.ref = ws.dimensions
            for row in ws:
                for c in row:
                    japanese = isinstance(c.value, str) and any(ord(x) > 255 for x in c.value)
                    c.font = Font(name="Meiryo" if japanese else "Times New Roman", size=10,
                                  bold=c.row == 1)
                    c.alignment = Alignment(vertical="top", wrap_text=True)
                    if c.row == 1:
                        c.fill = PatternFill("solid", fgColor="E8EDF2")
                        c.border = Border(bottom=Side(style="thin", color="8899AA"))
                    elif isinstance(c.value, float):
                        c.number_format = "0.0000"
            for col in ws.columns:
                width = max([len(str(c.value)) if c.value is not None else 0 for c in col[:100]] + [10])
                ws.column_dimensions[col[0].column_letter].width = min(42, max(12, width + 2))
            ws.row_dimensions[1].height = 32
    os.replace(tmp, path)
    return path


def parse_mixed_date_series(s):
    s = s.copy()
    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="coerce")
    else:
        numeric = pd.to_numeric(s, errors="coerce")
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        ymd = numeric.between(19000101, 21001231) & (numeric == np.floor(numeric))
        out.loc[ymd] = pd.to_datetime(numeric.loc[ymd].astype("int64").astype(str),
                                    format="%Y%m%d", errors="coerce")
        excel = numeric.between(20000, 80000) & ~ymd
        out.loc[excel] = pd.to_datetime(numeric.loc[excel], unit="D", origin="1899-12-30", errors="coerce")
        text = s.notna() & ~ymd & ~excel & numeric.isna()
        if text.any():
            out.loc[text] = pd.to_datetime(s.loc[text].astype(str).str.strip(), format="mixed", errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()




def prepare_source_df(source):
    d = source.copy().reset_index(drop=True)

    required = [
        "INDEX", "施設名", "データ識別番号",
        "入院日", "退院日", "死亡日", "再入院日", "LastDate",
        "Age", "Male", "BMI", "LVEF", "af", "DM",
        "維持透析", "PD", "SGLT2", "β遮断薬", "RAS_ARNI",
        "MRA_内服", "Loop_内服",
        "LM30_対象", "LM90_対象",
        "Day30_外来受診日", "Day90_外来受診日",
        "Day30_外来_退院後日数", "Day90_外来_退院後日数",
    ]

    for day in (30, 90):
        for lab in SOURCE_LABS.values():
            required += [f"d{day}_{lab}", f"d{day}_{lab}_date"]

    required += [f"dis_{lab}" for lab in SOURCE_LABS.values()]

    missing = sorted(set(required) - set(d.columns))
    if missing:
        raise KeyError(f"Missing source columns: {missing}")

    if d.columns.duplicated().any():
        raise ValueError("Source contains duplicated column names.")

    if d["INDEX"].isna().any() or d["INDEX"].duplicated().any():
        raise ValueError("Source must contain one nonmissing row per INDEX.")

    if d[["施設名", "データ識別番号"]].isna().any().any():
        raise ValueError("Patient and center identifiers must not be missing.")

    date_cols = [
        "入院日", "退院日", "死亡日", "再入院日", "LastDate",
        "Day30_外来受診日", "Day90_外来受診日",
    ]
    date_cols += [
        f"d{day}_{lab}_date"
        for day in (30, 90)
        for lab in SOURCE_LABS.values()
    ]

    for c in date_cols:
        parsed = parse_mixed_date_series(d[c])
        supplied = d[c].notna() & d[c].astype(str).str.strip().ne("")
        bad = supplied & parsed.isna()
        if bad.any():
            raise ValueError(
                f"{c}: unparseable nonmissing dates, "
                f"examples={d.loc[bad, c].head().tolist()}"
            )
        d[c] = parsed

    if d["退院日"].isna().any():
        raise ValueError("退院日 contains missing values.")

    binary = [
        "Male", "af", "DM", "維持透析", "PD",
        "SGLT2", "β遮断薬", "RAS_ARNI",
        "MRA_内服", "Loop_内服",
        "LM30_対象", "LM90_対象",
    ]

    numeric_cols = (
        binary
        + ["Age", "BMI", "LVEF",
           "Day30_外来_退院後日数",
           "Day90_外来_退院後日数"]
        + [
            f"{p}_{lab}"
            for p in ("dis", "d30", "d90")
            for lab in SOURCE_LABS.values()
        ]
    )

    for c in numeric_cols:
        v = pd.to_numeric(d[c], errors="coerce")
        bad = (
            d[c].notna()
            & d[c].astype(str).str.strip().ne("")
            & ~np.isfinite(v)
        )
        if bad.any():
            raise ValueError(
                f"{c}: nonnumeric/nonfinite values, "
                f"examples={d.loc[bad, c].head().tolist()}"
            )
        d[c] = v

    for c in binary:
        if (~d[c].dropna().isin([0, 1])).any():
            raise ValueError(f"{c} must be 0/1 or missing.")

    if d[["LM30_対象", "LM90_対象"]].isna().any().any():
        raise ValueError("LM eligibility flags must not be missing.")

    for day, (low, high) in {
        30: EARLY_WINDOW,
        90: LATER_WINDOW,
    }.items():

        flag = d[f"LM{day}_対象"].eq(1)
        visit_day = d[f"Day{day}_外来_退院後日数"]
        visit_date = d[f"Day{day}_外来受診日"]

        invalid = (
            flag
            & (
                visit_day.isna()
                | visit_date.isna()
                | ~visit_day.between(low, high, inclusive="both")
                | visit_day.ne(np.floor(visit_day))
            )
        )

        if invalid.any():
            raise ValueError(
                f"LM{day}_対象=1 but the selected outpatient visit is "
                f"outside days {low}-{high} or missing for "
                f"{int(invalid.sum())} rows."
            )

        reconstructed = (
            d["退院日"]
            + pd.to_timedelta(visit_day, unit="D")
        )

        mismatch = (
            flag
            & reconstructed.notna()
            & visit_date.notna()
            & reconstructed.ne(visit_date)
        )

        if mismatch.any():
            raise ValueError(
                f"Day{day}_外来受診日 does not match 退院日 + "
                f"Day{day}_外来_退院後日数 for "
                f"{int(mismatch.sum())} rows."
            )

    d["__loop_diuretic"] = d["Loop_内服"]
    d["__ref"] = (
        (d["LVEF"] < 50)
        .astype(float)
        .where(d["LVEF"].notna())
    )

    dial = d[["維持透析", "PD"]]
    d["__dialysis"] = np.where(
        dial.eq(1).any(axis=1),
        1.0,
        np.where(dial.eq(0).all(axis=1), 0.0, np.nan),
    )

    d["__patient_key"] = (
        d["施設名"].astype(str)
        + "|"
        + d["データ識別番号"].astype(str)
    )

    return d


def load_source_df():
    if not PARQUET_PATH.is_file():
        raise FileNotFoundError(PARQUET_PATH)

    source = pd.read_parquet(PARQUET_PATH)

    log_message(
        f"Loaded parquet: {PARQUET_PATH}\n"
        f"Original N={len(source):,}; columns={source.shape[1]}"
    )

    return prepare_source_df(source)


def create_source_landmark_survival_cohort(data, landmark_day):
    if landmark_day == 30:
        low, high = EARLY_WINDOW
        label = "Early"
    elif landmark_day == 90:
        low, high = LATER_WINDOW
        label = "Later"
    else:
        raise ValueError("landmark_day must be 30 or 90.")

    flag_col = f"LM{landmark_day}_対象"
    visit_day_col = f"Day{landmark_day}_外来_退院後日数"
    visit_date_col = f"Day{landmark_day}_外来受診日"

    d = data.loc[data[flag_col].eq(1)].copy()

    flow = {
        "Landmark": f"Day{landmark_day}",
        "Window": f"Day {low}-{high}",
        "Source N": int(len(data)),
        "Eligible outpatient flags N": int(len(d)),
    }

    valid_visit = (
        d[visit_day_col].between(low, high, inclusive="both")
        & d[visit_day_col].eq(np.floor(d[visit_day_col]))
        & d[visit_date_col].notna()
    )

    flow["Invalid/missing index visit excluded"] = int((~valid_visit).sum())
    d = d.loc[valid_visit].copy()

    d["index_outpatient_date"] = d[visit_date_col]
    d["prediction_date"] = d["index_outpatient_date"]
    d["visit_day"] = d[visit_day_col].astype(float)

    pre_r = (
        d["再入院日"].notna()
        & (d["再入院日"] <= d["prediction_date"])
    )
    pre_d = (
        d["死亡日"].notna()
        & (d["死亡日"] <= d["prediction_date"])
    )

    flow["Pre-index readmission excluded"] = int(pre_r.sum())
    flow["Pre-index death excluded"] = int(pre_d.sum())
    flow["Pre-index either excluded"] = int((pre_r | pre_d).sum())

    d = d.loc[~(pre_r | pre_d)].copy()

    first = d[["再入院日", "死亡日"]].min(axis=1)
    admin = d["prediction_date"] + pd.Timedelta(days=HORIZON)
    follow = d["LastDate"].copy()

    after_last = first.notna() & (
        follow.isna() | (first > follow)
    )

    flow[
        "Known first event after LastDate or LastDate missing"
    ] = int(after_last.sum())

    if KNOWN_FIRST_EVENT_EXTENDS_LASTDATE:
        follow = pd.concat([follow, first], axis=1).max(axis=1)

    valid_follow = (
        follow.notna()
        & (follow > d["prediction_date"])
    )

    flow["No positive post-index follow-up excluded"] = int(
        (~valid_follow).sum()
    )

    d = d.loc[valid_follow].copy()

    first = first.loc[d.index]
    follow = follow.loc[d.index]
    admin = admin.loc[d.index]

    end = pd.concat([follow, admin], axis=1).min(axis=1)

    event = first.notna() & first.le(end)

    d["event"] = event.astype(int)
    d["event_date"] = first.where(event)
    d["followup_end"] = end.where(~event, first)

    d["time"] = (
        d["followup_end"] - d["prediction_date"]
    ).dt.days.astype(float)

    if (d["time"] <= 0).any() or (d["time"] > HORIZON).any():
        raise ValueError("Invalid analysis follow-up time.")

    rfirst = event & d["再入院日"].eq(first)
    dfirst = event & d["死亡日"].eq(first)

    d["event_component"] = np.select(
        [rfirst & dfirst, rfirst, dfirst],
        [
            "Readmission and death on same day",
            "Readmission",
            "Death",
        ],
        default="None",
    )

    flow.update({
        "Final N": int(len(d)),
        "Composite events": int(event.sum()),
        "Readmission-first events": int((rfirst & ~dfirst).sum()),
        "Death-first events": int((dfirst & ~rfirst).sum()),
        "Same-day readmission/death events": int((rfirst & dfirst).sum()),
        "Censored before 180 days": int((~event & (d["time"] < HORIZON)).sum()),
        "Event-free through 180 days": int((~event & (d["time"] == HORIZON)).sum()),
    })

    expected_n = (
        flow["Eligible outpatient flags N"]
        - flow["Invalid/missing index visit excluded"]
        - flow["Pre-index either excluded"]
        - flow["No positive post-index follow-up excluded"]
    )

    if expected_n != len(d):
        raise RuntimeError(f"{label}: cohort-flow count mismatch.")

    log_message(
        f"{label} Day{low}-{high}: "
        f"time=0 at first outpatient visit; "
        f"N={len(d):,}; composite events={int(event.sum()):,}"
    )

    return d.reset_index(drop=True), flow


def build_source_internal_for_s1(source):
    out = pd.DataFrame(index=source.index)
    out["id"] = source["INDEX"].astype(str)

    mapping = {
        "age": "Age",
        "male": "Male",
        "bmi": "BMI",
        "ref": "__ref",
        "af": "af",
        "dm": "DM",
        "dialysis": "__dialysis",
        "sglt2": "SGLT2",
        "beta_blocker": "β遮断薬",
        "ras_arni": "RAS_ARNI",
        "mra": "MRA_内服",
        "loop_diuretic": "__loop_diuretic",
    }

    for new, old in mapping.items():
        out[new] = source[old]

    for lab, source_lab in SOURCE_LABS.items():
        out[f"discharge_{lab}"] = source[f"dis_{source_lab}"]

    out["discharge_ntprobnp_log2"] = np.log2(
        out["discharge_ntprobnp"].where(
            out["discharge_ntprobnp"] > 0
        )
    )

    return out


def prepare_internal_landmark(cohort, landmark_day):
    out = build_source_internal_for_s1(cohort)

    out["center"] = cohort["施設名"].astype(str)
    out["patient_key"] = cohort["__patient_key"].astype(str)

    for c in [
        "time",
        "event",
        "prediction_date",
        "index_outpatient_date",
        "event_component",
    ]:
        out[c] = cohort[c]

    out["visit_day"] = cohort["visit_day"].astype(float)

    audit = []
    bad_rows = []

    for lab, source_lab in SOURCE_LABS.items():

        value_col = f"d{landmark_day}_{source_lab}"
        date_col = f"d{landmark_day}_{source_lab}_date"

        value = cohort[value_col].copy()
        dt = cohort[date_col]

        measurement_day = (
            dt - cohort["退院日"]
        ).dt.days.astype(float)

        window_start = (
            cohort["prediction_date"]
            - pd.Timedelta(days=LAB_LOOKBACK_DAYS)
        )

        timing = (
            dt.notna()
            & dt.ge(cohort["退院日"])
            & dt.ge(window_start)
            & dt.le(cohort["prediction_date"])
        )

        supplied = value.notna()
        invalid = supplied & ~timing

        age_days = (
            cohort["prediction_date"] - dt
        ).dt.days.astype(float)

        audit.append({
            "Landmark": f"Day{landmark_day}",
            "Laboratory": source_lab,
            "N": int(len(cohort)),
            "Source nonmissing N": int(supplied.sum()),
            "Date missing among nonmissing values": int(
                (supplied & dt.isna()).sum()
            ),
            "Before discharge": int(
                (supplied & dt.notna() & dt.lt(cohort["退院日"])).sum()
            ),
            "Earlier than look-back window": int(
                (supplied & dt.notna() & dt.lt(window_start)).sum()
            ),
            "After index outpatient visit": int(
                (supplied & dt.notna() & dt.gt(cohort["prediction_date"])).sum()
            ),
            "Invalid timing N": int(invalid.sum()),
            "Retained at prediction time": int((supplied & timing).sum()),
            "Median laboratory age at index visit, days": (
                float(age_days.loc[supplied & timing].median())
                if (supplied & timing).any()
                else np.nan
            ),
        })

        if invalid.any():
            examples = cohort.loc[
                invalid,
                [
                    "INDEX",
                    "退院日",
                    "index_outpatient_date",
                    value_col,
                    date_col,
                ],
            ].copy()
            examples["Laboratory"] = source_lab
            bad_rows.append(examples)

        out[f"current_{lab}"] = value.where(timing)
        out[f"current_day_{lab}"] = measurement_day.where(timing)

    if bad_rows:
        export_excel(
            {
                "Timing audit": pd.DataFrame(audit),
                "Invalid rows": pd.concat(
                    bad_rows,
                    ignore_index=True,
                ),
            },
            ARTIFACT_DIR
            / f"Day{landmark_day}_invalid_lab_timing.xlsx",
        )

        raise ValueError(
            f"Day{landmark_day}: source outpatient laboratory dates "
            "do not match the approved 14-day look-back window ending "
            "on the index outpatient visit. "
            "See invalid_lab_timing.xlsx. "
            "No source values were overwritten and fitting did not start."
        )

    out["current_ntprobnp_log2"] = np.log2(
        out["current_ntprobnp"].where(
            out["current_ntprobnp"] > 0
        )
    )

    return out.reset_index(drop=True), pd.DataFrame(audit)


def load_landmark_inputs():
    global SOURCE_DF
    global SOURCE_INTERNAL
    global FLOW_TABLE
    global LAB_TIMING_AUDIT
    global COHORT_AUDIT

    SOURCE_DF = load_source_df()
    SOURCE_INTERNAL = build_source_internal_for_s1(SOURCE_DF)

    raw30, f30 = create_source_landmark_survival_cohort(
        SOURCE_DF, 30
    )
    raw90, f90 = create_source_landmark_survival_cohort(
        SOURCE_DF, 90
    )

    d30, a30 = prepare_internal_landmark(raw30, 30)
    d90, a90 = prepare_internal_landmark(raw90, 90)

    if d30.empty or d90.empty:
        raise ValueError("An analysis cohort is empty.")

    FLOW_TABLE = pd.DataFrame([f30, f90])

    LAB_TIMING_AUDIT = pd.concat(
        [a30, a90],
        ignore_index=True,
    )

    COHORT_AUDIT = pd.concat(
        [
            d30.assign(Landmark="Day30"),
            d90.assign(Landmark="Day90"),
        ],
        ignore_index=True,
    )

    return d30, d90


# %% [markdown]
# ## PMM chained equations: baseline and current blocks

# %%
def safe_sd(x):
    sd = float(np.nanstd(x))
    return sd if np.isfinite(sd) and sd > 1e-12 else 1.0


def get_imputation_columns(run_longitudinal=True):
    continuous = FIXED_CONTINUOUS + ALWAYS_LINEAR_CONTINUOUS + [f"{p}_{lab}" for p in ("discharge", "current") for lab in LAB_NAMES]
    if run_longitudinal:
        continuous += [f"current_day_{lab}" for lab in SOURCE_LABS]
    return continuous, FIXED_BINARY.copy()


def _center_array(df, levels):
    return np.column_stack([df["center"].astype(str).eq(x).to_numpy(dtype=float) for x in levels])


def _pmm_draws(observed_y, observed_pred, missing_pred, rng, k):
    k = min(max(int(k), 1), len(observed_y))
    if k == 0:
        raise ValueError("PMM has no training donors.")
    # Exact nearest-k donors. Chunking limits temporary memory.
    out = np.empty(len(missing_pred), dtype=float)
    for i, p in enumerate(missing_pred):
        ids = np.argpartition(np.abs(observed_pred - p), k - 1)[:k]
        out[i] = observed_y[rng.choice(ids)]
    return out


def _clip_imputed_days(df, landmark):
    """Keep imputed measurement days within the patient's 14-day look-back window."""
    visit_day = df["visit_day"].to_numpy(dtype=float)
    lower = np.maximum(0.0, visit_day - LAB_LOOKBACK_DAYS)
    upper = visit_day

    for lab in SOURCE_LABS:
        c = f"current_day_{lab}"
        if c in df:
            x = df[c].to_numpy(dtype=float)
            df[c] = np.minimum(
                np.maximum(x, lower),
                upper,
            )


    for lab in SOURCE_LABS:
        c = f"current_day_{lab}"
        if c in df:
            x = df[c].to_numpy(dtype=float)
            df[c] = np.minimum(
                np.maximum(
                    x,
                    lower,
                ),
                upper,
            )



def _initial_fill(train, valid, continuous_cols, binary_cols):
    tr, va = train.copy(), valid.copy() if valid is not None else None
    fill = {}
    for c in list(continuous_cols) + list(binary_cols):
        vals = pd.to_numeric(tr[c], errors="coerce")
        if not np.isfinite(vals.dropna()).all() or vals.notna().sum() == 0:
            raise ValueError(f"{c}: no finite training observations. Check the source timing audit; do not invent donors.")
        fill[c] = float(vals.median() if c in continuous_cols else vals.mode().iloc[0])
        tr[c] = vals.fillna(fill[c])
        if va is not None:
            va[c] = pd.to_numeric(va[c], errors="coerce").fillna(fill[c])
    return tr, va, fill


def _fit_imputation_equation(X, y, continuous, rng):
    scaler = StandardScaler().fit(X)
    xs = scaler.transform(X)
    if continuous:
        model = BayesianRidge().fit(xs, y)
        cov = (model.sigma_ + model.sigma_.T) / 2
        eig, vectors = np.linalg.eigh(cov)
        draw = model.coef_ + vectors @ (np.sqrt(np.maximum(eig, 0)) * rng.normal(size=len(eig)))
        return {"kind": "pmm", "scaler": scaler, "model": model,
                "draw_coef": draw, "observed_y": y,
                "observed_pred": model.predict(xs)}
    if np.unique(y).size == 1:
        return {"kind": "constant", "value": float(y[0]), "scaler": scaler}
    model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs").fit(xs, y.astype(int))
    # Bernoulli imputation with coefficient uncertainty from the fitted information matrix.
    xx = np.column_stack([np.ones(len(y)), xs])
    p = model.predict_proba(xs)[:, 1]
    info = (xx.T * np.maximum(p * (1-p), 1e-8)) @ xx + np.eye(xx.shape[1]) * 1e-6
    cov = np.linalg.pinv(info)
    eig, vectors = np.linalg.eigh((cov + cov.T) / 2)
    beta = np.r_[model.intercept_, model.coef_.ravel()]
    draw = beta + vectors @ (np.sqrt(np.maximum(eig, 0)) * rng.normal(size=len(eig)))
    return {"kind": "binary", "scaler": scaler, "draw_coef": draw}


def _apply_imputation_equation(equation, X, rng, donors):
    if equation["kind"] == "constant":
        return np.repeat(equation["value"], len(X))
    x = equation["scaler"].transform(X)
    if equation["kind"] == "pmm":
        pred = x @ equation["draw_coef"] + equation["model"].intercept_
        return _pmm_draws(equation["observed_y"], equation["observed_pred"], pred, rng, donors)
    z = equation["draw_coef"][0] + x @ equation["draw_coef"][1:]
    p = 1 / (1 + np.exp(-np.clip(z, -35, 35)))
    return rng.binomial(1, p)


def mice_impute_train_valid(train_raw, valid_raw, center_levels, m, n_iter, seed,
                           landmark, k_pmm=5, return_states=False):
    """Train-only PMM chained equations; baseline block cannot see outpatient values.

    Outcomes, time, patient IDs and the held-out center never enter an imputation equation.
    Baseline/discharge variables are completed first, then frozen while current values are imputed.
    Both Cox models use the SAME completed baseline block.
    """
    continuous, binary = get_imputation_columns(RUN_LONGITUDINAL_SENSITIVITY)
    baseline = FIXED_CONTINUOUS + ALWAYS_LINEAR_CONTINUOUS + [f"discharge_{lab}" for lab in LAB_NAMES] + binary
    current = [f"current_{lab}" for lab in LAB_NAMES]
    if RUN_LONGITUDINAL_SENSITIVITY:
        current += [f"current_day_{lab}" for lab in SOURCE_LABS]
    targets = baseline + current
    tr_raw = train_raw.reset_index(drop=True)
    va_raw = valid_raw.reset_index(drop=True) if valid_raw is not None else None
    tr_missing = {c: tr_raw[c].isna().to_numpy() for c in targets}
    va_missing = {c: va_raw[c].isna().to_numpy() for c in targets} if va_raw is not None else None
    ctr = _center_array(tr_raw, center_levels)
    cva = _center_array(va_raw, center_levels) if va_raw is not None else None
    completed, validation, states, trace = [], ([] if va_raw is not None else None), [], []
    for j in range(int(m)):
        rng = np.random.default_rng(int(seed) + 10007 * (j+1))
        tr, va, fill = _initial_fill(tr_raw, va_raw, continuous, binary)
        frozen_steps = []
        for block_name, block in [("baseline", baseline), ("current", current)]:
            for cycle in range(int(n_iter)):
                for target in block:
                    missing_any = tr_missing[target].any() or (va is not None and va_missing[target].any())
                    save_eq = return_states and cycle == n_iter - 1
                    if not missing_any and not save_eq:
                        continue
                    predictors = [c for c in (baseline if block_name == "baseline" else targets) if c != target]
                    xtr = np.column_stack([tr[predictors].to_numpy(dtype=float), ctr])
                    obs = ~tr_missing[target]
                    equation = _fit_imputation_equation(xtr[obs], tr_raw.loc[obs, target].to_numpy(dtype=float),
                                                         target in continuous, rng)
                    equation.update({"target": target, "predictors": predictors, "block": block_name})
                    if tr_missing[target].any():
                        tr.loc[tr_missing[target], target] = _apply_imputation_equation(
                            equation, xtr[tr_missing[target]], rng, k_pmm)
                    if va is not None and va_missing[target].any():
                        xva = np.column_stack([va[predictors].to_numpy(dtype=float), cva])
                        va.loc[va_missing[target], target] = _apply_imputation_equation(
                            equation, xva[va_missing[target]], rng, k_pmm)
                    if save_eq:
                        frozen_steps.append(equation)
                if block_name == "current":
                    _clip_imputed_days(tr, landmark)
                    if va is not None:
                        _clip_imputed_days(va, landmark)
                if return_states:
                    for target in block:
                        mask = tr_missing[target]
                        if mask.any():
                            v = tr.loc[mask, target].to_numpy(dtype=float)
                            trace.append({"MI": j+1, "Cycle": cycle+1, "Block": block_name,
                                          "Variable": target, "Imputed mean": v.mean(), "Imputed SD": v.std()})
        completed.append(tr)
        if validation is not None:
            validation.append(va)
        if return_states:
            states.append({"fill": fill, "steps": frozen_steps, "center_levels": list(center_levels),
                           "n_iter": int(n_iter), "donors": k_pmm, "landmark": landmark})
        if VERBOSE_FITS and ((j+1) % 5 == 0 or j+1 == m):
            log_message(f"  MICE {j+1}/{m} completed (baseline block excludes outpatient variables).")
    if return_states:
        return completed, validation, states, pd.DataFrame(trace)
    return completed, validation


def transform_saved_mice(raw, states, seed=MASTER_SEED):
    """Apply frozen final-cycle imputation equations; no fitting and no outcome use."""
    out = []
    for j, state in enumerate(states):
        rng = np.random.default_rng(seed + 10007*(j+1))
        d = raw.copy().reset_index(drop=True)
        missing = {c: d[c].isna().to_numpy() for c in state["fill"]}
        for c, value in state["fill"].items():
            d[c] = d[c].fillna(value)
        center = _center_array(d, state["center_levels"])
        for block in ("baseline", "current"):
            for _ in range(state["n_iter"]):
                for step in state["steps"]:
                    c = step["target"]
                    if step["block"] != block or not missing[c].any():
                        continue
                    x = np.column_stack([d[step["predictors"]].to_numpy(dtype=float), center])
                    d.loc[missing[c], c] = _apply_imputation_equation(step, x[missing[c]], rng, state["donors"])
                if block == "current":
                    _clip_imputed_days(d, state["landmark"])
        out.append(d)
    return out


def single_imputation(train_raw, valid_raw=None):
    continuous, binary = get_imputation_columns(RUN_LONGITUDINAL_SENSITIVITY)
    tr, va, _ = _initial_fill(train_raw, valid_raw, continuous, binary)
    return tr, va


# %% [markdown]
# ## Standard Cox and functional-form assessment

# %%
def rcs_nonlinear_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)

    if len(k) < 3:
        return np.empty((len(x), 0), dtype=float)

    if np.any(np.diff(k) <= 0):
        return np.empty((len(x), 0), dtype=float)

    denom = (k[-1] - k[0]) ** 2
    cols = []

    for j in range(len(k) - 2):
        kj = k[j]
        term = np.maximum(x - kj, 0.0) ** 3
        term -= (
            (k[-1] - kj) / (k[-1] - k[-2])
        ) * np.maximum(x - k[-2], 0.0) ** 3
        term += (
            (k[-2] - kj) / (k[-1] - k[-2])
        ) * np.maximum(x - k[-1], 0.0) ** 3
        cols.append(term / denom)

    return np.column_stack(cols)


def conceptual_column(concept: str, representation: str) -> str:
    if concept in {"age", "bmi"}:
        return concept
    return f"{representation}_{concept}"


def compute_knots(
    imputed_train: Sequence[pd.DataFrame],
    concept: str,
) -> Optional[List[float]]:
    # Use the first imputed dataset to avoid multiplying each patient M times.
    d = imputed_train[0]

    if concept in {"age", "bmi"}:
        values = d[concept].to_numpy(dtype=float)
    else:
        values = np.concatenate(
            [
                d[f"discharge_{concept}"].to_numpy(dtype=float),
                d[f"current_{concept}"].to_numpy(dtype=float),
            ]
        )

    values = values[np.isfinite(values)]
    if len(values) < 10:
        return None

    q = np.quantile(values, [0.10, 0.50, 0.90]).astype(float)

    if np.any(np.diff(q) <= 1e-12):
        return None

    return q.tolist()


@dataclass
class DesignSpec:
    representation: str
    forms: Dict[str, str]
    knots: Dict[str, Optional[List[float]]]
    feature_names: List[str]
    continuous_feature_names: List[str]
    means: Dict[str, float]
    sds: Dict[str, float]
    extra_continuous: List[str]


def _unscaled_design(
    df: pd.DataFrame,
    representation: str,
    forms: Dict[str, str],
    knots: Dict[str, Optional[List[float]]],
    extra_continuous: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    data = {}

    for concept in CONTINUOUS_CONCEPTS:
        c = conceptual_column(concept, representation)
        x = df[c].to_numpy(dtype=float)
        data[concept] = x

        if forms.get(concept, "linear") == "spline":
            k = knots.get(concept)
            if k is not None:
                z = rcs_nonlinear_basis(x, k)
                for j in range(z.shape[1]):
                    data[f"{concept}_rcs{j + 1}"] = z[:, j]

    for c in ALWAYS_LINEAR_CONTINUOUS:
        data[c] = df[c].to_numpy(dtype=float)

    for c in ALWAYS_LINEAR_CONTINUOUS:
        data[c] = df[c].to_numpy(dtype=float)

    for c in FIXED_BINARY:
        data[c] = df[c].to_numpy(dtype=float)

    for c in list(extra_continuous or []):
        data[c] = df[c].to_numpy(dtype=float)

    return pd.DataFrame(data, index=df.index)


def fit_design_spec(
    df: pd.DataFrame,
    representation: str,
    forms: Dict[str, str],
    knots: Dict[str, Optional[List[float]]],
    extra_continuous: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, DesignSpec]:
    Xdf = _unscaled_design(
        df,
        representation,
        forms,
        knots,
        extra_continuous,
    )

    continuous_names = [
        c for c in Xdf.columns
        if c not in FIXED_BINARY
    ]

    means = {}
    sds = {}

    for c in continuous_names:
        mu = float(Xdf[c].mean())
        sd = safe_sd(Xdf[c].to_numpy(dtype=float))
        means[c] = mu
        sds[c] = sd
        Xdf[c] = (Xdf[c] - mu) / sd

    spec = DesignSpec(
        representation=representation,
        forms=forms.copy(),
        knots={k: (v.copy() if isinstance(v, list) else v) for k, v in knots.items()},
        feature_names=Xdf.columns.tolist(),
        continuous_feature_names=continuous_names,
        means=means,
        sds=sds,
        extra_continuous=list(extra_continuous or []),
    )

    return Xdf.to_numpy(dtype=float), spec


def apply_design_spec(
    df: pd.DataFrame,
    spec: DesignSpec,
) -> np.ndarray:
    Xdf = _unscaled_design(
        df,
        spec.representation,
        spec.forms,
        spec.knots,
        spec.extra_continuous,
    )

    missing = [c for c in spec.feature_names if c not in Xdf.columns]
    if missing:
        raise KeyError(f"Design columns missing: {missing}")

    Xdf = Xdf[spec.feature_names].copy()

    for c in spec.continuous_feature_names:
        Xdf[c] = (Xdf[c] - spec.means[c]) / spec.sds[c]

    return Xdf.to_numpy(dtype=float)


def _estimable_column_indices(X: np.ndarray) -> np.ndarray:
    """Return a full-rank subset of non-constant design columns.

    Exact zero-variance or exact linear-dependence can occur in bootstrap or
    center-specific training samples. Such columns are non-estimable in that
    training sample and are assigned coefficient 0 in the saved full-length
    coefficient vector. This is not variable selection; it is numerical handling
    of non-identifiable columns in a particular resample.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")

    finite = np.all(np.isfinite(X), axis=0)
    if not finite.all():
        raise ValueError("Non-finite values in Cox design matrix.")
    sd = np.nanstd(X, axis=0, ddof=0)
    candidates = np.where(finite & (sd > 1e-12))[0]
    if len(candidates) == 0:
        raise RuntimeError("No estimable Cox predictors remain after removing constant columns.")

    Xc = X[:, candidates] - X[:, candidates].mean(axis=0)
    _, R, piv = qr(Xc, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    if len(diag) == 0:
        raise RuntimeError("Unable to determine Cox design-matrix rank.")
    tol = max(Xc.shape) * np.finfo(float).eps * max(float(diag.max()), 1.0)
    rank = int(np.sum(diag > tol))
    if rank < 1:
        raise RuntimeError("Cox design matrix has rank 0.")

    keep = candidates[np.asarray(piv[:rank], dtype=int)]
    return np.sort(keep)


def fit_standard_cox_phreg(time, event, X):
    time, event, X = np.asarray(time, float), np.asarray(event, int), np.asarray(X, float)
    if len(time) != X.shape[0] or len(event) != len(time) or (time <= 0).any():
        raise ValueError("Invalid Cox outcome/design dimensions or nonpositive times.")
    if not event.sum():
        raise RuntimeError("No events in training data.")
    keep = _estimable_column_indices(X)
    model = PHReg(time, X[:, keep], status=event, ties=COX_TIES)
    last_error = None
    for method in ("newton", "bfgs"):
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                fitted = model.fit(method=method, maxiter=300, disp=0)
            beta = np.asarray(fitted.params, float)
            cov = np.asarray(fitted.cov_params(), float)
            score = np.asarray(model.score(beta), float)
            if (not np.isfinite(beta).all() or not np.isfinite(cov).all()
                    or not np.isfinite(score).all() or (np.diag(cov) <= 0).any()
                    or np.max(np.abs(score)) > max(0.01, 1e-5*event.sum())):
                raise RuntimeError("Unpenalized Cox did not converge to finite coefficients/covariance.")
            break
        except Exception as exc:
            last_error = exc
    else:
        raise RuntimeError(f"Unpenalized Cox fitting failed; no penalized fallback used: {last_error}")
    p = X.shape[1]
    beta_full = np.zeros(p)
    cov_full = np.full((p, p), np.nan)
    estimable = np.zeros(p, bool)
    beta_full[keep] = beta
    cov_full[np.ix_(keep, keep)] = cov
    estimable[keep] = True
    return beta_full, cov_full, estimable



def breslow_baseline(
    time: np.ndarray,
    event: np.ndarray,
    lp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.exp(np.clip(lp, -30, 30))

    event_times = np.sort(np.unique(time[event == 1]))
    increments = []
    for t in event_times:
        d = np.sum((time == t) & (event == 1))
        risk_sum = np.sum(risk[time >= t])
        inc = d / risk_sum if risk_sum > 0 else 0.0
        increments.append(inc)
    return event_times, np.cumsum(np.asarray(increments, dtype=float))


def cumulative_hazard_at(
    query_times: np.ndarray,
    event_times: np.ndarray,
    cumhaz: np.ndarray,
) -> np.ndarray:
    q = np.asarray(query_times, dtype=float)
    idx = np.searchsorted(event_times, q, side="right") - 1
    out = np.zeros(len(q), dtype=float)
    mask = idx >= 0
    out[mask] = cumhaz[idx[mask]]
    return out


@dataclass
class SingleImputationCoxModel:
    coefficients: np.ndarray
    covariance: np.ndarray
    estimable_mask: np.ndarray
    design_spec: DesignSpec
    baseline_event_times: np.ndarray
    baseline_cumhaz: np.ndarray


@dataclass
class MICoxBundle:
    representation: str
    models: List[SingleImputationCoxModel]
    extra_continuous: List[str]


def fit_mi_cox_bundle(
    imputed_train: Sequence[pd.DataFrame],
    raw_train: pd.DataFrame,
    representation: str,
    forms: Dict[str, str],
    knots: Dict[str, Optional[List[float]]],
    seed: int,
    extra_continuous: Optional[Sequence[str]] = None,
) -> MICoxBundle:
    """Fit one conventional Cox PH model per imputed dataset.

    `seed` is retained in the signature for compatibility with the supplied
    pipeline but is not used by deterministic PHReg fitting.
    """
    del seed
    models = []
    time = raw_train["time"].to_numpy(dtype=float)
    event = raw_train["event"].to_numpy(dtype=int)

    total = len(imputed_train)
    for m_idx, d in enumerate(imputed_train):
        X, spec = fit_design_spec(
            d,
            representation,
            forms,
            knots,
            extra_continuous=extra_continuous,
        )

        beta, cov, estimable = fit_standard_cox_phreg(time, event, X)
        lp = X @ beta
        bt, bh = breslow_baseline(time, event, lp)

        models.append(
            SingleImputationCoxModel(
                coefficients=beta,
                covariance=cov,
                estimable_mask=estimable,
                design_spec=spec,
                baseline_event_times=bt,
                baseline_cumhaz=bh,
            )
        )

        if VERBOSE_FITS and ((m_idx + 1) % 5 == 0 or (m_idx + 1) == total):
            dropped = int((~estimable).sum())
            log_message(
                f"  {representation}: Cox MI {m_idx + 1}/{total} completed"
                + (f"; non-estimable columns={dropped}" if dropped else "")
            )

    return MICoxBundle(
        representation=representation,
        models=models,
        extra_continuous=list(extra_continuous or []),
    )


def predict_mi_bundle(
    bundle: MICoxBundle,
    imputed_data: Sequence[pd.DataFrame],
    eval_times: np.ndarray = EVAL_TIMES,
) -> Dict[str, np.ndarray]:
    if len(bundle.models) != len(imputed_data):
        raise ValueError("Number of fitted MI models and imputed datasets must match.")

    lp_list = []
    surv_list = []

    for model, d in zip(bundle.models, imputed_data):
        X = apply_design_spec(d, model.design_spec)
        lp = X @ model.coefficients

        H0 = cumulative_hazard_at(
            eval_times,
            model.baseline_event_times,
            model.baseline_cumhaz,
        )
        surv = np.exp(
            -np.exp(np.clip(lp, -30, 30))[:, None]
            * H0[None, :]
        )
        lp_list.append(lp)
        surv_list.append(surv)

    return {
        "lp": np.mean(np.stack(lp_list, axis=0), axis=0),
        "survival": np.mean(np.stack(surv_list, axis=0), axis=0),
    }

def nonlinear_test_for_representation(imputed_train, time, event, representation, concept, knots):
    betas, variances = [], []
    for d in list(imputed_train)[:NONLINEARITY_MI_DATASETS]:
        forms = {c: "linear" for c in CONTINUOUS_CONCEPTS}
        forms[concept] = "spline"
        kd = {c: None for c in CONTINUOUS_CONCEPTS}
        kd[concept] = knots
        X, spec = fit_design_spec(d, representation, forms, kd)
        beta, cov, estimable = fit_standard_cox_phreg(time, event, X)
        term = f"{concept}_rcs1"
        if term not in spec.feature_names:
            continue
        i = spec.feature_names.index(term)
        if estimable[i] and np.isfinite(cov[i, i]) and cov[i, i] > 0:
            scale = spec.sds[term]
            betas.append(beta[i] / scale)
            variances.append(cov[i, i] / scale**2)
    if not betas:
        return np.nan, 1, 0
    m = len(betas)
    q = float(np.mean(betas))
    within = float(np.mean(variances))
    between = float(np.var(betas, ddof=1)) if m > 1 else 0.0
    total = within + (1+1/m)*between
    stat = abs(q) / np.sqrt(total)
    if m > 1 and between > 0:
        df = (m-1) * (1 + within / ((1+1/m)*between))**2
        p = float(2 * student_t.sf(stat, df))
    else:
        p = float(chi2.sf(stat**2, 1))
    return p, 1, m


def select_functional_forms(imputed_train, raw_train, landmark):
    forms, knots_dict, rows = {}, {}, []
    time, event = raw_train["time"].to_numpy(float), raw_train["event"].to_numpy(int)
    for concept in CONTINUOUS_CONCEPTS:
        knots = compute_knots(imputed_train, concept)
        knots_dict[concept] = knots
        pdv = pcv = np.nan
        nd = nc = 0
        status = "Insufficient distinct values for three knots"
        if knots is not None:
            pdv, _, nd = nonlinear_test_for_representation(imputed_train, time, event, "discharge", concept, knots)
            pcv, _, nc = nonlinear_test_for_representation(imputed_train, time, event, "current", concept, knots)
            if not np.isfinite(pdv) or not np.isfinite(pcv):
                raise RuntimeError(f"{landmark} / {concept}: nonlinearity test not estimable; see data/design matrix.")
            status = "Pooled unstandardized nonlinear coefficient; Rubin t test"
        use = knots is not None and min(pdv, pcv) < NONLINEARITY_ALPHA
        forms[concept] = "spline" if use else "linear"
        rows.append({"Landmark": landmark, "Predictor": concept, "Discharge nonlinearity P": pdv,
            "Current-value nonlinearity P": pcv, "Selected functional form": forms[concept],
            "Knot 1": knots[0] if knots else np.nan, "Knot 2": knots[1] if knots else np.nan,
            "Knot 3": knots[2] if knots else np.nan, "Successful MI fits - discharge": nd,
            "Successful MI fits - current": nc, "Test status": status})
        if VERBOSE_FITS:
            log_message(f"  functional form {concept}: {forms[concept]} (P_dis={pdv:.4g}, P_cur={pcv:.4g})")
    return forms, knots_dict, pd.DataFrame(rows)

# %% [markdown]
# ## Survival metrics and conditional bootstrap CIs

# %%
class KMSurvival:
    def __init__(self, time: np.ndarray, status: np.ndarray):
        time = np.asarray(time, dtype=float)
        status = np.asarray(status, dtype=int)

        uniq = np.sort(np.unique(time))
        surv_after = []
        s = 1.0

        for t in uniq:
            at_risk = np.sum(time >= t)
            d = np.sum((time == t) & (status == 1))
            if at_risk > 0 and d > 0:
                s *= (1.0 - d / at_risk)
            surv_after.append(s)

        self.times = uniq
        self.surv_after = np.asarray(surv_after, dtype=float)

    def eval(self, q, side: str = "right"):
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))

        search_side = "right" if side == "right" else "left"
        idx = np.searchsorted(self.times, q_arr, side=search_side) - 1

        out = np.ones(len(q_arr), dtype=float)
        mask = idx >= 0
        out[mask] = self.surv_after[idx[mask]]
        out = np.clip(out, IPCW_MIN_G if getattr(self, "is_censoring", False) else 0.0, 1.0)

        if np.ndim(q) == 0:
            return float(out[0])
        return out


def censoring_km(time: np.ndarray, event: np.ndarray) -> KMSurvival:
    censor_status = 1 - np.asarray(event, dtype=int)
    km = KMSurvival(time, censor_status)
    km.is_censoring = True
    return km


class FenwickTree:
    def __init__(self, n: int):
        self.n = int(n)
        self.bit = np.zeros(self.n + 1, dtype=float)

    def add(self, idx: int, value: float):
        i = idx + 1
        while i <= self.n:
            self.bit[i] += value
            i += i & -i

    def sum(self, idx: int) -> float:
        if idx < 0:
            return 0.0
        i = idx + 1
        s = 0.0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


def harrell_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
) -> float:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk_score, dtype=float)

    valid = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time = time[valid]
    event = event[valid]
    risk = risk[valid]

    if len(time) < 2:
        return np.nan

    _, rank = np.unique(risk, return_inverse=True)
    tree = FenwickTree(int(rank.max()) + 1)

    order = np.argsort(-time, kind="mergesort")
    time_s = time[order]
    event_s = event[order]
    rank_s = rank[order]

    concordant = 0.0
    comparable = 0.0
    n_longer = 0.0

    start = 0
    while start < len(time_s):
        end = start + 1
        while end < len(time_s) and time_s[end] == time_s[start]:
            end += 1

        for i in range(start, end):
            if event_s[i] == 0:
                tree.add(int(rank_s[i]), 1.0)
                n_longer += 1.0

        for i in range(start, end):
            if event_s[i] != 1:
                continue

            r = int(rank_s[i])
            less = tree.sum(r - 1)
            leq = tree.sum(r)
            equal = leq - less

            concordant += less + 0.5 * equal
            comparable += n_longer

        for i in range(start, end):
            if event_s[i] == 1:
                tree.add(int(rank_s[i]), 1.0)
                n_longer += 1.0

        start = end

    return (
        concordant / comparable
        if comparable > 0 else np.nan
    )


def calibration_ipcw(
    time: np.ndarray,
    event: np.ndarray,
    risk_prob: np.ndarray,
    horizon: float,
    censor_km_model: KMSurvival,
) -> Tuple[float, float]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    p = np.clip(
        np.asarray(risk_prob, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    event_before, at_risk, endpoint = _horizon_masks(time, event, horizon)

    y = np.zeros(len(time), dtype=float)
    y[event_before] = 1.0

    w = np.zeros(len(time), dtype=float)

    if event_before.any():
        w[event_before] = 1.0 / censor_km_model.eval(
            time[event_before],
            side="left",
        )

    if at_risk.any():
        w[at_risk] = 1.0 / censor_km_model.eval(
            horizon,
            side="left" if endpoint else "right",
        )

    use = w > 0
    if use.sum() < 20 or len(np.unique(y[use])) < 2:
        return np.nan, np.nan

    z = np.log(-np.log(1.0 - p))

    try:
        X0 = np.ones((use.sum(), 1))
        mod0 = sm.GLM(
            y[use],
            X0,
            family=sm.families.Binomial(
                link=sm.families.links.CLogLog()
            ),
            freq_weights=w[use],
            offset=z[use],
        ).fit()
        intercept = float(mod0.params[0])
    except Exception:
        intercept = np.nan

    try:
        X1 = sm.add_constant(z[use], has_constant="add")
        mod1 = sm.GLM(
            y[use],
            X1,
            family=sm.families.Binomial(
                link=sm.families.links.CLogLog()
            ),
            freq_weights=w[use],
        ).fit()
        slope = float(mod1.params[1])
    except Exception:
        slope = np.nan

    return intercept, slope


def km_event_risk(
    time: np.ndarray,
    event: np.ndarray,
    horizon: float,
) -> float:
    km = KMSurvival(time, event)
    return 1.0 - km.eval(horizon, side="right")


def calibration_table(
    time: np.ndarray,
    event: np.ndarray,
    risk_prob: np.ndarray,
    horizon: float,
    n_groups: int = 10,
) -> pd.DataFrame:
    d = pd.DataFrame({
        "time": time,
        "event": event,
        "pred": risk_prob,
    }).dropna()

    if len(d) == 0:
        return pd.DataFrame()

    q = min(n_groups, max(2, d["pred"].nunique()))
    try:
        d["group"] = pd.qcut(
            d["pred"],
            q=q,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        d["group"] = 0

    rows = []
    for g, x in d.groupby("group", sort=True):
        rows.append({
            "group": int(g) + 1,
            "n": len(x),
            "mean_predicted_risk": x["pred"].mean(),
            "observed_risk": km_event_risk(
                x["time"].to_numpy(),
                x["event"].to_numpy(),
                horizon,
            ),
        })

    return pd.DataFrame(rows)


def favorable_delta(
    metrics_discharge: Dict[str, float],
    metrics_current: Dict[str, float],
) -> Dict[str, float]:
    return {
        "Delta C-index": (
            metrics_current["C-index"] - metrics_discharge["C-index"]
        ),
        "Delta AUC180": (
            metrics_current["AUC180"] - metrics_discharge["AUC180"]
        ),
        "Delta Brier180": (
            metrics_discharge["Brier180"] - metrics_current["Brier180"]
        ),
        "Delta IBS": (
            metrics_discharge["IBS"] - metrics_current["IBS"]
        ),
    }


def _horizon_masks(time, event, horizon):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    case = (event == 1) & (time <= horizon)
    control = time > horizon
    # Source cohorts are administratively truncated at HORIZON. At this
    # boundary, event=0 and time=HORIZON means known event-free through day 180.
    # Without this explicit convention, ALL endpoint controls disappear.
    endpoint = bool(float(horizon) == float(HORIZON))
    if endpoint:
        control = control | ((event == 0) & (time == horizon))
    return case, control, endpoint


def time_dependent_auc(time, event, risk_score, horizon, censor_km_model):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    score = np.asarray(risk_score, dtype=float)
    case, control, endpoint = _horizon_masks(time, event, horizon)
    if not case.any() or not control.any():
        return np.nan
    wc = np.zeros(len(time), dtype=float)
    wn = np.zeros(len(time), dtype=float)
    wc[case] = 1.0 / censor_km_model.eval(time[case], side="left")
    wn[control] = 1.0 / censor_km_model.eval(horizon, side="left" if endpoint else "right")
    order = np.argsort(score, kind="mergesort")
    ss, wc, wn = score[order], wc[order], wn[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    cw = np.add.reduceat(wc, starts)
    nw = np.add.reduceat(wn, starts)
    before = np.cumsum(nw) - nw
    return float(np.sum(cw * (before + 0.5 * nw)) / (wc.sum() * wn.sum()))


def _brier_loss_matrix(time, event, survival, eval_times, censor_model):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    ts = np.asarray(eval_times, dtype=float)
    s = np.asarray(survival, dtype=float)
    if s.shape != (len(time), len(ts)):
        raise ValueError("Survival matrix and evaluation times do not agree.")
    before = (event[:, None] == 1) & (time[:, None] <= ts[None, :])
    control = time[:, None] > ts[None, :]
    endpoint_columns = ts == HORIZON
    if endpoint_columns.any():
        control[:, endpoint_columns] |= ((event == 0) & (time == HORIZON))[:, None]
    g_event = censor_model.eval(time, side="left")
    g_grid = np.asarray(censor_model.eval(ts, side="right"), dtype=float)
    if endpoint_columns.any():
        g_grid[endpoint_columns] = censor_model.eval(ts[endpoint_columns], side="left")
    return (before * (s ** 2) / g_event[:, None]
            + control * ((1.0 - s) ** 2) / g_grid[None, :])


def brier_score_ipcw(time, event, survival_prob, horizon, censor_km_model):
    loss = _brier_loss_matrix(time, event, np.asarray(survival_prob)[:, None],
                              np.asarray([horizon], dtype=float), censor_km_model)
    return float(loss.mean())


def evaluate_predictions(raw_eval, pred, censor_km_model, eval_times=EVAL_TIMES,
                         include_calibration=True, make_calibration_table=True):
    time = raw_eval["time"].to_numpy(dtype=float)
    event = raw_eval["event"].to_numpy(dtype=int)
    surv = pred["survival"]
    risk180 = 1.0 - surv[:, -1]
    brier_values = _brier_loss_matrix(time, event, surv, eval_times, censor_km_model).mean(axis=0)
    ibs = (float(trapezoid(brier_values, x=eval_times) / (eval_times[-1] - eval_times[0]))
           if len(eval_times) > 1 else float(brier_values[0]))
    metrics = {
        "C-index": harrell_c_index(time, event, pred["lp"]),
        "AUC180": time_dependent_auc(time, event, risk180, HORIZON, censor_km_model),
        "Brier180": float(brier_values[-1]),
        "IBS": ibs,
    }
    cal = pd.DataFrame()
    if include_calibration:
        intercept, slope = calibration_ipcw(time, event, risk180, HORIZON, censor_km_model)
        metrics["Calibration intercept"] = intercept
        metrics["Calibration slope"] = slope
        if make_calibration_table:
            cal = calibration_table(time, event, risk180, HORIZON, n_groups=10)
    return metrics, cal

def require_prediction_alignment(raw, pred):
    n = len(raw)
    if pred["survival"].shape != (n, len(EVAL_TIMES)) or len(pred["lp"]) != n:
        raise ValueError("Prediction arrays are not aligned with evaluation rows.")
    if not np.isfinite(pred["survival"]).all() or not np.isfinite(pred["lp"]).all():
        raise ValueError("Nonfinite predictions.")
    if (pred["survival"] < 0).any() or (pred["survival"] > 1).any():
        raise ValueError("Survival predictions outside [0,1].")
    if (np.diff(pred["survival"], axis=1) > 1e-10).any():
        raise ValueError("Survival predictions are not nonincreasing.")


def subset_prediction(pred, idx):
    return {"lp": np.asarray(pred["lp"])[idx], "survival": np.asarray(pred["survival"])[idx]}


def stratified_bootstrap_indices(df, center_col, seed):
    # Resample patients within center. Multiple INDEX rows for one patient stay together.
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True)
    chunks = []
    for _, g in d.groupby(center_col, sort=True):
        key = "patient_key" if "patient_key" in g else "id"
        groups = [x.index.to_numpy(dtype=int) for _, x in g.groupby(key, sort=True)]
        draws = rng.integers(0, len(groups), len(groups))
        chunks.extend(groups[j] for j in draws)
    return np.concatenate(chunks)


def percentile_ci(values, level=0.95):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return tuple(np.quantile(x, [(1-level)/2, 1-(1-level)/2]).tolist())


def ci_summary(boot, column):
    if column not in boot:
        raise KeyError(f"Missing bootstrap metric column: {column}")
    x = pd.to_numeric(boot[column], errors="coerce").to_numpy(float)
    valid = np.isfinite(x)
    required = max(2, math.ceil(CI_MIN_VALID_FRACTION * len(x)))
    if valid.sum() < required:
        raise RuntimeError(f"{column}: only {int(valid.sum())}/{len(x)} finite bootstrap values. "
                           "No CI is fabricated; inspect bootstrap diagnostics.")
    lo, hi = percentile_ci(x[valid])
    return float(lo), float(hi), int(valid.sum())


def fixed_prediction_bootstrap_ci(raw, pred_dis, pred_cur, n_boot, seed, cache_tag):
    """Conditional patient-sampling CI, not a model-development/generalization CI.

    Both models use each identical patient resample. Censoring KM is refitted to the resample.
    Only evaluation/calibration models are fitted; primary Cox and MICE are not fitted.
    """
    require_prediction_alignment(raw, pred_dis)
    require_prediction_alignment(raw, pred_cur)
    signature = _config_digest({"version": ANALYSIS_VERSION, "raw": _frame_digest(raw), "seed": seed,
        "tag": cache_tag, "dis": _array_digest(pred_dis["survival"]), "cur": _array_digest(pred_cur["survival"])})
    path = CACHE_DIR / f"CI_{cache_tag}_{signature[:16]}.pkl"
    records = []
    if RESUME_FROM_CHECKPOINT and path.exists():
        with path.open("rb") as f:
            saved = pickle.load(f)
        if saved["signature"] != signature:
            raise ValueError("CI cache signature mismatch.")
        records = saved["records"]
    start_n = len(records)
    started = _clock.perf_counter()
    try:
        for b in range(start_n, int(n_boot)):
            idx = stratified_bootstrap_indices(raw, "center", seed+b+1)
            rb = raw.iloc[idx].reset_index(drop=True)
            ck = censoring_km(rb["time"].to_numpy(float), rb["event"].to_numpy(int))
            row = {"iteration": b+1}
            ms = []
            for name, pr in [("Discharge", pred_dis), ("Current", pred_cur)]:
                met, _ = evaluate_predictions(rb, subset_prediction(pr, idx), ck, include_calibration=True, make_calibration_table=False)
                row.update({f"{name}_{k}": met[k] for k in METRICS})
                ms.append(met)
            row.update(favorable_delta(*ms))
            records.append(row)
            if (b+1) % 50 == 0 or b+1 == n_boot:
                _atomic_pickle({"signature": signature, "records": records}, path)
            if (b+1) % 100 == 0 or b+1 == n_boot:
                elapsed = _clock.perf_counter()-started
                eta = elapsed / max(1, b+1-start_n) * (n_boot-b-1)
                log_message(f"[{cache_tag}] CI {b+1}/{n_boot}; elapsed={elapsed/60:.1f} min; ETA={eta/60:.1f} min")
    except BaseException:
        _atomic_pickle({"signature": signature, "records": records}, path)
        raise
    return pd.DataFrame(records[:n_boot])


# %% [markdown]
# ## Primary fitting and model saving

# %%
def evaluate_pair_from_imputed(raw, imputed, landmark, seed, forms=None, knots=None):
    if forms is None:
        forms, knots, nonlinear = select_functional_forms(imputed, raw, landmark)
    else:
        nonlinear = pd.DataFrame()
    bundles, predictions, metrics, curves = {}, {}, {}, {}
    ck = censoring_km(raw["time"].to_numpy(float), raw["event"].to_numpy(int))
    for j, rep in enumerate(("discharge", "current")):
        if VERBOSE_FITS:
            log_message(f"[{landmark}] Fitting {rep} model...")
        bundle = fit_mi_cox_bundle(imputed, raw, rep, forms, knots, seed+10000*(j+1))
        pred = predict_mi_bundle(bundle, imputed)
        require_prediction_alignment(raw, pred)
        met, cal = evaluate_predictions(raw, pred, ck)
        if not all(np.isfinite(met[k]) for k in METRICS):
            raise RuntimeError(f"{landmark}/{rep}: non-estimable primary performance: {met}")
        bundles[rep], predictions[rep], metrics[rep], curves[rep] = bundle, pred, met, cal
    result = {"raw": raw, "imputed": imputed, "forms": forms, "knots": knots,
              "nonlinearity": nonlinear, "censor_km": ck,
              "center_levels": sorted(raw["center"].astype(str).unique().tolist())}
    for rep in ("discharge", "current"):
        result.update({f"{rep}_bundle": bundles[rep], f"{rep}_pred": predictions[rep],
                       f"{rep}_metrics": metrics[rep], f"{rep}_calibration": curves[rep]})
    result["delta"] = favorable_delta(metrics["discharge"], metrics["current"])
    return result


def run_primary_landmark(raw, landmark, seed):
    levels = sorted(raw["center"].astype(str).unique().tolist())
    log_message(f"[{landmark}] Multiple imputation...")
    imp, _, states, trace = mice_impute_train_valid(raw, None, levels, MICE_M, MICE_ITERATIONS,
                                                  seed, landmark, PMM_DONORS, return_states=True)
    result = evaluate_pair_from_imputed(raw, imp, landmark, seed)
    result["imputation_states"] = states
    result["imputation_trace"] = trace
    return result


def reuse_or_fit_primary(raw, landmark, seed):
    return cached_stage("primary_models", landmark, raw, seed,
                        lambda: run_primary_landmark(raw, landmark, seed))


def parameter_tables(bundle, landmark, model_name):
    coefs, scales, knots, baseline, manifest = [], [], [], [], []
    base = MODEL_DIR / landmark / model_name
    base.mkdir(parents=True, exist_ok=True)
    for j, model in enumerate(bundle.models, 1):
        spec = model.design_spec
        tag = f"MI{j:02d}"
        np.savez_compressed(base / f"{tag}_parameters.npz", coefficients=model.coefficients,
            covariance=model.covariance, estimable_mask=model.estimable_mask,
            baseline_event_times=model.baseline_event_times, baseline_cumhaz=model.baseline_cumhaz)
        payload = {"version": ANALYSIS_VERSION, "landmark": landmark, "model": model_name,
            "ties": COX_TIES, "representation": spec.representation, "forms": spec.forms, "knots": spec.knots,
            "feature_names": spec.feature_names, "continuous_feature_names": spec.continuous_feature_names,
            "means": spec.means, "sds": spec.sds, "extra_continuous": spec.extra_continuous}
        (base / f"{tag}_design.json").write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        for k, name in enumerate(spec.feature_names):
            sd = spec.sds.get(name, 1.0)
            b, var = model.coefficients[k]/sd, model.covariance[k,k]/sd**2
            coefs.append({"Landmark": landmark, "Model": model_name, "MI": j, "Term": name,
                "Coefficient on basis scale": b, "Variance on basis scale": var,
                "Estimable": bool(model.estimable_mask[k]),
                "Standardized coefficient": model.coefficients[k]})
            scales.append({"Landmark": landmark, "Model": model_name, "MI": j, "Term": name,
                           "Mean": spec.means.get(name, 0.0), "SD": sd})
        for name, ks in spec.knots.items():
            knots.append({"Landmark": landmark, "Model": model_name, "MI": j, "Predictor": name,
                          "Form": spec.forms[name], "Knots": json.dumps(ks)})
        baseline.extend({"Landmark": landmark, "Model": model_name, "MI": j, "Time": t,
                         "Baseline cumulative hazard": h} for t, h in zip(model.baseline_event_times, model.baseline_cumhaz))
        manifest.append({"Landmark": landmark, "Model": model_name, "MI": j,
                         "Parameters": str(base / f"{tag}_parameters.npz"), "Design": str(base / f"{tag}_design.json")})
    coef = pd.DataFrame(coefs)
    pooled = []
    for term, g in coef.groupby("Term", sort=False):
        valid = g["Estimable"] & np.isfinite(g["Variance on basis scale"])
        x = g.loc[valid]
        if len(x) != len(bundle.models):
            pooled.append({"Landmark": landmark, "Model": model_name, "Term": term,
                           "Status": "Not pooled: non-estimable in at least one imputation", "MI N": len(x)})
            continue
        m = len(x)
        b = x["Coefficient on basis scale"].mean()
        w = x["Variance on basis scale"].mean()
        v = x["Coefficient on basis scale"].var(ddof=1) if m > 1 else 0.0
        se = np.sqrt(w + (1+1/m)*v)
        df = (m-1)*(1+w/((1+1/m)*v))**2 if m > 1 and v > 0 else np.inf
        cut = student_t.ppf(.975, df)
        pooled.append({"Landmark": landmark, "Model": model_name, "Term": term, "MI N": m,
            "Coefficient": b, "Standard error": se, "95% CI lower": b-cut*se,
            "95% CI upper": b+cut*se, "Status": "Rubin pooling on common unstandardized basis; spline terms are not standalone clinical HRs"})
    return {"Manifest": pd.DataFrame(manifest), "Coefficients by MI": coef,
            "Pooled coefficients": pd.DataFrame(pooled), "Standardization": pd.DataFrame(scales),
            "Functional forms": pd.DataFrame(knots), "Baseline hazards": pd.DataFrame(baseline)}


def save_primary_outputs(primary_result, landmark):
    p = primary_result
    # This state contains model parameters, imputation equations, completed data and predictions.
    _atomic_pickle({"version": ANALYSIS_VERSION, "settings": _settings_signature(), "primary": p},
                   MODEL_DIR / f"{landmark}_complete_model_state.pkl")
    tables = {}
    for name, rep in [("Discharge", "discharge"), ("Current", "current")]:
        for sheet, frame in parameter_tables(p[f"{rep}_bundle"], landmark, name).items():
            tables.setdefault(sheet, []).append(frame)
    export_excel({k: pd.concat(v, ignore_index=True) for k,v in tables.items()},
                 PARAMETER_DIR / f"{landmark}_model_parameters.xlsx")
    patient = p["raw"][["id", "center", "patient_key", "prediction_date", "index_outpatient_date", "visit_day", "time", "event", "event_component"]].copy()
    for rep in ("discharge", "current"):
        patient[f"{rep}_lp"] = p[f"{rep}_pred"]["lp"]
        patient[f"{rep}_risk180"] = 1 - p[f"{rep}_pred"]["survival"][:, -1]
    patient.to_parquet(ARTIFACT_DIR / f"{landmark}_primary_predictions.parquet", index=False)
    np.savez_compressed(ARTIFACT_DIR / f"{landmark}_primary_prediction_arrays.npz", eval_times=EVAL_TIMES,
        discharge_lp=p["discharge_pred"]["lp"], discharge_survival=p["discharge_pred"]["survival"],
        current_lp=p["current_pred"]["lp"], current_survival=p["current_pred"]["survival"])
    cal = pd.concat([p["discharge_calibration"].assign(Model="Discharge"),
                     p["current_calibration"].assign(Model="Current")], ignore_index=True).assign(Landmark=landmark)
    cal.to_pickle(ARTIFACT_DIR / f"{landmark}_primary_calibration.pkl")
    export_excel({"Calibration": cal, "MICE trace": p["imputation_trace"]},
                 ARTIFACT_DIR / f"{landmark}_primary_diagnostics.xlsx")


# %% [markdown]
# ## Bootstrap redevelopment and optimism correction

# %%
@contextmanager
def _temporary_compute_settings(values):
    old = {k: globals()[k] for k in values}
    try:
        globals().update(values)
        yield
    finally:
        globals().update(old)


def _bootstrap_compute_config():
    return {"m": BOOTSTRAP_MICE_M, "n_iter": BOOTSTRAP_MICE_ITERATIONS,
        "form_m": BOOTSTRAP_NONLINEARITY_MI_DATASETS, "reselect": RESELECT_FORMS_IN_BOOTSTRAP,
        "version": ANALYSIS_VERSION, "worker_threads": BOOTSTRAP_WORKER_THREADS,
        "primary_settings": _settings_signature(), "nonlinearity_alpha": NONLINEARITY_ALPHA, "pmm_donors": PMM_DONORS}


def _one_bootstrap_redevelopment(raw, forms_original, knots_original, landmark, seed, b, config):
    start = _clock.perf_counter()
    row = {"iteration": b+1}
    stage = "resampling"
    try:
        with _temporary_compute_settings({"NONLINEARITY_MI_DATASETS": config["form_m"], "VERBOSE_FITS": False}), \
                threadpool_limits(limits=config["worker_threads"]):
            idx = stratified_bootstrap_indices(raw, "center", seed+100000+b)
            boot = raw.iloc[idx].reset_index(drop=True)
            stage = "MICE"
            t0 = _clock.perf_counter()
            ib, io = mice_impute_train_valid(boot, raw, sorted(raw["center"].unique().tolist()),
                config["m"], config["n_iter"], seed+100000+b, landmark, PMM_DONORS)
            row["MICE seconds"] = _clock.perf_counter()-t0
            stage = "functional forms"
            t0 = _clock.perf_counter()
            if config["reselect"]:
                forms, knots, _ = select_functional_forms(ib, boot, landmark)
            else:
                forms, knots = forms_original, knots_original
            row["Form selection seconds"] = _clock.perf_counter()-t0
            stage = "Cox"
            t0 = _clock.perf_counter()
            train_metrics, test_metrics = [], []
            # Each evaluation population estimates its own censoring distribution.
            ckb = censoring_km(boot["time"].to_numpy(float), boot["event"].to_numpy(int))
            cko = censoring_km(raw["time"].to_numpy(float), raw["event"].to_numpy(int))
            for name, rep in [("Discharge", "discharge"), ("Current", "current")]:
                bundle = fit_mi_cox_bundle(ib, boot, rep, forms, knots, seed+b)
                train, _ = evaluate_predictions(boot, predict_mi_bundle(bundle, ib), ckb,
                                                  make_calibration_table=False)
                test, _ = evaluate_predictions(raw, predict_mi_bundle(bundle, io), cko,
                                                 make_calibration_table=False)
                for metric in METRICS:
                    row[f"{name}_{metric}_train"] = train[metric]
                    row[f"{name}_{metric}_test"] = test[metric]
                train_metrics.append(train)
                test_metrics.append(test)
            row["Cox and scoring seconds"] = _clock.perf_counter()-t0
            dt, dv = favorable_delta(*train_metrics), favorable_delta(*test_metrics)
            row.update(dt)
            row.update({f"{k}_test": v for k,v in dv.items()})
            row.update({f"{k}_optimism": dt[k]-dv[k] for k in dt})
            row.update({f"Form_{k}": v for k,v in forms.items()})
            row["Error"] = ""
    except Exception as exc:
        row["Error"] = f"{stage}: {type(exc).__name__}: {exc}"
        row["Traceback"] = traceback.format_exc()
    row["Total seconds"] = _clock.perf_counter()-start
    return row


def _summarize_primary_bootstrap(primary, landmark, records):
    boot = pd.DataFrame(records).sort_values("iteration").reset_index(drop=True)
    if "Error" in boot and boot["Error"].ne("").any():
        export_excel({"Failed replicates": boot.loc[boot["Error"].ne("")]},
                     ARTIFACT_DIR / f"{landmark}_bootstrap_failures.xlsx")
    opt = boot.iloc[:INTERNAL_BOOTSTRAP_N]
    corrected, deltas = [], []
    for name, rep in [("Discharge", "discharge"), ("Current", "current")]:
        for metric in METRICS:
            train_col, test_col = f"{name}_{metric}_train", f"{name}_{metric}_test"
            ci_summary(opt, train_col)  # enforce a minimum number of valid fits
            v = opt[train_col] - opt[test_col]
            v = v[np.isfinite(v)]
            if len(v) < math.ceil(CI_MIN_VALID_FRACTION * len(opt)):
                raise RuntimeError(f"Too few valid paired train/test metrics: {landmark}/{name}/{metric}")
            mean, mcse = float(v.mean()), float(v.std(ddof=1)/np.sqrt(len(v)))
            app = primary[f"{rep}_metrics"][metric]
            corrected.append({"Landmark": landmark, "Model": name, "Metric": metric,
                "Apparent": app, "Mean optimism": mean, "Optimism-corrected": app-mean,
                "Valid bootstrap N": len(v), "Optimism Monte Carlo SE": mcse,
                "Bootstrap MI": BOOTSTRAP_MICE_M, "Bootstrap cycles": BOOTSTRAP_MICE_ITERATIONS,
                "Form reselection": RESELECT_FORMS_IN_BOOTSTRAP,
                "Validation scope": "Numerically reduced MI redevelopment; includes form selection" if RESELECT_FORMS_IN_BOOTSTRAP else
                                    "Numerically reduced MI redevelopment; conditional on selected forms"})
    for metric in DELTA_METRICS:
        lo, hi, n = ci_summary(boot.iloc[:PAIRED_BOOTSTRAP_N], metric)
        v = opt[f"{metric}_optimism"].to_numpy(float)
        v = v[np.isfinite(v)]
        if len(v) < math.ceil(CI_MIN_VALID_FRACTION * len(opt)):
            raise RuntimeError(f"Too few valid delta optimism values: {landmark}/{metric}")
        deltas.append({"Landmark": landmark, "Metric": metric, "Point estimate": primary["delta"][metric],
            "95% CI lower": lo, "95% CI upper": hi, "Valid bootstrap N": n,
            "Mean delta optimism": float(v.mean()),
            "Optimism-corrected difference": primary["delta"][metric]-float(v.mean()),
            "CI target": "Apparent paired difference; reduced-MI redevelopment percentile interval"})
    return {"bootstrap_records": boot, "corrected": pd.DataFrame(corrected), "delta_ci": pd.DataFrame(deltas)}


def run_primary_bootstrap(primary, landmark, seed):
    config = _bootstrap_compute_config()
    requested = max(INTERNAL_BOOTSTRAP_N, PAIRED_BOOTSTRAP_N)
    if config["m"] < config["form_m"] or min(config["m"], config["n_iter"]) < 1:
        raise ValueError("Incompatible bootstrap MI settings.")
    sig = _config_digest({"raw": _frame_digest(primary["raw"]), "config": config, "seed": seed,
        "pred": [_array_digest(primary[f"{r}_pred"]["survival"]) for r in ("discharge", "current")],
        "forms": primary["forms"], "knots": primary["knots"]})
    path = CACHE_DIR / f"{landmark}_redevelopment_{sig[:16]}.pkl"
    records = {}
    if RESUME_FROM_CHECKPOINT and path.exists():
        with path.open("rb") as f:
            saved = pickle.load(f)
        if saved["signature"] != sig:
            raise ValueError("Bootstrap checkpoint mismatch.")
        records = {int(x["iteration"]): x for x in saved["records"]}
    pending = [b for b in range(requested) if b+1 not in records]
    log_message(f"[{landmark}] redevelopment completed cache={requested-len(pending)}/{requested}; "
                f"MI={config['m']}x{config['n_iter']}; form reselection={config['reselect']}")
    started = _clock.perf_counter()
    completed_call = 0
    def consume(row):
        nonlocal completed_call
        records[row["iteration"]] = row
        completed_call += 1
        _atomic_pickle({"signature": sig, "records": list(records.values())}, path)
        done = sum(i <= requested for i in records)
        elapsed = _clock.perf_counter()-started
        eta = elapsed/max(1, completed_call)*(requested-done)
        log_message(f"[{landmark}] bootstrap {done}/{requested}; elapsed={elapsed/60:.1f} min; ETA={eta/60:.1f} min"
                    + (f"; FAILED: {row['Error']}" if row.get("Error") else ""))
    jobs = min(BOOTSTRAP_N_JOBS, len(pending)) if pending else 1
    if jobs <= 1:
        for b in pending:
            consume(_one_bootstrap_redevelopment(primary["raw"], primary["forms"], primary["knots"], landmark, seed, b, config))
    else:
        with parallel_config(backend="loky", inner_max_num_threads=BOOTSTRAP_WORKER_THREADS):
            with Parallel(n_jobs=jobs, return_as="generator_unordered", batch_size=1, pre_dispatch=jobs) as parallel:
                for row in parallel(delayed(_one_bootstrap_redevelopment)(primary["raw"], primary["forms"], primary["knots"],
                                         landmark, seed, b, config) for b in pending):
                    consume(row)
    result = _summarize_primary_bootstrap(primary, landmark, [records[i] for i in range(1, requested+1)])
    return result


# %% [markdown]
# ## IECV, subgroups and sensitivity models

# %%
def pair_ci_table(raw, pred_dis, pred_cur, landmark, boot, labels=("Discharge", "Current"), context=None):
    ck = censoring_km(raw["time"].to_numpy(float), raw["event"].to_numpy(int))
    rows, mets = [], []
    for prefix, name, pred in [("Discharge", labels[0], pred_dis), ("Current", labels[1], pred_cur)]:
        met, _ = evaluate_predictions(raw, pred, ck, make_calibration_table=False)
        mets.append(met)
        for metric in METRICS:
            lo, hi, count = ci_summary(boot, f"{prefix}_{metric}")
            if not np.isfinite(met[metric]):
                raise RuntimeError(f"{landmark}/{name}/{metric}: estimate is not finite.")
            rows.append({"Landmark": landmark, **(context or {}), "N": len(raw), "Events": int(raw["event"].sum()),
                "Model": name, "Metric": metric, "Estimate": met[metric], "95% CI lower": lo,
                "95% CI upper": hi, "Valid bootstrap N": count,
                "CI method": "Patient bootstrap within center; predictions fixed; censoring KM re-estimated"})
    for metric, value in favorable_delta(*mets).items():
        lo, hi, count = ci_summary(boot, metric)
        rows.append({"Landmark": landmark, **(context or {}), "N": len(raw), "Events": int(raw["event"].sum()),
            "Model": "Current vs Discharge improvement" if labels == ("Discharge", "Current") else "Augmented vs Current improvement",
            "Metric": metric, "Estimate": value, "95% CI lower": lo, "95% CI upper": hi,
            "Valid bootstrap N": count, "CI method": "Paired patient bootstrap within center; predictions fixed"})
    return pd.DataFrame(rows)


def _fit_iecv_fold(raw, heldout, landmark, seed):
    train = raw.loc[raw["center"].ne(heldout)].reset_index(drop=True)
    valid = raw.loc[raw["center"].eq(heldout)].reset_index(drop=True)
    levels = sorted(train["center"].unique().tolist())
    # Donors, coefficients, center effects, knots and form selection use training rows only.
    it, iv, states, trace = mice_impute_train_valid(train, valid, levels, MICE_M, MICE_ITERATIONS,
        seed, landmark, PMM_DONORS, return_states=True)
    fitted = evaluate_pair_from_imputed(train, it, landmark, seed)
    pred_dis = predict_mi_bundle(fitted["discharge_bundle"], iv)
    pred_cur = predict_mi_bundle(fitted["current_bundle"], iv)
    return {"raw": valid, "discharge_pred": pred_dis, "current_pred": pred_cur,
            "discharge_bundle": fitted["discharge_bundle"], "current_bundle": fitted["current_bundle"],
            "imputation_states": states, "forms": fitted["forms"], "knots": fitted["knots"],
            "nonlinearity": fitted["nonlinearity"], "trace": trace}


def run_iecv(raw, landmark, seed):
    centers = sorted(raw["center"].unique().tolist())
    if len(centers) < 2:
        raise ValueError("IECV requires at least two centers.")
    tables, curves, predictions = [], [], []
    for j, center in enumerate(centers):
        s = seed+10000*(j+1)
        log_message(f"[{landmark}] IECV held-out center={center}")
        fold = cached_stage(f"iecv_model_{j}", landmark, raw, s,
            lambda center=center, s=s: _fit_iecv_fold(raw, center, landmark, s), extra={"heldout": center})
        d = fold["raw"]
        boot = fixed_prediction_bootstrap_ci(d, fold["discharge_pred"], fold["current_pred"],
            IECV_CI_BOOTSTRAP_N, s+5000, f"{landmark}_IECV_{j}")
        tables.append(pair_ci_table(d, fold["discharge_pred"], fold["current_pred"], landmark, boot,
                                    context={"Validation center": center}))
        ck = censoring_km(d["time"].to_numpy(float), d["event"].to_numpy(int))
        for model, rep in [("Discharge", "discharge"), ("Current", "current")]:
            risk = 1-fold[f"{rep}_pred"]["survival"][:,-1]
            cal = calibration_table(d["time"].to_numpy(float), d["event"].to_numpy(int), risk,
                                    HORIZON, n_groups=5)
            curves.append(cal.assign(Landmark=landmark, Model=model, **{"Validation center": center}))
            predictions.append(d[["id", "patient_key", "center", "time", "event"]].assign(
                Landmark=landmark, Model=model, risk180=risk))
    return {"results": pd.concat(tables, ignore_index=True),
            "calibration": pd.concat(curves, ignore_index=True),
            "predictions": pd.concat(predictions, ignore_index=True)}


SUBGROUP_SPECS = [
    ("Age <75 years", lambda d: d["age"].notna() & (d["age"] < 75)),
    ("Age >=75 years", lambda d: d["age"].notna() & (d["age"] >= 75)),
    ("Male", lambda d: d["male"].eq(1)), ("Female", lambda d: d["male"].eq(0)),
    ("LVEF <50%", lambda d: d["ref"].eq(1)), ("LVEF >=50%", lambda d: d["ref"].eq(0)),
    ("Discharge eGFR <45", lambda d: d["discharge_egfr"].notna() & (d["discharge_egfr"] < 45)),
    ("Discharge eGFR >=45", lambda d: d["discharge_egfr"].notna() & (d["discharge_egfr"] >= 45)),
]


def run_subgroups(primary, landmark, seed):
    tables, skipped = [], []
    for j, (name, select) in enumerate(SUBGROUP_SPECS):
        idx = np.flatnonzero(select(primary["raw"]).to_numpy())
        d = primary["raw"].iloc[idx].reset_index(drop=True)
        if len(d) < MIN_SUBGROUP_N or d["event"].sum() < 10:
            skipped.append({"Landmark": landmark, "Subgroup": name, "N": len(d),
                            "Events": int(d["event"].sum()), "Reason": "Insufficient subgroup size/events"})
            continue
        pdv, pcv = subset_prediction(primary["discharge_pred"], idx), subset_prediction(primary["current_pred"], idx)
        boot = fixed_prediction_bootstrap_ci(d, pdv, pcv, SUBGROUP_BOOTSTRAP_N, seed+10000*j,
                                             f"{landmark}_subgroup_{j}")
        tables.append(pair_ci_table(d, pdv, pcv, landmark, boot, context={"Subgroup": name}))
    return {"results": pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(),
            "skipped": pd.DataFrame(skipped)}


def primary_required_model_columns():
    return FIXED_CONTINUOUS + ALWAYS_LINEAR_CONTINUOUS + FIXED_BINARY + [f"{rep}_{lab}" for rep in ("discharge", "current") for lab in LAB_NAMES]


def add_change_variables(imputed):
    extra = [f"delta30_{lab}" for lab in LAB_NAMES]
    out = []
    for d0 in imputed:
        d = d0.copy()
        for lab in LAB_NAMES:
            day_lab = "ntprobnp" if lab == "ntprobnp_log2" else lab
            days = d[f"current_day_{day_lab}"].to_numpy(float)
            # Existing 30-day-standardized change convention retained.
            # A day-0 value uses a one-day floor, explicitly recorded in Settings/notes.
            days = np.maximum(days, 1.0)
            d[f"delta30_{lab}"] = (d[f"current_{lab}"] - d[f"discharge_{lab}"]) * 30.0 / days
        out.append(d)
    return out, extra


def _fit_longitudinal(primary, landmark, seed):
    augmented_data, extra = add_change_variables(primary["imputed"])
    bundle = fit_mi_cox_bundle(augmented_data, primary["raw"], "current", primary["forms"], primary["knots"],
                              seed, extra_continuous=extra)
    return {"raw": primary["raw"], "discharge_pred": primary["current_pred"],
            "current_pred": predict_mi_bundle(bundle, augmented_data), "augmented_bundle": bundle}


def sensitivity_longitudinal(primary, landmark, seed):
    pair = cached_stage("longitudinal_models", landmark, primary["raw"], seed,
        lambda: _fit_longitudinal(primary, landmark, seed),
        extra={"reference": _array_digest(primary["current_pred"]["survival"]), "forms": primary["forms"]})
    boot = fixed_prediction_bootstrap_ci(pair["raw"], pair["discharge_pred"], pair["current_pred"],
                                         SENSITIVITY_BOOTSTRAP_N, seed+5000, f"{landmark}_longitudinal")
    return pair_ci_table(pair["raw"], pair["discharge_pred"], pair["current_pred"], landmark, boot,
        labels=("Primary current-value", "Current-value + 30-day-standardized changes"),
        context={"Analysis": "Longitudinal-change sensitivity"})


def _fit_other_sensitivity(raw, landmark, seed, mode):
    if mode == "Complete-case":
        imp = [raw.copy()]
    elif mode == "Single imputation":
        one, _ = single_imputation(raw)
        imp = [one]
    else:
        imp, _ = mice_impute_train_valid(raw, None, sorted(raw["center"].unique().tolist()),
                                         MICE_M, MICE_ITERATIONS, seed, landmark, PMM_DONORS)
    return evaluate_pair_from_imputed(raw, imp, landmark, seed)


def sensitivity_missing_and_dialysis(raw, landmark, seed):
    rows, skipped = [], []
    cases = [("Complete-case", raw.loc[raw[primary_required_model_columns()].notna().all(axis=1)]),
             ("Single imputation", raw), ("Excluding dialysis", raw.loc[raw["dialysis"].eq(0)])]
    for j, (name, frame) in enumerate(cases):
        d = frame.reset_index(drop=True)
        if len(d) < MIN_SENSITIVITY_N or d["event"].sum() < MIN_SENSITIVITY_EVENTS:
            skipped.append({"Landmark": landmark, "Analysis": name, "N": len(d), "Events": int(d["event"].sum()),
                            "Reason": "Insufficient sample size/events"})
            continue
        s = seed+10000*(j+1)
        pair = cached_stage(f"sensitivity_models_{j}", landmark, d, s,
            lambda d=d, s=s, name=name: _fit_other_sensitivity(d, landmark, s, name), extra={"analysis": name})
        boot = fixed_prediction_bootstrap_ci(d, pair["discharge_pred"], pair["current_pred"],
                                             SENSITIVITY_BOOTSTRAP_N, s+5000, f"{landmark}_sensitivity_{j}")
        rows.append(pair_ci_table(d, pair["discharge_pred"], pair["current_pred"], landmark, boot,
                                  context={"Analysis": name}))
    return {"results": pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), "skipped": pd.DataFrame(skipped)}


# %% [markdown]
# ## Table assembly functions: no model fitting

# %%
def format_continuous(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    q1, med, q3 = np.quantile(x, [0.25, 0.50, 0.75])
    return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"


def format_binary(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    n = int((x == 1).sum())
    return f"{n} ({100*n/len(x):.1f}%)"


def make_table1(day30: pd.DataFrame, day90: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("Age, years", "continuous", "age"),
        ("Male sex", "binary", "male"),
        ("BMI, kg/m2", "continuous", "bmi"),
        ("Days from discharge to index outpatient visit", "continuous", "visit_day"),
        ("Days from discharge to index outpatient visit", "continuous", "visit_day"),
        ("LVEF <50%", "binary", "ref"),
        ("Atrial fibrillation", "binary", "af"),
        ("Diabetes mellitus", "binary", "dm"),
        ("Dialysis", "binary", "dialysis"),
        ("SGLT2 inhibitor", "binary", "sglt2"),
        ("Beta-blocker", "binary", "beta_blocker"),
        ("RAS inhibitor/ARNI", "binary", "ras_arni"),
        ("MRA", "binary", "mra"),
        ("Loop diuretic", "binary", "loop_diuretic"),
        ("Discharge albumin", "continuous", "discharge_albumin"),
        ("Current albumin", "continuous", "current_albumin"),
        ("Discharge BUN", "continuous", "discharge_bun"),
        ("Current BUN", "continuous", "current_bun"),
        ("Discharge hemoglobin", "continuous", "discharge_hemoglobin"),
        ("Current hemoglobin", "continuous", "current_hemoglobin"),
        ("Discharge potassium", "continuous", "discharge_potassium"),
        ("Current potassium", "continuous", "current_potassium"),
        ("Discharge sodium", "continuous", "discharge_sodium"),
        ("Current sodium", "continuous", "current_sodium"),
        ("Discharge eGFR", "continuous", "discharge_egfr"),
        ("Current eGFR", "continuous", "current_egfr"),
        ("Discharge NT-proBNP", "continuous", "discharge_ntprobnp"),
        ("Current NT-proBNP", "continuous", "current_ntprobnp"),
    ]

    rows = []
    for label, typ, col in specs:
        formatter = format_binary if typ == "binary" else format_continuous
        rows.append({
            "Variable": label,
            "Day 30": formatter(day30[col]),
            "Day 90": formatter(day90[col]),
        })

    return pd.DataFrame(rows)


def make_missingness_table(
    cohorts: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    cols = primary_required_model_columns()

    for landmark, d in cohorts.items():
        for c in cols:
            rows.append({
                "Landmark": landmark,
                "Predictor": c,
                "Missing n": int(d[c].isna().sum()),
                "Missing %": 100.0 * d[c].isna().mean(),
            })

    return pd.DataFrame(rows)


def make_predictor_definition_table() -> pd.DataFrame:
    rows = [
        ["age", "Age at index hospitalization", "Index hospitalization", "Continuous"],
        ["male", "Male sex", "Index hospitalization", "Binary"],
        ["bmi", "Body mass index", "Index hospitalization", "Continuous"],
        ["visit_day", "Days from discharge to the index outpatient visit", "Index outpatient visit", "Continuous; prespecified linear term in both models"],
        ["visit_day", "Days from discharge to the index outpatient visit", "Index outpatient visit", "Continuous; prespecified linear term in both models"],
        ["ref", "LVEF <50%", "LVEF source column from the index admission", "Binary"],
        ["af", "Atrial fibrillation", "Index hospitalization", "Binary"],
        ["dm", "Diabetes mellitus", "Index hospitalization", "Binary"],
        ["dialysis", "Maintenance hemodialysis or peritoneal dialysis", "Index hospitalization/discharge", "Binary"],
        ["sglt2", "SGLT2 inhibitor prescribed at discharge", "Discharge prescription", "Binary"],
        ["beta_blocker", "Beta-blocker prescribed at discharge", "Discharge prescription", "Binary"],
        ["ras_arni", "ACE inhibitor/ARB/ARNI prescribed at discharge", "Discharge prescription", "Binary"],
        ["mra", "Mineralocorticoid receptor antagonist prescribed at discharge", "Discharge prescription", "Binary"],
        ["loop_diuretic", "Loop diuretic prescribed at discharge", "Discharge prescription", "Binary"],
    ]

    for lab in [
        "albumin",
        "bun",
        "hemoglobin",
        "potassium",
        "sodium",
        "egfr",
        "ntprobnp_log2",
    ]:
        rows.append([
            f"discharge_{lab}",
            f"Discharge {LAB_LABELS[lab]} from the supplied dis_* source column",
            "Electronic medical record",
            "Continuous",
        ])
        rows.append([
            f"current_{lab}",
            f"Latest outpatient {LAB_LABELS[lab]} within the 14-day look-back window ending on the index outpatient visit",
            "Electronic medical record",
            "Continuous",
        ])

    return pd.DataFrame(
        rows,
        columns=["Predictor", "Definition", "Data source/timing", "Coding"],
    )


def make_table_s1(
    day30: pd.DataFrame,
    day90: pd.DataFrame,
) -> pd.DataFrame:
    if SOURCE_INTERNAL is None:
        raise RuntimeError("SOURCE_INTERNAL is not available.")

    src = SOURCE_INTERNAL.copy()
    ids30 = set(day30["id"].astype(str))
    ids90 = set(day90["id"].astype(str))

    src["Day30_status"] = np.where(
        src["id"].isin(ids30), "Included", "Not included"
    )
    src["Day90_status"] = np.where(
        src["id"].isin(ids90), "Included", "Not included"
    )

    specs = [
        ("Age, years", "continuous", "age"),
        ("Male sex", "binary", "male"),
        ("BMI, kg/m2", "continuous", "bmi"),
        ("LVEF <50%", "binary", "ref"),
        ("Atrial fibrillation", "binary", "af"),
        ("Diabetes mellitus", "binary", "dm"),
        ("Dialysis", "binary", "dialysis"),
        ("SGLT2 inhibitor", "binary", "sglt2"),
        ("Beta-blocker", "binary", "beta_blocker"),
        ("RAS inhibitor/ARNI", "binary", "ras_arni"),
        ("MRA", "binary", "mra"),
        ("Loop diuretic", "binary", "loop_diuretic"),
        ("Discharge albumin", "continuous", "discharge_albumin"),
        ("Discharge BUN", "continuous", "discharge_bun"),
        ("Discharge hemoglobin", "continuous", "discharge_hemoglobin"),
        ("Discharge potassium", "continuous", "discharge_potassium"),
        ("Discharge sodium", "continuous", "discharge_sodium"),
        ("Discharge eGFR", "continuous", "discharge_egfr"),
        ("Discharge NT-proBNP", "continuous", "discharge_ntprobnp"),
    ]

    def summarize(x: pd.Series, typ: str) -> str:
        return format_binary(x) if typ == "binary" else format_continuous(x)

    rows = []
    for label, typ, col in specs:
        row = {"Variable": label}
        for landmark in ["Day30", "Day90"]:
            status_col = f"{landmark}_status"
            for status in ["Included", "Not included"]:
                subset = src.loc[src[status_col] == status, col]
                row[f"{landmark} {status}"] = summarize(subset, typ)
        rows.append(row)

    return pd.DataFrame(rows)

def make_table2(primary_results, bootstrap_results, primary_ci):
    rows = []
    for landmark in ("Day30", "Day90"):
        p, b = primary_results[landmark], bootstrap_results[landmark]
        for model, rep in [("Discharge", "discharge"), ("Current", "current")]:
            for metric in METRICS:
                if metric in ("Calibration intercept", "Calibration slope"):
                    lo, hi, n = ci_summary(primary_ci[landmark], f"{model}_{metric}")
                    target = "Apparent calibration; conditional patient-sampling CI with predictions fixed"
                else:
                    lo, hi, n = ci_summary(b["bootstrap_records"], f"{model}_{metric}_train")
                    target = "Apparent performance; percentile CI from reduced-MI redevelopment bootstrap"
                r = b["corrected"].loc[lambda x: x["Model"].eq(model) & x["Metric"].eq(metric)].iloc[0]
                rows.append({"Landmark": landmark, "Model": model, "Metric": metric,
                    "Apparent estimate": p[f"{rep}_metrics"][metric], "95% CI lower": lo, "95% CI upper": hi,
                    "Optimism-corrected estimate": r["Optimism-corrected"], "Valid bootstrap N": n,
                    "CI target": target})
        for _, r in b["delta_ci"].iterrows():
            rows.append({"Landmark": landmark, "Model": "Current vs Discharge improvement", "Metric": r["Metric"],
                "Apparent estimate": r["Point estimate"], "95% CI lower": r["95% CI lower"],
                "95% CI upper": r["95% CI upper"], "Optimism-corrected estimate": r["Optimism-corrected difference"],
                "Valid bootstrap N": r["Valid bootstrap N"], "CI target": r["CI target"]})
    return pd.DataFrame(rows)


def validate_ci_table(df, name):
    if df.empty:
        return
    required = ["95% CI lower", "95% CI upper"]
    if not set(required).issubset(df.columns):
        raise KeyError(f"{name}: CI schema mismatch; missing {required}.")
    x = df[required].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    bad = ~np.isfinite(x).all(axis=1) | (x[:,0] > x[:,1])
    if bad.any():
        export_excel({"Invalid CI rows": df.loc[bad]}, ARTIFACT_DIR / f"{name.replace(' ', '_')}_CI_error.xlsx")
        raise ValueError(f"{name}: {int(bad.sum())} invalid CI rows. No recalculation occurs in export cells.")


def analysis_settings_table():
    settings = _settings_signature()
    settings.update({"Time zero": "first outpatient visit in days 15-45 / 75-105",
        "Later cohort": "separately defined from Early cohort; patients may contribute to both; first outpatient visit in days 75-105",
        "Current laboratory cutoff": "14-day look-back window ending on the index outpatient visit",
        "Early window": "15-45 days inclusive", "Later window": "75-105 days inclusive",
        "Follow-up": "(index outpatient visit, index outpatient visit + 180 days]",
        "Known first event extends LastDate": KNOWN_FIRST_EVENT_EXTENDS_LASTDATE,
        "Internal bootstrap": INTERNAL_BOOTSTRAP_N, "Paired bootstrap (same bank)": PAIRED_BOOTSTRAP_N,
        "Bootstrap MI": BOOTSTRAP_MICE_M, "Bootstrap cycles": BOOTSTRAP_MICE_ITERATIONS,
        "Bootstrap nonlinearity MI": BOOTSTRAP_NONLINEARITY_MI_DATASETS,
        "Bootstrap form reselection": RESELECT_FORMS_IN_BOOTSTRAP,
        "Primary conditional CI bootstrap": PRIMARY_FIXED_CI_BOOTSTRAP_N,
        "IECV CI bootstrap": IECV_CI_BOOTSTRAP_N, "Subgroup bootstrap": SUBGROUP_BOOTSTRAP_N,
        "Sensitivity bootstrap": SENSITIVITY_BOOTSTRAP_N,
        "Change formula": "(current - discharge) * 30 / max(measurement days since discharge, 1); log2 difference for NT-proBNP",
        "IPCW censoring model": "Marginal KM in evaluation population; re-estimated in fixed-prediction resamples",
        "IBS integration": "Trapezoidal average over days 1-180",
        "Primary coefficient penalty": "None", "Cox tuning CV": "None",
        "Source parquet": str(PARQUET_PATH), "Python": __import__('sys').version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__, "statsmodels": sm.__version__})
    return pd.DataFrame({"Parameter": list(settings), "Value": [str(v) for v in settings.values()]})


def table_notes():
    return pd.DataFrame([
        ["Cohorts", "Early: first outpatient visit on days 15-45 after discharge. Later: first outpatient visit on days 75-105. Patients may contribute to both cohorts."],
        ["Time zero", "The actual selected outpatient visit date. Same-day and earlier composite events are excluded."],
        ["Endpoint", "First recorded emergency readmission OR all-cause death strictly after the index outpatient visit and within 180 days."],
        ["Current laboratory values", "Latest laboratory value within the 14-day look-back window ending on the index outpatient visit. Same-day laboratory tests are allowed."],
        ["Visit timing", "Days from discharge to the index outpatient visit is included as a common prespecified linear predictor in both Discharge and Current models."],
        ["Follow-up", "Known-first-event extension of LastDate is explicit. It assumes both composite components are ascertainable through that recorded event. Otherwise set KNOWN_FIRST_EVENT_EXTENDS_LASTDATE=False before fitting."],
        ["Table 1", "Observed data: continuous variables median [IQR]; binary variables n (% of observed). Missingness is in Table S3."],
        ["Table 2", "95% CI columns target APPARENT estimates, not the adjacent optimism-corrected point estimates. Corrected-estimand CIs are not estimated or fabricated."],
        ["Primary CI scope", "Redevelopment uses the explicitly reduced MI numerical settings. Calibration CI keeps fitted predictions fixed and resamples patients; it excludes model-development uncertainty."],
        ["Table S4", "Three-knot RCS (10th/50th/90th percentiles); common functional form for both representations when either nonlinear component test meets P<0.05. Visit day is prespecified linear."],
        ["Table S5", "Optimism and Monte Carlo SE are not confidence intervals. The bootstrap repeats functional-form selection by default, with the stated reduced MI computation profile."],
        ["Table S6", "IECV point estimates use held-out-center predictions. CI conditions on each training model; only held-out patients are resampled."],
        ["Table S7", "Exploratory evaluation of primary fitted predictions by subgroup; no subgroup model refit and no interaction test."],
        ["Table S8-S9", "Conditional paired patient-bootstrap CI, not full model-redevelopment CI. Results are apparent in each analysis sample and may be optimistic."],
        ["Delta sign", "Current minus Discharge for C-index/AUC; Discharge minus Current for Brier/IBS. All positive improvements favor the updated model."],
        ["MI", "Baseline/discharge completion does not use outpatient values or outcomes. Current values are completed after the shared baseline block is frozen."],
        ["Change rates", "Longitudinal change is standardized to 30 days using the actual laboratory measurement day since discharge."],
        ["Repeated admissions", "INDEX is the analysis unit. Early and Later models are fitted separately."],
    ], columns=["Scope", "Note"])

def export_table_sources(day30, day90, primary, bootstrap, primary_ci, iecv, subgroup, longitudinal, sensitivity_other):
    # This function only constructs tables and writes files. No fits, MICE or resampling.
    tables = {"Table1": make_table1(day30, day90).rename(columns={"Day 30": f"Early (Day 15-45; N={len(day30):,}; events={int(day30['event'].sum()):,})", "Day 90": f"Later (Day 75-105; N={len(day90):,}; events={int(day90['event'].sum()):,})"}), "Table2": make_table2(primary, bootstrap, primary_ci),
        "TableS1": make_table_s1(day30, day90), "TableS2": make_predictor_definition_table(),
        "TableS3": make_missingness_table({"Day30": day30, "Day90": day90}),
        "TableS4": pd.concat([primary[x]["nonlinearity"] for x in ("Day30", "Day90")], ignore_index=True),
        "TableS5": pd.concat([bootstrap[x]["corrected"] for x in ("Day30", "Day90")], ignore_index=True),
        "TableS6": pd.concat([iecv[x]["results"] for x in ("Day30", "Day90")], ignore_index=True),
        "TableS7": subgroup.copy(), "TableS8": longitudinal.copy(), "TableS9": sensitivity_other.copy()}
    for name in ("Table2", "TableS6", "TableS7", "TableS8", "TableS9"):
        validate_ci_table(tables[name], name)
    tables["Figure1_source_table"] = FLOW_TABLE.copy()
    tables["Lab_timing_audit"] = LAB_TIMING_AUDIT.copy()
    tables["Settings"] = analysis_settings_table()
    tables["Notes"] = table_notes()
    for name, df in tables.items():
        df.to_pickle(ARTIFACT_DIR / f"{name}.pkl")
    export_excel(tables, RESULT_XLSX)
    provenance = {"analysis_version": ANALYSIS_VERSION, "schema_version": SCHEMA_VERSION,
        "source_digest": _frame_digest(SOURCE_DF), "settings": _settings_signature(),
        "outcome": OUTCOME_MODE, "time_zero": "fixed discharge+30/90", "tables": list(tables),
        "completed": True}
    (ARTIFACT_DIR / "run_provenance.json").write_text(json.dumps(_json_safe(provenance), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Export complete (no analysis rerun): {RESULT_XLSX}")
    return tables


# %% [markdown]
# ## 1. Load and audit visit-triggered cohorts

# %%
# 1. Source data, visit-triggered landmarks, and timing audit. No model fitting.
# Mark the run as unfinished so Notebook 02 cannot silently read old outputs.
(ARTIFACT_DIR / "run_provenance.json").write_text(
    json.dumps({"analysis_version": ANALYSIS_VERSION, "completed": False}, indent=2), encoding="utf-8")
day30, day90 = load_landmark_inputs()
cohorts = {"Day30": day30, "Day90": day90}
seed_map = {"Day30": MASTER_SEED + 3000, "Day90": MASTER_SEED + 9000}
export_excel({"Flow": FLOW_TABLE, "Lab timing": LAB_TIMING_AUDIT,
              "Patient timing": COHORT_AUDIT[["id", "center", "Landmark", "prediction_date", "index_outpatient_date",
                                                "visit_day", "time", "event", "event_component"]]},
             ARTIFACT_DIR / "Landmark_design_audit.xlsx")
export_excel({"Figure 1 flow": FLOW_TABLE}, OUTPUT_DIR / "Figure1_source_table.xlsx")
display(FLOW_TABLE)
display(LAB_TIMING_AUDIT)


# %% [markdown]
# ## 2. Primary development and immediate saving

# %%
# 2. Primary development; save each completed landmark immediately.
primary = {}
for landmark, d in cohorts.items():
    primary[landmark] = reuse_or_fit_primary(d, landmark, seed_map[landmark])
    save_primary_outputs(primary[landmark], landmark)
print("Primary models, imputation states, parameters, predictions and diagnostics saved.")


# %% [markdown]
# ## 3. Internal validation

# %%
# 3. Internal validation. Completed bootstrap replicates are checkpointed.
bootstrap = {}
for landmark in ("Day30", "Day90"):
    bootstrap[landmark] = run_primary_bootstrap(primary[landmark], landmark, seed_map[landmark] + 500000)
    for name, tbl in bootstrap[landmark].items():
        tbl.to_pickle(ARTIFACT_DIR / f"{landmark}_{name}.pkl")
    export_excel(bootstrap[landmark], ARTIFACT_DIR / f"{landmark}_internal_validation.xlsx")
print("Internal validation completed.")


# %% [markdown]
# ## 4. Conditional calibration/performance CIs

# %%
# 4. Conditional sampling CIs, including overall calibration intercept and slope.
# No primary Cox fitting, no MICE, and no tuning in this cell.
primary_apparent_ci = {}
for landmark in ("Day30", "Day90"):
    p = primary[landmark]
    primary_apparent_ci[landmark] = fixed_prediction_bootstrap_ci(
        p["raw"], p["discharge_pred"], p["current_pred"], PRIMARY_FIXED_CI_BOOTSTRAP_N,
        seed_map[landmark] + 700000, f"{landmark}_primary")
    primary_apparent_ci[landmark].to_pickle(ARTIFACT_DIR / f"{landmark}_primary_conditional_CI_replicates.pkl")
print("Primary conditional sampling CIs saved.")


# %% [markdown]
# ## 5. Internal-external validation

# %%
# 5. Internal-external validation. Training model and CI caches are separate.
iecv = {}
for landmark, d in cohorts.items():
    iecv[landmark] = run_iecv(d, landmark, seed_map[landmark] + 1000000)
    for name, frame in iecv[landmark].items():
        frame.to_pickle(ARTIFACT_DIR / f"{landmark}_IECV_{name}.pkl")
    export_excel(iecv[landmark], ARTIFACT_DIR / f"{landmark}_IECV.xlsx")
print("IECV completed.")


# %% [markdown]
# ## 6. Subgroup evaluation

# %%
# 6. Exploratory subgroup evaluations; primary predictions are fixed.
subgroup_parts, subgroup_skipped = [], []
if RUN_SUBGROUPS:
    for landmark in ("Day30", "Day90"):
        r = run_subgroups(primary[landmark], landmark, seed_map[landmark] + 1500000)
        subgroup_parts.append(r["results"])
        subgroup_skipped.append(r["skipped"])
subgroup = pd.concat(subgroup_parts, ignore_index=True) if subgroup_parts else pd.DataFrame()
subgroup.to_pickle(ARTIFACT_DIR / "subgroup_results.pkl")
export_excel({"Subgroups": subgroup, "Skipped": pd.concat(subgroup_skipped, ignore_index=True)
              if subgroup_skipped else pd.DataFrame()}, ARTIFACT_DIR / "Subgroup_results.xlsx")


# %% [markdown]
# ## 7. Sensitivity analyses and CIs

# %%
# 7. Sensitivity models and CIs are computed here, never in the table-export cell.
longitudinal_parts, other_parts, skipped_parts = [], [], []
for landmark, d in cohorts.items():
    if RUN_LONGITUDINAL_SENSITIVITY:
        longitudinal_parts.append(sensitivity_longitudinal(primary[landmark], landmark, seed_map[landmark] + 2000000))
    if RUN_OTHER_SENSITIVITY:
        r = sensitivity_missing_and_dialysis(d, landmark, seed_map[landmark] + 2500000)
        other_parts.append(r["results"])
        skipped_parts.append(r["skipped"])
longitudinal = pd.concat(longitudinal_parts, ignore_index=True) if longitudinal_parts else pd.DataFrame()
sensitivity_other = pd.concat(other_parts, ignore_index=True) if other_parts else pd.DataFrame()
longitudinal.to_pickle(ARTIFACT_DIR / "longitudinal_change_sensitivity.pkl")
sensitivity_other.to_pickle(ARTIFACT_DIR / "other_sensitivity_results.pkl")
export_excel({"Longitudinal": longitudinal, "Other sensitivity": sensitivity_other,
              "Skipped": pd.concat(skipped_parts, ignore_index=True) if skipped_parts else pd.DataFrame()},
             ARTIFACT_DIR / "Sensitivity_results.xlsx")
print("Sensitivity analyses and confidence intervals saved.")


# %% [markdown]
# ## 8. Final source-table export (no analyses)

# %%
# 8. CREATE AND SAVE TABLE SOURCES: no model fitting, MICE or resampling.
table_sources = export_table_sources(day30, day90, primary, bootstrap, primary_apparent_ci,
                                    iecv, subgroup, longitudinal, sensitivity_other)
print("01 completed. Run Notebook 02 to format tables and figures.")


# %% [markdown]
# ## Implementation references and reporting notes
# 
# - Standard Cox proportional hazards regression is fitted with `statsmodels.PHReg`.
# - No L2 penalty and no tuning cross-validation are used.
# - Early and Later prediction problems are modeled separately; patients may contribute to both.
# - Time zero is the actual first outpatient visit in the prespecified window.
# - Current laboratory values are limited to the 14-day look-back period ending on that visit.
# - `visit_day` is included in both models as a common prespecified linear predictor.
# - Internal validation uses bootstrap redevelopment. The reduced bootstrap MI profile is an explicit computational approximation and is reported in Settings.
# - Apparent calibration intercept/slope CIs use patient resampling with fitted predictions held fixed and are not presented as optimism-corrected CIs.
# 


