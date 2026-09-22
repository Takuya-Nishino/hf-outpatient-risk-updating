# %% [markdown]
# # HF outpatient-window Cox v9: tables and figures
# 
# Use only after Notebook 01 v9 completes.
# 
# - Early: first outpatient visit on days 15-45.
# - Later: first outpatient visit on days 75-105.
# - Time zero is the selected outpatient visit date.
# - No model fitting, MICE, bootstrap, or CI calculation occurs here.
# 

# %%
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ANALYSIS_VERSION = "v9_visit_window_first_outpatient_composite"
DATA_DIR = Path(os.environ.get("HF_STUDY_DATA_DIR", ".")).expanduser().resolve()
OUTPUT_DIR = DATA_DIR / "HF_visit_window_Cox_FULL_v9_composite"
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
FINAL_XLSX = OUTPUT_DIR / "HF_visit_window_Cox_FULL_v9_tables_figures.xlsx"
for p in (FIGURE_DIR, TABLE_DIR):
    p.mkdir(parents=True, exist_ok=True)
CENTER_DISPLAY = {"付属": "Nippon Medical School Hospital", "北総": "Chiba Hokusoh Hospital",
                  "小杉": "Musashi-Kosugi Hospital", "永山": "Tama Nagayama Hospital"}
plt.rcParams["font.family"] = ["Times New Roman", "Meiryo"]
plt.rcParams["axes.unicode_minus"] = False
FIGURE_DPI = 600

provenance_path = ARTIFACT_DIR / "run_provenance.json"
if not provenance_path.is_file():
    raise FileNotFoundError(f"01 has not completed its table export: {provenance_path}")
provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
if provenance.get("analysis_version") != ANALYSIS_VERSION or not provenance.get("completed"):
    raise ValueError("Artifact version mismatch. Use outputs from Notebook 01 v9.")


def read_source(name):
    path = ARTIFACT_DIR / f"{name}.pkl"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = pd.read_pickle(path)  # trusted artifacts generated locally by Notebook 01
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"Not a table: {path}")
    return result


def require_columns(df, columns, name):
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"{name}: missing columns {sorted(missing)}; actual={df.columns.tolist()}")


def save_figure(fig, name):
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(path)
    return path


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


# %%
def plot_calibration_panel(ax, d, title, lim=None):
    require_columns(d, ["Model", "mean_predicted_risk", "observed_risk"], title)
    if d.empty:
        raise ValueError(f"No calibration data: {title}")
    if lim is None:
        x = d[["mean_predicted_risk", "observed_risk"]].to_numpy(float)
        lim = min(1.0, max(0.05, float(np.nanmax(x))) * 1.15)
    # Plot models before the ideal line to keep the default model-color order consistent.
    for model, marker in [("Discharge", "o"), ("Current", "s")]:
        g = d.loc[d["Model"].eq(model)].sort_values("mean_predicted_risk")
        ax.plot(g["mean_predicted_risk"], g["observed_risk"], marker=marker, linewidth=1.4, label=model)
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1, label="Ideal")
    ax.set(xlim=(0,lim), ylim=(0,lim), xlabel="Predicted 180-day risk",
           ylabel="Observed 180-day risk", title=title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)


def plot_iecv_panel(ax, df, landmark, metric):
    require_columns(df, ["Landmark", "Validation center", "Model", "Metric", "Estimate", "95% CI lower", "95% CI upper"], "IECV")
    d = df.loc[df["Landmark"].eq(landmark) & df["Metric"].eq(metric) & df["Model"].isin(["Discharge", "Current"])]
    centers = d["Validation center"].drop_duplicates().tolist()
    for j, (model, marker) in enumerate([("Discharge", "o"), ("Current", "s")]):
        g = d.loc[d["Model"].eq(model)].set_index("Validation center").reindex(centers)
        y = np.arange(len(centers)) + (-.12 if j == 0 else .12)
        # Segments + point avoid invalid negative xerr when a percentile CI does not contain the estimate.
        points, = ax.plot(g["Estimate"], y, marker=marker, linestyle="none", label=model)
        for yy, lo, hi in zip(y, g["95% CI lower"], g["95% CI upper"]):
            ax.plot([lo, hi], [yy, yy], linewidth=1.2, linestyle="-", c=points.get_color())
    ax.set_yticks(np.arange(len(centers)), [CENTER_DISPLAY.get(x, str(x)) for x in centers])
    ax.invert_yaxis()
    ax.set_xlabel("C-index" if metric == "C-index" else "180-day AUC")
    ax.set_title(f"{landmark.replace('Day', 'Day ')}: {metric}")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=.2)


# %% [markdown]
# ## Supplementary table helpers
# 
# The following output-only helpers add the manuscript-facing supplementary tables that are not directly exported by Notebook 01:
# 
# - **Table S1:** predictor definitions and ascertainment windows.
# - **Table S2:** comparison of included and non-included patients using absolute standardized mean differences (SMDs).
# - **Table S5:** pooled primary-model coefficients and hazard ratios where a single hazard ratio is interpretable.
# 
# No Cox model, imputation model, bootstrap, or confidence interval is recalculated here.

# %%

# ============================================================
# Additional manuscript-facing supplementary tables
# ============================================================

PARAMETER_DIR = OUTPUT_DIR / "model_parameters"
SOURCE_PARQUET = DATA_DIR / "df_LR.parquet"


# ------------------------------------------------------------
# Table S2: cohort selection comparison with absolute SMD
# ------------------------------------------------------------

def _format_continuous_for_table(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    q1, med, q3 = np.quantile(x, [0.25, 0.50, 0.75])
    return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"


def _format_binary_for_table(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return ""
    n = int((x == 1).sum())
    return f"{n} ({100.0*n/len(x):.1f}%)"


def _absolute_smd_continuous(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    denom = np.sqrt((va + vb) / 2.0)

    if not np.isfinite(denom) or denom <= 0:
        return 0.0 if np.isclose(np.mean(a), np.mean(b)) else np.nan

    return float(abs(np.mean(a) - np.mean(b)) / denom)


def _absolute_smd_binary(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()

    if len(a) == 0 or len(b) == 0:
        return np.nan

    pa = float((a == 1).mean())
    pb = float((b == 1).mean())

    denom = np.sqrt(
        (
            pa * (1.0 - pa)
            + pb * (1.0 - pb)
        ) / 2.0
    )

    if not np.isfinite(denom) or denom <= 0:
        return 0.0 if np.isclose(pa, pb) else np.nan

    return float(abs(pa - pb) / denom)


def _source_comparison_frame():
    if not SOURCE_PARQUET.is_file():
        raise FileNotFoundError(
            f"Source parquet not found: {SOURCE_PARQUET}"
        )

    d = pd.read_parquet(SOURCE_PARQUET).copy()

    if d["INDEX"].duplicated().any():
        raise ValueError("df_LR.parquet contains duplicated INDEX values.")

    d["INDEX"] = d["INDEX"].astype("string")

    # Derived variables used in the prediction models.
    d["__ref"] = np.where(
        pd.to_numeric(d["LVEF"], errors="coerce").notna(),
        (
            pd.to_numeric(d["LVEF"], errors="coerce") < 50
        ).astype(float),
        np.nan,
    )

    hd = pd.to_numeric(d["維持透析"], errors="coerce")
    pdial = pd.to_numeric(d["PD"], errors="coerce")

    d["__dialysis"] = np.where(
        (hd == 1) | (pdial == 1),
        1.0,
        np.where(
            (hd == 0) & (pdial == 0),
            0.0,
            np.nan,
        ),
    )

    return d


def make_selection_smd_table():
    """
    Compare each final prediction cohort with patients from the
    post-discharge source cohort who were not included in that cohort.

    Continuous-variable SMDs use differences in means divided by
    the square root of the average group variance.
    Binary-variable SMDs use the corresponding pooled Bernoulli variance.
    Absolute SMDs are reported; no P values are calculated.
    """

    src = _source_comparison_frame()

    pred30_path = ARTIFACT_DIR / "Day30_primary_predictions.parquet"
    pred90_path = ARTIFACT_DIR / "Day90_primary_predictions.parquet"

    if not pred30_path.is_file() or not pred90_path.is_file():
        raise FileNotFoundError(
            "Primary prediction parquet files are required for Table S2."
        )

    ids30 = set(
        pd.read_parquet(pred30_path)["id"].astype("string")
    )
    ids90 = set(
        pd.read_parquet(pred90_path)["id"].astype("string")
    )

    specs = [
        ("Age, years", "continuous", "Age"),
        ("Male sex", "binary", "Male"),
        ("BMI, kg/m2", "continuous", "BMI"),
        ("LVEF <50%", "binary", "__ref"),
        ("Atrial fibrillation", "binary", "af"),
        ("Diabetes mellitus", "binary", "DM"),
        ("Dialysis", "binary", "__dialysis"),
        ("SGLT2 inhibitor", "binary", "SGLT2"),
        ("Beta-blocker", "binary", "β遮断薬"),
        ("RAS inhibitor/ARNI", "binary", "RAS_ARNI"),
        ("MRA", "binary", "MRA_内服"),
        ("Loop diuretic", "binary", "Loop_内服"),
        ("Discharge albumin", "continuous", "dis_Alb"),
        ("Discharge BUN", "continuous", "dis_BUN"),
        ("Discharge hemoglobin", "continuous", "dis_Hb"),
        ("Discharge potassium", "continuous", "dis_K"),
        ("Discharge sodium", "continuous", "dis_Na"),
        ("Discharge eGFR", "continuous", "dis_eGFR"),
        ("Discharge NT-proBNP", "continuous", "dis_NT-proBNP"),
    ]

    cohort_specs = [
        ("Early", ids30),
        ("Later", ids90),
    ]

    rows = []

    for label, typ, col in specs:
        row = {"Variable": label}

        for cohort_label, included_ids in cohort_specs:
            included = src.loc[
                src["INDEX"].isin(included_ids),
                col,
            ]
            not_included = src.loc[
                ~src["INDEX"].isin(included_ids),
                col,
            ]

            formatter = (
                _format_binary_for_table
                if typ == "binary"
                else _format_continuous_for_table
            )

            smd = (
                _absolute_smd_binary(included, not_included)
                if typ == "binary"
                else _absolute_smd_continuous(included, not_included)
            )

            row[f"{cohort_label} included"] = formatter(included)
            row[f"{cohort_label} not included"] = formatter(not_included)
            row[f"{cohort_label} absolute SMD"] = smd

        rows.append(row)

    out = pd.DataFrame(rows)

    out.attrs["note"] = (
        "Continuous variables are median [IQR] and binary variables are n (%). "
        "Absolute standardized mean differences are descriptive; no P values "
        "are calculated. Not-included patients comprise all post-discharge "
        "source-cohort patients who did not enter the corresponding final "
        "prediction cohort."
    )

    return out


# ------------------------------------------------------------
# Table S5: pooled model coefficients and hazard ratios
# ------------------------------------------------------------

HR_LABELS = {
    "age": ("Age", 10.0, "per 10-year increase"),
    "bmi": ("BMI", 5.0, "per 5-kg/m2 increase"),
    "visit_day": (
        "Days from discharge to index outpatient visit",
        10.0,
        "per 10-day increase",
    ),
    "albumin": ("Albumin", 1.0, "per 1-g/dL increase"),
    "bun": ("BUN", 10.0, "per 10-mg/dL increase"),
    "hemoglobin": ("Hemoglobin", 1.0, "per 1-g/dL increase"),
    "potassium": ("Potassium", 1.0, "per 1-mEq/L increase"),
    "sodium": ("Sodium", 1.0, "per 1-mEq/L increase"),
    "egfr": ("eGFR", 10.0, "per 10-mL/min/1.73 m2 increase"),
    "ntprobnp_log2": ("NT-proBNP", 1.0, "per doubling"),
    "male": ("Male sex", 1.0, "male vs female"),
    "ref": ("LVEF <50%", 1.0, "<50% vs >=50%"),
    "af": ("Atrial fibrillation", 1.0, "yes vs no"),
    "dm": ("Diabetes mellitus", 1.0, "yes vs no"),
    "dialysis": ("Dialysis", 1.0, "yes vs no"),
    "sglt2": ("SGLT2 inhibitor", 1.0, "use vs non-use"),
    "beta_blocker": ("Beta-blocker", 1.0, "use vs non-use"),
    "ras_arni": ("RAS inhibitor/ARNI", 1.0, "use vs non-use"),
    "mra": ("MRA", 1.0, "use vs non-use"),
    "loop_diuretic": ("Loop diuretic", 1.0, "use vs non-use"),
}

CONTINUOUS_MODEL_CONCEPTS = {
    "age",
    "bmi",
    "albumin",
    "bun",
    "hemoglobin",
    "potassium",
    "sodium",
    "egfr",
    "ntprobnp_log2",
}


def _parameter_workbook(landmark):
    path = PARAMETER_DIR / f"{landmark}_model_parameters.xlsx"
    if not path.is_file():
        raise FileNotFoundError(
            f"Model parameter workbook not found: {path}"
        )
    return path


def _model_form_map(forms_df, model_name):
    d = forms_df.loc[
        forms_df["Model"].astype(str).eq(model_name)
    ].copy()

    form_map = {}

    if d.empty:
        return form_map

    for predictor, g in d.groupby("Predictor", sort=False):
        vals = (
            g["Form"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        if len(vals) > 1:
            raise ValueError(
                f"Inconsistent functional form for {model_name}/{predictor}: "
                f"{vals}"
            )
        if vals:
            form_map[str(predictor)] = vals[0]

    return form_map


def make_model_hr_table():
    """
    Create a manuscript-facing pooled coefficient / HR table.

    For predictors modeled with restricted cubic splines, a single HR is not
    meaningful. Those predictors are retained as one row with HR fields left
    blank and are linked to Table S4 for the spline specification.

    For linear continuous predictors, HRs are reported for prespecified
    clinically readable increments. Binary predictors are reported as the
    modeled contrast.
    """

    rows = []

    for landmark in ["Day30", "Day90"]:
        path = _parameter_workbook(landmark)

        pooled = pd.read_excel(
            path,
            sheet_name="Pooled coefficients",
        )

        forms = pd.read_excel(
            path,
            sheet_name="Functional forms",
        )

        required = {
            "Landmark",
            "Model",
            "Term",
            "Coefficient",
            "Standard error",
            "95% CI lower",
            "95% CI upper",
            "Status",
        }
        missing = required - set(pooled.columns)

        if missing:
            raise KeyError(
                f"{path.name}: missing pooled coefficient columns "
                f"{sorted(missing)}"
            )

        for model_name in ["Discharge", "Current"]:
            dcoef = pooled.loc[
                pooled["Model"].astype(str).eq(model_name)
            ].copy()

            form_map = _model_form_map(
                forms,
                model_name,
            )

            # Model terms correspond to common clinical concepts in both
            # Discharge and Current models.
            base_terms = []
            for term in dcoef["Term"].astype(str):
                base = re.sub(r"_rcs\d+$", "", term)
                if base not in base_terms:
                    base_terms.append(base)

            for base in base_terms:
                label, scale, contrast = HR_LABELS.get(
                    base,
                    (base, 1.0, "per 1-unit increase"),
                )

                form = (
                    form_map.get(base, "linear")
                    if base in CONTINUOUS_MODEL_CONCEPTS
                    else "binary/linear"
                )

                term_rows = dcoef.loc[
                    dcoef["Term"].astype(str).eq(base)
                    | dcoef["Term"].astype(str).str.match(
                        rf"^{re.escape(base)}_rcs\d+$"
                    )
                ].copy()

                if str(form).lower() == "spline":
                    status_values = (
                        term_rows["Status"]
                        .dropna()
                        .astype(str)
                        .drop_duplicates()
                        .tolist()
                    )

                    rows.append({
                        "Landmark": landmark,
                        "Model": model_name,
                        "Predictor": label,
                        "Functional form": "Restricted cubic spline",
                        "Reported contrast": "",
                        "Pooled beta": np.nan,
                        "Standard error": np.nan,
                        "Hazard ratio": np.nan,
                        "HR 95% CI lower": np.nan,
                        "HR 95% CI upper": np.nan,
                        "Status / note": (
                            "No single HR reported for a spline-modeled predictor; "
                            "see Table S4 for knots and saved model-parameter files "
                            "for basis coefficients."
                            + (
                                " " + "; ".join(status_values)
                                if status_values
                                else ""
                            )
                        ),
                    })
                    continue

                base_row = term_rows.loc[
                    term_rows["Term"].astype(str).eq(base)
                ]

                if base_row.empty:
                    rows.append({
                        "Landmark": landmark,
                        "Model": model_name,
                        "Predictor": label,
                        "Functional form": str(form),
                        "Reported contrast": contrast,
                        "Pooled beta": np.nan,
                        "Standard error": np.nan,
                        "Hazard ratio": np.nan,
                        "HR 95% CI lower": np.nan,
                        "HR 95% CI upper": np.nan,
                        "Status / note": "Base model term not available.",
                    })
                    continue

                r = base_row.iloc[0]

                status = str(r["Status"])
                beta = pd.to_numeric(
                    pd.Series([r["Coefficient"]]),
                    errors="coerce",
                ).iloc[0]
                se = pd.to_numeric(
                    pd.Series([r["Standard error"]]),
                    errors="coerce",
                ).iloc[0]
                lo = pd.to_numeric(
                    pd.Series([r["95% CI lower"]]),
                    errors="coerce",
                ).iloc[0]
                hi = pd.to_numeric(
                    pd.Series([r["95% CI upper"]]),
                    errors="coerce",
                ).iloc[0]

                estimable = (
                    np.isfinite(beta)
                    and np.isfinite(lo)
                    and np.isfinite(hi)
                    and not status.startswith("Not pooled")
                )

                if estimable:
                    hr = float(np.exp(beta * scale))
                    hr_lo = float(np.exp(lo * scale))
                    hr_hi = float(np.exp(hi * scale))
                else:
                    hr = hr_lo = hr_hi = np.nan

                rows.append({
                    "Landmark": landmark,
                    "Model": model_name,
                    "Predictor": label,
                    "Functional form": str(form),
                    "Reported contrast": contrast,
                    "Pooled beta": beta,
                    "Standard error": se,
                    "Hazard ratio": hr,
                    "HR 95% CI lower": hr_lo,
                    "HR 95% CI upper": hr_hi,
                    "Status / note": status,
                })

    out = pd.DataFrame(rows)

    landmark_order = pd.Categorical(
        out["Landmark"],
        categories=["Day30", "Day90"],
        ordered=True,
    )
    model_order = pd.Categorical(
        out["Model"],
        categories=["Discharge", "Current"],
        ordered=True,
    )

    out = (
        out.assign(
            _landmark_order=landmark_order,
            _model_order=model_order,
        )
        .sort_values(
            ["_landmark_order", "_model_order"],
            kind="stable",
        )
        .drop(
            columns=["_landmark_order", "_model_order"]
        )
        .reset_index(drop=True)
    )

    return out


# %% [markdown]
# ## Table 1

# %%
# Table 1: use inferential values saved by Notebook 01 without alteration.
Table1 = read_source("Table1")
display(Table1)
export_excel({"Table 1": Table1}, TABLE_DIR / "Table1.xlsx")


# %% [markdown]
# ## Table 2

# %%
# Table 2: use inferential values saved by Notebook 01 without alteration.
Table2 = read_source("Table2")
display(Table2)
export_excel({"Table 2": Table2}, TABLE_DIR / "Table2.xlsx")


# %% [markdown]
# ## Figure 1 flow

# %%
# Figure 1 flow: use inferential values saved by Notebook 01 without alteration.
Figure1_source = read_source("Figure1_source_table")
display(Figure1_source)
export_excel({"Figure 1 flow": Figure1_source}, TABLE_DIR / "Figure1_source_table.xlsx")


# %% [markdown]
# ## Table S1 - Predictor definitions and ascertainment windows

# %%
# Table S1: predictor definitions and ascertainment windows.
TableS1 = read_source("TableS2").copy()
display(TableS1)
export_excel(
    {"Table S1": TableS1},
    TABLE_DIR / "TableS1_predictor_definitions.xlsx",
)


# %% [markdown]
# ## Table S2 - Included versus non-included patients

# %%
# Table S2: included versus non-included patients with absolute SMDs.
TableS2 = make_selection_smd_table()
display(TableS2)
export_excel(
    {"Table S2": TableS2},
    TABLE_DIR / "TableS2_selection_SMD.xlsx",
)


# %% [markdown]
# ## Table S3 - Predictor missingness

# %%
# Table S3: use inferential values saved by Notebook 01 without alteration.
TableS3 = read_source("TableS3")
display(TableS3)
export_excel({"Table S3": TableS3}, TABLE_DIR / "TableS3.xlsx")


# %% [markdown]
# ## Table S4 - Functional-form assessment

# %%
# Table S4: use inferential values saved by Notebook 01 without alteration.
TableS4 = read_source("TableS4")
display(TableS4)
export_excel({"Table S4": TableS4}, TABLE_DIR / "TableS4.xlsx")


# %% [markdown]
# ## Table S5 - Final primary-model coefficients and hazard ratios

# %%
import re
# Table S5: pooled coefficients and interpretable hazard ratios.
TableS5 = make_model_hr_table()
display(TableS5)
export_excel(
    {"Table S5": TableS5},
    TABLE_DIR / "TableS5_model_coefficients_HR.xlsx",
)


# %% [markdown]
# ## Table S6 - Bootstrap internal validation

# %%
# Table S6: bootstrap internal validation.
TableS6 = read_source("TableS5").copy()
display(TableS6)
export_excel(
    {"Table S6": TableS6},
    TABLE_DIR / "TableS6_internal_validation.xlsx",
)


# %% [markdown]
# ## Table S7 - Internal-external validation

# %%
# Table S7: internal-external validation.
TableS7 = read_source("TableS6").copy()
display(TableS7)
export_excel(
    {"Table S7": TableS7},
    TABLE_DIR / "TableS7_IECV.xlsx",
)


# %% [markdown]
# ## Table S8 - Exploratory subgroup analyses

# %%
# Table S8: exploratory subgroup analyses.
TableS8 = read_source("TableS7").copy()
display(TableS8)
export_excel(
    {"Table S8": TableS8},
    TABLE_DIR / "TableS8_subgroup.xlsx",
)


# %% [markdown]
# ## Table S9 - Longitudinal-change sensitivity analysis

# %%
# Table S9: longitudinal-change sensitivity analysis.
TableS9 = read_source("TableS8").copy()
display(TableS9)
export_excel(
    {"Table S9": TableS9},
    TABLE_DIR / "TableS9_longitudinal_change.xlsx",
)


# %% [markdown]
# ## Table S10 - Other sensitivity analyses

# %%
# Table S10: complete-case, single-imputation, and dialysis-exclusion analyses.
TableS10 = read_source("TableS9").copy()
display(TableS10)
export_excel(
    {"Table S10": TableS10},
    TABLE_DIR / "TableS10_other_sensitivity.xlsx",
)


# %% [markdown]
# ## Figure2

# %%
# ============================================================
# Figure 2: Overall apparent calibration
# ============================================================

cal = pd.concat(
    [
        read_source("Day30_primary_calibration"),
        read_source("Day90_primary_calibration"),
    ],
    ignore_index=True,
)

lim = min(
    1.0,
    max(
        0.05,
        float(
            np.nanmax(
                cal[
                    [
                        "mean_predicted_risk",
                        "observed_risk",
                    ]
                ].to_numpy(float)
            )
        ) * 1.15,
    ),
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 5),
)

for ax, landmark, title, panel in zip(
    axes,
    ["Day30", "Day90"],
    [
        "Early (Day 15-45)",
        "Later (Day 75-105)",
    ],
    [
        "(A)",
        "(B)",
    ],
):

    plot_calibration_panel(
        ax,
        cal.loc[
            cal["Landmark"].eq(landmark)
        ],
        title,
        lim,
    )

    # --------------------------------------------------------
    # Remove grid lines
    # --------------------------------------------------------
    ax.grid(False)

    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------
    ax.text(
        -0.10,
        1.04,
        panel,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


# ============================================================
# Layout
# ============================================================

fig.tight_layout(
    w_pad=2.5
)


# ============================================================
# Submission-quality outputs
# ============================================================

PDF_PATH = (
    FIGURE_DIR
    / "Figure2_calibration_v9.pdf"
)

TIFF_PATH = (
    FIGURE_DIR
    / "Figure2_calibration_v9.tiff"
)

fig.savefig(
    PDF_PATH,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    TIFF_PATH,
    dpi=1000,
    bbox_inches="tight",
    pad_inches=0.02,
)


# PNG for routine checking
save_figure(
    fig,
    "Figure2_calibration_v9.png",
)

plt.show()

print(PDF_PATH)
print(TIFF_PATH)


# %% [markdown]
# ## Figure3

# %%
# ============================================================
# Figure 3: Center-held-out AUC at 180 days
# ============================================================

iecv_table = read_source("TableS6").copy()


# ============================================================
# Facility labels
# ============================================================

CENTER_MAP = {
    "付属": "Center A",
    "北総": "Center B",
    "小杉": "Center C",
    "永山": "Center D",

    # English namesにも対応
    "Nippon Medical School Hospital": "Center A",
    "Chiba Hokusoh Hospital": "Center B",
    "Musashi-Kosugi Hospital": "Center C",
    "Tama Nagayama Hospital": "Center D",
}


# Table S6の施設列
center_col = "Validation center"

iecv_table[center_col] = (
    iecv_table[center_col]
    .replace(CENTER_MAP)
)


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 5.0),
)


for ax, landmark, title, panel in zip(
    axes,
    ["Day30", "Day90"],
    [
        "Early (Day 15-45)",
        "Later (Day 75-105)",
    ],
    [
        "(A)",
        "(B)",
    ],
):

    plot_iecv_panel(
        ax,
        iecv_table,
        landmark,
        "AUC180",
    )

    # --------------------------------------------
    # Panel title
    # --------------------------------------------
    ax.set_title(
        title,
        fontsize=12,
    )

    # --------------------------------------------
    # Panel label
    # --------------------------------------------
    ax.text(
        -0.12,
        1.04,
        panel,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # --------------------------------------------
    # Grid削除
    # --------------------------------------------
    ax.grid(False)

    # --------------------------------------------
    # X-axis label
    # --------------------------------------------
    ax.set_xlabel(
        "AUC at 180 days",
        fontsize=11,
    )


# ============================================================
# Layout
# ============================================================

fig.tight_layout(
    w_pad=2.5,
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    FIGURE_DIR / "Figure3_IECV_AUC180_v9.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    FIGURE_DIR / "Figure3_IECV_AUC180_v9.tiff",
    dpi=1000,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    FIGURE_DIR / "Figure3_IECV_AUC180_v9.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()

# %% [markdown]
# ## FigureS1_Day30

# %%
# Figure S1: full-cohort missingness, two-panel version (A) Early / (B) Later
miss = read_source("TableS3")

fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharex=False)

panel_specs = [
    ("Day30", "Early (Day 15-45)", "(A)"),
    ("Day90", "Later (Day 75-105)", "(B)"),
]

for ax, (landmark, title, panel) in zip(axes, panel_specs):
    d = (
        miss.loc[miss["Landmark"].eq(landmark)]
        .sort_values("Missing %", ascending=True)
        .copy()
    )

    ax.barh(d["Predictor"], d["Missing %"])
    ax.set_xlabel("Missing values (%)")
    ax.set_title(title)

    ax.set_xlim(0, max(5, float(d["Missing %"].max()) * 1.15))

    # panel label
    ax.text(
        -0.12, 1.03, panel,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    # 必要なら罫線を消す
    ax.grid(False)

fig.tight_layout(w_pad=2.5)
save_figure(fig, "FigureS1_missingness_v9.png")

# %%
fig.tight_layout(w_pad=2.5)

fig.savefig(
    FIGURE_DIR / "FigureS1_missingness_v9.pdf",
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    FIGURE_DIR / "FigureS1_missingness_v9.tiff",
    dpi=1000,
    bbox_inches="tight",
    pad_inches=0.02,
)

save_figure(fig, "FigureS1_missingness_v9.png")
plt.show()

# %% [markdown]
# ## FigureS3

# %%
# Figure S3: Early IECV calibration
cal = read_source("Day30_IECV_calibration").copy()

# ------------------------------------------------------------
# Center label mapping
# ------------------------------------------------------------
CENTER_MAP = {
    "付属": "Center A",
    "北総": "Center B",
    "小杉": "Center C",
    "永山": "Center D",
    "Nippon Medical School Hospital": "Center A",
    "Chiba Hokusoh Hospital": "Center B",
    "Musashi-Kosugi Hospital": "Center C",
    "Tama Nagayama Hospital": "Center D",
}

cal["Validation center display"] = (
    cal["Validation center"]
    .replace(CENTER_MAP)
)

# 表示順を固定
center_order = ["Center A", "Center B", "Center C", "Center D"]
centers = [
    c for c in center_order
    if c in cal["Validation center display"].drop_duplicates().tolist()
]

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, axes = plt.subplots(
    2, 2,
    figsize=(10.5, 9.0),
    squeeze=False
)

panel_labels = ["(A)", "(B)", "(C)", "(D)"]

for ax, center, panel in zip(axes.flat, centers, panel_labels):
    d = cal.loc[cal["Validation center display"].eq(center)].copy()

    plot_calibration_panel(
        ax,
        d,
        center,
    )

    # panel label
    ax.text(
        -0.12, 1.03, panel,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # remove grid
    ax.grid(False)

# 余ったaxesを非表示
for ax in list(axes.flat)[len(centers):]:
    ax.set_visible(False)

fig.tight_layout(
    w_pad=2.0,
    h_pad=2.2,
)

save_figure(fig, "FigureS3_Early_IECV_calibration_v9.png")

# %% [markdown]
# ## FigureS4

# %%
# Figure S4: Later IECV calibration
cal = read_source("Day90_IECV_calibration").copy()

# ------------------------------------------------------------
# Center label mapping
# ------------------------------------------------------------
CENTER_MAP = {
    "付属": "Center A",
    "北総": "Center B",
    "小杉": "Center C",
    "永山": "Center D",
    "Nippon Medical School Hospital": "Center A",
    "Chiba Hokusoh Hospital": "Center B",
    "Musashi-Kosugi Hospital": "Center C",
    "Tama Nagayama Hospital": "Center D",
}

cal["Validation center display"] = (
    cal["Validation center"]
    .replace(CENTER_MAP)
)

# 表示順を固定
center_order = ["Center A", "Center B", "Center C", "Center D"]

centers = [
    c
    for c in center_order
    if c in cal["Validation center display"].drop_duplicates().tolist()
]

# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(10.5, 9.0),
    squeeze=False,
)

panel_labels = [
    "(A)",
    "(B)",
    "(C)",
    "(D)",
]

visible_axes = []

for ax, center, panel in zip(
    axes.flat,
    centers,
    panel_labels,
):

    d = cal.loc[
        cal["Validation center display"].eq(center)
    ].copy()

    plot_calibration_panel(
        ax,
        d,
        center,
    )

    # panel label
    ax.text(
        -0.12,
        1.03,
        panel,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # remove grid
    ax.grid(False)

    visible_axes.append((center, ax))

# ------------------------------------------------------------
# Common scale
# Center Cは除外し、A/B/Dの自動スケールから共通範囲を決定
# ------------------------------------------------------------
reference_axes = [
    ax
    for center, ax in visible_axes
    if center != "Center C"
]

if reference_axes:
    xmins = [ax.get_xlim()[0] for ax in reference_axes]
    xmaxs = [ax.get_xlim()[1] for ax in reference_axes]
    ymins = [ax.get_ylim()[0] for ax in reference_axes]
    ymaxs = [ax.get_ylim()[1] for ax in reference_axes]

    common_xlim = (
        min(xmins),
        max(xmaxs),
    )
    common_ylim = (
        min(ymins),
        max(ymaxs),
    )

    # A-Dすべてに同じスケールを適用
    for _, ax in visible_axes:
        ax.set_xlim(common_xlim)
        ax.set_ylim(common_ylim)

# ------------------------------------------------------------
# 余ったaxesを非表示
# ------------------------------------------------------------
for ax in list(axes.flat)[len(centers):]:
    ax.set_visible(False)

fig.tight_layout(
    w_pad=2.0,
    h_pad=2.2,
)

save_figure(
    fig,
    "FigureS4_Later_IECV_calibration_v9.png",
)

# %% [markdown]
# ## FigureS5_Day30

# %%
# ============================================================
# Figure S5: exploratory subgroup differences
# (A) Early / (B) Later
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Font settings
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"

# 数学記号をTimes系に近いSTIXで描画
plt.rcParams["mathtext.fontset"] = "stix"


# ============================================================
# Load data
# ============================================================

sg = read_source("TableS7")

if sg.empty:

    print(
        "Subgroup analysis disabled or no eligible subgroups; "
        "no figure saved."
    )

else:

    # ========================================================
    # Subgroup label formatting
    # ========================================================

    def format_subgroup_label(x):

        s = str(x)

        # 表記ゆれを統一
        s = s.replace("> =", ">=")
        s = s.replace("< =", "<=")
        s = s.replace("=>", ">=")
        s = s.replace("=<", "<=")

        # 記号だけmathtextで描画
        s = s.replace(">=", r"$\geq$")
        s = s.replace("<=", r"$\leq$")

        return s


    # ========================================================
    # Data preparation
    # ========================================================

    def prepare_subgroup_data(data, landmark):

        d = data.loc[
            data["Landmark"].eq(landmark)
            & data["Model"].eq(
                "Current vs Discharge improvement"
            )
            & data["Metric"].eq(
                "Delta AUC180"
            )
        ].copy()

        if d.empty:
            raise ValueError(
                f"No {landmark} subgroup delta-AUC rows."
            )

        d["Subgroup_display"] = (
            d["Subgroup"]
            .map(format_subgroup_label)
        )

        return d


    d30 = prepare_subgroup_data(
        sg,
        "Day30",
    )

    d90 = prepare_subgroup_data(
        sg,
        "Day90",
    )


    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        sharex=True,
    )


    panel_specs = [
        (
            axes[0],
            d30,
            "Early (Day 15-45)",
            "(A)",
        ),
        (
            axes[1],
            d90,
            "Later (Day 75-105)",
            "(B)",
        ),
    ]


    for ax, d, title, panel in panel_specs:

        y = np.arange(len(d))

        est = pd.to_numeric(
            d["Estimate"],
            errors="coerce",
        ).to_numpy(float)

        lo = pd.to_numeric(
            d["95% CI lower"],
            errors="coerce",
        ).to_numpy(float)

        hi = pd.to_numeric(
            d["95% CI upper"],
            errors="coerce",
        ).to_numpy(float)


        # ----------------------------------------------------
        # Point estimate + 95% CI
        # ----------------------------------------------------

        xerr = np.vstack(
            [
                est - lo,
                hi - est,
            ]
        )

        ax.errorbar(
            est,
            y,
            xerr=xerr,
            fmt="o",
            linestyle="none",
            linewidth=1.2,
            capsize=0,
        )


        # ----------------------------------------------------
        # Reference line
        # ----------------------------------------------------

        ax.axvline(
            0,
            linestyle="--",
            linewidth=1,
            color="black",
        )


        # ----------------------------------------------------
        # Y labels
        # Times New Romanを維持
        # ----------------------------------------------------

        ax.set_yticks(y)

        ax.set_yticklabels(
            d["Subgroup_display"],
            fontsize=10.5,
            fontfamily="Times New Roman",
        )

        ax.invert_yaxis()


        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        ax.set_xlabel(
            "AUC improvement (Current - Discharge), 95% CI",
            fontsize=11,
            fontfamily="Times New Roman",
        )

        ax.set_title(
            title,
            fontsize=12,
            fontfamily="Times New Roman",
        )


        # ----------------------------------------------------
        # Panel label
        # ----------------------------------------------------

        ax.text(
            -0.12,
            1.03,
            panel,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            fontfamily="Times New Roman",
            ha="left",
            va="bottom",
        )


        # ----------------------------------------------------
        # Grid off
        # ----------------------------------------------------

        ax.grid(False)


    fig.tight_layout(
        w_pad=2.5,
    )

    save_figure(
        fig,
        "FigureS5_subgroup_deltaAUC_v9.png",
    )

    plt.show()

# %% [markdown]
# ## Consolidated Excel workbook

# %% [markdown]
# ### Discharge-laboratory ascertainment window
# 
# The primary discharge laboratory definition remains the prespecified discharge-period definition generated upstream. A 14-day discharge look-back sensitivity analysis is **not** calculated in Notebook 02 because it requires reconstructing predictors and refitting the models; if required, it should be added to Notebook 01 rather than performed during table/figure generation.

# %%
# Consolidated manuscript-facing workbook.
# No model fitting, MICE, bootstrap, or CI recalculation occurs here.

# Recreate output-only tables if this cell is run independently.
TableS1 = read_source("TableS2").copy()
TableS2 = make_selection_smd_table()
TableS3 = read_source("TableS3").copy()
TableS4 = read_source("TableS4").copy()
TableS5 = make_model_hr_table()
TableS6 = read_source("TableS5").copy()
TableS7 = read_source("TableS6").copy()
TableS8 = read_source("TableS7").copy()
TableS9 = read_source("TableS8").copy()
TableS10 = read_source("TableS9").copy()

workbook_tables = {
    "Table 1": read_source("Table1"),
    "Table 2": read_source("Table2"),
    "Figure 1 flow": read_source("Figure1_source_table"),
    "Table S1": TableS1,
    "Table S2": TableS2,
    "Table S3": TableS3,
    "Table S4": TableS4,
    "Table S5": TableS5,
    "Table S6": TableS6,
    "Table S7": TableS7,
    "Table S8": TableS8,
    "Table S9": TableS9,
    "Table S10": TableS10,
    "Settings": read_source("Settings"),
    "Notes": read_source("Notes"),
    "Lab timing audit": read_source("Lab_timing_audit"),
}

export_excel(
    workbook_tables,
    FINAL_XLSX,
)

print(FINAL_XLSX)


# %% [markdown]
# ## Figure output manifest

# %%
expected_figures = [
    "Figure2_calibration_v9.png",
    "Figure3_IECV_AUC180_v9.png",
    "FigureS1_missingness_v9.png",
    "FigureS3_Early_IECV_calibration_v9.png",
    "FigureS4_Later_IECV_calibration_v9.png",
    "FigureS5_subgroup_deltaAUC_v9.png",
]

manifest = pd.DataFrame({
    "File": expected_figures,
    "Path": [str(FIGURE_DIR / name) for name in expected_figures],
    "Exists": [(FIGURE_DIR / name).is_file() for name in expected_figures],
})

display(manifest)

export_excel(
    {"Figure files": manifest},
    FIGURE_DIR / "Figure_output_manifest.xlsx",
)

print(FIGURE_DIR)


# %%
# -*- coding: utf-8 -*-

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================
# 1. Output
# ============================================================
FLOW_OUTPUT_DIR = DATA_DIR
PNG_PATH = FLOW_OUTPUT_DIR / "Figure1_study_flow_compact_v3.png"
TIFF_PATH = FLOW_OUTPUT_DIR / "Figure1_study_flow_compact_v3.tiff"

# ============================================================
# 2. Figure settings
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11

fig, ax = plt.subplots(figsize=(10.4, 7.2))   # ← 少しだけ縦を増やす
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ============================================================
# 3. Helper functions
# ============================================================
def add_box(ax, x, y, w, h, text, fontsize=11, fontweight="normal", lw=1.2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.006,rounding_size=0.010",
        linewidth=lw,
        edgecolor="black",
        facecolor="white",
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        linespacing=1.18,
    )

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "left": x,
        "right": x + w,
        "bottom": y,
        "top": y + h,
        "cx": x + w / 2,
        "cy": y + h / 2,
    }


def v_arrow(ax, x, y0, y1, lw=1.2, ms=11):
    arr = FancyArrowPatch(
        (x, y0),
        (x, y1),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color="black",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)


def h_line(ax, x0, x1, y, lw=1.2):
    ax.plot([x0, x1], [y, y], color="black", linewidth=lw)


def v_line(ax, x, y0, y1, lw=1.2):
    ax.plot([x, x], [y0, y1], color="black", linewidth=lw)


# ============================================================
# 4. Top boxes
# ============================================================
top_w = 0.36
top_x = 0.5 - top_w / 2

source = add_box(
    ax,
    x=top_x,
    y=0.885,
    w=top_w,
    h=0.070,
    text="Patients hospitalized with heart failure\nN = 9,395",
    fontsize=12.2,
    fontweight="bold",
)

excluded_top = add_box(
    ax,
    x=top_x,
    y=0.765,
    w=top_w,
    h=0.085,
    text=(
        "Excluded, n = 2,282\n"
        "Death during index hospitalization, n = 839\n"
        "Transfer to another hospital, n = 1,443"
    ),
    fontsize=10.8,
)

post = add_box(
    ax,
    x=top_x,
    y=0.645,
    w=top_w,
    h=0.075,
    text="Post-discharge source cohort\nN = 7,113",
    fontsize=12.2,
    fontweight="bold",
)

v_arrow(ax, source["cx"], source["bottom"] - 0.004, excluded_top["top"] + 0.004)
v_arrow(ax, excluded_top["cx"], excluded_top["bottom"] - 0.004, post["top"] + 0.004)

# ============================================================
# 5. Branch geometry
#    3→4段目の余裕を少し広げる
# ============================================================
branch_y = 0.575

left_cx = 0.245
right_cx = 0.755

# post box から中央縦線
v_line(ax, post["cx"], post["bottom"], branch_y)

# 左右分岐の横線
h_line(ax, left_cx, right_cx, branch_y)

# ============================================================
# 6. Left branch
# ============================================================
branch_w = 0.40
branch_h1 = 0.095
branch_h2 = 0.100
branch_h3 = 0.115

left_x = left_cx - branch_w / 2

early = add_box(
    ax,
    x=left_x,
    y=0.455,
    w=branch_w,
    h=branch_h1,
    text=(
        "Early outpatient follow-up\n"
        "First outpatient visit on days 15-45\n"
        "n = 4,555"
    ),
    fontsize=11.4,
    fontweight="bold",
)

early_excl = add_box(
    ax,
    x=left_x,
    y=0.285,
    w=branch_w,
    h=branch_h2,
    text=(
        "Excluded, n = 277\n"
        "Composite event on/before index visit, n = 101\n"
        "No valid post-index follow-up, n = 176"
    ),
    fontsize=10.5,
)

early_final = add_box(
    ax,
    x=left_x,
    y=0.090,
    w=branch_w,
    h=branch_h3,
    text=(
        "Early analysis cohort\n"
        "n = 4,278\n"
        "180-day composite events, n = 411\n"
        "(Readmission-first, n = 369; death-first, n = 42)"
    ),
    fontsize=10.8,
    fontweight="bold",
)

v_arrow(ax, left_cx, branch_y, early["top"] + 0.004)
v_arrow(ax, left_cx, early["bottom"] - 0.004, early_excl["top"] + 0.004)
v_arrow(ax, left_cx, early_excl["bottom"] - 0.004, early_final["top"] + 0.004)

# ============================================================
# 7. Right branch
# ============================================================
right_x = right_cx - branch_w / 2

later = add_box(
    ax,
    x=right_x,
    y=0.455,
    w=branch_w,
    h=branch_h1,
    text=(
        "Later outpatient follow-up\n"
        "First outpatient visit on days 75-105\n"
        "n = 3,324"
    ),
    fontsize=11.4,
    fontweight="bold",
)

later_excl = add_box(
    ax,
    x=right_x,
    y=0.285,
    w=branch_w,
    h=branch_h2,
    text=(
        "Excluded, n = 316\n"
        "Composite event on/before index visit, n = 225\n"
        "No valid post-index follow-up, n = 91"
    ),
    fontsize=10.5,
)

later_final = add_box(
    ax,
    x=right_x,
    y=0.090,
    w=branch_w,
    h=branch_h3,
    text=(
        "Later analysis cohort\n"
        "n = 3,008\n"
        "180-day composite events, n = 250\n"
        "(Readmission-first, n = 221; death-first, n = 29)"
    ),
    fontsize=10.8,
    fontweight="bold",
)

v_arrow(ax, right_cx, branch_y, later["top"] + 0.004)
v_arrow(ax, right_cx, later["bottom"] - 0.004, later_excl["top"] + 0.004)
v_arrow(ax, right_cx, later_excl["bottom"] - 0.004, later_final["top"] + 0.004)

# ============================================================
# 8. Save
# ============================================================
fig.subplots_adjust(left=0.02, right=0.98, top=0.985, bottom=0.02)

fig.savefig(
    PNG_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.01,
)

fig.savefig(
    TIFF_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.01,
)

# %%
# ============================================================
# Early / Later final cohort overlap
# 元データ読み込み → 最終コホート再構築 → 重複症例数集計
# 単独セルで実行可能
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display

# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------
# DATA_DIR is defined at the top of this script.
SOURCE_PATH = DATA_DIR / "df_LR.parquet"
OUTPUT_PATH = DATA_DIR / "Early_Later_cohort_overlap.xlsx"

HORIZON = 180
EARLY_WINDOW = (15, 45)
LATER_WINDOW = (75, 105)

# 元解析と同じ設定
KNOWN_FIRST_EVENT_EXTENDS_LASTDATE = True


# ------------------------------------------------------------
# 1. Date parser
#    元ノートブックと同じ考え方
# ------------------------------------------------------------
def parse_mixed_date_series(s):
    s = s.copy()

    if pd.api.types.is_datetime64_any_dtype(s):
        out = pd.to_datetime(s, errors="coerce")

    else:
        numeric = pd.to_numeric(s, errors="coerce")
        out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

        # YYYYMMDD
        ymd = (
            numeric.between(19000101, 21001231)
            & (numeric == np.floor(numeric))
        )
        out.loc[ymd] = pd.to_datetime(
            numeric.loc[ymd].astype("int64").astype(str),
            format="%Y%m%d",
            errors="coerce",
        )

        # Excel serial date
        excel = numeric.between(20000, 80000) & ~ymd
        out.loc[excel] = pd.to_datetime(
            numeric.loc[excel],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

        # Text date
        text_mask = (
            s.notna()
            & ~ymd
            & ~excel
            & numeric.isna()
        )

        if text_mask.any():
            out.loc[text_mask] = pd.to_datetime(
                s.loc[text_mask].astype(str).str.strip(),
                format="mixed",
                errors="coerce",
            )

    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)

    return out.dt.normalize()


# ------------------------------------------------------------
# 2. Load source data
# ------------------------------------------------------------
if not SOURCE_PATH.is_file():
    raise FileNotFoundError(f"Source file not found:\n{SOURCE_PATH}")

df = pd.read_parquet(SOURCE_PATH).copy().reset_index(drop=True)

print(f"Source file : {SOURCE_PATH}")
print(f"Source N    : {len(df):,}")
print(f"Columns     : {df.shape[1]:,}")


# ------------------------------------------------------------
# 3. Required columns
# ------------------------------------------------------------
required_cols = [
    "INDEX",
    "施設名",
    "データ識別番号",
    "退院日",
    "死亡日",
    "再入院日",
    "LastDate",
    "LM30_対象",
    "LM90_対象",
    "Day30_外来受診日",
    "Day90_外来受診日",
    "Day30_外来_退院後日数",
    "Day90_外来_退院後日数",
]

missing_cols = sorted(set(required_cols) - set(df.columns))

if missing_cols:
    raise KeyError(
        "Required columns are missing:\n"
        + "\n".join(missing_cols)
    )

if df["INDEX"].isna().any():
    raise ValueError("INDEX contains missing values.")

if df["INDEX"].duplicated().any():
    raise ValueError(
        f"INDEX is not unique: "
        f"{int(df['INDEX'].duplicated().sum())} duplicated rows."
    )


# ------------------------------------------------------------
# 4. Type conversion
# ------------------------------------------------------------
date_cols = [
    "退院日",
    "死亡日",
    "再入院日",
    "LastDate",
    "Day30_外来受診日",
    "Day90_外来受診日",
]

for col in date_cols:
    original = df[col].copy()
    df[col] = parse_mixed_date_series(df[col])

    supplied = (
        original.notna()
        & original.astype(str).str.strip().ne("")
    )

    bad = supplied & df[col].isna()

    if bad.any():
        raise ValueError(
            f"{col}: {int(bad.sum())} unparseable dates. "
            f"Examples: {original.loc[bad].head().tolist()}"
        )

numeric_cols = [
    "LM30_対象",
    "LM90_対象",
    "Day30_外来_退院後日数",
    "Day90_外来_退院後日数",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ------------------------------------------------------------
# 5. Function to reconstruct final cohort
# ------------------------------------------------------------
def make_final_cohort(data, day, window):

    low, high = window

    flag_col = f"LM{day}_対象"
    visit_date_col = f"Day{day}_外来受診日"
    visit_day_col = f"Day{day}_外来_退院後日数"

    # Eligible outpatient visit
    d = data.loc[data[flag_col].eq(1)].copy()

    n_flag = len(d)

    # Valid index visit
    valid_visit = (
        d[visit_day_col].between(low, high, inclusive="both")
        & d[visit_day_col].eq(np.floor(d[visit_day_col]))
        & d[visit_date_col].notna()
    )

    n_invalid_visit = int((~valid_visit).sum())

    d = d.loc[valid_visit].copy()

    d["prediction_date"] = d[visit_date_col]

    # --------------------------------------------------------
    # Exclude outcome on or before index visit
    # --------------------------------------------------------
    pre_readmission = (
        d["再入院日"].notna()
        & (d["再入院日"] <= d["prediction_date"])
    )

    pre_death = (
        d["死亡日"].notna()
        & (d["死亡日"] <= d["prediction_date"])
    )

    pre_either = pre_readmission | pre_death

    n_pre_event = int(pre_either.sum())

    d = d.loc[~pre_either].copy()

    # --------------------------------------------------------
    # Follow-up availability
    # --------------------------------------------------------
    first_event = d[["再入院日", "死亡日"]].min(axis=1)

    follow = d["LastDate"].copy()

    # 元解析と同じ:
    # known eventがLastDateより後なら、そのeventまでfollow-upを延長
    if KNOWN_FIRST_EVENT_EXTENDS_LASTDATE:
        follow = pd.concat(
            [follow, first_event],
            axis=1,
        ).max(axis=1)

    valid_follow = (
        follow.notna()
        & (follow > d["prediction_date"])
    )

    n_no_followup = int((~valid_follow).sum())

    d = d.loc[valid_follow].copy()

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------
    expected_n = (
        n_flag
        - n_invalid_visit
        - n_pre_event
        - n_no_followup
    )

    if expected_n != len(d):
        raise RuntimeError(
            f"Day{day}: cohort count mismatch "
            f"({expected_n:,} != {len(d):,})"
        )

    flow = {
        "Cohort": "Early" if day == 30 else "Later",
        "Window": f"Day {low}-{high}",
        "Eligible outpatient N": n_flag,
        "Invalid/missing index visit excluded": n_invalid_visit,
        "Pre-index outcome excluded": n_pre_event,
        "No positive post-index follow-up excluded": n_no_followup,
        "Final N": len(d),
    }

    return d.reset_index(drop=True), flow


# ------------------------------------------------------------
# 6. Reconstruct Early and Later final cohorts
# ------------------------------------------------------------
early, flow_early = make_final_cohort(
    df,
    day=30,
    window=EARLY_WINDOW,
)

later, flow_later = make_final_cohort(
    df,
    day=90,
    window=LATER_WINDOW,
)

flow_table = pd.DataFrame(
    [flow_early, flow_later]
)


# ------------------------------------------------------------
# 7. Overlap by INDEX
# ------------------------------------------------------------
early_ids = set(early["INDEX"].astype(str))
later_ids = set(later["INDEX"].astype(str))

both_ids = early_ids & later_ids
early_only_ids = early_ids - later_ids
later_only_ids = later_ids - early_ids
either_ids = early_ids | later_ids

n_early = len(early_ids)
n_later = len(later_ids)
n_both = len(both_ids)
n_early_only = len(early_only_ids)
n_later_only = len(later_only_ids)
n_either = len(either_ids)

pct_both_early = 100 * n_both / n_early
pct_both_later = 100 * n_both / n_later


# ------------------------------------------------------------
# 8. Summary
# ------------------------------------------------------------
summary = pd.DataFrame(
    {
        "Measure": [
            "Early cohort N",
            "Later cohort N",
            "Included in both cohorts N",
            "Both / Early (%)",
            "Both / Later (%)",
            "Early only N",
            "Later only N",
            "Unique INDEX in either cohort N",
        ],
        "Value": [
            n_early,
            n_later,
            n_both,
            pct_both_early,
            pct_both_later,
            n_early_only,
            n_later_only,
            n_either,
        ],
    }
)


# ------------------------------------------------------------
# 9. Patient-level membership
# ------------------------------------------------------------
membership = df.loc[
    df["INDEX"].astype(str).isin(either_ids),
    [
        "INDEX",
        "施設名",
        "データ識別番号",
    ],
].copy()

membership["INDEX"] = membership["INDEX"].astype(str)

membership["Early cohort"] = (
    membership["INDEX"].isin(early_ids)
).astype(int)

membership["Later cohort"] = (
    membership["INDEX"].isin(later_ids)
).astype(int)

membership["Cohort membership"] = np.select(
    [
        (membership["Early cohort"] == 1)
        & (membership["Later cohort"] == 1),

        (membership["Early cohort"] == 1)
        & (membership["Later cohort"] == 0),

        (membership["Early cohort"] == 0)
        & (membership["Later cohort"] == 1),
    ],
    [
        "Both",
        "Early only",
        "Later only",
    ],
    default="Check",
)

membership = membership.sort_values(
    ["Cohort membership", "INDEX"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 10. Manuscript-ready sentence
# ------------------------------------------------------------
manuscript_sentence = (
    f"A total of {n_both:,} patients were included in both cohorts, "
    f"representing {pct_both_early:.1f}% of the Early cohort "
    f"and {pct_both_later:.1f}% of the Later cohort."
)


# ------------------------------------------------------------
# 11. Sanity check against current manuscript counts
# ------------------------------------------------------------
print("\n===== Cohort reconstruction =====")
display(flow_table)

print("\n===== Overlap summary =====")
display(summary)

if n_early != 4278:
    print(f"WARNING: Early N is {n_early:,}, expected 4,278.")

if n_later != 3008:
    print(f"WARNING: Later N is {n_later:,}, expected 3,008.")


# ------------------------------------------------------------
# 12. Excel output
# ------------------------------------------------------------
with pd.ExcelWriter(
    OUTPUT_PATH,
    engine="openpyxl",
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Overlap summary",
        index=False,
    )

    flow_table.to_excel(
        writer,
        sheet_name="Cohort reconstruction",
        index=False,
    )

    membership.to_excel(
        writer,
        sheet_name="Patient membership",
        index=False,
    )


# ------------------------------------------------------------
# 13. Final output
# ------------------------------------------------------------
print("\n===== Manuscript-ready sentence =====")
print(manuscript_sentence)

print(f"\nSaved: {OUTPUT_PATH}")


