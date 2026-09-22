# Model Specification

## Prediction cohorts

| Cohort | Window | Prediction time |
|---|---|---|
| Early | Days 15-45 after discharge | First outpatient visit in the window |
| Later | Days 75-105 after discharge | First outpatient visit in the window |

The cohorts are separately defined; a patient may contribute to both.

## Outcome

First emergency rehospitalization for heart failure or all-cause death during the 180 days after the index outpatient visit.

## Prespecified nonlaboratory predictors

Age, sex, BMI, LVEF <50%, atrial fibrillation, diabetes mellitus, dialysis, discharge SGLT2 inhibitor, beta-blocker, RAS inhibitor/ARNI, MRA, loop diuretic, and days from discharge to the index outpatient visit.

## Laboratory predictors

Albumin, BUN, hemoglobin, potassium, sodium, eGFR, and NT-proBNP (log2-transformed).

- **Discharge model:** discharge laboratory values.
- **Current-value model:** latest outpatient values within the 14-day look-back window ending on the prediction time.

## Functional forms

Continuous predictors are modeled continuously. Three-knot restricted cubic splines are considered using the 10th, 50th, and 90th percentiles. The same selected functional form is applied to each paired model within a cohort. Visit timing is prespecified as linear.

## Missing data

Primary MICE settings:

- 20 imputed datasets
- 20 chained-equation cycles per block
- predictive mean matching for continuous variables
- logistic/Bernoulli imputation for binary variables

Baseline/discharge variables are completed without outpatient laboratory values or outcomes, then frozen before current outpatient measurements are imputed.

## Cox model

- Standard unpenalized Cox proportional hazards model
- Breslow handling of ties
- No automated variable selection
- No tuning cross-validation
- No penalized fallback

## Performance

- Time-dependent AUC at 180 days
- Harrell's C-index
- Brier score at 180 days
- Calibration-in-the-large
- Calibration slope
- Calibration plots

The pipeline additionally calculates integrated Brier score as an auxiliary metric.

## Validation

### Internal validation

200 patient-level bootstrap redevelopment replicates, sampled within hospital. Functional-form assessment is repeated. Bootstrap redevelopment uses 5 imputed datasets x 5 cycles as a computational approximation.

### Internal-external cross-validation

Leave-one-center-out validation across four hospitals. Imputation relationships, donor pools, functional forms, spline knots, standardization, Cox coefficients, and baseline cumulative hazards are estimated only from development hospitals.

## Sensitivity analyses

- 30-day-standardized discharge-to-current laboratory changes
- Complete-case analysis
- Single-imputation analysis
- Exclusion of dialysis patients

## Random seed

`20260916`
