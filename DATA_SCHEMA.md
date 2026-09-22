# Data Schema

The source file must be named `df_LR.parquet` and contain one row per unique `INDEX`.

## Identification and follow-up

- `INDEX`
- `施設名`
- `データ識別番号`
- `入院日`
- `退院日`
- `死亡日`
- `再入院日`
- `LastDate`

## Baseline clinical variables

- `Age`
- `Male`
- `BMI`
- `LVEF`
- `af`
- `DM`
- `維持透析`
- `PD`

## Discharge medications

- `SGLT2`
- `β遮断薬`
- `RAS_ARNI`
- `MRA_内服`
- `Loop_内服`

## Outpatient eligibility/timing

- `LM30_対象`
- `LM90_対象`
- `Day30_外来受診日`
- `Day90_外来受診日`
- `Day30_外来_退院後日数`
- `Day90_外来_退院後日数`

## Laboratory source labels

`Alb`, `BUN`, `Hb`, `K`, `Na`, `eGFR`, `NT-proBNP`

### Discharge

`dis_Alb`, `dis_BUN`, `dis_Hb`, `dis_K`, `dis_Na`, `dis_eGFR`, `dis_NT-proBNP`

### Early outpatient values and dates

For each laboratory item:

`d30_<LAB>` and `d30_<LAB>_date`

### Later outpatient values and dates

For each laboratory item:

`d90_<LAB>` and `d90_<LAB>_date`

## Timing rule

An outpatient laboratory value is retained only when its measurement date falls from 14 days before the index outpatient visit through the index visit itself, and not before discharge.

The code aborts when a supplied nonmissing outpatient laboratory value violates the timing policy.

## Privacy

The source data and several generated outputs contain patient-level identifiers and dates. Do not upload them to a public repository.
