# Simple Recurrent Neural Networks for Univariate Time Series Forecasting

Code for RW441 Assignment 3 (Aidan Cohen, 27187950).

The project compares three simple recurrent neural network architectures on
one-step-ahead forecasting of five univariate time series:

- **Elman** (ERNN): feeds back the hidden layer
- **Jordan** (JRNN): feeds back the output
- **Multi-recurrent** (MRNN): feeds back both through four decaying memory banks

All three are implemented as one parameterised network (`src/models.py`), so
they differ only in the values fed back to the hidden layer. Accuracy is
measured by the Theil U statistic relative to the persistence forecast, on
expanding-window test folds that were never used during hyperparameter tuning.
A lag study measures which past observations each network relies on and tests
whether direct access to a known lag improves the forecast.

## Datasets

| Series | Source | Frequency |
|---|---|---|
| Mackey-Glass | generated (RK4, delay 17) | time step |
| Electricity | PJM Interconnection hourly load, PJME region | hourly |
| Temperature | Melbourne daily minimum temperature, 1981-1990 | daily |
| Page views | Wikimedia pageviews API, article Masters_Tournament | daily |
| S&P 500 | Yahoo Finance, ticker ^GSPC, 2010-2025 | trading day |

The loaders in `src/data.py` download each dataset on first use and cache it in
`data/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training runs on the CPU. When several scripts run in parallel, limit the
threads per process to avoid oversubscription:

```bash
export OMP_NUM_THREADS=1
```

## Reproducing the results

Run from `src/`, in this order:

```bash
python sanity_check.py                # 12 checks: windows, folds, leakage, transforms
python tuning.py                      # Optuna search on the first 60% of each series
python run.py                         # final runs: 5 folds x 5 seeds per architecture
python analysis.py                    # tests, Table III, box plots, forecast figures
python lag_study.py sensitivity --datasets mackey_glass electricity pageviews temperature sp500
python lag_study.py select            # residual rule (reported, then rejected)
python lag_study.py shortcut --datasets mackey_glass electricity pageviews temperature sp500
python lag_study.py analyse           # stage 2 tests and the sensitivity figure
python fold_robustness.py             # fold-level robustness check
python report_tables.py               # Tables I and II, drift-only baseline
python cv_diagram.py                  # Figure 1 (data partitioning)
```

`tuning.py` and `run.py` accept `--datasets` to run a subset, for example
`python run.py --datasets mackey_glass electricity`. The `sensitivity` and
`shortcut` stages of `lag_study.py` run only the datasets listed after
`--datasets`.
All random seeds are fixed, so repeated runs give identical results.

## Repository layout

```
src/
  common.py            paths, seed, plotting defaults
  data.py              dataset loaders, cleaning, ADF and KPSS tests
  transforms.py        log and log-return transforms and their inverses
  models.py            the parameterised recurrent network
  pipeline.py          windowing, expanding-window folds, training, metrics
  sanity_check.py      automated checks of the pipeline
  tuning.py            hyperparameter search
  run.py               final runs on the held-out test folds
  analysis.py          statistical tests, tables and figures
  lag_study.py         sensitivity analysis and shortcut experiment
  fold_robustness.py   fold-level robustness check
  report_tables.py     LaTeX tables and drift-only baseline
  cv_diagram.py        partitioning diagram
results/               CSV outputs of every script
figures/               PDF figures used in the report
tables/                LaTeX tables used in the report
report/                LaTeX source of the report

## Use of AI tools

I developed this project with the assistance of Claude (Anthropic), used as a
programming and writing assistant.

Claude provided the code for this project, which I typed out, worked through
and debugged myself, and modified where needed, for example the
hyperparameter search space and the lag study settings.
Claude also explained the underlying concepts, discussed the experimental
design with me, and reviewed and edited the report text.

I made the final design decisions, ran every experiment, checked the outputs
and revised both the code and the report.
All results in the report come from running the code in this repository.