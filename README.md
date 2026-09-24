# Preparing Historical Equity Data for Machine Learning (Project 2)

An end-to-end time-series feature engineering and data transformation pipeline designed to turn nearly five decades of raw daily equity trading records into a high-quality, leakage-free dataset for predictive modeling.

This repository serves as the bridge between exploratory data analysis (Project 1) and predictive model development (Project 3), transforming over 20 million raw stock transactions into an engineered, memory-optimized format.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Data Cleaning & Integrity Decisions](#data-cleaning--integrity-decisions)
- [Engineered Features](#engineered-features)
- [Technical Challenges & Memory Optimization](#technical-challenges--memory-optimization)
- [Dataset Summary & Verification](#dataset-summary--verification)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Next Steps (Project 3 Preview)](#next-steps-project-3-preview)
- [License & Acknowledgments](#license--acknowledgments)

---

## Project Overview

Raw stock data (open, high, low, close, volume) reflects historical pricing snapshots at the end of each session. However, machine learning models require context: momentum, trends, risk, and relative performance.

Key project objectives:
- **Clean and validate** 48 years of historical trade records (1970–2018) across NASDAQ and NYSE listings.
- **Isolate time-series calculations** strictly by company ticker (`groupby('ticker')`) to prevent future-to-past lookahead and cross-ticker data leakage.
- **Engineer normalized momentum, trend, and volatility features** that allow models to evaluate penny stocks and multi-thousand-dollar equities on an equal footing.
- **Prepare and export** a clean, leak-free dataset (`stock_prepared.csv`, 2.85 GB) for downstream machine learning in Project 3.

---

## Pipeline Architecture

The pipeline processes raw data through four distinct sequential stages:

```text
[ 1. Ingestion & Preprocessing ]
├── Load historical_stock_prices.csv (20.97M rows) & historical_stocks.csv (6.46K rows)
├── Purge duplicate header record at index 74
├── Convert dates to datetime64[ns]
└── Sort chronologically by ['ticker', 'date']
              │
              ▼
[ 2. Grouped Time-Series Engineering ]
├── Isolate calculation windows strictly by ticker
├── Momentum: close_lag_1, price_change, daily_return
├── Trends: 5-day & 20-day rolling moving averages
└── Turbulence/Risk: 20-day rolling return volatility & intraday range spread %
              │
              ▼
[ 3. Integration & Categorical Encoding ]
├── Inner join metadata (exchange, sector) on 'ticker' key
├── Prune lookback warm-up rows (20-day cold starts, ~0.54% pruned)
├── One-hot encode categorical features (exchange, sector)
└── Downcast numeric schemas (float32, int8) to preserve RAM
              │
              ▼
[ 4. Production Verification & Disk Export ]
├── Comprehensive data audit (0 nulls, 0 duplicate trading days)
└── Stream write to disk: stock_prepared.csv (20,860,334 rows, 2.85 GB)
```

---

## Data Cleaning & Integrity Decisions

Building upon the initial exploratory audit from Project 1, three key data-quality decisions were made:

### 1. Injected Header Purge
An upstream concatenation error had inserted an extra column header row (`['Symbol', 'NYSE', ...]`) at index 74 of the security catalog. Dropping this single artifact brought the master company registry to exactly 6,459 valid, unique tickers.

### 2. Imputing Unclassified Securities as `'Unknown'`
A total of 1,440 securities lacked formal Global Industry Classification Standard (GICS) sector and industry assignments. Rather than deleting them—which would have created severe survivorship bias by eliminating ETFs, holding trusts, and SPACs—these assets were explicitly categorized as `'Unknown'` to preserve their market histories.

### 3. Investigation of Extreme Numerical Observations
* **Multi-Million Dollar Closes (`SCON`, `CODA`, `TOPS`):** Audited and confirmed to be the result of repeated historical reverse stock splits (e.g., 1-for-50 consolidations) executed by micro-caps to maintain exchange listing compliance. These records were retained, while shifting model emphasis to normalized returns and ratios rather than nominal price levels.
* **Vendor Calculation Overflow (`AAN`):** Aaron’s Inc. exhibited an adjusted close of 18.9 quintillion USD on February 27, 1987, caused by an upstream vendor divide-by-zero split calculation error. The underlying trading record was retained because the raw trade figures were accurate ($1.30 USD close), but the corrupted `adj_close` column was systematically excluded from the feature set.

---

## Engineered Features

All engineered indicators were calculated chronologically and grouped strictly by stock ticker:

| Feature Name | Category | Formula / Definition | Analytical Purpose |
| :--- | :--- | :--- | :--- |
| `close_lag_1` | Lag (1-Day) | Close<sub>t-1</sub> | Provides yesterday’s baseline benchmark for sequential time-series modeling. |
| `price_change` | Absolute Delta | Close<sub>t</sub> − Close<sub>t-1</sub> | Tracks net dollar movement between consecutive trading sessions. |
| `daily_return` | Normalized Return | (Close<sub>t</sub> − Close<sub>t-1</sub>) / Close<sub>t-1</sub> | Normalizes performance across price levels, removing nominal price bias. |
| `ma_5` | Moving Average | (1/5) &Sigma;<sub>i=0..4</sub> Close<sub>t-i</sub> | Captures short-term (1-week) directional price momentum. |
| `ma_20` | Moving Average | (1/20) &Sigma;<sub>i=0..19</sub> Close<sub>t-i</sub> | Establishes a medium-term (~1 trading month) baseline trend. |
| `volatility_20` | Rolling Risk | &sigma;<sub>20</sub>(daily_return) | Measures 20-day return dispersion to detect market turbulence and regime shifts. |
| `daily_range_pct` | Intraday Spread | (High<sub>t</sub> − Low<sub>t</sub>) / Low<sub>t</sub> | Measures intraday volatility relative to share price, independent of net close. |

---

## Technical Challenges & Memory Optimization

Processing 20.9M rows across 29 features can easily exceed standard memory limits (such as Google Colab's ~12.7 GB RAM cap). The following engineering practices were implemented to maintain system stability:

1. **Numeric Precision Downcasting:** 64-bit floating-point columns (`float64`) were downcasted to 32-bit (`float32`), cutting numeric memory usage in half without loss of practical precision.
2. **Compact Categorical Encoding:** One-hot encoded binary indicator columns were stored as 8-bit integers (`int8`), utilizing 1 byte per record instead of standard 8-byte integers.
3. **In-Place Operations & Explicit Garbage Collection:** Merges, drops, and transforms were executed in-place or cleaned up using `gc.collect()`, preventing multiple duplicate 21M-row DataFrames from occupying RAM simultaneously.
4. **Handling Warm-Up Missing Values:** Rolling 20-day metrics create `NaN` values for the first 19 trading days of every stock. Imputing zeros was avoided because `0.0` represents a valid market condition (0% return, zero volatility). Instead, initial warm-up rows were dropped, pruning 113,555 rows (<0.55% of the data) and leaving 20,860,334 fully populated records.

---

## Dataset Summary & Verification

The finalized dataset was validated against pre-export verification standards:

```text
======================================================================
FINAL DATASET AUDIT SUMMARY
======================================================================
Total Processed Records : 20,860,334 rows
Total Features / Columns: 29 columns
Temporal Coverage       : 1970-01-02 to 2018-08-24
Unique Tickers          : 6,459 companies
Missing Values (NaNs)   : 0 (Zero nulls remaining)
Duplicate Records       : 0 (Zero duplicate [ticker, date] entries)
Peak Working Memory     : ~1.65 GB
Exported File Size      : 2.85 GB (stock_prepared.csv)
======================================================================
```

### Complete Feature Inventory (29 Columns)
- **Identifiers & Temporal (2):** `ticker`, `date`
- **Market Core Metrics (5):** `open`, `high`, `low`, `close`, `volume`
- **Engineered Time-Series Indicators (7):** `close_lag_1`, `price_change`, `daily_return`, `ma_5`, `ma_20`, `volatility_20`, `daily_range_pct`
- **One-Hot Encoded Exchanges (2):** `exchange_NASDAQ`, `exchange_NYSE`
- **One-Hot Encoded Sectors (13):** `sector_BASIC INDUSTRIES`, `sector_CAPITAL GOODS`, `sector_CONSUMER DURABLES`, `sector_CONSUMER NON-DURABLES`, `sector_CONSUMER SERVICES`, `sector_ENERGY`, `sector_FINANCE`, `sector_HEALTH CARE`, `sector_MISCELLANEOUS`, `sector_PUBLIC UTILITIES`, `sector_TECHNOLOGY`, `sector_TRANSPORTATION`, `sector_Unknown`

---

## Repository Structure

```text
├── notebooks/
│   └── Project2_CesarJuarez.ipynb     # Complete data prep & feature engineering pipeline
├── reports/
│   ├── Project2_CesarJuarez.pdf       # Formal technical report
│   └── Project2_CesarJuarez.docx      # Word document with embedded native equations
├── src/
│   └── pipeline.py                    # Modular Python script for feature generation
├── docs/
│   └── pipeline_architecture.png      # High-level architecture and flow diagram
├── .gitignore                         # Excludes large CSVs and environment artifacts
├── README.md                          # Comprehensive project documentation
└── requirements.txt                   # Environment dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Google Colab or local environment with at least 8 GB RAM

### Installation
Clone the repository and install required packages:

```bash
git clone [https://github.com/your-username/stock-data-preparation-ml.git](https://github.com/your-username/stock-data-preparation-ml.git)
cd stock-data-preparation-ml
pip install -r requirements.txt
```

### Running the Pipeline
Open `notebooks/Project2_CesarJuarez.ipynb` in Google Colab or your local Jupyter environment:
1. Mount Google Drive or point data paths to your local raw CSV files.
2. Execute the sequential cells:
   - Part 1: Loading, Cleaning, and Chronological Sorting
   - Part 2: Vectorized Feature Engineering (Grouped by Ticker)
   - Part 3: Memory-Optimized Merging, Pruning, and One-Hot Encoding
   - Part 4: Quality Verification & Final Export to `stock_prepared.csv`

---

## Next Steps (Project 3 Preview)

The finalized `stock_prepared.csv` dataset is structured directly for machine learning modeling:
1. **Chronological Splitting:** Enforce walk-forward temporal splits (e.g., Train 1970–2012, Validate 2013–2015, Test 2016–2018) rather than random k-fold cross-validation to prevent temporal lookahead leakage.
2. **Target Definition:** Frame modeling objectives around normalized forward returns (e.g., $t+1$ or $t+5$ daily returns) or trend-following binary classification targets (e.g., `close > ma_20`).
3. **Macro Integration:** Combine with external economic indices (Federal Reserve interest rates, S&P 500 benchmark returns) to capture broader market regimes.

---

## License & Acknowledgments

This project was developed as part of the Data Analytics and Artificial Intelligence Program at Willis College.
Dataset sourced from historical NYSE and NASDAQ price records (1970–2018).
