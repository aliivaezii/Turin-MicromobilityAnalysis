<div align="center"><div align="center">



# 🛴 Turin Micromobility Analysis# Turin Micromobility Analysis



**Comprehensive Spatial-Temporal Analysis of Shared E-Scooter Services****Comprehensive Spatial-Temporal Analysis of E-Scooter Sharing Services**



[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)

[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14+-139C5A?style=for-the-badge&logo=geopandas&logoColor=white)](https://geopandas.org)[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14+-139C5A?style=for-the-badge&logo=geopandas&logoColor=white)](https://geopandas.org)

[![DOI](https://img.shields.io/badge/DOI-10.34740%2FKAGGLE%2FDSV%2F14486163-blue?style=for-the-badge)](https://doi.org/10.34740/KAGGLE/DSV/14486163)[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)](#)



*Politecnico di Torino | Department of Regional and Urban Studies and Planning | 2024–2025**Politecnico di Torino | Transport Engineering | 2024-2025*



[Overview](#-overview) • [Key Findings](#-key-findings) • [Data](#-data-download) • [Installation](#-installation) • [Usage](#-usage) • [Citation](#-citation)[Overview](#overview) • [Key Findings](#key-findings) • [Methodology](#methodology) • [Installation](#installation) • [Usage](#usage)



</div></div>



------



## 📖 Overview## Overview



This repository presents a comprehensive data-driven investigation of shared micro-mobility patterns in **Turin, Italy**. We analyze **2,548,650 e-scooter trips** across three major operators (Lime, Bird, Voi) to understand urban mobility dynamics, service integration with public transport, and economic viability.This project presents a comprehensive data-driven investigation of shared micro-mobility patterns in Turin, Italy. We analyze **2,548,650 e-scooter trips** across three major operators (Lime, Bird, Voi) to understand service availability, pricing structures, and the relationship between e-scooters and public transport.



### Research Question### Research Objectives



> **"Are e-scooters competitors or complementary allies to public transport in Turin?"**1. **Service Level Analysis**: Compare fleet density, availability, and battery levels across operators

2. **Temporal Patterns**: Identify peak usage hours, weekly cycles, and seasonal trends

### Research Objectives3. **Public Transport Integration**: Determine if e-scooters compete with or complement public transit

4. **Economic Viability**: Evaluate operator profitability and cost structures

| # | Objective | Analysis Focus |5. **Optimal Mode Choice**: Recommend the best operator for specific commuting scenarios

|:-:|:----------|:---------------|

| 1 | **Service Level Analysis** | Fleet density, availability, battery levels |### Central Research Question

| 2 | **Temporal Patterns** | Peak hours, weekly cycles, seasonal trends |

| 3 | **Spatial Flows** | Origin-destination matrices, mobility corridors |> **"Are e-scooters competitors or allies to public transport in Turin?"**

| 4 | **PT Integration** | Proximity analysis, feeder service potential |

| 5 | **Economic Viability** | Revenue modeling, profitability assessment |---



---## Key Findings



## 📊 Key Findings### Summary Statistics



### Summary Metrics| Metric | Value | Interpretation |

|--------|-------|----------------|

| Metric | Value | Interpretation || **Total Trips Analyzed** | 2,548,650 | After data cleaning (87-99% retention) |

|:-------|------:|:---------------|| **Integration Index (200m)** | 99.7% | Near-universal proximity to PT stops |

| **Total Trips** | 2,548,650 | After cleaning (87-99% retention) || **Feeder Rate** | 95.1% | Strong first/last-mile connector role |

| **Integration Index** | 99.7% | Trips within 200m of PT stops || **Peak Hours** | 8-9 AM, 17-19 PM | Clear commuting patterns |

| **Feeder Rate** | 95.1% | First/last-mile connector role || **Market Size** | €8.30M/year | Combined operator revenue |

| **Peak Hours** | 8-9 AM, 17-19 PM | Clear commuting patterns || **Probability of Loss** | 0.28% | Very low financial risk (Monte Carlo) |

| **Market Size** | €8.30M/year | Combined operator revenue |

| **Loss Probability** | 0.28% | Monte Carlo simulation (n=10,000) |### Operator Comparison



### Operator Performance| Operator | Total Trips | Revenue (€/year) | Profit Margin | Market Share |

|----------|-------------|------------------|---------------|--------------|

| Operator | Trips | Revenue (€/yr) | Profit Margin | Market Share || **LIME** | 1,421,374 | 4,245,099 | 55.7% | 55.8% |

|:---------|------:|---------------:|--------------:|-------------:|| **BIRD** | 852,751 | 3,217,369 | 61.5% | 33.5% |

| 🟢 **Lime** | 1,421,374 | 4,245,099 | 55.7% | 55.8% || **VOI** | 274,525 | 837,898 | 53.5% | 10.8% |

| 🔵 **Bird** | 852,751 | 3,217,369 | 61.5% | 33.5% |

| 🟣 **Voi** | 274,525 | 837,898 | 53.5% | 10.8% |### Main Conclusion



### Main ConclusionE-scooters function primarily as **first/last-mile connectors** rather than direct competitors to public transport. Over 95% of trips start or end within 200 meters of a transit stop.



> E-scooters function primarily as **first/last-mile connectors** rather than direct competitors to public transport. Over 95% of trips originate or terminate within 200 meters of a transit stop.---



---## Methodology



## 🔬 Methodology### Study Area



### Study Area- **City**: Turin, Italy (Metropolitan Area)

- **Operators**: Lime, Bird, Voi

| Parameter | Value |- **Data Period**: 12 months (2023-2024)

|:----------|:------|- **Commute Route**: Codegone to Castello del Valentino (3.6 km)

| **City** | Turin (Torino), Metropolitan Area |

| **Operators** | Lime, Bird, Voi |### Snapshot Analysis Approach

| **Data Period** | January 2024 – November 2025 |

| **Sample Route** | Codegone → Castello del Valentino (3.6 km) |Our methodology follows a systematic snapshot analysis framework:



### Analysis Framework```

┌─────────────────────────────────────────────────────────────────┐

```│  1. DEFINE STUDY AREA                                           │

┌─────────────────────────────────────────────────────────────────┐│     └── Select zone where all operators are active (~1 km²)     │

│  1. DATA COLLECTION                                             │├─────────────────────────────────────────────────────────────────┤

│     └── Trip records, vehicle status, pricing from operator APIs││  2. DATA COLLECTION                                             │

├─────────────────────────────────────────────────────────────────┤│     └── Capture vehicle locations, battery, pricing from apps   │

│  2. DATA CLEANING                                               │├─────────────────────────────────────────────────────────────────┤

│     └── Geographic filtering, outlier removal, harmonization    ││  3. CALCULATE KPIs                                              │

├─────────────────────────────────────────────────────────────────┤│     └── Fleet density, market share, battery levels, costs      │

│  3. SPATIAL ANALYSIS                                            │├─────────────────────────────────────────────────────────────────┤

│     └── OD matrices, zone aggregation, PT proximity analysis    ││  4. APPLY TO COMMUTE                                            │

├─────────────────────────────────────────────────────────────────┤│     └── Calculate trip costs and generalized cost per operator  │

│  4. STATISTICAL MODELING                                        │└─────────────────────────────────────────────────────────────────┘

│     └── Survival analysis, Monte Carlo simulation, hypothesis   │```

├─────────────────────────────────────────────────────────────────┤

│  5. ECONOMIC ASSESSMENT                                         │### Key Performance Indicators

│     └── Revenue modeling, break-even analysis, risk assessment  │

└─────────────────────────────────────────────────────────────────┘| KPI | Formula | Purpose |

```|-----|---------|---------|

| **Fleet Density** | `Vehicles / Area (km²)` | Measures ease of finding a scooter |

### Statistical Methods| **Market Share** | `Operator Vehicles / Total Vehicles` | Shows operator dominance |

| **Average Battery** | `Mean(Battery Readings)` | Indicates fleet quality |

| Method | Application || **Trip Cost** | `Unlock Fee + (Duration × Rate)` | Determines value for money |

|:-------|:------------|| **Integration Index** | `Trips near PT / Total Trips` | PT complementarity |

| Kruskal-Wallis H-test | Cross-operator distribution comparison |

| Chi-square Test | Categorical independence testing |### Statistical Methods

| Monte Carlo Simulation | Risk analysis (10,000 iterations) |

| Kaplan-Meier Analysis | Parking duration survival curves |- **Kruskal-Wallis H-test**: Compare distributions across operators

| Weibull Distribution | Parametric survival modeling |- **Chi-square Test**: Categorical variable independence

| Moran's I | Spatial autocorrelation detection |- **Mann-Whitney U Test**: Pairwise non-parametric comparison

| STL Decomposition | Seasonal-trend decomposition |- **Monte Carlo Simulation**: 10,000 iterations for risk analysis

| Bootstrap CI | Robust confidence intervals (n=1,000) |- **Bootstrap Confidence Intervals**: 95% CI (n=1,000 resamples)

- **Kaplan-Meier Survival Analysis**: Parking duration modeling

---- **Weibull Distribution Fitting**: Parametric survival model

- **Log-rank Test**: Survival curve comparison

## 📁 Project Structure- **Moran's I**: Spatial autocorrelation detection

- **STL Decomposition**: Seasonal-trend analysis

```

Turin-MicromobilityAnalysis/---

│

├── 📄 run_pipeline.py           # Master pipeline controller## Project Structure

├── 📄 requirements.txt          # Python dependencies

├── 📄 README.md                 # This file```text

├── 📄 LICENSE                   # MIT LicenseTurin-MicromobilityAnalysis/

││

├── 📁 data/                     # ⬇️ Download from Kaggle├── run_pipeline.py              # Master pipeline controller

│   ├── processed/               # Cleaned trip data├── requirements.txt             # Python dependencies

│   └── raw/                     # Original files + GTFS + zones├── README.md                    # This file

│├── ARCHITECTURE.md              # Technical documentation

├── 📁 src/│

│   ├── 📁 analysis/             # Core analysis modules├── data/                        # Download from Kaggle (see below)

│   │   ├── 01_temporal_analysis.py│   ├── raw/                     # Raw e-scooter trip data

│   │   ├── 02_od_matrix_analysis.py│   └── processed/               # Cleaned and processed data

│   │   ├── 03_integration_analysis.py│

│   │   ├── 04_parking_analysis.py├── src/

│   │   └── 05_economic_analysis.py│   ├── analysis/                # Core analysis modules

│   ││   │   ├── 01_temporal_analysis.py      # Hourly/daily/monthly patterns

│   ├── 📁 data/│   │   ├── 02_od_matrix_analysis.py     # Origin-destination flows

│   │   └── 00_data_cleaning.py│   │   ├── 03_integration_analysis.py   # E-scooter & PT comparison

│   ││   │   ├── 04_parking_analysis.py       # Parking duration & turnover

│   ├── 📁 utils/                # Helper functions│   │   └── 05_economic_analysis.py      # Revenue & profitability

│   ││   │

│   └── 📁 visualization/        # Plotting scripts│   ├── data/

│       ├── 01_temporal_dashboard.py│   │   └── 00_data_cleaning.py          # Data preprocessing

│       ├── 02_od_spatial_flows.py│   │

│       ├── 03_integration_maps.py│   ├── utils/                   # Helper functions

│       ├── 04_parking_maps.py│   │

│       └── 05_economic_maps.py│   └── visualization/           # Plotting scripts

││       ├── 00_data_cleaning.py          # Data quality visualizations

└── 📁 outputs/                  # Generated on run│       ├── 01_temporal_dashboard.py     # Temporal pattern charts

    ├── figures/                 # Visualizations (PNG)│       ├── 02_od_spatial_flows.py       # Spatial flow maps

    └── reports/                 # Analysis checkpoints│       ├── 03_integration_maps.py       # PT integration maps

```│       ├── 04_parking_maps.py           # Parking analysis maps

│       └── 05_economic_maps.py          # Economic analysis charts

---│

└── outputs/                     # Generated outputs (created on run)

## 📥 Data Download    ├── figures/                 # Visualizations (PNG)

    └── reports/                 # Analysis checkpoints (CSV/PKL)

The complete dataset (~6.8 GB) is hosted on **Kaggle**.```



### Option 1: Kaggle CLI (Recommended)---



```bash## Data Download

# Install Kaggle CLI

pip install kaggleThe complete dataset (raw + processed) is hosted on Kaggle (~6.8 GB).



# Configure API key (see: kaggle.com/docs/api)### Option 1: Kaggle CLI (Recommended)

# Download and extract dataset

kaggle datasets download -d aliivaezii/turin-escooter-trips -p data/ --unzip```bash

```# Install Kaggle CLI

pip install kaggle

### Option 2: Manual Download

# Download dataset (requires Kaggle API key)

1. Visit: **[kaggle.com/datasets/aliivaezii/turin-escooter-trips](https://www.kaggle.com/datasets/aliivaezii/turin-escooter-trips)**kaggle datasets download -d aliivaezii/turin-escooter-trips -p data/ --unzip

2. Click **Download**```

3. Extract to `data/` folder

### Option 2: Manual Download

### Dataset Contents

1. Visit: [https://www.kaggle.com/datasets/aliivaezii/turin-escooter-trips](https://www.kaggle.com/datasets/aliivaezii/turin-escooter-trips)

<table>2. Click "Download" 

<tr>3. Extract contents to `data/` folder

<th>Processed Data (4.4 GB)</th>

<th>Raw Data (2.4 GB)</th>### Dataset Contents

</tr>

<tr>**Total Size**: ~6.8 GB

<td>

#### Processed Data (4.4 GB)

| File | Size | Records |

|:-----|:----:|--------:|| File | Size | Trips | Description |

| `lime_cleaned.csv` | 2.0 GB | 1.4M ||------|------|-------|-------------|

| `bird_cleaned.csv` | 106 MB | 850K || `processed/lime_cleaned.csv` | 2.0 GB | ~1.4M | Lime e-scooter trips (2024-2025) |

| `voi_cleaned.csv` | 53 MB | 275K || `processed/bird_cleaned.csv` | 106 MB | ~850K | Bird e-scooter trips (2024-2025) |

| `df_all.pkl` | 2.2 GB | 2.5M || `processed/voi_cleaned.csv` | 53 MB | ~275K | Voi e-scooter trips (2024-2025) |

| `processed/df_all.pkl` | 2.2 GB | ~2.5M | Combined DataFrame (pickle) |

</td>

<td>#### Raw Data (2.4 GB)



| Folder | Size | Contents || Folder | Size | Description |

|:-------|:----:|:---------||--------|------|-------------|

| `lime/` | 2.2 GB | CSV + monthly || `raw/lime/` | 2.2 GB | Original Lime CSV with route paths + 16 monthly files |

| `bird/` | 86 MB | 2024 + 2025 CSV || `raw/bird/` | 86 MB | Original Bird CSV files (2024 + 2025) |

| `voi/` | 28 MB | 22 Excel files || `raw/voi/` | 28 MB | 22 monthly Excel files (Jan 2024 - Oct 2025) |

| `gtfs/` | 70 MB | PT network || `raw/gtfs/` | 70 MB | Turin GTFS public transport data |

| `zone_statistiche/` | 344 KB | Shapefile || `raw/zone_statistiche_geo/` | 344 KB | Turin statistical zones shapefile |



</td>### Data Directory Structure

</tr>

</table>After downloading, your `data/` folder should look like:



---```

data/

## ⚙️ Installation├── processed/

│   ├── lime_cleaned.csv

### Prerequisites│   ├── bird_cleaned.csv

│   ├── voi_cleaned.csv

- Python 3.10+│   └── df_all.pkl

- pip package manager└── raw/

- Git    ├── lime/

    │   ├── Torino_Corse24-25.csv

### Quick Start    │   └── monthly/

    ├── bird/

```bash    │   ├── Bird Torino - 2024 - Sheet1.csv

# Clone repository    │   └── Bird Torino - 2025 (fino il 18_11_2025) - Sheet1.csv

git clone https://github.com/aliivaezii/Turin-MicromobilityAnalysis.git    ├── voi/

cd Turin-MicromobilityAnalysis    │   └── DATINOLEGGI_YYYYMM.xlsx (22 files)

    ├── gtfs/

# Create virtual environment    │   ├── stops.txt

python -m venv .venv    │   ├── routes.txt

source .venv/bin/activate  # macOS/Linux    │   └── ...

# .venv\Scripts\activate   # Windows    └── zone_statistiche_geo/

        ├── zone_statistiche_geo.shp

# Install dependencies        └── ...

pip install -r requirements.txt```



# Download data from Kaggle (see above)---

```

## Analysis Pipeline

### Core Dependencies

### Exercise 1: Temporal Pattern Analysis

| Package | Version | Purpose |

|:--------|:-------:|:--------|Analyzes hourly, daily, and monthly usage patterns for each operator.

| pandas | ≥2.0.0 | Data manipulation |

| numpy | ≥1.24.0 | Numerical computing |**Methods**:

| scipy | ≥1.10.0 | Statistical analysis |- Seasonal-Trend decomposition using Loess (STL)

| geopandas | ≥0.14.0 | Geospatial analysis |- Peak detection using scipy signal processing

| shapely | ≥2.0.0 | Geometric operations |- Statistical comparison with Kruskal-Wallis test

| matplotlib | ≥3.7.0 | Visualization |

| seaborn | ≥0.12.0 | Statistical graphics |**Key Outputs**:

| contextily | ≥1.3.0 | Basemap tiles |- Hourly trip distribution with confidence intervals

- Weekly usage heatmaps

---- Monthly trend analysis

- Fleet utilization rates by operator

## 🚀 Usage

### Exercise 2: Origin-Destination Matrix

### Run Complete Pipeline

Maps mobility corridors and zone-to-zone flows across Turin.

```bash

python run_pipeline.py**Methods**:

```- Spatial joining to official Turin Zone Statistiche

- Advanced OD metrics (Gini, Entropy, Flow Asymmetry)

### Run Specific Stages- Chi-square tests for temporal independence

- Hierarchical clustering for zone grouping

```bash

# Run selected exercises**Key Outputs**:

python run_pipeline.py --stages 1 2 3- OD flow matrices (zone-to-zone)

- Spatial flow maps with arrow thickness

# Run from specific stage- Trip density hexbin maps

python run_pipeline.py --from-stage 3- Top corridor identification



# Visualization only### Exercise 3: Public Transport Integration

python run_pipeline.py --viz-only

Evaluates e-scooter proximity to public transport stops.

# Analysis only (no plots)

python run_pipeline.py --no-viz**Methods**:

```- Multi-buffer sensitivity analysis (50m, 100m, 200m)

- Moran's I for spatial autocorrelation

### Pipeline Stages- Chi-square tests for integration independence

- Permutation tests (1,000 iterations)

| Stage | Module | Description | Runtime |

|:-----:|:-------|:------------|--------:|**Key Outputs**:

| 0 | Data Cleaning | Harmonize raw data | ~2 min |- Integration index by buffer distance

| 1 | Temporal Analysis | Hourly/daily/monthly patterns | ~5 min |- Competition map (PT-parallel routes)

| 2 | OD Matrix | Origin-destination flows | ~8 min |- Feeder service identification

| 3 | Integration | E-scooter ↔ PT comparison | ~10 min |- Temporal comparison with PT schedules

| 4 | Parking | Duration & turnover analysis | ~5 min |

| 5 | Economic | Revenue & profitability | ~15 min |### Exercise 4: Parking Analysis



---Studies parking duration patterns and turnover rates.



## 💰 Cost Comparison**Methods**:

- Kaplan-Meier survival curves with 95% CI

**Commute Scenario**: Home → University (18 minutes)- Weibull distribution fitting (MLE)

- Log-rank test for group comparisons

| Operator | Unlock | Rate/min | Trip Cost | Monthly (44 trips) |- Ghost vehicle detection (>120 hours)

|:---------|-------:|---------:|----------:|-------------------:|

| 🟢 Lime | €1.00 | €0.25 | €5.50 | €242.00 |**Key Outputs**:

| 🔵 Bird | €1.00 | €0.19 | €4.42 | €194.48 |- Parking duration histograms

| 🟣 Voi | €1.00 | €0.19 | €4.42 | €194.48 |- Survival curves by operator

- Parking intensity heatmaps

### Recommendations- Turnover vs. demand scatter plots



| Use Case | Best Choice | Reason |### Exercise 5: Economic Analysis

|:---------|:------------|:-------|

| Occasional (< 10/mo) | Any | Similar pricing |Calculates revenue, fleet economics, and profitability.

| Regular commute | Lime Pro | Subscription value |

| Short trips (< 5 min) | Bird/Voi | Lower per-minute rate |**Methods**:

| High availability | Lime | Largest fleet |- Monte Carlo simulation (10,000 iterations)

- Break-even analysis with sensitivity testing

---- Pareto analysis (80/20 rule for zones)

- Scenario modeling (base/optimistic/pessimistic)

## 📖 Citation

**Key Outputs**:

If you use this code or dataset in your research, please cite:- Operator P&L waterfall charts

- Revenue yield maps

```bibtex- Break-even scatter plots

@misc{ali_vaezi_2026,- Sensitivity tornado diagrams

    title     = {Turin Escooter Trips 2024},

    author    = {Ali Vaezi},---

    year      = {2026},

    url       = {https://www.kaggle.com/dsv/14486163},## Installation

    doi       = {10.34740/KAGGLE/DSV/14486163},

    publisher = {Kaggle}### Prerequisites

}

```- Python 3.10 or higher

- pip package manager

**DOI**: [10.34740/KAGGLE/DSV/14486163](https://doi.org/10.34740/KAGGLE/DSV/14486163)- Git



---### Quick Start



## 🤝 Contributing```bash

# Clone the repository

Contributions are welcome! Please follow these steps:git clone https://github.com/aliivaezii/Turin-MicromobilityAnalysis.git

cd Turin-MicromobilityAnalysis

1. Fork the repository

2. Create a feature branch (`git checkout -b feature/new-analysis`)# Create and activate virtual environment

3. Commit changes (`git commit -m 'Add new analysis module'`)python -m venv .venv

4. Push to branch (`git push origin feature/new-analysis`)source .venv/bin/activate  # macOS/Linux

5. Open a Pull Request# .venv\Scripts\activate   # Windows



---# Install dependencies

pip install -r requirements.txt

## 👤 Author```



<table>### Dependencies

<tr>

<td>| Package | Version | Purpose |

|---------|---------|---------|

**Ali Vaezi**  | pandas | ≥2.0.0 | Data manipulation |

Interuniversity Department of Regional and Urban Studies and Planning  | numpy | ≥1.24.0 | Numerical computing |

Politecnico di Torino  | scipy | ≥1.10.0 | Statistical analysis |

Corso Duca degli Abruzzi, 24, Turin 10129, Italy| geopandas | ≥0.14.0 | Geospatial analysis |

| shapely | ≥2.0.0 | Geometric operations |

📧 [ali.vaezi@studenti.polito.it](mailto:ali.vaezi@studenti.polito.it)| matplotlib | ≥3.7.0 | Visualization |

| seaborn | ≥0.12.0 | Statistical graphics |

</td>| contextily | ≥1.3.0 | Basemap tiles |

<td>| tqdm | ≥4.65.0 | Progress bars |



[![GitHub](https://img.shields.io/badge/GitHub-aliivaezii-181717?style=for-the-badge&logo=github)](https://github.com/aliivaezii)---

[![Kaggle](https://img.shields.io/badge/Kaggle-aliivaezii-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/aliivaezii)

## Usage

</td>

</tr>### Run Complete Pipeline

</table>

```bash

---python run_pipeline.py

```

## 📜 License

### Run Specific Exercises

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

```bash

---# Run only exercises 1, 2, and 3

python run_pipeline.py --stages 1 2 3

## 🙏 Acknowledgments

# Run from exercise 3 onwards

- **Politecnico di Torino** — Department of Regional and Urban Studies and Planningpython run_pipeline.py --from-stage 3

- **Turin Open Data Portal** — Public transport and administrative data

- **Lime, Bird, Voi** — E-scooter trip data access# Run only visualizations (skip analysis)

- Course instructors for methodology guidancepython run_pipeline.py --viz-only



---# Run analysis without visualizations

python run_pipeline.py --no-viz

<div align="center">```



**⭐ Star this repository if you find it useful!**### Pipeline Stages



</div>| Stage | Module | Description | Runtime |

|-------|--------|-------------|---------|
| 0 | Data Cleaning | Clean and harmonize raw data | ~2 min |
| 1 | Temporal Analysis | Hourly/daily/monthly patterns | ~5 min |
| 2 | OD Matrix | Origin-destination flow analysis | ~8 min |
| 3 | Integration | E-scooter and PT comparison | ~10 min |
| 4 | Parking | Duration and turnover analysis | ~5 min |
| 5 | Economic | Revenue and profitability | ~15 min |

---

## Cost Comparison for Commuters

For a typical Home-to-University commute (Codegone to Valentino, 18 minutes):

| Operator | Unlock Fee | Rate/min | Trip Cost | Monthly (44 trips) |
|----------|------------|----------|-----------|-------------------|
| **Lime** | €1.00 | €0.25 | €5.50 | €242.00 |
| **Bird** | €1.00 | €0.19 | €4.42 | €194.48 |
| **Voi** | €1.00 | €0.19 | €4.42 | €194.48 |

### Subscription Recommendations

| Scenario | Best Choice | Reason |
|----------|-------------|--------|
| **Occasional use** (< 10 trips/month) | Any operator | Similar pricing |
| **Regular commute** (20+ trips/month) | Lime Pro subscription | Fixed monthly fee |
| **Short trips** (< 5 min) | Bird or Voi | Lower per-minute rate |
| **High availability priority** | Lime | Largest fleet |

---

## Technical Notes

### Data Quality

- Raw data cleaned using multi-stage pipeline
- Geographic filtering to Turin metropolitan boundary
- Temporal filtering for valid trip durations (1-60 minutes)
- Battery level validation (0-100%)

### Statistical Rigor

- All confidence intervals at 95% level
- Effect sizes reported (Cohen's d, eta-squared)
- Non-parametric tests used for non-normal distributions
- Bootstrap resampling for robust estimates

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Commit changes (`git commit -m 'Add new analysis module'`)
4. Push to branch (`git push origin feature/new-analysis`)
5. Open a Pull Request

---

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{ali_vaezi_2026,
  title = {Turin Escooter Trips 2024},
  url = {https://www.kaggle.com/dsv/14486163},
  DOI = {10.34740/KAGGLE/DSV/14486163},
  publisher = {Kaggle},
  author = {Ali Vaezi},
  year = {2026}
}
```

**DOI**: [10.34740/KAGGLE/DSV/14486163](https://doi.org/10.34740/KAGGLE/DSV/14486163)

---

## Author

**Ali Vaezi**  
Interuniversity Department of Regional and Urban Studies and Planning  
Politecnico di Torino  
Corso Duca degli Abruzzi, 24, Turin 10129, Italy  
📧 ali.vaezi@studenti.polito.it

[![GitHub](https://img.shields.io/badge/GitHub-aliivaezii-181717?style=flat&logo=github)](https://github.com/aliivaezii)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Politecnico di Torino**, Department of Transport Engineering
- **Turin Open Data Portal** for public transport stop locations
- **Lime, Bird, and Voi** for providing accessible mobile applications
- Course instructors for methodology guidance

