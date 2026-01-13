<div align="center">

# Turin Micromobility Analysis

**Comprehensive Spatial-Temporal Analysis of E-Scooter Sharing Services**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14+-139C5A?style=for-the-badge&logo=geopandas&logoColor=white)](https://geopandas.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)](#)

*Politecnico di Torino | Transport Engineering | 2024-2025*

[Overview](#overview) • [Key Findings](#key-findings) • [Methodology](#methodology) • [Installation](#installation) • [Usage](#usage)

</div>

---

## Overview

This project presents a comprehensive data-driven investigation of shared micro-mobility patterns in Turin, Italy. We analyze **2,548,650 e-scooter trips** across three major operators (Lime, Bird, Voi) to understand service availability, pricing structures, and the relationship between e-scooters and public transport.

### Research Objectives

1. **Service Level Analysis**: Compare fleet density, availability, and battery levels across operators
2. **Temporal Patterns**: Identify peak usage hours, weekly cycles, and seasonal trends
3. **Public Transport Integration**: Determine if e-scooters compete with or complement public transit
4. **Economic Viability**: Evaluate operator profitability and cost structures
5. **Optimal Mode Choice**: Recommend the best operator for specific commuting scenarios

### Central Research Question

> **"Are e-scooters competitors or allies to public transport in Turin?"**

---

## Key Findings

### Summary Statistics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Total Trips Analyzed** | 2,548,650 | After data cleaning (87-99% retention) |
| **Integration Index (200m)** | 99.7% | Near-universal proximity to PT stops |
| **Feeder Rate** | 95.1% | Strong first/last-mile connector role |
| **Peak Hours** | 8-9 AM, 17-19 PM | Clear commuting patterns |
| **Market Size** | €8.30M/year | Combined operator revenue |
| **Probability of Loss** | 0.28% | Very low financial risk (Monte Carlo) |

### Operator Comparison

| Operator | Total Trips | Revenue (€/year) | Profit Margin | Market Share |
|----------|-------------|------------------|---------------|--------------|
| **LIME** | 1,421,374 | 4,245,099 | 55.7% | 55.8% |
| **BIRD** | 852,751 | 3,217,369 | 61.5% | 33.5% |
| **VOI** | 274,525 | 837,898 | 53.5% | 10.8% |

### Main Conclusion

E-scooters function primarily as **first/last-mile connectors** rather than direct competitors to public transport. Over 95% of trips start or end within 200 meters of a transit stop.

---

## Methodology

### Study Area

- **City**: Turin, Italy (Metropolitan Area)
- **Operators**: Lime, Bird, Voi
- **Data Period**: 12 months (2023-2024)
- **Commute Route**: Codegone to Castello del Valentino (3.6 km)

### Snapshot Analysis Approach

Our methodology follows a systematic snapshot analysis framework:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. DEFINE STUDY AREA                                           │
│     └── Select zone where all operators are active (~1 km²)     │
├─────────────────────────────────────────────────────────────────┤
│  2. DATA COLLECTION                                             │
│     └── Capture vehicle locations, battery, pricing from apps   │
├─────────────────────────────────────────────────────────────────┤
│  3. CALCULATE KPIs                                              │
│     └── Fleet density, market share, battery levels, costs      │
├─────────────────────────────────────────────────────────────────┤
│  4. APPLY TO COMMUTE                                            │
│     └── Calculate trip costs and generalized cost per operator  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Performance Indicators

| KPI | Formula | Purpose |
|-----|---------|---------|
| **Fleet Density** | `Vehicles / Area (km²)` | Measures ease of finding a scooter |
| **Market Share** | `Operator Vehicles / Total Vehicles` | Shows operator dominance |
| **Average Battery** | `Mean(Battery Readings)` | Indicates fleet quality |
| **Trip Cost** | `Unlock Fee + (Duration × Rate)` | Determines value for money |
| **Integration Index** | `Trips near PT / Total Trips` | PT complementarity |

### Statistical Methods

- **Kruskal-Wallis H-test**: Compare distributions across operators
- **Chi-square Test**: Categorical variable independence
- **Mann-Whitney U Test**: Pairwise non-parametric comparison
- **Monte Carlo Simulation**: 10,000 iterations for risk analysis
- **Bootstrap Confidence Intervals**: 95% CI (n=1,000 resamples)
- **Kaplan-Meier Survival Analysis**: Parking duration modeling
- **Weibull Distribution Fitting**: Parametric survival model
- **Log-rank Test**: Survival curve comparison
- **Moran's I**: Spatial autocorrelation detection
- **STL Decomposition**: Seasonal-trend analysis

---

## Project Structure

```text
Turin-MicromobilityAnalysis/
│
├── run_pipeline.py              # Master pipeline controller
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── ARCHITECTURE.md              # Technical documentation
│
├── data/                        # Download from Kaggle (see below)
│   ├── raw/                     # Raw e-scooter trip data
│   └── processed/               # Cleaned and processed data
│
├── src/
│   ├── analysis/                # Core analysis modules
│   │   ├── 01_temporal_analysis.py      # Hourly/daily/monthly patterns
│   │   ├── 02_od_matrix_analysis.py     # Origin-destination flows
│   │   ├── 03_integration_analysis.py   # E-scooter & PT comparison
│   │   ├── 04_parking_analysis.py       # Parking duration & turnover
│   │   └── 05_economic_analysis.py      # Revenue & profitability
│   │
│   ├── data/
│   │   └── 00_data_cleaning.py          # Data preprocessing
│   │
│   ├── utils/                   # Helper functions
│   │
│   └── visualization/           # Plotting scripts
│       ├── 00_data_cleaning.py          # Data quality visualizations
│       ├── 01_temporal_dashboard.py     # Temporal pattern charts
│       ├── 02_od_spatial_flows.py       # Spatial flow maps
│       ├── 03_integration_maps.py       # PT integration maps
│       ├── 04_parking_maps.py           # Parking analysis maps
│       └── 05_economic_maps.py          # Economic analysis charts
│
└── outputs/                     # Generated outputs (created on run)
    ├── figures/                 # Visualizations (PNG)
    └── reports/                 # Analysis checkpoints (CSV/PKL)
```

---

## Data Download

The complete dataset (raw + processed) is hosted on Kaggle (~6.8 GB).

### Option 1: Kaggle CLI (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API key)
kaggle datasets download -d aliivaezii/turin-escooter-trips -p data/ --unzip
```

### Option 2: Manual Download

1. Visit: [https://www.kaggle.com/datasets/aliivaezii/turin-escooter-trips](https://www.kaggle.com/datasets/aliivaezii/turin-escooter-trips)
2. Click "Download" 
3. Extract contents to `data/` folder

### Dataset Contents

**Total Size**: ~6.8 GB

#### Processed Data (4.4 GB)

| File | Size | Trips | Description |
|------|------|-------|-------------|
| `processed/lime_cleaned.csv` | 2.0 GB | ~1.4M | Lime e-scooter trips (2024-2025) |
| `processed/bird_cleaned.csv` | 106 MB | ~850K | Bird e-scooter trips (2024-2025) |
| `processed/voi_cleaned.csv` | 53 MB | ~275K | Voi e-scooter trips (2024-2025) |
| `processed/df_all.pkl` | 2.2 GB | ~2.5M | Combined DataFrame (pickle) |

#### Raw Data (2.4 GB)

| Folder | Size | Description |
|--------|------|-------------|
| `raw/lime/` | 2.2 GB | Original Lime CSV with route paths + 16 monthly files |
| `raw/bird/` | 86 MB | Original Bird CSV files (2024 + 2025) |
| `raw/voi/` | 28 MB | 22 monthly Excel files (Jan 2024 - Oct 2025) |
| `raw/gtfs/` | 70 MB | Turin GTFS public transport data |
| `raw/zone_statistiche_geo/` | 344 KB | Turin statistical zones shapefile |

### Data Directory Structure

After downloading, your `data/` folder should look like:

```
data/
├── processed/
│   ├── lime_cleaned.csv
│   ├── bird_cleaned.csv
│   ├── voi_cleaned.csv
│   └── df_all.pkl
└── raw/
    ├── lime/
    │   ├── Torino_Corse24-25.csv
    │   └── monthly/
    ├── bird/
    │   ├── Bird Torino - 2024 - Sheet1.csv
    │   └── Bird Torino - 2025 (fino il 18_11_2025) - Sheet1.csv
    ├── voi/
    │   └── DATINOLEGGI_YYYYMM.xlsx (22 files)
    ├── gtfs/
    │   ├── stops.txt
    │   ├── routes.txt
    │   └── ...
    └── zone_statistiche_geo/
        ├── zone_statistiche_geo.shp
        └── ...
```

---

## Analysis Pipeline

### Exercise 1: Temporal Pattern Analysis

Analyzes hourly, daily, and monthly usage patterns for each operator.

**Methods**:
- Seasonal-Trend decomposition using Loess (STL)
- Peak detection using scipy signal processing
- Statistical comparison with Kruskal-Wallis test

**Key Outputs**:
- Hourly trip distribution with confidence intervals
- Weekly usage heatmaps
- Monthly trend analysis
- Fleet utilization rates by operator

### Exercise 2: Origin-Destination Matrix

Maps mobility corridors and zone-to-zone flows across Turin.

**Methods**:
- Spatial joining to official Turin Zone Statistiche
- Advanced OD metrics (Gini, Entropy, Flow Asymmetry)
- Chi-square tests for temporal independence
- Hierarchical clustering for zone grouping

**Key Outputs**:
- OD flow matrices (zone-to-zone)
- Spatial flow maps with arrow thickness
- Trip density hexbin maps
- Top corridor identification

### Exercise 3: Public Transport Integration

Evaluates e-scooter proximity to public transport stops.

**Methods**:
- Multi-buffer sensitivity analysis (50m, 100m, 200m)
- Moran's I for spatial autocorrelation
- Chi-square tests for integration independence
- Permutation tests (1,000 iterations)

**Key Outputs**:
- Integration index by buffer distance
- Competition map (PT-parallel routes)
- Feeder service identification
- Temporal comparison with PT schedules

### Exercise 4: Parking Analysis

Studies parking duration patterns and turnover rates.

**Methods**:
- Kaplan-Meier survival curves with 95% CI
- Weibull distribution fitting (MLE)
- Log-rank test for group comparisons
- Ghost vehicle detection (>120 hours)

**Key Outputs**:
- Parking duration histograms
- Survival curves by operator
- Parking intensity heatmaps
- Turnover vs. demand scatter plots

### Exercise 5: Economic Analysis

Calculates revenue, fleet economics, and profitability.

**Methods**:
- Monte Carlo simulation (10,000 iterations)
- Break-even analysis with sensitivity testing
- Pareto analysis (80/20 rule for zones)
- Scenario modeling (base/optimistic/pessimistic)

**Key Outputs**:
- Operator P&L waterfall charts
- Revenue yield maps
- Break-even scatter plots
- Sensitivity tornado diagrams

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/aliivaezii/Turin-MicromobilityAnalysis.git
cd Turin-MicromobilityAnalysis

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical computing |
| scipy | ≥1.10.0 | Statistical analysis |
| geopandas | ≥0.14.0 | Geospatial analysis |
| shapely | ≥2.0.0 | Geometric operations |
| matplotlib | ≥3.7.0 | Visualization |
| seaborn | ≥0.12.0 | Statistical graphics |
| contextily | ≥1.3.0 | Basemap tiles |
| tqdm | ≥4.65.0 | Progress bars |

---

## Usage

### Run Complete Pipeline

```bash
python run_pipeline.py
```

### Run Specific Exercises

```bash
# Run only exercises 1, 2, and 3
python run_pipeline.py --stages 1 2 3

# Run from exercise 3 onwards
python run_pipeline.py --from-stage 3

# Run only visualizations (skip analysis)
python run_pipeline.py --viz-only

# Run analysis without visualizations
python run_pipeline.py --no-viz
```

### Pipeline Stages

| Stage | Module | Description | Runtime |
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
Politecnico di Torino, Italy

[![GitHub](https://img.shields.io/badge/GitHub-aliivaezii-181717?style=flat&logo=github)](https://github.com/aliivaezii)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


