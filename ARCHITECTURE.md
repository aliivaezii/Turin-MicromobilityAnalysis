# Technical Architecture

## Turin Micromobility Analysis - System Design Document

**Version 3.0** | **December 2025** | **Politecnico di Torino**

---

## Document Purpose

This document provides the complete technical specification for the Turin Micromobility analysis pipeline. It serves as a reference for:

- **Developers**: Extending or maintaining the codebase
- **Reviewers**: Assessing the technical rigor of the methodology
- **Researchers**: Replicating the analysis for other cities

---

## System Architecture Overview

### High-Level Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TURIN MICROMOBILITY PIPELINE v3.0                        │
└─────────────────────────────────────────────────────────────────────────────┘

  RAW DATA                 PREPROCESSING              ANALYSIS STAGES
  ════════                 ═════════════              ═══════════════
  
  ┌─────────────┐         ┌─────────────┐           ┌─────────────────────────┐
  │ LIME CSV    │         │             │           │  STAGE 1: TEMPORAL      │
  │ (1.62M raw) │────────▶│   Data      │──────────▶│  01_temporal_analysis   │
  ├─────────────┤         │  Cleaning   │           ├─────────────────────────┤
  │ VOI XLSX    │────────▶│             │           │  STAGE 2: OD MATRIX     │
  │ (18 files)  │         │  Retention: │           │  02_od_matrix_analysis  │
  ├─────────────┤         │  87-99%     │           ├─────────────────────────┤
  │ BIRD CSV    │────────▶│             │           │  STAGE 3: INTEGRATION   │
  │ (858K raw)  │         └─────────────┘           │  03_integration_analysis│
  └─────────────┘               │                   ├─────────────────────────┤
                                │                   │  STAGE 4: PARKING       │
  ┌─────────────┐               │                   │  04_parking_analysis    │
  │ GTFS Bundle │───────────────┼──────────────────▶├─────────────────────────┤
  │ (stops.txt) │               │                   │  STAGE 5: ECONOMICS     │
  ├─────────────┤               │                   │  05_economic_analysis   │
  │ Zone SHP    │───────────────┘                   └─────────────────────────┘
  │ (94 zones)  │                                              │
  └─────────────┘                                              ▼
                          ┌────────────────────────────────────────────────────┐
                          │                    OUTPUTS                         │
                          ├────────────────┬────────────────┬──────────────────┤
                          │   figures/     │   reports/     │    tables/       │
                          │   (PNG)        │   (CSV/PKL)    │    (CSV)         │
                          └────────────────┴────────────────┴──────────────────┘
```

---

## Data Sources

### Raw Data Summary

| Source | Format | Records (Raw) | Records (Clean) | Retention |
|--------|--------|---------------|-----------------|-----------|
| LIME | CSV | 1,624,528 | 1,421,374 | 87.5% |
| BIRD | CSV | 858,020 | 852,751 | 99.4% |
| VOI | XLSX (18 files) | 291,765 | 274,525 | 94.1% |
| **Total** | - | **2,774,313** | **2,548,650** | **91.9%** |

### Auxiliary Data

| Dataset | Format | Description |
|---------|--------|-------------|
| GTFS Bundle | TXT | Public transport stops, routes, schedules |
| Zone Statistiche | Shapefile | 94 official Turin statistical zones |
| Turin Boundary | GeoJSON | Metropolitan area boundary |

### Data Schema (Harmonized)

After preprocessing, all operator data conforms to this schema:

| Column | Type | Description |
|--------|------|-------------|
| `trip_id` | string | Unique trip identifier |
| `operator` | string | LIME, BIRD, or VOI |
| `start_time` | datetime | Trip start timestamp |
| `end_time` | datetime | Trip end timestamp |
| `duration_min` | float | Trip duration in minutes |
| `start_lat` | float | Origin latitude (WGS84) |
| `start_lon` | float | Origin longitude (WGS84) |
| `end_lat` | float | Destination latitude (WGS84) |
| `end_lon` | float | Destination longitude (WGS84) |
| `distance_m` | float | Trip distance in meters |

---

## Directory Structure

```text
Turin-MicromobilityAnalysis/
│
├── run_pipeline.py                   # Master pipeline controller
├── requirements.txt                  # Python dependencies
├── README.md                         # Project overview
├── ARCHITECTURE.md                   # This file
│
├── src/
│   ├── analysis/                     # Core analysis modules
│   │   ├── 01_temporal_analysis.py   # Exercise 1: Temporal patterns
│   │   ├── 02_od_matrix_analysis.py  # Exercise 2: O-D flows
│   │   ├── 03_integration_analysis.py# Exercise 3: PT integration
│   │   ├── 04_parking_analysis.py    # Exercise 4: Parking duration
│   │   └── 05_economic_analysis.py   # Exercise 5: Economics
│   │
│   ├── visualization/                # Plotting modules
│   │   ├── 01_temporal_dashboard.py
│   │   ├── 01_temporal_statistics.py
│   │   ├── 02_od_spatial_flows.py
│   │   ├── 02_od_statistics.py
│   │   ├── 03_integration_maps.py
│   │   ├── 03_integration_statistics.py
│   │   ├── 04_parking_maps.py
│   │   ├── 04_parking_survival.py
│   │   ├── 05_economic_maps.py
│   │   └── 05_economic_sensitivity.py
│   │
│   ├── data/
│   │   └── 00_data_cleaning.py       # ETL pipeline
│   │
│   └── utils/                        # Shared utilities
│
├── data/                             # Data directory (git-ignored)
│   ├── raw/                          # Original operator data
│   │   ├── bird/
│   │   ├── lime/
│   │   ├── voi/
│   │   ├── gtfs/
│   │   └── zone_statistiche_geo/
│   └── processed/                    # Cleaned datasets
│
├── outputs/
│   ├── figures/                      # Generated visualizations
│   │   ├── exercise0/
│   │   ├── exercise1/
│   │   ├── exercise2/
│   │   ├── exercise3/
│   │   ├── exercise4/
│   │   └── exercise5/
│   │
│   ├── reports/                      # Checkpoints and reports
│   │   ├── exercise1/
│   │   ├── exercise2/
│   │   ├── exercise3/
│   │   ├── exercise4/
│   │   └── exercise5/
│   │
│   └── tables/                       # Statistical summary tables
│       ├── exercise0/
│       ├── exercise1/
│       ├── exercise2/
│       ├── exercise3/
│       ├── exercise4/
│       └── exercise5/
│
└── docs/                             # Additional documentation
```

---

## Decoupled Design Pattern

### Architecture Philosophy

The pipeline uses a decoupled architecture that separates computation from visualization:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DECOUPLED ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ANALYSIS LAYER                        VISUALIZATION LAYER                 │
│   src/analysis/                         src/visualization/                  │
│   ══════════════                        ══════════════════                  │
│                                                                             │
│   01_temporal_analysis.py    ─────▶     01_temporal_dashboard.py            │
│   02_od_matrix_analysis.py   ─────▶     02_od_spatial_flows.py              │
│   03_integration_analysis.py ─────▶     03_integration_maps.py              │
│   04_parking_analysis.py     ─────▶     04_parking_maps.py                  │
│   05_economic_analysis.py    ─────▶     05_economic_maps.py                 │
│                                                                             │
│   Runtime: 5-15 min each           CHECKPOINTS       Runtime: 1-2 min each │
│   CPU-bound (computation)           (PKL/CSV)         I/O-bound (plotting) │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Benefits

| Aspect | Monolithic Script | Decoupled Architecture |
|--------|-------------------|------------------------|
| Plot iteration time | 30+ min | 1-2 min |
| Parallel development | Difficult | Easy |
| Testing | Complex | Modular |
| Debugging | Slow | Fast |

---

## Analysis Modules

### Exercise 1: Temporal Pattern Analysis

**File**: `src/analysis/01_temporal_analysis.py`

**Purpose**: Analyze hourly, daily, and monthly usage patterns for each operator.

**Statistical Methods**:

| Method | Implementation | Purpose |
|--------|----------------|---------|
| STL Decomposition | `statsmodels.tsa.seasonal.STL` | Seasonal-trend separation |
| Peak Detection | `scipy.signal.find_peaks` | Identify usage peaks |
| Kruskal-Wallis H-test | `scipy.stats.kruskal` | Compare operator distributions |
| Mann-Whitney U | `scipy.stats.mannwhitneyu` | Pairwise comparison |
| Cohen's d | Custom function | Effect size calculation |

**Outputs**:

| File | Description |
|------|-------------|
| `table01_descriptive_stats.csv` | Mean, median, std by operator |
| `table02_statistical_tests.csv` | Test results with p-values |
| `table03_peak_hours.csv` | Peak hour identification |
| `checkpoint_exercise1.pkl` | Pickle checkpoint for visualization |

---

### Exercise 2: Origin-Destination Matrix

**File**: `src/analysis/02_od_matrix_analysis.py`

**Purpose**: Map mobility corridors and zone-to-zone flows across Turin.

**Statistical Methods**:

| Method | Implementation | Purpose |
|--------|----------------|---------|
| Spatial Join | `geopandas.sjoin` | Assign trips to zones |
| Chi-square Test | `scipy.stats.chi2_contingency` | Temporal independence |
| Gini Coefficient | Custom function | Flow concentration |
| Entropy | Custom function | Flow diversity |
| Hierarchical Clustering | `scipy.cluster.hierarchy` | Zone grouping |

**Outputs**:

| File | Description |
|------|-------------|
| `table01_top_od_pairs.csv` | Top 20 O-D corridors |
| `table02_zone_statistics.csv` | Zone-level metrics |
| `table03_gravity_parameters.csv` | Gravity model parameters |
| `checkpoint_od_matrix.pkl` | Full O-D matrix |

---

### Exercise 3: Public Transport Integration

**File**: `src/analysis/03_integration_analysis.py`

**Purpose**: Evaluate e-scooter proximity to public transport stops.

**Statistical Methods**:

| Method | Implementation | Purpose |
|--------|----------------|---------|
| Multi-Buffer Analysis | `shapely.buffer` | 50m, 100m, 200m catchments |
| Moran's I | `scipy` + custom | Spatial autocorrelation |
| Chi-square Test | `scipy.stats.chi2_contingency` | Integration independence |
| Permutation Test | Custom (n=1000) | Non-parametric significance |
| Cramér's V | Custom function | Effect size |

**Key Metrics**:

| Metric | Formula | Value (200m) |
|--------|---------|--------------|
| Integration Index | Trips within buffer / Total trips | 99.7% |
| Feeder Rate | First/last-mile trips / Total | 95.1% |

**Outputs**:

| File | Description |
|------|-------------|
| `table01_integration_summary.csv` | Integration metrics by operator |
| `table02_buffer_analysis.csv` | Results at different buffer distances |
| `table03_temporal_comparison.csv` | Peak vs off-peak integration |

---

### Exercise 4: Parking Duration Analysis

**File**: `src/analysis/04_parking_analysis.py`

**Purpose**: Study parking duration patterns and turnover rates.

**Statistical Methods**:

| Method | Implementation | Purpose |
|--------|----------------|---------|
| Kaplan-Meier | `lifelines.KaplanMeierFitter` | Survival curves |
| Weibull Fitting | `lifelines.WeibullFitter` | Parametric model |
| Log-rank Test | `lifelines.statistics.logrank_test` | Curve comparison |
| Bootstrap CI | Custom (n=1000) | Confidence intervals |
| Moran's I | Custom | Spatial autocorrelation |

**Key Metrics**:

| Metric | Description |
|--------|-------------|
| Median Duration | 50th percentile parking time |
| Abandonment Rate | Vehicles parked >120 hours |
| Turnover Rate | Trips per vehicle per day |

**Outputs**:

| File | Description |
|------|-------------|
| `table01_survival_summary.csv` | Survival statistics |
| `table02_bootstrap_statistics.csv` | Bootstrap CIs |
| `table03_logrank_pairwise.csv` | Pairwise comparisons |
| `table04_weibull_parameters.csv` | Shape and scale parameters |

---

### Exercise 5: Economic Analysis

**File**: `src/analysis/05_economic_analysis.py`

**Purpose**: Calculate revenue, fleet economics, and profitability.

**Statistical Methods**:

| Method | Implementation | Purpose |
|--------|----------------|---------|
| Monte Carlo Simulation | Custom (n=10,000) | Risk analysis |
| VaR Calculation | 5th percentile | Value at Risk |
| CVaR | Mean below VaR | Conditional VaR |
| Pareto Analysis | 80/20 rule | Zone profitability |
| Sensitivity Analysis | Tornado diagram | Parameter importance |

**Key Results**:

| Metric | LIME | BIRD | VOI |
|--------|------|------|-----|
| Gross Revenue | €4,245,099 | €3,217,369 | €837,898 |
| Net Profit | €2,362,745 | €1,978,662 | €448,251 |
| Profit Margin | 55.7% | 61.5% | 53.5% |

**Outputs**:

| File | Description |
|------|-------------|
| `table01_operator_summary.csv` | P&L by operator |
| `table02_monte_carlo_stats.csv` | Simulation results |
| `table03_sensitivity_ranking.csv` | Parameter sensitivity |
| `table04_scenario_comparison.csv` | Scenario analysis |

---

## Visualization Modules

### Naming Convention

Each analysis module has corresponding visualization scripts:

| Analysis | Statistics Plots | Spatial Plots |
|----------|------------------|---------------|
| 01_temporal | 01_temporal_statistics.py | 01_temporal_dashboard.py |
| 02_od_matrix | 02_od_statistics.py | 02_od_spatial_flows.py |
| 03_integration | 03_integration_statistics.py | 03_integration_maps.py |
| 04_parking | 04_parking_survival.py | 04_parking_maps.py |
| 05_economic | 05_economic_sensitivity.py | 05_economic_maps.py |

### Output Formats

| Type | Format | Resolution | Location |
|------|--------|------------|----------|
| Charts | PNG | 300 DPI | outputs/figures/exerciseN/ |
| Maps | PNG | 300 DPI | outputs/figures/exerciseN/ |
| Dashboards | PNG | 300 DPI | outputs/figures/exerciseN/dashboard/ |

---

## Pipeline Controller

### run_pipeline.py

The master controller orchestrates all analysis and visualization stages.

**Usage**:

```bash
# Run complete pipeline
python run_pipeline.py

# Run specific stages
python run_pipeline.py --stages 1 2 3

# Run from specific stage
python run_pipeline.py --from-stage 3

# Run only visualizations
python run_pipeline.py --viz-only

# Skip visualizations
python run_pipeline.py --no-viz
```

**Stage Execution Order**:

| Order | Stage | Module | Depends On |
|-------|-------|--------|------------|
| 1 | Data Cleaning | 00_data_cleaning.py | Raw data |
| 2 | Temporal Analysis | 01_temporal_analysis.py | Cleaned data |
| 3 | OD Matrix | 02_od_matrix_analysis.py | Cleaned data |
| 4 | Integration | 03_integration_analysis.py | Cleaned data + GTFS |
| 5 | Parking | 04_parking_analysis.py | Cleaned data |
| 6 | Economic | 05_economic_analysis.py | All previous |

---

## Dependencies

### Core Requirements

```text
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
scipy>=1.10.0          # Statistical analysis
geopandas>=0.14.0      # Geospatial analysis
shapely>=2.0.0         # Geometric operations
pyproj>=3.3.0          # Coordinate transformations
```

### Visualization

```text
matplotlib>=3.7.0      # Base plotting
seaborn>=0.12.0        # Statistical graphics
contextily>=1.3.0      # Basemap tiles
```

### Optional (Advanced Features)

```text
lifelines>=0.27.0      # Survival analysis
statsmodels>=0.14.0    # Time series decomposition
scikit-learn>=1.3.0    # Machine learning utilities
```

---

## Performance Considerations

### Runtime Estimates

| Stage | Typical Runtime | Memory Usage |
|-------|-----------------|--------------|
| Data Cleaning | 2-3 min | 2 GB |
| Temporal Analysis | 5-8 min | 1 GB |
| OD Matrix | 8-12 min | 3 GB |
| Integration | 10-15 min | 4 GB |
| Parking | 5-8 min | 2 GB |
| Economic | 15-20 min | 2 GB |
| **Total** | **45-65 min** | **Peak: 4 GB** |

### Optimization Strategies

1. **Checkpointing**: Intermediate results saved as pickle files
2. **Spatial Indexing**: R-tree for efficient spatial queries
3. **Vectorization**: NumPy/Pandas operations over loops
4. **Lazy Loading**: Data loaded only when needed

---

## Quality Assurance

### Statistical Rigor

| Standard | Implementation |
|----------|----------------|
| Confidence Level | 95% for all intervals |
| Effect Sizes | Cohen's d, eta-squared, Cramér's V |
| Multiple Testing | Bonferroni correction |
| Non-parametric | Used for non-normal distributions |
| Bootstrap | 1,000 resamples for robust estimates |

### Data Quality Checks

| Check | Threshold | Action |
|-------|-----------|--------|
| Coordinate bounds | Turin bbox | Remove outliers |
| Trip duration | 1-60 minutes | Filter invalid |
| Battery level | 0-100% | Cap values |
| Distance | 0-50 km | Remove outliers |

---

## References

### Statistical Methods

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Cleveland, R. B. et al. (1990). STL: A Seasonal-Trend Decomposition
- Hollander, M. et al. (2013). Nonparametric Statistical Methods
- Moran, P.A.P. (1950). Notes on continuous stochastic phenomena
- Anselin, L. (1995). Local indicators of spatial association

### Python Libraries

- McKinney, W. (2010). Data Structures for Statistical Computing in Python
- Jordahl, K. (2014). GeoPandas: Python tools for geographic data
- Davidson-Pilon, C. (2019). Lifelines: Survival analysis in Python

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Oct 2025 | Initial pipeline structure |
| 2.0 | Nov 2025 | Added survival analysis, Monte Carlo |
| 3.0 | Dec 2025 | Decoupled architecture, full documentation |

---

## Author

**Ali Vaezi**  
MSc Transport Engineering  
Politecnico di Torino, Italy

---

*Last Updated: December 2025*
