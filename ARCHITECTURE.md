# 🏗️ Technical Architecture# 🏗️ Technical Architecture# 🏗️ Technical Architecture# 🏗️ Technical Architecture



## Turin Smart Mobility — System Design Document## Turin Smart Mobility — System Design Document



<div align="center">## Turin Smart Mobility — System Design Document## Turin Smart Mobility — System Design Document



**Version 3.0** | **December 2025** | **Politecnico di Torino**<div align="center">



</div>



---**Version 3.0** | **December 2025** | **Politecnico di Torino**



## 📐 Document Purpose<div align="center"><div align="center">



This document provides the technical specification for the Turin Smart Mobility analysis pipeline. It is intended for:</div>



- **Developers** extending or maintaining the codebase

- **Reviewers** assessing the technical rigor of the analysis

- **Researchers** replicating the methodology for other cities---



---**Version 3.0** | **December 2025** | **Politecnico di Torino****Version 2.0** | **December 2025** | **Politecnico di Torino**



## 🔄 System Architecture Overview## 📐 Document Purpose



### Pipeline Flow Diagram



```This document provides the complete technical specification for the Turin Smart Mobility analysis pipeline. It is intended for:

┌─────────────────────────────────────────────────────────────────────────────────┐

│                       TURIN SMART MOBILITY PIPELINE v3.0                        │</div></div>

└─────────────────────────────────────────────────────────────────────────────────┘

- **Developers** extending or maintaining the codebase

  RAW DATA                 PREPROCESSING              ANALYSIS STAGES

  ════════                 ═════════════              ═══════════════- **Reviewers** assessing the technical rigor of the analysis

  

  ┌─────────────┐         ┌─────────────┐           ┌─────────────────────────────┐- **Researchers** replicating the methodology for other cities

  │ LIME CSV    │         │             │           │  STAGE 1: TEMPORAL          │

  │ (~312K)     │────────▶│   Data      │──────────▶│  01_temporal_analysis.py    │------

  ├─────────────┤         │  Cleaning   │           ├─────────────────────────────┤

  │ VOI XLSX    │────────▶│             │           │  STAGE 2: OD MATRIX         │---

  │ (18 files)  │         │  Creates:   │           │  02_od_matrix_analysis.py   │

  ├─────────────┤         │  *_cleaned  │           ├─────────────────────────────┤

  │ BIRD CSV    │────────▶│  .csv       │           │  STAGE 3: INTEGRATION       │

  │ (2 files)   │         │             │           │  03_integration_analysis.py │## 🔄 System Architecture Overview

  └─────────────┘         └─────────────┘           ├─────────────────────────────┤

                                                    │  STAGE 4: PARKING           │## 📐 Document Purpose## 📐 Document Purpose

  ┌─────────────┐                                   │  04_parking_analysis.py     │

  │ GTFS Bundle │──────────────────────────────────▶├─────────────────────────────┤### Pipeline Flow Diagram

  │ (stops.txt) │                                   │  STAGE 5: ECONOMICS         │

  ├─────────────┤                                   │  05_economic_analysis.py    │

  │ Zone SHP    │──────────────────────────────────▶└─────────────────────────────┘

  │ (94 zones)  │                                                │```

  └─────────────┘                                                │

                          ┌──────────────────────────────────────┴────────────────┐┌─────────────────────────────────────────────────────────────────────────────────┐This document provides the complete technical specification for the Turin Smart Mobility analysis pipeline. It is intended for:This document provides the technical specification for the Turin Smart Mobility analysis pipeline. It is intended for:

                          ▼                          ▼                            ▼

                   ┌─────────────┐           ┌─────────────┐              ┌─────────────┐│                       TURIN SMART MOBILITY PIPELINE v3.0                        │

                   │  FIGURES    │           │  REPORTS    │              │ CHECKPOINTS │

                   │  (PNG)      │           │  (CSV/MD)   │              │  (PKL)      │└─────────────────────────────────────────────────────────────────────────────────┘

                   └─────────────┘           └─────────────┘              └─────────────┘

```



---  RAW DATA                 PREPROCESSING              ANALYSIS STAGES- **Developers** extending or maintaining the codebase- **Developers** extending or maintaining the codebase



## 📂 Complete Directory Structure  ════════                 ═════════════              ═══════════════



```  - **Reviewers** assessing the technical rigor of the analysis- **Reviewers** assessing the technical rigor of the analysis

DATI MONOPATTINI SHARING-2/

│  ┌─────────────┐         ┌─────────────┐           ┌─────────────────────────────┐

├── 📄 README.md                      # Project overview & quick start

├── 📄 ARCHITECTURE.md                # This file - technical documentation  │ LIME CSV    │         │             │           │  STAGE 1: TEMPORAL          │- **Researchers** replicating the methodology for other cities- **Researchers** replicating the methodology for other cities

├── 📄 requirements.txt               # Python dependencies

├── 📄 run_pipeline.py                # Master pipeline controller  │ (~312K)     │────────▶│   Data      │──────────▶│  01_temporal_analysis.py    │

│

├── 📂 src/  ├─────────────┤         │  Cleaning   │           ├─────────────────────────────┤

│   ├── analysis/                     # Statistical analysis modules

│   │   ├── 01_temporal_analysis.py   # Temporal pattern analysis  │ VOI XLSX    │────────▶│             │           │  STAGE 2: OD MATRIX         │

│   │   ├── 02_od_matrix_analysis.py  # OD flow analysis

│   │   ├── 03_integration_analysis.py # PT integration metrics  │ (18 files)  │         │  Creates:   │           │ 02_od_matrix_analysis.py   │------

│   │   ├── 04_parking_analysis.py    # Parking duration analysis

│   │   └── 05_economic_analysis.py   # Economic modeling  ├─────────────┤         │  *_cleaned  │           ├─────────────────────────────┤

│   │

│   ├── utils/                        # Utility modules (spatial, general)  │ BIRD CSV    │────────▶│  .csv       │           │  STAGE 3: INTEGRATION       │

│   │   ├── spatial_utils.py          # CRS, zone handling, spatial helpers  │ (2 files)   │         │             │           │  03_integration_analysis.py │

│   │   └── __init__.py               # Utils package init

│   ├── visualization/                # Visualization modules  │ GTFS Bundle │──────────────────────────────────▶├─────────────────────────────┤

│   │   ├── 00_data_cleaning.py       # Data cleaning waterfall & bad data charts

│   │   ├── 01_temporal_dashboard.py  # Temporal dashboard  │ (stops.txt) │                                   │  STAGE 5: ECONOMICS         │

│   │   ├── 01_temporal_statistics.py # Temporal stats figures

│   │   ├── 02_od_spatial_flows.py    # OD flow maps  ├─────────────┤                                   │  05_economic_analysis.py    │

│   │   ├── 02_od_statistics.py       # OD statistics figures

│   │   ├── 03_integration_maps.py    # Integration maps                                                    │  STAGE 4: PARKING           │

│   │   ├── 03_integration_statistics.py # Integration stats

│   │   ├── 04_parking_survival.py    # Survival analysis plots

│   │   ├── 04_parking_maps.py        # Parking heatmaps  ┌─────────────┐                                   │  04_parking_analysis.py     │

│   │   ├── 05_economic_sensitivity.py # Sensitivity analysis

│   │   └── 05_economic_maps.py       # Economic visualizations  │ GTFS Bundle │──────────────────────────────────▶├─────────────────────────────┤

│   │

│   └── data/                         # Data processing  │ (stops.txt) │                                   │  STAGE 5: ECONOMICS         │### High-Level Pipeline Architecture### High-Level Pipeline Architecture

│       └── 01_data_cleaning.py       # ETL pipeline

│  ├─────────────┤                                   │  05_economic_analysis.py    │

├── 📂 data/                          # Data directory (git-ignored)

│   ├── raw/                          # Original operator data  │ Zone SHP    │──────────────────────────────────▶└─────────────────────────────┘

│   └── processed/                    # Cleaned datasets

│  │ (94 zones)  │                                                │

└── 📂 outputs/

    ├── figures/                      # Generated visualizations  └─────────────┘                                                │```mermaid```mermaid

    │   ├── exercise1/statistical/

    │   ├── exercise2/statistical/                          ┌──────────────────────────────────────┴────────────────┐

    │   ├── exercise3/statistical/

    │   ├── exercise4/statistical/                          ▼                          ▼                            ▼flowchart TBflowchart LR

    │   └── exercise5/statistical/

    └── reports/                      # Analysis reports                   ┌─────────────┐           ┌─────────────┐              ┌─────────────┐

```

                   │  FIGURES    │           │  REPORTS    │              │ CHECKPOINTS │    subgraph INPUT ["📥 RAW DATA LAYER"]    subgraph INPUT ["📥 RAW DATA"]

---

                   │  (PNG)      │           │  (CSV/MD)   │              │  (PKL)      │

## 🎯 The 5 Research Questions

                   └─────────────┘           └─────────────┘              └─────────────┘        A1[LIME CSV<br/>1.2M trips]        A1[LIME CSV<br/>1.2M trips]

### Exercise 1: Temporal Pattern Analysis

**Research Question**: *How do e-scooter usage patterns vary by time?*```



**Methods**: Kruskal-Wallis H-test, Chi-square, Bootstrap CI        A2[VOI XLSX<br/>18 monthly files]        A2[VOI XLSX<br/>18 monthly files]



**Key Metrics**: Peak hours, weekend share, monthly trends---



---        A3[BIRD CSV<br/>2 files]        A3[BIRD CSV<br/>2 files]



### Exercise 2: Origin-Destination Matrix Analysis## 📂 Complete Directory Structure

**Research Question**: *What are the primary mobility corridors?*

        A4[GTFS Bundle<br/>stops, routes, shapes]        A4[GTFS Bundle<br/>stops, routes, shapes]

**Methods**: Chi-square test, Cramér's V, Gini coefficient

```

**Key Metrics**: Zone flows, corridor rankings, concentration

DATI MONOPATTINI SHARING-2/        A5[Zone Shapefile<br/>94 zones]        A5[Zone Shapefile<br/>94 zones]

---

│

### Exercise 3: Public Transport Integration Analysis

**Research Question**: *Are e-scooters competitors or allies?*├── 📄 README.md                      # Project overview & quick start    end    end



**Methods**: Buffer analysis, temporal segmentation├── 📄 ARCHITECTURE.md                # This file - technical documentation



**Key Metrics**: Integration Index, Feeder Rate├── 📄 requirements.txt               # Python dependencies



### Exercise 4: Parking Duration Analysis│    subgraph STAGE0 ["🔧 STAGE 0: PREPROCESSING"]    subgraph STAGE1 ["🔧 STAGE 1: PREPROCESSING"]

**Research Question**: *How long do e-scooters remain parked?*

├── 📂 src/

**Methods**: Weibull survival, Kaplan-Meier, Log-rank test

│   ├── analysis/                     # Statistical analysis modules        B1[01_preprocessing.py]        B1[01_preprocessing.py]

**Key Metrics**: Median duration, abandonment rate

│   │   ├── 01_temporal_analysis.py   # Temporal pattern analysis

---

│   │   ├── 02_od_matrix_analysis.py  # OD flow analysis    end        B2[Schema Harmonization]

### Exercise 5: Economic Analysis

**Research Question**: *What is the financial viability?*│   │   ├── 03_integration_analysis.py # PT integration metrics



**Methods**: Monte Carlo simulation, sensitivity analysis│   │   ├── 04_parking_analysis.py    # Parking duration analysis        B3[Coordinate Validation]



**Key Metrics**: Revenue, profit margin, P(loss)│   │   └── 05_economic_analysis.py   # Economic modeling



---│   │    subgraph STAGE1 ["📊 STAGE 1: TEMPORAL ANALYSIS"]        B4[Temporal Cleaning]



## 🎨 Decoupled Design Pattern│   ├── visualization/                # Visualization modules



### The Problem: Monolithic Scripts│   │   ├── 01_temporal_statistics.py # Temporal stats figures        C1[02_analysis.py]    end



```python│   │   ├── 01_temporal_dashboard.py  # Temporal dashboard

# ❌ ANTI-PATTERN: Monolithic Script

def main():│   │   ├── 02_od_statistics.py       # OD statistics figures        C2[src/analysis/01_temporal_q1.py]

    df = load_data()           # 2 min

    results = heavy_calc(df)   # 30 min  ← Must re-run for any change│   │   ├── 02_od_spatial_flows.py    # OD flow maps

    plot_results(results)      # 1 min

    │   │   ├── 03_integration_statistics.py # Integration stats        C3[src/visualization/01_temporal_plots.py]    subgraph STAGE2 ["📊 STAGE 2: ANALYSIS"]

# Total: 33 min for a single plot color change!

```│   │   ├── 03_integration_maps.py    # Integration maps



### Our Solution: Separated Layers│   │   ├── 04_parking_survival.py    # Survival analysis plots    end        C1[02_analysis.py<br/>Descriptive Stats]



```│   │   ├── 04_parking_maps.py        # Parking heatmaps

┌─────────────────────────────────────────────────────────────────────────────────┐

│                         DECOUPLED ARCHITECTURE                                   ││   │   ├── 05_economic_sensitivity.py # Sensitivity analysis        C2[03_od_matrices.py<br/>O-D Flows]

├─────────────────────────────────────────────────────────────────────────────────┤

│                                                                                  ││   │   └── 05_economic_maps.py       # Economic visualizations

│   ANALYSIS LAYER (src/analysis/)      VISUALIZATION LAYER (src/visualization/) │

│   ══════════════════════════════      ═════════════════════════════════════════││   │    subgraph STAGE2 ["🗺️ STAGE 2: OD MATRIX"]    end

│                                                                                  │

│   01_temporal_analysis.py              01_temporal_statistics.py                 ││   └── data/                         # Data processing

│   02_od_matrix_analysis.py             02_od_statistics.py                       │

│   03_integration_analysis.py  ─────▶   03_integration_statistics.py              ││       └── 01_data_cleaning.py       # ETL pipeline        D1[03_od_matrices.py]

│   04_parking_analysis.py     CHECKPOINTS 04_parking_survival.py                  │

│   05_economic_analysis.py              05_economic_sensitivity.py                ││

│                                                                                  │

│   Runtime: ~30 min each                Runtime: ~2 min each                      │├── 📂 data/                          # Data directory (git-ignored)        D2[src/analysis/02_od_matrix_q1.py]    subgraph STAGE3 ["⚙️ STAGE 3: CALCULATION"]

│   CPU-bound (computation)              I/O-bound (plotting)                      │  ├── raw/                          # Original operator data

│   Run ONCE per data update             Run MANY times for styling                │   │   ├── bird/                     # BIRD CSV files        D3[src/visualization/02_od_matrix_plots.py]        D1[04_transport_comparison.py]

```

│   │   ├── lime/                     # LIME CSV files

### Benefits

│   │   ├── voi/                      # VOI XLSX files    end        D2[Buffer Analysis]

| Benefit | Monolithic | Decoupled |

|---------|------------|-----------|│   │   ├── gtfs/                     # GTFS bundle

| **Visualization Iteration** | 30+ min | ~2 min |

| **Fault Recovery** | Start over | Resume from checkpoint |│   │   └── zone_statistiche_geo/     # Zone shapefile        D3[Temporal Segmentation]

| **Memory Usage** | High peak | Isolated per stage |

│   │

---

│   └── processed/                    # Cleaned datasets    subgraph STAGE3 ["🔗 STAGE 3: INTEGRATION"]        D4[Tortuosity Calculation]

## 📚 Data Dictionary

│       ├── lime_cleaned.csv

### Standardized Schema (Post-Preprocessing)

│       ├── voi_cleaned.csv        E1[04_transport_comparison.py]    end

| Column | Type | Description |

|--------|------|-------------|│       ├── bird_cleaned.csv

| `operator` | str | BIRD, LIME, VOI |

| `start_time` | datetime | Trip start (UTC+1) |│       └── df_all.pkl        E2[src/analysis/03_integration_q1.py]

| `end_time` | datetime | Trip end (UTC+1) |

| `start_lat`, `start_lon` | float | Origin (WGS84) |│

| `end_lat`, `end_lon` | float | Destination (WGS84) |

| `distance_km` | float | Trip distance |├── 📂 outputs/        E3[src/visualization/03_integration_plots.py]    subgraph CHECKPOINTS ["💾 CHECKPOINTS"]

| `duration_min` | float | Trip duration |

| `hour` | int | Hour of day (0-23) |│   ├── figures/                      # Generated visualizations

| `day_of_week` | int | Day (0=Mon, 6=Sun) |

| `is_weekend` | bool | Saturday or Sunday |│   │   ├── exercise1/                # ~10 PNG files    end        E1[.pkl files]



### Checkpoint Files Reference│   │   ├── exercise2/                # ~15 PNG files



| Exercise | Checkpoint File | Contents |│   │   ├── exercise3/                # ~17 PNG files        E2[.geojson files]

|----------|-----------------|----------|

| 1 | `checkpoint_hourly_stats.csv` | Hourly aggregations |│   │   ├── exercise4/                # ~12 PNG files

| 2 | `checkpoint_od_matrix.pkl` | Full OD matrix |

| 3 | `checkpoint_buffer_sensitivity.pkl` | Multi-buffer results |│   │   └── exercise5/                # ~10 PNG files    subgraph STAGE4 ["🅿️ STAGE 4: PARKING"]        E3[.csv summaries]

| 4 | `checkpoint_parking_stats.csv` | Duration statistics |

| 5 | `checkpoint_monte_carlo_summary.csv` | Risk analysis |│   │



---│   └── reports/                      # Analysis reports        F1[src/analysis/04_parking_q1.py]    end



## ⚡ Key Algorithms│       ├── exercise1/



### 1. Vectorized Buffer Analysis│       ├── exercise2/        F2[src/visualization/04_parking_plots.py]



**Challenge**: 549K trips × 1,500 PT stops = 824M distance checks│       ├── exercise3/



**Solution**: Pre-computed coverage zones with vectorized containment│       ├── exercise4/    end    subgraph STAGE4 ["🎨 STAGE 4: VISUALIZATION"]



```python│       └── exercise5/

for buffer_distance in [50, 100, 200]:

    pt_coverage = unary_union([stop.buffer(buffer_distance) for stop in stops])│        F1[04_visualization.py]

    prepared_coverage = prep(pt_coverage)

    is_near = trips_gdf.geometry.within(prepared_coverage)└── 📂 archive/                       # Deprecated scripts (git-ignored)

```

```    subgraph STAGE5 ["💰 STAGE 5: ECONOMICS"]        F2[Professional Figures]

**Speedup**: 100× faster than naive approach



### 2. Weibull Survival Analysis

---        G1[src/analysis/05_economic_q1.py]        F3[Report Tables]

$$S(t) = e^{-(t/\lambda)^k}$$



| Parameter | BIRD | LIME | VOI |

|-----------|------|------|-----|## 🎯 The 5 Research Questions        G2[src/visualization/05_economic_plots.py]    end

| Shape (k) | 0.615 | 0.628 | 0.570 |

| Scale (λ) | 12.0h | 6.5h | 22.8h |



### 3. Monte Carlo Profit Simulation### Exercise 1: Temporal Pattern Analysis    end



10,000 iterations with random parameter sampling**Research Question**: *How do e-scooter usage patterns vary by time?



**Risk Metrics**: P(loss) = 0.52%, VaR(5%) = €1.23M    subgraph OUTPUT ["📤 OUTPUTS"]



---**Methods**: Kruskal-Wallis H-test, Chi-square, Bootstrap CI



## 🛠️ Technology Stack    subgraph OUTPUT ["📤 OUTPUT LAYER"]        G1[PNG Figures]



### Core Libraries**Key Metrics**: Peak hours, weekend share, monthly trends



| Library | Version | Purpose |        H1[PNG Figures]        G2[CSV Reports]

|---------|---------|---------|

| **pandas** | ≥2.0 | Data manipulation |---

| **geopandas** | ≥0.14 | Spatial DataFrames |

| **shapely** | ≥2.0 | Geometry operations |        H2[CSV Reports]        G3[GeoJSON Maps]

| **numpy** | ≥1.24 | Numerical computing |

| **scipy** | ≥1.10 | Statistical analysis |### Exercise 2: Origin-Destination Matrix Analysis

| **matplotlib** | ≥3.7 | Visualization |

| **seaborn** | ≥0.12 | Statistical plots |**Research Question**: *What are the primary mobility corridors?*        H3[Markdown Reports]    end



### Coordinate Reference Systems



| CRS | EPSG | Usage |**Methods**: Chi-square test, Cramér's V, Gini coefficient        H4[Pickle Checkpoints]

|-----|------|-------|

| WGS84 | 4326 | Input/storage |

| UTM 32N | 32632 | Metric calculations |

| Web Mercator | 3857 | Basemap visualization |**Key Metrics**: Zone flows, corridor rankings, concentration    end    A1 & A2 & A3 --> B1



---



## 🚀 Pipeline Execution---    A4 & A5 --> D1



### Full Pipeline



```bash### Exercise 3: Public Transport Integration Analysis    A1 & A2 & A3 --> B1    B1 --> B2 --> B3 --> B4

python run_pipeline.py --stages 0,1,2,3,4,5

```**Research Question**: *Are e-scooters competitors or allies to public transport?*



### Resource Requirements    A4 & A5 --> E1    B4 --> C1 & C2



| Stage | Peak RAM | Runtime |**Methods**: Buffer analysis, temporal segmentation

|-------|----------|---------|

| 0 (Preprocessing) | 4 GB | 5 min |    B1 --> C1 --> C2 --> C3    C1 & C2 --> D1

| 1 (Temporal) | 3 GB | 10 min |

| 2 (OD Matrix) | 6 GB | 15 min |**Key Metrics**: Integration Index, Feeder Rate

| 3 (Integration) | 8 GB | 30 min |

| 4 (Parking) | 4 GB | 20 min |    C1 --> D1 --> D2 --> D3    D1 --> D2 & D3 & D4

| 5 (Economics) | 2 GB | 10 min |

---

---

    D1 --> E1 --> E2 --> E3    D2 & D3 & D4 --> E1 & E2 & E3

## 📊 Output Artifacts

### Figures by Exercise

**Research Question**: *How long do e-scooters remain parked?*    E1 --> F1 --> F2    E1 & E2 & E3 --> F1

| Exercise | Count | Key Figures |

|----------|-------|-------------|

| 1 | ~10 | Hourly patterns, heatmaps, data cleaning waterfall, bad data breakdown |

| 2 | ~15 | OD flows, choropleths |

| 3 | ~17 | Buffer sensitivity, integration |

| 4 | ~12 | Survival curves, hazard |

| 5 | ~10 | Monte Carlo, sensitivity |

**Key Metrics**: Median duration, abandonment rate    C3 & D3 & E3 & F2 & G2 --> H1 & H2 & H3    F2 & F3 --> G1 & G2 & G3

### Reports



All exercises have detailed Markdown reports with:

- Statistical test results---    C2 & D2 & E2 & F1 & G1 --> H4```

- LaTeX-ready tables

- Figure references



---### Exercise 5: Economic Analysis```



## 🔒 Quality Assurance**Research Question**: *What is the financial viability?*



### Data Validation### Text-Based Alternative (for non-Mermaid renderers)



| Check | Stage | Action |**Methods**: Monte Carlo simulation, sensitivity analysis

|-------|-------|--------|

| Coordinate bounds | Preprocessing | Drop invalid |### Text-Based Alternative

| Temporal consistency | Preprocessing | Correct dates |

| Missing values | Preprocessing | Impute or flag |**Key Metrics**: Revenue, profit margin, P(loss)

| Duplicate trips | Preprocessing | Deduplicate |

```

### Statistical Rigor

---

- Bonferroni-corrected p-values

- Effect sizes (η², Cramér's V)```┌─────────────────────────────────────────────────────────────────────────────────┐

- 95% bootstrap confidence intervals

- Non-parametric tests for non-normal data## 🎨 Decoupled Design Pattern



---┌─────────────────────────────────────────────────────────────────────────────────┐│                           SYSTEM ARCHITECTURE                                    │



## 📚 References### The Problem: Monolithic Analysis Scripts



1. **Buffer Analysis**: EU Standard EN13816│                       TURIN SMART MOBILITY PIPELINE v3.0                        ││                      Turin Smart Mobility Pipeline                               │

2. **Survival Analysis**: Weibull distribution, Kaplan-Meier

3. **Economic Modeling**: Monte Carlo methods```python

4. **Spatial Indexing**: Shapely STRtree, GEOS

# ❌ ANTI-PATTERN: Monolithic Script└─────────────────────────────────────────────────────────────────────────────────┘└─────────────────────────────────────────────────────────────────────────────────┘

---

def main():

<div align="center">

    df = load_data()           # 2 min

**Technical Architecture Document v3.0**

    results = heavy_calc(df)   # 30 min  ← Must re-run for any change

*Turin Smart Mobility Project • December 2025*

    plot_results(results)      # 1 min

    │   │   ├── 03_integration_statistics.py # Integration stats        C3[src/visualization/01_temporal_plots.py]    subgraph STAGE2 ["📊 STAGE 2: ANALYSIS"]

# Total: 33 min for a single plot color change!

```│   │   ├── 03_integration_maps.py    # Integration maps



### Our Solution: Separated Layers│   │   ├── 04_parking_survival.py    # Survival analysis plots    end        C1[02_analysis.py<br/>Descriptive Stats]



```│   │   ├── 04_parking_maps.py        # Parking heatmaps

┌─────────────────────────────────────────────────────────────────────────────────┐

│                         DECOUPLED ARCHITECTURE                                   ││   │   ├── 05_economic_sensitivity.py # Sensitivity analysis        C2[03_od_matrices.py<br/>O-D Flows]

├─────────────────────────────────────────────────────────────────────────────────┤

│                                                                                  ││   │   └── 05_economic_maps.py       # Economic visualizations

│   ANALYSIS LAYER (src/analysis/)      VISUALIZATION LAYER (src/visualization/) │

│   ══════════════════════════════      ═════════════════════════════════════════││   │    subgraph STAGE2 ["🗺️ STAGE 2: OD MATRIX"]    end

│                                                                                  │

│   01_temporal_analysis.py              01_temporal_statistics.py                 ││   └── data/                         # Data processing

│   02_od_matrix_analysis.py             02_od_statistics.py                       │

│   03_integration_analysis.py  ─────▶   03_integration_statistics.py              ││       └── 01_data_cleaning.py       # ETL pipeline        D1[03_od_matrices.py]

│   04_parking_analysis.py     CHECKPOINTS 04_parking_survival.py                  │

│   05_economic_analysis.py              05_economic_sensitivity.py                ││

│                                                                                  │

│   Runtime: ~30 min each                Runtime: ~2 min each                      │├── 📂 data/                          # Data directory (git-ignored)        D2[src/analysis/02_od_matrix_q1.py]    subgraph STAGE3 ["⚙️ STAGE 3: CALCULATION"]

│   CPU-bound (computation)              I/O-bound (plotting)                      │  ├── raw/                          # Original operator data

│   Run ONCE per data update             Run MANY times for styling                │   │   ├── bird/                     # BIRD CSV files        D3[src/visualization/02_od_matrix_plots.py]        D1[04_transport_comparison.py]

```

│   │   ├── lime/                     # LIME CSV files

### Benefits

│   │   ├── voi/                      # VOI XLSX files    end        D2[Buffer Analysis]

| Benefit | Monolithic | Decoupled |

|---------|------------|-----------|│   │   ├── gtfs/                     # GTFS bundle

| **Visualization Iteration** | 30+ min | ~2 min |

| **Fault Recovery** | Start over | Resume from checkpoint |│   │   └── zone_statistiche_geo/     # Zone shapefile        D3[Temporal Segmentation]

| **Memory Usage** | High peak | Isolated per stage |

│   │

---

│   └── processed/                    # Cleaned datasets    subgraph STAGE3 ["🔗 STAGE 3: INTEGRATION"]        D4[Tortuosity Calculation]

## 📚 Data Dictionary

│       ├── lime_cleaned.csv

### Standardized Schema (Post-Preprocessing)

│       ├── voi_cleaned.csv        E1[04_transport_comparison.py]    end

| Column | Type | Description |

|--------|------|-------------|│       ├── bird_cleaned.csv

| `operator` | str | BIRD, LIME, VOI |

| `start_time` | datetime | Trip start (UTC+1) |│       └── df_all.pkl        E2[src/analysis/03_integration_q1.py]

| `end_time` | datetime | Trip end (UTC+1) |

| `start_lat`, `start_lon` | float | Origin (WGS84) |│

| `end_lat`, `end_lon` | float | Destination (WGS84) |

| `distance_km` | float | Trip distance |├── 📂 outputs/        E3[src/visualization/03_integration_plots.py]    subgraph CHECKPOINTS ["💾 CHECKPOINTS"]

| `duration_min` | float | Trip duration |

| `hour` | int | Hour of day (0-23) |│   ├── figures/                      # Generated visualizations

| `day_of_week` | int | Day (0=Mon, 6=Sun) |

| `is_weekend` | bool | Saturday or Sunday |│   │   ├── exercise1/                # ~10 PNG files    end        E1[.pkl files]



### Checkpoint Files Reference│   │   ├── exercise2/                # ~15 PNG files



| Exercise | Checkpoint File | Contents |│   │   ├── exercise3/                # ~17 PNG files        E2[.geojson files]

|----------|-----------------|----------|

| 1 | `checkpoint_hourly_stats.csv` | Hourly aggregations |│   │   ├── exercise4/                # ~12 PNG files

| 2 | `checkpoint_od_matrix.pkl` | Full OD matrix |

| 3 | `checkpoint_buffer_sensitivity.pkl` | Multi-buffer results |│   │   └── exercise5/                # ~10 PNG files    subgraph STAGE4 ["🅿️ STAGE 4: PARKING"]        E3[.csv summaries]

| 4 | `checkpoint_parking_stats.csv` | Duration statistics |

| 5 | `checkpoint_monte_carlo_summary.csv` | Risk analysis |│   │



---│   └── reports/                      # Analysis reports        F1[src/analysis/04_parking_q1.py]    end



## ⚡ Key Algorithms│       ├── exercise1/



### 1. Vectorized Buffer Analysis│       ├── exercise2/        F2[src/visualization/04_parking_plots.py]



**Challenge**: 549K trips × 1,500 PT stops = 824M distance checks│       ├── exercise3/



**Solution**: Pre-computed coverage zones with vectorized containment│       ├── exercise4/    end    subgraph STAGE4 ["🎨 STAGE 4: VISUALIZATION"]



```python│       └── exercise5/

for buffer_distance in [50, 100, 200]:

    pt_coverage = unary_union([stop.buffer(buffer_distance) for stop in stops])│        F1[04_visualization.py]

    prepared_coverage = prep(pt_coverage)

    is_near = trips_gdf.geometry.within(prepared_coverage)└── 📂 archive/                       # Deprecated scripts (git-ignored)

```

```    subgraph STAGE5 ["💰 STAGE 5: ECONOMICS"]        F2[Professional Figures]

**Speedup**: 100× faster than naive approach



### 2. Weibull Survival Analysis

---        G1[src/analysis/05_economic_q1.py]        F3[Report Tables]

$$S(t) = e^{-(t/\lambda)^k}$$



| Parameter | BIRD | LIME | VOI |

|-----------|------|------|-----|## 🎯 The 5 Research Questions        G2[src/visualization/05_economic_plots.py]    end

| Shape (k) | 0.615 | 0.628 | 0.570 |

| Scale (λ) | 12.0h | 6.5h | 22.8h |



### 3. Monte Carlo Profit Simulation### Exercise 1: Temporal Pattern Analysis    end



10,000 iterations with random parameter sampling**Research Question**: *How do e-scooter usage patterns vary by time?



**Risk Metrics**: P(loss) = 0.52%, VaR(5%) = €1.23M    subgraph OUTPUT ["📤 OUTPUTS"]



---**Methods**: Kruskal-Wallis H-test, Chi-square, Bootstrap CI



## 🛠️ Technology Stack    subgraph OUTPUT ["📤 OUTPUT LAYER"]        G1[PNG Figures]



### Core Libraries**Key Metrics**: Peak hours, weekend share, monthly trends



| Library | Version | Purpose |        H1[PNG Figures]        G2[CSV Reports]

|---------|---------|---------|

| **pandas** | ≥2.0 | Data manipulation |---

| **geopandas** | ≥0.14 | Spatial DataFrames |

| **shapely** | ≥2.0 | Geometry operations |        H2[CSV Reports]        G3[GeoJSON Maps]

| **numpy** | ≥1.24 | Numerical computing |

| **scipy** | ≥1.10 | Statistical analysis |### Exercise 2: Origin-Destination Matrix Analysis

| **matplotlib** | ≥3.7 | Visualization |

| **seaborn** | ≥0.12 | Statistical plots |**Research Question**: *What are the primary mobility corridors?*        H3[Markdown Reports]    end



### Coordinate Reference Systems



| CRS | EPSG | Usage |**Methods**: Chi-square test, Cramér's V, Gini coefficient        H4[Pickle Checkpoints]

|-----|------|-------|

| WGS84 | 4326 | Input/storage |

| UTM 32N | 32632 | Metric calculations |

| Web Mercator | 3857 | Basemap visualization |**Key Metrics**: Zone flows, corridor rankings, concentration    end    A1 & A2 & A3 --> B1



---



## 🚀 Pipeline Execution---    A4 & A5 --> D1



### Full Pipeline



```bash### Exercise 3: Public Transport Integration Analysis    A1 & A2 & A3 --> B1    B1 --> B2 --> B3 --> B4

python run_pipeline.py --stages 0,1,2,3,4,5

```**Research Question**: *Are e-scooters competitors or allies to public transport?*



### Resource Requirements    A4 & A5 --> E1    B4 --> C1 & C2



| Stage | Peak RAM | Runtime |**Methods**: Buffer analysis, temporal segmentation

|-------|----------|---------|

| 0 (Preprocessing) | 4 GB | 5 min |    B1 --> C1 --> C2 --> C3    C1 & C2 --> D1

| 1 (Temporal) | 3 GB | 10 min |

| 2 (OD Matrix) | 6 GB | 15 min |**Key Metrics**: Integration Index, Feeder Rate

| 3 (Integration) | 8 GB | 30 min |

| 4 (Parking) | 4 GB | 20 min |    C1 --> D1 --> D2 --> D3    D1 --> D2 & D3 & D4

| 5 (Economics) | 2 GB | 10 min |

---

---

    D1 --> E1 --> E2 --> E3    D2 & D3 & D4 --> E1 & E2 & E3

## 📊 Output Artifacts

### Figures by Exercise

**Research Question**: *How long do e-scooters remain parked?*    E1 --> F1 --> F2    E1 & E2 & E3 --> F1

| Exercise | Count | Key Figures |

|----------|-------|-------------|

| 1 | ~10 | Hourly patterns, heatmaps, data cleaning waterfall, bad data breakdown |

| 2 | ~15 | OD flows, choropleths |

| 3 | ~17 | Buffer sensitivity, integration |

| 4 | ~12 | Survival curves, hazard |

| 5 | ~10 | Monte Carlo, sensitivity |

**Key Metrics**: Median duration, abandonment rate    C3 & D3 & E3 & F2 & G2 --> H1 & H2 & H3    F2 & F3 --> G1 & G2 & G3

### Reports



All exercises have detailed Markdown reports with:

- Statistical test results---    C2 & D2 & E2 & F1 & G1 --> H4```

- LaTeX-ready tables

- Figure references



---### Exercise 5: Economic Analysis```



## 🔒 Quality Assurance**Research Question**: *What is the financial viability?*



### Data Validation### Text-Based Alternative (for non-Mermaid renderers)



| Check | Stage | Action |**Methods**: Monte Carlo simulation, sensitivity analysis

|-------|-------|--------|

| Coordinate bounds | Preprocessing | Drop invalid |### Text-Based Alternative

| Temporal consistency | Preprocessing | Correct dates |

| Missing values | Preprocessing | Impute or flag |**Key Metrics**: Revenue, profit margin, P(loss)

| Duplicate trips | Preprocessing | Deduplicate |

```

### Statistical Rigor

---

- Bonferroni-corrected p-values

- Effect sizes (η², Cramér's V)```┌─────────────────────────────────────────────────────────────────────────────────┐

- 95% bootstrap confidence intervals

- Non-parametric tests for non-normal data## 🎨 Decoupled Design Pattern



---┌─────────────────────────────────────────────────────────────────────────────────┐│                           SYSTEM ARCHITECTURE                                    │



## 📚 References### The Problem: Monolithic Analysis Scripts



1. **Buffer Analysis**: EU Standard EN13816│                       TURIN SMART MOBILITY PIPELINE v3.0                        ││                      Turin Smart Mobility Pipeline                               │

2. **Survival Analysis**: Weibull distribution, Kaplan-Meier

3. **Economic Modeling**: Monte Carlo methods```python

4. **Spatial Indexing**: Shapely STRtree, GEOS

# ❌ ANTI-PATTERN: Monolithic Script└─────────────────────────────────────────────────────────────────────────────────┘└─────────────────────────────────────────────────────────────────────────────────┘

---

def main():

<div align="center">

    df = load_data()           # 2 min

**Technical Architecture Document v3.0**

    results = heavy_calc(df)   # 30 min  ← Must re-run for any change

*Turin Smart Mobility Project • December 2025*

    plot_results(results)      # 1 min

    │   │   ├── 03_integration_statistics.py # Integration stats        C3[src/visualization/01_temporal_plots.py]    subgraph STAGE2 ["📊 STAGE 2: ANALYSIS"]

# Total: 33 min for a single plot color change!

```│   │   ├── 03_integration_maps.py    # Integration maps



### Our Solution: Separated Layers│   │   ├── 04_parking_survival.py    # Survival analysis plots    end        C1[02_analysis.py<br/>Descriptive Stats]



```│   │   ├── 04_parking_maps.py        # Parking heatmaps

┌─────────────────────────────────────────────────────────────────────────────────┐

│                         DECOUPLED ARCHITECTURE                                   ││   │   ├── 05_economic_sensitivity.py # Sensitivity analysis        C2[03_od_matrices.py<br/>O-D Flows]

├─────────────────────────────────────────────────────────────────────────────────┤

│                                                                                  ││   │   └── 05_economic_maps.py       # Economic visualizations

│   ANALYSIS LAYER (src/analysis/)      VISUALIZATION LAYER (src/visualization/) │

│   ══════════════════════════════      ═════════════════════════════════════════││   │    subgraph STAGE2 ["🗺️ STAGE 2: OD MATRIX"]    end

│                                                                                  │

│   01_temporal_analysis.py              01_temporal_statistics.py                 ││   └── data/                         # Data processing

│   02_od_matrix_analysis.py             02_od_statistics.py                       │

│   03_integration_analysis.py  ─────▶   03_integration_statistics.py              ││       └── 01_data_cleaning.py       # ETL pipeline        D1[03_od_matrices.py]

│   04_parking_analysis.py     CHECKPOINTS 04_parking_survival.py                  │

│   05_economic_analysis.py              05_economic_sensitivity.py                ││

│                                                                                  │

│   Runtime: ~30 min each                Runtime: ~2 min each                      │├── 📂 data/                          # Data directory (git-ignored)        D2[src/analysis/02_od_matrix_q1.py]    subgraph STAGE3 ["⚙️ STAGE 3: CALCULATION"]

│   CPU-bound (computation)              I/O-bound (plotting)                      │  ├── raw/                          # Original operator data

│   Run ONCE per data update             Run MANY times for styling                │   │   ├── bird/                     # BIRD CSV files        D3[src/visualization/02_od_matrix_plots.py]        D1[04_transport_comparison.py]

```

│   │   ├── lime/                     # LIME CSV files

### Benefits

│   │   ├── voi/                      # VOI XLSX files    end        D2[Buffer Analysis]

| Benefit | Monolithic | Decoupled |

|---------|------------|-----------|│   │   ├── gtfs/                     # GTFS bundle

| **Visualization Iteration** | 30+ min | ~2 min |

| **Fault Recovery** | Start over | Resume from checkpoint |│   │   └── zone_statistiche_geo/     # Zone shapefile        D3[Temporal Segmentation]

| **Memory Usage** | High peak | Isolated per stage |

│   │

---

│   └── processed/                    # Cleaned datasets    subgraph STAGE3 ["🔗 STAGE 3: INTEGRATION"]        D4[Tortuosity Calculation]

## 📚 Data Dictionary

│       ├── lime_cleaned.csv

### Standardized Schema (Post-Preprocessing)

│       ├── voi_cleaned.csv        E1[04_transport_comparison.py]    end

| Column | Type | Description |

|--------|------|-------------|│       ├── bird_cleaned.csv

| `operator` | str | BIRD, LIME, VOI |

| `start_time` | datetime | Trip start (UTC+1) |│       └── df_all.pkl        E2[src/analysis/03_integration_q1.py]

| `end_time` | datetime | Trip end (UTC+1) |

| `start_lat`, `start_lon` | float | Origin (WGS84) |│

| `end_lat`, `end_lon` | float | Destination (WGS84) |

| `distance_km` | float | Trip distance |├── 📂 outputs/        E3[src/visualization/03_integration_plots.py]    subgraph CHECKPOINTS ["💾 CHECKPOINTS"]

| `duration_min` | float | Trip duration |

| `hour` | int | Hour of day (0-23) |│   ├── figures/                      # Generated visualizations

| `day_of_week` | int | Day (0=Mon, 6=Sun) |

| `is_weekend` | bool | Saturday or Sunday |│   │   ├── exercise1/                # ~10 PNG files    end        E1[.pkl files]



### Checkpoint Files Reference│   │   ├── exercise2/                # ~15 PNG files



| Exercise | Checkpoint File | Contents |│   │   ├── exercise3/                # ~17 PNG files        E2[.geojson files]

|----------|-----------------|----------|

| 1 | `checkpoint_hourly_stats.csv` | Hourly aggregations |│   │   ├── exercise4/                # ~12 PNG files

| 2 | `checkpoint_od_matrix.pkl` | Full OD matrix |

| 3 | `checkpoint_buffer_sensitivity.pkl` | Multi-buffer results |│   │   └── exercise5/                # ~10 PNG files    subgraph STAGE4 ["🅿️ STAGE 4: PARKING"]        E3[.csv summaries]

| 4 | `checkpoint_parking_stats.csv` | Duration statistics |

| 5 | `checkpoint_monte_carlo_summary.csv` | Risk analysis |│   │



---│   └── reports/                      # Analysis reports        F1[src/analysis/04_parking_q1.py]    end



## ⚡ Key Algorithms│       ├── exercise1/



### 1. Vectorized Buffer Analysis│       ├── exercise2/        F2[src/visualization/04_parking_plots.py]



**Challenge**: 549K trips × 1,500 PT stops = 824M distance checks│       ├── exercise3/



**Solution**: Pre-computed coverage zones with vectorized containment│       ├── exercise4/    end    subgraph STAGE4 ["🎨 STAGE 4: VISUALIZATION"]



```python│       └── exercise5/

for buffer_distance in [50, 100, 200]:

    pt_coverage = unary_union([stop.buffer(buffer_distance) for stop in stops])│        F1[04_visualization.py]

    prepared_coverage = prep(pt_coverage)

    is_near = trips_gdf.geometry.within(prepared_coverage)└── 📂 archive/                       # Deprecated scripts (git-ignored)

```

```    subgraph STAGE5 ["💰 STAGE 5: ECONOMICS"]        F2[Professional Figures]

**Speedup**: 100× faster than naive approach



### 2. Weibull Survival Analysis

---        G1[src/analysis/05_economic_q1.py]        F3[Report Tables]

$$S(t) = e^{-(t/\lambda)^k}$$



| Parameter | BIRD | LIME | VOI |

|-----------|------|------|-----|## 🎯 The 5 Research Questions        G2[src/visualization/05_economic_plots.py]    end

| Shape (k) | 0.615 | 0.628 | 0.570 |

| Scale (λ) | 12.0h | 6.5h | 22.8h |



### 3. Monte Carlo Profit Simulation### Exercise 1: Temporal Pattern Analysis    end



10,000 iterations with random parameter sampling**Research Question**: *How do e-scooter usage patterns vary by time?



**Risk Metrics**: P(loss) = 0.52%, VaR(5%) = €1.23M    subgraph OUTPUT ["📤 OUTPUTS"]



---**Methods**: Kruskal-Wallis H-test, Chi-square, Bootstrap CI



## 🛠️ Technology Stack    subgraph OUTPUT ["📤 OUTPUT LAYER"]        G1[PNG Figures]



### Core Libraries**Key Metrics**: Peak hours, weekend share, monthly trends



| Library | Version | Purpose |        H1[PNG Figures]        G2[CSV Reports]

|---------|---------|---------|

| **pandas** | ≥2.0 | Data manipulation |---

| **geopandas** | ≥0.14 | Spatial DataFrames |

| **shapely** | ≥2.0 | Geometry operations |        H2[CSV Reports]        G3[GeoJSON Maps]

| **numpy** | ≥1.24 | Numerical computing |

| **scipy** | ≥1.10 | Statistical analysis |### Exercise 2: Origin-Destination Matrix Analysis

| **matplotlib** | ≥3.7 | Visualization |

| **seaborn** | ≥0.12 | Statistical plots |**Research Question**: *What are the primary mobility corridors?*        H3[Markdown Reports]    end



### Coordinate Reference Systems



| CRS | EPSG | Usage |**Methods**: Chi-square test, Cramér's V, Gini coefficient        H4[Pickle Checkpoints]

|-----|------|-------|

| WGS84 | 4326 | Input/storage |

| UTM 32N | 32632 | Metric calculations |

| Web Mercator | 3857 | Basemap visualization |**Key Metrics**: Zone flows, corridor rankings, concentration    end    A1 & A2 & A3 --> B1



---



## 🚀 Pipeline Execution---    A4 & A5 --> D1



### Full Pipeline



```bash### Exercise 3: Public Transport Integration Analysis    A1 & A2 & A3 --> B1    B1 --> B2 --> B3 --> B4

python run_pipeline.py --stages 0,1,2,3,4,5

```**Research Question**: *Are e-scooters competitors or allies to public transport?*



### Resource Requirements    A4 & A5 --> E1    B4 --> C1 & C2



| Stage | Peak RAM | Runtime |**Methods**: Buffer analysis, temporal segmentation

|-------|----------|---------|

| 0 (Preprocessing) | 4 GB | 5 min |    B1 --> C1 --> C2 --> C3    C1 & C2 --> D1

| 1 (Temporal) | 3 GB | 10 min |

| 2 (OD Matrix) | 6 GB | 15 min |**Key Metrics**: Integration Index, Feeder Rate

| 3 (Integration) | 8 GB | 30 min |

| 4 (Parking) | 4 GB | 20 min |    C1 --> D1 --> D2 --> D3    D1 --> D2 & D3 & D4

| 5 (Economics) | 2 GB | 10 min |

---

---

    D1 --> E1 --> E2 --> E3    D2 & D3 & D4 --> E1 & E2 & E3

## 📊 Output Artifacts

### Figures by Exercise

**Research Question**: *How long do e-scooters remain parked?*    E1 --> F1 --> F2    E1 & E2 & E3 --> F1

| Exercise | Count | Key Figures |

|----------|-------|-------------|

| 1 | ~10 | Hourly patterns, heatmaps, data cleaning waterfall, bad data breakdown |

| 2 | ~15 | OD flows, choropleths |

| 3 | ~17 | Buffer sensitivity, integration |

| 4 | ~12 | Survival curves, hazard |

| 5 | ~10 | Monte Carlo, sensitivity |

**Key Metrics**: Median duration, abandonment rate    C3 & D3 & E3 & F2 & G2 --> H1 & H2 & H3    F2 & F3 --> G1 & G2 & G3

### Reports



All exercises have detailed Markdown reports with:

- Statistical test results---    C2 & D2 & E2 & F1 & G1 --> H4```

- LaTeX-ready tables

- Figure references



---### Exercise 5: Economic Analysis```



## 🔒 Quality Assurance**Research Question**: *What is the financial viability?*



### Data Validation### Text-Based Alternative (for non-Mermaid renderers)



| Check | Stage | Action |**Methods**: Monte Carlo simulation, sensitivity analysis

|-------|-------|--------|

| Coordinate bounds | Preprocessing | Drop invalid |### Text-Based Alternative

| Temporal consistency | Preprocessing | Correct dates |

| Missing values | Preprocessing | Impute or flag |**Key Metrics**: Revenue, profit margin, P(loss)

| Duplicate trips | Preprocessing | Deduplicate |

```

### Statistical Rigor

---

- Bonferroni-corrected p-values

- Effect sizes (η², Cramér's V)```┌─────────────────────────────────────────────────────────────────────────────────┐

- 95% bootstrap confidence intervals

- Non-parametric tests for non-normal data## 🎨 Decoupled Design Pattern



---┌─────────────────────────────────────────────────────────────────────────────────┐│                           SYSTEM ARCHITECTURE                                    │



## 📚 References### The Problem: Monolithic Analysis Scripts



1. **Buffer Analysis**: EU Standard EN13816│                       TURIN SMART MOBILITY PIPELINE v3.0                        ││                      Turin Smart Mobility Pipeline                               │

2. **Survival Analysis**: Weibull distribution, Kaplan-Meier

3. **Economic Modeling**: Monte Carlo methods```python

4. **Spatial Indexing**: Shapely STRtree, GEOS

# ❌ ANTI-PATTERN: Monolithic Script└─────────────────────────────────────────────────────────────────────────────────┘└─────────────────────────────────────────────────────────────────────────────────┘

---

def main():

<div align="center">

    df = load_data()           # 2 min

**Technical Architecture Document v3.0**

    results = heavy_calc(df)   # 30 min  ← Must re-run for any change

*Turin Smart Mobility Project • December 2025*

    plot_results(results)      # 1 min

    │   │   ├── 03_integration_statistics.py # Integration stats        C3[src/visualization/01_temporal_plots.py]    subgraph STAGE2 ["📊 STAGE 2: ANALYSIS"]

# Total: 33 min for a single plot color change!

```│   │   ├── 03_integration_maps.py    # Integration maps



### Our Solution: Separated Layers│   │   ├── 04_parking_survival.py    # Survival analysis plots    end        C1[02_analysis.py<br/>Descriptive Stats]



```│   │   ├── 04_parking_maps.py        # Parking heatmaps

┌─────────────────────────────────────────────────────────────────────────────────┐

│                         DECOUPLED ARCHITECTURE                                   ││   │   ├── 05_economic_sensitivity.py # Sensitivity analysis        C2[03_od_matrices.py<br/>O-D Flows]

├─────────────────────────────────────────────────────────────────────────────────┤

│                                                                                  ││   │   └── 05_economic_maps.py       # Economic visualizations

│   ANALYSIS LAYER (src/analysis/)      VISUALIZATION LAYER (src/visualization/) │

│   ══════════════════════════════      ═════════════════════════════════════════││   │    subgraph STAGE2 ["🗺️ STAGE 2: OD MATRIX"]    end

│                                                                                  │

│   01_temporal_analysis.py              01_temporal_statistics.py                 ││   └── data/                         # Data processing

│   02_od_matrix_analysis.py             02_od_statistics.py                       │

│   03_integration_analysis.py  ─────▶   03_integration_statistics.py              ││       └── 01_data_cleaning.py       # ETL pipeline        D1[03_od_matrices.py]

│   04_parking_analysis.py     CHECKPOINTS 04_parking_survival.py                  │

│   05_economic_analysis.py              05_economic_sensitivity.py                ││

│                                                                                  │

│   Runtime: ~30 min each                Runtime: ~2 min each                      │├── 📂 data/                          # Data directory (git-ignored)        D2[src/analysis/02_od_matrix_q1.py]    subgraph STAGE3 ["⚙️ STAGE 3: CALCULATION"]

│   CPU-bound (computation)              I/O-bound (plotting)                      │  ├── raw/                          # Original operator data

│   Run ONCE per data update             Run MANY times for styling                │   │   ├── bird/                     # BIRD CSV files        D3[src/visualization/02_od_matrix_plots.py]        D1[04_transport_comparison.py]

```

│   │   ├── lime/                     # LIME CSV files

### Benefits

│   │   ├── voi/                      # VOI XLSX files    end        D2[Buffer Analysis]

| Benefit | Monolithic | Decoupled |

|---------|------------|-----------|│   │   ├── gtfs/                     # GTFS bundle

| **Visualization Iteration** | 30+ min | ~2 min |

| **Fault Recovery** | Start over | Resume from checkpoint |│   │   └── zone_statistiche_geo/     # Zone shapefile        D3[Temporal Segmentation]

| **Memory Usage** | High peak | Isolated per stage |

│   │

---

│   └── processed/                    # Cleaned datasets    subgraph STAGE3 ["🔗 STAGE 3: INTEGRATION"]        D4[Tortuosity Calculation]

## 📚 Data Dictionary

│       ├── lime_cleaned.csv

### Standardized Schema (Post-Preprocessing)

│       ├── voi_cleaned.csv        E1[04_transport_comparison.py]    end

| Column | Type | Description |

|--------|------|-------------|│       ├── bird_cleaned.csv

| `operator` | str | BIRD, LIME, VOI |

| `start_time` | datetime | Trip start (UTC+1) |│       └── df_all.pkl        E2[src/analysis/03_integration_q1.py]

| `end_time` | datetime | Trip end (UTC+1) |

| `start_lat`, `start_lon` | float | Origin (WGS84) |│

| `end_lat`, `end_lon` | float | Destination (WGS84) |

| `distance_km` | float | Trip distance |├── 📂 outputs/        E3[src/visualization/03_integration_plots.py]    subgraph CHECKPOINTS ["💾 CHECKPOINTS"]

| `duration_min` | float | Trip duration |

| `hour` | int | Hour of day (0-23) |│   ├── figures/                      # Generated visualizations

| `day_of_week` | int | Day (0=Mon, 6=Sun) |

| `is_weekend` | bool | Saturday or Sunday |│   │   ├── exercise1/                # ~10 PNG files    end        E1[.pkl files]



### Checkpoint Files Reference│   │   ├── exercise2/                # ~15 PNG files



| Exercise | Checkpoint File | Contents |│   │   ├── exercise3/                # ~17 PNG files        E2[.geojson files]

|----------|-----------------|----------|

| 1 | `checkpoint_hourly_stats.csv` | Hourly aggregations |│   │   ├── exercise4/                # ~12 PNG files

| 2 | `checkpoint_od_matrix.pkl` | Full OD matrix |

| 3 | `checkpoint_buffer_sensitivity.pkl` | Multi-buffer results |│   │   └── exercise5/                # ~10 PNG files    subgraph STAGE4 ["🅿️ STAGE 4: PARKING"]        E3[.csv summaries]

| 4 | `checkpoint_parking_stats.csv` | Duration statistics |

| 5 | `checkpoint_monte_carlo_summary.csv` | Risk analysis |│   │



---│   └── reports/                      # Analysis reports        F1[src/analysis/04_parking_q1.py]    end



## ⚡ Key Algorithms│       ├── exercise1/



### 1. Vectorized Buffer Analysis│       ├── exercise2/        F2[src/visualization/04_parking_plots.py]



**Challenge**: 549K trips × 1,500 PT stops = 824M distance checks│       ├── exercise3/



**Solution**: Pre-computed coverage zones with vectorized containment│       ├── exercise4/    end    subgraph STAGE4 ["🎨 STAGE 4: VISUALIZATION"]



```python│       └── exercise5/

for buffer_distance in [50, 100, 200]:

    pt_coverage = unary_union([stop.buffer(buffer_distance) for stop in stops])│        F1[04_visualization.py]

    prepared_coverage = prep(pt_coverage)

    is_near = trips_gdf.geometry.within(prepared_coverage)└── 📂 archive/                       # Deprecated scripts (git-ignored)

```

```    subgraph STAGE5 ["💰 STAGE 5: ECONOMICS"]        F2[Professional Figures]

**Speedup**: 100× faster than naive approach



### 2. Weibull Survival Analysis

---        G1[src/analysis/05_economic_q1.py]        F3[Report Tables]

$$S(t) = e^{-(t/\lambda)^k}$$



| Parameter | BIRD | LIME | VOI |

|-----------|------|------|-----|## 🎯 The 5 Research Questions        G2[src/visualization/05_economic_plots.py]    end

| Shape (k) | 0.615 | 0.628 | 0.570 |

| Scale (λ) | 12.0h | 6.5h | 22.8h |



### 3. Monte Carlo Profit Simulation### Exercise 1: Temporal Pattern Analysis    end



10,000 iterations with random parameter sampling**Research Question**: *How do e-scooter usage patterns vary by time?



**Risk Metrics**: P(loss) = 0.52%, VaR(5%) = €1.23M    subgraph OUTPUT ["📤 OUTPUTS"]



---**Methods**: Kruskal-Wallis H-test, Chi-square, Bootstrap CI



## 🛠️ Technology Stack    subgraph OUTPUT ["📤 OUTPUT LAYER"]        G1[PNG Figures]



### Core Libraries**Key Metrics**: Peak hours, weekend share, monthly trends



| Library | Version | Purpose |        H1[PNG Figures]        G2[CSV Reports]

|---------|---------|---------|

| **pandas** | ≥2.0 | Data manipulation |---

| **geopandas** | ≥0.14 | Spatial DataFrames |

| **shapely** | ≥2.0 | Geometry operations |        H2[CSV Reports]        G3[GeoJSON Maps]

| **numpy** | ≥1.24 | Numerical computing |

| **scipy** | ≥1.10 | Statistical analysis |### Exercise 2: Origin-Destination Matrix Analysis

| **matplotlib** | ≥3.7 | Visualization |

| **seaborn** | ≥0.12 | Statistical plots |**Research Question**: *What are the primary mobility corridors?*        H3[Markdown Reports]    end



### Coordinate Reference Systems



| CRS | EPSG | Usage |**Methods**: Chi-square test, Cramér's V, Gini coefficient        H4[Pickle Checkpoints]

|-----|------|-------|

| WGS84 | 4326 | Input/storage |

| UTM 32N | 32632 | Metric calculations |

| Web Mercator | 3857 | Basemap visualization |**Key Metrics**: Zone flows, corridor rankings, concentration    end    A1 & A2 & A3 --> B1



---



## 🚀 Pipeline Execution---    A4 & A5 --> D1



### Full Pipeline



```bash### Exercise 3: Public Transport Integration Analysis    A1 & A2 & A3 --> B1    B1 --> B2 --> B3 --> B4

python run_pipeline.py --stages 0,1,2,3,4,5

```**Research Question**: *Are e-scooters competitors or allies to public transport?*



### Resource Requirements    A4 & A5 --> E1    B4 --> C1 & C2



| Stage | Peak RAM | Runtime |**Methods**: Buffer analysis, temporal segmentation

|-------|----------|---------|

| 0 (Preprocessing) | 4 GB | 5 min |    B1 --> C1 --> C2 --> C3    C1 & C2 --> D1

| 1 (Temporal) | 3 GB | 10 min |

| 2 (OD Matrix) | 6 GB | 15 min |**Key Metrics**: Integration Index, Feeder Rate

| 3 (Integration) | 8 GB | 30 min |

| 4 (Parking) | 4 GB | 20 min |    C1 --> D1 --> D2 --> D3    D1 --> D2 & D3 & D4

| 5 (Economics) | 2 GB | 10 min |

---

---

    D1 --> E1 --> E2 --> E3    D2 & D3 & D4 --> E1 & E2 & E3

## 📊 Output Artifacts

### Figures by Exercise

**Research Question**: *How long do e-scooters remain parked?*    E1 --> F1 --> F2    E1 & E2 & E3 --> F1

| Exercise | Count | Key Figures |

|----------|-------|-------------|

| 1 | ~10 | Hourly patterns, heatmaps, data cleaning waterfall, bad data breakdown |

| 2 | ~15 | OD flows, choropleths |

| 3 | ~17 | Buffer sensitivity, integration |

| 4 | ~12 | Survival curves, hazard |

| 5 | ~10 | Monte Carlo, sensitivity |

**Key Metrics**: Median duration, abandonment rate    C3 & D3 & E3 & F2 & G2 --> H1 & H2 & H3    F2 & F3 --> G1 & G2 & G3

### Reports



All exercises have detailed Markdown reports with:

- Statistical test results---    C2 & D2 & E2 & F1 & G1 --> H4```

- LaTeX-ready tables

- Figure references



---### Exercise 5: Economic Analysis```



## 🔒 Quality Assurance**Research Question**: *What is the financial viability?*



### Data Validation### Text-Based Alternative (for non-Mermaid renderers)



| Check | Stage | Action |**Methods**: Monte Carlo simulation, sensitivity analysis

|-------|-------|--------|

| Coordinate bounds | Preprocessing | Drop invalid |### Text-Based Alternative

| Temporal consistency | Preprocessing | Correct dates |

| Missing values | Preprocessing | Impute or flag |**Key Metrics**: Revenue, profit margin, P(loss)

| Duplicate trips | Preprocessing | Deduplicate |

```

### Statistical Rigor

---

- Bonferroni-corrected p-values

- Effect sizes (η², Cramér's V)```┌─────────────────────────────────────────────────────────────────────────────────┐

- 95% bootstrap confidence intervals

- Non-parametric tests for non-normal data## 🎨 Decoupled Design Pattern



---┌─────────────────────────────────────────────────────────────────────────────────┐│                           SYSTEM ARCHITECTURE                                    │



## 📚 References### The Problem: Monolithic Analysis Scripts



1. **Buffer Analysis**: EU Standard EN13816│                       TURIN SMART MOBILITY PIPELINE v3.0                        ││                      Turin Smart Mobility Pipeline                               │

2. **Survival Analysis**: Weibull distribution, Kaplan-Meier

3. **Economic Modeling**: Monte Carlo methods```python

4. **Spatial Indexing**: Shapely STRtree, GEOS

# ❌ ANTI-PATTERN: Monolithic Script└─────────────────────────────────────────────────────────────────────────────────┘└─────────────────────────────────────────────────────────────────────────────────┘

---

def main():

<div align="center">

    df = load_data()           # 2 min

**Technical Architecture Document v3.0**

    results = heavy_calc(df)   # 30 min  ← Must re-run for any change

*Turin Smart Mobility Project • December 2025*

    plot_results(results)      # 1 min

    │   │   ├── 03_integration_statistics.py # Integration stats        C3[src/visualization/01_temporal_plots.py]    subgraph STAGE2 ["📊 STAGE 2: ANALYSIS"]

# Total: 33 min for a single plot color change!

```│   │   ├── 03_integration_maps.py    # Integration maps



### Our Solution: Separated Layers│   │   ├── 04_parking_survival.py    # Survival analysis plots    end        C1[02_analysis.py<br/>Descriptive Stats]



```│   │   ├── 04_parking_maps.py        # Parking heatmaps

┌─────────────────────────────────────────────────────────────────────────────────┐

│                         DECOUPLED ARCHITECTURE                                   ││   │   ├── 05_economic_sensitivity.py # Sensitivity analysis        C2[03_od_matrices.py<br/>O-D Flows]

├─────────────────────────────────────────────────────────────────────────────────┤

│                                                                                  ││   │   └── 05_economic_maps.py       # Economic visualizations

│   ANALYSIS LAYER (src/analysis/)      VISUALIZATION LAYER (src/visualization/) │

│   ══════════════════════════════      ═════════════════════════════════════════││   │    subgraph STAGE2 ["🗺️ STAGE 2: OD MATRIX"]    end

│                                                                                  │

│   01_temporal_analysis.py              01_temporal_statistics.py                 ││   └── data/                         # Data processing

│   02_od_matrix_analysis.py             02_od_statistics.py                       │

│   03_integration_analysis.py  ─────▶   03_integration_statistics.py              ││       └── 01_data_cleaning.py       # ETL pipeline        D1[03_od_matrices.py]

│   04_parking_analysis.py     CHECKPOINTS 04_parking_survival.py                  │

│   05_economic_analysis.py              05_economic_sensitivity.py                ││

│                                                                                  │

│   Runtime: ~30 min each                Runtime: ~2 min each                      │├── 📂 data/                          # Data directory (git-ignored)        D2[src/analysis/02_od_matrix_q1.py]    subgraph STAGE3 ["⚙️ STAGE 3: CALCULATION"]

│   CPU-bound (computation)              I/O-bound (plotting)                      │  ├── raw/                          # Original operator data

│   Run ONCE per data update             Run MANY times for styling                │   │   ├── bird/                     # BIRD CSV files        D3[src/visualization/02_od_matrix_plots.py]        D1[04_transport_comparison.py]

```

│   │   ├── lime/                     # LIME CSV files

### Benefits

│   │   ├── voi/                      # VOI XLSX files    end        D2[Buffer Analysis]

| Benefit | Monolithic | Decoupled |

|---------|------------|-----------|│   │   ├── gtfs/                     # GTFS bundle

| **Visualization Iteration** | 30+ min | ~2 min |

| **Fault Recovery** | Start over | Resume from checkpoint |│   │   └── zone_statistiche_geo/     # Zone shapefile        D3[Temporal Segmentation]

| **Memory Usage** | High peak | Isolated per stage |

│   │

---

│   └── processed/                    # Cleaned datasets    subgraph STAGE3 ["🔗 STAGE 3: INTEGRATION"]        D4[Tortuosity Calculation]

## 📚 Data Dictionary

│       ├── lime_cleaned.csv

### Standardized Schema (Post-Preprocessing)

│       ├── voi_cleaned.csv        E1[04_transport_comparison.py]    end

| Column | Type | Description |

|--------|------|-------------|│       ├── bird_cleaned.csv

| `operator` | str | BIRD, LIME, VOI |

| `start_time` | datetime | Trip start (UTC+1) |│       └── df_all.pkl        E2[src/analysis/03_integration_q1.py]

| `end_time` | datetime | Trip end (UTC+1) |

| `start_lat`, `start_lon` | float | Origin (WGS84) |│

| `end_lat`, `end_lon` | float | Destination (WGS84) |

| `distance_km` | float | Trip distance |├── 📂 outputs/        E3[src/visualization/03_integration_plots.py]    subgraph CHECKPOINTS ["💾 CHECKPOINTS"]

| `duration_min` | float | Trip duration |

| `hour` | int | Hour of day (0-23) |│   ├── figures/                      # Generated visualizations

| `day_of_week` | int | Day (0=Mon, 6=Sun) |

| `is_weekend` | bool | Saturday or Sunday |│   │   ├── exercise1/                # ~10 PNG files    end        E1[.pkl files]



### Checkpoint Files Reference│   │   ├── exercise2/                # ~15 PNG files



| Exercise | Checkpoint File | Contents |│   │   ├── exercise3/                # ~17 PNG files        E2[.geojson files]

|----------|-----------------|----------|

| 1 | `checkpoint_hourly_stats.csv` | Hourly aggregations |│   │   ├── exercise4/                # ~12 PNG files

| 2 | `checkpoint_od_matrix.pkl` | Full OD matrix |

| 3 | `checkpoint_buffer_sensitivity.pkl` | Multi-buffer results |│   │   └── exercise5/                # ~10 PNG files    subgraph STAGE4 ["🅿️ STAGE 4: PARKING"]        E3[.csv summaries]

| 4 | `checkpoint_parking_stats.csv` | Duration statistics |

| 5 | `checkpoint_monte_carlo_summary.csv` | Risk analysis |│   │



---│   └── reports/                      # Analysis reports        F1[src/analysis/04_parking_q1.py]    end



## ⚡ Key Algorithms│       ├── exercise1/



### 1. Vectorized Buffer Analysis│       ├── exercise2/        F2[src/visualization/04_parking_plots.py]



**Challenge**: 549K trips × 1,500 PT stops = 824M distance checks│       ├── exercise3/



**Solution**: Pre-computed coverage zones with vectorized containment│       ├── exercise4/    end    subgraph STAGE4 ["🎨 STAGE 4: VISUALIZATION"]



```python│       └── exercise5/

for buffer_distance in [50, 100, 200]:

    pt_coverage = unary_union([stop.buffer(buffer_distance) for stop in stops])│        F1[04_visualization.py]

    prepared_coverage = prep(pt_coverage)

    is_near = trips_gdf.geometry.within(prepared_coverage)└── 📂 archive/                       # Deprecated scripts (git-ignored)

```

```    subgraph STAGE5 ["💰 STAGE 5: ECONOMICS"]        F2[Professional Figures]

**Speedup**: 100× faster than naive approach



### 2. Weibull Survival Analysis

---        G1[src/analysis/05_economic_q1.py]        F3[Report Tables]

$$S(t) = e^{-(t/\lambda)^k}$$



| Parameter | BIRD | LIME | VOI |

|-----------|------|------|-----|## 🎯 The 5 Research Questions        G2[src/visualization/05_economic_plots.py]    end

| Shape (k) | 0.615 | 0.628 | 0.570 |

| Scale (λ) | 12.0h | 6.5h | 22.8h |



### 3. Monte Carlo Profit Simulation### Exercise 1: Temporal Pattern Analysis    end



10,000 iterations with random parameter sampling**Research Question**: *How do e-scooter usage patterns vary by time?



**Risk Metrics**: P(loss) = 0.52%, VaR(5%) = €1.23M    subgraph OUTPUT ["📤 OUTPUTS"]



---**Methods**: Kruskal-Wallis H-test, Chi-square, Bootstrap CI



## 🛠️ Technology Stack    subgraph OUTPUT ["📤 OUTPUT LAYER"]        G1[PNG Figures]



### Core Libraries**Key Metrics**: Peak hours, weekend share, monthly trends



| Library | Version | Purpose |        H1[PNG Figures]        G2[CSV Reports]

|---------|---------|---------|

| **pandas** | ≥2.0 | Data manipulation |---

| **geopandas** | ≥0.14 | Spatial DataFrames |

| **shapely** | ≥2.0 | Geometry operations |        H2[CSV Reports]        G3[GeoJSON Maps]

| **numpy** | ≥1.24 | Numerical computing |

| **scipy** | ≥1.10 | Statistical analysis |### Exercise 2: Origin-Destination Matrix Analysis

| **matplotlib** | ≥3.7 | Visualization |

| **seaborn** | ≥0.12 | Statistical plots |**Research Question**: *What are the primary mobility corridors?*        H3[Markdown Reports]    end



### Coordinate Reference Systems



| CRS | EPSG | Usage |**Methods**: Chi-square test, Cramér's V, Gini coefficient        H4[Pickle Checkpoints]

|-----|------|-------|

| WGS84 | 4326 | Input/storage |

| UTM 32N | 32632 | Metric calculations |

| Web Mercator | 3857 | Basemap visualization |**Key Metrics**: Zone flows, corridor rankings, concentration    end    A1 & A2 & A3 --> B1



---



## 🚀 Pipeline Execution---    A4 & A5 --> D1



### Full Pipeline



```bash### Exercise 3: Public Transport Integration Analysis    A1 & A2 & A3 --> B1    B1 --> B2 --> B3 --> B4

python run_pipeline.py --stages 0,1,2,3,4,5

```**Research Question**: *Are e-scooters competitors or allies to public transport?*



### Resource Requirements    A4 & A5 --> E1    B4 --> C1 & C2



| Stage | Peak RAM | Runtime |**Methods**: Buffer analysis, temporal segmentation

|-------|----------|---------|

| 0 (Preprocessing) | 4 GB | 5 min |    B1 --> C1 --> C2 --> C3    C1 & C2 --> D1

| 1 (Temporal) | 3 GB | 10 min |

| 2 (OD Matrix) | 6 GB | 15 min |**Key Metrics**: Integration Index, Feeder Rate

| 3 (Integration) | 8 GB | 30 min |

| 4 (Parking) | 4 GB | 20 min |    C1 --> D1 --> D2 --> D3    D1 --> D2 & D3 & D4

| 5 (Economics) | 2 GB | 10 min |

---

---

    D1 --> E1 --> E2 --> E3    D2 & D3 & D4 --> E1 & E2 & E3

## 📊 Output Artifacts

### Figures by Exercise

**Research Question**: *How long do e-scooters remain parked?*    E1 --> F1 --> F2    E1 & E2 & E3 --> F1

| Exercise | Count | Key Figures |

|----------|-------|-------------|

| 1 | ~10 | Hourly patterns, heatmaps, data cleaning waterfall, bad data breakdown |

| 2 | ~15 | OD flows, choropleths |

| 3 | ~17 | Buffer sensitivity, integration |

| 4 | ~12 | Survival curves, hazard |

| 5 | ~10 | Monte Carlo, sensitivity |

**Key Metrics**: Median duration, abandonment rate    C3 & D3 & E3 & F2 & G2 --> H1 & H2 & H3    F2 & F3 --> G1 & G2 & G3

### Reports



All exercises have detailed Markdown reports with:

- Statistical test results---    C2 & D2 & E2 & F1 & G1 --> H4```

- LaTeX-ready tables

- Figure references



---### Exercise 5: Economic Analysis```



## 🔒 Quality Assurance**Research Question**: *What is the financial viability?*



### Data Validation### Text-Based Alternative (for non-Mermaid renderers)



| Check | Stage | Action |**Methods**: Monte Carlo simulation, sensitivity analysis

|-------|-------|--------|

| Coordinate bounds | Preprocessing | Drop invalid |### Text-Based Alternative

| Temporal consistency | Preprocessing | Correct dates |

| Missing values | Preprocessing | Impute or flag |**Key Metrics**: Revenue, profit margin, P(loss)

| Duplicate trips | Preprocessing | Deduplicate |

```

### Statistical Rigor

---

- Bonferroni-corrected p-values

- Effect sizes (η², Cramér's V)```┌─────────────────────────────────────────────────────────────────────────────────┐

- 95% bootstrap confidence intervals

- Non-parametric tests for non-normal data## 🎨 Decoupled Design Pattern



---┌─────────────────────────────────────────────────────────────────────────────────┐│                           SYSTEM ARCHITECTURE                                    │



## 📚 References### The Problem: Monolithic Analysis Scripts



1. **Buffer Analysis**: EU Standard EN13816│                       TURIN SMART MOBILITY PIPELINE v3.0                        ││                      Turin Smart Mobility Pipeline                               │

2. **Survival Analysis**: Weibull distribution, Kaplan-Meier

3. **Economic Modeling**: Monte Carlo methods```python

4. **Spatial Indexing**: Shapely STRtree, GEOS

# ❌ ANTI-PATTERN: Monolithic Script└─────────────────────────────────────────────────────────────────────────────────┘└─────────────────────────────────────────────────────────────────────────────────┘

---

def main():

<div align="center">

    df = load_data()           # 2 min

**Technical Architecture Document v3.0**

    results = heavy_calc(df)   # 30 min  ← Must re-run for any change

*Turin Smart Mobility Project • December 2025*

    plot_results(results)      # 1 min

    │   │   ├── 03_integration_statistics.py # Integration stats        C3[src/visualization/01_temporal_plots.py]    subgraph STAGE2 ["📊 STAGE 2: ANALYSIS"]

# Total: 33 min for a single plot color change!

```│   │   ├── 03_integration_maps.py    # Integration maps



### Our Solution: Separated Layers│   │   ├── 04_parking_survival.py    # Survival analysis plots    end        C1[02_analysis.py<br/>Descriptive Stats]



```│   │   ├── 04_parking_maps.py        # Parking heatmaps

┌─────────────────────────────────────────────────────────────────────────────────┐

│                         DECOUPLED ARCHITECTURE                                   ││   │   ├── 05_economic_sensitivity.py # Sensitivity analysis        C2[03_od_matrices.py<br/>O-D Flows]

├─────────────────────────────────────────────────────────────────────────────────┤

│                                                                                  ││   │   └── 05_economic_maps.py       # Economic visualizations

│   ANALYSIS LAYER (src/analysis/)      VISUALIZATION LAYER (src/visualization/) │

│   ══════════════════════════════      ═════════════════════════════════════════││   │    subgraph STAGE2 ["🗺️ STAGE 2: OD MATRIX"]    end

│                                                                                  │

│   01_temporal_analysis.py              01_temporal_statistics.py                 ││   └── data/                         # Data processing

│   02_od_matrix_analysis.py             02_od_statistics.py                       │

│   03_integration_analysis.py  ─────▶   03_integration_statistics.py              ││       └── 01_data_cleaning.py       # ETL pipeline        D1[03_od_matrices.py]

│   04_parking_analysis.py     CHECKPOINTS 04_parking_survival.py                  │

│   05_economic_analysis.py              05_economic_sensitivity.py                ││

│                                                                                  │

│   Runtime: ~30 min each                Runtime: ~2 min each                      │├── 📂 data/                          # Data directory (git-ignored)        D2[src/analysis/02_od_matrix_q1.py]    subgraph STAGE3 ["⚙️ STAGE 3: CALCULATION"]

│   CPU-bound (computation)              I/O-bound (plotting)                      │  ├── raw/                          # Original operator data

│   Run ONCE per data update             Run MANY times for styling                │   │   ├── bird/                     # BIRD CSV files        D3[src/visualization/02_od_matrix_plots.py]        D1[04_transport_comparison.py]

```

│   │   ├── lime/                     # LIME CSV files

### Benefits

│   │   ├── voi/                      # VOI XLSX files    end        D2[Buffer Analysis]

| Benefit | Monolithic | Decoupled |

|---------|------------|-----------|│   │   ├── gtfs/                     # GTFS bundle

| **Visualization Iteration** | 30+ min | ~2 min |

| **Fault Recovery** | Start over | Resume from checkpoint |│   │   └── zone_statistiche_geo/     # Zone shapefile        D3[Temporal Segmentation]

| **Memory Usage** | High peak | Isolated per stage |

│   │

---

│   └── processed/                    # Cleaned datasets    subgraph STAGE3 ["🔗 STAGE 3: INTEGRATION"]        D4[Tortuosity Calculation]

## 📚 Data Dictionary

│       ├── lime_cleaned.csv

### Standardized Schema (Post-Preprocessing)

│       ├── voi_cleaned.csv        E1[04_transport_comparison.py]    end

| Column | Type | Description |

|--------|------|-------------|│       ├── bird_cleaned.csv

| `operator` | str | BIRD, LIME, VOI |

| `start_time` | datetime | Trip start (UTC+1) |│       └── df_all.pkl        E2[src/analysis/03_integration_q1.py]

| `end_time` | datetime | Trip end (UTC+1) |

| `start_lat`, `start_lon` | float | Origin (WGS84) |│

| `end_lat`, `end_lon` | float | Destination (WGS84) |

| `distance_km` | float | Trip distance |├── 📂 outputs/        E3[src/visualization/03_integration_plots.py]    subgraph CHECKPOINTS ["💾 CHECKPOINTS"]

| `duration_min` | float | Trip duration |

| `hour` | int | Hour of day (0-23) |│   ├── figures/                      # Generated visualizations

| `day_of_week` | int | Day (0=Mon, 6=Sun) |

| `is_weekend` | bool | Saturday or Sunday |│   │   ├── exercise1/                # ~10 PNG files    end        E1[.pkl files]



### Checkpoint Files Reference│   │   ├── exercise2/                # ~15 PNG files



| Exercise | Checkpoint File | Contents |│   │   ├── exercise3/                # ~17 PNG files        E2[.geojson files]

|----------|-----------------|----------|

| 1 | `checkpoint_hourly_stats.csv` | Hourly aggregations |│   │   ├── exercise4/                # ~12 PNG files

| 2 | `checkpoint_od_matrix.pkl` | Full OD matrix |

| 3 | `checkpoint_buffer_sensitivity.pkl` | Multi-buffer results |│   │   └── exercise5/                # ~10 PNG files    subgraph STAGE4 ["🅿️ STAGE 4: PARKING"]        E3[.csv summaries]

| 4 | `checkpoint_parking_stats.csv` | Duration statistics |

| 5 | `checkpoint_monte_carlo_summary.csv` | Risk analysis |│   │



---│   └── reports/                      # Analysis reports        F1[src/analysis/04_parking_q1.py]    end



## ⚡ Key Algorithms│       ├── exercise1/



### 1. Vectorized Buffer Analysis│       ├── exercise2/        F2[src/visualization/04_parking_plots.py]



**Challenge**: 549K trips × 1,500 PT stops = 824M distance checks│       ├── exercise3/



**Solution**: Pre-computed coverage zones with vectorized containment│       ├── exercise4/    end    subgraph STAGE4 ["🎨 STAGE 4: VISUALIZATION"]



```python│       └── exercise5/

for buffer_distance in [50, 100, 200]:

    pt_coverage = unary_union([stop.buffer(buffer_distance) for stop in stops])│        F1[04_visualization.py]

    prepared_coverage = prep(pt_coverage)

    is_near = trips_gdf.geometry.within(prepared_coverage)└── 📂 archive/                       # Deprecated scripts (git-ignored)

```

```    subgraph STAGE5 ["💰 STAGE 5: ECONOMICS"]        F2[Professional Figures]

**Speedup**: 100× faster than naive approach



### 2. Weibull Survival Analysis

---        G1[src/analysis/05_economic_q1.py]        F3[Report Tables]

$$S(t) = e^{-(t/\lambda)^k}$$



| Parameter | BIRD | LIME | VOI |

|-----------|------|------|-----|## 🎯 The 5 Research Questions        G2[src/visualization/05_economic_plots.py]    end

| Shape (k) | 0.615 | 0.628 | 0.570 |

| Scale (λ) | 12.0h | 6.5h | 22.8h |



### 3. Monte Carlo Profit Simulation### Exercise 1: Temporal Pattern Analysis    end



10,000 iterations with random parameter sampling**Research Question**: *How do e-scooter usage patterns vary by time?



**Risk Metrics**: P(loss) = 0.52%, VaR(5%) = €1.23M    subgraph OUTPUT ["📤 OUTPUTS"]



---**Methods**: Kruskal-Wallis H-test, Chi-square, Bootstrap CI



## 🛠️ Technology Stack    subgraph OUTPUT ["📤 OUTPUT LAYER"]        G1[PNG Figures]



### Core Libraries**Key Metrics**: Peak hours, weekend share, monthly trends



| Library | Version | Purpose |        H1[PNG Figures]        G2[CSV Reports]

|---------|---------|---------|

| **pandas** | ≥2.0 | Data manipulation |---

| **geopandas** | ≥0.14 | Spatial DataFrames |

| **shapely** | ≥2.0 | Geometry operations |        H2[CSV Reports]        G3[GeoJSON Maps]

| **numpy** | ≥1.24 | Numerical computing |

| **scipy** | ≥1.10 | Statistical analysis |### Exercise 2: Origin-Destination Matrix Analysis

| **matplotlib** | ≥3.7 | Visualization |

| **seaborn** | ≥0.12 | Statistical plots |**Research Question**: *What are the primary mobility corridors?*        H3[Markdown Reports]    end



### Coordinate Reference Systems



| CRS | EPSG | Usage |**Methods**: Chi-square test, Cramér's V, Gini coefficient        H4[Pickle Checkpoints]

|-----|------|-------|

| WGS84 | 4326 | Input/storage |

| UTM 32N | 32632 | Metric calculations |

| Web Mercator | 3857 | Basemap visualization |**Key Metrics**: Zone flows, corridor rankings, concentration    end    A1 & A2 & A3 --> B1



---



## 🚀 Pipeline Execution---    A4 & A5 --> D1



### Full Pipeline



```bash### Exercise 3: Public Transport Integration Analysis    A1 & A2 & A3 --> B1    B1 --> B2 --> B3 --> B4

python run_pipeline.py --stages 0,1,2,3,4,5

```**Research Question**: *Are e-scooters competitors or allies to public transport?*



### Resource Requirements    A4 & A5 --> E1    B4 --> C1 & C2



| Stage | Peak RAM | Runtime |**Methods**: Buffer analysis, temporal segmentation

|-------|----------|---------|

| 0 (Preprocessing) | 4 GB | 5 min |    B1 --> C1 --> C2 --> C3    C1 & C2 --> D1

| 1 (Temporal) | 3 GB | 10 min |

| 2 (OD Matrix) | 6 GB | 15 min |**Key Metrics**: Integration Index, Feeder Rate

| 3 (Integration) | 8 GB | 30 min |

| 4 (Parking) | 4 GB | 20 min |    C1 --> D1 --> D2 --> D3    D1 --> D2 & D3 & D4

| 5 (Economics) | 2 GB | 10 min |

---

---

    D1 --> E1 --> E2 --> E3    D2 & D3 & D4 --> E1 & E2 & E3

## 📊 Output Artifacts

### Figures by Exercise

**Research Question**: *How long do e-scooters remain parked?*    E1 --> F1 --> F2    E1 & E2 & E3 --> F1

| Exercise | Count | Key Figures |

|----------|-------|-------------|

| 1 | ~10 | Hourly patterns, heatmaps, data cleaning waterfall, bad data breakdown |

| 2 | ~15 | OD flows, choropleths |

| 3 | ~17 | Buffer sensitivity, integration |

| 4 | ~12 | Survival curves, hazard |

| 5 | ~10 | Monte Carlo, sensitivity |

**Key Metrics**: Median duration, abandonment rate    C3 & D3 & E3 & F2 & G2 --> H1 & H2 & H3    F2 & F3 --> G1 & G2 & G3

### Reports



All exercises have detailed Markdown reports with:

- Statistical test results---    C2 & D2 & E2 & F1 & G1 --> H4```

- LaTeX-ready tables

- Figure references



---### Exercise 5: Economic Analysis```



## 🔒 Quality Assurance**Research Question**: *What is the financial viability?*



### Data Validation### Text-Based Alternative (for non-Mermaid renderers)



| Check | Stage | Action |**Methods**: Monte Carlo simulation, sensitivity analysis

|-------|-------|--------|

| Coordinate bounds | Preprocessing | Drop invalid |### Text-Based Alternative

| Temporal consistency | Preprocessing | Correct dates |

| Missing values | Preprocessing | Impute or flag |**Key Metrics**: Revenue, profit margin, P(loss)

| Duplicate trips | Preprocessing | Deduplicate |

```

### Statistical Rigor

---

- Bonferroni-corrected p-values

- Effect sizes (η², Cramér's V)```┌─────────────────────────────────────────────────────────────────────────────────┐

- 95% bootstrap confidence intervals

- Non-parametric tests for non-normal data## 🎨 Decoupled Design Pattern



---┌─────────────────────────────────────────────────────────────────────────────────┐│                           SYSTEM ARCHITECTURE                                    │



## 📚 References### The Problem: Monolithic Analysis Scripts



1. **Buffer Analysis**: EU Standard EN13816, Stockholm PT Study (2024): "Impact of catchment area definition on micro-mobility integration metrics"

```   - EU Standard EN13816: "Transportation — Logistics and services"



### Resource Requirements2. **Tortuosity Index**

   - Schläpfer et al. (2021): "The universal visitation law of human mobility"

| Stage | Peak RAM | Runtime | Dependencies |   - Applied to distinguish commuting from exploration in urban micro-mobility

|-------|----------|---------|--------------|

| 0 | 4 GB | 5 min | None |3. **Spatial Indexing**

| 1 | 3 GB | 10 min | Stage 0 |   - Shapely Documentation: STRtree and Prepared Geometries

| 2 | 6 GB | 15 min | Stage 0 |   - PostGIS GEOS: Computational geometry algorithms

| 3 | 8 GB | 30 min | Stage 0, GTFS |

| 4 | 4 GB | 20 min | Stage 0 |---

| 5 | 2 GB | 10 min | Stage 0 |

<div align="center">

---

**Technical Architecture Document v2.0**

## 📊 Output Artifacts

*Turin Smart Mobility Project • December 2025*

### Generated Figures by Exercise

</div>

**Exercise 1 (Temporal)**:
- `hourly_distribution_by_operator.png`
- `daily_pattern_heatmap.png`
- `monthly_trend_analysis.png`
- `weekend_vs_weekday.png`

**Exercise 2 (OD Matrix)**:
- `od_flow_map_combined.png`
- `top_corridors_sankey.png`
- `zone_choropleth.png`

**Exercise 3 (Integration)**:
- `buffer_sensitivity_professional.png`
- `temporal_feeder_comparison.png`
- `hexbin_density_map.png`
- `correlation_heatmap.png`

**Exercise 4 (Parking)**:
- `survival_curves_comparison.png`
- `parking_duration_distribution.png`
- `hazard_functions.png`
- `abandonment_rates.png`

**Exercise 5 (Economics)**:
- `revenue_breakdown.png`
- `profit_margin_comparison.png`
- `monte_carlo_distribution.png`
- `scenario_tornado.png`

### Generated Reports

| Report | Location | Format |
|--------|----------|--------|
| Temporal Analysis | `outputs/reports/exercise1/EXERCISE1_DETAILED_REPORT.md` | Markdown |
| OD Matrix Analysis | `outputs/reports/exercise2/EXERCISE2_DETAILED_REPORT.md` | Markdown |
| Integration Analysis | `outputs/reports/exercise3/EXERCISE3_DETAILED_REPORT.md` | Markdown |
| Parking Analysis | `outputs/reports/exercise4/EXERCISE4_DETAILED_REPORT.md` | Markdown |
| Economic Analysis | `outputs/reports/exercise5/EXERCISE5_DETAILED_REPORT.md` | Markdown |

---

## 🔒 Quality Assurance

### Data Validation Pipeline

| Check | Stage | Action |
|-------|-------|--------|
| Coordinate bounds (Turin) | Preprocessing | Drop invalid |
| Temporal consistency | Preprocessing | Correct VOI dates |
| Missing values | Preprocessing | Impute or flag |
| Duplicate trips | Preprocessing | Deduplicate |
| Zero-length trips | Analysis | Exclude |
| Invalid geometries | Analysis | Repair or exclude |

### Statistical Rigor

- All p-values Bonferroni-corrected for multiple comparisons
- Effect sizes reported (η², Cramér's V)
- 95% confidence intervals via bootstrap (10,000 resamples)
- Non-parametric tests for non-normal distributions

---

## 📚 References

1. **Buffer Analysis**: EU Standard EN13816, Stockholm PT Study (2024): "Impact of catchment area definition on micro-mobility integration metrics"
2. **Survival Analysis**: Weibull distribution, Kaplan-Meier estimator
3. **Economic Modeling**: Monte Carlo methods, sensitivity analysis
4. **Spatial Indexing**: Shapely STRtree, GEOS algorithms

---

<div align="center">

**Technical Architecture Document v3.0**

*Turin Smart Mobility Project • December 2025*

</div>
