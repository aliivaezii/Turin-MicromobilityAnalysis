# 🛴 Turin Smart Mobility# 🛴 Turin Smart Mobility# 🛴 Turin Smart Mobility# 🛴 Turin Smart Mobility



## E-Scooter & Public Transport Integration Analysis## E-Scooter Sharing & Public Transport Integration Analysis



<div align="center">## E-Scooter Sharing & Public Transport Integration Analysis## E-Scooter & Public Transport Integration Analysis



**A Data-Driven Investigation into Shared Micro-Mobility Patterns in Turin, Italy**<div align="center">



[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14+-green.svg)](https://geopandas.org)

[![License](https://img.shields.io/badge/License-Academic-orange.svg)](#license)**A Comprehensive Data-Driven Investigation of Shared Micro-Mobility Patterns in Turin, Italy**

[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)](#)

<div align="center"><div align="center">

*Politecnico di Torino • Transport Engineering • Academic Year 2024-2025*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

</div>

[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14+-green.svg)](https://geopandas.org)

---

[![License](https://img.shields.io/badge/License-Academic-orange.svg)](#license)

## 📋 Executive Summary

[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)](#)**A Comprehensive Data-Driven Investigation of Shared Micro-Mobility Patterns in Turin, Italy****A Data-Driven Investigation into Shared Micro-Mobility Patterns in Turin, Italy**

This project presents a comprehensive spatial-temporal analysis of **549,513 e-scooter trips** across Turin's metropolitan area, investigating the integration patterns between shared micro-mobility services (BIRD, LIME, VOI) and the public transport network.



### 🎯 Central Research Question

*Politecnico di Torino • Transport Engineering • Academic Year 2024-2025*

> **"Are e-scooters competitors or allies to public transport in Turin?"**



### 📊 Key Findings

</div>[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)

| Metric | Value | Interpretation |

|--------|-------|----------------|

| **Integration Index (200m)** | 95.3% | Near-universal PT proximity |

| **Feeder Rate** | 82.4% | Strong first/last-mile role |---[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14+-green.svg)](https://geopandas.org)[![GeoPandas](https://img.shields.io/badge/GeoPandas-0.14+-green.svg)](https://geopandas.org)

| **Peak Hour Concentration** | 38.5% | Clear commuting patterns |

| **Market Size** | €8.31M/year | Sustainable business model |

| **Probability of Loss** | 0.52% | Low financial risk |

## 📋 Executive Summary[![License](https://img.shields.io/badge/License-Academic-orange.svg)](#license)[![License](https://img.shields.io/badge/License-Academic-orange.svg)](#)

**Conclusion**: E-scooters predominantly function as **first/last-mile connectors** rather than direct competitors to public transport.



---

This project presents a comprehensive spatial-temporal analysis of **549,513 e-scooter trips** across Turin's metropolitan area, investigating the integration patterns between shared micro-mobility services (BIRD, LIME, VOI) and the public transport network.[![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)](#)

## 🔬 Research Framework



### The 5 Research Exercises

### 🎯 Central Research Question*Politecnico di Torino • Transport Engineering • 2024-2025*

| Exercise | Topic | Research Focus |

|----------|-------|----------------|

| **1** | Temporal Pattern Analysis | Hourly, daily, and monthly usage patterns |

| **2** | Origin-Destination Matrix | Mobility corridors and zone flows |> **"Are e-scooters competitors or allies to public transport in Turin?"***Politecnico di Torino • Transport Engineering • Academic Year 2024-2025*

| **3** | Public Transport Integration | E-scooter proximity to PT stops |

| **4** | Parking Duration Analysis | Fleet turnover and survival analysis |

| **5** | Economic Analysis | Revenue modeling and Monte Carlo simulation |

### 📊 Key Findings</div>

---



## 📁 Project Structure

| Metric | Value | Interpretation |</div>

DATI MONOPATTINI SHARING-2/|--------|-------|----------------|

│

├── 📄 README.md                      # This file| **Integration Index (200m)** | 95.3% | Near-universal PT proximity |---

├── 📄 ARCHITECTURE.md                # Technical documentation

├── 📄 requirements.txt               # Python dependencies| **Feeder Rate** | 82.4% | Strong first/last-mile role |

├── 📄 run_pipeline.py                # Master pipeline controller

│| **Peak Hour Concentration** | 38.5% | Clear commuting patterns |---

├── 📂 src/

│   ├── analysis/                     # Statistical analysis modules| **Market Size** | €8.31M/year | Sustainable business model |

│   │   ├── 01_temporal_analysis.py

│   │   ├── 02_od_matrix_analysis.py| **Probability of Loss** | 0.52% | Low financial risk |## 📋 Executive Summary

│   │   ├── 03_integration_analysis.py

│   │   ├── 04_parking_analysis.py

│   │   └── 05_economic_analysis.py

│   │**Conclusion**: E-scooters predominantly function as **first/last-mile connectors** rather than direct competitors to public transport.## 📋 Executive Summary

│   ├── utils/                        # Utility modules (spatial, general)

│   │   ├── spatial_utils.py          # CRS, zone handling, spatial helpers

│   │   └── __init__.py               # Utils package init
│   ├── visualization/                # Visualization modules
│   │   ├── 00_data_cleaning.py       # Data cleaning waterfall & bad data charts
│   │   ├── 01_temporal_statistics.py
│   │   ├── 01_temporal_dashboard.py
│   │   ├── 02_od_statistics.py
│   │   ├── 02_od_spatial_flows.py
│   │   ├── 03_integration_statistics.py
│   │   ├── 03_integration_maps.py
│   │   ├── 04_parking_survival.py
│   │   ├── 04_parking_maps.py
│   │   ├── 05_economic_sensitivity.py
│   │   └── 05_economic_maps.py
│   └── data/                         # Data processing
│       └── 01_data_cleaning.py

├── 📂 data/                          # Data directory (git-ignored)| Exercise | Topic | Status |### 🎯 Central Research Question

│   ├── raw/                          # Original operator data

│   └── processed/                    # Cleaned datasets|----------|-------|--------|

│

└── 📂 outputs/| **1** | Temporal Pattern Analysis | ✅ Completed |> **"Are e-scooters competitors or allies to public transport in Turin?"**

    ├── figures/                      # Generated visualizations

    │   ├── exercise1/                # Descriptive analysis plots, data cleaning waterfall, bad data breakdown charts
    │   ├── exercise2/                # O-D flow maps
    │   ├── exercise3/                # Integration analysis figures
    │   ├── exercise4/                # Parking analysis figures
    │   ├── exercise5/                # Economic analysis figures
    └── reports/                      # Analysis reports

```| **5** | Economic & Sensitivity Analysis | ✅ Completed |Our analysis reveals that e-scooters predominantly function as **first/last-mile connectors** rather than direct competitors, with **95%+ of trips originating within 200m of public transport stops** during peak commuting hours.



---



## 🚀 Quick Start---### 📊 Key Findings



### Prerequisites



- Python 3.10+## 📊 Detailed Results Summary---

- Virtual environment (recommended)



### Installation

### Exercise 1: Temporal Pattern Analysis| Metric | Value | Interpretation |

```bash

# Clone repository*How do e-scooter usage patterns vary across time dimensions?*

git clone https://github.com/YOUR_USERNAME/turin-smart-mobility.git

cd turin-smart-mobility|--------|-------|----------------|## 🔬 Mission Statement



# Create virtual environment| Operator | Total Trips | Peak Hour | Peak Day | Weekend Share |

python -m venv .venv

source .venv/bin/activate  # macOS/Linux|----------|-------------|-----------|----------|---------------|| **Integration Index (200m)** | 95.3% | Near-universal PT proximity |

# .venv\Scripts\activate   # Windows

| **BIRD** | 147,823 | 18:00 (8.2%) | Friday | 28.3% |

# Install dependencies

pip install -r requirements.txt| **LIME** | 312,456 | 18:00 (9.1%) | Thursday | 31.2% || **Feeder Rate** | 82.4% | Strong first/last-mile role |Urban mobility is undergoing a fundamental transformation. As cities worldwide grapple with congestion, emissions, and accessibility challenges, shared micro-mobility has emerged as a potential solution—or a new problem.

```

| **VOI** | 89,234 | 17:00 (7.8%) | Friday | 26.8% |

### Running the Pipeline

| **Peak Hour Concentration** | 38.5% | Clear commuting patterns |

```bash

# Run complete pipeline**Statistical Test**: Kruskal-Wallis H = 12,456.7, p < 0.001

python run_pipeline.py

| **Market Size** | €8.31M/year | Sustainable business model |**This project aims to:**

# Run specific exercises

python run_pipeline.py --stages 1 2 3---



# Run from a specific stage| **Probability of Loss** | 0.52% | Low financial risk |

python run_pipeline.py --from-stage 3

### Exercise 2: Origin-Destination Matrix

# Only visualizations (skip analysis)

python run_pipeline.py --viz-only*What are the primary mobility corridors in Turin?*1. **Quantify** the spatial relationship between e-scooter usage and public transport infrastructure



# Skip visualizations

python run_pipeline.py --no-viz

```| Rank | Corridor | Daily Trips | Share |**Conclusion**: E-scooters predominantly function as **first/last-mile connectors** rather than direct competitors to public transport.2. **Identify** temporal patterns that reveal user behavior (commuting vs. leisure)



---|------|----------|-------------|-------|



## 📊 Data Sources| 1 | Porta Nuova ↔ Centro | 2,847 | 12.3% |3. **Map** origin-destination flows to understand city-wide mobility demand



| Source | Records | Period | Format || 2 | San Salvario ↔ Politecnico | 1,923 | 8.4% |

|--------|---------|--------|--------|

| **LIME** | 312,000+ | Jan 2024 - Nov 2025 | CSV || 3 | Crocetta ↔ Centro | 1,456 | 6.3% |---4. **Assess** route efficiency to distinguish functional trips from exploratory rides

| **VOI** | 180,000+ | Jan 2024 - Oct 2025 | XLSX (monthly) |

| **BIRD** | 58,000+ | 2024 - Nov 2025 | CSV || 4 | Lingotto ↔ Porta Nuova | 1,234 | 5.4% |

| **GTFS** | 1,500+ stops | Current | Standard GTFS |

| **Zones** | 94 zones | Current | Shapefile || 5 | Aurora ↔ Centro | 1,087 | 4.7% |5. **Inform** policy recommendations for sustainable multi-modal integration



**Total**: ~549,513 trips after cleaning



---**Geographic Concentration**: 60% of trips occur within 5 central zones## 🔬 Research Framework



## 🎨 Key Visualizations



### Exercise 1: Temporal Patterns------

- Hourly trip distribution by operator

- Day-of-week heatmaps

- Monthly trend analysis

### Exercise 3: Public Transport Integration### The 5 Research Questions (Exercises)

### Exercise 2: OD Matrix

- Zone-to-zone flow heatmaps*Do e-scooters complement or compete with public transport?*

- Mobility corridor identification

- Gini concentration analysis## 🏗️ Project Structure: The 5 Pillars



### Exercise 3: PT Integration| Buffer Distance | Integration Index | Feeder Rate |

- Buffer sensitivity analysis (50m, 100m, 200m)

- Integration index choropleth|-----------------|------------------|-------------|```

- Peak vs off-peak comparison

| **50m** | 78.4% | 56.2% |

### Exercise 4: Parking Duration

- Kaplan-Meier survival curves| **100m** | 89.2% | 67.8% |┌─────────────────────────────────────────────────────────────────────────────────┐```

- Weibull distribution fitting

- Abandoned vehicle detection| **200m** | 95.3% | 82.4% |



### Exercise 5: Economics│                         TURIN SMART MOBILITY PROJECT                            │┌─────────────────────────────────────────────────────────────────────────────────┐

- Monte Carlo profit simulation

- Sensitivity tornado charts**Conclusion**: Strong evidence of complementary relationship

- Revenue by zone analysis

│                              5 Research Questions                                ││                         TURIN SMART MOBILITY PROJECT                            │

---

---

## 📈 Statistical Methods

├─────────────────────────────────────────────────────────────────────────────────┤│                              Analysis Framework                                  │

| Exercise | Methods |

|----------|---------|### Exercise 4: Parking Duration Analysis

| **1** | Kruskal-Wallis H-test, Chi-square, Bootstrap CI |

| **2** | Cramér's V, Gini coefficient, Chi-square |*How efficiently is the fleet utilized?*│                                                                                  │├─────────────────────────────────────────────────────────────────────────────────┤

| **3** | Buffer analysis, Temporal segmentation |

| **4** | Weibull survival, Kaplan-Meier, Log-rank test |

| **5** | Monte Carlo (10,000 iterations), VaR analysis |

| Operator | Median (h) | Mean (h) | Abandonment (>48h) |│   EX.1              EX.2              EX.3              EX.4          EX.5      ││                                                                                  │

---

|----------|------------|----------|-------------------|

## 🛠 Technology Stack

| **BIRD** | 6.0 | 17.9 | 2.0% |│  ─────────        ─────────        ─────────        ─────────      ─────────   ││   PILLAR 1          PILLAR 2          PILLAR 3          PILLAR 4    PILLAR 5   │

| Category | Technologies |

|----------|--------------|| **LIME** | 3.1 | 9.9 | 0.6% |

| **Core** | Python, Pandas, NumPy |

| **Spatial** | GeoPandas, Shapely, PyProj || **VOI** | 11.6 | 37.5 | 8.0% |│  │TEMPORAL│       │ O-D     │      │INTEGRA-│       │PARKING │     │ECONOMIC│  ││  ───────────       ───────────       ───────────       ─────────   ─────────   │

| **Statistics** | SciPy, Statsmodels |

| **Visualization** | Matplotlib, Seaborn |

| **Maps** | Contextily (basemaps) |

**Statistical Test**: Kruskal-Wallis H = 95,913.47, p < 0.001│  │PATTERNS│  ───▶ │ MATRIX  │ ───▶ │  TION  │  ───▶ │DURATION│ ───▶│ANALYSIS│  │ MODEL  │  │

---



## 📚 References

---│  │        │       │         │      │        │       │        │     │        │  ││  │ CLEANING │ ───▶ │ & FLOWS  │ ───▶ │ ANALYSIS │ ───▶ │ANALYSIS│  │ MODEL  │  │

1. NACTO (2019). *Guidelines for Regulating Shared Micromobility*

2. ITF (2020). *Safe Micromobility*

3. EU Standard EN13816 - Buffer analysis methodology

### Exercise 5: Economic Analysis│  ─────────        ─────────        ─────────        ─────────      ─────────   ││  │          │      │          │      │          │      │        │  │        │  │

---

*What is the financial viability of the market?*

## 👥 Authors

│      ✅               ✅               ✅               ✅             ✅       ││  ───────────       ───────────       ───────────       ─────────   ─────────   │

**Ali Vaezi** — Politecnico di Torino, Transport Engineering

| Operator | Revenue (€) | Net Profit (€) | Margin |

---

|----------|-------------|----------------|--------|│  COMPLETED        COMPLETED        COMPLETED        COMPLETED      COMPLETED   ││       ✅                ✅                ✅              🔜          🔜        │

## 📄 License

| **BIRD** | 3,224,567 | 1,898,593 | 58.9% |

This project is part of academic coursework at Politecnico di Torino.  

For academic use only. Contact author for permissions.| **LIME** | 4,254,890 | 2,208,597 | 51.9% |│                                                                                  ││   COMPLETED         COMPLETED         COMPLETED        PLANNED     PLANNED     │



---| **VOI** | 837,654 | 423,395 | 50.5% |



<div align="center">| **Total** | **8,317,111** | **4,530,585** | **54.5%** |└─────────────────────────────────────────────────────────────────────────────────┘│                                                                                  │



**Turin Smart Mobility Project** • December 2025



*Powered by Python & GeoPandas***Monte Carlo Risk**: Mean profit €4.92M, P(loss) = 0.52%```└─────────────────────────────────────────────────────────────────────────────────┘



</div>


---```



## 📁 Repository Structure---



```---

turin-smart-mobility/

│## 📊 Detailed Results Summary

├── 📄 README.md                      # This file

├── 📄 ARCHITECTURE.md                # Technical documentation### 📊 Pillar 1: Big Data Cleaning & Harmonization

├── 📄 requirements.txt               # Python dependencies

├── 📄 run_pipeline.py                # Master pipeline controller### Exercise 1: Temporal Pattern Analysis**Status:** ✅ Completed | **Script:** `01_preprocessing.py`

│

├── 📂 src/*How do e-scooter usage patterns vary across time dimensions?*

│   ├── analysis/                     # Statistical analysis modules

│   │   ├── 01_temporal_analysis.py**Challenge:** Three operators (LIME, VOI, BIRD) with different data formats, schemas, and quality issues.

│   │   ├── 02_od_matrix_analysis.py

│   │   ├── 03_integration_analysis.py| Operator | Total Trips | Peak Hour | Peak Day | Weekend Share |

│   │   ├── 04_parking_analysis.py

│   │   └── 05_economic_analysis.py|----------|-------------|-----------|----------|---------------|| Operator | Raw Format | Records | Key Challenges |

│   │

│   ├── utils/                        # Utility modules (spatial, general)

│   │   ├── spatial_utils.py          # CRS, zone handling, spatial helpers

│   │   └── __init__.py               # Utils package init
│   ├── visualization/                # Visualization modules
│   │   ├── 00_data_cleaning.py       # Data cleaning waterfall & bad data charts
│   │   ├── 01_temporal_statistics.py
│   │   ├── 01_temporal_dashboard.py
│   │   ├── 02_od_statistics.py
│   │   ├── 02_od_spatial_flows.py
│   │   ├── 03_integration_statistics.py
│   │   ├── 03_integration_maps.py
│   │   ├── 04_parking_survival.py
│   │   ├── 04_parking_maps.py
│   │   ├── 05_economic_sensitivity.py
│   │   └── 05_economic_maps.py
│   └── data/                         # Data processing
│       └── 01_data_cleaning.py

├── 📂 data/                          # Data directory (git-ignored)| Exercise | Topic | Status |### 🎯 Central Research Question

│   ├── raw/                          # Original operator data

│   └── processed/                    # Cleaned datasets|----------|-------|--------|

│

└── 📂 outputs/| **1** | Temporal Pattern Analysis | ✅ Completed |> **"Are e-scooters competitors or allies to public transport in Turin?"**

    ├── figures/                      # Generated visualizations

    │   ├── exercise1/                # Descriptive analysis plots, data cleaning waterfall, bad data breakdown charts
    │   ├── exercise2/                # O-D flow maps
    │   ├── exercise3/                # Integration analysis figures
    │   ├── exercise4/                # Parking analysis figures
    │   ├── exercise5/                # Economic analysis figures
    └── reports/                      # Analysis reports

```| **5** | Economic & Sensitivity Analysis | ✅ Completed |Our analysis reveals that e-scooters predominantly function as **first/last-mile connectors** rather than direct competitors, with **95%+ of trips originating within 200m of public transport stops** during peak commuting hours.



---



## 🚀 Quick Start---### 📊 Key Findings



### Prerequisites



- Python 3.10+## 📊 Detailed Results Summary---

- Virtual environment (recommended)



### Installation

### Exercise 1: Temporal Pattern Analysis| Metric | Value | Interpretation |

```bash

# Clone repository*How do e-scooter usage patterns vary across time dimensions?*

git clone https://github.com/YOUR_USERNAME/turin-smart-mobility.git

cd turin-smart-mobility|--------|-------|----------------|## 🔬 Mission Statement



# Create virtual environment| Operator | Total Trips | Peak Hour | Peak Day | Weekend Share |

python -m venv .venv

source .venv/bin/activate  # macOS/Linux|----------|-------------|-----------|----------|---------------|| **Integration Index (200m)** | 95.3% | Near-universal PT proximity |

# .venv\Scripts\activate   # Windows

| **BIRD** | 147,823 | 18:00 (8.2%) | Friday | 28.3% |

# Install dependencies

pip install -r requirements.txt| **LIME** | 312,456 | 18:00 (9.1%) | Thursday | 31.2% || **Feeder Rate** | 82.4% | Strong first/last-mile role |Urban mobility is undergoing a fundamental transformation. As cities worldwide grapple with congestion, emissions, and accessibility challenges, shared micro-mobility has emerged as a potential solution—or a new problem.

```

| **VOI** | 89,234 | 17:00 (7.8%) | Friday | 26.8% |

### Running the Pipeline

| **Peak Hour Concentration** | 38.5% | Clear commuting patterns |

```bash

# Run complete pipeline**Statistical Test**: Kruskal-Wallis H = 12,456.7, p < 0.001

python run_pipeline.py

| **Market Size** | €8.31M/year | Sustainable business model |**This project aims to:**

# Run specific exercises

python run_pipeline.py --stages 1 2 3---



# Run from a specific stage| **Probability of Loss** | 0.52% | Low financial risk |

python run_pipeline.py --from-stage 3

### Exercise 2: Origin-Destination Matrix

# Only visualizations (skip analysis)

python run_pipeline.py --viz-only*What are the primary mobility corridors in Turin?*1. **Quantify** the spatial relationship between e-scooter usage and public transport infrastructure



# Skip visualizations

python run_pipeline.py --no-viz

```| Rank | Corridor | Daily Trips | Share |**Conclusion**: E-scooters predominantly function as **first/last-mile connectors** rather than direct competitors to public transport.2. **Identify** temporal patterns that reveal user behavior (commuting vs. leisure)



---|------|----------|-------------|-------|



## 📊 Data Sources| 1 | Porta Nuova ↔ Centro | 2,847 | 12.3% |3. **Map** origin-destination flows to understand city-wide mobility demand



| Source | Records | Period | Format || 2 | San Salvario ↔ Politecnico | 1,923 | 8.4% |

|--------|---------|--------|--------|

| **LIME** | 312,000+ | Jan 2024 - Nov 2025 | CSV || 3 | Crocetta ↔ Centro | 1,456 | 6.3% |---4. **Assess** route efficiency to distinguish functional trips from exploratory rides

| **VOI** | 180,000+ | Jan 2024 - Oct 2025 | XLSX (monthly) |

| **BIRD** | 58,000+ | 2024 - Nov 2025 | CSV || 4 | Lingotto ↔ Porta Nuova | 1,234 | 5.4% |

| **GTFS** | 1,500+ stops | Current | Standard GTFS |

| **Zones** | 94 zones | Current | Shapefile || 5 | Aurora ↔ Centro | 1,087 | 4.7% |5. **Inform** policy recommendations for sustainable multi-modal integration



**Total**: ~549,513 trips after cleaning



---**Geographic Concentration**: 60% of trips occur within 5 central zones## 🔬 Research Framework



## 🎨 Key Visualizations



### Exercise 1: Temporal Patterns------

- Hourly trip distribution by operator

- Day-of-week heatmaps

- Monthly trend analysis

### Exercise 3: Public Transport Integration### The 5 Research Questions (Exercises)

### Exercise 2: OD Matrix

- Zone-to-zone flow heatmaps*Do e-scooters complement or compete with public transport?*

- Mobility corridor identification

- Gini concentration analysis## 🏗️ Project Structure: The 5 Pillars



### Exercise 3: PT Integration| Buffer Distance | Integration Index | Feeder Rate |

- Buffer sensitivity analysis (50m, 100m, 200m)

- Integration index choropleth|-----------------|------------------|-------------|```

- Peak vs off-peak comparison

| **50m** | 78.4% | 56.2% |

### Exercise 4: Parking Duration

- Kaplan-Meier survival curves| **100m** | 89.2% | 67.8% |┌─────────────────────────────────────────────────────────────────────────────────┐```

- Weibull distribution fitting

- Abandoned vehicle detection| **200m** | 95.3% | 82.4% |



### Exercise 5: Economics│                         TURIN SMART MOBILITY PROJECT                            │┌─────────────────────────────────────────────────────────────────────────────────┐

- Monte Carlo profit simulation

- Sensitivity tornado charts**Conclusion**: Strong evidence of complementary relationship

- Revenue by zone analysis

│                              5 Research Questions                                ││                         TURIN SMART MOBILITY PROJECT                            │

---

---

## 📈 Statistical Methods

├─────────────────────────────────────────────────────────────────────────────────┤│                              Analysis Framework                                  │

| Exercise | Methods |

|----------|---------|### Exercise 4: Parking Duration Analysis

| **1** | Kruskal-Wallis H-test, Chi-square, Bootstrap CI |

| **2** | Cramér's V, Gini coefficient, Chi-square |*How efficiently is the fleet utilized?*│                                                                                  │├─────────────────────────────────────────────────────────────────────────────────┤

| **3** | Buffer analysis, Temporal segmentation |

| **4** | Weibull survival, Kaplan-Meier, Log-rank test |

| **5** | Monte Carlo (10,000 iterations), VaR analysis |

| Operator | Median (h) | Mean (h) | Abandonment (>48h) |│   EX.1              EX.2              EX.3              EX.4          EX.5      ││                                                                                  │

---

|----------|------------|----------|-------------------|

## 🛠 Technology Stack

| **BIRD** | 6.0 | 17.9 | 2.0% |│  ─────────        ─────────        ─────────        ─────────      ─────────   ││   PILLAR 1          PILLAR 2          PILLAR 3          PILLAR 4    PILLAR 5   │

| Category | Technologies |

|----------|--------------|| **LIME** | 3.1 | 9.9 | 0.6% |

| **Core** | Python, Pandas, NumPy |

| **Spatial** | GeoPandas, Shapely, PyProj || **VOI** | 11.6 | 37.5 | 8.0% |│  │TEMPORAL│       │ O-D     │      │INTEGRA-│       │PARKING │     │ECONOMIC│  ││  ───────────       ───────────       ───────────       ─────────   ─────────   │

| **Statistics** | SciPy, Statsmodels |

| **Visualization** | Matplotlib, Seaborn |

| **Maps** | Contextily (basemaps) |

**Statistical Test**: Kruskal-Wallis H = 95,913.47, p < 0.001│  │PATTERNS│  ───▶ │ MATRIX  │ ───▶ │  TION  │  ───▶ │DURATION│ ───▶│ANALYSIS│  │ MODEL  │  │

---



## 📚 References

---│  │        │       │         │      │        │       │        │     │        │  ││  │ CLEANING │ ───▶ │ & FLOWS  │ ───▶ │ ANALYSIS │ ───▶ │ANALYSIS│  │ MODEL  │  │

1. NACTO (2019). *Guidelines for Regulating Shared Micromobility*

2. ITF (2020). *Safe Micromobility*

3. EU Standard EN13816 - Buffer analysis methodology

### Exercise 5: Economic Analysis│  ─────────        ─────────        ────

