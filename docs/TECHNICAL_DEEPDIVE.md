# 🔬 Deep-Dive Technical Documentation

## Turin Shared E-Scooter Micromobility Analysis System

> **Version**: 1.0.0 | **Last Updated**: December 2025  
> **Author**: Ali Vaezi | **Affiliation**: Transport Research  
> **License**: Academic Research Use

---

## Table of Contents

1. [🧬 Data Pipeline Architecture](#-1-data-pipeline-architecture)
2. [⏱️ Temporal & Spatial Dynamics](#️-2-temporal--spatial-dynamics-exercise-1--2)
3. [🚌 Multimodal Integration Analysis](#-3-multimodal-integration-analysis-exercise-3)
4. [🅿️ Fleet Survival Analysis](#️-4-fleet-survival-analysis-exercise-4)
5. [💶 Micro-Economic Modeling](#-5-micro-economic-modeling-exercise-5)

---

## 🧬 1. Data Pipeline Architecture

### 1.1 Dataset Overview

This analysis processes **2,548,650 e-scooter trips** across three major operators in Turin, Italy, spanning the 2024-2025 observation period.

| Operator | Raw Records | Cleaned Records | Data Format | Temporal Coverage |
|----------|-------------|-----------------|-------------|-------------------|
| **LIME** | 1,450,000+ | 1,421,374 | CSV (single file) | Jan 2024 – Nov 2025 |
| **BIRD** | 870,000+ | 852,751 | CSV (2 annual files) | Jan 2024 – Nov 2025 |
| **VOI** | 285,000+ | 274,525 | XLSX (18 monthly files) | Jan 2024 – May 2025 |
| **Total** | ~2,600,000 | **2,548,650** | Mixed | 23 months |

> **💡 Key Insight**: The 2% data loss during cleaning primarily stems from coordinate validation failures and timestamp parsing errors.

---

### 1.2 Data Validation Pipeline

The preprocessing pipeline implements a **5-stage validation cascade** designed to ensure spatial and temporal integrity:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DATA VALIDATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   RAW DATA                                                                      │
│      │                                                                          │
│      ▼                                                                          │
│   ┌────────────────────────────────────────────────────────────┐               │
│   │  STAGE 1: FORMAT NORMALIZATION                             │               │
│   │  • LIME: UTF-8 CSV parsing with low_memory=False           │               │
│   │  • BIRD: Multi-file concatenation (2024 + 2025)            │               │
│   │  • VOI: Excel parsing with 14-digit timestamp handling     │               │
│   └────────────────────────────────────────────────────────────┘               │
│      │                                                                          │
│      ▼                                                                          │
│   ┌────────────────────────────────────────────────────────────┐               │
│   │  STAGE 2: DATETIME PARSING & VOI ANOMALY CORRECTION        │               │
│   │  • Parse ISO 8601 timestamps (LIME/BIRD)                   │               │
│   │  • Handle VOI's YYYYMMDDHHmmss integer format              │               │
│   │  • Reject future dates (>= 2026-01-01)                     │               │
│   └────────────────────────────────────────────────────────────┘               │
│      │                                                                          │
│      ▼                                                                          │
│   ┌────────────────────────────────────────────────────────────┐               │
│   │  STAGE 3: COORDINATE VALIDATION                            │               │
│   │  • Turin bounding box: [44.97°N–45.14°N, 7.57°E–7.77°E]    │               │
│   │  • Reject null/NaN coordinates                             │               │
│   │  • Flag suspicious precision (<4 decimal places)           │               │
│   └────────────────────────────────────────────────────────────┘               │
│      │                                                                          │
│      ▼                                                                          │
│   ┌────────────────────────────────────────────────────────────┐               │
│   │  STAGE 4: DURATION SANITY CHECKS                           │               │
│   │  • Reject trips < 60 seconds (unlock errors)               │               │
│   │  • Reject trips > 24 hours (data errors)                   │               │
│   │  • Flag trips > 4 hours for review                         │               │
│   └────────────────────────────────────────────────────────────┘               │
│      │                                                                          │
│      ▼                                                                          │
│   ┌────────────────────────────────────────────────────────────┐               │
│   │  STAGE 5: SPATIAL ZONE ASSIGNMENT                          │               │
│   │  • Spatial join with 94 Zone Statistiche polygons          │               │
│   │  • CRS transformation: WGS84 → EPSG:3003 → EPSG:32632      │               │
│   │  • 89 zones contain trip data (5 zones are parks/rivers)   │               │
│   └────────────────────────────────────────────────────────────┘               │
│      │                                                                          │
│      ▼                                                                          │
│   CLEANED DATA (2,548,650 validated trips)                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 VOI 2024 Date Anomaly Handling

> **⚠️ Critical Data Issue**: VOI data uses a non-standard timestamp format requiring custom parsing logic.

VOI's raw Excel files encode timestamps as **14-digit integers** in the format `YYYYMMDDHHmmss`:

```python
# Example: 20240630215349 → 2024-06-30 21:53:49
# VOI dates are in format YYYYMMDDHHmmss (14-digit integer like 20240630215349)

def parse_voi_timestamp(value):
    """
    Convert VOI's 14-digit integer timestamp to datetime.
    
    Input:  20240630215349 (int64)
    Output: 2024-06-30 21:53:49 (datetime64)
    """
    s = str(int(value))
    if len(s) != 14:
        return pd.NaT
    
    return pd.Timestamp(
        year=int(s[0:4]),
        month=int(s[4:6]),
        day=int(s[6:8]),
        hour=int(s[8:10]),
        minute=int(s[10:12]),
        second=int(s[12:14])
    )
```

> **📁 Note**: VOI file `DATINOLEGGI_202503.xlsx` (March 2025) was not provided and is skipped during processing.

---

### 1.4 Coordinate Cleaning Protocol

| Validation Rule | Threshold | Action |
|-----------------|-----------|--------|
| Latitude bounds | 44.97°N – 45.14°N | Reject if outside |
| Longitude bounds | 7.57°E – 7.77°E | Reject if outside |
| Null coordinates | `NaN` or `None` | Reject row |
| Precision check | < 4 decimal places | Flag for review |
| Zero coordinates | `(0.0, 0.0)` | Reject (GPS error) |

---

## ⏱️ 2. Temporal & Spatial Dynamics (Exercise 1 & 2)

### 2.1 Statistical Testing Framework

The temporal analysis employs **non-parametric statistical tests** to compare demand patterns across operators, given the non-normal distribution of trip counts.

> **📊 Key Algorithm: Kruskal-Wallis H-Test**
>
> The Kruskal-Wallis test is a rank-based non-parametric alternative to one-way ANOVA, used when the assumption of normality is violated.
>
> **Null Hypothesis (H₀)**: The hourly trip distributions are identical across all operators.
>
> $$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$
>
> Where:
> - $N$ = total number of observations
> - $k$ = number of groups (operators)
> - $R_i$ = sum of ranks in group $i$
> - $n_i$ = sample size of group $i$

#### Statistical Results: Operator Comparison

| Test | Statistic | p-value | Effect Size | Interpretation |
|------|-----------|---------|-------------|----------------|
| Kruskal-Wallis H | 95,913.47 | < 0.001*** | η² = 0.039 | Significant difference |
| Mann-Whitney U (LIME vs BIRD) | 2.1×10⁸ | < 0.001*** | r = 0.31 | Medium effect |
| Mann-Whitney U (LIME vs VOI) | 1.8×10⁸ | < 0.001*** | r = 0.42 | Medium effect |
| Mann-Whitney U (BIRD vs VOI) | 9.4×10⁷ | < 0.001*** | r = 0.28 | Small-medium effect |

> **📈 Key Finding**: The evening peak at **18:00** accounts for **7.2%** of daily demand, with the 17:00–20:00 window capturing **20.5%** of all trips—significantly exceeding the morning rush (9.9%).

---

### 2.2 Post-Hoc Analysis: Eta-Squared Effect Size

```python
def calculate_eta_squared(h_statistic: float, n_total: int, k_groups: int) -> float:
    """
    Calculate eta-squared (η²) effect size from Kruskal-Wallis H statistic.
    
    η² = (H - k + 1) / (N - k)
    
    Interpretation:
        η² < 0.01: negligible
        0.01 ≤ η² < 0.06: small
        0.06 ≤ η² < 0.14: medium
        η² ≥ 0.14: large
    """
    return (h_statistic - k_groups + 1) / (n_total - k_groups)
```

---

### 2.3 Origin-Destination Matrix Methodology

#### 2.3.1 Matrix Construction

The O-D matrix $\mathbf{T}$ is a $90 \times 89$ matrix where entry $T_{ij}$ represents trips from origin zone $i$ to destination zone $j$.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Matrix Dimensions | 90 × 89 | 89 zones + TOTAL row |
| Total OD Pairs | 8,010 | Theoretical maximum |
| Non-Zero Pairs | 6,613 | Active corridors |
| Matrix Sparsity | 17.4% | Relatively dense |
| Total Trips | 2,509,948 | All-day period |

---

#### 2.3.2 Hierarchical Clustering for Zone Grouping

> **🧮 Key Algorithm: Ward's Hierarchical Clustering**
>
> Zones are clustered based on their **origin-destination profiles** using Ward's minimum variance method, which minimizes within-cluster variance at each merge step.
>
> **Distance Metric**: Euclidean distance on row-normalized O-D profiles
>
> **Linkage**: Ward's method minimizes:
> $$\Delta(A,B) = \frac{n_A n_B}{n_A + n_B} \| \bar{x}_A - \bar{x}_B \|^2$$

```python
from scipy.cluster.hierarchy import linkage, fcluster

def perform_hierarchical_clustering(od_matrix, n_clusters=5):
    """
    Perform hierarchical clustering on zones based on O-D patterns.
    Groups zones with similar origin-destination profiles.
    """
    # Remove TOTAL row/column
    matrix_clean = od_matrix.drop('TOTAL', axis=0, errors='ignore')
    matrix_clean = matrix_clean.drop('TOTAL', axis=1, errors='ignore')
    
    # Normalize rows (origin profiles)
    row_sums = matrix_clean.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix_norm = matrix_clean.div(row_sums, axis=0)
    
    # Compute distance matrix and perform clustering
    Z = linkage(matrix_norm.values, method='ward')
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    return clusters, Z, matrix_clean.index.tolist()
```

---

#### 2.3.3 Gravity Model Application

The **doubly-constrained gravity model** describes the relationship between trip flows and inter-zonal distance:

$$T_{ij} = K \cdot P_i \cdot A_j \cdot e^{-\beta \cdot d_{ij}}$$

Where:
- $T_{ij}$ = trips from zone $i$ to zone $j$
- $P_i$ = production (trips originating) at zone $i$
- $A_j$ = attraction (trips destinating) at zone $j$
- $d_{ij}$ = distance between zone centroids
- $\beta$ = distance decay parameter
- $K$ = balancing factor

| Parameter | Estimated Value | 95% CI |
|-----------|-----------------|--------|
| β (decay) | 1.50 | [1.42, 1.58] |
| R² | 0.72 | - |

---

#### 2.3.4 Flow Concentration Metrics

| Metric | Formula | Value | Interpretation |
|--------|---------|-------|----------------|
| **Gini Coefficient** | $G = \frac{\sum_{i,j}\|T_i - T_j\|}{2n^2\bar{T}}$ | 0.774 | High inequality |
| **Shannon Entropy** | $H = -\sum p_{ij} \log(p_{ij})$ | 0.858 | Moderate diversity |
| **Flow Asymmetry** | $A = \frac{\sum_{i<j}\|T_{ij} - T_{ji}\|}{\sum T_{ij}}$ | 0.231 | Moderate imbalance |
| **Top 10 Corridors Share** | - | 3.75% | Concentrated demand |

---

#### 2.3.5 Top Trip Generators and Attractors

| Rank | Zone | Name | Role | Daily Trips |
|------|------|------|------|-------------|
| 1 | 04 | Piazza San Carlo – Piazza Carlo Felice | Generator/Attractor | 126,229 |
| 2 | 08 | Porta Susa Station | Transit Hub | 99,777 |
| 3 | 01 | Municipio (City Hall) | Civic Center | 94,787 |
| 4 | 33 | San Salvario | Residential | 92,884 |
| 5 | 03 | Piazza Castello | Mixed-Use | 88,992 |

---

## 🚌 3. Multimodal Integration Analysis (Exercise 3)

### 3.1 Vectorized Buffer Analysis

Public transport integration is assessed using **buffer-based spatial proximity analysis**. Trips are classified as "integrated" if their origin or destination falls within a defined catchment distance of transit stops.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-BUFFER SENSITIVITY FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   PT STOP                                                                       │
│      ●                                                                          │
│     /│\         ┌──────────────────────────────────────────┐                   │
│    / │ \        │  BUFFER THRESHOLDS                       │                   │
│   /  │  \       │                                          │                   │
│  ╱   │   ╲      │  🔴 50m  - Conservative (walking 1 min)  │                   │
│ ╱    │    ╲     │  🟡 100m - Standard (literature norm)    │                   │
│╱     │     ╲    │  🟢 200m - Liberal (extended catchment)  │                   │
│      │      ────┴──────────────────────────────────────────┘                   │
│   50m│100m  200m                                                               │
│                                                                                 │
│   VECTORIZED SPATIAL JOIN:                                                      │
│   • Load 2,500+ PT stops from GTFS (stops.txt)                                 │
│   • Create buffered polygons using shapely.buffer()                            │
│   • Perform spatial join with STRtree for O(log n) performance                 │
│   • Classify trips: origin_near_pt, dest_near_pt, both_near_pt                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 Integration Metrics Definitions

> **📐 Key Metrics**
>
> **Integration Index (I)**: Probability that either trip endpoint is near PT
> $$I = \frac{N_{\text{origin} \leq d} + N_{\text{dest} \leq d}}{2 \cdot N_{\text{total}}}$$
>
> **Feeder Percentage (F)**: Probability that **both** endpoints are near PT (strict criterion)
> $$F = \frac{N_{\text{both} \leq d}}{N_{\text{total}}}$$

---

### 3.3 Buffer Sensitivity Results

| Operator | Buffer (m) | Integration Index | Feeder % | Origin Near PT | Destination Near PT |
|----------|------------|-------------------|----------|----------------|---------------------|
| LIME | 50 | 55.7% | 34.1% | 484,016 | 472,541 |
| LIME | 100 | 86.6% | 63.9% | 908,392 | 905,342 |
| LIME | 200 | 99.8% | 95.9% | 1,362,447 | 1,360,981 |
| VOI | 50 | 52.2% | 31.8% | 85,677 | 82,321 |
| VOI | 100 | 86.8% | 63.1% | 169,968 | 169,624 |
| VOI | 200 | 99.8% | 94.4% | 254,374 | 253,818 |
| BIRD | 50 | 53.8% | 34.2% | 291,880 | 262,622 |
| BIRD | 100 | 84.7% | 63.3% | 540,147 | 520,139 |
| BIRD | 200 | 99.5% | 95.2% | 811,551 | 806,631 |

> **💡 Key Finding**: At the 100-meter threshold (standard in literature), **85–87%** of e-scooter trips demonstrate potential PT integration, with **63–64%** classified as strict "feeder" trips.

---

### 3.4 Tortuosity Index Calculation

The **Tortuosity Index** measures route efficiency by comparing actual travel distance to the Euclidean (straight-line) distance.

> **📐 Key Algorithm: Tortuosity Index**
>
> $$\tau = \frac{D_{\text{actual}}}{D_{\text{euclidean}}}$$
>
> Where:
> - $D_{\text{actual}}$ = cumulative length of GPS trajectory (sum of segment distances)
> - $D_{\text{euclidean}}$ = straight-line distance from origin to destination
>
> **Interpretation:**
> - $\tau = 1.0$: Perfectly direct route
> - $\tau = 1.3$: 30% longer than straight-line (typical urban value)
> - $\tau > 2.0$: Highly circuitous (leisure/exploration behavior)

```python
def calculate_tortuosity(linestring, start_coords, end_coords):
    """
    Calculate tortuosity index for a GPS trajectory.
    
    Parameters:
    - linestring: Shapely LineString of GPS points
    - start_coords: (lon, lat) of trip origin
    - end_coords: (lon, lat) of trip destination
    
    Returns:
    - tortuosity_index: float >= 1.0
    """
    from shapely.geometry import Point
    from pyproj import Geod
    
    geod = Geod(ellps='WGS84')
    
    # Actual distance: sum of all segments
    actual_distance = linestring.length  # In projected CRS (meters)
    
    # Euclidean distance: straight line
    _, _, euclidean_distance = geod.inv(
        start_coords[0], start_coords[1],
        end_coords[0], end_coords[1]
    )
    
    if euclidean_distance < 10:  # Minimum 10m to avoid division errors
        return np.nan
    
    return actual_distance / euclidean_distance
```

| Tortuosity Range | Interpretation | Typical Behavior |
|------------------|----------------|------------------|
| 1.00 – 1.15 | Very direct | Commute/purpose-driven |
| 1.15 – 1.30 | Normal urban | Standard navigation |
| 1.30 – 1.50 | Moderate detour | Traffic avoidance |
| 1.50 – 2.00 | Circuitous | Exploration/leisure |
| > 2.00 | Highly indirect | Joy-riding/errors |

**System Average Tortuosity**: **1.31** (indicating normal urban navigation patterns)

---

## 🅿️ 4. Fleet Survival Analysis (Exercise 4)

### 4.1 Survival Analysis Framework

Survival analysis models the time-to-event distribution for vehicle rental after parking. The "event" is a rental, and "survival" means the vehicle remains unrented.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SURVIVAL ANALYSIS FRAMEWORK                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   SURVIVAL FUNCTION S(t):                                                       │
│   "What is the probability that a parked vehicle remains unrented at time t?"   │
│                                                                                 │
│   HAZARD FUNCTION h(t):                                                         │
│   "Given survival to time t, what is the instantaneous rental probability?"     │
│                                                                                 │
│   RELATIONSHIP:                                                                 │
│   S(t) = exp(-∫₀ᵗ h(u) du)                                                      │
│                                                                                 │
│   ┌───────────────────────────────────────────────────────────────────┐        │
│   │                METHODOLOGY PIPELINE                               │        │
│   │                                                                   │        │
│   │   1. IDLE TIME CALCULATION                                        │        │
│   │      Idle = Start_Time(Next Trip) - End_Time(Current Trip)        │        │
│   │      → 2.2M parking events extracted                              │        │
│   │                                                                   │        │
│   │   2. KAPLAN-MEIER ESTIMATION                                      │        │
│   │      Non-parametric survival curves per operator                  │        │
│   │      95% CI via Greenwood's formula                               │        │
│   │                                                                   │        │
│   │   3. WEIBULL PARAMETRIC FIT                                       │        │
│   │      S(t) = exp(-(t/λ)^k)                                         │        │
│   │      MLE estimation of shape (k) and scale (λ)                    │        │
│   │                                                                   │        │
│   │   4. LOG-RANK TEST                                                │        │
│   │      χ² comparison of survival curves between operators           │        │
│   │                                                                   │        │
│   │   5. GHOST VEHICLE DETECTION                                      │        │
│   │      Flag parking > 120 hours as "abandoned"                      │        │
│   └───────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### 4.2 Kaplan-Meier Estimator

> **📐 Key Algorithm: Kaplan-Meier Survival Estimator**
>
> The Kaplan-Meier estimator is a non-parametric maximum likelihood estimate of the survival function:
>
> $$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$
>
> Where:
> - $t_i$ = distinct event times
> - $d_i$ = number of events (rentals) at time $t_i$
> - $n_i$ = number of vehicles "at risk" just before $t_i$
>
> **95% Confidence Interval** (Greenwood's formula):
> $$\text{Var}[\hat{S}(t)] = \hat{S}(t)^2 \sum_{t_i \leq t} \frac{d_i}{n_i(n_i - d_i)}$$

---

### 4.3 Survival Curve Results

| Time (hours) | BIRD Survival | LIME Survival | VOI Survival | System Average |
|--------------|---------------|---------------|--------------|----------------|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 0.801 | 0.734 | 0.843 | 0.766 |
| 2 | 0.699 | 0.597 | 0.773 | 0.646 |
| 4 | 0.577 | 0.446 | 0.686 | 0.510 |
| 8 | 0.447 | 0.302 | 0.580 | 0.375 |
| 12 | 0.365 | 0.216 | 0.493 | 0.290 |
| 24 | 0.202 | 0.096 | 0.339 | 0.152 |
| 48 | 0.092 | 0.038 | 0.211 | 0.071 |
| 72 | 0.050 | 0.018 | 0.145 | 0.040 |

**Median Survival Times:**
- **LIME**: ~3.5 hours (fastest turnover)
- **BIRD**: ~6.0 hours
- **VOI**: ~10.5 hours
- **System**: ~5.0 hours

---

### 4.4 Weibull Distribution Fitting

> **📐 Key Algorithm: Weibull Distribution**
>
> The Weibull distribution is a flexible parametric survival model:
>
> $$f(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1} e^{-(t/\lambda)^k}$$
>
> $$S(t) = e^{-(t/\lambda)^k}$$
>
> **Parameters:**
> - $k$ (shape): Controls hazard rate behavior
>   - $k < 1$: Decreasing hazard (vehicles parked longer become less likely to rent)
>   - $k = 1$: Constant hazard (exponential distribution)
>   - $k > 1$: Increasing hazard (vehicles become more likely to rent over time)
> - $\lambda$ (scale): Characteristic life (time at which 63.2% have been rented)

```python
from scipy.optimize import minimize

def weibull_negative_log_likelihood(params, data):
    """Negative log-likelihood for Weibull distribution MLE."""
    shape, scale = params
    if shape <= 0 or scale <= 0:
        return 1e10
    n = len(data)
    ll = n * np.log(shape / scale)
    ll += (shape - 1) * np.sum(np.log(data / scale))
    ll -= np.sum((data / scale) ** shape)
    return -ll

def fit_weibull(data):
    """Fit Weibull distribution to duration data using MLE."""
    data = data[data > 0]
    
    # Initial estimates via method of moments
    mean_data = np.mean(data)
    cv = np.std(data) / mean_data
    shape_init = 1 / cv if cv > 0 else 1.0
    scale_init = mean_data
    
    result = minimize(
        weibull_negative_log_likelihood,
        [shape_init, scale_init],
        args=(data,),
        method='L-BFGS-B',
        bounds=[(0.01, 10), (0.01, 1000)]
    )
    
    return {'shape': result.x[0], 'scale': result.x[1]}
```

| Operator | Shape (k) | Scale (λ) | Log-Likelihood | AIC | Interpretation |
|----------|-----------|-----------|----------------|-----|----------------|
| BIRD | 0.615 | 12.0 | -2.98×10⁶ | 5.95×10⁶ | Decreasing hazard |
| LIME | 0.628 | 6.5 | -4.29×10⁶ | 8.58×10⁶ | Decreasing hazard |
| VOI | 0.570 | 22.8 | -8.77×10⁵ | 1.75×10⁶ | Decreasing hazard |
| ALL | 0.593 | 9.1 | -8.23×10⁶ | 1.65×10⁷ | Decreasing hazard |

> **💡 Key Insight**: All operators exhibit $k < 1$ (decreasing hazard), meaning vehicles parked for longer periods become **less likely** to be rented—indicating they are in low-demand locations.

---

### 4.5 Log-Rank Test for Operator Comparison

> **📐 Key Algorithm: Log-Rank Test**
>
> The log-rank test compares survival curves between groups:
>
> $$\chi^2 = \frac{\left(\sum_{i}(O_{1i} - E_{1i})\right)^2}{\sum_{i}V_i}$$
>
> Where:
> - $O_{1i}$ = observed events in group 1 at time $i$
> - $E_{1i}$ = expected events in group 1 at time $i$ (under null hypothesis)
> - $V_i$ = variance of the difference

| Comparison | Chi-Square (χ²) | p-value | Significant? |
|------------|-----------------|---------|--------------|
| BIRD vs LIME | 3.28 × 10⁹ | < 0.001*** | Yes |
| BIRD vs VOI | 3.41 × 10⁸ | < 0.001*** | Yes |
| LIME vs VOI | 1.43 × 10⁹ | < 0.001*** | Yes |

> **📊 Conclusion**: Survival curves differ **significantly** between all operator pairs, reflecting distinct fleet management strategies, vehicle positioning, and customer bases.

---

### 4.6 Ghost Vehicle Detection

**Ghost Vehicles** are scooters that remain parked for extended periods, indicating potential abandonment, vandalism, or operational neglect.

> **⚠️ Ghost Vehicle Threshold: 120 hours (5 days)**

```python
GHOST_THRESHOLD_HOURS = 120  # 5 days = abandoned/ghost vehicle

def detect_ghost_vehicles(df):
    """
    Flag parking events > 120 hours as abandoned/ghost vehicles.
    
    These vehicles require operational intervention (retrieval, maintenance).
    """
    df['is_abandoned'] = df['idle_hours'] > GHOST_THRESHOLD_HOURS
    
    ghost_df = df[df['is_abandoned']][
        ['operator', 'vehicle_id', 'end_datetime', 'end_lat', 'end_lon', 
         'idle_hours', 'is_abandoned']
    ].copy()
    
    return df, ghost_df
```

**Ghost Vehicle Analysis:**
- Ghost events represent **< 2%** of total parking events
- Higher ghost rates in peripheral zones (lower demand)
- Used to inform rebalancing operations

---

## 💶 5. Micro-Economic Modeling (Exercise 5)

### 5.1 Unit Economics Model

The profitability of each trip is computed using an **Activity-Based Costing (ABC)** framework:

> **📐 Unit Economics Formula**
>
> $$\pi_{\text{trip}} = R_{\text{trip}} - VC_{\text{trip}} - \frac{FC}{N_{\text{trips}}}$$
>
> **Revenue Model:**
> $$R_{\text{trip}} = \text{Unlock Fee} + (\text{Minute Rate} \times \text{Duration}_{\text{min}})$$
> $$R_{\text{trip}} = €1.00 + (€0.15 \times t)$$
>
> **Cost Model:**
> $$VC_{\text{trip}} = C_{\text{battery}} + C_{\text{maintenance}} + C_{\text{insurance}} + C_{\text{operations}}$$
> $$VC_{\text{trip}} = €0.05 + €0.35 + €0.40 + €0.40 = €1.20$$
>
> $$FC_{\text{annual}} = C_{\text{permits}} + C_{\text{depreciation}} = €50 + €200 = €250/\text{vehicle}$$

---

### 5.2 Cost Structure Breakdown

| Cost Category | Cost per Trip | Cost Type | Notes |
|---------------|---------------|-----------|-------|
| **Unlock Fee** | +€1.00 | Revenue | Fixed per trip |
| **Time Revenue** | +€0.15/min | Revenue | Variable with duration |
| Battery/Charging | -€0.05 | Variable | Per-trip energy cost |
| Maintenance | -€0.35 | Variable | Wear and repairs |
| Insurance | -€0.40 | Variable | Per-trip liability |
| Operations | -€0.40 | Variable | Rebalancing, support |
| Permits/Licenses | -€50/yr | Fixed | Municipal fees |
| Depreciation | -€200/yr | Fixed | €600 vehicle / 3-year life |

---

### 5.3 System-Wide Economics

| Metric | Value |
|--------|-------|
| **Total Trips** | 2,543,648 |
| **Total Revenue** | €8,300,365 |
| **Total Variable Costs** | €3,052,378 |
| **Total Fixed Costs** | €715,185 |
| **Total Costs** | €3,767,563 |
| **Net Profit** | €4,532,803 |
| **Profit Margin** | 54.6% |
| **Avg Revenue/Trip** | €3.26 |
| **Avg Cost/Trip** | €1.48 |
| **Avg Profit/Trip** | €1.78 |
| **Profitable Trips** | 97.1% |

---

### 5.4 Operator-Level Profitability

| Operator | Trips | Revenue | Total Cost | Net Profit | Profit/Trip | Margin |
|----------|-------|---------|------------|------------|-------------|--------|
| LIME | 1,421,372 | €4,245,099 | €2,037,115 | €2,207,983 | €1.55 | 52.0% |
| BIRD | 852,751 | €3,217,369 | €1,316,081 | €1,901,288 | €2.23 | 59.1% |
| VOI | 269,525 | €837,898 | €414,367 | €423,532 | €1.57 | 50.6% |

> **💡 Key Finding**: BIRD achieves the highest per-trip profit (€2.23) due to longer average trip durations (13.9 min vs. 10.5 min for LIME).

---

### 5.5 Monte Carlo Simulation for Risk Assessment

> **📐 Key Algorithm: Monte Carlo Simulation**
>
> Monte Carlo simulation quantifies profit uncertainty by sampling from stochastic distributions of key inputs:
>
> **Simulation Parameters (10,000 iterations):**
> - Revenue variation: ±15% (uniform distribution)
> - Variable cost variation: ±10% (uniform distribution)
> - Trip volume variation: ±5% (normal distribution)

```python
def monte_carlo_profit_simulation(base_revenue, base_var_cost, base_fixed_cost, 
                                   n_simulations=10000):
    """
    Monte Carlo simulation for profit risk assessment.
    
    Returns distribution of possible profit outcomes.
    """
    np.random.seed(42)
    
    profits = []
    for _ in range(n_simulations):
        # Stochastic inputs
        rev_factor = np.random.uniform(0.85, 1.15)
        cost_factor = np.random.uniform(0.90, 1.10)
        volume_factor = np.random.normal(1.0, 0.05)
        volume_factor = np.clip(volume_factor, 0.85, 1.15)
        
        # Calculate profit
        revenue = base_revenue * rev_factor * volume_factor
        var_cost = base_var_cost * cost_factor * volume_factor
        profit = revenue - var_cost - base_fixed_cost
        
        profits.append(profit)
    
    return np.array(profits)
```

| Metric | Value |
|--------|-------|
| Mean Profit | €4,923,913 |
| Standard Deviation | €2,885,759 |
| Median Profit | €4,434,099 |
| Value at Risk (5%) | €1,210,844 |
| Value at Risk (1%) | €334,594 |
| Conditional VaR (5%) | €639,406 |
| **Probability of Loss** | **0.52%** |

> **📊 Conclusion**: Monte Carlo analysis reveals a **99.48% probability of profitability** under stochastic variation, with the worst 5% of scenarios still generating over €1.2 million in profit.

---

### 5.6 Scenario Analysis

Four strategic scenarios were evaluated:

| Scenario | Description | Net Profit | Margin | Δ vs Base |
|----------|-------------|-----------|--------|-----------|
| **Base Case** | Current operations | €4,532,803 | 54.6% | - |
| **Optimistic** | +10% revenue, -10% OpEx | €5,668,077 | 62.1% | +25.0% |
| **Pessimistic** | -10% revenue, +10% OpEx | €3,397,528 | 45.5% | -25.0% |
| **No Subsidy** | Exit bottom 17 zones | €4,526,021 | 55.0% | -0.15% |

> **💡 Strategic Insight**: The "No Subsidy" scenario demonstrates that exiting the lowest-performing 17 zones (19% of zones) reduces profit by only 0.15% while eliminating 0.8% of trips—confirming peripheral zones still contribute positively.

---

### 5.7 Subsidy vs. Profit Center Spatial Analysis

**Zone Classification:**

| Category | Zones | Description |
|----------|-------|-------------|
| **Profit Centers** | 89/89 (100%) | All zones generate positive contribution |
| **Total Zone Revenue** | €8,244,707 | |
| **Total Zone Profit** | €4,498,594 | |

> **📊 Key Finding**: Universal profitability across all 89 operational zones indicates robust demand distribution without geographic subsidization requirements.

---

## 📚 References

1. Bowman, J. L., & Ben-Akiva, M. E. (2001). Activity-based disaggregate travel demand model system with activity schedules. *Transportation Research Part A: Policy and Practice*, 35(1), 1-28.

2. Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481.

3. Ortúzar, J. D., & Willumsen, L. G. (2011). *Modelling transport* (4th ed.). John Wiley & Sons.

4. Reck, D. J., Haitao, H., Guidon, S., & Axhausen, K. W. (2021). Explaining shared micromobility usage, competition and mode choice by modelling empirical data from Zurich, Switzerland. *Transportation Research Part C: Emerging Technologies*, 124, 102947.

5. Shaheen, S., Cohen, A., Chan, N., & Bansal, A. (2020). Sharing strategies: carsharing, shared micromobility (bikesharing and scooter sharing), transportation network companies, microtransit, and other innovative mobility modes. *Transportation, Land Use, and Environmental Planning*, 237-262.

6. Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*, 37(1/2), 17-23.

7. Anselin, L. (1995). Local indicators of spatial association—LISA. *Geographical Analysis*, 27(2), 93-115.

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Data Processing | pandas, numpy |
| Spatial Analysis | geopandas, shapely, pyproj |
| Statistical Analysis | scipy.stats, lifelines |
| Visualization | matplotlib, seaborn, contextily |
| Progress Tracking | tqdm |

---

*Document generated: December 2025*  
*For academic use in PhD defense and portfolio demonstration*
