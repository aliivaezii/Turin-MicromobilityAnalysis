# Comprehensive Analysis of Shared E-Scooter Micromobility Services in Turin, Italy: Methodology and Results

## Abstract

This study presents a comprehensive empirical analysis of shared electric scooter (e-scooter) operations in Turin, Italy, examining 2,548,650 trips across three major operators (LIME, VOI, and BIRD) over the 2024-2025 period. We employ a multi-dimensional analytical framework encompassing temporal demand analysis, origin-destination flow modeling, public transport integration assessment, vehicle parking behavior through survival analysis, and microeconomic profitability evaluation. Our findings reveal distinct temporal usage patterns with pronounced evening peak demand, significant spatial heterogeneity in trip origins and destinations, high rates of multimodal integration with public transport, and overall positive economic viability across all operational zones.

---

## 1. Introduction and Data Description

### 1.1 Study Context

The proliferation of shared micromobility services, particularly dockless e-scooters, has fundamentally transformed urban transportation landscapes across European cities. Turin, as Italy's fourth-largest metropolitan area with approximately 870,000 inhabitants, provides an ideal laboratory for analyzing the operational dynamics, integration potential, and economic sustainability of multi-operator e-scooter systems.

### 1.2 Data Sources

This analysis utilizes trip-level data from three licensed e-scooter operators in Turin:

| Operator | Total Trips | Unique Vehicles | Fleet Utilization (trips/vehicle) | Study Period |
|----------|-------------|-----------------|-----------------------------------|--------------|
| LIME | 1,421,374 | 2,399 | 592.5 | 2024-2025 |
| BIRD | 852,751 | 2,819 | 302.5 | 2024-2025 |
| VOI | 274,525 | 1,441 | 190.5 | 2024-2025 |
| **Total** | **2,548,650** | **6,659** | **382.8** | - |

The dataset includes trip start and end timestamps, GPS coordinates, duration, and vehicle identifiers, enabling comprehensive spatial-temporal analysis.

---

## 2. Exercise 1: Temporal Demand Analysis

### 2.1 Theoretical Framework

Temporal analysis of shared mobility demand draws upon activity-based travel behavior theory (Bowman & Ben-Akiva, 2001), which posits that travel demand is derived from the need to participate in spatially distributed activities. The temporal distribution of e-scooter usage reflects underlying activity patterns, work schedules, and leisure behaviors of urban populations.

We examine demand across three temporal scales:
- **Hourly patterns**: Capture within-day variation and peak period identification
- **Weekly patterns**: Reveal weekday/weekend differentiation
- **Monthly trends**: Illuminate seasonal effects and growth trajectories

### 2.2 Methodology

Trips were aggregated by hour of day (0-23), day of week (Monday-Sunday), and calendar month. Peak periods were defined as the morning rush (07:00-10:00) and evening rush (17:00-20:00). Fleet utilization was computed as total trips divided by unique vehicle identifiers.

### 2.3 Results

#### 2.3.1 Hourly Demand Patterns

All three operators exhibit pronounced **unimodal evening peak demand**, with maximum ridership occurring between 16:00 and 18:00:

| Operator | Peak Hour | Peak Trips | Share of Daily Demand | Morning Rush (7-10) | Evening Rush (17-20) |
|----------|-----------|------------|----------------------|---------------------|---------------------|
| LIME | 18:00 | 102,488 | 7.2% | 9.9% | 20.5% |
| BIRD | 18:00 | 61,418 | 7.2% | 9.1% | 20.6% |
| VOI | 16:00 | 19,118 | 7.0% | 11.0% | 17.2% |

The concentration of demand in the evening peak (approximately 20% of daily trips) significantly exceeds morning peak usage (9-11%), suggesting e-scooters in Turin are predominantly used for return commuting and evening leisure activities rather than morning work trips.

#### 2.3.2 Weekly Patterns

| Operator | Weekday Avg (trips/day) | Weekend Avg (trips/day) | Weekend/Weekday Ratio |
|----------|------------------------|------------------------|----------------------|
| LIME | 201,907 | 205,920 | 1.02 |
| BIRD | 119,710 | 127,100 | 1.06 |
| VOI | 41,071 | 34,584 | 0.84 |

LIME and BIRD show relatively balanced weekday-weekend usage with slight weekend increases, indicating a mix of utilitarian and recreational use. VOI exhibits stronger weekday orientation, suggesting more commute-focused ridership.

#### 2.3.3 Fleet Utilization Distribution

Vehicle-level analysis reveals substantial heterogeneity in utilization:

| Operator | Mean Trips/Vehicle | Median | Max | Standard Deviation |
|----------|-------------------|--------|-----|-------------------|
| LIME | 592.5 | 334 | 1,649 | High variance |
| BIRD | 302.5 | 181 | 1,026 | Moderate variance |
| VOI | 190.5 | 164 | 1,858 | Moderate variance |

The right-skewed distribution (mean > median) indicates a subset of high-performing vehicles contributing disproportionately to system output, consistent with spatial clustering of demand in high-activity zones.

---

## 3. Exercise 2: Origin-Destination Flow Analysis

### 3.1 Theoretical Framework

Origin-Destination (OD) matrices form the cornerstone of transportation demand modeling (Ortúzar & Willumsen, 2011). For micromobility systems, OD analysis reveals:
- **Spatial demand patterns**: Identification of trip generators and attractors
- **Flow concentration**: Pareto-like distributions of demand across zone pairs
- **Trip purpose inference**: Intra-zonal vs. inter-zonal movement characteristics

We employ entropy-based diversity metrics (Shannon entropy) and inequality measures (Gini coefficient) to characterize flow distributions.

### 3.2 Methodology

Trip origins and destinations were geocoded to 90 statistical zones (*Zone Statistiche*) defined by the City of Turin. The OD matrix $T_{ij}$ represents trips from zone $i$ to zone $j$. Key metrics include:

**Shannon Entropy (Diversity):**
$$H = -\sum_{i,j} p_{ij} \log(p_{ij})$$

where $p_{ij} = T_{ij} / \sum_{i,j} T_{ij}$

**Gini Coefficient (Inequality):**
$$G = \frac{\sum_{i=1}^{n}\sum_{j=1}^{n}|T_i - T_j|}{2n^2\bar{T}}$$

**Flow Asymmetry:**
$$A = \frac{\sum_{i<j}|T_{ij} - T_{ji}|}{\sum_{i,j}T_{ij}}$$

### 3.3 Results

#### 3.3.1 Matrix Structure

| Metric | All-Day | Peak Hours | Off-Peak |
|--------|---------|------------|----------|
| Total Trips | 2,509,948 | 752,924 | 1,757,024 |
| Active Zones | 89 | 88 | 86 |
| Non-Zero OD Pairs | 6,435 | 5,848 | 6,335 |
| Matrix Sparsity | 17.4% | - | - |

#### 3.3.2 Intra-zonal vs. Inter-zonal Trips

| Category | Trips | Percentage |
|----------|-------|------------|
| Intra-zonal (within same zone) | 262,211 | 10.4% |
| Inter-zonal (between zones) | 2,247,737 | 89.6% |

The low intra-zonal share (10.4%) indicates e-scooters are primarily used for **medium-distance urban trips** rather than very short within-neighborhood movements, consistent with their role as "last-mile" and "first-mile" connectors.

#### 3.3.3 Flow Concentration and Inequality

| Metric | All-Day Value | Interpretation |
|--------|---------------|----------------|
| Gini Coefficient | 0.774 | High inequality - demand concentrated in few corridors |
| Shannon Entropy | 0.858 | Moderate diversity relative to maximum possible |
| Flow Asymmetry | 0.231 | Moderate directional imbalance |
| Top 10 Corridors Share | 3.75% | Concentrated demand in key routes |
| Top 50 Corridors Share | 13.2% | |

The high Gini coefficient (0.774) reveals that trip flows are highly concentrated along specific corridors, primarily connecting central zones (04, 08, 01, 33) to peripheral areas.

#### 3.3.4 Top Origin-Destination Pairs

| Rank | Origin Zone | Destination Zone | Daily Trips | Corridor Type |
|------|-------------|------------------|-------------|---------------|
| 1 | 04 (Piazza San Carlo) | Multiple | 126,229 | Central hub |
| 2 | 08 (Porta Susa Station) | Multiple | 99,777 | Transit hub |
| 3 | 01 (Municipio) | Multiple | 94,787 | Civic center |
| 4 | 33 | Multiple | 92,884 | Residential |
| 5 | 03 | Multiple | 88,992 | Mixed-use |

---

## 4. Exercise 3: Public Transport Integration Analysis

### 4.1 Theoretical Framework

Multimodal integration between micromobility and public transport (PT) is a key policy objective for sustainable urban mobility (Shaheen et al., 2020). E-scooters can serve as **first-mile/last-mile (FMLM) connectors**, extending the catchment area of transit stops and enabling seamless door-to-door journeys.

Integration potential is assessed through spatial proximity analysis using buffer-based methodologies (Reck et al., 2021), where trips originating or terminating within a defined distance of PT stops are classified as potentially integrated.

### 4.2 Methodology

Public transport stop locations were extracted from Turin's GTFS (General Transit Feed Specification) data. For each e-scooter trip, we calculated:
- Distance from trip origin to nearest PT stop
- Distance from trip destination to nearest PT stop

Trips were classified as **PT-integrated** if either origin or destination fell within a specified buffer distance (50m, 100m, or 200m) of a transit stop.

**Integration Index:**
$$I = \frac{N_{origin \leq d} + N_{dest \leq d}}{2N_{total}}$$

**Feeder Percentage (strict):**
$$F = \frac{N_{both \leq d}}{N_{total}}$$

### 4.3 Results

#### 4.3.1 Buffer Sensitivity Analysis

| Operator | Buffer (m) | Integration Index | Feeder % | Trips Near PT Origin | Trips Near PT Dest |
|----------|-----------|------------------|----------|---------------------|-------------------|
| LIME | 50 | 55.7% | 34.1% | 484,016 | 472,541 |
| LIME | 100 | 86.6% | 63.9% | 908,392 | 905,342 |
| LIME | 200 | 99.8% | 95.9% | 1,362,447 | 1,360,981 |
| VOI | 50 | 52.2% | 31.8% | 85,677 | 82,321 |
| VOI | 100 | 86.8% | 63.1% | 169,968 | 169,624 |
| VOI | 200 | 99.8% | 94.4% | 254,374 | 253,818 |
| BIRD | 50 | 53.8% | 34.2% | 291,880 | 262,622 |
| BIRD | 100 | 84.7% | 63.3% | 540,147 | 520,139 |
| BIRD | 200 | 99.5% | 95.2% | 811,551 | 806,631 |

#### 4.3.2 Zone-Level Integration Metrics

Analysis across 89 zones revealed:

| Metric | Mean | Std Dev | Max |
|--------|------|---------|-----|
| Total Trips per Zone | 28,422 | 27,987 | 126,533 |
| Integrated Trips per Zone | 18,096 | - | - |
| Integration Rate (%) | 63.8% | - | - |
| Average Tortuosity | 1.31 | - | - |

**Key Finding:** At the 100-meter buffer (commonly used in literature), approximately **85-87% of e-scooter trips** demonstrate potential integration with public transport, with **63-64%** classified as strict "feeder" trips (both endpoints near PT).

#### 4.3.3 Spatial Variation

The highest integration rates were observed in:
- **Zone 08 (Porta Susa Station)**: 88.2% integration rate - major railway hub
- **Zone 01 (Municipio)**: 64.4% - city center with dense transit
- **Zone 04 (Piazza San Carlo)**: 60.8% - commercial core

This suggests e-scooters effectively complement the existing transit network, particularly around major transport nodes.

---

## 5. Exercise 4: Parking Behavior and Vehicle Availability (Survival Analysis)

### 5.1 Theoretical Framework

Survival analysis, originally developed for biomedical research (Kaplan & Meier, 1958), provides a powerful framework for analyzing time-to-event data in transportation contexts. For shared micromobility, the "survival" function $S(t)$ represents the probability that a parked vehicle remains available (not rented) at time $t$ after being parked.

This analysis informs:
- **Fleet rebalancing strategies**: Identifying vehicles likely to remain idle
- **Demand prediction**: Understanding availability patterns by location
- **Operational efficiency**: Optimizing vehicle-trip matching

### 5.2 Methodology

#### 5.2.1 Kaplan-Meier Estimator

The non-parametric Kaplan-Meier estimator was used to estimate survival curves:

$$\hat{S}(t) = \prod_{t_i \leq t} \left(1 - \frac{d_i}{n_i}\right)$$

where $d_i$ is the number of "events" (rentals) at time $t_i$ and $n_i$ is the number of vehicles at risk.

#### 5.2.2 Weibull Distribution Fitting

Parametric modeling employed the Weibull distribution, characterized by shape ($k$) and scale ($\lambda$) parameters:

$$f(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1}e^{-(t/\lambda)^k}$$

- $k < 1$: Decreasing hazard (rental probability decreases over time - "aging" vehicles become less likely to be rented)
- $k = 1$: Constant hazard (exponential distribution)
- $k > 1$: Increasing hazard (vehicles become more likely to be rented over time)

#### 5.2.3 Log-Rank Test

Inter-operator differences were tested using the log-rank test statistic:

$$\chi^2 = \frac{\left(\sum_{i}(O_{1i} - E_{1i})\right)^2}{\sum_{i}V_i}$$

### 5.3 Results

#### 5.3.1 Kaplan-Meier Survival Curves

| Time (hours) | BIRD Survival | LIME Survival | VOI Survival | ALL Survival |
|--------------|---------------|---------------|--------------|--------------|
| 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| 1 | 0.801 | 0.734 | 0.843 | 0.766 |
| 2 | 0.699 | 0.597 | 0.773 | 0.646 |
| 4 | 0.577 | 0.446 | 0.686 | 0.510 |
| 8 | 0.447 | 0.302 | 0.580 | 0.375 |
| 12 | 0.365 | 0.216 | 0.493 | 0.290 |
| 24 | 0.202 | 0.096 | 0.339 | 0.152 |
| 48 | 0.092 | 0.038 | 0.211 | 0.071 |
| 72 | 0.050 | 0.018 | 0.145 | 0.040 |

**Key Finding:** LIME exhibits the fastest "turnover" with only 9.6% of vehicles remaining unrented after 24 hours, compared to 20.2% for BIRD and 33.9% for VOI. This indicates LIME achieves higher fleet utilization in Turin.

#### 5.3.2 Median Survival Times

- **LIME**: ~3.5 hours (50% of parked vehicles rented within 3.5 hours)
- **BIRD**: ~6.0 hours
- **VOI**: ~10.5 hours
- **System-wide**: ~5.0 hours

#### 5.3.3 Weibull Distribution Parameters

| Operator | Shape (k) | Scale (λ) | Interpretation |
|----------|-----------|-----------|----------------|
| BIRD | 0.615 | 12.0 | Decreasing hazard |
| LIME | 0.628 | 6.5 | Decreasing hazard |
| VOI | 0.570 | 22.8 | Decreasing hazard |
| ALL | 0.593 | 9.1 | Decreasing hazard |

All operators exhibit shape parameters $k < 1$, indicating **decreasing hazard rates**. This implies that as parking duration increases, the instantaneous probability of rental decreases—vehicles parked for extended periods are in low-demand locations and become increasingly unlikely to be rented.

#### 5.3.4 Log-Rank Test Results

| Comparison | Chi-Square | p-value | Significant |
|------------|-----------|---------|-------------|
| BIRD vs LIME | 3.28 × 10⁹ | <0.001 | Yes |
| BIRD vs VOI | 3.41 × 10⁸ | <0.001 | Yes |
| LIME vs VOI | 1.43 × 10⁹ | <0.001 | Yes |

The extremely high chi-square values and p < 0.001 confirm that survival curves differ **significantly** between all operator pairs, reflecting distinct fleet management strategies, vehicle positioning, and customer bases.

---

## 6. Exercise 5: Economic and Sensitivity Analysis

### 6.1 Theoretical Framework

Microeconomic analysis of shared mobility platforms employs activity-based costing (ABC) to allocate revenues and costs at the trip level (Kaplan & Anderson, 2007). The unit economics framework decomposes profitability into:

$$\pi_{trip} = R_{trip} - VC_{trip} - \frac{FC}{N_{trips}}$$

where:
- $R_{trip}$ = Revenue per trip (unlock fee + time-based fare)
- $VC_{trip}$ = Variable costs (battery charging, maintenance, insurance)
- $FC$ = Fixed costs (permits, overhead, vehicle depreciation)
- $N_{trips}$ = Total trips

Scenario analysis and Monte Carlo simulation quantify uncertainty and strategic options.

### 6.2 Methodology

#### 6.2.1 Revenue Model

Based on typical Italian e-scooter pricing:
- Unlock fee: €1.00 per trip
- Per-minute rate: €0.15/minute

$$R_{trip} = 1.00 + 0.15 \times duration_{minutes}$$

#### 6.2.2 Cost Model

**Variable Costs:**
- Battery/charging: €0.05 per trip
- Maintenance: €0.35 per trip
- Insurance: €0.40 per trip
- Operations: €0.40 per trip

**Fixed Costs (per vehicle-year):**
- Permits/licenses: €50
- Depreciation: €200 (assuming €600 vehicle cost, 3-year life)

#### 6.2.3 Scenario Analysis

Four scenarios were evaluated:
1. **Base Case**: Current operations
2. **Optimistic**: +10% revenue, -10% variable costs
3. **Pessimistic**: -10% revenue, +10% variable costs
4. **No Subsidy**: Exit from lowest-performing 20% of zones

#### 6.2.4 Monte Carlo Simulation

10,000 iterations with stochastic variation in:
- Revenue (±15% uniform distribution)
- Variable costs (±10% uniform distribution)
- Trip volumes (±5% normal distribution)

### 6.3 Results

#### 6.3.1 System-Wide Economics

| Metric | Value |
|--------|-------|
| **Total Trips** | 2,543,648 |
| **Total Revenue** | €8,300,365 |
| **Total Variable Costs** | €3,052,378 |
| **Total Fixed Costs** | €715,185 |
| **Total Costs** | €3,767,563 |
| **Net Profit** | €4,532,803 |
| **Profit Margin** | 54.6% |

#### 6.3.2 Operator-Level Profitability

| Operator | Trips | Revenue | Total Cost | Net Profit | Profit/Trip | Margin |
|----------|-------|---------|------------|------------|-------------|--------|
| LIME | 1,421,372 | €4,245,099 | €2,037,115 | €2,207,983 | €1.55 | 52.0% |
| BIRD | 852,751 | €3,217,369 | €1,316,081 | €1,901,288 | €2.23 | 59.1% |
| VOI | 269,525 | €837,898 | €414,367 | €423,532 | €1.57 | 50.6% |

**Key Finding:** All three operators achieve positive profitability with margins exceeding 50%. BIRD achieves the highest per-trip profit (€2.23) due to longer average trip durations (13.9 minutes vs. 10.5 for LIME).

#### 6.3.3 Trip-Level Economics

| Metric | Value |
|--------|-------|
| Average Revenue per Trip | €3.26 |
| Average Cost per Trip | €1.48 |
| Average Profit per Trip | €1.78 |
| Profitable Trip Share | 97.1% |

#### 6.3.4 Spatial Profitability (Zone-Level)

| Category | Zones | Interpretation |
|----------|-------|----------------|
| Profitable Zones | 89/89 (100%) | All zones generate positive contribution |
| Total Zone Revenue | €8,244,707 | |
| Total Zone Profit | €4,498,594 | |

The universal profitability across all 89 operational zones indicates robust demand distribution without geographic subsidization requirements.

#### 6.3.5 Temporal Profitability Matrix

| Day | Peak Profit Hour | Peak Profit (€) | Low Profit Hour | Low Profit (€) |
|-----|-----------------|-----------------|-----------------|----------------|
| Friday | 17:00 | €56,178 | - | - |
| Tuesday | - | - | 04:00 | €3,131 |

The temporal analysis reveals a 18× variation between peak and trough profitability, emphasizing the importance of temporal demand management.

#### 6.3.6 Scenario Analysis Results

| Scenario | Net Profit | Margin | Δ vs Base | Δ% |
|----------|-----------|--------|-----------|-----|
| Base Case | €4,532,803 | 54.6% | - | - |
| Optimistic | €5,668,077 | 62.1% | +€1,135,274 | +25.0% |
| Pessimistic | €3,397,528 | 45.5% | -€1,135,274 | -25.0% |
| No Subsidy (drop 17 zones) | €4,526,021 | 55.0% | -€6,782 | -0.15% |

The "No Subsidy" scenario demonstrates that exiting the lowest-performing 17 zones (19% of zones) reduces profit by only 0.15% while eliminating 0.8% of trips—confirming that peripheral zones, while less profitable, still contribute positively to the system.

#### 6.3.7 Monte Carlo Risk Analysis

| Metric | Value |
|--------|-------|
| Mean Profit (10,000 simulations) | €4,923,913 |
| Standard Deviation | €2,885,759 |
| Median Profit | €4,434,099 |
| Value at Risk (5%) | €1,210,844 |
| Value at Risk (1%) | €334,594 |
| Conditional VaR (5%) | €639,406 |
| Probability of Loss | 0.52% |

**Key Finding:** Monte Carlo analysis reveals a **99.48% probability of profitability** under stochastic variation, with the worst 5% of scenarios still generating over €1.2 million in profit.

---

## 7. Conclusions

This comprehensive analysis of Turin's shared e-scooter system reveals:

1. **Temporal Patterns**: Pronounced evening peak demand (17:00-18:00) with 20% of daily trips concentrated in the 17:00-20:00 period, indicating strong commute-return and leisure use cases.

2. **Spatial Concentration**: High Gini coefficient (0.774) demonstrates demand clustering in central zones, with 89.6% of trips crossing zone boundaries, confirming e-scooters' role in medium-distance urban mobility.

3. **Multimodal Integration**: At 100-meter proximity, 85-87% of trips demonstrate potential public transport integration, with 63-64% classified as strict feeder trips—validating the first-mile/last-mile value proposition.

4. **Vehicle Turnover**: Weibull analysis reveals decreasing hazard rates (k < 1) across all operators, with LIME achieving fastest turnover (median survival 3.5 hours) and highest fleet utilization.

5. **Economic Viability**: System-wide profit margin of 54.6% with 97.1% of individual trips profitable. Monte Carlo simulation confirms 99.48% probability of annual profitability.

These findings support the continued expansion of e-scooter services in Turin while highlighting opportunities for optimized fleet rebalancing, enhanced transit integration, and demand-responsive pricing strategies.

---

## References

Bowman, J. L., & Ben-Akiva, M. E. (2001). Activity-based disaggregate travel demand model system with activity schedules. *Transportation Research Part A: Policy and Practice*, 35(1), 1-28.

Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*, 53(282), 457-481.

Kaplan, R. S., & Anderson, S. R. (2007). *Time-driven activity-based costing: a simpler and more powerful path to higher profits*. Harvard Business Press.

Ortúzar, J. D., & Willumsen, L. G. (2011). *Modelling transport* (4th ed.). John Wiley & Sons.

Reck, D. J., Haitao, H., Guidon, S., & Axhausen, K. W. (2021). Explaining shared micromobility usage, competition and mode choice by modelling empirical data from Zurich, Switzerland. *Transportation Research Part C: Emerging Technologies*, 124, 102947.

Shaheen, S., Cohen, A., Chan, N., & Bansal, A. (2020). Sharing strategies: carsharing, shared micromobility (bikesharing and scooter sharing), transportation network companies, microtransit, and other innovative mobility modes. *Transportation, Land Use, and Environmental Planning*, 237-262.

---

*Document generated: December 2025*
*Analysis performed using Python 3.9 with pandas, geopandas, scipy, lifelines, and matplotlib*
