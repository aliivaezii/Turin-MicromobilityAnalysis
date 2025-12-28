# Appendix A: Table of Model Constants and Parameters

## For LaTeX Integration

> **Study**: Turin Shared E-Scooter Micromobility Analysis  
> **Period**: January 2024 – November 2025  
> **Sample Size**: N = 2,543,648 trips

---

## Table A.1: Economic Model Assumptions (2025)

| Parameter | Symbol | Value | Unit | Source |
|-----------|--------|-------|------|--------|
| Unlock Fee | $F_u$ | 1.00 | € per trip | Operator tariff |
| LIME Minute Rate | $r_{LIME}$ | 0.19 | € per minute | Operator tariff |
| VOI Minute Rate | $r_{VOI}$ | 0.22 | € per minute | Operator tariff |
| BIRD Minute Rate | $r_{BIRD}$ | 0.20 | € per minute | Operator tariff |
| **Average Minute Rate** | $\bar{r}$ | **0.203** | € per minute | Weighted mean |
| Variable Cost per Trip | $VC_{trip}$ | 1.20 | € per trip | Industry estimate |
| Vehicle Unit Cost | $C_{vehicle}$ | 600.00 | € | Hardware cost |
| Vehicle Lifespan | $L$ | 30 | months | Economic life |
| Daily Depreciation | $D_{daily}$ | 0.667 | € per vehicle-day | $C_{vehicle}/(L \times 30)$ |
| Annual City Fee per Vehicle | $F_{city}$ | 50.00 | € per vehicle-year | Municipal tender |
| Daily City Fee | $F_{city,d}$ | 0.137 | € per vehicle-day | $F_{city}/365$ |
| **Total Daily Fixed Cost** | $FC_{daily}$ | **0.804** | € per vehicle-day | $D_{daily} + F_{city,d}$ |

### LaTeX-Ready Formula Block

```latex
\begin{equation}
R_{trip} = F_u + (r_{operator} \times t_{duration})
\end{equation}

\begin{equation}
\pi_{trip} = R_{trip} - VC_{trip} - \frac{FC_{daily} \times N_{vehicles}}{N_{trips}}
\end{equation}

\text{Where: } F_u = €1.00, \quad \bar{r} = €0.203/\text{min}, \quad VC_{trip} = €1.20
```

---

## Table A.2: Spatial Model Parameters

| Parameter | Symbol | Value | Unit | Notes |
|-----------|--------|-------|------|-------|
| Gravity Model Distance Decay | $\beta$ | 1.50 | dimensionless | Exponential decay |
| Gravity Model Fit | $R^2$ | 0.72 | dimensionless | Goodness of fit |
| Average Trip Distance | $\bar{d}$ | 2.615 | km | Haversine calculation |
| Number of Statistical Zones | $n_{zones}$ | 89 | zones | Active zones |
| OD Matrix Sparsity | - | 17.4 | % | Zero-flow pairs |
| Gini Coefficient (Flow Inequality) | $G$ | 0.774 | dimensionless | High concentration |
| Shannon Entropy | $H$ | 0.858 | dimensionless | Moderate diversity |
| Flow Asymmetry Index | $A$ | 0.231 | dimensionless | Moderate imbalance |

### LaTeX-Ready Formula Block

```latex
\begin{equation}
T_{ij} = K \cdot P_i \cdot A_j \cdot e^{-\beta \cdot d_{ij}}
\end{equation}

\text{Where: } \beta = 1.50, \quad R^2 = 0.72, \quad \bar{d} = 2.615 \text{ km}
```

---

## Table A.3: Network Efficiency and Integration Metrics

| Parameter | Symbol | Value | Unit | Notes |
|-----------|--------|-------|------|-------|
| **Average Tortuosity Index** | $\bar{\tau}$ | **1.311** | dimensionless | All operators |
| LIME Tortuosity | $\tau_{LIME}$ | 1.299 | dimensionless | Most efficient |
| Tortuosity Std. Dev. | $\sigma_\tau$ | 0.101 | dimensionless | Zone-level |
| Tortuosity Range | - | [1.10, 1.81] | dimensionless | Min–Max |
| PT Integration (100m buffer) | $I_{100}$ | 85.4 | % | System average |
| Feeder Trips (100m buffer) | $F_{100}$ | 63.4 | % | Both endpoints |
| LIME Integration Index (100m) | - | 86.6 | % | - |
| VOI Integration Index (100m) | - | 86.8 | % | - |
| BIRD Integration Index (100m) | - | 84.7 | % | - |
| **LIME vs BIRD Efficiency Δ** | - | **1.8%** | percentage points | $\tau_{BIRD} - \tau_{LIME}$ |

### LaTeX-Ready Formula Block

```latex
\begin{equation}
\tau = \frac{D_{actual}}{D_{euclidean}} \geq 1.0
\end{equation}

\begin{equation}
I = \frac{N_{origin \leq d} + N_{dest \leq d}}{2 \cdot N_{total}}
\end{equation}

\text{At } d = 100m: \quad \bar{I} = 85.4\%, \quad \bar{F} = 63.4\%
```

---

## Table A.4: Survival Analysis Parameters

| Parameter | Symbol | Value | Unit | Notes |
|-----------|--------|-------|------|-------|
| **Ghost Vehicle Threshold** | $T_{ghost}$ | **5** | days (120 hours) | Operational definition |
| **LIME Weibull Shape** | $k_{LIME}$ | **0.6276** | dimensionless | Decreasing hazard |
| LIME Weibull Scale | $\lambda_{LIME}$ | 6.543 | hours | Characteristic life |
| **VOI Weibull Shape** | $k_{VOI}$ | **0.5696** | dimensionless | Decreasing hazard |
| VOI Weibull Scale | $\lambda_{VOI}$ | 22.841 | hours | Characteristic life |
| BIRD Weibull Shape | $k_{BIRD}$ | 0.6147 | dimensionless | Decreasing hazard |
| BIRD Weibull Scale | $\lambda_{BIRD}$ | 11.997 | hours | Characteristic life |
| System Weibull Shape | $k_{ALL}$ | 0.5928 | dimensionless | Aggregate |
| System Weibull Scale | $\lambda_{ALL}$ | 9.145 | hours | Aggregate |
| LIME Median Survival | $t_{50,LIME}$ | ~3.5 | hours | 50% rental threshold |
| VOI Median Survival | $t_{50,VOI}$ | ~10.5 | hours | 50% rental threshold |
| **LIME vs VOI Shape Δ** | - | **+10.2%** | relative | $(k_{LIME}/k_{VOI})-1$ |

### LaTeX-Ready Formula Block

```latex
\begin{equation}
S(t) = \exp\left[-\left(\frac{t}{\lambda}\right)^k\right]
\end{equation}

\begin{equation}
h(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1}
\end{equation}

\text{LIME: } k = 0.628, \lambda = 6.54h \quad \text{VOI: } k = 0.570, \lambda = 22.84h
```

---

## Table A.5: Operator-Specific Duration Parameters

| Operator | Avg Duration (min) | Median Survival (h) | Weibull k | Weibull λ (h) |
|----------|-------------------|---------------------|-----------|---------------|
| LIME | 10.46 | ~3.5 | 0.6276 | 6.54 |
| VOI | 9.59 | ~10.5 | 0.5696 | 22.84 |
| BIRD | 13.86 | ~6.0 | 0.6147 | 12.00 |
| **System** | **11.30** | **~5.0** | **0.5928** | **9.15** |

---

## Table A.6: Monte Carlo Simulation Parameters

| Parameter | Value | Distribution |
|-----------|-------|--------------|
| Number of Iterations | 10,000 | - |
| Revenue Variation | ±15% | Uniform |
| Variable Cost Variation | ±10% | Uniform |
| Volume Variation | ±5% | Normal (clipped) |
| Random Seed | 42 | Fixed for reproducibility |
| Probability of Loss | 0.52% | Simulated result |
| VaR (5%) | €1,210,844 | 5th percentile |
| CVaR (5%) | €639,406 | Mean below 5th percentile |

---

## Summary Constants for Quick Reference

```
┌────────────────────────────────────────────────────────────────────────┐
│                    CRITICAL MODEL CONSTANTS                            │
├────────────────────────────────────────────────────────────────────────┤
│  ECONOMIC                                                              │
│    • Unlock Fee:           €1.00/trip                                  │
│    • Avg Minute Rate:      €0.203/min                                  │
│    • Variable Cost:        €1.20/trip                                  │
│    • Daily Fixed Cost:     €0.804/vehicle-day                          │
│    • Scooter Lifespan:     30 months                                   │
│                                                                        │
│  SPATIAL                                                               │
│    • Gravity β:            1.50                                        │
│    • Avg Trip Distance:    2.615 km                                    │
│    • Avg Tortuosity:       1.311                                       │
│                                                                        │
│  SURVIVAL                                                              │
│    • Ghost Threshold:      5 days (120 hours)                          │
│    • LIME Weibull k:       0.6276                                      │
│    • VOI Weibull k:        0.5696                                      │
│    • LIME vs VOI Δk:       +10.2% (faster turnover)                    │
│                                                                        │
│  INTEGRATION                                                           │
│    • PT Buffer (standard): 100 meters                                  │
│    • Integration Rate:     85.4% (100m)                                │
│    • Feeder Rate:          63.4% (100m)                                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## LaTeX Table Template

```latex
\begin{table}[htbp]
\centering
\caption{Economic Model Parameters for Turin E-Scooter Analysis (2025)}
\label{tab:economic_params}
\begin{tabular}{llrl}
\toprule
\textbf{Parameter} & \textbf{Symbol} & \textbf{Value} & \textbf{Unit} \\
\midrule
Unlock Fee & $F_u$ & 1.00 & € per trip \\
Average Minute Rate & $\bar{r}$ & 0.203 & € per minute \\
Variable Cost per Trip & $VC_{trip}$ & 1.20 & € per trip \\
Daily Fixed Cost & $FC_{daily}$ & 0.804 & € per vehicle-day \\
Vehicle Lifespan & $L$ & 30 & months \\
\midrule
Gravity Decay Parameter & $\beta$ & 1.50 & dimensionless \\
Average Trip Distance & $\bar{d}$ & 2.615 & km \\
Average Tortuosity & $\bar{\tau}$ & 1.311 & dimensionless \\
\midrule
Ghost Vehicle Threshold & $T_{ghost}$ & 5 & days \\
LIME Weibull Shape & $k_{LIME}$ & 0.628 & dimensionless \\
VOI Weibull Shape & $k_{VOI}$ & 0.570 & dimensionless \\
\bottomrule
\end{tabular}
\end{table}
```

---

*Generated: December 2025*  
*Source: Turin Micromobility Analysis Pipeline v1.0*
