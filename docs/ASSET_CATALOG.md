# 📚 Asset Catalog: Turin E-Scooter Micromobility Analysis

> **Master Index of All Figures, Tables, and Documents**  
> Total Assets: 150+ Items | Generated: Research Pipeline v1.0  
> Use this catalog to locate and cite assets for your thesis chapters.

---

## 🗂️ Table of Contents

1. [Document Assets](#1-document-assets)
2. [Exercise 1: Temporal Analysis](#2-exercise-1-temporal-analysis)
3. [Exercise 2: OD Matrix & Spatial Patterns](#3-exercise-2-od-matrix--spatial-patterns)
4. [Exercise 3: Public Transit Integration](#4-exercise-3-public-transit-integration)
5. [Exercise 4: Parking & Survival Analysis](#5-exercise-4-parking--survival-analysis)
6. [Exercise 5: Economic Viability](#6-exercise-5-economic-viability)
7. [Quick Reference by Figure Type](#7-quick-reference-by-figure-type)
8. [Recommended Chapter Placement](#8-recommended-chapter-placement)

---

## 1. Document Assets

### 1.1 Methodology Documents

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `PAPER_METHODOLOGY_AND_RESULTS.md` | Report | Comprehensive methodology covering all 5 exercises with formulas, theory, and results tables | Core thesis content | Ch. 3 Methodology + Ch. 4 Results |
| `TECHNICAL_DEEPDIVE.md` | Report | 5-pillar technical documentation (Data Pipeline, Temporal, Spatial, Survival, Economic) | Technical appendix/supplement | Appendix A: Technical Notes |
| `APPENDIX_CONSTANTS.md` | Report | LaTeX-ready tables of all model constants and parameters | Reproducibility | Appendix B: Parameters |
| `ARCHITECTURE.md` | Report | Project structure and code organization overview | Supplementary | Appendix: Codebase Structure |
| `README.md` | Report | Project overview and execution instructions | GitHub/Portfolio | Not for thesis |

### 1.2 Exercise-Specific Reports

| Asset Name | Path | Description | Strategic Value |
|------------|------|-------------|-----------------|
| `figure_descriptions.md` | exercise2/ | Detailed descriptions of OD matrix figures | Figure captions reference |
| `integration_analysis_report.md` | exercise3/ | Full integration analysis narrative | Ch. 4.3 Results text |

---

## 2. Exercise 1: Temporal Analysis

### 2.1 Publication-Ready Dashboards

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `statistical_dashboard.png` | Dashboard | 6-panel statistical overview: bootstrap CI, effect sizes, weekday/weekend, variance | **HERO FIGURE** for temporal chapter | Ch. 4.1 Figure 1 |
| `operator_comparison_overview.png` | Dashboard | Combined view of all operators' key metrics | Multi-operator comparison | Ch. 4.1 Figure 2 |

### 2.2 Individual Extracted Figures (dashboard/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `fig_total_trip_volume.png` | Bar Chart | Total trips by operator (LIME: 1.42M, BIRD: 853K, VOI: 270K) | Market size visualization | Ch. 4.1 |
| `fig_weekly_pattern_comparison.png` | Line Chart | Day-of-week patterns for all operators | Behavioral insights | Ch. 4.1 |
| `fig_hourly_pattern_comparison.png` | Line Chart | Hourly demand curves (0-23h) by operator | Peak hour identification | Ch. 4.1 |
| `fig_fleet_utilization_comparison.png` | Bar Chart | Daily trips-per-vehicle comparison | Operational efficiency | Ch. 4.1 |
| `fig_fleet_size_comparison.png` | Bar Chart | Fleet size by operator | Scale comparison | Ch. 4.1 |
| `fig_summary_statistics_table.png` | Table | Key descriptive statistics rendered as figure | Summary table | Ch. 4.1 |
| `monthly_trend.png` | Line Chart | Monthly trip volume evolution (Jan 2024 - Nov 2025) | Seasonal patterns | Ch. 4.1 |
| `weekly_comparison.png` | Grouped Bar | Weekday vs weekend comparison | Usage patterns | Ch. 4.1 |
| `ridgeline_hourly.png` | Ridgeline | Hourly distributions by operator (ridgeline plot) | Distributional comparison | Ch. 4.1 |
| `temporal_heatmap.png` | Heatmap | Hour × Day-of-Week demand heatmap | Temporal clustering | Ch. 4.1 |
| `violin_daily_trips.png` | Violin | Daily trip count distributions | Variability analysis | Ch. 4.1 |
| `fleet_utilization.png` | Combined | Fleet utilization overview | Fleet efficiency | Ch. 4.1 |

### 2.3 Per-Operator Dashboards

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `LIME_Exercise1_Analysis.png` | Dashboard | 4-panel LIME analysis (monthly, weekly, hourly, fleet) | Operator deep-dive | Appendix: Operator Details |
| `VOI_Exercise1_Analysis.png` | Dashboard | 4-panel VOI analysis | Operator deep-dive | Appendix: Operator Details |
| `BIRD_Exercise1_Analysis.png` | Dashboard | 4-panel BIRD analysis | Operator deep-dive | Appendix: Operator Details |

### 2.4 Individual Operator Figures

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `lime_monthly_trend.png` | Line Chart | LIME monthly trip evolution | Individual operator analysis | Appendix |
| `lime_weekly_pattern.png` | Bar Chart | LIME day-of-week pattern | Individual operator analysis | Appendix |
| `lime_hourly_pattern.png` | Line Chart | LIME hourly demand curve | Individual operator analysis | Appendix |
| `lime_fleet_utilization.png` | Bar Chart | LIME fleet efficiency over time | Individual operator analysis | Appendix |
| `voi_monthly_trend.png` | Line Chart | VOI monthly trip evolution | Individual operator analysis | Appendix |
| `voi_weekly_pattern.png` | Bar Chart | VOI day-of-week pattern | Individual operator analysis | Appendix |
| `voi_hourly_pattern.png` | Line Chart | VOI hourly demand curve | Individual operator analysis | Appendix |
| `voi_fleet_utilization.png` | Bar Chart | VOI fleet efficiency over time | Individual operator analysis | Appendix |
| `bird_monthly_trend.png` | Line Chart | BIRD monthly trip evolution | Individual operator analysis | Appendix |
| `bird_weekly_pattern.png` | Bar Chart | BIRD day-of-week pattern | Individual operator analysis | Appendix |
| `bird_hourly_pattern.png` | Line Chart | BIRD hourly demand curve | Individual operator analysis | Appendix |
| `bird_fleet_utilization.png` | Bar Chart | BIRD fleet efficiency over time | Individual operator analysis | Appendix |

### 2.5 Statistical Figures (statistical/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `fig01_violin_distribution.png` | Violin | Daily trip distributions by operator with quartiles | Distributional comparison | Ch. 4.1 Statistical Results |
| `fig02_hourly_bird.png` | Line/Area | BIRD hourly pattern with confidence bands | Hourly precision | Ch. 4.1 |
| `fig03_hourly_lime.png` | Line/Area | LIME hourly pattern with confidence bands | Hourly precision | Ch. 4.1 |
| `fig04_hourly_voi.png` | Line/Area | VOI hourly pattern with confidence bands | Hourly precision | Ch. 4.1 |
| `fig05_heatmap_bird.png` | Heatmap | BIRD hour × day heatmap | Temporal clustering | Ch. 4.1 |
| `fig06_heatmap_lime.png` | Heatmap | LIME hour × day heatmap | Temporal clustering | Ch. 4.1 |
| `fig07_heatmap_voi.png` | Heatmap | VOI hour × day heatmap | Temporal clustering | Ch. 4.1 |
| `fig08_trend_decomposition.png` | Decomposition | STL decomposition: trend + seasonal + residual | Time series analysis | Ch. 4.1 |
| `fig09_kruskal_wallis.png` | Forest Plot | Kruskal-Wallis test results (H=47,832, p<0.001) | Statistical inference | Ch. 4.1 Statistical Tests |
| `fig10_bootstrap_ci.png` | Error Bars | Bootstrap 95% CI for mean daily trips | Uncertainty quantification | Ch. 4.1 |
| `fig11_peak_hours.png` | Bar Chart | Peak hour identification and intensity | Peak analysis | Ch. 4.1 |
| `fig12_weekend_weekday.png` | Grouped Bar | Weekend vs weekday comparison with effect sizes | Behavioral segmentation | Ch. 4.1 |

### 2.6 Additional Dashboard Figures

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `fig02_hourly_lime.png` (dashboard/) | Line | LIME hourly pattern | Per-operator hourly | Appendix |
| `fig02_hourly_voi.png` (dashboard/) | Line | VOI hourly pattern | Per-operator hourly | Appendix |
| `fig02_hourly_bird.png` (dashboard/) | Line | BIRD hourly pattern | Per-operator hourly | Appendix |
| `fig03_heatmap_lime.png` (dashboard/) | Heatmap | LIME temporal heatmap | Per-operator temporal | Appendix |
| `fig03_heatmap_voi.png` (dashboard/) | Heatmap | VOI temporal heatmap | Per-operator temporal | Appendix |
| `fig03_heatmap_bird.png` (dashboard/) | Heatmap | BIRD temporal heatmap | Per-operator temporal | Appendix |
| `fig05a_bootstrap_ci.png` | Error Bars | Bootstrap confidence intervals detailed | Statistical precision | Ch. 4.1 |
| `fig05b_effect_sizes.png` | Bar Chart | Cohen's d effect sizes between operators | Magnitude of differences | Ch. 4.1 |
| `fig05c_weekday_weekend.png` | Grouped Bar | Weekday/weekend split | Behavioral patterns | Ch. 4.1 |
| `fig05d_variance.png` | Bar Chart | Variance comparison across operators | Operational consistency | Ch. 4.1 |
| `fig07_fleet_lime.png` | Line | LIME fleet utilization over time | Fleet efficiency | Appendix |
| `fig07_fleet_voi.png` | Line | VOI fleet utilization over time | Fleet efficiency | Appendix |
| `fig07_fleet_bird.png` | Line | BIRD fleet utilization over time | Fleet efficiency | Appendix |

### 2.7 Tables (exercise1/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `table01_descriptive_stats.csv` | Table | Mean, median, std, min, max by operator | Summary statistics | Ch. 4.1 Table 1 |
| `table02_statistical_tests.csv` | Table | Kruskal-Wallis, Mann-Whitney, Cohen's d results | Hypothesis testing | Ch. 4.1 Table 2 |
| `table03_peak_hours.csv` | Table | Peak hour identification by operator | Peak analysis | Ch. 4.1 Table 3 |

---

## 3. Exercise 2: OD Matrix & Spatial Patterns

### 3.1 Combined Flow Maps

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `flow_map_allday_ALL.png` | Flow Map | All-day OD flows across all operators | **HERO FIGURE** for spatial chapter | Ch. 4.2 Figure 1 |
| `flow_map_allday_improved.png` | Flow Map | Enhanced all-day flow visualization | Alternative main figure | Ch. 4.2 |
| `flow_map_peak_improved.png` | Flow Map | Peak hour flows (7-9, 17-19) | Peak spatial patterns | Ch. 4.2 |
| `flow_map_offpeak_improved.png` | Flow Map | Off-peak flows | Off-peak patterns | Ch. 4.2 |

### 3.2 OD Heatmaps (combined/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `od_heatmap_allday_full.png` | Heatmap | Full 89×89 zone OD matrix (all day) | Complete matrix | Ch. 4.2 |
| `od_heatmap_allday_top30.png` | Heatmap | Top 30 zones OD matrix | Focused analysis | Ch. 4.2 |
| `od_heatmap_peak_top30.png` | Heatmap | Peak hours top 30 zones | Peak patterns | Ch. 4.2 |
| `od_heatmap_offpeak_top30.png` | Heatmap | Off-peak top 30 zones | Off-peak patterns | Ch. 4.2 |

### 3.3 Operator Comparison (combined/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `operator_comparison_combined.png` | Multi-panel | Side-by-side operator OD comparison | Cross-operator analysis | Ch. 4.2 |
| `operator_hourly_distribution.png` | Line Chart | Hourly trip distribution by operator | Temporal-spatial link | Ch. 4.2 |
| `operator_market_share.png` | Choropleth | Market share by zone | Competitive landscape | Ch. 4.2 |
| `operator_peak_offpeak.png` | Grouped Bar | Peak vs off-peak by operator | Temporal segmentation | Ch. 4.2 |
| `operator_zone_coverage.png` | Choropleth | Zone coverage by operator | Service area analysis | Ch. 4.2 |

### 3.4 Per-Operator Maps (per_operator/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `flow_map_lime_allday.png` | Flow Map | LIME-only OD flows | Operator-specific | Appendix |
| `flow_map_voi_allday.png` | Flow Map | VOI-only OD flows | Operator-specific | Appendix |
| `flow_map_bird_allday.png` | Flow Map | BIRD-only OD flows | Operator-specific | Appendix |
| `od_heatmap_lime_allday.png` | Heatmap | LIME OD matrix heatmap | Operator-specific | Appendix |
| `od_heatmap_voi_allday.png` | Heatmap | VOI OD matrix heatmap | Operator-specific | Appendix |
| `od_heatmap_bird_allday.png` | Heatmap | BIRD OD matrix heatmap | Operator-specific | Appendix |

### 3.5 Spatial Analysis (spatial/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `fig02_combined_peak_offpeak.png` | Combined | Peak vs off-peak spatial comparison | Temporal-spatial patterns | Ch. 4.2 |
| `fig02a_peak_od_heatmap.png` | Heatmap | Peak hours OD heatmap | Peak analysis | Ch. 4.2 |
| `fig02b_offpeak_od_heatmap.png` | Heatmap | Off-peak OD heatmap | Off-peak analysis | Ch. 4.2 |
| `fig05_combined_metrics.png` | 4-panel | Gini, Shannon, Asymmetry, Concentration | **KEY METRICS FIGURE** | Ch. 4.2 |
| `fig05a_gini_coefficient.png` | Choropleth | Gini coefficient by zone (inequality) | Spatial equity | Ch. 4.2 |
| `fig05b_shannon_entropy.png` | Choropleth | Shannon entropy (diversity) | Destination diversity | Ch. 4.2 |
| `fig05c_flow_asymmetry.png` | Choropleth | Flow asymmetry (|O-D|/Total) | Directional imbalance | Ch. 4.2 |
| `fig05d_spatial_concentration.png` | Choropleth | Spatial concentration index | Activity clustering | Ch. 4.2 |
| `fig06_combined_operators.png` | 6-panel | Multi-operator comparison dashboard | Comparative analysis | Ch. 4.2 |
| `fig06a_operator_trips.png` | Bar Chart | Total trips by operator | Volume comparison | Ch. 4.2 |
| `fig06b_operator_corridors.png` | Bar Chart | Active corridors by operator | Network coverage | Ch. 4.2 |
| `fig06c_operator_gini.png` | Bar Chart | Gini by operator | Concentration comparison | Ch. 4.2 |
| `fig06d_operator_entropy.png` | Bar Chart | Entropy by operator | Diversity comparison | Ch. 4.2 |
| `fig06e_operator_asymmetry.png` | Bar Chart | Asymmetry by operator | Balance comparison | Ch. 4.2 |
| `fig06f_operator_intrazonal.png` | Bar Chart | Intra-zonal trips by operator | Short trip analysis | Ch. 4.2 |
| `flow_map_professional.png` | Flow Map | Publication-quality flow map | **SUBMISSION FIGURE** | Ch. 4.2 Figure 1 |
| `od_asymmetry_heatmap.png` | Heatmap | Asymmetry matrix visualization | Directional patterns | Ch. 4.2 |
| `od_heatmap_clustered.png` | Heatmap | Hierarchically clustered OD matrix | Cluster analysis | Ch. 4.2 |

### 3.6 Statistical Figures (statistical/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `fig01_od_heatmap_bird.png` | Heatmap | BIRD OD matrix | Per-operator OD | Appendix |
| `fig02_od_heatmap_lime.png` | Heatmap | LIME OD matrix | Per-operator OD | Appendix |
| `fig03_od_heatmap_voi.png` | Heatmap | VOI OD matrix | Per-operator OD | Appendix |
| `fig04_top_origins.png` | Bar Chart | Top origin zones by trip volume | Key generators | Ch. 4.2 |
| `fig05_top_destinations.png` | Bar Chart | Top destination zones | Key attractors | Ch. 4.2 |
| `fig06_flow_imbalance.png` | Bar Chart | Net flow imbalance by zone | Rebalancing needs | Ch. 4.2 |
| `fig07_trip_distance_distribution.png` | Histogram | Trip distance distribution | Distance analysis | Ch. 4.2 |
| `fig08_gravity_model.png` | Scatter | Gravity model fit (α=0.47, β=-1.63, R²=0.82) | Model validation | Ch. 4.2 |
| `fig09_zone_connectivity.png` | Network | Zone connectivity graph | Network topology | Ch. 4.2 |
| `fig10_internal_vs_external.png` | Stacked Bar | Internal vs external trips by zone | Trip typology | Ch. 4.2 |

### 3.7 Tables (exercise2/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `table01_top_od_pairs.csv` | Table | Top 20 OD pairs by volume | Key corridors | Ch. 4.2 Table 1 |
| `table02_zone_statistics.csv` | Table | Zone-level summary statistics | Zone profiles | Ch. 4.2 Table 2 |
| `table03_gravity_parameters.csv` | Table | Gravity model parameters (α, β, R²) | Model summary | Ch. 4.2 Table 3 |

### 3.8 Data Exports (reports/exercise2/)

| Asset Name | Type | Description | Strategic Value |
|------------|------|-------------|-----------------|
| `OD_Matrix_AllDay.csv` | Matrix | Full OD matrix (all day) | Raw data |
| `OD_Matrix_Peak.csv` | Matrix | Peak hours OD matrix | Raw data |
| `OD_Matrix_OffPeak.csv` | Matrix | Off-peak OD matrix | Raw data |
| `OD_Matrix_LIME_AllDay.csv` | Matrix | LIME-specific OD matrix | Raw data |
| `OD_Matrix_VOI_AllDay.csv` | Matrix | VOI-specific OD matrix | Raw data |
| `OD_Matrix_BIRD_AllDay.csv` | Matrix | BIRD-specific OD matrix | Raw data |
| `corridor_dominance.csv` | Table | Corridor market share analysis | Competition analysis |
| `operator_market_share_by_zone.csv` | Table | Per-zone market shares | Competitive landscape |
| `checkpoint_chi_square.csv` | Table | Chi-square test results | Statistical tests |
| `checkpoint_zone_clusters.csv` | Table | Zone clustering assignments | Spatial clustering |
| `checkpoint_od_metrics.csv` | Table | OD matrix summary metrics | Aggregated metrics |

---

## 4. Exercise 3: Public Transit Integration

### 4.1 Main Maps

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `integration_map.png` | Choropleth | Zone-level PT integration score | **HERO FIGURE** | Ch. 4.3 Figure 1 |
| `competition_map.png` | Choropleth | E-scooter vs PT competition zones | Modal competition | Ch. 4.3 Figure 2 |
| `inefficient_routes_map.png` | Map | High tortuosity routes (feeder behavior) | Route efficiency | Ch. 4.3 |
| `trip_density_hexbin.png` | Hexbin | Trip density near PT stops | Spatial clustering | Ch. 4.3 |
| `summary_dashboard.png` | Dashboard | Multi-panel integration overview | Summary | Ch. 4.3 |

### 4.2 Analysis Charts

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `buffer_sensitivity_curve.png` | Line Chart | Integration % vs buffer distance (25-500m) | Sensitivity analysis | Ch. 4.3 |
| `operator_comparison_bar.png` | Bar Chart | Integration rates by operator | Cross-operator comparison | Ch. 4.3 |
| `route_competition_bar.png` | Bar Chart | Route-level competition with PT | Modal split | Ch. 4.3 |
| `temporal_comparison.png` | Grouped Bar | Peak vs off-peak integration | Temporal patterns | Ch. 4.3 |
| `tortuosity_histogram.png` | Histogram | Route tortuosity distribution | Efficiency analysis | Ch. 4.3 |
| `zone_scatter_integration.png` | Scatter | Zone size vs integration score | Spatial correlation | Ch. 4.3 |

### 4.3 Statistical Figures (statistical/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `fig01_competition_zones.png` | Choropleth | Zones with high PT competition | Competition hotspots | Ch. 4.3 |
| `fig02_integration_zones.png` | Choropleth | Zones with high integration | Integration hotspots | Ch. 4.3 |
| `fig03_buffer_sensitivity.png` | Line Chart | Buffer sensitivity curve with CI | Methodology validation | Ch. 3.3 |
| `fig04_buffer_by_operator.png` | Multi-line | Buffer sensitivity per operator | Operator differences | Ch. 4.3 |
| `fig05_temporal_peak.png` | Choropleth | Peak hour integration map | Temporal-spatial | Ch. 4.3 |
| `fig06_temporal_offpeak.png` | Choropleth | Off-peak integration map | Temporal-spatial | Ch. 4.3 |
| `fig07_peak_vs_offpeak.png` | Comparison | Peak vs off-peak integration rates | Temporal patterns | Ch. 4.3 |
| `fig08_operator_integration.png` | Bar Chart | Overall integration by operator | Operator summary | Ch. 4.3 |
| `fig09_feeder_percentage.png` | Bar Chart | Feeder trip percentage by operator | Feeder behavior | Ch. 4.3 |
| `fig10_tortuosity_histogram.png` | Histogram | Tortuosity distribution (μ=1.34) | Route efficiency | Ch. 4.3 |
| `fig11_integration_scatter.png` | Scatter | PT density vs integration score | Correlation analysis | Ch. 4.3 |
| `fig12_modal_split.png` | Pie/Bar | Modal split between e-scooter and PT | Modal competition | Ch. 4.3 |

### 4.4 Tables (exercise3/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `table01_integration_summary.csv` | Table | Overall integration metrics by operator | Summary statistics | Ch. 4.3 Table 1 |
| `table02_buffer_analysis.csv` | Table | Buffer sensitivity results | Methodology | Ch. 3.3 Table 2 |
| `table03_temporal_comparison.csv` | Table | Peak vs off-peak integration | Temporal analysis | Ch. 4.3 Table 3 |

### 4.5 Data Exports (reports/exercise3/)

| Asset Name | Type | Description | Strategic Value |
|------------|------|-------------|-----------------|
| `zone_integration_metrics.csv` | Table | Per-zone integration scores | Zone analysis |
| `buffer_sensitivity_results.csv` | Table | Full buffer analysis results | Sensitivity data |
| `route_competition_analysis_50m.csv` | Table | 50m buffer competition analysis | Route analysis |
| `top_competitor_routes_50m.csv` | Table | Top competing routes | Competition details |
| `temporal_analysis_results.csv` | Table | Temporal integration results | Peak/off-peak data |
| `lime_tortuosity_analysis.csv` | Table | LIME route tortuosity | Efficiency data |
| `full_integration_matrix.csv` | Matrix | Full integration score matrix | Raw data |
| `checkpoint_zones_with_metrics.geojson` | GeoJSON | Zones with computed metrics | GIS export |
| `checkpoint_turin_pt_stops.csv` | Table | PT stop locations | Reference data |

---

## 5. Exercise 4: Parking & Survival Analysis

### 5.1 Main Maps

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `map_parking_intensity.png` | Choropleth | Parking event intensity by zone | **HERO FIGURE** | Ch. 4.4 Figure 1 |
| `map_parking_turnover.png` | Choropleth | Parking turnover rate by zone | Turnover analysis | Ch. 4.4 |
| `map_abandoned_scooters.png` | Point Map | Long-duration parking (>120h) locations | Ghost scooter identification | Ch. 4.4 |

### 5.2 Analysis Charts

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `survival_curves.png` | Kaplan-Meier | Survival curves by operator | **KEY FIGURE** | Ch. 4.4 Figure 2 |
| `parking_duration_histogram.png` | Histogram | Parking duration distribution | Duration analysis | Ch. 4.4 |
| `parking_rhythm_curve.png` | Line Chart | Temporal pattern of parking events | Circadian rhythm | Ch. 4.4 |
| `operator_comparison.png` | Multi-bar | Operator comparison on survival metrics | Cross-operator | Ch. 4.4 |
| `turnover_vs_demand_scatter.png` | Scatter | Turnover vs demand relationship | Correlation | Ch. 4.4 |

### 5.3 Statistical Figures (statistical/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `fig01_weibull_survival.png` | Survival | Weibull-fitted survival curves (k=0.593) | **KEY FIGURE** | Ch. 4.4 Figure 1 |
| `fig02_logrank_forest.png` | Forest Plot | Log-rank test results (χ²=12,847, p<0.001) | Statistical inference | Ch. 4.4 |
| `fig03_bootstrap_median.png` | Error Bars | Bootstrap CI for median parking time | Uncertainty | Ch. 4.4 |
| `fig04_bootstrap_mean.png` | Error Bars | Bootstrap CI for mean parking time | Uncertainty | Ch. 4.4 |
| `fig05_bootstrap_cv.png` | Error Bars | Bootstrap CI for CV | Variability | Ch. 4.4 |
| `fig06_hazard_function.png` | Curve | Hazard function h(t) by operator | Reuse probability | Ch. 4.4 |
| `fig07_shape_interpretation.png` | Diagram | Weibull shape parameter interpretation | Methodology | Ch. 3.4 |
| `fig08_survival_quantiles.png` | Multi-line | Survival curves with quantile bands | Distributional | Ch. 4.4 |
| `fig09_operator_median.png` | Bar Chart | Median parking time by operator | Summary | Ch. 4.4 |
| `fig10_operator_cv.png` | Bar Chart | Coefficient of variation by operator | Consistency | Ch. 4.4 |
| `fig11_kruskal_wallis.png` | Forest Plot | Kruskal-Wallis test for parking times | Statistical test | Ch. 4.4 |

### 5.4 Tables (exercise4/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `table01_survival_summary.csv` | Table | Median, mean, 95th percentile by operator | Summary statistics | Ch. 4.4 Table 1 |
| `table02_bootstrap_statistics.csv` | Table | Bootstrap CI for all metrics | Uncertainty | Ch. 4.4 Table 2 |
| `table03_logrank_pairwise.csv` | Table | Pairwise log-rank test results | Statistical tests | Ch. 4.4 Table 3 |
| `table04_weibull_parameters.csv` | Table | Weibull shape (k) and scale (λ) | Model parameters | Ch. 4.4 Table 4 |

---

## 6. Exercise 5: Economic Viability

### 6.1 Main Maps

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `map_profitability_hotspots.png` | Choropleth | Net profit by zone | **HERO FIGURE** | Ch. 4.5 Figure 1 |
| `map_revenue_yield.png` | Choropleth | Revenue per trip by zone | Revenue analysis | Ch. 4.5 |

### 6.2 Analysis Charts

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `temporal_profitability_heatmap.png` | Heatmap | Hour × Day-of-Week profit heatmap | **KEY FIGURE** | Ch. 4.5 Figure 2 |
| `operator_pnl_waterfall.png` | Waterfall | P&L breakdown by operator | Financial summary | Ch. 4.5 |
| `pareto_value_curve.png` | Pareto | Zone contribution to total profit | 80/20 analysis | Ch. 4.5 |
| `break_even_scatter.png` | Scatter | Break-even analysis by zone | Viability threshold | Ch. 4.5 |
| `scenario_comparison_bridge.png` | Bridge | Scenario comparison (Base vs High Fee) | Policy analysis | Ch. 4.5 |
| `unit_economics_distribution.png` | Histogram | Per-trip profit distribution | Unit economics | Ch. 4.5 |

### 6.3 Statistical Figures (statistical/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `fig01_revenue_cost_operator.png` | Stacked Bar | Revenue vs cost breakdown by operator | Financial structure | Ch. 4.5 |
| `fig02_profit_margin_distribution.png` | Histogram | Profit margin distribution | Profitability spread | Ch. 4.5 |
| `fig03_monte_carlo_histogram.png` | Histogram | Monte Carlo profit distribution (N=1000) | Risk analysis | Ch. 4.5 |
| `fig04_tornado_sensitivity.png` | Tornado | Sensitivity analysis ranking | Key drivers | Ch. 4.5 |
| `fig05_bootstrap_ci_revenue.png` | Error Bars | Bootstrap CI for total revenue | Uncertainty | Ch. 4.5 |
| `fig06_bootstrap_ci_profit.png` | Error Bars | Bootstrap CI for net profit | Uncertainty | Ch. 4.5 |
| `fig07_bootstrap_ci_margin.png` | Error Bars | Bootstrap CI for profit margin | Uncertainty | Ch. 4.5 |
| `fig08_regression_duration_profit.png` | Scatter | Trip duration vs profit regression | Correlation | Ch. 4.5 |
| `fig09_pareto_zone_curve.png` | Lorenz | Zone contribution Pareto curve | Concentration | Ch. 4.5 |
| `fig10_scenario_comparison.png` | Grouped Bar | Multi-scenario comparison | Policy scenarios | Ch. 4.5 |

### 6.4 Tables (exercise5/)

| Asset Name | Type | Description | Strategic Value | Recommended Section |
|------------|------|-------------|-----------------|---------------------|
| `table01_operator_summary.csv` | Table | Revenue, cost, profit by operator | Financial summary | Ch. 4.5 Table 1 |
| `table02_monte_carlo_stats.csv` | Table | Monte Carlo summary statistics | Risk metrics | Ch. 4.5 Table 2 |
| `table03_sensitivity_ranking.csv` | Table | Sensitivity ranking of parameters | Key drivers | Ch. 4.5 Table 3 |
| `table04_scenario_comparison.csv` | Table | Scenario comparison results | Policy analysis | Ch. 4.5 Table 4 |

### 6.5 Data Exports (reports/exercise5/)

| Asset Name | Type | Description | Strategic Value |
|------------|------|-------------|-----------------|
| `checkpoint_operator_pnl.csv` | Table | Detailed P&L by operator | Financial details |
| `checkpoint_economics_zones.csv` | Table | Zone-level economics | Spatial economics |
| `checkpoint_monte_carlo_summary.csv` | Table | Monte Carlo simulation results | Risk data |
| `checkpoint_monte_carlo_simulations.csv` | Matrix | Full 1000 simulations | Raw Monte Carlo |
| `checkpoint_sensitivity_analysis.csv` | Table | Full sensitivity results | Sensitivity data |
| `checkpoint_bootstrap_ci.csv` | Table | Bootstrap CI results | Uncertainty data |
| `checkpoint_regression_analysis.csv` | Table | Regression model results | Statistical models |
| `checkpoint_economics_pareto.csv` | Table | Pareto analysis results | Concentration data |
| `checkpoint_economics_scenarios.csv` | Table | Scenario analysis results | Policy data |
| `checkpoint_economics_temporal.csv` | Table | Temporal profitability data | Hourly economics |

---

## 7. Quick Reference by Figure Type

### 7.1 Maps (Choropleths, Flow Maps)
| Count | Exercise | Key Assets |
|-------|----------|------------|
| 8 | Exercise 2 | `flow_map_*.png`, `operator_market_share.png` |
| 5 | Exercise 3 | `integration_map.png`, `competition_map.png` |
| 3 | Exercise 4 | `map_parking_*.png` |
| 2 | Exercise 5 | `map_profitability_hotspots.png`, `map_revenue_yield.png` |

### 7.2 Heatmaps
| Count | Exercise | Key Assets |
|-------|----------|------------|
| 9 | Exercise 1 | `temporal_heatmap.png`, `fig0[5-7]_heatmap_*.png` |
| 10 | Exercise 2 | `od_heatmap_*.png` |
| 1 | Exercise 5 | `temporal_profitability_heatmap.png` |

### 7.3 Statistical Charts
| Count | Exercise | Key Assets |
|-------|----------|------------|
| 12 | Exercise 1 | `statistical/fig01-12_*.png` |
| 10 | Exercise 2 | `statistical/fig01-10_*.png` |
| 12 | Exercise 3 | `statistical/fig01-12_*.png` |
| 11 | Exercise 4 | `statistical/fig01-11_*.png` |
| 10 | Exercise 5 | `statistical/fig01-10_*.png` |

### 7.4 Dashboards
| Exercise | Key Assets |
|----------|------------|
| Exercise 1 | `statistical_dashboard.png`, `operator_comparison_overview.png` |
| Exercise 3 | `summary_dashboard.png` |

---

## 8. Recommended Chapter Placement

### Chapter 3: Methodology
| Section | Assets |
|---------|--------|
| 3.1 Data Collection | `ARCHITECTURE.md`, data cleaning scripts |
| 3.2 Temporal Analysis Methods | `fig08_trend_decomposition.png`, `fig09_kruskal_wallis.png` |
| 3.3 OD Matrix Methods | `fig08_gravity_model.png`, `od_heatmap_clustered.png` |
| 3.4 Integration Methods | `fig03_buffer_sensitivity.png`, `fig07_shape_interpretation.png` |
| 3.5 Survival Analysis Methods | `fig01_weibull_survival.png`, `fig06_hazard_function.png` |
| 3.6 Economic Model | `APPENDIX_CONSTANTS.md`, economic parameters |

### Chapter 4: Results

| Section | Hero Figure | Supporting Figures | Key Tables |
|---------|-------------|-------------------|------------|
| 4.1 Temporal | `statistical_dashboard.png` | Hourly patterns, heatmaps, decomposition | Tables 1-3 |
| 4.2 Spatial | `flow_map_professional.png` | OD heatmaps, Gini/entropy maps | Tables 1-3 |
| 4.3 Integration | `integration_map.png` | Buffer sensitivity, operator comparison | Tables 1-3 |
| 4.4 Survival | `fig01_weibull_survival.png` | Log-rank forest, hazard function | Tables 1-4 |
| 4.5 Economics | `temporal_profitability_heatmap.png` | Tornado, Monte Carlo, Pareto | Tables 1-4 |

### Appendices

| Appendix | Content |
|----------|---------|
| A: Technical Notes | `TECHNICAL_DEEPDIVE.md` |
| B: Model Parameters | `APPENDIX_CONSTANTS.md` |
| C: Operator Details | Per-operator figures (12 individual charts) |
| D: Raw Data Exports | All checkpoint CSV files |
| E: Supplementary Maps | Additional flow maps and heatmaps |

---

## 📋 Asset Statistics Summary

| Category | Exercise 1 | Exercise 2 | Exercise 3 | Exercise 4 | Exercise 5 | Total |
|----------|------------|------------|------------|------------|------------|-------|
| **Figures** | 48 | 35 | 23 | 19 | 18 | **143** |
| **Tables** | 3 | 3 | 3 | 4 | 4 | **17** |
| **Data Exports** | 0 | 10 | 15 | 0 | 12 | **37** |
| **Documents** | - | 1 | 1 | - | - | **2** |

**Total Assets: 199**

---

*Generated for Turin E-Scooter Micromobility Analysis*  
*Last Updated: 2025*
