# E-Scooter & Public Transport Integration Analysis Report

    ## Executive Summary

    This analysis examines the integration between e-scooter services and public transport in Turin, using standard scientific buffer values for accessibility analysis.

    **Analysis Date:** 2025-12-28 15:10

    **Mode:** Single Pass Optimized - Standard Values (50m PT stops, 50m route corridors)

    ---

    ## Configuration

    - **PT Stop Buffer Distance:** 50 meters (standard first/last-mile catchment)
    - **Route Corridor Buffer:** 50 meters (street corridor width)
    - **Peak Hours Definition:** [7, 8, 9, 17, 18, 19] (Morning: 7-9, Evening: 17-19)
    - **Total Trips Analyzed:** 2,543,648

    ---

    ## Integration Analysis Results

    | Operator | Buffer (m) | Integration Index (%) | Feeder (%) | Total Trips |
    |----------|------------|----------------------|------------|-------------|
    | LIME | 50 | 55.74 | 34.05 | 1,421,372 |
| LIME | 100 | 86.63 | 63.91 | 1,421,372 |
| LIME | 200 | 99.77 | 95.85 | 1,421,372 |
| VOI | 50 | 52.20 | 31.79 | 269,525 |
| VOI | 100 | 86.79 | 63.06 | 269,525 |
| VOI | 200 | 99.81 | 94.38 | 269,525 |
| BIRD | 50 | 53.81 | 34.23 | 852,751 |
| BIRD | 100 | 84.74 | 63.34 | 852,751 |
| BIRD | 200 | 99.54 | 95.17 | 852,751 |

    ---

    ## Temporal Segmentation Results

    | Operator | Time Period | Integration Index (%) | Feeder (%) |
    |----------|-------------|----------------------|------------|
    | LIME | Peak | 56.53 | 33.86 |
| LIME | Off-Peak | 55.39 | 34.14 |
| LIME | Peak | 86.92 | 63.49 |
| LIME | Off-Peak | 86.50 | 64.10 |
| LIME | Peak | 99.77 | 95.87 |
| LIME | Off-Peak | 99.77 | 95.85 |
| VOI | Peak | 53.92 | 32.80 |
| VOI | Off-Peak | 51.51 | 31.38 |
| VOI | Peak | 87.14 | 64.55 |
| VOI | Off-Peak | 86.65 | 62.46 |
| VOI | Peak | 99.84 | 95.62 |
| VOI | Off-Peak | 99.79 | 93.88 |
| BIRD | Peak | 54.13 | 33.66 |
| BIRD | Off-Peak | 53.68 | 34.47 |
| BIRD | Peak | 84.80 | 62.38 |
| BIRD | Off-Peak | 84.72 | 63.75 |
| BIRD | Peak | 99.54 | 94.84 |
| BIRD | Off-Peak | 99.55 | 95.31 |

    ---

    ## Key Findings

    1. **Buffer Sensitivity**: Integration metrics show consistent patterns across buffer distances
    2. **Temporal Patterns**: Peak hours may show different integration characteristics
    3. **Operator Comparison**: Different operators exhibit varying levels of PT integration

    ---

    *Report generated automatically by 04_transport_comparison.py*
    *Visualizations available separately via 04_visualization.py*
    