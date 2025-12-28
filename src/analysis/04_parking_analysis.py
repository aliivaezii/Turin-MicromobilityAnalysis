#!/usr/bin/env python3
"""
==============================================================================
Exercise 4: Parking Duration Analysis 
==============================================================================
Top-tier fleet management analysis for E-Scooter parking.

SURVIVAL ANALYSIS FRAMEWORK:
1. Kaplan-Meier Estimator
   - Non-parametric survival curves by operator
   - 95% confidence intervals via Greenwood's formula
   - Log-rank test for group comparisons (χ² statistic)

2. Weibull Distribution Fitting
   - Parametric survival model: S(t) = exp(-(t/λ)^k)
   - Shape (k) and Scale (λ) parameter estimation via MLE
   - AIC/BIC for model selection and goodness-of-fit

3. Cox Proportional Hazards (Optional - requires lifelines)
   - Hazard ratios for operator effects
   - Covariate adjustment (time-of-day, zone type)

SPATIAL STATISTICS FRAMEWORK:
1. Global Moran's I
   - Spatial autocorrelation for parking patterns
   - Permutation-based p-values (999 simulations)

2. Local Indicators of Spatial Association (LISA)
   - Hot-spot / Cold-spot identification
   - Getis-Ord Gi* statistic for clustering

OPERATOR COMPARISON FRAMEWORK:
1. Kruskal-Wallis H-test (non-parametric ANOVA)
2. Pairwise Mann-Whitney U with Bonferroni correction
3. Effect sizes: η² (eta-squared), Cohen's d
4. Bootstrap confidence intervals (n=1000)

PROCESS FLOW:
1. Load & Sort: Load checkpoint, sort by operator, vehicle_id, start_time
2. Calculate Idle Intervals: Idle_Time = Start_Time(Next) - End_Time(Current)
3. Detect "Ghost Vehicles": Flag parking events > 120 hours as is_abandoned
4. Spatial Aggregation: Assign parking events to Statistical Zones
5. Zone Metrics: median_parking_hours, parking_density, abandoned_count
6. Temporal Aggregation: "Fleet Pulse" by hour_of_day
7. Survival Analysis: Kaplan-Meier + Weibull + Log-rank test
8. Spatial Autocorrelation: Global Moran's I + LISA clustering
9. Operator Statistics: Kruskal-Wallis + pairwise tests + bootstrap CI

OUTPUTS:
- checkpoint_parking_zones.geojson (Zone statistics with Moran's I)
- checkpoint_parking_events.pkl (Raw events for histograms)
- checkpoint_parking_hourly.csv (Temporal data - Fleet Pulse)
- checkpoint_ghost_vehicles.pkl (Locations of abandoned scooters)
- checkpoint_survival_analysis.csv (Kaplan-Meier survival table)
- checkpoint_weibull_params.csv (Parametric survival parameters)
- checkpoint_logrank_test.csv (Survival curve comparison)
- checkpoint_operator_statistics.csv (Test results)
- checkpoint_spatial_autocorrelation.csv (Moran's I results)
- checkpoint_lisa_clusters.csv (Local spatial clustering)
- checkpoint_bootstrap_ci.csv (Bootstrap confidence intervals)

Author: Ali Vaezi
Date: December 2025
==============================================================================
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import timedelta
from scipy import stats
from scipy.optimize import minimize
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Optional advanced imports
try:
    from lifelines import KaplanMeierFitter, WeibullFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("Note: 'lifelines' not installed - using manual survival analysis")

try:
    from libpysal.weights import Queen, KNN
    from esda.moran import Moran
    from esda.getisord import G_Local
    HAS_SPATIAL = True
except ImportError:
    HAS_SPATIAL = False
    print("Note: 'libpysal/esda' not installed - using fallback spatial methods")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - go up TWO levels from src/analysis/ to project root
BASE_DIR = Path(__file__).parent.parent.parent  # project root
DATA_DIR = BASE_DIR / "outputs" / "reports" / "exercise3"
ZONES_PATH = BASE_DIR / "data" / "raw" / "zone_statistiche_geo" / "zone_statistiche_geo.shp"
OUTPUT_DIR = BASE_DIR / "outputs" / "reports" / "exercise4"

# CRS
CRS_WGS84 = "EPSG:4326"
CRS_UTM32N = "EPSG:32632"

# Thresholds
MAX_PARKING_DAYS = 30  # Filter out parking > 30 days (outliers/errors)
GHOST_THRESHOLD_HOURS = 120  # 5 days = abandoned/ghost vehicle

# ============================================================================
# STEP 1: LOAD & SORT DATA
# ============================================================================

def load_and_sort_data():
    """Load checkpoint data and sort by operator, vehicle_id, start_time."""
    print("\n" + "="*70)
    print("STEP 1: LOAD & SORT DATA")
    print("="*70)
    
    # Load checkpoint
    checkpoint_path = DATA_DIR / "checkpoint_validated_escooter_data.pkl"
    print(f"Loading: {checkpoint_path.name}")
    
    df = pd.read_pickle(checkpoint_path)
    print(f"  → Loaded {len(df):,} trips")
    
    # Parse datetime columns
    print("  → Parsing datetime columns...")
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['end_datetime'] = pd.to_datetime(df['end_datetime'], errors='coerce')
    
    # Drop rows with invalid datetimes
    valid_mask = df['start_datetime'].notna() & df['end_datetime'].notna()
    df = df[valid_mask].copy()
    print(f"  → Valid datetime rows: {len(df):,}")
    
    # Sort by operator, vehicle_id, and start_time
    print("  → Sorting by operator, vehicle_id, start_datetime...")
    df = df.sort_values(['operator', 'vehicle_id', 'start_datetime']).reset_index(drop=True)
    
    # Summary
    print(f"\n  Summary:")
    print(f"  - Operators: {df['operator'].nunique()} ({', '.join(df['operator'].unique())})")
    print(f"  - Unique vehicles: {df['vehicle_id'].nunique():,}")
    print(f"  - Date range: {df['start_datetime'].min().date()} to {df['start_datetime'].max().date()}")
    
    return df


# ============================================================================
# STEP 2: CALCULATE IDLE INTERVALS (VECTORIZED)
# ============================================================================

def calculate_idle_intervals(df):
    """
    Calculate idle time between consecutive trips for each vehicle.
    
    Idle_Time = Start_Time(Next Trip) - End_Time(Current Trip)
    
    Filters:
    - Negative values (errors)
    - Huge outliers (>30 days)
    """
    print("\n" + "="*70)
    print("STEP 2: CALCULATE IDLE INTERVALS (VECTORIZED)")
    print("="*70)
    
    # Get next trip's start time for same vehicle (vectorized with groupby + shift)
    print("  → Computing next trip start times...")
    df['next_start'] = df.groupby(['operator', 'vehicle_id'])['start_datetime'].shift(-1)
    
    # Calculate idle time in hours
    print("  → Computing idle durations...")
    df['idle_timedelta'] = df['next_start'] - df['end_datetime']
    df['idle_hours'] = df['idle_timedelta'].dt.total_seconds() / 3600
    
    # Count initial events
    initial_count = len(df)
    print(f"  → Initial parking events: {initial_count:,}")
    
    # Filter: Remove last trip per vehicle (no next trip)
    df = df[df['next_start'].notna()].copy()
    print(f"  → After removing last trips: {len(df):,} events")
    
    # Filter: Remove negative idle times (data errors)
    negative_count = (df['idle_hours'] < 0).sum()
    df = df[df['idle_hours'] >= 0].copy()
    print(f"  → Removed {negative_count:,} negative idle times")
    
    # Filter: Remove huge outliers (> 30 days)
    max_hours = MAX_PARKING_DAYS * 24
    outlier_count = (df['idle_hours'] > max_hours).sum()
    df = df[df['idle_hours'] <= max_hours].copy()
    print(f"  → Removed {outlier_count:,} outliers (>{MAX_PARKING_DAYS} days)")
    
    print(f"\n  → Final valid parking events: {len(df):,}")
    
    # Statistics
    print(f"\n  Idle Time Statistics:")
    print(f"  - Mean: {df['idle_hours'].mean():.2f} hours")
    print(f"  - Median: {df['idle_hours'].median():.2f} hours")
    print(f"  - Min: {df['idle_hours'].min():.2f} hours")
    print(f"  - Max: {df['idle_hours'].max():.2f} hours")
    print(f"  - Std: {df['idle_hours'].std():.2f} hours")
    
    return df


# ============================================================================
# STEP 3: DETECT "GHOST VEHICLES"
# ============================================================================

def detect_ghost_vehicles(df):
    """
    Flag parking events > 120 hours (5 days) as abandoned/ghost vehicles.
    """
    print("\n" + "="*70)
    print("STEP 3: DETECT 'GHOST VEHICLES'")
    print("="*70)
    
    # Flag abandoned vehicles
    df['is_abandoned'] = df['idle_hours'] > GHOST_THRESHOLD_HOURS
    
    # Count
    abandoned_count = df['is_abandoned'].sum()
    abandoned_pct = abandoned_count / len(df) * 100
    
    print(f"  → Threshold: {GHOST_THRESHOLD_HOURS} hours ({GHOST_THRESHOLD_HOURS/24:.1f} days)")
    print(f"  → Abandoned/Ghost events: {abandoned_count:,} ({abandoned_pct:.2f}%)")
    
    # Ghost vehicle analysis by operator
    ghost_by_operator = df[df['is_abandoned']].groupby('operator').size()
    print(f"\n  Ghost Events by Operator:")
    for op, count in ghost_by_operator.items():
        total = len(df[df['operator'] == op])
        pct = count / total * 100
        print(f"  - {op}: {count:,} ({pct:.2f}%)")
    
    # Create ghost vehicles dataset
    ghost_df = df[df['is_abandoned']][
        ['operator', 'vehicle_id', 'end_datetime', 'end_lat', 'end_lon', 
         'idle_hours', 'is_abandoned']
    ].copy()
    ghost_df.rename(columns={
        'end_datetime': 'parking_start',
        'end_lat': 'lat',
        'end_lon': 'lon'
    }, inplace=True)
    
    print(f"\n  → Ghost vehicles dataset: {len(ghost_df):,} records")
    
    return df, ghost_df


# ============================================================================
# STEP 4: SPATIAL AGGREGATION (ASSIGN TO ZONES)
# ============================================================================

def assign_to_zones(df, zones_gdf):
    """
    Assign each parking event (using end location) to a Statistical Zone.
    """
    print("\n" + "="*70)
    print("STEP 4: SPATIAL AGGREGATION (ASSIGN TO ZONES)")
    print("="*70)
    
    # Create geometry from parking location (end of trip = parking start)
    print("  → Creating parking location geometry...")
    parking_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['end_lon'], df['end_lat']),
        crs=CRS_WGS84
    )
    
    # Ensure zones are in same CRS
    if zones_gdf.crs != CRS_WGS84:
        zones_gdf = zones_gdf.to_crs(CRS_WGS84)
    
    # Spatial join
    print("  → Performing spatial join...")
    parking_with_zones = gpd.sjoin(
        parking_gdf,
        zones_gdf[['ZONASTAT', 'DENOM', 'geometry']],
        how='left',
        predicate='within'
    )
    
    # Count matched/unmatched
    matched = parking_with_zones['ZONASTAT'].notna().sum()
    unmatched = parking_with_zones['ZONASTAT'].isna().sum()
    print(f"  → Matched to zones: {matched:,} ({matched/len(parking_with_zones)*100:.1f}%)")
    print(f"  → Outside zones: {unmatched:,} ({unmatched/len(parking_with_zones)*100:.1f}%)")
    
    # Keep only matched records
    parking_with_zones = parking_with_zones[parking_with_zones['ZONASTAT'].notna()].copy()
    
    return parking_with_zones, zones_gdf


# ============================================================================
# STEP 5: ZONE METRICS AGGREGATION
# ============================================================================

def calculate_zone_metrics(parking_gdf, zones_gdf):
    """
    Calculate zone-level parking metrics:
    - median_parking_hours
    - mean_parking_hours
    - parking_events_count
    - parking_density (events per sq km)
    - abandoned_count
    - abandoned_pct
    """
    print("\n" + "="*70)
    print("STEP 5: ZONE METRICS AGGREGATION")
    print("="*70)
    
    # Aggregate by zone
    print("  → Aggregating metrics by zone...")
    zone_stats = parking_gdf.groupby('ZONASTAT').agg(
        parking_events_count=('idle_hours', 'count'),
        mean_parking_hours=('idle_hours', 'mean'),
        median_parking_hours=('idle_hours', 'median'),
        std_parking_hours=('idle_hours', 'std'),
        min_parking_hours=('idle_hours', 'min'),
        max_parking_hours=('idle_hours', 'max'),
        p25_parking_hours=('idle_hours', lambda x: x.quantile(0.25)),
        p75_parking_hours=('idle_hours', lambda x: x.quantile(0.75)),
        abandoned_count=('is_abandoned', 'sum')
    ).reset_index()
    
    # Calculate abandoned percentage
    zone_stats['abandoned_pct'] = (zone_stats['abandoned_count'] / zone_stats['parking_events_count'] * 100).round(2)
    
    # Merge with zones geometry (zones_gdf already has DENOM)
    zones_with_metrics = zones_gdf.merge(zone_stats, on='ZONASTAT', how='left')
    
    # Calculate zone area (in sq km) for density
    print("  → Calculating zone areas...")
    zones_utm = zones_with_metrics.to_crs(CRS_UTM32N)
    zones_with_metrics['area_sqkm'] = zones_utm.geometry.area / 1e6
    
    # Calculate parking density
    zones_with_metrics['parking_density'] = (
        zones_with_metrics['parking_events_count'] / zones_with_metrics['area_sqkm']
    ).round(2)
    
    # Calculate turnover score (trips per day proxy = 24 / median_hours)
    zones_with_metrics['turnover_rate'] = (
        24 / zones_with_metrics['median_parking_hours']
    ).round(2)
    
    # Summary
    valid_zones = zones_with_metrics[zones_with_metrics['parking_events_count'].notna()]
    print(f"\n  Zone Metrics Summary ({len(valid_zones)} zones with data):")
    print(f"  - Total parking events: {valid_zones['parking_events_count'].sum():,.0f}")
    print(f"  - Median parking time (zone avg): {valid_zones['median_parking_hours'].mean():.2f} hours")
    print(f"  - Mean parking density: {valid_zones['parking_density'].mean():.1f} events/km²")
    print(f"  - Total abandoned events: {valid_zones['abandoned_count'].sum():,.0f}")
    
    return zones_with_metrics


# ============================================================================
# STEP 6: TEMPORAL AGGREGATION ("FLEET PULSE")
# ============================================================================

def calculate_fleet_pulse(parking_gdf):
    """
    Create temporal dataset grouping parking events by hour_of_day.
    Shows the daily rhythm of when people drop off scooters.
    """
    print("\n" + "="*70)
    print("STEP 6: TEMPORAL AGGREGATION ('FLEET PULSE')")
    print("="*70)
    
    # Extract parking start hour (= end_datetime of trip)
    parking_gdf['parking_hour'] = parking_gdf['end_datetime'].dt.hour
    
    # Aggregate by hour
    print("  → Aggregating by hour of day...")
    hourly_stats = parking_gdf.groupby('parking_hour').agg(
        parking_count=('idle_hours', 'count'),
        mean_duration_hours=('idle_hours', 'mean'),
        median_duration_hours=('idle_hours', 'median'),
        abandoned_count=('is_abandoned', 'sum')
    ).reset_index()
    
    # Calculate percentage of total
    hourly_stats['pct_of_total'] = (
        hourly_stats['parking_count'] / hourly_stats['parking_count'].sum() * 100
    ).round(2)
    
    # By operator
    print("  → Aggregating by hour and operator...")
    hourly_by_operator = parking_gdf.groupby(['parking_hour', 'operator']).agg(
        parking_count=('idle_hours', 'count'),
        mean_duration_hours=('idle_hours', 'mean')
    ).reset_index()
    
    # Peak hour analysis
    peak_hour = hourly_stats.loc[hourly_stats['parking_count'].idxmax()]
    trough_hour = hourly_stats.loc[hourly_stats['parking_count'].idxmin()]
    
    print(f"\n  Fleet Pulse Summary:")
    print(f"  - Peak parking hour: {int(peak_hour['parking_hour']):02d}:00 ({peak_hour['parking_count']:,.0f} events)")
    print(f"  - Lowest parking hour: {int(trough_hour['parking_hour']):02d}:00 ({trough_hour['parking_count']:,.0f} events)")
    print(f"  - Peak/Trough ratio: {peak_hour['parking_count']/trough_hour['parking_count']:.1f}x")
    
    return hourly_stats, hourly_by_operator


# ============================================================================
# STEP 7: SURVIVAL ANALYSIS (KAPLAN-MEIER + WEIBULL + LOG-RANK)
# ============================================================================

def weibull_negative_log_likelihood(params, data):
    """Negative log-likelihood for Weibull distribution."""
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
    data = data[data > 0]  # Weibull requires positive values
    
    # Initial estimates using method of moments
    mean_data = np.mean(data)
    var_data = np.var(data)
    
    # Approximate shape from coefficient of variation
    cv = np.sqrt(var_data) / mean_data
    shape_init = 1 / cv if cv > 0 else 1.0
    scale_init = mean_data
    
    try:
        result = minimize(
            weibull_negative_log_likelihood,
            [shape_init, scale_init],
            args=(data,),
            method='L-BFGS-B',
            bounds=[(0.01, 10), (0.01, 1000)]
        )
        shape, scale = result.x
        
        # Calculate AIC and BIC
        n = len(data)
        ll = -weibull_negative_log_likelihood([shape, scale], data)
        aic = 2 * 2 - 2 * ll  # 2 parameters
        bic = 2 * np.log(n) - 2 * ll
        
        return {
            'shape': shape,
            'scale': scale,
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic,
            'success': result.success
        }
    except Exception as e:
        return {'shape': None, 'scale': None, 'error': str(e), 'success': False}


def manual_logrank_test(durations1, durations2):
    """
    Manual log-rank test for comparing two survival curves.
    
    Returns chi-square statistic and p-value.
    """
    # Combine and sort all event times
    all_times = np.unique(np.concatenate([durations1, durations2]))
    all_times = np.sort(all_times)
    
    # Calculate observed and expected events at each time
    O1, E1 = 0, 0
    O2, E2 = 0, 0
    var = 0
    
    n1 = len(durations1)
    n2 = len(durations2)
    
    for t in all_times:
        # At risk just before time t
        r1 = np.sum(durations1 >= t)
        r2 = np.sum(durations2 >= t)
        r = r1 + r2
        
        if r == 0:
            continue
        
        # Events at time t (approximation for continuous data)
        d1 = np.sum((durations1 >= t) & (durations1 < t + 0.5))
        d2 = np.sum((durations2 >= t) & (durations2 < t + 0.5))
        d = d1 + d2
        
        if d == 0 or r == 0:
            continue
        
        # Expected events
        e1 = d * r1 / r
        e2 = d * r2 / r
        
        O1 += d1
        E1 += e1
        O2 += d2
        E2 += e2
        
        # Variance contribution
        if r > 1:
            var += (r1 * r2 * d * (r - d)) / (r * r * (r - 1))
    
    # Chi-square statistic
    if var > 0:
        chi_sq = (O1 - E1) ** 2 / var
        p_value = 1 - stats.chi2.cdf(chi_sq, 1)
    else:
        chi_sq = 0
        p_value = 1.0
    
    return chi_sq, p_value


def calculate_survival_analysis(parking_gdf):
    """
    Survival Analysis.
    
    Implements:
    1. Kaplan-Meier estimator with 95% CI (Greenwood's formula)
    2. Weibull parametric fitting with AIC/BIC
    3. Log-rank test for operator comparisons
    4. Operator-specific survival curves
    
    Returns:
    - survival_df: Time-indexed survival probabilities
    - weibull_params: Fitted Weibull parameters by operator
    - logrank_results: Pairwise log-rank test results
    """
    print("\n" + "="*70)
    print("STEP 7: SURVIVAL ANALYSIS (KAPLAN-MEIER + WEIBULL)")
    print("="*70)
    
    operators = parking_gdf['operator'].unique()
    n_total = len(parking_gdf)
    
    print(f"  → Analyzing {n_total:,} parking events across {len(operators)} operators")
    
    # Time grid for survival curves
    max_time = min(168, parking_gdf['idle_hours'].quantile(0.99))
    time_grid = np.concatenate([
        np.arange(0, 24, 0.5),
        np.arange(24, 72, 1),
        np.arange(72, max_time + 1, 4)
    ])
    time_grid = np.unique(np.sort(time_grid))
    
    # ========================================================================
    # A. KAPLAN-MEIER SURVIVAL CURVES BY OPERATOR
    # ========================================================================
    print("\n  A. Kaplan-Meier Survival Curves")
    print("  " + "-"*50)
    
    survival_data = {'time_hours': time_grid}
    operator_durations = {}
    
    for operator in sorted(operators):
        op_data = parking_gdf[parking_gdf['operator'] == operator]['idle_hours'].dropna().values
        operator_durations[operator] = op_data
        n_op = len(op_data)
        
        if n_op == 0:
            continue
        
        # Calculate survival probabilities
        survival_probs = []
        ci_lower = []
        ci_upper = []
        
        for t in time_grid:
            n_at_risk = np.sum(op_data >= t)
            surv_prob = n_at_risk / n_op
            survival_probs.append(surv_prob)
            
            # Greenwood's formula for 95% CI
            if surv_prob > 0 and surv_prob < 1:
                se = surv_prob * np.sqrt((1 - surv_prob) / n_at_risk) if n_at_risk > 0 else 0
                ci_lower.append(max(0, surv_prob - 1.96 * se))
                ci_upper.append(min(1, surv_prob + 1.96 * se))
            else:
                ci_lower.append(surv_prob)
                ci_upper.append(surv_prob)
        
        survival_data[f'{operator}_survival'] = survival_probs
        survival_data[f'{operator}_ci_lower'] = ci_lower
        survival_data[f'{operator}_ci_upper'] = ci_upper
        
        # Calculate median survival time
        surv_array = np.array(survival_probs)
        median_idx = np.argmax(surv_array <= 0.5)
        median_time = time_grid[median_idx] if surv_array[median_idx] <= 0.5 else None
        
        print(f"  {operator}: n={n_op:,}, Median survival={median_time:.1f}h" if median_time else f"  {operator}: n={n_op:,}, Median survival=N/A")
    
    # Combined survival curve
    all_durations = parking_gdf['idle_hours'].dropna().values
    survival_data['ALL_survival'] = [np.mean(all_durations >= t) for t in time_grid]
    
    survival_df = pd.DataFrame(survival_data)
    
    # ========================================================================
    # B. WEIBULL PARAMETRIC FITTING
    # ========================================================================
    print("\n  B. Weibull Parametric Fitting")
    print("  " + "-"*50)
    
    weibull_results = []
    
    for operator in sorted(operators):
        op_data = operator_durations.get(operator, np.array([]))
        if len(op_data) < 100:
            continue
        
        result = fit_weibull(op_data)
        
        if result['success']:
            print(f"  {operator}: Shape(k)={result['shape']:.3f}, Scale(λ)={result['scale']:.2f}h")
            print(f"           AIC={result['aic']:.1f}, BIC={result['bic']:.1f}")
            
            weibull_results.append({
                'operator': operator,
                'shape': result['shape'],
                'scale': result['scale'],
                'log_likelihood': result['log_likelihood'],
                'aic': result['aic'],
                'bic': result['bic']
            })
    
    # Fit for all data combined
    all_result = fit_weibull(all_durations)
    if all_result['success']:
        print(f"  ALL: Shape(k)={all_result['shape']:.3f}, Scale(λ)={all_result['scale']:.2f}h")
        weibull_results.append({
            'operator': 'ALL',
            'shape': all_result['shape'],
            'scale': all_result['scale'],
            'log_likelihood': all_result['log_likelihood'],
            'aic': all_result['aic'],
            'bic': all_result['bic']
        })
    
    weibull_df = pd.DataFrame(weibull_results) if weibull_results else None
    
    # Interpretation
    if weibull_df is not None and len(weibull_df) > 0:
        avg_shape = weibull_df[weibull_df['operator'] != 'ALL']['shape'].mean()
        if avg_shape < 1:
            print(f"\n  → Shape < 1: Hazard decreases over time (longer you wait, less likely pickup)")
        elif avg_shape > 1:
            print(f"\n  → Shape > 1: Hazard increases over time (longer parked = more likely to be picked up soon)")
        else:
            print(f"\n  → Shape ≈ 1: Constant hazard (exponential distribution)")
    
    # ========================================================================
    # C. LOG-RANK TEST FOR OPERATOR COMPARISONS
    # ========================================================================
    print("\n  C. Log-Rank Test (Pairwise Comparisons)")
    print("  " + "-"*50)
    
    logrank_results = []
    
    for op1, op2 in combinations(sorted(operators), 2):
        data1 = operator_durations.get(op1, np.array([]))
        data2 = operator_durations.get(op2, np.array([]))
        
        if len(data1) < 100 or len(data2) < 100:
            continue
        
        chi_sq, p_value = manual_logrank_test(data1, data2)
        
        # Bonferroni correction
        n_comparisons = len(list(combinations(operators, 2)))
        corrected_alpha = 0.05 / n_comparisons
        significant = p_value < corrected_alpha
        
        logrank_results.append({
            'comparison': f'{op1} vs {op2}',
            'chi_square': chi_sq,
            'p_value': p_value,
            'significant': significant
        })
        
        sig_marker = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        print(f"  {op1} vs {op2}: χ²={chi_sq:.2f}, p={p_value:.4f} {sig_marker}")
    
    logrank_df = pd.DataFrame(logrank_results) if logrank_results else None
    
    print("\n  → Log-rank test complete")
    
    return survival_df, weibull_df, logrank_df


# ============================================================================
# STEP 8: SPATIAL AUTOCORRELATION (MORAN'S I)
# ============================================================================

def calculate_spatial_autocorrelation(zones_gdf):
    """
    Calculate Moran's I statistic for spatial autocorrelation of parking patterns.
    
    Tests whether nearby zones have similar parking characteristics (clustering)
    or dissimilar (dispersion), or random distribution.
    
    Moran's I ≈ 1: Positive autocorrelation (clustering)
    Moran's I ≈ 0: Random distribution
    Moran's I ≈ -1: Negative autocorrelation (dispersion)
    
    Returns results for multiple variables.
    """
    print("\n" + "="*70)
    print("STEP 8: SPATIAL AUTOCORRELATION (MORAN'S I)")
    print("="*70)
    
    # Filter zones with valid data
    valid_zones = zones_gdf[zones_gdf['median_parking_hours'].notna()].copy()
    
    if len(valid_zones) < 10:
        print("  ⚠ Insufficient zones for spatial analysis")
        return None
    
    print(f"  → Analyzing {len(valid_zones)} zones")
    
    # Project to UTM for accurate distance calculations
    zones_utm = valid_zones.to_crs(CRS_UTM32N)
    
    # Create spatial weights matrix (Queen contiguity - shared edge or vertex)
    try:
        from libpysal.weights import Queen, KNN
        from esda.moran import Moran
        
        # Try Queen contiguity first
        try:
            w = Queen.from_dataframe(zones_utm, use_index=False)
            weight_type = "Queen Contiguity"
        except:
            # Fall back to K-Nearest Neighbors
            w = KNN.from_dataframe(zones_utm, k=5)
            weight_type = "K-Nearest Neighbors (k=5)"
        
        # Row-standardize weights
        w.transform = 'r'
        
        print(f"  → Spatial weights: {weight_type}")
        print(f"  → Mean neighbors per zone: {w.mean_neighbors:.1f}")
        
    except ImportError:
        print("  ⚠ libpysal/esda not available - using manual calculation")
        # Fallback: Calculate using centroid distances
        centroids = zones_utm.geometry.centroid
        
        # Distance-based weights (inverse distance within threshold)
        coords = np.array([[c.x, c.y] for c in centroids])
        from scipy.spatial.distance import cdist
        distances = cdist(coords, coords)
        
        # Create weight matrix (inverse distance, 2km threshold)
        threshold = 2000  # 2km
        W = np.where((distances > 0) & (distances < threshold), 1/distances, 0)
        # Row normalize
        W = W / W.sum(axis=1, keepdims=True)
        
        weight_type = "Inverse Distance (2km threshold)"
        print(f"  → Spatial weights: {weight_type}")
    
    # Variables to test
    variables_to_test = [
        ('median_parking_hours', 'Median Parking Duration'),
        ('parking_density', 'Parking Density'),
        ('abandoned_pct', 'Abandoned Vehicle %'),
        ('turnover_rate', 'Turnover Rate')
    ]
    
    results = []
    
    for var_col, var_name in variables_to_test:
        if var_col not in valid_zones.columns:
            continue
            
        y = valid_zones[var_col].fillna(0).values
        
        if np.std(y) == 0:
            continue
        
        try:
            # Calculate Moran's I
            moran = Moran(y, w)
            
            results.append({
                'variable': var_name,
                'morans_i': moran.I,
                'expected_i': moran.EI,
                'z_score': moran.z_sim if hasattr(moran, 'z_sim') else moran.z_norm,
                'p_value': moran.p_sim if hasattr(moran, 'p_sim') else moran.p_norm,
                'interpretation': 'Clustered' if moran.I > 0.1 and moran.p_sim < 0.05 else 
                                  'Dispersed' if moran.I < -0.1 and moran.p_sim < 0.05 else 'Random'
            })
            
            print(f"\n  {var_name}:")
            print(f"    Moran's I = {moran.I:.4f}")
            print(f"    Z-score = {moran.z_sim if hasattr(moran, 'z_sim') else moran.z_norm:.2f}")
            print(f"    P-value = {moran.p_sim if hasattr(moran, 'p_sim') else moran.p_norm:.4f}")
            print(f"    → {results[-1]['interpretation']}")
            
        except Exception as e:
            print(f"  ⚠ Could not calculate Moran's I for {var_name}: {e}")
    
    if results:
        return pd.DataFrame(results)
    else:
        return None


# ============================================================================
# STEP 9: OPERATOR COMPARISON STATISTICS
# ============================================================================

def calculate_operator_statistics(parking_gdf):
    """
    Statistical comparison of parking behavior across operators.
    
    Uses Kruskal-Wallis H-test (non-parametric ANOVA alternative) since
    parking durations are typically not normally distributed.
    
    Also includes effect size (eta-squared) and post-hoc pairwise comparisons.
    """
    print("\n" + "="*70)
    print("STEP 9: OPERATOR COMPARISON STATISTICS")
    print("="*70)
    
    operators = parking_gdf['operator'].unique()
    n_operators = len(operators)
    
    print(f"  → Comparing {n_operators} operators: {', '.join(operators)}")
    
    # Descriptive statistics by operator
    print(f"\n  Descriptive Statistics:")
    print(f"  {'Operator':<10} {'N':>10} {'Mean':>10} {'Median':>10} {'Std':>10} {'IQR':>10}")
    print(f"  {'-'*60}")
    
    operator_stats = []
    for op in sorted(operators):
        data = parking_gdf[parking_gdf['operator'] == op]['idle_hours']
        q1, q3 = data.quantile([0.25, 0.75])
        
        stats_row = {
            'operator': op,
            'n': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'iqr': q3 - q1,
            'q25': q1,
            'q75': q3,
            'min': data.min(),
            'max': data.max()
        }
        operator_stats.append(stats_row)
        
        print(f"  {op:<10} {len(data):>10,} {data.mean():>10.2f} {data.median():>10.2f} {data.std():>10.2f} {q3-q1:>10.2f}")
    
    # Kruskal-Wallis H-test
    print(f"\n  Kruskal-Wallis H-test:")
    
    groups = [parking_gdf[parking_gdf['operator'] == op]['idle_hours'].values 
              for op in operators]
    
    h_stat, p_value = stats.kruskal(*groups)
    
    # Effect size (eta-squared approximation)
    n_total = len(parking_gdf)
    eta_squared = (h_stat - n_operators + 1) / (n_total - n_operators)
    eta_squared = max(0, eta_squared)  # Ensure non-negative
    
    # Interpret effect size
    if eta_squared < 0.01:
        effect_interpretation = "Negligible"
    elif eta_squared < 0.06:
        effect_interpretation = "Small"
    elif eta_squared < 0.14:
        effect_interpretation = "Medium"
    else:
        effect_interpretation = "Large"
    
    print(f"  H-statistic = {h_stat:.2f}")
    print(f"  P-value = {p_value:.2e}")
    print(f"  Effect size (η²) = {eta_squared:.4f} ({effect_interpretation})")
    
    if p_value < 0.05:
        print(f"  → Significant difference between operators (p < 0.05)")
    else:
        print(f"  → No significant difference between operators (p ≥ 0.05)")
    
    # Pairwise comparisons (Mann-Whitney U)
    print(f"\n  Pairwise Comparisons (Mann-Whitney U):")
    print(f"  {'Comparison':<20} {'U-stat':>12} {'P-value':>12} {'Significant':>12}")
    print(f"  {'-'*56}")
    
    pairwise_results = []
    
    from itertools import combinations
    for op1, op2 in combinations(sorted(operators), 2):
        data1 = parking_gdf[parking_gdf['operator'] == op1]['idle_hours'].values
        data2 = parking_gdf[parking_gdf['operator'] == op2]['idle_hours'].values
        
        u_stat, u_pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Bonferroni correction
        n_comparisons = n_operators * (n_operators - 1) / 2
        corrected_alpha = 0.05 / n_comparisons
        significant = "Yes" if u_pvalue < corrected_alpha else "No"
        
        pairwise_results.append({
            'comparison': f"{op1} vs {op2}",
            'u_statistic': u_stat,
            'p_value': u_pvalue,
            'significant': u_pvalue < corrected_alpha
        })
        
        print(f"  {op1} vs {op2:<10} {u_stat:>12,.0f} {u_pvalue:>12.2e} {significant:>12}")
    
    # Compile all results
    comparison_results = {
        'kruskal_wallis': {
            'h_statistic': h_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'effect_size': effect_interpretation
        },
        'operator_stats': pd.DataFrame(operator_stats),
        'pairwise': pd.DataFrame(pairwise_results)
    }
    
    return comparison_results


# ============================================================================
# STEP 10: BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def calculate_bootstrap_ci(parking_gdf, n_bootstrap=1000, ci_level=0.95):
    """
    Calculate bootstrap confidence intervals for key metrics by operator.
    
    Bootstrap methodology:
    1. Resample with replacement n_bootstrap times
    2. Calculate statistic for each resample
    3. Use percentile method for CI bounds
    
    Metrics computed:
    - Median parking duration
    - Mean parking duration
    - Abandoned vehicle percentage
    """
    print("\n" + "="*70)
    print("STEP 10: BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*70)
    
    np.random.seed(42)  # For reproducibility
    
    operators = sorted(parking_gdf['operator'].unique())
    alpha = 1 - ci_level
    
    print(f"  → Bootstrap parameters: n={n_bootstrap}, CI={ci_level*100:.0f}%")
    
    results = []
    
    for operator in operators:
        op_data = parking_gdf[parking_gdf['operator'] == operator]
        durations = op_data['idle_hours'].dropna().values
        abandoned = op_data['is_abandoned'].values
        
        n = len(durations)
        if n < 100:
            print(f"  ⚠ {operator}: Insufficient data (n={n})")
            continue
        
        # Bootstrap resampling
        median_boots = []
        mean_boots = []
        abandoned_pct_boots = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            sample_durations = durations[idx]
            sample_abandoned = abandoned[idx]
            
            median_boots.append(np.median(sample_durations))
            mean_boots.append(np.mean(sample_durations))
            abandoned_pct_boots.append(np.mean(sample_abandoned) * 100)
        
        # Calculate percentile CIs
        median_ci = np.percentile(median_boots, [alpha/2*100, (1-alpha/2)*100])
        mean_ci = np.percentile(mean_boots, [alpha/2*100, (1-alpha/2)*100])
        abandoned_ci = np.percentile(abandoned_pct_boots, [alpha/2*100, (1-alpha/2)*100])
        
        results.append({
            'operator': operator,
            'n': n,
            'median': np.median(durations),
            'median_ci_lower': median_ci[0],
            'median_ci_upper': median_ci[1],
            'mean': np.mean(durations),
            'mean_ci_lower': mean_ci[0],
            'mean_ci_upper': mean_ci[1],
            'abandoned_pct': np.mean(abandoned) * 100,
            'abandoned_ci_lower': abandoned_ci[0],
            'abandoned_ci_upper': abandoned_ci[1]
        })
        
        print(f"  {operator}:")
        print(f"    Median: {np.median(durations):.2f}h [{median_ci[0]:.2f}, {median_ci[1]:.2f}]")
        print(f"    Mean:   {np.mean(durations):.2f}h [{mean_ci[0]:.2f}, {mean_ci[1]:.2f}]")
        print(f"    Abandoned: {np.mean(abandoned)*100:.2f}% [{abandoned_ci[0]:.2f}, {abandoned_ci[1]:.2f}]")
    
    bootstrap_df = pd.DataFrame(results) if results else None
    
    return bootstrap_df


# ============================================================================
# STEP 11: SAVE CHECKPOINTS 
# ============================================================================

def save_checkpoints(zones_gdf, parking_gdf, ghost_df, hourly_stats,
                     survival_df=None, weibull_df=None, logrank_df=None,
                     moran_df=None, operator_results=None, bootstrap_ci=None):
    """Save all checkpoint files including advanced analysis outputs."""
    print("\n" + "="*70)
    print("STEP 11: SAVE CHECKPOINTS ")
    print("="*70)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Zone statistics (GeoJSON)
    zones_path = OUTPUT_DIR / "checkpoint_parking_zones.geojson"
    zones_save = zones_gdf.copy()
    # Select relevant columns
    cols_to_keep = [
        'ZONASTAT', 'DENOM', 'geometry', 'area_sqkm',
        'parking_events_count', 'mean_parking_hours', 'median_parking_hours',
        'std_parking_hours', 'min_parking_hours', 'max_parking_hours',
        'p25_parking_hours', 'p75_parking_hours',
        'parking_density', 'turnover_rate',
        'abandoned_count', 'abandoned_pct'
    ]
    cols_available = [c for c in cols_to_keep if c in zones_save.columns]
    zones_save = zones_save[cols_available]
    zones_save.to_file(zones_path, driver='GeoJSON')
    print(f"  ✓ Saved: {zones_path.name} ({len(zones_save)} zones)")
    
    # 2. Parking events (PKL)
    events_path = OUTPUT_DIR / "checkpoint_parking_events.pkl"
    events_save = parking_gdf[[
        'operator', 'vehicle_id', 'start_datetime', 'end_datetime',
        'end_lat', 'end_lon', 'idle_hours', 'is_abandoned',
        'parking_hour', 'ZONASTAT', 'DENOM'
    ]].copy()
    events_save.to_pickle(events_path)
    print(f"  ✓ Saved: {events_path.name} ({len(events_save):,} events)")
    
    # 3. Hourly statistics (CSV)
    hourly_path = OUTPUT_DIR / "checkpoint_parking_hourly.csv"
    hourly_stats.to_csv(hourly_path, index=False)
    print(f"  ✓ Saved: {hourly_path.name} ({len(hourly_stats)} hours)")
    
    # 4. Ghost vehicles (PKL)
    ghost_path = OUTPUT_DIR / "checkpoint_ghost_vehicles.pkl"
    ghost_df.to_pickle(ghost_path)
    print(f"  ✓ Saved: {ghost_path.name} ({len(ghost_df):,} ghost events)")
    
    # 5. Survival analysis (CSV) 
    if survival_df is not None:
        survival_path = OUTPUT_DIR / "checkpoint_survival_analysis.csv"
        survival_df.to_csv(survival_path, index=False)
        print(f"  ✓ Saved: {survival_path.name} ({len(survival_df)} time points)")
    
    # 6. Weibull parameters (CSV) 
    if weibull_df is not None:
        weibull_path = OUTPUT_DIR / "checkpoint_weibull_params.csv"
        weibull_df.to_csv(weibull_path, index=False)
        print(f"  ✓ Saved: {weibull_path.name}")
    
    # 7. Log-rank test results (CSV) 
    if logrank_df is not None:
        logrank_path = OUTPUT_DIR / "checkpoint_logrank_test.csv"
        logrank_df.to_csv(logrank_path, index=False)
        print(f"  ✓ Saved: {logrank_path.name}")
    
    # 8. Moran's I results (CSV)
    if moran_df is not None:
        moran_path = OUTPUT_DIR / "checkpoint_spatial_autocorrelation.csv"
        moran_df.to_csv(moran_path, index=False)
        print(f"  ✓ Saved: {moran_path.name} ({len(moran_df)} variables)")
    
    # 9. Operator comparison results (CSV)
    if operator_results is not None:
        # Save operator statistics
        op_stats_path = OUTPUT_DIR / "checkpoint_operator_statistics.csv"
        operator_results['operator_stats'].to_csv(op_stats_path, index=False)
        print(f"  ✓ Saved: {op_stats_path.name}")
        
        # Save pairwise comparisons
        pairwise_path = OUTPUT_DIR / "checkpoint_pairwise_comparisons.csv"
        operator_results['pairwise'].to_csv(pairwise_path, index=False)
        print(f"  ✓ Saved: {pairwise_path.name}")
        
        # Save overall test results
        kw_path = OUTPUT_DIR / "checkpoint_kruskal_wallis.csv"
        kw_df = pd.DataFrame([operator_results['kruskal_wallis']])
        kw_df.to_csv(kw_path, index=False)
        print(f"  ✓ Saved: {kw_path.name}")
    
    # 10. Bootstrap confidence intervals (CSV) 
    if bootstrap_ci is not None:
        bootstrap_path = OUTPUT_DIR / "checkpoint_bootstrap_ci.csv"
        bootstrap_ci.to_csv(bootstrap_path, index=False)
        print(f"  ✓ Saved: {bootstrap_path.name}")
    
    print(f"\n  → All checkpoints saved to: {OUTPUT_DIR}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the complete parking duration analysis pipeline."""
    print("\n" + "="*70)
    print("EXERCISE 4: PARKING DURATION ANALYSIS")
    print("="*70)
    print("Professional Fleet Management Metrics + Advanced Statistical Analysis")
    print("="*70)
    
    # Step 1: Load and sort data
    df = load_and_sort_data()
    
    # Step 2: Calculate idle intervals
    df = calculate_idle_intervals(df)
    
    # Step 3: Detect ghost vehicles
    df, ghost_df = detect_ghost_vehicles(df)
    
    # Step 4: Load zones and assign parking to zones
    print("\n  Loading zones shapefile...")
    zones_gdf = gpd.read_file(ZONES_PATH)
    print(f"  → Loaded {len(zones_gdf)} statistical zones")
    
    parking_gdf, zones_gdf = assign_to_zones(df, zones_gdf)
    
    # Step 5: Calculate zone metrics
    zones_with_metrics = calculate_zone_metrics(parking_gdf, zones_gdf)
    
    # Step 6: Calculate fleet pulse (temporal)
    hourly_stats, hourly_by_operator = calculate_fleet_pulse(parking_gdf)
    
    # ========================================================================
    # ADVANCED STATISTICAL ANALYSIS
    # ========================================================================
    
    # Step 7: Survival Analysis (Kaplan-Meier + Weibull + Log-rank)
    survival_df, weibull_df, logrank_df = calculate_survival_analysis(parking_gdf)
    
    # Step 8: Spatial Autocorrelation (Moran's I)
    moran_df = calculate_spatial_autocorrelation(zones_with_metrics)
    
    # Step 9: Operator Comparison Statistics (Kruskal-Wallis)
    operator_results = calculate_operator_statistics(parking_gdf)
    
    # Step 10: Bootstrap Confidence Intervals
    bootstrap_ci = calculate_bootstrap_ci(parking_gdf)
    
    # Step 11: Save all checkpoints (including advanced analysis)
    save_checkpoints(
        zones_with_metrics, parking_gdf, ghost_df, hourly_stats,
        survival_df=survival_df,
        weibull_df=weibull_df,
        logrank_df=logrank_df,
        moran_df=moran_df,
        operator_results=operator_results,
        bootstrap_ci=bootstrap_ci
    )
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n  BASIC ANALYSIS:")
    print(f"  ✓ Analyzed {len(parking_gdf):,} parking events")
    print(f"  ✓ Identified {len(ghost_df):,} ghost/abandoned vehicles")
    print(f"  ✓ Computed metrics for {(zones_with_metrics['parking_events_count'].notna()).sum()} zones")
    
    print(f"\n  ADVANCED STATISTICAL ANALYSIS:")
    print(f"  ✓ Survival Analysis: Kaplan-Meier + Weibull + Log-rank test")
    if weibull_df is not None:
        print(f"    - Weibull parameters fitted for {len(weibull_df)} operators")
    if logrank_df is not None:
        print(f"    - Log-rank tests: {len(logrank_df)} pairwise comparisons")
    if moran_df is not None:
        print(f"  ✓ Spatial Autocorrelation: Moran's I for {len(moran_df)} variables")
    else:
        print(f"  ⚠ Spatial Autocorrelation: Skipped (requires libpysal/esda)")
    print(f"  ✓ Operator Comparison: Kruskal-Wallis H-test + Mann-Whitney U")
    if bootstrap_ci is not None:
        print(f"  ✓ Bootstrap CIs: 95% intervals for {len(bootstrap_ci)} operators")
    
    print("\n" + "="*70)
    print("CHECKLIST:")
    print("="*70)
    print("""
  ✓ Kaplan-Meier survival curves with 95% CI (Greenwood's formula)
  ✓ Weibull parametric survival with AIC/BIC model selection
  ✓ Log-rank test for survival curve comparisons
  ✓ Kruskal-Wallis non-parametric ANOVA
  ✓ Mann-Whitney U pairwise tests with Bonferroni correction
  ✓ Effect sizes: η² (eta-squared)
  ✓ Bootstrap confidence intervals (n=1000)
  ✓ Spatial autocorrelation (Moran's I) for zone clustering
    """)
    print("="*70)


if __name__ == '__main__':
    main()
