#!/usr/bin/env python3
"""
==============================================================================
EXERCISE 1: E-SCOOTER TEMPORAL ANALYSIS
==============================================================================

This module performs comprehensive temporal and descriptive analysis of 
e-scooter trip data from three operators (LIME, VOI, BIRD) in Turin, Italy.

ACADEMIC METHODOLOGY:
    1. Data Cleaning & Quality Assessment
       - Multi-source data harmonization
       - Geographic and temporal filtering
       - Quality metrics calculation
    
    2. Descriptive Statistics
       - Central tendency with confidence intervals
       - Effect sizes (Cohen's d, eta-squared)
       - Non-parametric tests (Kruskal-Wallis, Mann-Whitney U)
    
    3. Temporal Pattern Analysis
       - Seasonal decomposition (STL)
       - Peak detection (scipy signal processing)
       - Trend analysis with statistical significance
    
    4. Cross-Operator Statistical Comparison
       - Chi-square tests for independence
       - Cramér's V effect size
       - Bootstrap confidence intervals

OUTPUT FILES:
    - Cleaned CSV files (per operator)
    - Checkpoint pickle files for visualization
    - Statistical summary reports

STATISTICAL TESTS:
    - Kruskal-Wallis H-test: Compare distributions across 3+ groups
    - Mann-Whitney U: Pairwise non-parametric comparison
    - Chi-square: Categorical variable independence
    - Levene's test: Homogeneity of variances
    - Effect sizes: Cohen's d, eta-squared, Cramér's V

REFERENCES:
    - Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
    - Cleveland, R. B. et al. (1990). STL: A Seasonal-Trend Decomposition
    - Hollander, M. et al. (2013). Nonparametric Statistical Methods

Author: Ali Vaezi
Date: December 2025
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Statistical imports
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, levene

# Optional: Advanced time series decomposition
try:
    from statsmodels.tsa.seasonal import STL
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Note: statsmodels not installed. STL decomposition will be skipped.")

# Get the project root directory (TWO levels up from src/analysis/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)  # src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # project root

# Define data paths
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUTS_FIGURES = os.path.join(PROJECT_ROOT, 'outputs', 'figures', 'exercise1')
OUTPUTS_REPORTS = os.path.join(PROJECT_ROOT, 'outputs', 'reports', 'exercise1')
OUTPUTS_TABLES = os.path.join(PROJECT_ROOT, 'outputs', 'tables', 'exercise1')

# Create output directories
os.makedirs(OUTPUTS_FIGURES, exist_ok=True)
os.makedirs(OUTPUTS_REPORTS, exist_ok=True)
os.makedirs(OUTPUTS_TABLES, exist_ok=True)
os.makedirs(os.path.join(OUTPUTS_FIGURES), exist_ok=True)


# ==============================================================================
# STATISTICAL FUNCTIONS
# ==============================================================================

def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size for two groups.
    
    d = (M1 - M2) / pooled_std
    
    Interpretation (Cohen, 1988):
        |d| < 0.2: negligible
        0.2 <= |d| < 0.5: small
        0.5 <= |d| < 0.8: medium
        |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def calculate_eta_squared(h_statistic: float, n_total: int, k_groups: int) -> float:
    """
    Calculate eta-squared effect size from Kruskal-Wallis H statistic.
    
    η² = (H - k + 1) / (n - k)
    
    Interpretation:
        η² < 0.01: negligible
        0.01 <= η² < 0.06: small
        0.06 <= η² < 0.14: medium
        η² >= 0.14: large
    """
    if n_total <= k_groups:
        return 0.0
    return (h_statistic - k_groups + 1) / (n_total - k_groups)


def calculate_cramers_v(contingency_table: np.ndarray) -> float:
    """
    Calculate Cramér's V for chi-square test effect size.
    
    V = sqrt(χ² / (n * min(r-1, c-1)))
    
    Interpretation:
        V < 0.1: negligible
        0.1 <= V < 0.3: small
        0.3 <= V < 0.5: medium
        V >= 0.5: large
    """
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum()
    min_dim = min(contingency_table.shape) - 1
    
    if n == 0 or min_dim == 0:
        return 0.0
    
    return np.sqrt(chi2 / (n * min_dim))


def bootstrap_ci(data: np.ndarray, statistic_func=np.mean, 
                 n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for any statistic.
    
    Returns (lower, upper) bounds of the confidence interval.
    """
    n = len(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)
    
    return lower, upper


def perform_kruskal_wallis_test(groups: Dict[str, np.ndarray]) -> Dict:
    """
    Perform Kruskal-Wallis H-test with effect size and post-hoc analysis.
    
    Non-parametric alternative to one-way ANOVA.
    """
    group_names = list(groups.keys())
    group_values = list(groups.values())
    
    # Main test
    h_stat, p_value = kruskal(*group_values)
    
    # Effect size
    n_total = sum(len(g) for g in group_values)
    k_groups = len(groups)
    eta_sq = calculate_eta_squared(h_stat, n_total, k_groups)
    
    # Post-hoc pairwise Mann-Whitney U tests
    pairwise_results = []
    for i, name1 in enumerate(group_names):
        for j, name2 in enumerate(group_names):
            if i < j:
                u_stat, u_pval = mannwhitneyu(groups[name1], groups[name2], 
                                              alternative='two-sided')
                cohens_d = calculate_cohens_d(groups[name1], groups[name2])
                pairwise_results.append({
                    'comparison': f"{name1} vs {name2}",
                    'U_statistic': u_stat,
                    'p_value': u_pval,
                    'cohens_d': cohens_d,
                    'significant': u_pval < 0.05
                })
    
    return {
        'H_statistic': h_stat,
        'p_value': p_value,
        'eta_squared': eta_sq,
        'significant': p_value < 0.05,
        'pairwise_comparisons': pairwise_results
    }


def detect_temporal_peaks(time_series: pd.Series, prominence: float = 0.1) -> Dict:
    """
    Detect peaks in temporal data using scipy signal processing.
    
    Uses Savitzky-Golay filter for smoothing and find_peaks for detection.
    """
    values = time_series.values.astype(float)
    
    # Smooth data
    if len(values) >= 7:
        smoothed = savgol_filter(values, window_length=min(7, len(values) // 2 * 2 + 1), 
                                  polyorder=2)
    else:
        smoothed = values
    
    # Normalize for peak detection
    normalized = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-10)
    
    # Find peaks
    peaks, properties = find_peaks(normalized, prominence=prominence, distance=1)
    
    # Find valleys (troughs)
    valleys, _ = find_peaks(-normalized, prominence=prominence, distance=1)
    
    return {
        'peak_indices': peaks,
        'peak_values': values[peaks] if len(peaks) > 0 else [],
        'valley_indices': valleys,
        'valley_values': values[valleys] if len(valleys) > 0 else [],
        'smoothed': smoothed
    }


def perform_stl_decomposition(time_series: pd.Series, period: int = 7) -> Optional[Dict]:
    """
    Perform Seasonal-Trend decomposition using Loess (STL).
    
    Decomposes time series into: trend + seasonal + residual
    """
    if not HAS_STATSMODELS:
        return None
    
    if len(time_series) < period * 2:
        return None
    
    try:
        stl = STL(time_series, period=period, robust=True)
        result = stl.fit()
        
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid,
            'period': period
        }
    except Exception as e:
        print(f"    Warning: STL decomposition failed: {e}")
        return None


print("="*100)
print("TURIN E-SCOOTER ANALYSIS - EXERCISE 1")
print(" Data Cleaning & Descriptive Analysis with Statistical Rigor")
print("="*100)

# ==========================================
# STEP 0: LOAD DATA
# ==========================================

print("\n" + "="*100)
print(" STEP 0: LOADING DATA")
print("="*100)

# Load LIME
print("\n[1/3] Loading LIME...")
df_lime = pd.read_csv(
    os.path.join(DATA_RAW, 'lime', 'Torino_Corse24-25.csv'),
    encoding='utf-8',
    low_memory=False
)
print(f"    Loaded: {len(df_lime):,} records")

# Load VOI
print("\n[2/3] Loading VOI (22 files)...")
voi_base_path = os.path.join(DATA_RAW, 'voi')
dfs_voi = []
for year in [2024, 2025]:
    months = range(1, 13) if year == 2024 else range(1, 11)
    for month in months:
        filename = f"DATINOLEGGI_{year}{month:02d}.xlsx"
        filepath = os.path.join(voi_base_path, filename)
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
            dfs_voi.append(df)

df_voi = pd.concat(dfs_voi, ignore_index=True)
print(f"   Loaded: {len(df_voi):,} records from {len(dfs_voi)} files")

# Load BIRD
print("\n[3/3] Loading BIRD (2 files)...")
bird_base_path = os.path.join(DATA_RAW, 'bird')
df_bird_2024 = pd.read_csv(os.path.join(bird_base_path, 'Bird Torino - 2024 - Sheet1.csv'), encoding='utf-8', low_memory=False)
df_bird_2025 = pd.read_csv(os.path.join(bird_base_path, 'Bird Torino - 2025 (fino il 18_11_2025) - Sheet1.csv'), encoding='utf-8', low_memory=False)
df_bird = pd.concat([df_bird_2024, df_bird_2025], ignore_index=True)
print(f"    Loaded: {len(df_bird):,} records from 2 files")

# ==========================================
# STEP 1: CLEAN DATA (OPERATOR-SPECIFIC)
# ==========================================

print("\n\n" + "="*100)
print(" STEP 1: DATA CLEANING")
print("="*100)

def clean_escooter_data(df, operator_name, col_mapping):
    """
    Clean e-scooter trip data with operator-specific handling
    """
    print(f"\n{'#'*100}")
    print(f"Cleaning: {operator_name}")
    print(f"{'#'*100}")

    initial_count = len(df)
    
    # Make a copy
    df = df.copy()
    
    # Step 1: Rename columns
    df = df.rename(columns=col_mapping)
    
    # Step 2: Parse dates (OPERATOR-SPECIFIC)
    print(" Parsing dates...")
    
    if operator_name == "VOI":
        # VOI dates are in format YYYYMMDDHHmmss (14-digit integer like 20240630215349)
        # Some dates are anonymized (e.g., 20240100000000 where day=00) - use day=15 as mid-month
        def parse_voi_date(x):
            if pd.isna(x):
                return pd.NaT
            # If already datetime, return as is
            if isinstance(x, (datetime, pd.Timestamp)):
                return pd.Timestamp(x)
            # If it's a 14-digit number (YYYYMMDDHHmmss format)
            try:
                if isinstance(x, (int, float)):
                    x_str = str(int(x))
                    if len(x_str) == 14:
                        # Check for anonymized dates where day=00 (e.g., 20240100000000)
                        day_part = x_str[6:8]
                        if day_part == '00':
                            # Anonymized date - use middle of month (day 15) for approximation
                            x_str = x_str[:6] + '15' + x_str[8:]
                        # Format: YYYYMMDDHHmmss
                        return pd.to_datetime(x_str, format='%Y%m%d%H%M%S', errors='coerce')
            except:
                pass
            # Try to parse as string
            try:
                return pd.to_datetime(x)
            except:
                return pd.NaT
        
        df['start_datetime'] = df['start_datetime'].apply(parse_voi_date)
        df['end_datetime'] = df['end_datetime'].apply(parse_voi_date)
    elif operator_name == "BIRD":
        # BIRD dates are "2024-1-1, 1:00" format
        df['start_datetime'] = pd.to_datetime(
            df['start_datetime'], 
            format='%Y-%m-%d, %H:%M',
            errors='coerce'
        )
        df['end_datetime'] = pd.to_datetime(
            df['end_datetime'], 
            format='%Y-%m-%d, %H:%M',
            errors='coerce'
        )
    else:
        # LIME dates are ISO format
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
        df['end_datetime'] = pd.to_datetime(df['end_datetime'], errors='coerce')
    
    # Step 3: Convert coordinates to numeric
    print("Converting coordinates...")
    for col in ['start_lat', 'start_lon', 'end_lat', 'end_lon']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Step 4: Convert distance and duration
    if 'distance_km' in df.columns:
        df['distance_km'] = pd.to_numeric(df['distance_km'], errors='coerce')
    if 'duration_min' in df.columns:
        df['duration_min'] = pd.to_numeric(df['duration_min'], errors='coerce')
    
    # Step 5: Remove rows with missing critical columns
    print("Removing nulls...")
    before_null = len(df)
    df = df.dropna(subset=['start_datetime', 'vehicle_id', 'start_lat', 'start_lon', 'end_lat', 'end_lon'])
    null_removed = before_null - len(df)
    
    # Step 6: Filter to Turin geographic bounds
    print("Filtering geographic bounds...")
    before_bounds = len(df)
    df = df[
        (df['start_lat'].between(44.9, 45.3)) &
        (df['start_lon'].between(7.5, 7.9)) &
        (df['end_lat'].between(44.9, 45.3)) &
        (df['end_lon'].between(7.5, 7.9))
    ]
    bounds_removed = before_bounds - len(df)
    
    # Step 7: Remove unrealistic trips
    print("Filtering unrealistic trips...")
    before_filter = len(df)
    
    # Min distance: 50m
    if 'distance_km' in df.columns:
        df = df[df['distance_km'] >= 0.05]
    
    # Min duration: 60 seconds
    if 'duration_min' in df.columns:
        df = df[df['duration_min'] >= 1.0]
    
    filter_removed = before_filter - len(df)
    
    # Step 8: Extract temporal features
    print("Extracting temporal features...")
    df['year'] = df['start_datetime'].dt.year
    df['month'] = df['start_datetime'].dt.month
    df['week'] = df['start_datetime'].dt.isocalendar().week
    df['day_of_week'] = df['start_datetime'].dt.dayofweek  # 0=Mon, 6=Sun
    df['hour'] = df['start_datetime'].dt.hour
    df['date'] = df['start_datetime'].dt.date
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    final_count = len(df)
    total_removed = initial_count - final_count
    retention_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
    
    print(f"\n  DATA QUALITY REPORT:")
    print(f"   Initial records:      {initial_count:>12,}")
    print(f"   Nulls removed:        {null_removed:>12,}")
    print(f"   Out-of-bounds:        {bounds_removed:>12,}")
    print(f"   Unrealistic trips:    {filter_removed:>12,}")
    print(f"   Final records:        {final_count:>12,}")
    print(f"   Retention rate:       {retention_rate:>11.1f}%")
    
    return df

# Column mappings (from inspection)
map_lime = {
    'ID_VEICOLO': 'vehicle_id',
    'DATAORA_INIZIO': 'start_datetime',
    'DATAORA_FINE': 'end_datetime',
    'LATITUDINE_INIZIO_CORSA': 'start_lat',
    'LONGITUTIDE_INIZIO_CORSA': 'start_lon',
    'LATITUDINE_FINE_CORSA': 'end_lat',
    'LONGITUTIDE_FINE_CORSA': 'end_lon',
    'DISTANZA_KM': 'distance_km',
    'DURATA_MIN': 'duration_min',
    'BATTERIA_INIZIO_CORSA': 'battery_start',
    'BATTERIA_FINE_CORSA': 'battery_end',
}

map_voi = {
    'Targa veicolo': 'vehicle_id',
    'Data inizio corsa': 'start_datetime',
    'Data fine corsa': 'end_datetime',
    'Lat inizio corsa_coordinate': 'start_lat',
    'Lon inizio corsa_coordinate': 'start_lon',
    'Lat fine corsa_coordinate': 'end_lat',
    'Lon fine corsa_coordinate': 'end_lon',
    'KM Tot': 'distance_km',
    'Tempo Tot': 'duration_min',
    'Batteria inizio': 'battery_start',
    'Batteria fine': 'battery_end',
}

map_bird = {
    'ID_VEICOLO': 'vehicle_id',
    'DATAORA_INIZIO': 'start_datetime',
    'DATAORA_FINE': 'end_datetime',
    'LATITUDINE_INIZIO_CORSA': 'start_lat',
    'LONGITUTIDE_INIZIO_CORSA': 'start_lon',
    'LATITUDINE_FINE_CORSA': 'end_lat',
    'LONGITUTIDE_FINE_CORSA': 'end_lon',
    'DISTANZA_KM': 'distance_km',
    'DURATA_MIN': 'duration_min',
}

# Clean all three operators
df_lime_clean = clean_escooter_data(df_lime, "LIME", map_lime)
df_voi_clean = clean_escooter_data(df_voi, "VOI", map_voi)
df_bird_clean = clean_escooter_data(df_bird, "BIRD", map_bird)

# ==============================================================================
# SAVE DATA QUALITY REPORT TO CSV (Gap #2 Fix)
# ==============================================================================
print("\n" + "="*100)
print(" STEP 1.5: SAVING DATA QUALITY REPORT")
print("="*100)

# Collect quality reports for all operators
quality_reports = []
for operator, df_raw, df_clean in [
    ('LIME', df_lime, df_lime_clean),
    ('VOI', df_voi, df_voi_clean),  
    ('BIRD', df_bird, df_bird_clean)
]:
    records_before = len(df_raw)
    records_after = len(df_clean)
    records_removed = records_before - records_after
    retention_rate = round(records_after / records_before * 100, 2) if records_before > 0 else 0
    
    report = {
        'operator': operator,
        'records_before': records_before,
        'records_after': records_after,
        'records_removed': records_removed,
        'retention_rate_pct': retention_rate,
        'removal_rate_pct': round(100 - retention_rate, 2)
    }
    quality_reports.append(report)
    print(f"   {operator}: {records_before:,} → {records_after:,} ({retention_rate:.1f}% retained)")

# Save to CSV
quality_df = pd.DataFrame(quality_reports)
quality_path = os.path.join(OUTPUTS_TABLES, 'data_quality_report.csv')
quality_df.to_csv(quality_path, index=False)
print(f"\n✓ Saved: {quality_path}")

# Also save a more detailed version
print(f"\n   Total across all operators:")
print(f"     Before: {quality_df['records_before'].sum():,}")
print(f"     After:  {quality_df['records_after'].sum():,}")
print(f"     Removed: {quality_df['records_removed'].sum():,}")

# ==========================================
# STEP 2: EXERCISE 1 ANALYSIS
# ==========================================

print("\n\n" + "="*100)
print(" STEP 2: DESCRIPTIVE ANALYSIS & VISUALIZATIONS")
print("="*100)

def analyze_operator(gdf, operator_name):
    """
    Perform Exercise 1 analysis on cleaned data
    """
    print(f"\n{'€'*100}")
    print(f"Analysis: {operator_name}")
    print(f"{'€'*100}")

    # Monthly trends
    print("\n  → Monthly Trends:")
    monthly_trips = gdf.groupby(gdf['start_datetime'].dt.to_period('M')).size()
    for period, count in monthly_trips.items():
        print(f"    {period}: {count:>8,} trips")
    
    # Weekly patterns
    print(f"\n  → Weekly Pattern (Day of Week):")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_trips = gdf['day_of_week'].value_counts().sort_index()
    for day_num, count in weekly_trips.items():
        print(f"    {day_names[day_num]:10s}: {count:>10,} trips")
    
    # Hourly patterns (peak hours)
    print(f"\n  → Hourly Pattern (24 hours):")
    hourly_trips = gdf['hour'].value_counts().sort_index()
    peak_hour = hourly_trips.idxmax()
    print(f"    Peak hour: {peak_hour}:00 ({hourly_trips.max():,} trips)")
    print(f"    Top 5 hours:")
    for hour, count in hourly_trips.nlargest(5).items():
        print(f"      {hour:2d}:00 â†’ {count:>8,} trips")
    
    # Vehicle usage statistics
    print(f"\n  → Vehicle Fleet Statistics:")
    trips_per_vehicle = gdf['vehicle_id'].value_counts()
    print(f"    Unique vehicles:      {len(trips_per_vehicle):>10,}")
    print(f"    Mean trips/vehicle:   {trips_per_vehicle.mean():>10.2f}")
    print(f"    Median trips/vehicle: {trips_per_vehicle.median():>10.2f}")
    print(f"    Min trips/vehicle:    {trips_per_vehicle.min():>10,}")
    print(f"    Max trips/vehicle:    {trips_per_vehicle.max():>10,}")
    
    # Distance and duration statistics
    print(f"\n  → Trip Statistics:")
    if 'distance_km' in gdf.columns:
        print(f"    Avg distance:         {gdf['distance_km'].mean():>10.2f} km")
        print(f"    Median distance:      {gdf['distance_km'].median():>10.2f} km")
    if 'duration_min' in gdf.columns:
        print(f"    Avg duration:         {gdf['duration_min'].mean():>10.2f} min")
        print(f"    Median duration:      {gdf['duration_min'].median():>10.2f} min")
    
    # Return data for visualization
    return {
        'monthly': monthly_trips,
        'weekly': weekly_trips,
        'hourly': hourly_trips,
        'vehicles': len(trips_per_vehicle),
        'trips_per_vehicle': trips_per_vehicle,
    }

# Analyze all operators
analysis_lime = analyze_operator(df_lime_clean, "LIME")
analysis_voi = analyze_operator(df_voi_clean, "VOI")
analysis_bird = analyze_operator(df_bird_clean, "BIRD")

# ==============================================================================
# STEP 3: ADVANCED STATISTICAL ANALYSIS
# ==============================================================================

print("\n\n" + "="*100)
print(" STEP 3: ADVANCED STATISTICAL ANALYSIS")
print("="*100)

# Store all statistical results for checkpoint
statistical_results = {}

# 3.1 Kruskal-Wallis Test: Compare hourly distributions across operators
print("\n" + "-"*80)
print("3.1 KRUSKAL-WALLIS H-TEST: Hourly Trip Distribution Comparison")
print("-"*80)

hourly_groups = {
    'LIME': df_lime_clean['hour'].values,
    'VOI': df_voi_clean['hour'].values,
    'BIRD': df_bird_clean['hour'].values
}

kw_hourly = perform_kruskal_wallis_test(hourly_groups)
statistical_results['kruskal_wallis_hourly'] = kw_hourly

print(f"\n  Hourly Distribution Comparison:")
print(f"    H-statistic: {kw_hourly['H_statistic']:.2f}")
print(f"    p-value: {kw_hourly['p_value']:.2e}")
print(f"    η² (effect size): {kw_hourly['eta_squared']:.4f}")

effect_interp = "large" if kw_hourly['eta_squared'] >= 0.14 else \
                "medium" if kw_hourly['eta_squared'] >= 0.06 else \
                "small" if kw_hourly['eta_squared'] >= 0.01 else "negligible"
print(f"    Effect interpretation: {effect_interp}")
print(f"    Significant difference: {'YES' if kw_hourly['significant'] else 'NO'}")

print("\n  Post-hoc Pairwise Comparisons (Mann-Whitney U):")
for comp in kw_hourly['pairwise_comparisons']:
    sig_marker = "*" if comp['significant'] else ""
    d_interp = "large" if abs(comp['cohens_d']) >= 0.8 else \
               "medium" if abs(comp['cohens_d']) >= 0.5 else \
               "small" if abs(comp['cohens_d']) >= 0.2 else "negligible"
    print(f"    {comp['comparison']:15s} | U={comp['U_statistic']:,.0f} | p={comp['p_value']:.2e}{sig_marker} | d={comp['cohens_d']:.3f} ({d_interp})")

# 3.2 Chi-Square Test: Weekday vs Weekend independence
print("\n" + "-"*80)
print("3.2 CHI-SQUARE TEST: Weekday vs Weekend Independence")
print("-"*80)

# Create contingency table
weekday_weekend_table = pd.DataFrame({
    'LIME': [
        len(df_lime_clean[df_lime_clean['is_weekend'] == 0]),
        len(df_lime_clean[df_lime_clean['is_weekend'] == 1])
    ],
    'VOI': [
        len(df_voi_clean[df_voi_clean['is_weekend'] == 0]),
        len(df_voi_clean[df_voi_clean['is_weekend'] == 1])
    ],
    'BIRD': [
        len(df_bird_clean[df_bird_clean['is_weekend'] == 0]),
        len(df_bird_clean[df_bird_clean['is_weekend'] == 1])
    ]
}, index=['Weekday', 'Weekend'])

chi2, p_chi, dof, expected = chi2_contingency(weekday_weekend_table.values)
cramers_v = calculate_cramers_v(weekday_weekend_table.values)

statistical_results['chi_square_weekend'] = {
    'chi2_statistic': chi2,
    'p_value': p_chi,
    'degrees_of_freedom': dof,
    'cramers_v': cramers_v,
    'contingency_table': weekday_weekend_table.to_dict()
}

print(f"\n  Contingency Table:")
print(weekday_weekend_table.to_string())
print(f"\n  χ² statistic: {chi2:.2f}")
print(f"  p-value: {p_chi:.2e}")
print(f"  Degrees of freedom: {dof}")
print(f"  Cramér's V: {cramers_v:.4f}")

v_interp = "large" if cramers_v >= 0.5 else \
           "medium" if cramers_v >= 0.3 else \
           "small" if cramers_v >= 0.1 else "negligible"
print(f"  Effect interpretation: {v_interp}")

# 3.3 Bootstrap Confidence Intervals
print("\n" + "-"*80)
print("3.3 BOOTSTRAP CONFIDENCE INTERVALS (95%)")
print("-"*80)

bootstrap_results = {}
for name, df in [('LIME', df_lime_clean), ('VOI', df_voi_clean), ('BIRD', df_bird_clean)]:
    print(f"\n  {name}:")
    
    # Trips per day
    daily_trips = df.groupby('date').size().values
    ci_daily = bootstrap_ci(daily_trips, n_bootstrap=1000)
    
    # Distance (if available)
    if 'distance_km' in df.columns:
        distances = df['distance_km'].dropna().values
        ci_distance = bootstrap_ci(distances, n_bootstrap=1000)
    else:
        ci_distance = (np.nan, np.nan)
    
    # Duration (if available)
    if 'duration_min' in df.columns:
        durations = df['duration_min'].dropna().values
        ci_duration = bootstrap_ci(durations, n_bootstrap=1000)
    else:
        ci_duration = (np.nan, np.nan)
    
    bootstrap_results[name] = {
        'daily_trips_mean': np.mean(daily_trips),
        'daily_trips_ci': ci_daily,
        'distance_mean': np.mean(distances) if 'distance_km' in df.columns else np.nan,
        'distance_ci': ci_distance,
        'duration_mean': np.mean(durations) if 'duration_min' in df.columns else np.nan,
        'duration_ci': ci_duration
    }
    
    print(f"    Daily trips: {np.mean(daily_trips):.1f} [95% CI: {ci_daily[0]:.1f} - {ci_daily[1]:.1f}]")
    if 'distance_km' in df.columns:
        print(f"    Distance (km): {np.mean(distances):.2f} [95% CI: {ci_distance[0]:.2f} - {ci_distance[1]:.2f}]")
    if 'duration_min' in df.columns:
        print(f"    Duration (min): {np.mean(durations):.2f} [95% CI: {ci_duration[0]:.2f} - {ci_duration[1]:.2f}]")

statistical_results['bootstrap_ci'] = bootstrap_results

# 3.4 Peak Detection Analysis
print("\n" + "-"*80)
print("3.4 PEAK DETECTION ANALYSIS (Signal Processing)")
print("-"*80)

peak_results = {}
for name, analysis in [('LIME', analysis_lime), ('VOI', analysis_voi), ('BIRD', analysis_bird)]:
    hourly_series = analysis['hourly'].sort_index()
    peaks = detect_temporal_peaks(hourly_series, prominence=0.15)
    
    peak_results[name] = {
        'peak_hours': hourly_series.index[peaks['peak_indices']].tolist() if len(peaks['peak_indices']) > 0 else [],
        'peak_values': peaks['peak_values'].tolist() if len(peaks['peak_values']) > 0 else [],
        'valley_hours': hourly_series.index[peaks['valley_indices']].tolist() if len(peaks['valley_indices']) > 0 else [],
        'valley_values': peaks['valley_values'].tolist() if len(peaks['valley_values']) > 0 else [],
    }
    
    print(f"\n  {name}:")
    if len(peaks['peak_indices']) > 0:
        peak_hours = [f"{h}:00" for h in hourly_series.index[peaks['peak_indices']]]
        print(f"    Peak hours detected: {', '.join(peak_hours)}")
    else:
        print(f"    No significant peaks detected")
    if len(peaks['valley_indices']) > 0:
        valley_hours = [f"{h}:00" for h in hourly_series.index[peaks['valley_indices']]]
        print(f"    Valley hours detected: {', '.join(valley_hours)}")

statistical_results['peak_detection'] = peak_results

# 3.5 STL Decomposition (if statsmodels available)
print("\n" + "-"*80)
print("3.5 STL SEASONAL DECOMPOSITION")
print("-"*80)

stl_results = {}
if HAS_STATSMODELS:
    for name, df in [('LIME', df_lime_clean), ('VOI', df_voi_clean), ('BIRD', df_bird_clean)]:
        daily_ts = df.groupby('date').size()
        daily_ts.index = pd.to_datetime(daily_ts.index)
        daily_ts = daily_ts.sort_index()
        
        stl_result = perform_stl_decomposition(daily_ts, period=7)
        
        if stl_result is not None:
            stl_results[name] = {
                'trend_strength': 1 - np.var(stl_result['residual']) / np.var(stl_result['trend'] + stl_result['residual']),
                'seasonal_strength': 1 - np.var(stl_result['residual']) / np.var(stl_result['seasonal'] + stl_result['residual']),
            }
            print(f"\n  {name}:")
            print(f"    Trend strength: {stl_results[name]['trend_strength']:.3f}")
            print(f"    Seasonal strength (weekly): {stl_results[name]['seasonal_strength']:.3f}")
        else:
            print(f"\n  {name}: Insufficient data for decomposition")
else:
    print("\n  Skipped: statsmodels not installed")

statistical_results['stl_decomposition'] = stl_results

# 3.6 Levene's Test: Variance Homogeneity
print("\n" + "-"*80)
print("3.6 LEVENE'S TEST: Variance Homogeneity")
print("-"*80)

# Test if variances of daily trips are equal across operators
daily_lime = df_lime_clean.groupby('date').size().values
daily_voi = df_voi_clean.groupby('date').size().values
daily_bird = df_bird_clean.groupby('date').size().values

levene_stat, levene_p = levene(daily_lime, daily_voi, daily_bird)
statistical_results['levene_test'] = {
    'statistic': levene_stat,
    'p_value': levene_p,
    'homogeneous': levene_p > 0.05
}

print(f"\n  Daily Trip Variance Comparison:")
print(f"    Levene statistic: {levene_stat:.2f}")
print(f"    p-value: {levene_p:.2e}")
print(f"    Variances homogeneous: {'YES' if levene_p > 0.05 else 'NO'}")

# Save statistical results checkpoint
checkpoint_path = os.path.join(OUTPUTS_REPORTS, 'checkpoint_statistics.pkl')
with open(checkpoint_path, 'wb') as f:
    pickle.dump(statistical_results, f)
print(f"\n✓ Saved: {checkpoint_path}")

# ==============================================================================
# STEP 4: VISUALIZATIONS
# ==============================================================================

print("\n\n" + "="*100)
print(" STEP 4: CREATING VISUALIZATIONS")
print("="*100)

def create_visualizations(gdf, operator_name, analysis_data):
    """
    Create 4-panel visualization for each operator.
    Also saves each chart as a separate individual figure.
    """
    op_lower = operator_name.lower()
    
    # =========================================================================
    # Individual Figure 1: Monthly Trends
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_data = analysis_data['monthly']
    ax.bar(range(len(monthly_data)), monthly_data.values, color='skyblue', edgecolor='black')
    ax.set_xticks(range(len(monthly_data)))
    ax.set_xticklabels([str(x) for x in monthly_data.index], rotation=45, ha='right')
    ax.set_title(f'{operator_name}: Monthly Trip Trend', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Trips', fontweight='bold')
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylim(0, monthly_data.max() * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUTS_FIGURES, f'{op_lower}_monthly_trend.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {op_lower}_monthly_trend.png")
    
    # =========================================================================
    # Individual Figure 2: Weekly Pattern
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly_data = analysis_data['weekly'].reindex(range(7)).fillna(0)
    ax.bar(range(7), weekly_data.values, color='orange', edgecolor='black')
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_title(f'{operator_name}: Weekly Trip Pattern', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Trips', fontweight='bold')
    ax.set_xlabel('Day of Week', fontweight='bold')
    ax.set_ylim(0, weekly_data.max() * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUTS_FIGURES, f'{op_lower}_weekly_pattern.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {op_lower}_weekly_pattern.png")
    
    # =========================================================================
    # Individual Figure 3: Hourly Pattern
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    hourly_data = analysis_data['hourly'].reindex(range(24)).fillna(0)
    ax.bar(range(24), hourly_data.values, color='green', edgecolor='black')
    ax.set_xticks(range(0, 24, 1))
    ax.set_xticklabels([f'{h}' for h in range(24)], fontsize=9)
    ax.set_title(f'{operator_name}: Hourly Trip Pattern', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Trips', fontweight='bold')
    ax.set_xlabel('Hour of Day', fontweight='bold')
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, hourly_data.max() * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUTS_FIGURES, f'{op_lower}_hourly_pattern.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {op_lower}_hourly_pattern.png")
    
    # =========================================================================
    # Individual Figure 4: Fleet Utilization Histogram
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    trips_per_vehicle = analysis_data['trips_per_vehicle']
    counts, bins, _ = ax.hist(trips_per_vehicle, bins=50, color='purple', edgecolor='black', alpha=0.7)
    
    # Add mean and median lines
    mean_val = trips_per_vehicle.mean()
    median_val = trips_per_vehicle.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='blue', linestyle=':', linewidth=2, label=f'Median: {median_val:.1f}')
    
    ax.set_title(f'{operator_name}: Fleet Utilization Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Trips per Vehicle', fontweight='bold')
    ax.set_ylabel('Number of Vehicles', fontweight='bold')
    ax.set_xlim(0, trips_per_vehicle.max() * 1.05)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUTS_FIGURES, f'{op_lower}_fleet_utilization.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {op_lower}_fleet_utilization.png")
    
    # =========================================================================
    # Combined 4-Panel Figure (for backward compatibility)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{operator_name} - Exercise 1: Descriptive Analysis', fontsize=16, fontweight='bold')
    
    # Chart 1: Monthly Trends
    axes[0,0].bar(range(len(monthly_data)), monthly_data.values, color='skyblue', edgecolor='black')
    axes[0,0].set_xticks(range(len(monthly_data)))
    axes[0,0].set_xticklabels([str(x) for x in monthly_data.index], rotation=45, ha='right')
    axes[0,0].set_title('A. Monthly Trips Trend', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Number of Trips')
    axes[0,0].set_xlabel('Month')
    axes[0,0].set_ylim(0, monthly_data.max() * 1.15)
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # Chart 2: Weekly Pattern
    axes[0,1].bar(range(7), weekly_data.values, color='orange', edgecolor='black')
    axes[0,1].set_xticks(range(7))
    axes[0,1].set_xticklabels(day_names)
    axes[0,1].set_title('B. Weekly Pattern', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Number of Trips')
    axes[0,1].set_xlabel('Day of Week')
    axes[0,1].set_ylim(0, weekly_data.max() * 1.15)
    axes[0,1].grid(axis='y', alpha=0.3)
    
    # Chart 3: Hourly Pattern
    axes[1,0].bar(range(24), hourly_data.values, color='green', edgecolor='black')
    axes[1,0].set_xticks(range(0, 24, 2))
    axes[1,0].set_xticklabels([f'{h}' for h in range(0, 24, 2)])
    axes[1,0].set_title('C. Hourly Pattern', fontsize=12, fontweight='bold')
    axes[1,0].set_ylabel('Number of Trips')
    axes[1,0].set_xlabel('Hour of Day')
    axes[1,0].set_xlim(-0.5, 23.5)
    axes[1,0].set_ylim(0, hourly_data.max() * 1.15)
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Chart 4: Histogram of Trips per Vehicle
    counts, bins, _ = axes[1,1].hist(trips_per_vehicle, bins=50, color='purple', edgecolor='black', alpha=0.7)
    axes[1,1].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    axes[1,1].axvline(median_val, color='blue', linestyle=':', linewidth=2, label=f'Median: {median_val:.1f}')
    axes[1,1].set_title('D. Fleet Utilization', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Trips per Vehicle')
    axes[1,1].set_ylabel('Number of Vehicles')
    axes[1,1].set_xlim(0, trips_per_vehicle.max() * 1.05)
    axes[1,1].set_ylim(0, counts.max() * 1.15)
    axes[1,1].legend(loc='upper right', fontsize=9)
    axes[1,1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUTS_FIGURES, f'{operator_name}_Exercise1_Analysis.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"    ✓ Saved: {operator_name}_Exercise1_Analysis.png (combined)")
    plt.close()

print("\n  Creating visualizations...")
create_visualizations(df_lime_clean, 'LIME', analysis_lime)
create_visualizations(df_voi_clean, 'VOI', analysis_voi)
create_visualizations(df_bird_clean, 'BIRD', analysis_bird)

# ==============================================================================
# STEP 5: COMPARATIVE ANALYSIS TABLE
# ==============================================================================

print("\n\n" + "="*100)
print(" STEP 5: COMPARATIVE ANALYSIS")
print("="*100)

# All operators comparison
comparison_data = {
    'Operator': ['LIME', 'VOI', 'BIRD'],
    'Total Trips': [len(df_lime_clean), len(df_voi_clean), len(df_bird_clean)],
    'Unique Vehicles': [df_lime_clean['vehicle_id'].nunique(), 
                        df_voi_clean['vehicle_id'].nunique(),
                        df_bird_clean['vehicle_id'].nunique()],
    'Date Range': [
        f"{df_lime_clean['date'].min()} to {df_lime_clean['date'].max()}",
        f"{df_voi_clean['date'].min()} to {df_voi_clean['date'].max()}",
        f"{df_bird_clean['date'].min()} to {df_bird_clean['date'].max()}",
    ],
    'Avg Trips/Vehicle': [
        f"{df_lime_clean['vehicle_id'].value_counts().mean():.2f}",
        f"{df_voi_clean['vehicle_id'].value_counts().mean():.2f}",
        f"{df_bird_clean['vehicle_id'].value_counts().mean():.2f}",
    ],
    'Peak Hour': [
        f"{analysis_lime['hourly'].idxmax()}:00",
        f"{analysis_voi['hourly'].idxmax()}:00",
        f"{analysis_bird['hourly'].idxmax()}:00",
    ],
    'Peak Day': [
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][analysis_lime['weekly'].idxmax()],
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][analysis_voi['weekly'].idxmax()],
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][analysis_bird['weekly'].idxmax()],
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

# Save comparison table
comparison_path = os.path.join(OUTPUTS_REPORTS, 'Comparison_All_Operators.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"\n    ✓ Saved: {comparison_path}")

# ==============================================================================
# STEP 6: SAVE CLEANED DATA & CHECKPOINTS
# ==============================================================================

print("\n\n" + "="*100)
print(" STEP 6: SAVING CLEANED DATA & CHECKPOINTS")
print("="*100)

print("\n  Saving cleaned datasets to data/processed/...")
lime_path = os.path.join(DATA_PROCESSED, 'lime_cleaned.csv')
voi_path = os.path.join(DATA_PROCESSED, 'voi_cleaned.csv')
bird_path = os.path.join(DATA_PROCESSED, 'bird_cleaned.csv')

df_lime_clean.to_csv(lime_path, index=False)
print(f"    ✓ Saved: {lime_path} ({len(df_lime_clean):,} records)")

df_voi_clean.to_csv(voi_path, index=False)
print(f"    ✓ Saved: {voi_path} ({len(df_voi_clean):,} records)")

df_bird_clean.to_csv(bird_path, index=False)
print(f"    ✓ Saved: {bird_path} ({len(df_bird_clean):,} records)")

# Save analysis checkpoints for visualization script
print("\n  Saving analysis checkpoints for visualization...")
analysis_checkpoint = {
    'lime': {
        'data': df_lime_clean,
        'analysis': analysis_lime,
    },
    'voi': {
        'data': df_voi_clean,
        'analysis': analysis_voi,
    },
    'bird': {
        'data': df_bird_clean,
        'analysis': analysis_bird,
    },
    'comparison': comparison_df,
    'statistical_results': statistical_results,
}

checkpoint_file = os.path.join(OUTPUTS_REPORTS, 'checkpoint_exercise1.pkl')
with open(checkpoint_file, 'wb') as f:
    pickle.dump(analysis_checkpoint, f)
print(f"    ✓ Saved: {checkpoint_file}")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

print("\n\n" + "="*100)
print("  EXERCISE 1 COMPLETE! ")
print("="*100)

print(f"""
SUMMARY:
  
  Data Cleaning:
      LIME:   {len(df_lime_clean):,} records (from {len(df_lime):,})
      VOI:    {len(df_voi_clean):,} records (from {len(df_voi):,})
      BIRD:   {len(df_bird_clean):,} records (from {len(df_bird):,})
      TOTAL:  {len(df_lime_clean) + len(df_voi_clean) + len(df_bird_clean):,} records

  Statistical Analysis :
      ✓ Kruskal-Wallis H-test (hourly distributions)
      ✓ Mann-Whitney U post-hoc comparisons
      ✓ Chi-square test (weekday/weekend independence)
      ✓ Bootstrap confidence intervals (95%)
      ✓ Peak detection (signal processing)
      ✓ STL seasonal decomposition
      ✓ Levene's test (variance homogeneity)
      ✓ Effect sizes (Cohen's d, η², Cramér's V)

  Outputs Created:
      {OUTPUTS_REPORTS}/
          ├── Comparison_All_Operators.csv
          ├── checkpoint_exercise1.pkl
          └── checkpoint_statistics.pkl
      
      {OUTPUTS_FIGURES}/
          ├── LIME_Exercise1_Analysis.png
          ├── VOI_Exercise1_Analysis.png
          └── BIRD_Exercise1_Analysis.png

  Next Steps:
    → Run visualization script for figures
    → Proceed to Exercise 2: O-D Matrix Construction
    → Use checkpoint files for advanced visualizations

EXERCISE 1 COMPLETE!  
""")

print("="*100)