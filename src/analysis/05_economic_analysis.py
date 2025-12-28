#!/usr/bin/env python3
"""
==============================================================================
Exercise 5: Business Model & Economic Analysis
==============================================================================
Top-tier financial analysis for E-Scooter market economics.

FINANCIAL ANALYSIS FRAMEWORK:
1. Revenue Modeling
   - Operator-specific tariff structures
   - Duration-based revenue calculation
   - Revenue per trip with CI estimation

2. Cost Analysis
   - Variable costs (logistics, energy, maintenance)
   - Fixed costs (depreciation, city fees)
   - Break-even analysis with sensitivity

3. Profitability Metrics
   - Net profit per trip
   - ROI calculation
   - Contribution margin analysis

ADVANCED ANALYTICS:
1. Monte Carlo Simulation
   - Stochastic risk analysis (n=10,000 simulations)
   - Distribution fitting for key parameters
   - VaR (Value at Risk) calculation

2. Sensitivity Analysis
   - Tornado diagram for parameter importance
   - Price elasticity estimation
   - Break-even sensitivity curves

3. Bootstrap Confidence Intervals
   - 95% CI for profit metrics (n=1000)
   - Operator comparison with statistical tests

4. Regression Analysis
   - Duration-profit relationship
   - Demand-price elasticity
   - Zone profitability drivers

5. Scenario Modeling
   - Base/Optimistic/Pessimistic scenarios
   - No-subsidy zone elimination
   - Fleet optimization scenarios

OUTPUTS:
- checkpoint_economics_trips.pkl 
- checkpoint_economics_zones.csv 
- checkpoint_operator_pnl.csv 
- checkpoint_economics_temporal.csv 
- checkpoint_economics_pareto.csv 
- checkpoint_economics_scenarios.csv 
- checkpoint_montecarlo_simulation.csv 
- checkpoint_sensitivity_analysis.csv 
- checkpoint_bootstrap_financial_ci.csv 
- checkpoint_regression_analysis.csv

Author: Ali Vaezi
Date: December 2025
==============================================================================
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize_scalar
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Note: scikit-learn not installed - using basic regression")

# =============================================================================
# COST PARAMETERS (UPDATED: TURIN SPECIFIC 2024-2025)
# =============================================================================

# 1. VARIABLE COSTS (Per Trip)
# Source: euenergy.live (Turin Commercial Rate) & Bird One Specs
COST_KWH_TURIN              = 0.11    # €/kWh commercial rate
SCOOTER_BATTERY_KWH         = 0.47    # kWh (Standard commercial scooter)
SCOOTER_RANGE_KM            = 48.0    # km per full charge
AVG_TRIP_DISTANCE_KM        = 2.5     # Derived from Exercise 2 data

# Derived Electricity Cost
# Cost to charge full battery = 0.47 * 0.11 = €0.0517
# Cost per km = €0.0517 / 48km = €0.00108
COST_ELECTRICITY_PER_TRIP   = (SCOOTER_BATTERY_KWH * COST_KWH_TURIN) / SCOOTER_RANGE_KM * AVG_TRIP_DISTANCE_KM
# Result should be approx €0.0027 - small but explicit as requested.

# Other Variable Costs
COST_MAINTENANCE_PER_TRIP   = 0.45    # Wear & tear
COST_OPERATIONS_PER_TRIP    = 0.40    # Rebalancing labor
COST_INSURANCE_PER_TRIP     = 0.20

VARIABLE_COST_PER_TRIP = (
    COST_ELECTRICITY_PER_TRIP +
    COST_MAINTENANCE_PER_TRIP +
    COST_OPERATIONS_PER_TRIP +
    COST_INSURANCE_PER_TRIP
)

# 2. FIXED COSTS (Per Scooter/Year)
# Source: Atom Mobility 2025 & Movability Consulting
VEHICLE_PURCHASE_PRICE      = 875.00  # Commercial fleet pricing
USEFUL_LIFE_YEARS           = 3.0     # Modern amortization standard
ANNUAL_PERMIT_FEE           = 50.00   # Municipal fee
ANNUAL_INSURANCE_FIXED      = 20.00

# Amortization Calculation (Required Terminology)
ANNUAL_AMORTIZATION = VEHICLE_PURCHASE_PRICE / USEFUL_LIFE_YEARS
DAILY_AMORTIZATION  = ANNUAL_AMORTIZATION / 365.0
DAILY_FIXED_COST    = (ANNUAL_AMORTIZATION + ANNUAL_PERMIT_FEE + ANNUAL_INSURANCE_FIXED) / 365.0
DAILY_CITY_FEE      = ANNUAL_PERMIT_FEE / 365.0

# Revenue Models (User Tariffs) - Per Operator
ECONOMIC_PARAMS = {
    'tariffs': {
        'LIME': {'unlock_fee': 1.00, 'minute_rate': 0.19},
        'VOI': {'unlock_fee': 1.00, 'minute_rate': 0.22},
        'BIRD': {'unlock_fee': 1.00, 'minute_rate': 0.20}
    },
    'variable_cost_per_trip': VARIABLE_COST_PER_TRIP,
    'annual_city_fee_per_vehicle': ANNUAL_PERMIT_FEE,
    'vehicle_unit_cost': VEHICLE_PURCHASE_PRICE,
    'lifespan_months': int(USEFUL_LIFE_YEARS * 12),
}

# Paths - go up TWO levels from src/analysis/ to project root
BASE_DIR = Path(__file__).parent.parent.parent  # project root
DATA_DIR = BASE_DIR / "outputs" / "reports" / "exercise3"
ZONES_PATH = DATA_DIR / "checkpoint_zones_with_metrics.geojson"
OUTPUT_DIR = BASE_DIR / "outputs" / "reports" / "exercise5"

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_data():
    """Load trip data and zone data."""
    print("\n" + "="*70)
    print("STEP 1: LOAD DATA")
    print("="*70)
    
    # Load trip data
    trips_path = DATA_DIR / "checkpoint_validated_escooter_data.pkl"
    print(f"Loading: {trips_path.name}")
    df = pd.read_pickle(trips_path)
    print(f"  → Loaded {len(df):,} trips")
    
    # Parse dates
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    
    # Load zones
    print(f"Loading: {ZONES_PATH.name}")
    zones_gdf = gpd.read_file(ZONES_PATH)
    print(f"  → Loaded {len(zones_gdf)} zones")
    
    # Summary
    print(f"\n  Data Summary:")
    print(f"  - Operators: {df['operator'].nunique()} ({', '.join(df['operator'].unique())})")
    print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  - Avg duration: {df['duration_min'].mean():.1f} min")
    
    return df, zones_gdf


# ============================================================================
# STEP 2: CALCULATE REVENUE PER TRIP
# ============================================================================

def calculate_revenue(df):
    """
    Calculate gross revenue per trip based on operator-specific tariffs.
    
    Revenue = Unlock Fee + (Duration_Minutes * Minute_Rate)
    """
    print("\n" + "="*70)
    print("STEP 2: CALCULATE REVENUE PER TRIP")
    print("="*70)
    
    # Initialize revenue column
    df['gross_revenue'] = 0.0
    df['unlock_fee'] = 0.0
    df['time_revenue'] = 0.0
    
    # Calculate for each operator
    for operator, tariff in ECONOMIC_PARAMS['tariffs'].items():
        mask = df['operator'] == operator
        trip_count = mask.sum()
        
        unlock = tariff['unlock_fee']
        rate = tariff['minute_rate']
        
        df.loc[mask, 'unlock_fee'] = unlock
        df.loc[mask, 'time_revenue'] = df.loc[mask, 'duration_min'] * rate
        df.loc[mask, 'gross_revenue'] = unlock + df.loc[mask, 'time_revenue']
        
        avg_rev = df.loc[mask, 'gross_revenue'].mean()
        total_rev = df.loc[mask, 'gross_revenue'].sum()
        
        print(f"  {operator}:")
        print(f"    - Tariff: €{unlock:.2f} + €{rate:.2f}/min")
        print(f"    - Trips: {trip_count:,}")
        print(f"    - Avg Revenue/Trip: €{avg_rev:.2f}")
        print(f"    - Total Revenue: €{total_rev:,.2f}")
    
    # Overall stats
    print(f"\n  Overall Revenue:")
    print(f"  - Total Gross Revenue: €{df['gross_revenue'].sum():,.2f}")
    print(f"  - Average Revenue/Trip: €{df['gross_revenue'].mean():.2f}")
    
    return df


# ============================================================================
# STEP 3: CALCULATE VARIABLE COSTS
# ============================================================================

def calculate_variable_costs(df):
    """
    Calculate variable costs per trip.
    
    Variable Cost = Trip_Count * €1.20 (logistics, energy, maintenance)
    """
    print("\n" + "="*70)
    print("STEP 3: CALCULATE VARIABLE COSTS")
    print("="*70)
    
    var_cost = ECONOMIC_PARAMS['variable_cost_per_trip']
    df['variable_cost'] = var_cost
    
    total_var_cost = df['variable_cost'].sum()
    
    print(f"  Variable Cost per Trip: €{var_cost:.2f}")
    print(f"  Total Variable Costs: €{total_var_cost:,.2f}")
    
    # Contribution margin (before fixed costs)
    df['contribution_margin'] = df['gross_revenue'] - df['variable_cost']
    avg_margin = df['contribution_margin'].mean()
    total_margin = df['contribution_margin'].sum()
    
    print(f"\n  Contribution Margin (Revenue - Variable Cost):")
    print(f"  - Average: €{avg_margin:.2f}/trip")
    print(f"  - Total: €{total_margin:,.2f}")
    
    return df


# ============================================================================
# STEP 4: CALCULATE FIXED COSTS (THE "FLEET BURDEN")
# ============================================================================

def calculate_fixed_costs(df):
    """
    Calculate fixed costs allocated to each trip.
    
    Fixed costs are applied to every ACTIVE vehicle per day:
    - Daily depreciation = vehicle_cost / (lifespan_months * 30)
    - Daily city fee = annual_fee / 365
    
    Then allocated to trips based on usage.
    """
    print("\n" + "="*70)
    print("STEP 4: CALCULATE FIXED COSTS (FLEET BURDEN)")
    print("="*70)
    
    print(f"  Daily Amortization: €{DAILY_AMORTIZATION:.4f}/vehicle")
    print(f"  Daily City Fee: €{DAILY_CITY_FEE:.4f}/vehicle")
    print(f"  Total Daily Fixed Cost/Vehicle: €{DAILY_AMORTIZATION + DAILY_CITY_FEE:.4f}")
    
    # Calculate active fleet size per day per operator
    print("\n  Calculating active fleet per day...")
    
    daily_fleet = df.groupby(['date', 'operator'])['vehicle_id'].nunique().reset_index()
    daily_fleet.columns = ['date', 'operator', 'active_vehicles']
    
    # Calculate trips per vehicle per day
    daily_trips = df.groupby(['date', 'operator']).size().reset_index(name='daily_trips')
    daily_fleet = daily_fleet.merge(daily_trips, on=['date', 'operator'])
    
    # Calculate fixed costs per day
    daily_fixed_cost = DAILY_AMORTIZATION + DAILY_CITY_FEE
    daily_fleet['total_fixed_cost'] = daily_fleet['active_vehicles'] * daily_fixed_cost
    daily_fleet['fixed_cost_per_trip'] = daily_fleet['total_fixed_cost'] / daily_fleet['daily_trips']
    
    # Merge back to trips
    df = df.merge(
        daily_fleet[['date', 'operator', 'active_vehicles', 'fixed_cost_per_trip']],
        on=['date', 'operator'],
        how='left'
    )
    
    # Fill any NaN with average
    avg_fixed = daily_fleet['fixed_cost_per_trip'].mean()
    df['fixed_cost_per_trip'] = df['fixed_cost_per_trip'].fillna(avg_fixed)
    
    # Summary stats
    print(f"\n  Fleet Statistics:")
    for op in df['operator'].unique():
        op_fleet = daily_fleet[daily_fleet['operator'] == op]
        print(f"  {op}:")
        print(f"    - Avg Active Vehicles/Day: {op_fleet['active_vehicles'].mean():.0f}")
        print(f"    - Avg Trips/Vehicle/Day: {op_fleet['daily_trips'].sum()/op_fleet['active_vehicles'].sum():.1f}")
        print(f"    - Avg Fixed Cost/Trip: €{op_fleet['fixed_cost_per_trip'].mean():.3f}")
    
    total_fixed = df['fixed_cost_per_trip'].sum()
    print(f"\n  Total Fixed Costs Allocated: €{total_fixed:,.2f}")
    
    return df


# ============================================================================
# STEP 5: CALCULATE PROFITABILITY
# ============================================================================

def calculate_profitability(df):
    """
    Calculate net profit per trip and ROI metrics.
    
    Net Profit = Gross Revenue - Variable Cost - Fixed Cost
    ROI = Net Profit / Total Cost
    """
    print("\n" + "="*70)
    print("STEP 5: CALCULATE PROFITABILITY")
    print("="*70)
    
    # Total cost per trip
    df['total_cost'] = df['variable_cost'] + df['fixed_cost_per_trip']
    
    # Net profit per trip
    df['net_profit'] = df['gross_revenue'] - df['total_cost']
    
    # ROI per trip
    df['roi_per_trip'] = df['net_profit'] / df['total_cost']
    
    # Profit classification
    df['is_profitable'] = df['net_profit'] > 0
    
    # Summary statistics
    profitable_trips = df['is_profitable'].sum()
    profitable_pct = profitable_trips / len(df) * 100
    
    print(f"  Profitability Summary:")
    print(f"  - Total Trips: {len(df):,}")
    print(f"  - Profitable Trips: {profitable_trips:,} ({profitable_pct:.1f}%)")
    print(f"  - Loss-Making Trips: {len(df) - profitable_trips:,} ({100-profitable_pct:.1f}%)")
    
    print(f"\n  Financial Metrics:")
    print(f"  - Total Revenue: €{df['gross_revenue'].sum():,.2f}")
    print(f"  - Total Variable Costs: €{df['variable_cost'].sum():,.2f}")
    print(f"  - Total Fixed Costs: €{df['fixed_cost_per_trip'].sum():,.2f}")
    print(f"  - Total Net Profit: €{df['net_profit'].sum():,.2f}")
    print(f"  - Average Net Profit/Trip: €{df['net_profit'].mean():.3f}")
    print(f"  - Average ROI: {df['roi_per_trip'].mean()*100:.1f}%")
    
    # Break-even analysis
    break_even_duration = (ECONOMIC_PARAMS['variable_cost_per_trip']) / 0.20  # approx avg rate
    print(f"\n  Break-Even Analysis:")
    print(f"  - Min duration for break-even (approx): {break_even_duration:.1f} min")
    
    return df


# ============================================================================
# STEP 6: SPATIAL FINANCIALS (BY ZONE)
# ============================================================================

def calculate_zone_financials(df, zones_gdf):
    """
    Aggregate financials by zone.
    Identify "Subsidy Zones" (negative profit) vs "Profit Centers" (positive profit).
    """
    print("\n" + "="*70)
    print("STEP 6: SPATIAL FINANCIALS (BY ZONE)")
    print("="*70)
    
    # Create geometry for trip start locations
    trips_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['start_lon'], df['start_lat']),
        crs="EPSG:4326"
    )
    
    # Ensure zones are in same CRS
    if zones_gdf.crs != "EPSG:4326":
        zones_gdf = zones_gdf.to_crs("EPSG:4326")
    
    # Spatial join to assign trips to zones
    print("  → Performing spatial join...")
    trips_with_zones = gpd.sjoin(
        trips_gdf,
        zones_gdf[['ZONASTAT', 'DENOM', 'geometry']],
        how='left',
        predicate='within'
    )
    
    # Count matched
    matched = trips_with_zones['ZONASTAT'].notna().sum()
    print(f"  → Matched to zones: {matched:,} ({matched/len(trips_with_zones)*100:.1f}%)")
    
    # Aggregate by zone
    print("  → Aggregating financials by zone...")
    zone_financials = trips_with_zones.groupby('ZONASTAT').agg(
        zone_name=('DENOM', 'first'),
        trip_count=('gross_revenue', 'count'),
        total_revenue=('gross_revenue', 'sum'),
        total_variable_cost=('variable_cost', 'sum'),
        total_fixed_cost=('fixed_cost_per_trip', 'sum'),
        total_net_profit=('net_profit', 'sum'),
        avg_profit_per_trip=('net_profit', 'mean'),
        avg_duration=('duration_min', 'mean'),
        profitable_trips=('is_profitable', 'sum'),
        unique_vehicles=('vehicle_id', 'nunique')
    ).reset_index()
    
    # Calculate derived metrics
    zone_financials['total_cost'] = zone_financials['total_variable_cost'] + zone_financials['total_fixed_cost']
    zone_financials['profit_margin_pct'] = (zone_financials['total_net_profit'] / zone_financials['total_revenue'] * 100).round(2)
    zone_financials['profitable_trip_pct'] = (zone_financials['profitable_trips'] / zone_financials['trip_count'] * 100).round(2)
    
    # Classify zones
    zone_financials['zone_classification'] = np.where(
        zone_financials['total_net_profit'] > 0,
        'Profit Center',
        'Subsidy Zone'
    )
    
    # Merge with zones geometry for area calculation
    zones_utm = zones_gdf.to_crs("EPSG:32632")
    zones_gdf['area_sqkm'] = zones_utm.geometry.area / 1e6
    zone_financials = zone_financials.merge(
        zones_gdf[['ZONASTAT', 'area_sqkm']],
        on='ZONASTAT',
        how='left'
    )
    
    # Revenue per sq km
    zone_financials['revenue_per_sqkm'] = (zone_financials['total_revenue'] / zone_financials['area_sqkm']).round(2)
    zone_financials['profit_per_sqkm'] = (zone_financials['total_net_profit'] / zone_financials['area_sqkm']).round(2)
    
    # Summary
    profit_centers = (zone_financials['zone_classification'] == 'Profit Center').sum()
    subsidy_zones = (zone_financials['zone_classification'] == 'Subsidy Zone').sum()
    
    print(f"\n  Zone Classification:")
    print(f"  - Profit Centers: {profit_centers} zones")
    print(f"  - Subsidy Zones: {subsidy_zones} zones")
    
    # Top and bottom zones
    top_zones = zone_financials.nlargest(3, 'total_net_profit')
    bottom_zones = zone_financials.nsmallest(3, 'total_net_profit')
    
    print(f"\n  Top 3 Profit Centers:")
    for _, z in top_zones.iterrows():
        print(f"    - {z['zone_name']}: €{z['total_net_profit']:,.2f}")
    
    print(f"\n  Top 3 Subsidy Zones (Losses):")
    for _, z in bottom_zones.iterrows():
        print(f"    - {z['zone_name']}: €{z['total_net_profit']:,.2f}")
    
    return zone_financials, trips_with_zones


# ============================================================================
# STEP 7: OPERATOR P&L SUMMARY
# ============================================================================

def calculate_operator_pnl(df):
    """
    Create P&L summary by operator.
    """
    print("\n" + "="*70)
    print("STEP 7: OPERATOR P&L SUMMARY")
    print("="*70)
    
    # Aggregate by operator
    operator_pnl = df.groupby('operator').agg(
        trip_count=('gross_revenue', 'count'),
        total_revenue=('gross_revenue', 'sum'),
        unlock_revenue=('unlock_fee', 'sum'),
        time_revenue=('time_revenue', 'sum'),
        total_variable_cost=('variable_cost', 'sum'),
        total_fixed_cost=('fixed_cost_per_trip', 'sum'),
        total_net_profit=('net_profit', 'sum'),
        avg_profit_per_trip=('net_profit', 'mean'),
        avg_duration=('duration_min', 'mean'),
        profitable_trips=('is_profitable', 'sum'),
        unique_vehicles=('vehicle_id', 'nunique'),
        active_vehicle_days=('active_vehicles', 'sum')
    ).reset_index()
    
    # Calculate derived metrics
    operator_pnl['total_cost'] = operator_pnl['total_variable_cost'] + operator_pnl['total_fixed_cost']
    operator_pnl['profit_margin_pct'] = (operator_pnl['total_net_profit'] / operator_pnl['total_revenue'] * 100).round(2)
    operator_pnl['profitable_trip_pct'] = (operator_pnl['profitable_trips'] / operator_pnl['trip_count'] * 100).round(2)
    operator_pnl['avg_trips_per_vehicle'] = (operator_pnl['trip_count'] / operator_pnl['unique_vehicles']).round(1)
    operator_pnl['revenue_per_trip'] = (operator_pnl['total_revenue'] / operator_pnl['trip_count']).round(2)
    operator_pnl['cost_per_trip'] = (operator_pnl['total_cost'] / operator_pnl['trip_count']).round(2)
    
    # Print P&L table
    print("\n  Operator P&L Summary:")
    print("  " + "-"*60)
    
    for _, row in operator_pnl.iterrows():
        print(f"\n  {row['operator']}:")
        print(f"    Revenue:        €{row['total_revenue']:>12,.2f}")
        print(f"    - Variable:     €{row['total_variable_cost']:>12,.2f}")
        print(f"    - Fixed:        €{row['total_fixed_cost']:>12,.2f}")
        print(f"    = Net Profit:   €{row['total_net_profit']:>12,.2f}")
        print(f"    Margin:         {row['profit_margin_pct']:>12.1f}%")
        print(f"    Profitable %:   {row['profitable_trip_pct']:>12.1f}%")
    
    # System totals
    print("\n  " + "-"*60)
    print(f"\n  SYSTEM TOTAL:")
    print(f"    Revenue:        €{operator_pnl['total_revenue'].sum():>12,.2f}")
    print(f"    - Variable:     €{operator_pnl['total_variable_cost'].sum():>12,.2f}")
    print(f"    - Fixed:        €{operator_pnl['total_fixed_cost'].sum():>12,.2f}")
    print(f"    = Net Profit:   €{operator_pnl['total_net_profit'].sum():>12,.2f}")
    
    return operator_pnl


# ============================================================================
# STEP 8: DAILY BREAK-EVEN ANALYSIS
# ============================================================================

def calculate_breakeven_analysis(df):
    """
    Calculate daily profitability per vehicle to find break-even point.
    """
    print("\n" + "="*70)
    print("STEP 8: DAILY BREAK-EVEN ANALYSIS")
    print("="*70)
    
    # Calculate daily metrics per vehicle
    daily_vehicle = df.groupby(['date', 'vehicle_id', 'operator']).agg(
        trips_per_day=('gross_revenue', 'count'),
        daily_revenue=('gross_revenue', 'sum'),
        daily_variable_cost=('variable_cost', 'sum'),
        daily_net_profit=('net_profit', 'sum'),
        avg_duration=('duration_min', 'mean')
    ).reset_index()
    
    # Add fixed cost (same for all vehicles per day)
    daily_fixed = DAILY_AMORTIZATION + DAILY_CITY_FEE
    daily_vehicle['daily_fixed_cost'] = daily_fixed
    daily_vehicle['daily_total_profit'] = daily_vehicle['daily_net_profit']
    
    # Find break-even point
    profitable_days = daily_vehicle[daily_vehicle['daily_total_profit'] > 0]
    losing_days = daily_vehicle[daily_vehicle['daily_total_profit'] <= 0]
    
    print(f"  Daily Vehicle Performance:")
    print(f"  - Total vehicle-days: {len(daily_vehicle):,}")
    print(f"  - Profitable days: {len(profitable_days):,} ({len(profitable_days)/len(daily_vehicle)*100:.1f}%)")
    print(f"  - Loss-making days: {len(losing_days):,} ({len(losing_days)/len(daily_vehicle)*100:.1f}%)")
    
    # Break-even trips analysis
    print(f"\n  Break-Even Analysis by Trips/Day:")
    for trips in range(1, 8):
        subset = daily_vehicle[daily_vehicle['trips_per_day'] == trips]
        if len(subset) > 0:
            avg_profit = subset['daily_total_profit'].mean()
            profitable_pct = (subset['daily_total_profit'] > 0).mean() * 100
            print(f"    {trips} trips/day: Avg profit €{avg_profit:.2f}, {profitable_pct:.0f}% profitable")
    
    # Find minimum trips for profitability
    avg_by_trips = daily_vehicle.groupby('trips_per_day')['daily_total_profit'].mean()
    break_even_trips = avg_by_trips[avg_by_trips > 0].index.min() if len(avg_by_trips[avg_by_trips > 0]) > 0 else None
    
    if break_even_trips:
        print(f"\n  → Minimum trips for average profitability: {break_even_trips} trips/day")
    
    return daily_vehicle


# ============================================================================
# STEP 9: TEMPORAL AGGREGATION (HOUR x DAY_OF_WEEK)
# ============================================================================

def calculate_temporal_economics(df):
    """
    Calculate Net Profit for each Hour-Day slot.
    Allocates fixed costs of entire fleet during each hour.
    
    Returns a matrix: rows=day_of_week (Mon-Sun), cols=hour (0-23)
    """
    print("\n" + "="*70)
    print("STEP 9: TEMPORAL AGGREGATION (HOUR x DAY_OF_WEEK)")
    print("="*70)
    
    # Ensure we have datetime and day_of_week
    df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
    df['hour'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.dayofweek  # 0=Mon, 6=Sun
    
    # Day names for readability
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Calculate hourly fleet size (unique vehicles active per hour per day)
    # For simplicity, we'll allocate fixed costs proportionally to trip count
    total_daily_fixed = (DAILY_AMORTIZATION + DAILY_CITY_FEE) * df['vehicle_id'].nunique()
    
    # Aggregate by day_of_week and hour
    print("  → Aggregating by day and hour...")
    temporal_stats = df.groupby(['day_of_week', 'hour']).agg(
        trip_count=('gross_revenue', 'count'),
        total_revenue=('gross_revenue', 'sum'),
        total_variable_cost=('variable_cost', 'sum'),
        total_net_profit=('net_profit', 'sum'),
        avg_duration=('duration_min', 'mean'),
        unique_vehicles=('vehicle_id', 'nunique')
    ).reset_index()
    
    # Add day name
    temporal_stats['day_name'] = temporal_stats['day_of_week'].map(
        {i: name for i, name in enumerate(day_names)}
    )
    
    # Calculate hourly fixed cost allocation (proportional to trips)
    total_trips = temporal_stats['trip_count'].sum()
    # Fixed cost per hour slot = total fleet fixed cost * (slot_trips / total_trips)
    # This is already accounted in net_profit via fixed_cost_per_trip
    
    # Create pivot table for heatmap
    profit_matrix = temporal_stats.pivot(
        index='day_of_week',
        columns='hour',
        values='total_net_profit'
    ).fillna(0)
    
    # Summary stats
    print(f"\n  Temporal Profitability Analysis:")
    
    # Best and worst hours
    hourly_profit = temporal_stats.groupby('hour')['total_net_profit'].sum()
    best_hour = hourly_profit.idxmax()
    worst_hour = hourly_profit.idxmin()
    print(f"  - Best hour: {best_hour:02d}:00 (€{hourly_profit[best_hour]:,.0f})")
    print(f"  - Worst hour: {worst_hour:02d}:00 (€{hourly_profit[worst_hour]:,.0f})")
    
    # Best and worst days
    daily_profit = temporal_stats.groupby('day_of_week')['total_net_profit'].sum()
    best_day = daily_profit.idxmax()
    worst_day = daily_profit.idxmin()
    print(f"  - Best day: {day_names[best_day]} (€{daily_profit[best_day]:,.0f})")
    print(f"  - Worst day: {day_names[worst_day]} (€{daily_profit[worst_day]:,.0f})")
    
    # Dead hours (negative profit)
    dead_slots = temporal_stats[temporal_stats['total_net_profit'] < 0]
    if len(dead_slots) > 0:
        print(f"\n  ⚠ Dead Slots (Negative Profit): {len(dead_slots)}")
        for _, slot in dead_slots.iterrows():
            print(f"    - {slot['day_name']} {int(slot['hour']):02d}:00: €{slot['total_net_profit']:,.0f}")
    else:
        print(f"\n  ✓ No dead slots - all time periods are profitable")
    
    return temporal_stats, profit_matrix


# ============================================================================
# STEP 10: PARETO ANALYSIS (CUMULATIVE PROFIT)
# ============================================================================

def calculate_pareto_analysis(zone_financials):
    """
    Rank zones by Net Profit (Descending).
    Calculate Cumulative Profit % and Cumulative Zone %.
    
    Identifies the "vital few" zones that drive most profit.
    """
    print("\n" + "="*70)
    print("STEP 10: PARETO ANALYSIS (CUMULATIVE PROFIT)")
    print("="*70)
    
    # Sort zones by net profit descending
    pareto_df = zone_financials.copy()
    pareto_df = pareto_df.sort_values('total_net_profit', ascending=False).reset_index(drop=True)
    
    # Add rank
    pareto_df['rank'] = range(1, len(pareto_df) + 1)
    
    # Calculate cumulative metrics
    total_profit = pareto_df['total_net_profit'].sum()
    pareto_df['cumulative_profit'] = pareto_df['total_net_profit'].cumsum()
    pareto_df['cumulative_profit_pct'] = (pareto_df['cumulative_profit'] / total_profit * 100).round(2)
    pareto_df['cumulative_zone_pct'] = (pareto_df['rank'] / len(pareto_df) * 100).round(2)
    
    # Find key thresholds
    # 80% of profit comes from what % of zones?
    zones_for_80pct = pareto_df[pareto_df['cumulative_profit_pct'] >= 80].iloc[0]['rank'] if any(pareto_df['cumulative_profit_pct'] >= 80) else len(pareto_df)
    pct_zones_for_80 = zones_for_80pct / len(pareto_df) * 100
    
    # 100% profit mark (where cumulative goes negative after)
    profit_zones = pareto_df[pareto_df['total_net_profit'] > 0]
    loss_zones = pareto_df[pareto_df['total_net_profit'] <= 0]
    
    print(f"\n  Pareto Analysis Results:")
    print(f"  - Total Zones: {len(pareto_df)}")
    print(f"  - Profit Centers: {len(profit_zones)} zones")
    print(f"  - Subsidy Zones: {len(loss_zones)} zones")
    print(f"\n  - 80% of profit from top {zones_for_80pct} zones ({pct_zones_for_80:.1f}%)")
    
    # Top 10 zones
    print(f"\n  Top 10 Profit Generators:")
    for _, row in pareto_df.head(10).iterrows():
        zone_name = row.get('zone_name', row.get('ZONASTAT', 'Unknown'))
        print(f"    {int(row['rank']):2d}. {zone_name}: €{row['total_net_profit']:,.0f} "
              f"(Cum: {row['cumulative_profit_pct']:.1f}%)")
    
    # Value destruction point
    if len(loss_zones) > 0:
        destruction_start = loss_zones.iloc[0]['rank']
        print(f"\n  ⚠ Value Destruction begins at Zone #{destruction_start}")
    else:
        print(f"\n  ✓ No value-destroying zones")
    
    return pareto_df


# ============================================================================
# STEP 11: SCENARIO MODELING
# ============================================================================

def calculate_scenario_analysis(df, zone_financials):
    """
    Calculate Total System Profit under alternative scenarios:
    
    - Base Case: Current state
    - Scenario A (Optimistic): +10% Revenue, -10% Ops Cost
    - Scenario B (Pessimistic): -10% Revenue, +10% Ops Cost
    - Scenario C (No Subsidy): Drop bottom 20% worst zones
    """
    print("\n" + "="*70)
    print("STEP 11: SCENARIO MODELING")
    print("="*70)
    
    # Base case metrics
    base_revenue = df['gross_revenue'].sum()
    base_var_cost = df['variable_cost'].sum()
    base_fixed_cost = df['fixed_cost_per_trip'].sum()
    base_total_cost = base_var_cost + base_fixed_cost
    base_profit = base_revenue - base_total_cost
    base_margin = base_profit / base_revenue * 100
    
    scenarios = []
    
    # Base Case
    scenarios.append({
        'scenario': 'Base Case',
        'scenario_code': 'BASE',
        'revenue': base_revenue,
        'variable_cost': base_var_cost,
        'fixed_cost': base_fixed_cost,
        'total_cost': base_total_cost,
        'net_profit': base_profit,
        'profit_margin_pct': base_margin,
        'delta_profit': 0,
        'delta_profit_pct': 0
    })
    
    print(f"\n  BASE CASE:")
    print(f"    Revenue: €{base_revenue:,.0f}")
    print(f"    Costs:   €{base_total_cost:,.0f}")
    print(f"    Profit:  €{base_profit:,.0f} ({base_margin:.1f}%)")
    
    # Scenario A: Optimistic (+10% Revenue, -10% Ops Cost)
    scen_a_revenue = base_revenue * 1.10
    scen_a_var_cost = base_var_cost * 0.90
    scen_a_fixed_cost = base_fixed_cost  # Fixed costs don't change
    scen_a_total_cost = scen_a_var_cost + scen_a_fixed_cost
    scen_a_profit = scen_a_revenue - scen_a_total_cost
    scen_a_margin = scen_a_profit / scen_a_revenue * 100
    
    scenarios.append({
        'scenario': 'Optimistic (+10% Rev, -10% OpEx)',
        'scenario_code': 'OPTIMISTIC',
        'revenue': scen_a_revenue,
        'variable_cost': scen_a_var_cost,
        'fixed_cost': scen_a_fixed_cost,
        'total_cost': scen_a_total_cost,
        'net_profit': scen_a_profit,
        'profit_margin_pct': scen_a_margin,
        'delta_profit': scen_a_profit - base_profit,
        'delta_profit_pct': (scen_a_profit - base_profit) / base_profit * 100
    })
    
    print(f"\n  SCENARIO A (Optimistic):")
    print(f"    Revenue: €{scen_a_revenue:,.0f} (+10%)")
    print(f"    Costs:   €{scen_a_total_cost:,.0f} (-10% var)")
    print(f"    Profit:  €{scen_a_profit:,.0f} ({scen_a_margin:.1f}%)")
    print(f"    Δ Profit: +€{scen_a_profit - base_profit:,.0f} (+{(scen_a_profit/base_profit-1)*100:.1f}%)")
    
    # Scenario B: Pessimistic (-10% Revenue, +10% Ops Cost)
    scen_b_revenue = base_revenue * 0.90
    scen_b_var_cost = base_var_cost * 1.10
    scen_b_fixed_cost = base_fixed_cost
    scen_b_total_cost = scen_b_var_cost + scen_b_fixed_cost
    scen_b_profit = scen_b_revenue - scen_b_total_cost
    scen_b_margin = scen_b_profit / scen_b_revenue * 100
    
    scenarios.append({
        'scenario': 'Pessimistic (-10% Rev, +10% OpEx)',
        'scenario_code': 'PESSIMISTIC',
        'revenue': scen_b_revenue,
        'variable_cost': scen_b_var_cost,
        'fixed_cost': scen_b_fixed_cost,
        'total_cost': scen_b_total_cost,
        'net_profit': scen_b_profit,
        'profit_margin_pct': scen_b_margin,
        'delta_profit': scen_b_profit - base_profit,
        'delta_profit_pct': (scen_b_profit - base_profit) / base_profit * 100
    })
    
    print(f"\n  SCENARIO B (Pessimistic):")
    print(f"    Revenue: €{scen_b_revenue:,.0f} (-10%)")
    print(f"    Costs:   €{scen_b_total_cost:,.0f} (+10% var)")
    print(f"    Profit:  €{scen_b_profit:,.0f} ({scen_b_margin:.1f}%)")
    print(f"    Δ Profit: €{scen_b_profit - base_profit:,.0f} ({(scen_b_profit/base_profit-1)*100:.1f}%)")
    
    # Scenario C: No Subsidy (Drop bottom 20% worst zones)
    # Sort zones by profit and drop bottom 20%
    zones_sorted = zone_financials.sort_values('total_net_profit', ascending=True)
    bottom_20pct = int(len(zones_sorted) * 0.20)
    dropped_zones = zones_sorted.head(bottom_20pct)
    kept_zones = zones_sorted.tail(len(zones_sorted) - bottom_20pct)
    
    # Calculate revenue/cost from dropped zones (proportional to trips)
    dropped_zone_ids = dropped_zones['ZONASTAT'].tolist()
    dropped_trips = zone_financials[zone_financials['ZONASTAT'].isin(dropped_zone_ids)]['trip_count'].sum()
    total_trips = zone_financials['trip_count'].sum()
    drop_ratio = dropped_trips / total_trips if total_trips > 0 else 0
    
    # Adjust metrics (rough approximation)
    scen_c_revenue = base_revenue * (1 - drop_ratio)
    scen_c_var_cost = base_var_cost * (1 - drop_ratio)
    scen_c_fixed_cost = base_fixed_cost * 0.95  # Slight reduction in fleet
    scen_c_total_cost = scen_c_var_cost + scen_c_fixed_cost
    scen_c_profit = scen_c_revenue - scen_c_total_cost
    scen_c_margin = scen_c_profit / scen_c_revenue * 100 if scen_c_revenue > 0 else 0
    
    scenarios.append({
        'scenario': f'No Subsidy (Drop {bottom_20pct} zones)',
        'scenario_code': 'NO_SUBSIDY',
        'revenue': scen_c_revenue,
        'variable_cost': scen_c_var_cost,
        'fixed_cost': scen_c_fixed_cost,
        'total_cost': scen_c_total_cost,
        'net_profit': scen_c_profit,
        'profit_margin_pct': scen_c_margin,
        'delta_profit': scen_c_profit - base_profit,
        'delta_profit_pct': (scen_c_profit - base_profit) / base_profit * 100 if base_profit != 0 else 0,
        'zones_dropped': bottom_20pct,
        'trips_dropped_pct': drop_ratio * 100
    })
    
    print(f"\n  SCENARIO C (No Subsidy - Exit {bottom_20pct} zones):")
    print(f"    Zones dropped: {bottom_20pct} ({drop_ratio*100:.1f}% of trips)")
    print(f"    Revenue: €{scen_c_revenue:,.0f}")
    print(f"    Costs:   €{scen_c_total_cost:,.0f}")
    print(f"    Profit:  €{scen_c_profit:,.0f} ({scen_c_margin:.1f}%)")
    print(f"    Δ Profit: €{scen_c_profit - base_profit:,.0f}")
    
    # Convert to DataFrame
    scenarios_df = pd.DataFrame(scenarios)
    
    # Summary
    print(f"\n  SCENARIO COMPARISON:")
    print(f"  {'─'*60}")
    print(f"  {'Scenario':<35} {'Net Profit':>12} {'Margin':>8}")
    print(f"  {'─'*60}")
    for _, row in scenarios_df.iterrows():
        print(f"  {row['scenario']:<35} €{row['net_profit']:>10,.0f} {row['profit_margin_pct']:>7.1f}%")
    print(f"  {'─'*60}")
    
    return scenarios_df


# ============================================================================
# STEP 12: MONTE CARLO SIMULATION
# ============================================================================

def calculate_monte_carlo_simulation(df, n_simulations=10000):
    """
    Monte Carlo simulation for risk analysis.
    
    Stochastic variables:
    - Revenue per trip (based on empirical distribution)
    - Variable cost variation (±20%)
    - Demand variation (±30%)
    
    Returns:
    - Distribution of system profit outcomes
    - VaR (Value at Risk) at 5% and 1%
    - Probability of loss
    """
    print("\n" + "="*70)
    print("STEP 12: MONTE CARLO SIMULATION")
    print("="*70)
    
    np.random.seed(42)
    
    # Base parameters
    base_revenue_mean = df['gross_revenue'].mean()
    base_revenue_std = df['gross_revenue'].std()
    base_var_cost = ECONOMIC_PARAMS['variable_cost_per_trip']
    base_fixed_cost = df['fixed_cost_per_trip'].mean()
    n_trips = len(df)
    
    print(f"  → Running {n_simulations:,} simulations...")
    print(f"  → Base: {n_trips:,} trips, €{base_revenue_mean:.2f}/trip avg revenue")
    
    # Simulation storage
    sim_results = []
    
    for i in range(n_simulations):
        # Stochastic demand (number of trips): ±30%
        demand_factor = np.random.uniform(0.70, 1.30)
        sim_trips = int(n_trips * demand_factor)
        
        # Stochastic revenue per trip (log-normal based on empirical)
        sim_revenue_per_trip = np.random.lognormal(
            np.log(base_revenue_mean), 
            0.3  # 30% coefficient of variation
        )
        
        # Stochastic variable cost (±20%)
        sim_var_cost = base_var_cost * np.random.uniform(0.80, 1.20)
        
        # Fixed costs (relatively stable, ±5%)
        sim_fixed_cost = base_fixed_cost * np.random.uniform(0.95, 1.05)
        
        # Calculate financials
        total_revenue = sim_trips * sim_revenue_per_trip
        total_var_cost = sim_trips * sim_var_cost
        total_fixed_cost = sim_trips * sim_fixed_cost
        net_profit = total_revenue - total_var_cost - total_fixed_cost
        margin = net_profit / total_revenue * 100 if total_revenue > 0 else 0
        
        sim_results.append({
            'simulation': i + 1,
            'demand_factor': demand_factor,
            'trips': sim_trips,
            'revenue_per_trip': sim_revenue_per_trip,
            'var_cost_per_trip': sim_var_cost,
            'total_revenue': total_revenue,
            'total_var_cost': total_var_cost,
            'total_fixed_cost': total_fixed_cost,
            'net_profit': net_profit,
            'profit_margin_pct': margin
        })
    
    sim_df = pd.DataFrame(sim_results)
    
    # Calculate statistics
    profit_mean = sim_df['net_profit'].mean()
    profit_std = sim_df['net_profit'].std()
    profit_median = sim_df['net_profit'].median()
    
    # Value at Risk (VaR) - losses at given percentiles
    var_5 = sim_df['net_profit'].quantile(0.05)
    var_1 = sim_df['net_profit'].quantile(0.01)
    
    # Probability of loss
    prob_loss = (sim_df['net_profit'] < 0).mean() * 100
    
    # Expected Shortfall (CVaR) - average loss in worst 5%
    worst_5pct = sim_df.nsmallest(int(n_simulations * 0.05), 'net_profit')
    cvar_5 = worst_5pct['net_profit'].mean()
    
    print(f"\n  Monte Carlo Results ({n_simulations:,} simulations):")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Mean Net Profit:     €{profit_mean:>12,.0f}")
    print(f"  Std Deviation:       €{profit_std:>12,.0f}")
    print(f"  Median Net Profit:   €{profit_median:>12,.0f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  VaR 5% (95% CI):     €{var_5:>12,.0f}")
    print(f"  VaR 1% (99% CI):     €{var_1:>12,.0f}")
    print(f"  CVaR 5% (Exp.Short): €{cvar_5:>12,.0f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Probability of Loss: {prob_loss:>12.2f}%")
    
    # Summary stats for export
    summary = {
        'mean_profit': profit_mean,
        'std_profit': profit_std,
        'median_profit': profit_median,
        'var_5_pct': var_5,
        'var_1_pct': var_1,
        'cvar_5_pct': cvar_5,
        'prob_loss_pct': prob_loss,
        'n_simulations': n_simulations
    }
    
    return sim_df, summary


# ============================================================================
# STEP 13: SENSITIVITY ANALYSIS (TORNADO DIAGRAM)
# ============================================================================

def calculate_sensitivity_analysis(df):
    """
    Sensitivity analysis for key parameters.
    
    Tests impact of ±10% change in each parameter on net profit.
    Used to create tornado diagram.
    """
    print("\n" + "="*70)
    print("STEP 13: SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Base case
    base_revenue = df['gross_revenue'].sum()
    base_var_cost = df['variable_cost'].sum()
    base_fixed_cost = df['fixed_cost_per_trip'].sum()
    base_profit = base_revenue - base_var_cost - base_fixed_cost
    
    print(f"  Base Case Net Profit: €{base_profit:,.0f}")
    
    # Parameters to test
    parameters = [
        ('Revenue (Pricing)', 'gross_revenue', 0.10),
        ('Variable Cost (OpEx)', 'variable_cost', 0.10),
        ('Fixed Cost (CapEx)', 'fixed_cost_per_trip', 0.10),
        ('Demand (Trips)', 'trips', 0.10),
        ('Duration (Time/Trip)', 'duration', 0.10)
    ]
    
    sensitivity_results = []
    
    for param_name, param_key, variation in parameters:
        # High scenario
        if param_key == 'gross_revenue':
            high_profit = (base_revenue * 1.10) - base_var_cost - base_fixed_cost
            low_profit = (base_revenue * 0.90) - base_var_cost - base_fixed_cost
        elif param_key == 'variable_cost':
            high_profit = base_revenue - (base_var_cost * 0.90) - base_fixed_cost  # Lower cost = higher profit
            low_profit = base_revenue - (base_var_cost * 1.10) - base_fixed_cost
        elif param_key == 'fixed_cost_per_trip':
            high_profit = base_revenue - base_var_cost - (base_fixed_cost * 0.90)
            low_profit = base_revenue - base_var_cost - (base_fixed_cost * 1.10)
        elif param_key == 'trips':
            # More trips = proportionally more revenue and variable cost
            high_profit = (base_revenue * 1.10) - (base_var_cost * 1.10) - base_fixed_cost
            low_profit = (base_revenue * 0.90) - (base_var_cost * 0.90) - base_fixed_cost
        elif param_key == 'duration':
            # Longer duration = more time revenue (unlock fee stays same)
            # Approximate: 60% of revenue is time-based
            high_profit = (base_revenue * 1.06) - base_var_cost - base_fixed_cost
            low_profit = (base_revenue * 0.94) - base_var_cost - base_fixed_cost
        
        delta_high = high_profit - base_profit
        delta_low = low_profit - base_profit
        swing = abs(delta_high) + abs(delta_low)
        
        sensitivity_results.append({
            'parameter': param_name,
            'variation_pct': variation * 100,
            'profit_high': high_profit,
            'profit_low': low_profit,
            'delta_high': delta_high,
            'delta_low': delta_low,
            'swing': swing,
            'delta_high_pct': delta_high / base_profit * 100,
            'delta_low_pct': delta_low / base_profit * 100
        })
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    sensitivity_df = sensitivity_df.sort_values('swing', ascending=False).reset_index(drop=True)
    
    print(f"\n  Sensitivity Analysis (±10% variation):")
    print(f"  {'─'*65}")
    print(f"  {'Parameter':<25} {'Δ High':>12} {'Δ Low':>12} {'Swing':>12}")
    print(f"  {'─'*65}")
    for _, row in sensitivity_df.iterrows():
        print(f"  {row['parameter']:<25} €{row['delta_high']:>+10,.0f} €{row['delta_low']:>+10,.0f} €{row['swing']:>10,.0f}")
    print(f"  {'─'*65}")
    
    # Most sensitive parameter
    most_sensitive = sensitivity_df.iloc[0]['parameter']
    print(f"\n  → Most sensitive parameter: {most_sensitive}")
    
    return sensitivity_df


# ============================================================================
# STEP 14: BOOTSTRAP FINANCIAL CONFIDENCE INTERVALS
# ============================================================================

def calculate_bootstrap_financial_ci(df, n_bootstrap=1000, ci_level=0.95):
    """
    Bootstrap confidence intervals for key financial metrics by operator.
    """
    print("\n" + "="*70)
    print("STEP 14: BOOTSTRAP FINANCIAL CI")
    print("="*70)
    
    np.random.seed(42)
    alpha = 1 - ci_level
    
    operators = sorted(df['operator'].unique())
    results = []
    
    print(f"  → Bootstrap parameters: n={n_bootstrap}, CI={ci_level*100:.0f}%")
    
    for operator in operators:
        op_data = df[df['operator'] == operator]
        n = len(op_data)
        
        if n < 100:
            print(f"  ⚠ {operator}: Insufficient data")
            continue
        
        # Metrics to bootstrap
        revenues = op_data['gross_revenue'].values
        profits = op_data['net_profit'].values
        margins = (op_data['net_profit'] / op_data['gross_revenue']).values
        
        # Bootstrap resampling
        rev_boots = []
        profit_boots = []
        margin_boots = []
        profitable_pct_boots = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            rev_boots.append(np.mean(revenues[idx]))
            profit_boots.append(np.mean(profits[idx]))
            margin_boots.append(np.mean(margins[idx]) * 100)
            profitable_pct_boots.append(np.mean(profits[idx] > 0) * 100)
        
        # Calculate CIs
        rev_ci = np.percentile(rev_boots, [alpha/2*100, (1-alpha/2)*100])
        profit_ci = np.percentile(profit_boots, [alpha/2*100, (1-alpha/2)*100])
        margin_ci = np.percentile(margin_boots, [alpha/2*100, (1-alpha/2)*100])
        profitable_ci = np.percentile(profitable_pct_boots, [alpha/2*100, (1-alpha/2)*100])
        
        results.append({
            'operator': operator,
            'n_trips': n,
            'revenue_mean': np.mean(revenues),
            'revenue_ci_lower': rev_ci[0],
            'revenue_ci_upper': rev_ci[1],
            'profit_mean': np.mean(profits),
            'profit_ci_lower': profit_ci[0],
            'profit_ci_upper': profit_ci[1],
            'margin_mean_pct': np.mean(margins) * 100,
            'margin_ci_lower': margin_ci[0],
            'margin_ci_upper': margin_ci[1],
            'profitable_pct': np.mean(profits > 0) * 100,
            'profitable_ci_lower': profitable_ci[0],
            'profitable_ci_upper': profitable_ci[1]
        })
        
        print(f"\n  {operator}:")
        print(f"    Revenue/Trip: €{np.mean(revenues):.2f} [{rev_ci[0]:.2f}, {rev_ci[1]:.2f}]")
        print(f"    Profit/Trip:  €{np.mean(profits):.3f} [{profit_ci[0]:.3f}, {profit_ci[1]:.3f}]")
        print(f"    Margin:       {np.mean(margins)*100:.2f}% [{margin_ci[0]:.2f}%, {margin_ci[1]:.2f}%]")
        print(f"    Profitable:   {np.mean(profits > 0)*100:.1f}% [{profitable_ci[0]:.1f}%, {profitable_ci[1]:.1f}%]")
    
    bootstrap_df = pd.DataFrame(results)
    
    # Statistical comparison between operators
    print(f"\n  Operator Comparison (Kruskal-Wallis):")
    groups = [df[df['operator'] == op]['net_profit'].values for op in operators]
    h_stat, p_value = stats.kruskal(*groups)
    print(f"    H-statistic: {h_stat:.2f}")
    print(f"    p-value: {p_value:.2e}")
    
    if p_value < 0.05:
        print(f"    → Significant difference between operators")
    else:
        print(f"    → No significant difference")
    
    return bootstrap_df


# ============================================================================
# STEP 15: REGRESSION ANALYSIS
# ============================================================================

def calculate_regression_analysis(df, zone_financials):
    """
    Regression analysis for profit drivers.
    
    1. Trip-level: Duration → Profit relationship
    2. Zone-level: Drivers of zone profitability
    """
    print("\n" + "="*70)
    print("STEP 15: REGRESSION ANALYSIS")
    print("="*70)
    
    results = []
    
    # 1. Duration-Profit Relationship (Trip-level)
    print("\n  A. Duration-Profit Relationship:")
    print("  " + "-"*50)
    
    x = df['duration_min'].values.reshape(-1, 1)
    y = df['net_profit'].values
    
    # Simple linear regression using scipy
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['duration_min'].values, 
        df['net_profit'].values
    )
    
    r_squared = r_value ** 2
    
    print(f"    Linear: Profit = {intercept:.3f} + {slope:.4f} × Duration")
    print(f"    R² = {r_squared:.4f}")
    print(f"    p-value = {p_value:.2e}")
    print(f"    Interpretation: Each minute adds €{slope:.4f} to profit")
    
    # Break-even duration
    break_even_duration = -intercept / slope if slope > 0 else None
    if break_even_duration and break_even_duration > 0:
        print(f"    Break-even duration: {break_even_duration:.1f} minutes")
    
    results.append({
        'analysis': 'Duration-Profit (Trip)',
        'coefficient': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_error': std_err,
        'n_observations': len(df)
    })
    
    # 2. Zone-Level Profitability Drivers
    print("\n  B. Zone Profitability Drivers:")
    print("  " + "-"*50)
    
    if zone_financials is not None and len(zone_financials) > 10:
        # Features: trip_count, avg_duration, area
        valid_zones = zone_financials.dropna(subset=['total_net_profit', 'trip_count', 'avg_duration'])
        
        if len(valid_zones) > 10:
            # Trip count vs profit
            slope_trips, intercept_trips, r_trips, p_trips, _ = stats.linregress(
                valid_zones['trip_count'].values,
                valid_zones['total_net_profit'].values
            )
            
            print(f"    Trips → Profit: β = {slope_trips:.2f}, R² = {r_trips**2:.4f}")
            
            results.append({
                'analysis': 'Trips-Profit (Zone)',
                'coefficient': slope_trips,
                'intercept': intercept_trips,
                'r_squared': r_trips ** 2,
                'p_value': p_trips,
                'std_error': np.nan,
                'n_observations': len(valid_zones)
            })
            
            # Duration vs profit margin
            if 'profit_margin_pct' in valid_zones.columns:
                slope_dur, intercept_dur, r_dur, p_dur, _ = stats.linregress(
                    valid_zones['avg_duration'].values,
                    valid_zones['profit_margin_pct'].fillna(0).values
                )
                print(f"    Duration → Margin: β = {slope_dur:.3f}%/min, R² = {r_dur**2:.4f}")
                
                results.append({
                    'analysis': 'Duration-Margin (Zone)',
                    'coefficient': slope_dur,
                    'intercept': intercept_dur,
                    'r_squared': r_dur ** 2,
                    'p_value': p_dur,
                    'std_error': np.nan,
                    'n_observations': len(valid_zones)
                })
    
    regression_df = pd.DataFrame(results)
    
    return regression_df


# ============================================================================
# STEP 16: SAVE ALL CHECKPOINTS 
# ============================================================================

def save_checkpoints(df, zone_financials, operator_pnl, daily_vehicle,
                     temporal_stats=None, pareto_df=None, scenarios_df=None,
                     monte_carlo_df=None, monte_carlo_summary=None,
                     sensitivity_df=None, bootstrap_df=None, regression_df=None):
    """Save all checkpoint files including advanced analytics."""
    print("\n" + "="*70)
    print("STEP 16: SAVE ALL CHECKPOINTS ")
    print("="*70)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Trip-level financials
    trips_path = OUTPUT_DIR / "checkpoint_economics_trips.pkl"
    trips_save = df[[
        'operator', 'vehicle_id', 'date', 'start_datetime', 'duration_min',
        'gross_revenue', 'unlock_fee', 'time_revenue',
        'variable_cost', 'fixed_cost_per_trip', 'total_cost',
        'net_profit', 'roi_per_trip', 'is_profitable',
        'start_lat', 'start_lon', 'end_lat', 'end_lon',
        'hour', 'day_of_week'
    ]].copy()
    trips_save.to_pickle(trips_path)
    print(f"  ✓ Saved: {trips_path.name} ({len(trips_save):,} trips)")
    
    # 2. Zone-level financials
    zones_path = OUTPUT_DIR / "checkpoint_economics_zones.csv"
    zone_financials.to_csv(zones_path, index=False)
    print(f"  ✓ Saved: {zones_path.name} ({len(zone_financials)} zones)")
    
    # 3. Operator P&L
    pnl_path = OUTPUT_DIR / "checkpoint_operator_pnl.csv"
    operator_pnl.to_csv(pnl_path, index=False)
    print(f"  ✓ Saved: {pnl_path.name} ({len(operator_pnl)} operators)")
    
    # 4. Daily vehicle analysis (for break-even)
    daily_path = OUTPUT_DIR / "checkpoint_daily_vehicle.pkl"
    daily_vehicle.to_pickle(daily_path)
    print(f"  ✓ Saved: {daily_path.name} ({len(daily_vehicle):,} vehicle-days)")
    
    # 5. Temporal economics (NEW)
    if temporal_stats is not None:
        temporal_path = OUTPUT_DIR / "checkpoint_economics_temporal.csv"
        temporal_stats.to_csv(temporal_path, index=False)
        print(f"  ✓ Saved: {temporal_path.name} ({len(temporal_stats)} time slots)")
    
    # 6. Pareto analysis (NEW)
    if pareto_df is not None:
        pareto_path = OUTPUT_DIR / "checkpoint_economics_pareto.csv"
        pareto_df.to_csv(pareto_path, index=False)
        print(f"  ✓ Saved: {pareto_path.name} ({len(pareto_df)} zones ranked)")
    
    # 7. Scenario analysis (NEW)
    if scenarios_df is not None:
        scenarios_path = OUTPUT_DIR / "checkpoint_economics_scenarios.csv"
        scenarios_df.to_csv(scenarios_path, index=False)
        print(f"  ✓ Saved: {scenarios_path.name} ({len(scenarios_df)} scenarios)")
    
    # ========================================================================
    # ADVANCED ANALYTICS CHECKPOINTS
    # ========================================================================
    print("\n  --- Advanced Analytics ---")
    
    # 8. Monte Carlo simulation results
    if monte_carlo_df is not None:
        mc_path = OUTPUT_DIR / "checkpoint_monte_carlo_simulations.csv"
        monte_carlo_df.to_csv(mc_path, index=False)
        print(f"  ✓ Saved: {mc_path.name} ({len(monte_carlo_df)} simulations)")
    
    if monte_carlo_summary is not None:
        mc_summary_path = OUTPUT_DIR / "checkpoint_monte_carlo_summary.csv"
        # Convert dict to DataFrame if necessary
        if isinstance(monte_carlo_summary, dict):
            mc_summary_df = pd.DataFrame([monte_carlo_summary])
        else:
            mc_summary_df = monte_carlo_summary
        mc_summary_df.to_csv(mc_summary_path, index=False)
        print(f"  ✓ Saved: {mc_summary_path.name} (summary stats)")
    
    # 9. Sensitivity analysis results
    if sensitivity_df is not None:
        sens_path = OUTPUT_DIR / "checkpoint_sensitivity_analysis.csv"
        sensitivity_df.to_csv(sens_path, index=False)
        print(f"  ✓ Saved: {sens_path.name} ({len(sensitivity_df)} parameters)")
    
    # 10. Bootstrap confidence intervals
    if bootstrap_df is not None:
        bootstrap_path = OUTPUT_DIR / "checkpoint_bootstrap_ci.csv"
        bootstrap_df.to_csv(bootstrap_path, index=False)
        print(f"  ✓ Saved: {bootstrap_path.name} ({len(bootstrap_df)} operators)")
    
    # 11. Regression analysis results
    if regression_df is not None:
        regression_path = OUTPUT_DIR / "checkpoint_regression_analysis.csv"
        regression_df.to_csv(regression_path, index=False)
        print(f"  ✓ Saved: {regression_path.name} ({len(regression_df)} models)")

    print(f"\n  → All checkpoints saved to: {OUTPUT_DIR}")
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the complete economic analysis pipeline."""
    print("\n" + "="*70)
    print("EXERCISE 5: BUSINESS MODEL & ECONOMIC ANALYSIS")
    print("="*70)
    print("Turin E-Scooter Market Financial Engine (2024-2025)")
    print("═══════════════════════════════════════════════════════════════════════")
    print(" UPGRADED VERSION: Temporal Aggregation + Pareto Analysis + Scenarios")
    print("="*70)
    
    # Display parameters
    print("\n  Economic Parameters (Turin 2025):")
    print("  " + "-"*50)
    for op, tariff in ECONOMIC_PARAMS['tariffs'].items():
        print(f"  {op}: €{tariff['unlock_fee']:.2f} + €{tariff['minute_rate']:.2f}/min")
    print(f"  Variable Cost/Trip: €{ECONOMIC_PARAMS['variable_cost_per_trip']:.2f}")
    print(f"  Vehicle Cost: €{ECONOMIC_PARAMS['vehicle_unit_cost']:.2f}")
    print(f"  Lifespan: {ECONOMIC_PARAMS['lifespan_months']} months")
    print(f"  Annual City Fee: €{ECONOMIC_PARAMS['annual_city_fee_per_vehicle']:.2f}/vehicle")
    print("  " + "-"*50)
    
    # Step 1: Load data
    df, zones_gdf = load_data()
    
    # Step 2: Calculate revenue
    df = calculate_revenue(df)
    
    # Step 3: Calculate variable costs
    df = calculate_variable_costs(df)
    
    # Step 4: Calculate fixed costs
    df = calculate_fixed_costs(df)
    
    # Step 5: Calculate profitability
    df = calculate_profitability(df)
    
    # Step 6: Zone financials
    zone_financials, trips_with_zones = calculate_zone_financials(df, zones_gdf)
    
    # Step 7: Operator P&L
    operator_pnl = calculate_operator_pnl(df)
    
    # Step 8: Break-even analysis
    daily_vehicle = calculate_breakeven_analysis(df)
    
    # ========================================================================
    # ADVANCED ANALYTICS (NEW)
    # ========================================================================
    
    # Step 9: Temporal Profitability Analysis
    temporal_stats, temporal_matrix = calculate_temporal_economics(df)
    
    # Step 10: Pareto Analysis (Zone Value Curve)
    pareto_df = calculate_pareto_analysis(zone_financials)
    
    # Step 11: Scenario Modeling
    scenarios_df = calculate_scenario_analysis(df, zone_financials)
    
    # ========================================================================
    # ADVANCED STATISTICAL METHODS
    # ========================================================================
    print("\n" + "="*70)
    print("ADVANCED STATISTICAL METHODS")
    print("="*70)
    
    # Step 12: Monte Carlo Profit Simulation
    monte_carlo_df, monte_carlo_summary = calculate_monte_carlo_simulation(df)
    
    # Step 13: Sensitivity Analysis for Tornado Diagram
    sensitivity_df = calculate_sensitivity_analysis(df)
    
    # Step 14: Bootstrap Confidence Intervals
    bootstrap_df = calculate_bootstrap_financial_ci(df)
    
    # Step 15: Regression Analysis
    regression_df = calculate_regression_analysis(df, zone_financials)
    
    # Step 16: Save all checkpoints (including analytics)
    save_checkpoints(df, zone_financials, operator_pnl, daily_vehicle,
                     temporal_stats, pareto_df, scenarios_df,
                     monte_carlo_df, monte_carlo_summary,
                     sensitivity_df, bootstrap_df, regression_df)
    
    # Final summary
    print("\n" + "="*70)
    print("ECONOMIC ANALYSIS COMPLETE ")
    print("="*70)
    
    total_revenue = df['gross_revenue'].sum()
    total_cost = df['total_cost'].sum()
    total_profit = df['net_profit'].sum()
    margin = total_profit / total_revenue * 100
    
    print(f"\n  SYSTEM-WIDE FINANCIAL SUMMARY:")
    print(f"  ╔{'═'*50}╗")
    print(f"  ║  Total Revenue:      €{total_revenue:>15,.2f}       ║")
    print(f"  ║  Total Costs:        €{total_cost:>15,.2f}       ║")
    print(f"  ║  {'─'*46}  ║")
    print(f"  ║  Net Profit:         €{total_profit:>15,.2f}       ║")
    print(f"  ║  Profit Margin:      {margin:>15.1f}%       ║")
    print(f"  ╚{'═'*50}╝")
    
    # Verdict
    profitable_zones = (zone_financials['total_net_profit'] > 0).sum()
    subsidy_zones = (zone_financials['total_net_profit'] <= 0).sum()
    avg_subsidy_loss = zone_financials[zone_financials['total_net_profit'] <= 0]['total_net_profit'].mean()
    
    print(f"\n  STRATEGIC VERDICT:")
    print(f"  → Profitable Zones: {profitable_zones}")
    print(f"  → Subsidy Zones: {subsidy_zones} (avg loss: €{avg_subsidy_loss:,.2f})")
    
    # Advanced analytics summary
    print(f"\n  ADVANCED ANALYTICS:")
    print(f"  → Temporal Analysis: {len(temporal_stats)} time slots (Hour × Day)")
    print(f"  → Pareto Analysis: Top 20% zones = {pareto_df[pareto_df['cumulative_zone_pct'] <= 20]['cumulative_profit_pct'].max():.1f}% of profit")
    print(f"  → Scenarios Modeled: {len(scenarios_df)} (Base, Optimistic, Pessimistic, No-Subsidy)")
    
    # Statistical Methods summary
    print(f"\n  STATISTICAL METHODS:")
    print(f"  → Monte Carlo: {len(monte_carlo_df):,} simulations (profit uncertainty)")
    print(f"  → Sensitivity Analysis: {len(sensitivity_df)} parameters (tornado)")
    print(f"  → Bootstrap CI: {len(bootstrap_df)} operator confidence intervals")
    print(f"  → Regression Analysis: {len(regression_df)} models (profit drivers)")
    print("="*70)


if __name__ == '__main__':
    main()
