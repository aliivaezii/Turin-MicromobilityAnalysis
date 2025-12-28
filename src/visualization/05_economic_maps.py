#!/usr/bin/env python3
"""
================================================================================
EXERCISE 5: ECONOMIC & SENSITIVITY ANALYSIS - Geospatial Mapping & Visualization
================================================================================

Geospatial visualization and cartographic analysis module.

This module generates publication-quality maps with professional cartographic
elements including scale bars, north arrows, and statistical overlays.

Creates profit/loss visualizations, Monte Carlo distributions,
and scenario comparison charts.

Output Directory: outputs/figures/exercise5/

Author: Transport Research Team
Version: 1.0.0  
Last Updated: December 2025
================================================================================
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import contextily as cx
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - visualization scripts are in src/visualization/, need to go up 3 levels to project root
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "outputs" / "reports" / "exercise5"
ZONES_PATH = BASE_DIR / "outputs" / "reports" / "exercise3" / "checkpoint_zones_with_metrics.geojson"
FIGURE_DIR = BASE_DIR / "outputs" / "figures" / "exercise5"

# CRS
CRS_WGS84 = "EPSG:4326"
CRS_WEB_MERCATOR = "EPSG:3857"

# Styling
FIGSIZE_MAP = (14, 10)
FIGSIZE_CHART = (12, 8)
DPI = 300

# Colors
OPERATOR_COLORS = {
    'BIRD': '#E53935',
    'LIME': '#43A047', 
    'VOI': '#1E88E5'
}

PROFIT_GREEN = '#2E7D32'
LOSS_RED = '#C62828'
NEUTRAL_GRAY = '#757575'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_matplotlib():
    """Configure matplotlib for professional output."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': DPI
    })


def load_checkpoints():
    """Load all required checkpoints for visualization."""
    print("\n" + "="*70)
    print("LOADING CHECKPOINTS FOR EXERCISE 5 VISUALIZATION")
    print("="*70)
    
    checkpoints = {}
    
    # 1. Trip-level economics
    trips_path = DATA_DIR / "checkpoint_economics_trips.pkl"
    if trips_path.exists():
        checkpoints['trips'] = pd.read_pickle(trips_path)
        print(f"✓ Loaded trip economics: {len(checkpoints['trips']):,} trips")
    else:
        print(f"✗ Missing: {trips_path.name}")
        checkpoints['trips'] = None
    
    # 2. Zone-level economics
    zones_path = DATA_DIR / "checkpoint_economics_zones.csv"
    if zones_path.exists():
        checkpoints['zones'] = pd.read_csv(zones_path)
        print(f"✓ Loaded zone economics: {len(checkpoints['zones'])} zones")
    else:
        print(f"✗ Missing: {zones_path.name}")
        checkpoints['zones'] = None
    
    # 3. Operator P&L
    pnl_path = DATA_DIR / "checkpoint_operator_pnl.csv"
    if pnl_path.exists():
        checkpoints['operator_pnl'] = pd.read_csv(pnl_path)
        print(f"✓ Loaded operator P&L: {len(checkpoints['operator_pnl'])} operators")
    else:
        print(f"✗ Missing: {pnl_path.name}")
        checkpoints['operator_pnl'] = None
    
    # 4. Daily vehicle analysis
    daily_path = DATA_DIR / "checkpoint_daily_vehicle.pkl"
    if daily_path.exists():
        checkpoints['daily_vehicle'] = pd.read_pickle(daily_path)
        print(f"✓ Loaded daily vehicle data: {len(checkpoints['daily_vehicle']):,} vehicle-days")
    else:
        print(f"✗ Missing: {daily_path.name}")
        checkpoints['daily_vehicle'] = None
    
    # 5. Zone geometries (for maps)
    if ZONES_PATH.exists():
        checkpoints['zones_geo'] = gpd.read_file(ZONES_PATH)
        print(f"✓ Loaded zone geometries: {len(checkpoints['zones_geo'])} zones")
    else:
        print(f"✗ Missing zone geometries")
        checkpoints['zones_geo'] = None
    
    # 6. Temporal economics (NEW - for heatmap)
    temporal_path = DATA_DIR / "checkpoint_economics_temporal.csv"
    if temporal_path.exists():
        checkpoints['temporal'] = pd.read_csv(temporal_path)
        print(f"✓ Loaded temporal data: {len(checkpoints['temporal'])} time slots")
    else:
        print(f"✗ Missing: {temporal_path.name}")
        checkpoints['temporal'] = None
    
    # 7. Pareto analysis (NEW - for value curve)
    pareto_path = DATA_DIR / "checkpoint_economics_pareto.csv"
    if pareto_path.exists():
        checkpoints['pareto'] = pd.read_csv(pareto_path)
        print(f"✓ Loaded Pareto data: {len(checkpoints['pareto'])} zones ranked")
    else:
        print(f"✗ Missing: {pareto_path.name}")
        checkpoints['pareto'] = None
    
    # 8. Scenario models (NEW - for bridge chart)
    scenarios_path = DATA_DIR / "checkpoint_economics_scenarios.csv"
    if scenarios_path.exists():
        checkpoints['scenarios'] = pd.read_csv(scenarios_path)
        print(f"✓ Loaded scenarios: {len(checkpoints['scenarios'])} scenarios")
    else:
        print(f"✗ Missing: {scenarios_path.name}")
        checkpoints['scenarios'] = None
    
    # Summary
    available = sum(1 for v in checkpoints.values() if v is not None)
    print(f"\n→ Loaded {available}/{len(checkpoints)} checkpoint files")
    
    return checkpoints


def add_basemap(ax, crs=CRS_WEB_MERCATOR):
    """Add CartoDB Positron basemap."""
    try:
        cx.add_basemap(ax, crs=crs, source=cx.providers.CartoDB.Positron, alpha=0.8)
    except Exception as e:
        print(f"  Warning: Could not add basemap: {e}")


def format_currency(x, pos=None):
    """Format number as currency."""
    if abs(x) >= 1e6:
        return f'€{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'€{x/1e3:.0f}K'
    else:
        return f'€{x:.0f}'


# ============================================================================
# FIGURE 1: MAP PROFITABILITY HOTSPOTS
# ============================================================================

def plot_profitability_map(zones_df, zones_geo, output_path):
    """
    Choropleth of Net Profit per Zone.
    
    Green = Profitable neighborhoods
    Red = Money-losing areas (Subsidy Zones)
    """
    print("\n" + "-"*50)
    print("Figure 1: Profitability Hotspots Map")
    print("-"*50)
    
    if zones_df is None or zones_geo is None:
        print("  ✗ Skipping: required data not available")
        return False
    
    # Merge economics with geometry
    zones_plot = zones_geo.merge(
        zones_df[['ZONASTAT', 'total_net_profit', 'zone_name', 'zone_classification']],
        on='ZONASTAT',
        how='left'
    )
    
    # Filter valid data
    valid_zones = zones_plot[zones_plot['total_net_profit'].notna()].copy()
    
    if len(valid_zones) == 0:
        print("  ✗ Skipping: no valid profit data")
        return False
    
    print(f"  → {len(valid_zones)} zones with profit data")
    
    # Project to Web Mercator
    zones_plot = valid_zones.to_crs(CRS_WEB_MERCATOR)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_MAP)
    
    # Diverging colormap centered at 0
    vmax = max(abs(valid_zones['total_net_profit'].min()), 
               abs(valid_zones['total_net_profit'].max()))
    
    zones_plot.plot(
        column='total_net_profit',
        ax=ax,
        cmap='RdYlGn',
        legend=True,
        legend_kwds={
            'label': 'Net Profit (€)',
            'orientation': 'horizontal',
            'shrink': 0.6,
            'pad': 0.02,
            'format': format_currency
        },
        edgecolor='white',
        linewidth=0.3,
        alpha=0.85,
        vmin=-vmax,
        vmax=vmax
    )
    
    # Add basemap
    add_basemap(ax)
    
    # Format
    ax.set_title('E-Scooter Profitability by Zone\nTurin Shared Mobility Economics', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    
    # Stats annotation
    profit_centers = (valid_zones['total_net_profit'] > 0).sum()
    subsidy_zones = (valid_zones['total_net_profit'] <= 0).sum()
    total_profit = valid_zones['total_net_profit'].sum()
    
    ax.text(
        0.02, 0.02,
        f'Profit Centers: {profit_centers} | Subsidy Zones: {subsidy_zones}\n'
        f'System Net Profit: €{total_profit:,.0f}',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 2: MAP REVENUE YIELD
# ============================================================================

def plot_revenue_yield_map(zones_df, zones_geo, output_path):
    """
    Choropleth of Revenue per Square Km.
    Shows where the cash flow density is highest.
    """
    print("\n" + "-"*50)
    print("Figure 2: Revenue Yield Map")
    print("-"*50)
    
    if zones_df is None or zones_geo is None:
        print("  ✗ Skipping: required data not available")
        return False
    
    # Merge economics with geometry
    zones_plot = zones_geo.merge(
        zones_df[['ZONASTAT', 'revenue_per_sqkm', 'zone_name', 'total_revenue']],
        on='ZONASTAT',
        how='left'
    )
    
    # Filter valid data
    valid_zones = zones_plot[zones_plot['revenue_per_sqkm'].notna()].copy()
    
    if len(valid_zones) == 0:
        print("  ✗ Skipping: no valid revenue data")
        return False
    
    print(f"  → {len(valid_zones)} zones with revenue data")
    
    # Project to Web Mercator
    zones_plot = valid_zones.to_crs(CRS_WEB_MERCATOR)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_MAP)
    
    # Use log scale for better visualization
    zones_plot['log_revenue'] = np.log10(zones_plot['revenue_per_sqkm'] + 1)
    
    zones_plot.plot(
        column='log_revenue',
        ax=ax,
        cmap='YlOrRd',
        legend=True,
        legend_kwds={
            'label': 'Revenue Yield (log₁₀ €/km²)',
            'orientation': 'horizontal',
            'shrink': 0.6,
            'pad': 0.02
        },
        edgecolor='white',
        linewidth=0.3,
        alpha=0.85
    )
    
    # Add basemap
    add_basemap(ax)
    
    # Format
    ax.set_title('Revenue Yield by Zone\nCash Flow Density Analysis', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    
    # Stats annotation
    top_zone = valid_zones.nlargest(1, 'revenue_per_sqkm').iloc[0]
    total_revenue = valid_zones['total_revenue'].sum()
    
    ax.text(
        0.02, 0.02,
        f'Total System Revenue: €{total_revenue:,.0f}\n'
        f'Highest Yield: {top_zone.get("zone_name", top_zone["ZONASTAT"])} (€{top_zone["revenue_per_sqkm"]:,.0f}/km²)',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 3: OPERATOR P&L WATERFALL
# ============================================================================

def plot_pnl_waterfall(operator_pnl, output_path):
    """
    Waterfall chart showing P&L breakdown:
    - Total Revenue (Positive)
    - Variable Costs (Negative)
    - Fixed Costs (Negative)
    - Net Profit (Result)
    """
    print("\n" + "-"*50)
    print("Figure 3: Operator P&L Waterfall")
    print("-"*50)
    
    if operator_pnl is None:
        print("  ✗ Skipping: operator P&L data not available")
        return False
    
    # Calculate system totals
    total_revenue = operator_pnl['total_revenue'].sum()
    total_var_cost = operator_pnl['total_variable_cost'].sum()
    total_fixed_cost = operator_pnl['total_fixed_cost'].sum()
    total_profit = operator_pnl['total_net_profit'].sum()
    
    print(f"  → Revenue: €{total_revenue:,.0f}")
    print(f"  → Variable Costs: €{total_var_cost:,.0f}")
    print(f"  → Fixed Costs: €{total_fixed_cost:,.0f}")
    print(f"  → Net Profit: €{total_profit:,.0f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_CHART)
    
    # Waterfall data
    categories = ['Revenue', 'Variable\nCosts', 'Fixed\nCosts', 'Net\nProfit']
    values = [total_revenue, -total_var_cost, -total_fixed_cost, total_profit]
    
    # Calculate running totals for waterfall positioning
    running_total = [0, total_revenue, total_revenue - total_var_cost, 
                     total_revenue - total_var_cost - total_fixed_cost]
    
    # Colors
    colors = [PROFIT_GREEN, LOSS_RED, LOSS_RED, 
              PROFIT_GREEN if total_profit > 0 else LOSS_RED]
    
    # Plot bars
    bar_width = 0.6
    for i, (cat, val, start, color) in enumerate(zip(categories, values, running_total, colors)):
        if i == 0:  # Revenue starts from 0
            ax.bar(i, val, bar_width, bottom=0, color=color, edgecolor='white', linewidth=2)
        elif i == len(categories) - 1:  # Net profit starts from 0
            ax.bar(i, val, bar_width, bottom=0, color=color, edgecolor='white', linewidth=2)
        else:  # Costs subtract from running total
            ax.bar(i, val, bar_width, bottom=start + val if val < 0 else start, 
                   color=color, edgecolor='white', linewidth=2)
        
        # Add value labels
        label_y = start + val/2 if i not in [0, 3] else val/2
        label_text = f'€{abs(val)/1e6:.2f}M'
        if val < 0:
            label_text = f'-€{abs(val)/1e6:.2f}M'
        ax.text(i, label_y, label_text, ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    # Connect bars with lines
    for i in range(len(categories) - 2):
        y_line = running_total[i+1]
        ax.hlines(y_line, i + bar_width/2, i + 1 - bar_width/2, 
                  colors='gray', linestyles='dashed', linewidth=1)
    
    # Formatting
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel('Amount (€)', fontsize=12)
    ax.set_title('System-Wide Profit & Loss Breakdown\nTurin E-Scooter Operations', 
                 fontsize=14, fontweight='bold')
    
    # Y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x/1e6:.1f}M'))
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Profit margin annotation
    margin = total_profit / total_revenue * 100
    ax.text(0.98, 0.95, f'Profit Margin: {margin:.1f}%', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 4: BREAK-EVEN SCATTER
# ============================================================================

def plot_break_even_scatter(daily_vehicle, output_path):
    """
    Scatter plot: Trips per Day vs Daily Net Profit.
    Shows how many trips/day a scooter needs to be profitable.
    """
    print("\n" + "-"*50)
    print("Figure 4: Break-Even Scatter")
    print("-"*50)
    
    if daily_vehicle is None:
        print("  ✗ Skipping: daily vehicle data not available")
        return False
    
    print(f"  → {len(daily_vehicle):,} vehicle-days to analyze")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_CHART)
    
    # Sample if too many points
    if len(daily_vehicle) > 10000:
        plot_data = daily_vehicle.sample(10000, random_state=42)
    else:
        plot_data = daily_vehicle
    
    # Plot by operator
    for operator, color in OPERATOR_COLORS.items():
        op_data = plot_data[plot_data['operator'] == operator]
        if len(op_data) > 0:
            ax.scatter(
                op_data['trips_per_day'],
                op_data['daily_total_profit'],
                c=color,
                s=20,
                alpha=0.3,
                label=operator,
                edgecolors='none'
            )
    
    # Add break-even line at Y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2, label='Break-Even')
    
    # Calculate and plot average by trips
    avg_by_trips = daily_vehicle.groupby('trips_per_day')['daily_total_profit'].mean()
    trips_x = avg_by_trips.index[avg_by_trips.index <= 15]  # Limit x-axis
    profit_y = avg_by_trips[avg_by_trips.index <= 15]
    
    ax.plot(trips_x, profit_y, 'k--', linewidth=2.5, label='Average', zorder=10)
    
    # Find break-even point
    break_even_trips = None
    for trips in sorted(avg_by_trips.index):
        if avg_by_trips[trips] > 0:
            break_even_trips = trips
            break
    
    if break_even_trips:
        ax.axvline(x=break_even_trips, color='green', linestyle=':', linewidth=2, alpha=0.8)
        ax.text(break_even_trips + 0.2, ax.get_ylim()[1] * 0.8,
                f'Break-even:\n{break_even_trips} trips/day',
                fontsize=11, fontweight='bold', color='green')
    
    # Formatting
    ax.set_xlabel('Trips per Day per Vehicle', fontsize=12)
    ax.set_ylabel('Daily Net Profit (€)', fontsize=12)
    ax.set_title('Vehicle Break-Even Analysis\nDaily Trips vs Profitability', 
                 fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 15)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Fill profitable/loss regions
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([xlim[0], xlim[1]], [0, 0], [ylim[1], ylim[1]], 
                    color='green', alpha=0.05)
    ax.fill_between([xlim[0], xlim[1]], [ylim[0], ylim[0]], [0, 0], 
                    color='red', alpha=0.05)
    
    ax.text(xlim[1]*0.95, ylim[1]*0.1, 'PROFITABLE', 
            ha='right', fontsize=10, color='green', alpha=0.5, fontweight='bold')
    ax.text(xlim[1]*0.95, ylim[0]*0.1, 'LOSS-MAKING', 
            ha='right', fontsize=10, color='red', alpha=0.5, fontweight='bold')
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 5: UNIT ECONOMICS DISTRIBUTION
# ============================================================================

def plot_unit_economics(trips_df, output_path):
    """
    Histogram of Profit per Trip.
    Highlights what % of trips are profitable vs loss-making.
    """
    print("\n" + "-"*50)
    print("Figure 5: Unit Economics Distribution")
    print("-"*50)
    
    if trips_df is None:
        print("  ✗ Skipping: trip economics data not available")
        return False
    
    print(f"  → {len(trips_df):,} trips to analyze")
    
    # Calculate stats
    profitable_trips = (trips_df['net_profit'] > 0).sum()
    profitable_pct = profitable_trips / len(trips_df) * 100
    avg_profit = trips_df['net_profit'].mean()
    median_profit = trips_df['net_profit'].median()
    
    print(f"  → Profitable: {profitable_pct:.1f}%")
    print(f"  → Avg profit/trip: €{avg_profit:.3f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_CHART)
    
    # Filter outliers for better visualization
    profit_data = trips_df['net_profit'].clip(-3, 5)
    
    # Create bins
    bins = np.linspace(-3, 5, 80)
    
    # Plot histogram with colors for profit/loss
    n, bins_edges, patches = ax.hist(
        profit_data,
        bins=bins,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8
    )
    
    # Color bars based on profit/loss
    for patch, left_edge in zip(patches, bins_edges[:-1]):
        if left_edge < 0:
            patch.set_facecolor(LOSS_RED)
        else:
            patch.set_facecolor(PROFIT_GREEN)
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Break-Even')
    
    # Add lines for mean and median
    ax.axvline(x=avg_profit, color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: €{avg_profit:.2f}')
    ax.axvline(x=median_profit, color='orange', linestyle=':', linewidth=2,
               label=f'Median: €{median_profit:.2f}')
    
    # Formatting
    ax.set_xlabel('Net Profit per Trip (€)', fontsize=12)
    ax.set_ylabel('Number of Trips', fontsize=12)
    ax.set_title('Unit Economics: Profit Distribution per Trip\nTurin E-Scooter Operations', 
                 fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Add annotation box
    textstr = (f'Profitable Trips: {profitable_pct:.1f}%\n'
               f'Loss-Making: {100-profitable_pct:.1f}%\n'
               f'System Avg: €{avg_profit:.3f}/trip')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 6: PARETO VALUE CURVE (Consulting-Grade)
# ============================================================================

def plot_pareto_curve(pareto_df, output_path):
    """
    Dual-Axis Pareto Chart showing Zone Value Concentration.
    
    Primary Y-axis: Net Profit per Zone (Bars - Green/Red)
    Secondary Y-axis: Cumulative Profit % (Line)
    
    Professional consulting-grade visualization.
    """
    print("\n" + "-"*50)
    print("Figure 6: Pareto Value Curve (Zone Profit Concentration)")
    print("-"*50)
    
    if pareto_df is None or len(pareto_df) == 0:
        print("  ✗ Skipping: Pareto data not available")
        return False
    
    # Ensure sorted by rank
    pareto_df = pareto_df.sort_values('rank').copy()
    n_zones = len(pareto_df)
    
    print(f"  → Analyzing {n_zones} zones")
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(16, 9))
    ax2 = ax1.twinx()
    
    # X positions
    x = np.arange(n_zones)
    
    # Determine bar colors (green=profit, red=loss)
    colors = [PROFIT_GREEN if p > 0 else LOSS_RED for p in pareto_df['total_net_profit']]
    
    # Primary axis: Net Profit Bars
    bars = ax1.bar(x, pareto_df['total_net_profit'], 
                   color=colors, alpha=0.75, width=0.7,
                   edgecolor='white', linewidth=0.5)
    
    # Secondary axis: Cumulative Line
    ax2.plot(x, pareto_df['cumulative_profit_pct'], 
             color='#1565C0', linewidth=3, marker='o', markersize=4,
             label='Cumulative Profit %', zorder=5)
    
    # Add 80% threshold line
    ax2.axhline(y=80, color='#FF6F00', linestyle='--', linewidth=2, 
                label='80% Profit Threshold', alpha=0.8)
    
    # Find where 80% is reached
    zones_80 = pareto_df[pareto_df['cumulative_profit_pct'] >= 80].head(1)
    if len(zones_80) > 0:
        idx_80 = zones_80.index[0]
        zone_count_80 = pareto_df.loc[idx_80, 'rank']
        ax1.axvline(x=zone_count_80-1, color='#FF6F00', linestyle=':', 
                    linewidth=2, alpha=0.8)
        ax1.annotate(f'{int(zone_count_80)} zones\n({zone_count_80/n_zones*100:.0f}%)',
                     xy=(zone_count_80-1, pareto_df['total_net_profit'].max() * 0.9),
                     fontsize=11, ha='center', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', alpha=0.9))
    
    # Formatting - Primary Axis
    ax1.set_xlabel('Zone Rank (by Net Profit, Descending)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Net Profit per Zone (€)', fontsize=12, fontweight='bold', color='#2E7D32')
    ax1.tick_params(axis='y', labelcolor='#2E7D32')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    
    # Formatting - Secondary Axis
    ax2.set_ylabel('Cumulative Profit (%)', fontsize=12, fontweight='bold', color='#1565C0')
    ax2.tick_params(axis='y', labelcolor='#1565C0')
    ax2.set_ylim(0, 105)
    
    # X-axis: Show every 5th zone
    step = max(1, n_zones // 15)
    ax1.set_xticks(x[::step])
    ax1.set_xticklabels([f'{i+1}' for i in x[::step]], rotation=45, ha='right')
    
    # Title
    fig.suptitle('Zone Value Concentration: Pareto Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    ax1.set_title('Turin E-Scooter Operations - 80/20 Rule Applied to Zone Profitability',
                  fontsize=12, color='gray', pad=10)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PROFIT_GREEN, alpha=0.75, label='Profit Zone'),
        Patch(facecolor=LOSS_RED, alpha=0.75, label='Loss Zone'),
        Line2D([0], [0], color='#1565C0', linewidth=3, label='Cumulative Profit %'),
        Line2D([0], [0], color='#FF6F00', linestyle='--', linewidth=2, label='80% Threshold')
    ]
    ax1.legend(handles=legend_elements, loc='center right', framealpha=0.95, fontsize=10)
    
    # Grid
    ax1.grid(True, alpha=0.3, axis='y', zorder=0)
    ax1.set_axisbelow(True)
    
    # Annotation box with key insights
    total_profit = pareto_df['total_net_profit'].sum()
    profit_zones = (pareto_df['total_net_profit'] > 0).sum()
    loss_zones = (pareto_df['total_net_profit'] <= 0).sum()
    
    textstr = (f"Total System Profit: €{total_profit:,.0f}\n"
               f"Profit Centers: {profit_zones} zones\n"
               f"Subsidy Zones: {loss_zones} zones")
    props = dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.95, edgecolor='#1565C0')
    ax1.text(0.98, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 7: TEMPORAL PROFITABILITY HEATMAP (Consulting-Grade)
# ============================================================================

def plot_temporal_heatmap(temporal_df, output_path):
    """
    Seaborn-style Heatmap showing Hourly Profitability by Day of Week.
    
    X-axis: Hour (0-23)
    Y-axis: Day (Monday-Sunday)
    Color: Net Profit (RdYlGn diverging, centered at 0)
    """
    print("\n" + "-"*50)
    print("Figure 7: Temporal Profitability Matrix")
    print("-"*50)
    
    if temporal_df is None or len(temporal_df) == 0:
        print("  ✗ Skipping: Temporal data not available")
        return False
    
    import seaborn as sns
    
    print(f"  → Analyzing {len(temporal_df)} time slots")
    
    # Prepare pivot table
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Pivot: rows=day_of_week, columns=hour, values=net_profit
    pivot_data = temporal_df.pivot(
        index='day_of_week',
        columns='hour',
        values='total_net_profit'
    ).fillna(0)
    
    # Ensure all hours are present
    all_hours = list(range(24))
    for h in all_hours:
        if h not in pivot_data.columns:
            pivot_data[h] = 0
    pivot_data = pivot_data[sorted(pivot_data.columns)]
    
    # Convert values to thousands for readability
    pivot_data_k = pivot_data / 1000
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Determine color scale center and limits
    vmax = pivot_data_k.max().max()
    vmin = pivot_data_k.min().min()
    
    # If all positive, center differently
    if vmin >= 0:
        # All profitable - use Greens
        cmap = 'YlGn'
        center = None
    else:
        # Mixed - use diverging centered at 0
        cmap = 'RdYlGn'
        center = 0
        # Symmetric limits
        abs_max = max(abs(vmax), abs(vmin))
        vmax = abs_max
        vmin = -abs_max
    
    # Create heatmap
    heatmap = sns.heatmap(
        pivot_data_k,
        ax=ax,
        cmap=cmap,
        center=center,
        vmin=vmin if center else None,
        vmax=vmax if center else None,
        annot=True,
        fmt='.0f',
        annot_kws={'fontsize': 8, 'fontweight': 'bold'},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Net Profit (€ thousands)', 'shrink': 0.8}
    )
    
    # Y-axis: Day names
    ax.set_yticklabels(day_names, rotation=0, fontsize=11, fontweight='bold')
    
    # X-axis: Hour labels
    hour_labels = [f'{h:02d}:00' for h in range(24)]
    ax.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=9)
    
    # Labels
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Day of Week', fontsize=12, fontweight='bold', labelpad=10)
    
    # Title
    fig.suptitle('Temporal Profitability Analysis: When Does Money Flow?', 
                 fontsize=16, fontweight='bold', y=0.98)
    ax.set_title('Net Profit by Hour and Day of Week - Turin E-Scooter Operations (€ thousands)',
                 fontsize=11, color='gray', pad=15)
    
    # Add peak/low annotations
    # Find peak hour-day
    peak_idx = pivot_data.stack().idxmax()
    peak_val = pivot_data.loc[peak_idx[0], peak_idx[1]]
    low_idx = pivot_data.stack().idxmin()
    low_val = pivot_data.loc[low_idx[0], low_idx[1]]
    
    textstr = (f" Peak: {day_names[peak_idx[0]]} {peak_idx[1]:02d}:00 → €{peak_val:,.0f}\n"
               f" Low:  {day_names[low_idx[0]]} {low_idx[1]:02d}:00 → €{low_val:,.0f}")
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, edgecolor='gray')
    # Position at top-left in figure space (above the heatmap in the white space)
    fig.text(0.02, 0.92, textstr, fontsize=10,
             verticalalignment='top', horizontalalignment='left', bbox=props, fontfamily='monospace')
    
    # Save
    plt.tight_layout(rect=[0, 0, 1.0, 0.90])
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 8: SCENARIO COMPARISON BRIDGE CHART (Consulting-Grade)
# ============================================================================

def plot_scenario_bridge(scenarios_df, output_path):
    """
    Executive Scenario Comparison Chart.
    
    Grouped bar chart comparing Net Profit across scenarios:
    - Base Case
    - Optimistic (+10% Revenue, -10% OpEx)
    - Pessimistic (-10% Revenue, +10% OpEx)
    - No Subsidy (Exit bottom 20% zones)
    
    With delta annotations and margin percentages.
    """
    print("\n" + "-"*50)
    print("Figure 8: Scenario Analysis Bridge Chart")
    print("-"*50)
    
    if scenarios_df is None or len(scenarios_df) == 0:
        print("  ✗ Skipping: Scenario data not available")
        return False
    
    print(f"  → Comparing {len(scenarios_df)} scenarios")
    
    # Sort by scenario code for consistent ordering
    scenario_order = ['BASE', 'OPTIMISTIC', 'PESSIMISTIC', 'NO_SUBSIDY']
    scenarios_df = scenarios_df.copy()
    scenarios_df['sort_order'] = scenarios_df['scenario_code'].map(
        {code: i for i, code in enumerate(scenario_order)}
    )
    scenarios_df = scenarios_df.sort_values('sort_order')
    
    # Colors for each scenario
    scenario_colors = {
        'BASE': '#546E7A',       # Gray
        'OPTIMISTIC': '#43A047', # Green
        'PESSIMISTIC': '#E53935', # Red
        'NO_SUBSIDY': '#FF9800'  # Orange
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Bar positions and data
    x = np.arange(len(scenarios_df))
    bar_width = 0.6
    
    profits = scenarios_df['net_profit'].values
    margins = scenarios_df['profit_margin_pct'].values
    deltas = scenarios_df['delta_profit'].values
    codes = scenarios_df['scenario_code'].values
    labels = scenarios_df['scenario'].values
    
    colors = [scenario_colors.get(code, '#757575') for code in codes]
    
    # Plot bars
    bars = ax.bar(x, profits / 1e6, width=bar_width, color=colors, 
                  edgecolor='white', linewidth=2, alpha=0.85)
    
    # Add value labels on bars
    for i, (bar, profit, margin, delta) in enumerate(zip(bars, profits, margins, deltas)):
        height = bar.get_height()
        
        # Main value (€M)
        ax.annotate(f'€{profit/1e6:.2f}M',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 10), textcoords="offset points",
                    ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        # Margin percentage
        ax.annotate(f'({margin:.1f}% margin)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 28), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, color='gray')
        
        # Delta (for non-base scenarios)
        if delta != 0:
            delta_color = PROFIT_GREEN if delta > 0 else LOSS_RED
            delta_sign = '+' if delta > 0 else ''
            ax.annotate(f'{delta_sign}€{delta/1e6:.2f}M',
                        xy=(bar.get_x() + bar.get_width() / 2, 0),
                        xytext=(0, -25), textcoords="offset points",
                        ha='center', va='top', fontsize=11, fontweight='bold',
                        color=delta_color,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                  edgecolor=delta_color, alpha=0.9))
    
    # Add horizontal line at base case
    base_profit = scenarios_df[scenarios_df['scenario_code'] == 'BASE']['net_profit'].values[0]
    ax.axhline(y=base_profit / 1e6, color='#546E7A', linestyle='--', 
               linewidth=2, alpha=0.6, label='Base Case Reference')
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Net Profit (€ Millions)', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:.1f}M'))
    
    # Y limits with padding
    y_max = max(profits) / 1e6 * 1.25
    y_min = min(0, min(profits) / 1e6 * 1.1)
    ax.set_ylim(y_min, y_max)
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.set_axisbelow(True)
    
    # Title
    fig.suptitle('Scenario Analysis: Strategic What-If Modeling', 
                 fontsize=16, fontweight='bold', y=0.98)
    ax.set_title('Net Profit Comparison Across Business Scenarios - Turin E-Scooter Operations',
                 fontsize=11, color='gray', pad=15)
    
    # Add insight box
    best_scenario = scenarios_df.loc[scenarios_df['net_profit'].idxmax(), 'scenario']
    worst_scenario = scenarios_df.loc[scenarios_df['net_profit'].idxmin(), 'scenario']
    spread = (max(profits) - min(profits)) / 1e6
    
    textstr = (f"Strategic Insights:\n"
               f"• Best Case: {best_scenario}\n"
               f"• Worst Case: {worst_scenario}\n"
               f"• Profit Range: €{spread:.2f}M")
    props = dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', alpha=0.95, edgecolor='#1565C0')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#546E7A', alpha=0.85, label='Base Case'),
        Patch(facecolor='#43A047', alpha=0.85, label='Optimistic'),
        Patch(facecolor='#E53935', alpha=0.85, label='Pessimistic'),
        Patch(facecolor='#FF9800', alpha=0.85, label='No Subsidy')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, fontsize=10)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all Exercise 5 visualization figures."""
    print("\n" + "="*70)
    print("EXERCISE 5: BUSINESS MODEL & ECONOMIC ANALYSIS")
    print("="*70)
    print("Executive Dashboard - Financial Visualization Suite (UPGRADED)")
    print("="*70)
    
    # Setup
    setup_matplotlib()
    
    # Ensure output directory exists
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {FIGURE_DIR}")
    
    # Load checkpoints
    checkpoints = load_checkpoints()
    
    # Check if we have minimum required data
    if all(v is None for v in checkpoints.values()):
        print("\n" + "="*70)
        print("ERROR: No checkpoint data available!")
        print("Please run 06_economics.py first to generate checkpoints.")
        print("="*70)
        return
    
    # Generate figures
    results = {}
    
    # A. STRATEGIC MAPS
    print("\n" + "="*70)
    print("A. STRATEGIC MAPS ('MONEY MAPS')")
    print("="*70)
    
    # Figure 1: Profitability Hotspots
    results['profitability_map'] = plot_profitability_map(
        checkpoints['zones'],
        checkpoints['zones_geo'],
        FIGURE_DIR / 'map_profitability_hotspots.png'
    )
    
    # Figure 2: Revenue Yield
    results['revenue_map'] = plot_revenue_yield_map(
        checkpoints['zones'],
        checkpoints['zones_geo'],
        FIGURE_DIR / 'map_revenue_yield.png'
    )
    
    # B. BUSINESS INTELLIGENCE CHARTS
    print("\n" + "="*70)
    print("B. BUSINESS INTELLIGENCE CHARTS")
    print("="*70)
    
    # Figure 3: P&L Waterfall
    results['pnl_waterfall'] = plot_pnl_waterfall(
        checkpoints['operator_pnl'],
        FIGURE_DIR / 'operator_pnl_waterfall.png'
    )
    
    # Figure 4: Break-Even Scatter
    results['break_even'] = plot_break_even_scatter(
        checkpoints['daily_vehicle'],
        FIGURE_DIR / 'break_even_scatter.png'
    )
    
    # Figure 5: Unit Economics
    results['unit_economics'] = plot_unit_economics(
        checkpoints['trips'],
        FIGURE_DIR / 'unit_economics_distribution.png'
    )
    
    # C. ADVANCED STRATEGIC VISUALIZATIONS (CONSULTING-GRADE)
    print("\n" + "="*70)
    print("C. CONSULTING-GRADE STRATEGIC VISUALIZATIONS")
    print("="*70)
    
    # Figure 6: Pareto Value Curve
    results['pareto_curve'] = plot_pareto_curve(
        checkpoints['pareto'],
        FIGURE_DIR / 'pareto_value_curve.png'
    )
    
    # Figure 7: Temporal Heatmap
    results['temporal_heatmap'] = plot_temporal_heatmap(
        checkpoints['temporal'],
        FIGURE_DIR / 'temporal_profitability_heatmap.png'
    )
    
    # Figure 8: Scenario Bridge
    results['scenario_bridge'] = plot_scenario_bridge(
        checkpoints['scenarios'],
        FIGURE_DIR / 'scenario_comparison_bridge.png'
    )
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION SUMMARY (UPGRADED)")
    print("="*70)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\nA. Strategic Maps:")
    print(f"  {'✓' if results.get('profitability_map') else '✗'} map_profitability_hotspots.png")
    print(f"  {'✓' if results.get('revenue_map') else '✗'} map_revenue_yield.png")
    
    print("\nB. Business Intelligence Charts:")
    print(f"  {'✓' if results.get('pnl_waterfall') else '✗'} operator_pnl_waterfall.png")
    print(f"  {'✓' if results.get('break_even') else '✗'} break_even_scatter.png")
    print(f"  {'✓' if results.get('unit_economics') else '✗'} unit_economics_distribution.png")
    
    print("\nC. Consulting-Grade Strategic Visualizations:")
    print(f"  {'✓' if results.get('pareto_curve') else '✗'} pareto_value_curve.png")
    print(f"  {'✓' if results.get('temporal_heatmap') else '✗'} temporal_profitability_heatmap.png")
    print(f"  {'✓' if results.get('scenario_bridge') else '✗'} scenario_comparison_bridge.png")
    
    print(f"\n→ Generated {successful}/{total} figures successfully")
    print(f"→ Output location: {FIGURE_DIR}")
    
    # Executive Summary
    if checkpoints['zones'] is not None and checkpoints['operator_pnl'] is not None:
        zones = checkpoints['zones']
        pnl = checkpoints['operator_pnl']
        
        profit_centers = (zones['total_net_profit'] > 0).sum()
        subsidy_zones = (zones['total_net_profit'] <= 0).sum()
        total_profit = zones['total_net_profit'].sum()
        
        # Calculate peripheral zone losses
        peripheral_loss = zones[zones['total_net_profit'] < 0]['total_net_profit'].sum()
        central_profit = zones[zones['total_net_profit'] > 0]['total_net_profit'].sum()
        
        # Calculate avg loss safely
        if subsidy_zones > 0:
            avg_loss_per_zone = abs(peripheral_loss / subsidy_zones)
            loss_message = f"€{avg_loss_per_zone:,.0f} per zone on average"
        else:
            loss_message = "N/A (no loss-making zones)"
        
        print("\n" + "="*70)
        print("EXECUTIVE SUMMARY FOR PRESENTATION:")
        print("="*70)
        print(f"""
  "Current market operations are highly profitable across the system.
   
   Key Findings:
   • Profit Centers: {profit_centers} zones (generating €{central_profit:,.0f})
   • Subsidy Zones: {subsidy_zones} zones (losing €{abs(peripheral_loss):,.0f})
   • Peripheral Zone Avg Loss: {loss_message}
   
   The {profit_centers} profitable zones generate consistent revenue
   streams across Turin's urban core and commercial areas.
   
   System-wide Net Profit: €{total_profit:,.0f}"
        """)
    print("="*70)


if __name__ == '__main__':
    main()
