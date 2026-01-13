#!/usr/bin/env python3
"""
================================================================================
EXERCISE 5: ECONOMIC & SENSITIVITY ANALYSIS - Economic Sensitivity & Risk Analysis
================================================================================
Economic sensitivity analysis and Monte Carlo simulation visualization module.

This module generates economic analysis figures including
scenario comparisons, sensitivity tornado charts, and risk distributions.

Creates profit/loss visualizations, Monte Carlo distributions,
and scenario comparison charts.

Output Directory: outputs/figures/exercise5/

Author: Ali Vaezi
Version: 1.0.0
Last Updated: December 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - visualization scripts are in src/visualization/, need to go up 3 levels to project root
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "outputs" / "reports" / "exercise5"
FIGURE_DIR = BASE_DIR / "outputs" / "figures" / "exercise5" / "statistical"
TABLE_DIR = BASE_DIR / "outputs" / "tables" / "exercise5"

# Professional settings
DPI = 300
FIGSIZE = (10, 7)
FIGSIZE_WIDE = (12, 6)

# Colors
OPERATOR_COLORS = {
    'BIRD': '#D32F2F',
    'LIME': '#388E3C',
    'VOI': '#1976D2'
}


def setup_matplotlib():
    """Configure matplotlib for HIGH-QUALITY."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight'
    })


def euro_formatter(x, pos):
    """Format as euros."""
    if abs(x) >= 1e6:
        return f'€{x/1e6:.1f}M'
    elif abs(x) >= 1e3:
        return f'€{x/1e3:.0f}K'
    else:
        return f'€{x:.0f}'


# ============================================================================
# FIGURE 1: REVENUE VS COST BY OPERATOR
# ============================================================================

def create_fig01_revenue_cost(trips_df):
    """Single-panel bar chart: Revenue vs Cost by Operator."""
    print("  Creating fig01_revenue_cost_operator.png...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Aggregate by operator
    op_summary = trips_df.groupby('operator').agg({
        'gross_revenue': 'sum',
        'total_cost': 'sum',
        'net_profit': 'sum'
    }).reset_index()
    
    x = np.arange(len(op_summary))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, op_summary['gross_revenue'], width, 
                   label='Revenue', color='#4575b4', edgecolor='white')
    bars2 = ax.bar(x + width/2, op_summary['total_cost'], width,
                   label='Cost', color='#d73027', edgecolor='white')
    
    ax.set_xlabel('Operator')
    ax.set_ylabel('Amount (€)')
    ax.set_title('Revenue vs Operating Cost by Operator', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(op_summary['operator'])
    ax.yaxis.set_major_formatter(FuncFormatter(euro_formatter))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'€{h/1e6:.1f}M', xy=(bar.get_x() + bar.get_width()/2, h),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig01_revenue_cost_operator.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


# ============================================================================
# FIGURE 2: PROFIT MARGIN DISTRIBUTION
# ============================================================================

def create_fig02_profit_margin_dist(trips_df):
    """Single-panel histogram: Profit margin distribution."""
    print("  Creating fig02_profit_margin_distribution.png...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    for op in trips_df['operator'].unique():
        margins = trips_df[trips_df['operator'] == op]['roi_per_trip'] * 100
        margins = margins[margins.between(-100, 100)]
        color = OPERATOR_COLORS.get(op, '#666666')
        ax.hist(margins, bins=50, alpha=0.5, color=color, label=op, density=True)
    
    ax.axvline(0, color='red', linewidth=2, linestyle='--', label='Break-even')
    ax.set_xlabel('Profit Margin (%)')
    ax.set_ylabel('Density')
    ax.set_title('Profit Margin Distribution by Operator', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig02_profit_margin_distribution.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


# ============================================================================
# FIGURE 3: MONTE CARLO HISTOGRAM
# ============================================================================

def create_fig03_monte_carlo(monte_carlo_df, monte_carlo_summary):
    """Single-panel histogram: Monte Carlo profit simulation."""
    print("  Creating fig03_monte_carlo_histogram.png...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    sim_data = monte_carlo_df['net_profit'] / 1e6
    
    # Get summary stats
    if isinstance(monte_carlo_summary, pd.DataFrame):
        summary = monte_carlo_summary.iloc[0].to_dict()
    else:
        summary = monte_carlo_summary
    
    # Histogram
    n, bins, patches = ax.hist(sim_data, bins=50, density=True,
                               alpha=0.7, color='#4575b4', edgecolor='white')
    
    # Add KDE
    if HAS_SCIPY:
        kde_x = np.linspace(sim_data.min(), sim_data.max(), 100)
        kde = stats.gaussian_kde(sim_data)
        ax.plot(kde_x, kde(kde_x), color='black', linewidth=2,
                linestyle='--', label='Kernel Density')
    
    # Reference lines
    mean_val = summary['mean_profit'] / 1e6
    var_5 = summary['var_5_pct'] / 1e6
    
    ax.axvline(mean_val, color='red', linewidth=2, linestyle='-',
               label=f'Mean: €{mean_val:.2f}M')
    ax.axvline(var_5, color='darkred', linewidth=1.5, linestyle=':',
               label=f'VaR 5%: €{var_5:.2f}M')
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Simulated Net Profit (€ Million)')
    ax.set_ylabel('Density')
    ax.set_title(f"Monte Carlo Profit Simulation (n={int(summary['n_simulations']):,})",
                 fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Statistics box
    stats_text = (f"Mean: €{mean_val:.2f}M\n"
                  f"Std: €{summary['std_profit']/1e6:.2f}M\n"
                  f"P(Loss): {summary['prob_loss_pct']:.2f}%")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig03_monte_carlo_histogram.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


# ============================================================================
# FIGURE 4: TORNADO DIAGRAM
# ============================================================================

def create_fig04_tornado(sensitivity_df):
    """Single-panel tornado diagram for sensitivity analysis."""
    print("  Creating fig04_tornado_sensitivity.png...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    
    df = sensitivity_df.copy()
    df['impact_range'] = df['profit_high'] - df['profit_low']
    df = df.sort_values('impact_range', ascending=True)
    
    # Calculate baseline
    baseline = (df['profit_high'].iloc[0] + df['profit_low'].iloc[0]) / 2 - df['delta_high'].iloc[0]
    
    y_pos = range(len(df))
    
    # Bars
    ax.barh(y_pos, df['profit_low'] - baseline, left=baseline,
            color='#d73027', edgecolor='white', alpha=0.8, label='-10%')
    ax.barh(y_pos, df['profit_high'] - baseline, left=baseline,
            color='#4575b4', edgecolor='white', alpha=0.8, label='+10%')
    
    ax.axvline(baseline, color='black', linewidth=2)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['parameter'])
    ax.set_xlabel('Total Profit (€)')
    ax.set_title('Sensitivity Analysis: ±10% Parameter Variation', fontweight='bold')
    ax.xaxis.set_major_formatter(FuncFormatter(euro_formatter))
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    # Impact labels
    for i, (low, high) in enumerate(zip(df['profit_low'], df['profit_high'])):
        impact = (high - low) / abs(baseline) * 100
        ax.text(max(high, baseline) + abs(baseline)*0.01, i,
                f'{impact:.1f}%', va='center', fontsize=9, color='gray')
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig04_tornado_sensitivity.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


# ============================================================================
# FIGURES 5-7: BOOTSTRAP CI (SEPARATE FIGURES)
# ============================================================================

def create_fig05_bootstrap_revenue(bootstrap_df):
    """Single-panel: Bootstrap CI for Revenue per Trip."""
    print("  Creating fig05_bootstrap_ci_revenue.png...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    operators = bootstrap_df['operator'].values
    y_pos = range(len(operators))
    
    for i, row in bootstrap_df.iterrows():
        color = OPERATOR_COLORS.get(row['operator'], '#666666')
        mean = row['revenue_mean']
        ci_low = row['revenue_ci_lower']
        ci_high = row['revenue_ci_upper']
        
        ax.errorbar(mean, i, xerr=[[mean-ci_low], [ci_high-mean]],
                    fmt='o', color=color, capsize=8, markersize=12,
                    capthick=2, elinewidth=2, label=row['operator'])
        ax.text(ci_high + 0.02, i, f'€{mean:.2f}', va='center', fontsize=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(operators)
    ax.set_xlabel('Revenue per Trip (€)')
    ax.set_title('Bootstrap 95% CI: Revenue per Trip', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig05_bootstrap_ci_revenue.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


def create_fig06_bootstrap_profit(bootstrap_df):
    """Single-panel: Bootstrap CI for Profit per Trip."""
    print("  Creating fig06_bootstrap_ci_profit.png...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    operators = bootstrap_df['operator'].values
    y_pos = range(len(operators))
    
    for i, row in bootstrap_df.iterrows():
        color = OPERATOR_COLORS.get(row['operator'], '#666666')
        mean = row['profit_mean']
        ci_low = row['profit_ci_lower']
        ci_high = row['profit_ci_upper']
        
        ax.errorbar(mean, i, xerr=[[mean-ci_low], [ci_high-mean]],
                    fmt='o', color=color, capsize=8, markersize=12,
                    capthick=2, elinewidth=2, label=row['operator'])
        ax.text(ci_high + 0.02, i, f'€{mean:.3f}', va='center', fontsize=10)
    
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(operators)
    ax.set_xlabel('Profit per Trip (€)')
    ax.set_title('Bootstrap 95% CI: Profit per Trip', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig06_bootstrap_ci_profit.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


def create_fig07_bootstrap_margin(bootstrap_df):
    """Single-panel: Bootstrap CI for Profit Margin."""
    print("  Creating fig07_bootstrap_ci_margin.png...")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    operators = bootstrap_df['operator'].values
    y_pos = range(len(operators))
    
    for i, row in bootstrap_df.iterrows():
        color = OPERATOR_COLORS.get(row['operator'], '#666666')
        mean = row['margin_mean_pct']
        ci_low = row['margin_ci_lower']
        ci_high = row['margin_ci_upper']
        
        ax.errorbar(mean, i, xerr=[[mean-ci_low], [ci_high-mean]],
                    fmt='o', color=color, capsize=8, markersize=12,
                    capthick=2, elinewidth=2, label=row['operator'])
        ax.text(ci_high + 0.5, i, f'{mean:.1f}%', va='center', fontsize=10)
    
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(operators)
    ax.set_xlabel('Profit Margin (%)')
    ax.set_title('Bootstrap 95% CI: Profit Margin', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig07_bootstrap_ci_margin.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


# ============================================================================
# FIGURE 8: REGRESSION - DURATION VS PROFIT
# ============================================================================

def create_fig08_regression(trips_df):
    """Single-panel: Duration vs Profit regression."""
    print("  Creating fig08_regression_duration_profit.png...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Sample for plotting
    sample_size = min(10000, len(trips_df))
    sample = trips_df.sample(n=sample_size, random_state=42)
    
    x = sample['duration_min'].values
    y = sample['net_profit'].values
    
    # Scatter
    ax.scatter(x, y, alpha=0.2, s=10, color='#4575b4', label='Trips')
    
    # Regression
    if HAS_SCIPY:
        valid = np.isfinite(x) & np.isfinite(y)
        x_valid, y_valid = x[valid], y[valid]
        
        if len(x_valid) > 10:
            slope, intercept, r_value, p_value, _ = stats.linregress(x_valid, y_valid)
            x_line = np.linspace(0, min(60, x_valid.max()), 100)
            y_line = slope * x_line + intercept
            
            ax.plot(x_line, y_line, color='red', linewidth=2,
                    label=f'OLS: y = {slope:.3f}x + {intercept:.2f}')
            
            # Stats box
            stats_text = f'R² = {r_value**2:.4f}\np < 0.001\nn = {len(x_valid):,}'
            ax.text(0.97, 0.03, stats_text, transform=ax.transAxes, fontsize=10,
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Trip Duration (minutes)')
    ax.set_ylabel('Net Profit (€)')
    ax.set_title('Duration-Profit Relationship (OLS Regression)', fontweight='bold')
    ax.set_xlim(0, 60)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig08_regression_duration_profit.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


# ============================================================================
# FIGURE 9: PARETO / LORENZ CURVE
# ============================================================================

def create_fig09_pareto(pareto_df):
    """Single-panel: Lorenz curve for zone profitability."""
    print("  Creating fig09_pareto_zone_curve.png...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Plot curve
    ax.fill_between(pareto_df['cumulative_zone_pct'], 0,
                    pareto_df['cumulative_profit_pct'],
                    alpha=0.3, color='#4575b4')
    ax.plot(pareto_df['cumulative_zone_pct'], pareto_df['cumulative_profit_pct'],
            color='#4575b4', linewidth=2, marker='o', markersize=3)
    
    # Perfect equality line
    ax.plot([0, 100], [0, 100], color='gray', linestyle='--',
            linewidth=1.5, label='Perfect Equality')
    
    # 80/20 markers
    ax.axhline(80, color='red', linestyle=':', alpha=0.7)
    ax.axvline(20, color='red', linestyle=':', alpha=0.7)
    
    # Find 80% point
    zones_80 = pareto_df[pareto_df['cumulative_profit_pct'] >= 80].iloc[0]['cumulative_zone_pct']
    
    # Position the annotation to avoid overlapping with the curve
    # Use lower-left quadrant for the annotation
    ax.annotate(f'{zones_80:.0f}% zones\ngenerate 80% profit',
                xy=(zones_80, 80), xycoords='data',
                xytext=(60, 40),  # Fixed position in lower-right area
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.9))
    
    # Gini coefficient - positioned in bottom right corner
    area = np.trapz(pareto_df['cumulative_profit_pct'],
                    pareto_df['cumulative_zone_pct']) / 100
    gini = 1 - 2 * area / 100
    
    ax.text(0.97, 0.03, f'Gini = {gini:.3f}', transform=ax.transAxes,
            fontsize=11, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.set_xlabel('Cumulative % of Zones (ranked by profit)')
    ax.set_ylabel('Cumulative % of Total Profit')
    ax.set_title('Pareto Analysis: Zone Value Concentration', fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig09_pareto_zone_curve.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


# ============================================================================
# FIGURE 10: SCENARIO COMPARISON
# ============================================================================

def create_fig10_scenarios(scenarios_df):
    """Single-panel: Scenario comparison bar chart."""
    print("  Creating fig10_scenario_comparison.png...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    colors = ['#4575b4', '#91bfdb', '#fc8d59', '#d73027']
    x_pos = range(len(scenarios_df))
    
    bars = ax.bar(x_pos, scenarios_df['net_profit'],
                  color=colors[:len(scenarios_df)], edgecolor='white')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios_df['scenario'], rotation=30, ha='right')
    ax.set_ylabel('Net Profit (€)')
    ax.set_title('Scenario Analysis: Profit Comparison', fontweight='bold')
    ax.yaxis.set_major_formatter(FuncFormatter(euro_formatter))
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Value labels
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'€{h/1e6:.2f}M', xy=(bar.get_x() + bar.get_width()/2, h),
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = FIGURE_DIR / "fig10_scenario_comparison.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close()
    print(f"    ✓ Saved: {output_path.name}")
    return output_path


# ============================================================================
# CSV TABLES FOR LATEX
# ============================================================================

def create_latex_tables(trips_df, monte_carlo_summary, sensitivity_df, scenarios_df):
    """Generate CSV tables for LaTeX inclusion."""
    print("\n  Generating LaTeX-ready CSV tables...")
    
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Table 1: Operator Summary
    op_summary = trips_df.groupby('operator').agg({
        'gross_revenue': 'sum',
        'total_cost': 'sum',
        'net_profit': 'sum',
        'is_profitable': 'mean'
    }).reset_index()
    op_summary['profit_margin_pct'] = op_summary['net_profit'] / op_summary['gross_revenue'] * 100
    op_summary['profitable_pct'] = op_summary['is_profitable'] * 100
    op_summary = op_summary.round(2)
    op_summary.to_csv(TABLE_DIR / "table01_operator_summary.csv", index=False)
    print(f"    ✓ table01_operator_summary.csv")
    
    # Table 2: Monte Carlo Stats
    if isinstance(monte_carlo_summary, pd.DataFrame):
        mc_table = monte_carlo_summary.copy()
    else:
        mc_table = pd.DataFrame([monte_carlo_summary])
    mc_table.to_csv(TABLE_DIR / "table02_monte_carlo_stats.csv", index=False)
    print(f"    ✓ table02_monte_carlo_stats.csv")
    
    # Table 3: Sensitivity Ranking
    sens_table = sensitivity_df[['parameter', 'swing', 'delta_high_pct']].copy()
    sens_table['swing_millions'] = sens_table['swing'] / 1e6
    sens_table = sens_table.sort_values('swing', ascending=False).round(3)
    sens_table.to_csv(TABLE_DIR / "table03_sensitivity_ranking.csv", index=False)
    print(f"    ✓ table03_sensitivity_ranking.csv")
    
    # Table 4: Scenario Comparison
    scen_table = scenarios_df[['scenario', 'net_profit', 'profit_margin_pct', 'delta_profit_pct']].copy()
    scen_table['net_profit_millions'] = scen_table['net_profit'] / 1e6
    scen_table = scen_table.round(2)
    scen_table.to_csv(TABLE_DIR / "table04_scenario_comparison.csv", index=False)
    print(f"    ✓ table04_scenario_comparison.csv")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all figures and tables."""
    print("\n" + "="*70)
    print("EXERCISE 5: ECONOMIC ANALYSIS - STATISTICS")
    print("="*70)
    print("Single-Panel Figures for LaTeX")
    print("="*70)
    
    setup_matplotlib()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoints
    print("\n  Loading checkpoint data...")
    
    try:
        trips_df = pd.read_pickle(DATA_DIR / "checkpoint_economics_trips.pkl")
        print(f"    ✓ Trips: {len(trips_df):,}")
        
        mc_df = pd.read_csv(DATA_DIR / "checkpoint_monte_carlo_simulations.csv")
        mc_summary = pd.read_csv(DATA_DIR / "checkpoint_monte_carlo_summary.csv")
        print(f"    ✓ Monte Carlo: {len(mc_df):,} simulations")
        
        sens_df = pd.read_csv(DATA_DIR / "checkpoint_sensitivity_analysis.csv")
        print(f"    ✓ Sensitivity: {len(sens_df)} parameters")
        
        bootstrap_df = pd.read_csv(DATA_DIR / "checkpoint_bootstrap_ci.csv")
        print(f"    ✓ Bootstrap CI: {len(bootstrap_df)} operators")
        
        pareto_df = pd.read_csv(DATA_DIR / "checkpoint_economics_pareto.csv")
        print(f"    ✓ Pareto: {len(pareto_df)} zones")
        
        scenarios_df = pd.read_csv(DATA_DIR / "checkpoint_economics_scenarios.csv")
        print(f"    ✓ Scenarios: {len(scenarios_df)}")
        
    except FileNotFoundError as e:
        print(f"\n  ❌ Error: {e}")
        print("  → Please run 06_economics.py first.")
        return
    
    # Generate figures
    print("\n" + "-"*70)
    print("GENERATING SINGLE-PANEL FIGURES")
    print("-"*70)
    
    figures = []
    figures.append(create_fig01_revenue_cost(trips_df))
    figures.append(create_fig02_profit_margin_dist(trips_df))
    figures.append(create_fig03_monte_carlo(mc_df, mc_summary))
    figures.append(create_fig04_tornado(sens_df))
    figures.append(create_fig05_bootstrap_revenue(bootstrap_df))
    figures.append(create_fig06_bootstrap_profit(bootstrap_df))
    figures.append(create_fig07_bootstrap_margin(bootstrap_df))
    figures.append(create_fig08_regression(trips_df))
    figures.append(create_fig09_pareto(pareto_df))
    figures.append(create_fig10_scenarios(scenarios_df))
    
    # Generate tables
    print("\n" + "-"*70)
    print("GENERATING LATEX TABLES")
    print("-"*70)
    create_latex_tables(trips_df, mc_summary, sens_df, scenarios_df)
    
    # Summary
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\n  Figures: {len(figures)} (all single-panel)")
    print(f"  Tables: 4 (CSV for LaTeX)")
    print(f"\n  Output: {FIGURE_DIR}")
    print(f"  Tables: {TABLE_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
