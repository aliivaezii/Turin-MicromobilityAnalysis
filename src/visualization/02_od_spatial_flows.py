#!/usr/bin/env python3
"""
================================================================================
EXERCISE 2: ORIGIN-DESTINATION MATRIX ANALYSIS - Spatial Flow & OD Matrix Visualization
================================================================================

Origin-Destination spatial flow visualization module.

This module creates OD flow maps, desire line visualizations,
and zone-based choropleth analysis figures.

Creates spatial flow maps, zone-based choropleths, and
OD matrix visualizations for mobility pattern analysis.

Output Directory: outputs/figures/exercise2/

Author: Ali Vaezi
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import contextily as cx
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - visualization scripts are in src/visualization/, need to go up 3 levels to project root
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "outputs" / "reports" / "exercise2" / "combined"
ZONES_PATH = BASE_DIR / "data" / "raw" / "zone_statistiche_geo" / "zone_statistiche_geo.shp"
FIGURE_DIR = BASE_DIR / "outputs" / "figures" / "exercise2" / "spatial"

# CRS
CRS_WGS84 = "EPSG:4326"
CRS_UTM32N = "EPSG:32632"
CRS_WEB_MERCATOR = "EPSG:3857"

# Publication-quality styling
FIGSIZE_STANDARD = (10, 8)  # Single-panel figures for LaTeX
FIGSIZE_HEATMAP = (14, 12)
FIGSIZE_MAP = (16, 14)
FIGSIZE_WIDE = (18, 10)
FIGSIZE_MULTI = (18, 14)
DPI = 300
FONT_FAMILY = 'DejaVu Sans'

# Publication Color palettes
OPERATOR_COLORS = {
    'LIME': '#388E3C',   # Material Green 700
    'BIRD': '#1976D2',   # Material Blue 700
    'VOI': '#D32F2F'     # Material Red 700
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_matplotlib():
    """Configure matplotlib for PUBLICATION-QUALITY output."""
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': DPI,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3
    })


def load_checkpoints():
    """Load all required checkpoints for visualization."""
    print("\n" + "="*70)
    print("LOADING CHECKPOINTS FOR EXERCISE 2 VISUALIZATION")
    print("="*70)
    
    checkpoints = {}
    
    # 1. O-D Matrices
    for period in ['AllDay', 'Peak', 'OffPeak']:
        matrix_path = DATA_DIR / f'OD_Matrix_{period}.csv'
        if matrix_path.exists():
            checkpoints[f'matrix_{period.lower()}'] = pd.read_csv(matrix_path, index_col=0)
            print(f"✓ Loaded O-D Matrix: {period}")
        else:
            print(f"✗ Missing: {matrix_path.name}")
            checkpoints[f'matrix_{period.lower()}'] = None
    
    # 2. Advanced metrics
    metrics_path = DATA_DIR / 'checkpoint_od_metrics.csv'
    if metrics_path.exists():
        checkpoints['metrics'] = pd.read_csv(metrics_path)
        print(f"✓ Loaded O-D metrics")
    else:
        print(f"✗ Missing: {metrics_path.name}")
        checkpoints['metrics'] = None
    
    # 3. Chi-square results
    chi2_path = DATA_DIR / 'checkpoint_chi_square.csv'
    if chi2_path.exists():
        checkpoints['chi_square'] = pd.read_csv(chi2_path)
        print(f"✓ Loaded Chi-square results")
    else:
        print(f"✗ Missing: {chi2_path.name}")
        checkpoints['chi_square'] = None
    
    # 4. Zone clusters
    cluster_path = DATA_DIR / 'checkpoint_zone_clusters.csv'
    if cluster_path.exists():
        checkpoints['clusters'] = pd.read_csv(cluster_path)
        print(f"✓ Loaded zone clusters")
    else:
        print(f"✗ Missing: {cluster_path.name}")
        checkpoints['clusters'] = None
    
    # 5. Operator metrics
    op_metrics_path = DATA_DIR / 'checkpoint_operator_od_metrics.csv'
    if op_metrics_path.exists():
        checkpoints['operator_metrics'] = pd.read_csv(op_metrics_path)
        print(f"✓ Loaded operator O-D metrics")
    else:
        print(f"✗ Missing: {op_metrics_path.name}")
        checkpoints['operator_metrics'] = None
    
    # 6. Zones shapefile
    if ZONES_PATH.exists():
        checkpoints['zones'] = gpd.read_file(ZONES_PATH)
        print(f"✓ Loaded zones shapefile: {len(checkpoints['zones'])} zones")
    else:
        print(f"✗ Missing zones shapefile")
        checkpoints['zones'] = None
    
    # Summary
    available = sum(1 for k, v in checkpoints.items() if v is not None)
    print(f"\n→ Loaded {available}/{len(checkpoints)} checkpoint files")
    
    return checkpoints


def add_statistical_annotation(ax, text, x=0.02, y=0.98, fontsize=10):
    """Add a professional statistical annotation box."""
    props = dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        edgecolor='gray',
        alpha=0.95,
        linewidth=1
    )
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='top', bbox=props, family='monospace')


def add_north_arrow(ax, x=0.95, y=0.95):
    """Add a professional north arrow to the map."""
    ax.annotate('N', xy=(x, y), xycoords='axes fraction',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.annotate('', xy=(x, y-0.01), xycoords='axes fraction',
                xytext=(x, y - 0.07), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='black', lw=2))


def add_scale_bar(ax, gdf_utm):
    """Add a manual scale bar to the map."""
    bounds = gdf_utm.total_bounds
    width_m = bounds[2] - bounds[0]
    
    if width_m > 20000:
        bar_length_m = 5000
        label = '5 km'
    elif width_m > 10000:
        bar_length_m = 2000
        label = '2 km'
    else:
        bar_length_m = 1000
        label = '1 km'
    
    x_start, y_start = 0.05, 0.05
    bar_width = bar_length_m / width_m * 0.8
    
    ax.add_patch(plt.Rectangle(
        (x_start, y_start), bar_width, 0.015,
        transform=ax.transAxes,
        facecolor='black',
        edgecolor='black'
    ))
    ax.text(x_start + bar_width/2, y_start + 0.025, label,
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9, fontweight='bold')


# ============================================================================
# FIGURE 1: HIERARCHICAL CLUSTERING HEATMAP
# ============================================================================

def plot_clustered_heatmap(matrix, clusters_df, output_path):
    """
    Heatmap with hierarchical clustering - PUBLICATION-QUALITY Quality.
    
    Reorders zones by cluster for clearer pattern visualization.
    """
    print("\n" + "-"*50)
    print("Figure 1: Hierarchical Clustering Heatmap")
    print("-"*50)
    
    if matrix is None:
        print("  ✗ Skipping: matrix data not available")
        return False
    
    # Remove TOTAL row/column
    matrix_clean = matrix.drop('TOTAL', axis=0, errors='ignore')
    matrix_clean = matrix_clean.drop('TOTAL', axis=1, errors='ignore')
    
    # Get top 40 zones by activity
    origin_totals = matrix_clean.sum(axis=1)
    dest_totals = matrix_clean.sum(axis=0)
    combined = origin_totals.add(dest_totals, fill_value=0)
    top_zones = combined.nlargest(40).index.tolist()
    
    matrix_filtered = matrix_clean.loc[
        matrix_clean.index.isin(top_zones),
        matrix_clean.columns.isin(top_zones)
    ]
    
    # Log transform
    matrix_log = np.log1p(matrix_filtered)
    
    print(f"  → Visualizing {len(matrix_filtered)} x {len(matrix_filtered.columns)} zones")
    
    # Create clustermap
    g = sns.clustermap(
        matrix_log,
        method='ward',
        cmap='YlOrRd',
        figsize=(14, 12),
        dendrogram_ratio=(0.15, 0.15),
        cbar_kws={'label': 'Log(Trips + 1)'},
        linewidths=0.5,
        linecolor='white',
        xticklabels=True,
        yticklabels=True
    )
    
    # Title
    g.fig.suptitle('O-D Matrix with Hierarchical Clustering\nTurin E-Scooter Shared Mobility', 
                   fontsize=16, fontweight='bold', y=1.02)
    
    # Rotate x labels
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    
    g.ax_heatmap.set_xlabel('Destination Zone', fontsize=12, fontweight='bold')
    g.ax_heatmap.set_ylabel('Origin Zone', fontsize=12, fontweight='bold')
    
    # Save
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 2: PEAK VS OFF-PEAK COMPARISON
# ============================================================================

def plot_peak_offpeak_comparison(matrix_peak, matrix_offpeak, chi2_results, output_path):
    """
    Side-by-side comparison of peak and off-peak patterns.
    """
    print("\n" + "-"*50)
    print("Figure 2: Peak vs Off-Peak Comparison")
    print("-"*50)
    
    if matrix_peak is None or matrix_offpeak is None:
        print("  ✗ Skipping: peak/off-peak data not available")
        return False
    
    # Clean matrices
    peak_clean = matrix_peak.drop('TOTAL', axis=0, errors='ignore').drop('TOTAL', axis=1, errors='ignore')
    offpeak_clean = matrix_offpeak.drop('TOTAL', axis=0, errors='ignore').drop('TOTAL', axis=1, errors='ignore')
    
    # Get top 25 zones
    combined = peak_clean.sum(axis=1).add(peak_clean.sum(axis=0), fill_value=0)
    top_zones = combined.nlargest(25).index.tolist()
    
    peak_filtered = peak_clean.loc[
        peak_clean.index.isin(top_zones),
        peak_clean.columns.isin(top_zones)
    ]
    offpeak_filtered = offpeak_clean.loc[
        offpeak_clean.index.isin(top_zones),
        offpeak_clean.columns.isin(top_zones)
    ]
    
    # Sort both the same way
    sorted_zones = peak_filtered.sum(axis=1).sort_values(ascending=False).index
    peak_filtered = peak_filtered.loc[sorted_zones, sorted_zones.intersection(peak_filtered.columns)]
    offpeak_filtered = offpeak_filtered.loc[sorted_zones, sorted_zones.intersection(offpeak_filtered.columns)]
    
    # Log transform
    peak_log = np.log1p(peak_filtered)
    offpeak_log = np.log1p(offpeak_filtered)
    
    # Get chi-square stats for annotation
    chi2_text = None
    if chi2_results is not None and len(chi2_results) > 0:
        chi2 = chi2_results['chi2_statistic'].values[0]
        cramers_v = chi2_results['cramers_v'].values[0]
        chi2_text = f"χ² = {chi2:,.0f}, Cramér's V = {cramers_v:.3f}"
    
    # Output directory from path
    output_dir = output_path.parent
    
    # ========== Figure 2a: Peak Hours O-D Heatmap ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    sns.heatmap(peak_log, ax=ax, cmap='Reds', 
                cbar_kws={'label': 'Log(Trips + 1)', 'shrink': 0.8},
                xticklabels=True, yticklabels=True,
                linewidths=0.3, linecolor='white')
    ax.set_title('Peak Hours O-D Matrix (7-9 AM, 5-7 PM)\nTurin E-Scooter Network', 
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Destination Zone', fontsize=11, fontweight='bold')
    ax.set_ylabel('Origin Zone', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
    
    if chi2_text:
        ax.text(0.5, -0.15, f'Chi-Square Test: {chi2_text}', transform=ax.transAxes,
               ha='center', fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig02a_peak_od_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig02a_peak_od_heatmap.png")
    
    # ========== Figure 2b: Off-Peak Hours O-D Heatmap ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    
    sns.heatmap(offpeak_log, ax=ax, cmap='Blues',
                cbar_kws={'label': 'Log(Trips + 1)', 'shrink': 0.8},
                xticklabels=True, yticklabels=True,
                linewidths=0.3, linecolor='white')
    ax.set_title('Off-Peak Hours O-D Matrix\nTurin E-Scooter Network', 
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Destination Zone', fontsize=11, fontweight='bold')
    ax.set_ylabel('Origin Zone', fontsize=11, fontweight='bold')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
    
    if chi2_text:
        ax.text(0.5, -0.15, f'Chi-Square Test: {chi2_text}', transform=ax.transAxes,
               ha='center', fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig02b_offpeak_od_heatmap.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig02b_offpeak_od_heatmap.png")
    
    # Also save combined for reference
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    
    sns.heatmap(peak_log, ax=axes[0], cmap='Reds', 
                cbar_kws={'label': 'Log(Trips + 1)', 'shrink': 0.7},
                xticklabels=True, yticklabels=True,
                linewidths=0.3, linecolor='white')
    axes[0].set_title('Peak Hours (7-9 AM, 5-7 PM)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Destination Zone', fontsize=11)
    axes[0].set_ylabel('Origin Zone', fontsize=11)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(axes[0].yaxis.get_majorticklabels(), rotation=0, fontsize=8)
    
    sns.heatmap(offpeak_log, ax=axes[1], cmap='Blues',
                cbar_kws={'label': 'Log(Trips + 1)', 'shrink': 0.7},
                xticklabels=True, yticklabels=True,
                linewidths=0.3, linecolor='white')
    axes[1].set_title('Off-Peak Hours', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Destination Zone', fontsize=11)
    axes[1].set_ylabel('Origin Zone', fontsize=11)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(axes[1].yaxis.get_majorticklabels(), rotation=0, fontsize=8)
    
    plt.suptitle('Temporal O-D Pattern Comparison\nTurin E-Scooter Network', 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig02_combined_peak_offpeak.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig02_combined_peak_offpeak.png (reference)")
    return True


# ============================================================================
# FIGURE 3: FLOW ASYMMETRY MAP
# ============================================================================

def plot_asymmetry_heatmap(matrix, output_path):
    """
    Visualize directional asymmetry in O-D flows.
    
    Asymmetry = (F_ij - F_ji) / (F_ij + F_ji)
    Red = more trips from i to j
    Blue = more trips from j to i
    """
    print("\n" + "-"*50)
    print("Figure 3: Flow Asymmetry Heatmap")
    print("-"*50)
    
    if matrix is None:
        print("  ✗ Skipping: matrix data not available")
        return False
    
    # Clean matrix
    matrix_clean = matrix.drop('TOTAL', axis=0, errors='ignore')
    matrix_clean = matrix_clean.drop('TOTAL', axis=1, errors='ignore')
    
    # Get top zones
    combined = matrix_clean.sum(axis=1).add(matrix_clean.sum(axis=0), fill_value=0)
    top_zones = combined.nlargest(30).index.tolist()
    
    matrix_filtered = matrix_clean.loc[
        matrix_clean.index.isin(top_zones),
        matrix_clean.columns.isin(top_zones)
    ].copy()
    
    # Calculate asymmetry matrix
    asymmetry = np.zeros_like(matrix_filtered.values, dtype=float)
    for i, origin in enumerate(matrix_filtered.index):
        for j, dest in enumerate(matrix_filtered.columns):
            f_ij = matrix_filtered.loc[origin, dest] if dest in matrix_filtered.columns else 0
            f_ji = matrix_filtered.loc[dest, origin] if origin in matrix_filtered.columns and dest in matrix_filtered.index else 0
            total = f_ij + f_ji
            if total > 0:
                asymmetry[i, j] = (f_ij - f_ji) / total
    
    asymmetry_df = pd.DataFrame(asymmetry, 
                                 index=matrix_filtered.index,
                                 columns=matrix_filtered.columns)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    
    # Use diverging colormap centered at 0
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    
    sns.heatmap(asymmetry_df, ax=ax, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Asymmetry Index\n← More O→D | More D→O →', 'shrink': 0.7},
                xticklabels=True, yticklabels=True,
                linewidths=0.3, linecolor='white')
    
    ax.set_title('O-D Flow Asymmetry Analysis\nTurin E-Scooter Network', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Destination Zone', fontsize=12, fontweight='bold')
    ax.set_ylabel('Origin Zone', fontsize=12, fontweight='bold')
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=9)
    
    # Add interpretation
    stats_text = "Asymmetry Interpretation:\n"
    stats_text += "Red: Net outflow (O → D)\n"
    stats_text += "Blue: Net inflow (D → O)\n"
    stats_text += "White: Balanced flows"
    add_statistical_annotation(ax, stats_text, x=0.02, y=0.25)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 4: PROFESSIONAL FLOW MAP
# ============================================================================

def plot_professional_flow_map(matrix, zones_gdf, output_path, top_n=30):
    """
    Publication-quality flow map with curved arrows.
    """
    print("\n" + "-"*50)
    print("Figure 4: Professional Flow Map")
    print("-"*50)
    
    if matrix is None or zones_gdf is None:
        print("  ✗ Skipping: data not available")
        return False
    
    # Clean matrix
    matrix_clean = matrix.drop('TOTAL', axis=0, errors='ignore')
    matrix_clean = matrix_clean.drop('TOTAL', axis=1, errors='ignore')
    
    # Get top corridors
    flows = []
    for origin in matrix_clean.index:
        for dest in matrix_clean.columns:
            if origin != dest:
                trips = matrix_clean.loc[origin, dest]
                if trips > 0:
                    flows.append({'origin': origin, 'dest': dest, 'trips': trips})
    
    flows_df = pd.DataFrame(flows).sort_values('trips', ascending=False).head(top_n)
    
    print(f"  → Showing top {len(flows_df)} corridors")
    
    # Project zones
    zones_plot = zones_gdf.to_crs(CRS_WEB_MERCATOR)
    zones_utm = zones_gdf.to_crs(CRS_UTM32N)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_MAP)
    
    # Plot zones
    zones_plot.plot(ax=ax, color='#f5f5f5', edgecolor='#cccccc', linewidth=0.5, alpha=0.9)
    
    # Highlight zones with flows
    active_zones = set(flows_df['origin'].tolist() + flows_df['dest'].tolist())
    zones_plot[zones_plot['ZONASTAT'].isin(active_zones)].plot(
        ax=ax, color='#ffe6cc', edgecolor='#ff9933', linewidth=1, alpha=0.7
    )
    
    # Draw flow lines
    max_trips = flows_df['trips'].max()
    min_trips = flows_df['trips'].min()
    
    cmap = LinearSegmentedColormap.from_list('flow', 
        ['#fee5d9', '#fcbba1', '#fc9272', '#fb6a4a', '#de2d26', '#a50f15'], N=256)
    
    for _, row in flows_df.iterrows():
        try:
            origin_geom = zones_plot[zones_plot['ZONASTAT'] == row['origin']]
            dest_geom = zones_plot[zones_plot['ZONASTAT'] == row['dest']]
            
            if len(origin_geom) == 0 or len(dest_geom) == 0:
                continue
            
            start = origin_geom.geometry.centroid.iloc[0]
            end = dest_geom.geometry.centroid.iloc[0]
            
            # Normalize for styling
            norm = (row['trips'] - min_trips) / (max_trips - min_trips) if max_trips > min_trips else 0.5
            
            # Line width and color
            lw = 1 + norm * 5
            color = cmap(norm)
            alpha = 0.4 + norm * 0.5
            
            # Draw curved line (simple bezier approximation)
            mid_x = (start.x + end.x) / 2 + (end.y - start.y) * 0.1
            mid_y = (start.y + end.y) / 2 - (end.x - start.x) * 0.1
            
            ax.plot([start.x, mid_x, end.x], [start.y, mid_y, end.y],
                   color=color, alpha=alpha, linewidth=lw, solid_capstyle='round')
            
        except Exception as e:
            continue
    
    # Add zone labels for top zones
    for zone in list(active_zones)[:10]:
        zone_row = zones_plot[zones_plot['ZONASTAT'] == zone]
        if len(zone_row) > 0:
            centroid = zone_row.geometry.centroid.iloc[0]
            ax.annotate(str(zone), xy=(centroid.x, centroid.y),
                       fontsize=8, fontweight='bold', ha='center', va='center',
                       bbox=dict(boxstyle='circle,pad=0.15', facecolor='white', 
                                edgecolor='gray', alpha=0.9), zorder=10)
    
    # Add basemap
    try:
        cx.add_basemap(ax, crs=CRS_WEB_MERCATOR, source=cx.providers.CartoDB.Positron, alpha=0.6)
    except:
        pass
    
    # Cartographic elements
    add_north_arrow(ax)
    add_scale_bar(ax, zones_utm)
    
    ax.set_title(f'Top {top_n} E-Scooter Corridors\nTurin Metropolitan Area', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_trips, vmax=max_trips))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=30, pad=0.02)
    cbar.set_label('Trip Count', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path.name}")
    return True


# ============================================================================
# FIGURE 5: O-D METRICS COMPARISON
# ============================================================================

def plot_metrics_comparison(metrics_df, output_path):
    """
    Bar chart comparison of O-D metrics across time periods.
    Generates separate single-panel figures for LaTeX compatibility.
    """
    print("\n" + "-"*50)
    print("Figure 5: O-D Metrics Comparison")
    print("-"*50)
    
    if metrics_df is None:
        print("  ✗ Skipping: metrics data not available")
        return False
    
    output_dir = output_path.parent
    periods = metrics_df['name'].tolist()
    colors = ['#2E7D32', '#FF8F00', '#1565C0']  # Green, Orange, Blue
    
    # ========== Figure 5a: Gini Coefficient ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    gini_vals = metrics_df['gini_coefficient'].values
    bars = ax.bar(periods, gini_vals, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax.set_title('O-D Flow Inequality by Time Period\nTurin E-Scooter Network (0 = Equal, 1 = Concentrated)', 
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, gini_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Moderate Inequality')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig05a_gini_coefficient.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig05a_gini_coefficient.png")
    
    # ========== Figure 5b: Shannon Entropy ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    entropy_vals = metrics_df['shannon_entropy'].values
    bars = ax.bar(periods, entropy_vals, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Normalized Entropy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax.set_title('O-D Flow Diversity by Time Period\nTurin E-Scooter Network (0 = Concentrated, 1 = Uniform)', 
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, entropy_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig05b_shannon_entropy.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig05b_shannon_entropy.png")
    
    # ========== Figure 5c: Flow Asymmetry ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    asym_vals = metrics_df['flow_asymmetry'].values
    bars = ax.bar(periods, asym_vals, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Asymmetry Index', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax.set_title('O-D Flow Directional Imbalance\nTurin E-Scooter Network (0 = Symmetric, 1 = One-way)', 
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, asym_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig05c_flow_asymmetry.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig05c_flow_asymmetry.png")
    
    # ========== Figure 5d: Spatial Concentration ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    x = np.arange(len(periods))
    width = 0.35
    conc10 = metrics_df['concentration_top10'].values
    conc50 = metrics_df['concentration_top50'].values
    
    bars_a = ax.bar(x - width/2, conc10 * 100, width, label='Top 10 Corridors', 
                   color='#E53935', edgecolor='black', linewidth=1)
    bars_b = ax.bar(x + width/2, conc50 * 100, width, label='Top 50 Corridors',
                   color='#1E88E5', edgecolor='black', linewidth=1)
    ax.set_ylabel('% of Total Trips', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
    ax.set_title('Spatial Concentration of O-D Flows\nTurin E-Scooter Network (% Captured by Top Corridors)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    for bar, val in zip(bars_a, conc10):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val*100:.1f}%', ha='center', fontsize=10)
    for bar, val in zip(bars_b, conc50):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val*100:.1f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig05d_spatial_concentration.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig05d_spatial_concentration.png")
    
    # Also save combined for reference
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Gini
    bars1 = axes[0, 0].bar(periods, gini_vals, color=colors, edgecolor='black', linewidth=1)
    axes[0, 0].set_ylabel('Gini Coefficient', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Flow Inequality', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim(0, 1)
    
    # Entropy
    bars2 = axes[0, 1].bar(periods, entropy_vals, color=colors, edgecolor='black', linewidth=1)
    axes[0, 1].set_ylabel('Normalized Entropy', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Flow Diversity', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 1)
    
    # Asymmetry
    bars3 = axes[1, 0].bar(periods, asym_vals, color=colors, edgecolor='black', linewidth=1)
    axes[1, 0].set_ylabel('Asymmetry Index', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Directional Imbalance', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim(0, 1)
    
    # Concentration
    axes[1, 1].bar(x - width/2, conc10 * 100, width, label='Top 10', color='#E53935')
    axes[1, 1].bar(x + width/2, conc50 * 100, width, label='Top 50', color='#1E88E5')
    axes[1, 1].set_ylabel('% of Total Trips', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Spatial Concentration', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(periods)
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 100)
    
    plt.suptitle('Advanced O-D Metrics Comparison\nTurin E-Scooter Network', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig05_combined_metrics.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig05_combined_metrics.png (reference)")
    
    return True


# ============================================================================
# FIGURE 6: OPERATOR O-D COMPARISON
# ============================================================================

def plot_operator_od_comparison(operator_metrics, output_path):
    """
    Comparison of O-D metrics across operators.
    Generates separate single-panel figures for LaTeX compatibility.
    """
    print("\n" + "-"*50)
    print("Figure 6: Operator O-D Comparison")
    print("-"*50)
    
    if operator_metrics is None:
        print("  ✗ Skipping: operator metrics not available")
        return False
    
    output_dir = output_path.parent
    operators = operator_metrics['name'].tolist()
    colors = [OPERATOR_COLORS.get(op, 'gray') for op in operators]
    
    # ========== Figure 6a: Total Trips by Operator ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    trips = operator_metrics['total_trips'].values
    bars = ax.bar(operators, trips, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Total Trips', fontsize=12, fontweight='bold')
    ax.set_xlabel('Operator', fontsize=12, fontweight='bold')
    ax.set_title('Trip Volume by Operator\nTurin E-Scooter Network', fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    for bar, val in zip(bars, trips):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + trips.max()*0.02,
               f'{val/1e6:.2f}M', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig06a_operator_trips.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig06a_operator_trips.png")
    
    # ========== Figure 6b: Active Corridors by Operator ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    corridors = operator_metrics['n_corridors'].values
    bars = ax.bar(operators, corridors, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Active Corridors', fontsize=12, fontweight='bold')
    ax.set_xlabel('Operator', fontsize=12, fontweight='bold')
    ax.set_title('Network Coverage by Operator\nTurin E-Scooter Network', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, corridors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + corridors.max()*0.02,
               f'{val:,}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig06b_operator_corridors.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig06b_operator_corridors.png")
    
    # ========== Figure 6c: Gini Coefficient by Operator ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    gini = operator_metrics['gini_coefficient'].values
    bars = ax.bar(operators, gini, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Gini Coefficient', fontsize=12, fontweight='bold')
    ax.set_xlabel('Operator', fontsize=12, fontweight='bold')
    ax.set_title('Flow Inequality by Operator\nTurin E-Scooter Network (0 = Equal, 1 = Concentrated)', 
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, gini):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig06c_operator_gini.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig06c_operator_gini.png")
    
    # ========== Figure 6d: Shannon Entropy by Operator ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    entropy = operator_metrics['shannon_entropy'].values
    bars = ax.bar(operators, entropy, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Normalized Entropy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Operator', fontsize=12, fontweight='bold')
    ax.set_title('Flow Diversity by Operator\nTurin E-Scooter Network (0 = Concentrated, 1 = Uniform)', 
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, entropy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig06d_operator_entropy.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig06d_operator_entropy.png")
    
    # ========== Figure 6e: Flow Asymmetry by Operator ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    asym = operator_metrics['flow_asymmetry'].values
    bars = ax.bar(operators, asym, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Asymmetry Index', fontsize=12, fontweight='bold')
    ax.set_xlabel('Operator', fontsize=12, fontweight='bold')
    ax.set_title('Directional Balance by Operator\nTurin E-Scooter Network (0 = Symmetric, 1 = One-way)', 
                fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, asym):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig06e_operator_asymmetry.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig06e_operator_asymmetry.png")
    
    # ========== Figure 6f: Intra-zonal % by Operator ==========
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    intra = operator_metrics['intra_zonal_pct'].values
    bars = ax.bar(operators, intra, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Intra-zonal %', fontsize=12, fontweight='bold')
    ax.set_xlabel('Operator', fontsize=12, fontweight='bold')
    ax.set_title('Local vs Long-distance Trips by Operator\nTurin E-Scooter Network', 
                fontsize=13, fontweight='bold')
    for bar, val in zip(bars, intra):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + intra.max()*0.02,
               f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig06f_operator_intrazonal.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig06f_operator_intrazonal.png")
    
    # Also save combined for reference
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_WIDE)
    
    # Trip Volume
    axes[0, 0].bar(operators, trips, color=colors, edgecolor='black', linewidth=1)
    axes[0, 0].set_ylabel('Total Trips', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('Trip Volume', fontsize=11, fontweight='bold')
    axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # Corridors
    axes[0, 1].bar(operators, corridors, color=colors, edgecolor='black', linewidth=1)
    axes[0, 1].set_ylabel('Active Corridors', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('Network Coverage', fontsize=11, fontweight='bold')
    
    # Gini
    axes[0, 2].bar(operators, gini, color=colors, edgecolor='black', linewidth=1)
    axes[0, 2].set_ylabel('Gini Coefficient', fontsize=10, fontweight='bold')
    axes[0, 2].set_title('Flow Inequality', fontsize=11, fontweight='bold')
    axes[0, 2].set_ylim(0, 1)
    
    # Entropy
    axes[1, 0].bar(operators, entropy, color=colors, edgecolor='black', linewidth=1)
    axes[1, 0].set_ylabel('Normalized Entropy', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('Flow Diversity', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylim(0, 1)
    
    # Asymmetry
    axes[1, 1].bar(operators, asym, color=colors, edgecolor='black', linewidth=1)
    axes[1, 1].set_ylabel('Asymmetry Index', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('Directional Balance', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylim(0, 1)
    
    # Intra-zonal
    axes[1, 2].bar(operators, intra, color=colors, edgecolor='black', linewidth=1)
    axes[1, 2].set_ylabel('Intra-zonal %', fontsize=10, fontweight='bold')
    axes[1, 2].set_title('Local vs Long-distance', fontsize=11, fontweight='bold')
    
    plt.suptitle('O-D Pattern Comparison by Operator\nTurin E-Scooter Network', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig06_combined_operators.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: fig06_combined_operators.png (reference)")
    
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all Exercise 2 publication visualization figures."""
    print("\n" + "="*70)
    print("EXERCISE 2: O-D MATRIX - PUBLICATION VISUALIZATION ENGINE")
    print("="*70)
    print("Publication-Quality Origin-Destination Visualizations")
    print("="*70)
    
    # Setup
    setup_matplotlib()
    
    # Ensure output directory exists
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {FIGURE_DIR}")
    
    # Load checkpoints
    checkpoints = load_checkpoints()
    
    # Check if we have minimum required data
    if checkpoints.get('matrix_allday') is None:
        print("\n" + "="*70)
        print("ERROR: No O-D matrix data available!")
        print("Please run 03_od_matrices.py first to generate checkpoints.")
        print("="*70)
        return
    
    # Generate figures
    results = {}
    
    # ========================================================================
    # A. HEATMAP ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("A. HEATMAP ANALYSIS")
    print("="*70)
    
    # Figure 1: Clustered Heatmap
    results['clustered_heatmap'] = plot_clustered_heatmap(
        checkpoints['matrix_allday'],
        checkpoints.get('clusters'),
        FIGURE_DIR / 'od_heatmap_clustered.png'
    )
    
    # Figure 2: Peak vs Off-Peak
    results['peak_offpeak'] = plot_peak_offpeak_comparison(
        checkpoints['matrix_peak'],
        checkpoints['matrix_offpeak'],
        checkpoints.get('chi_square'),
        FIGURE_DIR / 'od_peak_vs_offpeak.png'
    )
    
    # Figure 3: Asymmetry Heatmap
    results['asymmetry'] = plot_asymmetry_heatmap(
        checkpoints['matrix_allday'],
        FIGURE_DIR / 'od_asymmetry_heatmap.png'
    )
    
    # ========================================================================
    # B. FLOW MAPS
    # ========================================================================
    print("\n" + "="*70)
    print("B. FLOW MAPS")
    print("="*70)
    
    # Figure 4: Professional Flow Map
    results['flow_map'] = plot_professional_flow_map(
        checkpoints['matrix_allday'],
        checkpoints['zones'],
        FIGURE_DIR / 'flow_map_professional.png',
        top_n=30
    )
    
    # ========================================================================
    # C. STATISTICAL ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("C. STATISTICAL ANALYSIS")
    print("="*70)
    
    # Figure 5: Metrics Comparison
    results['metrics'] = plot_metrics_comparison(
        checkpoints['metrics'],
        FIGURE_DIR / 'od_metrics_comparison.png'
    )
    
    # Figure 6: Operator Comparison
    results['operator'] = plot_operator_od_comparison(
        checkpoints.get('operator_metrics'),
        FIGURE_DIR / 'operator_od_comparison.png'
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("PUBLICATION VISUALIZATION SUMMARY")
    print("="*70)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\nA. Heatmap Analysis:")
    print(f"  {'✓' if results.get('clustered_heatmap') else '✗'} od_heatmap_clustered.png")
    print(f"  {'✓' if results.get('peak_offpeak') else '✗'} od_peak_vs_offpeak.png")
    print(f"  {'✓' if results.get('asymmetry') else '✗'} od_asymmetry_heatmap.png")
    
    print("\nB. Flow Maps:")
    print(f"  {'✓' if results.get('flow_map') else '✗'} flow_map_professional.png")
    
    print("\nC. Statistical Analysis:")
    print(f"  {'✓' if results.get('metrics') else '✗'} od_metrics_comparison.png")
    print(f"  {'✓' if results.get('operator') else '✗'} operator_od_comparison.png")
    
    print(f"\n→ Generated {successful}/{total} figures successfully")
    print(f"→ Output location: {FIGURE_DIR}")
    
    # PUBLICATION-QUALITY Checklist
    print("\n" + "="*70)
    print("PUBLICATION-QUALITY CHECKLIST:")
    print("="*70)
    print("""
✓ Hierarchical clustering for zone grouping
✓ Diverging colormap for asymmetry visualization
✓ Chi-square test for temporal independence
✓ Gini coefficient for inequality measurement
✓ Shannon entropy for diversity quantification
✓ Professional cartographic elements
✓ 300 DPI resolution for print quality
    """)
    print("="*70)


if __name__ == '__main__':
    main()
