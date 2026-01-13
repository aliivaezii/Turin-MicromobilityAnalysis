#!/usr/bin/env python3
"""
================================================================================
EXERCISE 3: PUBLIC TRANSPORT INTEGRATION ANALYSIS - Geospatial Mapping & Visualization
================================================================================

Geospatial visualization and cartographic analysis module.

This module generates high-quality maps with professional cartographic
elements including scale bars, north arrows, and statistical overlays.

Produces buffer sensitivity maps, integration metrics, and
PT accessibility visualizations.

FIGURES GENERATED (11 Total):
    1. competition_map.png - Zones where scooters replace buses
    2. integration_map.png - Zones where scooters feed buses  
    3. buffer_sensitivity_curve.png - Integration % vs Buffer distance
    4. temporal_comparison.png - Peak vs Off-Peak stability
    5. inefficient_routes_map.png - High-tortuosity LIME routes
    6. trip_density_hexbin.png - Trip density heatmap
    7. operator_comparison_bar.png - Integration by operator
    8. route_competition_bar.png - Top competing transit routes
    9. tortuosity_histogram.png - Distribution of route efficiency
    10. zone_scatter_integration.png - Integration vs Competition scatter
    11. summary_dashboard.png - Multi-panel overview dashboard

Output Directory: outputs/figures/exercise3/

Author: Ali Vaezi
Version: 1.0.0  
Last Updated: December 2025
================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import contextily as cx

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib style for professional output
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# =============================================================================
# CONFIGURATION
# =============================================================================

# Get project paths - visualization scripts are in src/visualization/
# SCRIPT_DIR is already the directory (src/visualization), so we go up 2 more levels
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # src/visualization
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # -> src -> project root

# Input/Output paths
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'reports', 'exercise3')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'figures', 'exercise3')
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')

# Zone shapefile path
ZONES_SHAPEFILE = os.path.join(DATA_RAW, 'zone_statistiche_geo', 'zone_statistiche_geo.shp')

# Create output directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# Color schemes
OPERATOR_COLORS = {
    'LIME': '#32CD32',   # Lime green
    'VOI': '#FF6B6B',    # Coral red
    'BIRD': '#4ECDC4'    # Teal
}

# Professional color maps
COMPETITION_CMAP = 'Reds'
INTEGRATION_CMAP = 'RdYlGn'
DENSITY_CMAP = 'YlOrRd'

# =============================================================================
# CHECKPOINT LOADING FUNCTIONS
# =============================================================================

def load_checkpoints():
    """
    Load all checkpoint files with graceful error handling.
    
    Returns:
        dict: Dictionary containing all loaded data, with None for missing files.
    """
    print("\n" + "="*80)
    print(" LOADING CHECKPOINTS")
    print("="*80)
    
    checkpoints = {}
    
    # Define checkpoint files to load
    files_to_load = {
        # Pickle files
        'buffer_sensitivity': ('checkpoint_buffer_sensitivity.pkl', 'pkl'),
        'temporal': ('checkpoint_temporal.pkl', 'pkl'),
        'full_results': ('checkpoint_full_results.pkl', 'pkl'),
        'route_competition': ('checkpoint_route_competition.pkl', 'pkl'),
        'validated_trips': ('checkpoint_validated_escooter_data.pkl', 'pkl'),
        'trip_overlaps': ('checkpoint_trip_overlaps.pkl', 'pkl'),
        
        # CSV files
        'pt_stops': ('checkpoint_turin_pt_stops.csv', 'csv'),
        'zone_overlaps': ('checkpoint_zone_overlaps.csv', 'csv'),
        'lime_tortuosity': ('lime_tortuosity_analysis.csv', 'csv'),
        'route_competition_csv': ('route_competition_analysis_50m.csv', 'csv'),
        
        # GeoJSON files
        'zones_with_metrics': ('checkpoint_zones_with_metrics.geojson', 'geojson'),
        'routes_gdf': ('checkpoint_routes_gdf.geojson', 'geojson'),
    }
    
    loaded_count = 0
    missing_count = 0
    
    for key, (filename, filetype) in files_to_load.items():
        filepath = os.path.join(REPORTS_DIR, filename)
        
        if os.path.exists(filepath):
            try:
                if filetype == 'pkl':
                    checkpoints[key] = pd.read_pickle(filepath)
                elif filetype == 'csv':
                    checkpoints[key] = pd.read_csv(filepath)
                elif filetype == 'geojson':
                    checkpoints[key] = gpd.read_file(filepath)
                
                # Get record count
                if hasattr(checkpoints[key], '__len__'):
                    count = len(checkpoints[key])
                else:
                    count = 'N/A'
                    
                print(f"   ✓ Loaded: {filename} ({count} records)")
                loaded_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Error loading {filename}: {e}")
                checkpoints[key] = None
                missing_count += 1
        else:
            print(f"   ✗ Missing: {filename}")
            checkpoints[key] = None
            missing_count += 1
    
    # Load base zones shapefile if zones_with_metrics is missing
    if checkpoints.get('zones_with_metrics') is None and os.path.exists(ZONES_SHAPEFILE):
        try:
            checkpoints['zones_base'] = gpd.read_file(ZONES_SHAPEFILE)
            print(f"   ✓ Loaded: zone_statistiche_geo.shp (fallback)")
        except Exception as e:
            print(f"   ⚠️ Could not load zones shapefile: {e}")
            checkpoints['zones_base'] = None
    
    print(f"\n   Summary: {loaded_count} loaded, {missing_count} missing/failed")
    
    return checkpoints


def prepare_geodataframes(checkpoints):
    """
    Prepare GeoDataFrames with proper CRS for plotting.
    Converts to EPSG:3857 (Web Mercator) for contextily compatibility.
    
    Returns:
        dict: Updated checkpoints with CRS-converted GeoDataFrames
    """
    print("\n   [CRS] Converting to Web Mercator (EPSG:3857)...")
    
    gdf_keys = ['zones_with_metrics', 'routes_gdf', 'zones_base']
    
    for key in gdf_keys:
        if checkpoints.get(key) is not None:
            try:
                if checkpoints[key].crs is None:
                    checkpoints[key] = checkpoints[key].set_crs("EPSG:4326")
                checkpoints[key] = checkpoints[key].to_crs("EPSG:3857")
                print(f"         ✓ {key} → EPSG:3857")
            except Exception as e:
                print(f"         ⚠️ {key} CRS conversion failed: {e}")
    
    return checkpoints


# =============================================================================
# FIGURE 0: STUDY AREA MAP (Statistical Zones Overview)
# =============================================================================

def plot_study_area_zones(checkpoints, output_dir):
    """
    Generate a professional map showing Turin's 94 statistical zones.
    
    This is for the Introduction/Study Area section of the report.
    Zones are colored by trip activity (total trips) for clear visual distinction.
    Enhanced VERSION: Clear choropleth coloring with trip density.
    """
    print("\n[0/11] Generating Study Area Zones Map...")
    
    zones_gdf = checkpoints.get('zones_with_metrics')
    if zones_gdf is None:
        zones_gdf = checkpoints.get('zones_base')
    
    if zones_gdf is None:
        print("       ⚠️ Skipped: No zone data available")
        return None
    
    # Create larger figure for better zone visibility
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))
    
    zones_gdf = zones_gdf.copy()
    
    # Ensure we're in Web Mercator for display
    if zones_gdf.crs != 'EPSG:3857':
        zones_gdf = zones_gdf.to_crs('EPSG:3857')
    
    # Get trip activity data for coloring
    # Check for total_trips, total_activity, or compute from origins + destinations
    if 'total_trips' in zones_gdf.columns:
        color_col = 'total_trips'
    elif 'total_activity' in zones_gdf.columns:
        color_col = 'total_activity'
    elif 'total_origins' in zones_gdf.columns and 'total_destinations' in zones_gdf.columns:
        zones_gdf['total_activity'] = zones_gdf['total_origins'] + zones_gdf['total_destinations']
        color_col = 'total_activity'
    else:
        # Fallback: use zone area for visual distinction
        zones_gdf['zone_area'] = zones_gdf.geometry.area / 1e6  # km²
        color_col = 'zone_area'
        print("       ℹ️ Using zone area for coloring (no trip data)")
    
    # Use log scale for better visual distribution
    import numpy as np
    zones_gdf['color_value'] = np.log1p(zones_gdf[color_col])
    
    # Professional colormap - YlOrRd for trip intensity (yellow=low, red=high)
    cmap = 'YlOrRd'
    
    # Plot zones with choropleth coloring
    zones_gdf.plot(
        column='color_value',
        cmap=cmap,
        ax=ax,
        edgecolor='#333333',  # Dark gray borders
        linewidth=1.2,
        alpha=0.85,
        legend=False  # We'll add a custom legend
    )
    
    # Add zone ID labels in small circles
    for idx, row in zones_gdf.iterrows():
        centroid = row.geometry.centroid
        zone_id = row.get('ZONASTAT', row.get('zona_stat', idx))
        
        # Calculate zone area for label sizing
        zone_area = row.geometry.area / 1e6  # km²
        fontsize = 6 if zone_area < 1.5 else 7
        
        ax.annotate(
            str(zone_id),
            xy=(centroid.x, centroid.y),
            fontsize=fontsize,
            ha='center',
            va='center',
            color='#1A1A1A',
            fontweight='bold',
            bbox=dict(
                boxstyle='circle,pad=0.15',
                facecolor='white',
                edgecolor='#333333',
                linewidth=0.8,
                alpha=0.9
            )
        )
    
    # Add subtle basemap
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.25)
    except Exception as e:
        print(f"       ⚠️ Basemap failed: {e}")
    
    # Zoom to data with small buffer
    bounds = zones_gdf.total_bounds
    buffer_x = (bounds[2] - bounds[0]) * 0.02
    buffer_y = (bounds[3] - bounds[1]) * 0.02
    ax.set_xlim(bounds[0] - buffer_x, bounds[2] + buffer_x)
    ax.set_ylim(bounds[1] - buffer_y, bounds[3] + buffer_y)
    
    # Professional title
    n_zones = len(zones_gdf)
    ax.set_title(f'Study Area: Municipality of Turin\n{n_zones} Statistical Zones with E-Scooter Activity', 
                 fontsize=16, fontweight='bold', pad=15, color='#1A1A1A')
    ax.set_axis_off()
    
    # Add north arrow
    ax.annotate('N', xy=(0.97, 0.95), xycoords='axes fraction',
                fontsize=14, fontweight='bold', ha='center', va='center',
                color='#1A1A1A')
    ax.annotate('↑', xy=(0.97, 0.92), xycoords='axes fraction',
                fontsize=16, ha='center', va='center', color='#1A1A1A')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, 
                                norm=plt.Normalize(vmin=zones_gdf['color_value'].min(), 
                                                   vmax=zones_gdf['color_value'].max()))
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, aspect=20, pad=0.02)
    cbar.set_label('Trip Activity (log scale)', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Add legend with zone info
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFFFB2', edgecolor='#333333', linewidth=1.2, 
              label='Low activity zones'),
        Patch(facecolor='#FD8D3C', edgecolor='#333333', linewidth=1.2, 
              label='Medium activity zones'),
        Patch(facecolor='#BD0026', edgecolor='#333333', linewidth=1.2, 
              label='High activity zones'),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
                    framealpha=0.95, edgecolor='#7F8C8D', fancybox=True,
                    title=f'Zone Activity (n={n_zones})', title_fontsize=11)
    
    # Add data source note
    ax.annotate(
        'Data: City of Turin Zone Statistiche • E-Scooter Trip Data 2024-2025',
        xy=(0.98, 0.02), xycoords='axes fraction',
        fontsize=8, color='#666666', style='italic',
        ha='right', va='bottom'
    )
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, 'study_area_zones.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    print(f"       ✓ Saved: study_area_zones.png")
    return filepath


# =============================================================================
# FIGURE 1: COMPETITION MAP (Zones where scooters replace buses)
# =============================================================================

def plot_competition_map(checkpoints, output_dir):
    """
    Generate competition map showing zones where e-scooters compete with transit.
    
    SIMPLIFIED VERSION:
    - Clean choropleth showing competition intensity per zone
    - NO transit route lines (too cluttered)
    - NO zone name labels (unreadable)
    - Just zone IDs for top 5 zones only
    - Clear color legend
    """
    print("\n[1/12] Generating Competition Map...")
    
    zones_gdf = checkpoints.get('zones_with_metrics')
    zone_overlaps = checkpoints.get('zone_overlaps')
    
    if zones_gdf is None:
        zones_gdf = checkpoints.get('zones_base')
        if zones_gdf is None:
            print("       ⚠️ Skipped: No zone data available")
            return None
    
    # Join overlap data if available
    if zone_overlaps is not None and 'competitor_trip_count' in zone_overlaps.columns:
        # Ensure ZONASTAT is string type for merge
        zones_gdf['ZONASTAT'] = zones_gdf['ZONASTAT'].astype(str)
        zone_overlaps['ZONASTAT'] = zone_overlaps['ZONASTAT'].astype(str)
        
        zones_gdf = zones_gdf.merge(
            zone_overlaps[['ZONASTAT', 'competitor_trip_count', 'avg_overlap_length_m']],
            on='ZONASTAT',
            how='left'
        )
        zones_gdf['competitor_trip_count'] = zones_gdf['competitor_trip_count'].fillna(0)
        plot_col = 'competitor_trip_count'
        title = 'E-Scooter ↔ Transit Competition Hotspots'
    else:
        # Fallback to total_trips if competition data not available
        if 'total_trips' in zones_gdf.columns:
            plot_col = 'total_trips'
            title = 'E-Scooter Trip Volume by Zone'
        else:
            print("       ⚠️ Skipped: No competition or trip data in zones")
            return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Plot ALL zones with light background first
    zones_gdf.plot(
        ax=ax,
        facecolor='#F5F5F5',
        edgecolor='#CCCCCC',
        linewidth=0.5,
        alpha=1.0
    )
    
    # Plot zones with competition data as choropleth
    zones_with_data = zones_gdf[zones_gdf[plot_col] > 0].copy()
    if len(zones_with_data) > 0:
        zones_with_data.plot(
            column=plot_col,
            cmap='YlOrRd',
            linewidth=0.8,
            edgecolor='#333333',
            alpha=0.9,
            ax=ax,
            legend=True,
            legend_kwds={
                'label': 'Competitor Trips per Zone',
                'orientation': 'horizontal',
                'shrink': 0.6,
                'pad': 0.02,
                'aspect': 30
            }
        )
    
    # Label ONLY top 5 zones with zone ID (not names!)
    top_5_zones = zones_gdf.nlargest(5, plot_col)
    for rank, (idx, row) in enumerate(top_5_zones.iterrows(), 1):
        centroid = row.geometry.centroid
        zone_id = row['ZONASTAT']
        count = int(row[plot_col])
        
        ax.annotate(
            f"#{rank}\nZ{zone_id}",
            xy=(centroid.x, centroid.y),
            fontsize=9,
            ha='center',
            va='center',
            color='white',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#B71C1C', 
                      edgecolor='white', linewidth=1.5, alpha=0.95)
        )
    
    # Add basemap (subtle)
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.3)
    except Exception as e:
        print(f"       ⚠️ Basemap failed: {e}")
    
    # Zoom to data extent
    bounds = zones_gdf.total_bounds
    buffer_x = (bounds[2] - bounds[0]) * 0.03
    buffer_y = (bounds[3] - bounds[1]) * 0.03
    ax.set_xlim(bounds[0] - buffer_x, bounds[2] + buffer_x)
    ax.set_ylim(bounds[1] - buffer_y, bounds[3] + buffer_y)
    
    # Styling
    ax.set_title(f'{title}\nZones with Direct PT Route Overlap', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_axis_off()
    
    # Simple legend - just color interpretation
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFEB3B', edgecolor='gray', label='Low Competition'),
        Patch(facecolor='#FF9800', edgecolor='gray', label='Medium Competition'),
        Patch(facecolor='#D32F2F', edgecolor='gray', label='High Competition'),
        Patch(facecolor='#F5F5F5', edgecolor='#CCCCCC', label='No Competition Data'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, 
              framealpha=0.95, edgecolor='gray')
    
    # Add summary stats
    total_competitor = zones_gdf[plot_col].sum()
    n_zones_with_data = (zones_gdf[plot_col] > 0).sum()
    stats_text = f"Total Competitor Trips: {int(total_competitor):,}\n"
    stats_text += f"Zones with PT Overlap: {n_zones_with_data}/{len(zones_gdf)}"
    ax.annotate(stats_text, xy=(0.98, 0.98), xycoords='axes fraction',
                fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                          edgecolor='gray', alpha=0.95))
    
    # Save
    filepath = os.path.join(output_dir, 'competition_map.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    
    print(f"       ✓ Saved: competition_map.png")
    return filepath


# =============================================================================
# FIGURE 2: INTEGRATION MAP (Zones where scooters feed buses)
# =============================================================================

def plot_integration_map(checkpoints, output_dir):
    """
    Generate integration map showing zones where e-scooters feed transit.
    
    Uses integration_pct or integration_score from zones_with_metrics.
    """
    print("\n[2/11] Generating Integration Map...")
    
    zones_gdf = checkpoints.get('zones_with_metrics')
    
    if zones_gdf is None:
        print("       ⚠️ Skipped: No zone metrics data available")
        return None
    
    # Determine which column to plot
    if 'integration_pct' in zones_gdf.columns:
        plot_col = 'integration_pct'
        label = 'Integration %'
    elif 'integration_score' in zones_gdf.columns:
        plot_col = 'integration_score'
        label = 'Integration Score'
    else:
        print("       ⚠️ Skipped: No integration metric in zones data")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot zones (choropleth) with diverging colormap
    zones_gdf.plot(
        column=plot_col,
        cmap=INTEGRATION_CMAP,
        linewidth=0.3,
        edgecolor='gray',
        alpha=0.85,
        ax=ax,
        legend=True,
        legend_kwds={
            'label': label,
            'orientation': 'horizontal',
            'shrink': 0.6,
            'pad': 0.05
        }
    )
    
    # Add basemap
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.5)
    except Exception as e:
        print(f"       ⚠️ Basemap failed: {e}")
    
    # Styling
    ax.set_title('E-Scooter Integration with Public Transport\n(% of Trips Near PT Stops)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    
    # Save
    filepath = os.path.join(output_dir, 'integration_map.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: integration_map.png")
    return filepath


# =============================================================================
# FIGURE 3: BUFFER SENSITIVITY CURVE
# =============================================================================

def plot_sensitivity_curve(checkpoints, output_dir):
    """
    Generate sensitivity curve showing Integration % vs Buffer distance.
    
    Scientific validation of buffer selection methodology.
    """
    print("\n[3/11] Generating Buffer Sensitivity Curve...")
    
    df = checkpoints.get('buffer_sensitivity')
    
    if df is None or len(df) == 0:
        print("       ⚠️ Skipped: No buffer sensitivity data available")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot line for each operator
    for operator in df['operator'].unique():
        df_op = df[df['operator'] == operator].sort_values('buffer_m')
        
        color = OPERATOR_COLORS.get(operator, '#666666')
        
        ax.plot(
            df_op['buffer_m'], 
            df_op['integration_index'],
            marker='o',
            markersize=8,
            linewidth=2.5,
            color=color,
            label=operator
        )
    
    # Add reference lines
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=75, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Styling
    ax.set_xlabel('Buffer Distance (meters)', fontsize=11)
    ax.set_ylabel('Integration Index (%)', fontsize=11)
    ax.set_title('Buffer Sensitivity Analysis\nIntegration % by Catchment Distance', 
                 fontsize=13, fontweight='bold')
    
    ax.set_xlim(0, max(df['buffer_m']) + 20)
    ax.set_ylim(0, 100)
    ax.set_xticks(df['buffer_m'].unique())
    
    ax.legend(title='Operator', loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate(
        'Higher integration at smaller buffers\nindicates strategic stop placement',
        xy=(0.02, 0.98), xycoords='axes fraction',
        fontsize=9, alpha=0.7,
        verticalalignment='top'
    )
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, 'buffer_sensitivity_curve.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: buffer_sensitivity_curve.png")
    return filepath


# =============================================================================
# FIGURE 4: TEMPORAL COMPARISON (Peak vs Off-Peak)
# =============================================================================

def plot_temporal_comparison(checkpoints, output_dir):
    """
    Generate temporal comparison showing Peak vs Off-Peak integration.
    """
    print("\n[4/11] Generating Temporal Comparison...")
    
    df = checkpoints.get('temporal')
    
    if df is None or len(df) == 0:
        print("       ⚠️ Skipped: No temporal data available")
        return None
    
    # Use 200m buffer as reference (or largest available)
    if 'buffer_m' in df.columns:
        reference_buffer = df['buffer_m'].max()
        df_ref = df[df['buffer_m'] == reference_buffer]
    else:
        df_ref = df
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Prepare data for grouped bar chart
    operators = df_ref['operator'].unique()
    x = np.arange(len(operators))
    width = 0.35
    
    peak_values = []
    offpeak_values = []
    
    for op in operators:
        df_op = df_ref[df_ref['operator'] == op]
        peak = df_op[df_op['time_period'] == 'Peak']['integration_index'].values
        offpeak = df_op[df_op['time_period'] == 'Off-Peak']['integration_index'].values
        
        peak_values.append(peak[0] if len(peak) > 0 else 0)
        offpeak_values.append(offpeak[0] if len(offpeak) > 0 else 0)
    
    # Plot bars
    bars1 = ax.bar(x - width/2, peak_values, width, label='Peak Hours', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, offpeak_values, width, label='Off-Peak Hours', color='#4ECDC4', alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars1, peak_values):
        ax.annotate(f'{val:.1f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    for bar, val in zip(bars2, offpeak_values):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    # Styling
    ax.set_xlabel('Operator', fontsize=11)
    ax.set_ylabel('Integration Index (%)', fontsize=11)
    ax.set_title('Temporal Stability: Peak vs Off-Peak Integration\n(200m Buffer)', 
                 fontsize=13, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(operators, fontsize=10)
    ax.set_ylim(0, 100)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, 'temporal_comparison.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: temporal_comparison.png")
    return filepath


# =============================================================================
# FIGURE 5: INEFFICIENT ROUTES MAP (High Tortuosity LIME Routes)
# =============================================================================

def plot_inefficient_routes(checkpoints, output_dir):
    """
    Generate map showing high-tortuosity (inefficient) LIME routes.
    
    Filters routes with tortuosity > 1.5 and plots them in red.
    """
    print("\n[5/11] Generating Inefficient Routes Map...")
    
    lime_df = checkpoints.get('lime_tortuosity')
    zones_gdf = checkpoints.get('zones_with_metrics')
    
    if lime_df is None or 'tortuosity_index' not in lime_df.columns:
        print("       ⚠️ Skipped: No tortuosity data available")
        return None
    
    # Filter high-tortuosity routes
    TORTUOSITY_THRESHOLD = 1.5
    inefficient = lime_df[lime_df['tortuosity_index'] > TORTUOSITY_THRESHOLD].copy()
    
    if len(inefficient) == 0:
        print("       ⚠️ Skipped: No inefficient routes found")
        return None
    
    print(f"       Found {len(inefficient):,} inefficient routes (tortuosity > {TORTUOSITY_THRESHOLD})")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot zone boundaries if available
    if zones_gdf is not None:
        zones_gdf.plot(
            ax=ax,
            facecolor='lightgray',
            edgecolor='white',
            linewidth=0.5,
            alpha=0.3
        )
    
    # Create GeoDataFrame of inefficient routes (origin points)
    if 'start_lat' in inefficient.columns and 'start_lon' in inefficient.columns:
        gdf_inefficient = gpd.GeoDataFrame(
            inefficient,
            geometry=gpd.points_from_xy(inefficient['start_lon'], inefficient['start_lat']),
            crs="EPSG:4326"
        ).to_crs("EPSG:3857")
        
        # Color by tortuosity intensity
        gdf_inefficient.plot(
            ax=ax,
            column='tortuosity_index',
            cmap='Reds',
            markersize=5,
            alpha=0.6,
            legend=True,
            legend_kwds={
                'label': 'Tortuosity Index',
                'orientation': 'horizontal',
                'shrink': 0.5,
                'pad': 0.05
            }
        )
    
    # Add basemap
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.5)
    except Exception as e:
        print(f"       ⚠️ Basemap failed: {e}")
    
    # Styling
    ax.set_title(f'Inefficient E-Scooter Routes (Tortuosity > {TORTUOSITY_THRESHOLD})\n'
                 f'{len(inefficient):,} Routes with Significant Detours', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    
    # Save
    filepath = os.path.join(output_dir, 'inefficient_routes_map.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: inefficient_routes_map.png")
    return filepath


# =============================================================================
# FIGURE 6: TRIP DENSITY HEXBIN MAP
# =============================================================================

def plot_density_map(checkpoints, output_dir):
    """
    Generate hexbin density map of all e-scooter trips.
    """
    print("\n[6/11] Generating Trip Density Map...")
    
    trips_df = checkpoints.get('validated_trips')
    zones_gdf = checkpoints.get('zones_with_metrics')
    
    if trips_df is None or len(trips_df) == 0:
        print("       ⚠️ Skipped: No trip data available")
        return None
    
    # Sample if too large
    SAMPLE_SIZE = 100000
    if len(trips_df) > SAMPLE_SIZE:
        print(f"       Sampling {SAMPLE_SIZE:,} trips from {len(trips_df):,}")
        trips_sample = trips_df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        trips_sample = trips_df
    
    # Create GeoDataFrame and convert to Web Mercator
    gdf_trips = gpd.GeoDataFrame(
        trips_sample,
        geometry=gpd.points_from_xy(trips_sample['start_lon'], trips_sample['start_lat']),
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")
    
    # Extract coordinates
    x = gdf_trips.geometry.x
    y = gdf_trips.geometry.y
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Plot zone boundaries if available
    if zones_gdf is not None:
        zones_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor='gray',
            linewidth=0.3,
            alpha=0.5
        )
    
    # Create hexbin plot
    hb = ax.hexbin(
        x, y,
        gridsize=50,
        cmap=DENSITY_CMAP,
        mincnt=1,
        alpha=0.7,
        linewidths=0.1
    )
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label('Trip Count', fontsize=10)
    
    # Add basemap
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, alpha=0.4)
    except Exception as e:
        print(f"       ⚠️ Basemap failed: {e}")
    
    # Styling
    ax.set_title(f'E-Scooter Trip Density\n({len(trips_sample):,} trip origins)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_axis_off()
    
    # Save
    filepath = os.path.join(output_dir, 'trip_density_hexbin.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: trip_density_hexbin.png")
    return filepath


# =============================================================================
# FIGURE 7: OPERATOR COMPARISON BAR CHART
# =============================================================================

def plot_operator_comparison(checkpoints, output_dir):
    """
    Generate bar chart comparing integration metrics by operator.
    """
    print("\n[7/11] Generating Operator Comparison...")
    
    df = checkpoints.get('buffer_sensitivity')
    
    if df is None or len(df) == 0:
        print("       ⚠️ Skipped: No buffer sensitivity data available")
        return None
    
    # Use 200m buffer as reference
    if 'buffer_m' in df.columns:
        df_200 = df[df['buffer_m'] == 200]
    else:
        df_200 = df
    
    if len(df_200) == 0:
        df_200 = df[df['buffer_m'] == df['buffer_m'].max()]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    operators = df_200['operator'].unique()
    colors = [OPERATOR_COLORS.get(op, '#666666') for op in operators]
    
    # Plot 1: Integration Index
    ax1 = axes[0]
    integration_values = [df_200[df_200['operator'] == op]['integration_index'].values[0] for op in operators]
    bars1 = ax1.bar(operators, integration_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars1, integration_values):
        ax1.annotate(f'{val:.1f}%', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Integration Index (%)', fontsize=11)
    ax1.set_title('PT Integration by Operator', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Feeder Percentage
    ax2 = axes[1]
    feeder_values = [df_200[df_200['operator'] == op]['feeder_pct'].values[0] for op in operators]
    bars2 = ax2.bar(operators, feeder_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars2, feeder_values):
        ax2.annotate(f'{val:.1f}%',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Feeder Percentage (%)', fontsize=11)
    ax2.set_title('Feeder Trips by Operator', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Operator Comparison (200m Buffer)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, 'operator_comparison_bar.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: operator_comparison_bar.png")
    return filepath


# =============================================================================
# FIGURE 8: ROUTE COMPETITION BAR CHART
# =============================================================================

def plot_route_competition(checkpoints, output_dir):
    """
    Generate bar chart showing top competing transit routes.
    """
    print("\n[8/11] Generating Route Competition Chart...")
    
    df = checkpoints.get('route_competition')
    
    if df is None:
        df = checkpoints.get('route_competition_csv')
    
    if df is None or len(df) == 0:
        print("       ⚠️ Skipped: No route competition data available")
        return None
    
    # Get top 10 routes
    if 'Overlap_Count' in df.columns:
        df_top = df.nlargest(10, 'Overlap_Count')
        count_col = 'Overlap_Count'
    elif 'overlap_count' in df.columns:
        df_top = df.nlargest(10, 'overlap_count')
        count_col = 'overlap_count'
    else:
        print("       ⚠️ Skipped: No overlap count column found")
        return None
    
    # Get route names
    if 'Route_Name' in df_top.columns:
        route_names = df_top['Route_Name'].astype(str).tolist()
    elif 'route_name' in df_top.columns:
        route_names = df_top['route_name'].astype(str).tolist()
    else:
        route_names = [f'Route {i+1}' for i in range(len(df_top))]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Horizontal bar chart
    y_pos = np.arange(len(route_names))
    counts = df_top[count_col].values
    
    # Color gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(counts)))[::-1]
    
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, counts):
        ax.annotate(f'{int(val):,}',
                    xy=(val, bar.get_y() + bar.get_height()/2),
                    ha='left', va='center', fontsize=9,
                    xytext=(5, 0), textcoords='offset points')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(route_names, fontsize=10)
    ax.invert_yaxis()
    
    ax.set_xlabel('Overlapping E-Scooter Trips', fontsize=11)
    ax.set_title('Top 10 Transit Routes with E-Scooter Competition\n(50m Corridor Buffer)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, 'route_competition_bar.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: route_competition_bar.png")
    return filepath


# =============================================================================
# FIGURE 9: TORTUOSITY HISTOGRAM
# =============================================================================

def plot_tortuosity_histogram(checkpoints, output_dir):
    """
    Generate histogram showing distribution of route efficiency (tortuosity).
    """
    print("\n[9/11] Generating Tortuosity Histogram...")
    
    df = checkpoints.get('lime_tortuosity')
    
    if df is None or 'tortuosity_index' not in df.columns:
        print("       ⚠️ Skipped: No tortuosity data available")
        return None
    
    # Filter valid values
    tortuosity = df['tortuosity_index'].dropna()
    tortuosity = tortuosity[tortuosity <= 5]  # Cap at 5 for visualization
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(
        tortuosity, 
        bins=50, 
        color='#32CD32', 
        alpha=0.7, 
        edgecolor='black',
        linewidth=0.5
    )
    
    # Color bins by efficiency category
    for i, (patch, binval) in enumerate(zip(patches, bins[:-1])):
        if binval <= 1.2:
            patch.set_facecolor('#2ECC71')  # Green - Direct
        elif binval <= 1.5:
            patch.set_facecolor('#F39C12')  # Orange - Moderate
        elif binval <= 2.0:
            patch.set_facecolor('#E74C3C')  # Red - Detoured
        else:
            patch.set_facecolor('#8E44AD')  # Purple - Highly Inefficient
    
    # Add vertical lines for thresholds
    ax.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Perfect (1.0)')
    ax.axvline(x=1.2, color='orange', linestyle='--', linewidth=1.5, label='Moderate (1.2)')
    ax.axvline(x=1.5, color='red', linestyle='--', linewidth=1.5, label='Detoured (1.5)')
    
    # Add statistics annotation - positioned in upper LEFT to avoid legend overlap
    stats_text = (f"Mean: {tortuosity.mean():.2f}\n"
                  f"Median: {tortuosity.median():.2f}\n"
                  f"Std: {tortuosity.std():.2f}")
    ax.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=10, ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.9))
    
    ax.set_xlabel('Tortuosity Index (Actual / Euclidean Distance)', fontsize=11)
    ax.set_ylabel('Number of Trips', fontsize=11)
    ax.set_title('LIME Route Efficiency Distribution\n(Tortuosity Index)', 
                 fontsize=13, fontweight='bold')
    
    # Legend positioned in upper right, properly sized box
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, 
              edgecolor='gray', fancybox=True)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, 'tortuosity_histogram.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: tortuosity_histogram.png")
    return filepath


# =============================================================================
# FIGURE 10: ZONE SCATTER (Integration vs Competition)
# =============================================================================

def plot_zone_scatter(checkpoints, output_dir):
    """
    Generate scatter plot of Integration vs Competition by zone.
    """
    print("\n[10/11] Generating Zone Scatter Plot...")
    
    zones_gdf = checkpoints.get('zones_with_metrics')
    zone_overlaps = checkpoints.get('zone_overlaps')
    
    if zones_gdf is None:
        print("       ⚠️ Skipped: No zone metrics data available")
        return None
    
    # Merge overlap data if available
    if zone_overlaps is not None and 'competitor_trip_count' in zone_overlaps.columns:
        zones_gdf['ZONASTAT'] = zones_gdf['ZONASTAT'].astype(str)
        zone_overlaps['ZONASTAT'] = zone_overlaps['ZONASTAT'].astype(str)
        
        df_merged = zones_gdf.merge(
            zone_overlaps[['ZONASTAT', 'competitor_trip_count']],
            on='ZONASTAT',
            how='left'
        )
        df_merged['competitor_trip_count'] = df_merged['competitor_trip_count'].fillna(0)
        competition_col = 'competitor_trip_count'
    elif 'total_trips' in zones_gdf.columns:
        df_merged = zones_gdf.copy()
        competition_col = 'total_trips'
    else:
        print("       ⚠️ Skipped: No competition data available")
        return None
    
    if 'integration_pct' not in df_merged.columns:
        print("       ⚠️ Skipped: No integration data available")
        return None
    
    # Filter zones with data
    df_plot = df_merged[(df_merged['integration_pct'] > 0) | (df_merged[competition_col] > 0)].copy()
    
    if len(df_plot) == 0:
        print("       ⚠️ Skipped: No zones with data")
        return None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Scatter plot with size based on total trips
    if 'total_trips' in df_plot.columns:
        sizes = df_plot['total_trips'] / df_plot['total_trips'].max() * 200 + 20
    else:
        sizes = 50
    
    scatter = ax.scatter(
        df_plot['integration_pct'],
        df_plot[competition_col],
        s=sizes,
        c=df_plot['integration_pct'],
        cmap='RdYlGn',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Integration %', fontsize=10)
    
    # Add quadrant labels
    x_mid = df_plot['integration_pct'].median()
    y_mid = df_plot[competition_col].median()
    
    ax.axhline(y=y_mid, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=x_mid, color='gray', linestyle='--', alpha=0.5)
    
    # Quadrant annotations
    ax.annotate('High Integration\nLow Competition\n(FEEDER ZONES)', 
                xy=(0.95, 0.05), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=8, alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.annotate('High Integration\nHigh Competition\n(MIXED USE)', 
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=8, alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    ax.annotate('Low Integration\nHigh Competition\n(SUBSTITUTION)', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=8, alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    ax.set_xlabel('Integration Index (%)', fontsize=11)
    ax.set_ylabel('Competitor Trips', fontsize=11)
    ax.set_title('Zone Classification: Integration vs Competition\n(Bubble size = Total trips)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, 'zone_scatter_integration.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: zone_scatter_integration.png")
    return filepath


# =============================================================================
# FIGURE 11: SUMMARY DASHBOARD (Multi-Panel Overview)
# =============================================================================

def plot_summary_dashboard(checkpoints, output_dir):
    """
    Generate multi-panel summary dashboard.
    """
    print("\n[11/11] Generating Summary Dashboard...")
    
    # Load required data
    df_buffer = checkpoints.get('buffer_sensitivity')
    df_temporal = checkpoints.get('temporal')
    df_tortuosity = checkpoints.get('lime_tortuosity')
    zones_gdf = checkpoints.get('zones_with_metrics')
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Sensitivity Curve (top-left)
    ax1 = axes[0, 0]
    if df_buffer is not None:
        for operator in df_buffer['operator'].unique():
            df_op = df_buffer[df_buffer['operator'] == operator].sort_values('buffer_m')
            color = OPERATOR_COLORS.get(operator, '#666666')
            ax1.plot(df_op['buffer_m'], df_op['integration_index'],
                     marker='o', linewidth=2, color=color, label=operator)
        ax1.set_xlabel('Buffer (m)', fontsize=10)
        ax1.set_ylabel('Integration %', fontsize=10)
        ax1.set_title('A. Buffer Sensitivity', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
        ax1.set_title('A. Buffer Sensitivity', fontsize=11, fontweight='bold')
    
    # Panel 2: Temporal Comparison (top-right)
    ax2 = axes[0, 1]
    if df_temporal is not None:
        df_200 = df_temporal[df_temporal['buffer_m'] == df_temporal['buffer_m'].max()]
        operators = df_200['operator'].unique()
        x = np.arange(len(operators))
        width = 0.35
        
        peak_vals = [df_200[(df_200['operator'] == op) & (df_200['time_period'] == 'Peak')]['integration_index'].values[0] 
                     if len(df_200[(df_200['operator'] == op) & (df_200['time_period'] == 'Peak')]) > 0 else 0 
                     for op in operators]
        offpeak_vals = [df_200[(df_200['operator'] == op) & (df_200['time_period'] == 'Off-Peak')]['integration_index'].values[0]
                        if len(df_200[(df_200['operator'] == op) & (df_200['time_period'] == 'Off-Peak')]) > 0 else 0
                        for op in operators]
        
        ax2.bar(x - width/2, peak_vals, width, label='Peak', color='#FF6B6B', alpha=0.8)
        ax2.bar(x + width/2, offpeak_vals, width, label='Off-Peak', color='#4ECDC4', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(operators, fontsize=9)
        ax2.set_ylabel('Integration %', fontsize=10)
        ax2.set_title('B. Peak vs Off-Peak', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
        ax2.set_title('B. Peak vs Off-Peak', fontsize=11, fontweight='bold')
    
    # Panel 3: Tortuosity Distribution (bottom-left)
    ax3 = axes[1, 0]
    if df_tortuosity is not None and 'tortuosity_index' in df_tortuosity.columns:
        tort = df_tortuosity['tortuosity_index'].dropna()
        tort = tort[tort <= 5]
        ax3.hist(tort, bins=30, color='#32CD32', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.axvline(x=1.0, color='green', linestyle='--', linewidth=1.5)
        ax3.axvline(x=1.5, color='red', linestyle='--', linewidth=1.5)
        ax3.set_xlabel('Tortuosity Index', fontsize=10)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_title('C. Route Efficiency (LIME)', fontsize=11, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
        ax3.set_title('C. Route Efficiency', fontsize=11, fontweight='bold')
    
    # Panel 4: Key Metrics Summary (bottom-right)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    metrics_text = "KEY FINDINGS\n" + "="*40 + "\n\n"
    
    if df_buffer is not None:
        avg_integration = df_buffer[df_buffer['buffer_m'] == 200]['integration_index'].mean()
        metrics_text += f"• Avg Integration (200m): {avg_integration:.1f}%\n"
    
    if df_tortuosity is not None:
        mean_tort = df_tortuosity['tortuosity_index'].mean()
        high_tort_pct = (df_tortuosity['tortuosity_index'] > 1.5).mean() * 100
        metrics_text += f"• Mean Tortuosity: {mean_tort:.2f}\n"
        metrics_text += f"• High Tortuosity Routes: {high_tort_pct:.1f}%\n"
    
    if zones_gdf is not None and 'total_trips' in zones_gdf.columns:
        total_trips = zones_gdf['total_trips'].sum()
        metrics_text += f"• Total Trips Analyzed: {int(total_trips):,}\n"
    
    metrics_text += "\n" + "="*40 + "\n"
    metrics_text += "INTERPRETATION\n\n"
    metrics_text += "• High Integration = Feeder to PT\n"
    metrics_text += "• Low Tortuosity = Efficient Routes\n"
    metrics_text += "• Peak ≈ Off-Peak = Stable Usage\n"
    
    ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2))
    ax4.set_title('D. Summary Statistics', fontsize=11, fontweight='bold')
    
    plt.suptitle('E-Scooter & Public Transport Integration Analysis\nTurin, Italy (2024-2025)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    filepath = os.path.join(output_dir, 'summary_dashboard.png')
    plt.savefig(filepath, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"       ✓ Saved: summary_dashboard.png")
    return filepath


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution function - generates all 11 professional figures.
    """
    print("="*80)
    print(" E-SCOOTER VISUALIZATION SUITE")
    print(" Exercise 3: Professional Figure Generation")
    print("="*80)
    print(f"\n Output Directory: {FIGURES_DIR}")
    
    # Load all checkpoints
    checkpoints = load_checkpoints()
    
    # Prepare GeoDataFrames with proper CRS
    checkpoints = prepare_geodataframes(checkpoints)
    
    # Track generated figures
    generated_figures = []
    failed_figures = []
    
    print("\n" + "="*80)
    print(" GENERATING FIGURES")
    print("="*80)
    
    # Generate all 12 figures (including study area)
    figure_functions = [
        ('Study Area Zones', plot_study_area_zones),
        ('Competition Map', plot_competition_map),
        ('Integration Map', plot_integration_map),
        ('Buffer Sensitivity Curve', plot_sensitivity_curve),
        ('Temporal Comparison', plot_temporal_comparison),
        ('Inefficient Routes Map', plot_inefficient_routes),
        ('Trip Density Hexbin', plot_density_map),
        ('Operator Comparison', plot_operator_comparison),
        ('Route Competition Bar', plot_route_competition),
        ('Tortuosity Histogram', plot_tortuosity_histogram),
        ('Zone Scatter Plot', plot_zone_scatter),
        ('Summary Dashboard', plot_summary_dashboard),
    ]
    
    for name, func in figure_functions:
        try:
            result = func(checkpoints, FIGURES_DIR)
            if result is not None:
                generated_figures.append(result)
            else:
                failed_figures.append(name)
        except Exception as e:
            print(f"       ❌ Error in {name}: {e}")
            failed_figures.append(name)
    
    # Print summary
    print("\n" + "="*80)
    print(" GENERATION COMPLETE")
    print("="*80)
    
    print(f"\n ✓ Successfully generated: {len(generated_figures)} figures")
    for fig in generated_figures:
        print(f"   • {os.path.basename(fig)}")
    
    if failed_figures:
        print(f"\n ⚠️ Skipped/Failed: {len(failed_figures)} figures")
        for fig in failed_figures:
            print(f"   • {fig}")
    
    print(f"\n Output Location: {FIGURES_DIR}")
    print("\n" + "="*80)
    print(" VISUALIZATION SUITE COMPLETE")
    print("="*80)
    
    return generated_figures


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()
