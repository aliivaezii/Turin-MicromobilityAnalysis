#!/usr/bin/env python3
"""
==============================================================================
DATA CLEANING WATERFALL CHART
==============================================================================

This script generates a waterfall chart showing the data cleaning pipeline
attrition analysis. The waterfall chart style (floating bar segments) is for visualizing sequential data loss.

Satisfies Exercise 1 requirement: "Analysis of Bad Data types"

Output: outputs/figures/exercise1/data_cleaning_waterfall.png

Author: Ali Vaezi
Date: December 2025
==============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

# Get project root (3 levels up from src/visualization/)
BASE_PATH = Path(__file__).parent.parent.parent
FIGURES_DIR = BASE_PATH / "outputs" / "figures" / "exercise0"
TABLES_DIR = BASE_PATH / "outputs" / "tables" / "exercise0"

# Ensure output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_quality_data():
    """
    Load data quality report from Exercise 1 temporal analysis.
    If not available, use documented values from analysis logs.
    """
    quality_path = TABLES_DIR / "data_quality_report.csv"
    
    if quality_path.exists():
        df = pd.read_csv(quality_path)
        print(f"Loaded quality report: {quality_path}")
        return df
    else:
        print("Quality report not found, using documented values from analysis logs")
        return None


def plot_waterfall():
    """
    Generate a high-quality waterfall chart showing data cleaning stages.
    """
    print("\n" + "="*70)
    print(" GENERATING DATA CLEANING WATERFALL CHART")
    print("="*70)
    
    # Data from Temporal Analysis logs
    stages = [
        'Raw Records',
        'Format Errors\n(Invalid dates/coords)',
        'Temporal Filter\n(Missing datetime)',
        'Spatial Bounds\n(Outside Turin)',
        'Duration Limits\n(<1 min or >6 hrs)',
        'Cleaned Dataset'
    ]
    
    values = [2605000, -25000, -30000, -22000, -8000, 2520000]
    
    # Calculate cumulative values for bar positioning
    cumulative = []
    running_total = 0
    for i, v in enumerate(values):
        if i == 0:
            cumulative.append(0)
            running_total = v
        elif i == len(values) - 1:
            cumulative.append(0)
        else:
            running_total += v
            cumulative.append(running_total)
    
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))  # Slightly increased height
    
    colors = [
        '#3498db',  # Raw - Blue
        '#e74c3c',  # Format Errors - Red
        '#c0392b',  # Temporal Filter - Dark Red
        '#d35400',  # Spatial Bounds - Orange-Red
        '#e67e22',  # Duration Limits - Orange
        '#27ae60',  # Cleaned - Green
    ]
    
    bar_width = 0.65
    x_positions = np.arange(len(stages))
    
    # Plot each bar
    for i, (stage, value, bottom, color) in enumerate(zip(stages, values, cumulative, colors)):
        bar_height = abs(value)
        
        if value < 0:
            bar_bottom = cumulative[i]
        else:
            bar_bottom = bottom
        
        ax.bar(
            x_positions[i], 
            bar_height, 
            bar_width, 
            bottom=bar_bottom,
            color=color, 
            edgecolor='black', 
            linewidth=0.8,
            alpha=0.9
        )
        
        # --- FIX 1: LABEL POSITIONING AND COLOR ---
        # For the start and end bars, place label on top
        if i == 0 or i == len(values) - 1:
            label_y = bar_height + 40000
            label_text = f'{abs(value):,}'
            va = 'bottom'
        else:
            # For removal bars: 
            # Because the bars are very thin (25k vs 2.6M), placing text inside is hard.
            # We place it slightly below the "floating" bar so it's readable.
            label_y = bar_bottom - 40000 
            label_text = f'-{abs(value):,}'
            va = 'top'
        
        ax.text(
            x_positions[i], 
            label_y, 
            label_text,
            ha='center', 
            va=va, 
            fontsize=10, 
            fontweight='bold',
            color='black'  # <--- CHANGED: Always black so it's visible on white bg
        )
    
    # Add connector lines
    for i in range(len(values) - 2):
        if i == 0:
            y_connect = values[0] + values[1]
        else:
            y_connect = cumulative[i+1]
        
        ax.hlines(
            y=y_connect,
            xmin=x_positions[i] + bar_width/2,
            xmax=x_positions[i+1] - bar_width/2,
            color='gray',
            linestyle='--',
            linewidth=1,
            alpha=0.5
        )
    
    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel('Number of Trip Records', fontweight='bold')
    ax.set_title(
        'Data Cleaning Pipeline: Attrition Analysis\n'
        'E-Scooter Trip Records (LIME + VOI + BIRD Combined)',
        fontsize=14, 
        fontweight='bold',
        pad=15
    )
    
    # Y-axis formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
    
    # --- FIX 2: INCREASE Y-LIMIT TO STOP BOX OVERLAP ---
    # Increased multiplier from 1.12 to 1.35 to create head-space for the boxes
    ax.set_ylim(0, max(values) * 1.35)
    
    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Summary Annotation
    retention_rate = values[-1] / values[0] * 100
    total_removed = values[0] - values[-1]
    
    summary_text = (
        f"Data Quality Summary:\n"
        f"• Initial: {values[0]:,} records\n"
        f"• Removed: {total_removed:,} records\n"
        f"• Final: {values[-1]:,} records\n"
        f"• Retention Rate: {retention_rate:.1f}%"
    )
    
    ax.text(
        0.98, 0.96, 
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='gray',
            alpha=0.9
        )
    )
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Initial Dataset'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Records Removed'),
        Patch(facecolor='#27ae60', edgecolor='black', label='Clean Dataset'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=1)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'data_cleaning_waterfall.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Waterfall chart saved: {output_path}")
    print(f"  - Retention rate: {retention_rate:.1f}%")
    print(f"  - Total removed: {total_removed:,} records")
    
    return output_path


def plot_bad_data_breakdown():
    """
    Generate a high-quality donut chart showing the breakdown of removed data.
    Uses high-contrast colors to make categories distinct.
    """
    print("\n" + "-"*50)
    print(" GENERATING BAD DATA TYPES BREAKDOWN")
    print("-"*50)
    
    # Categories and Values
    categories = [
        'Format Errors\n(Invalid dates/coords)',
        'Temporal Issues\n(Missing datetime)',
        'Spatial Outliers\n(Outside Turin)',
        'Duration Outliers\n(<1 min or >6 hrs)'
    ]
    
    # Hardcoded values from analysis
    removals = [25000, 30000, 22000, 8000]
    total_removed = sum(removals)
    
    # --- CHANGED: High Contrast Colors ---
    # We use distinct hues instead of a gradient so they are easy to tell apart.
    colors = [
        '#e74c3c',  # Red (Format Errors)
        '#2980b9',  # Blue (Temporal Issues)
        '#8e44ad',  # Purple (Spatial Outliers)
        '#16a085'   # Teal (Duration Outliers)
    ]
    
    # Create Figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot Pie Chart
    # pctdistance=0.85 moves numbers closer to the edge of the slice
    wedges, texts, autotexts = ax.pie(
        removals,
        labels=None,  # We will use a legend instead of labels
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_removed):,})',
        colors=colors,
        startangle=90,
        pctdistance=0.78, 
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2) # Thinner donut, clean lines
    )
    
    # Style the text inside the slices
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    # Add a Legend 
    ax.legend(
        wedges, 
        categories,
        title="Reason for Removal",
        loc="center left",
        bbox_to_anchor=(0.9, 0, 0.5, 1),
        fontsize=10,
        frameon=False 
    )
    
    # Center Text (Total Removed)
    ax.text(
        0, 0, 
        f'Total Removed\n{total_removed:,}', 
        ha='center', 
        va='center', 
        fontsize=16, 
        fontweight='bold',
        color='#333333'
    )
    
    # Title
    ax.set_title(
        'Bad Data Types: Removal Breakdown',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / 'bad_data_types_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Bad data breakdown saved: {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" DATA CLEANING VISUALIZATION PIPELINE")
    print("="*70)
    
    # Generate both visualizations
    waterfall_path = plot_waterfall()
    breakdown_path = plot_bad_data_breakdown()
    
    print("\n" + "="*70)
    print(" COMPLETE")
    print("="*70)
    print(f"Generated files:")
    print(f"  1. {waterfall_path}")
    print(f"  2. {breakdown_path}")
