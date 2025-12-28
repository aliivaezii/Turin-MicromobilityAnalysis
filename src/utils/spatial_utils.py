#!/usr/bin/env python3
"""
================================================================================
SPATIAL UTILITIES - Standardized CRS and Zone Handling
================================================================================

This module provides standardized functions for:
- Loading zone shapefiles with consistent CRS
- Converting between coordinate reference systems
- Ensuring zone consistency across all analysis modules

COORDINATE REFERENCE SYSTEMS (CRS):
-----------------------------------
- EPSG:3003 - Monte Mario / Italy zone 1 (Original shapefile CRS)
- EPSG:4326 - WGS84 (GPS coordinates, lat/lon)
- EPSG:32632 - UTM zone 32N (Metric, for distance calculations)
- EPSG:3857 - Web Mercator (For map visualization with basemaps)

ZONE COUNT:
-----------
Turin has 94 statistical zones ("Zone Statistiche")

Author: Ali Vaezi
Version: 1.0.0
Last Updated: December 2025
================================================================================
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

# ============================================================================
# STANDARD CRS DEFINITIONS
# ============================================================================

CRS_ORIGINAL = "EPSG:3003"    # Original shapefile CRS (Monte Mario / Italy zone 1)
CRS_WGS84 = "EPSG:4326"       # GPS coordinates (lat/lon)
CRS_UTM32N = "EPSG:32632"     # Metric CRS for Turin (for distance/area calculations)
CRS_WEB_MERCATOR = "EPSG:3857"  # Web Mercator (for map display with basemaps)

# Standard zone count for Turin
EXPECTED_ZONE_COUNT = 94

# Zone column names
ZONE_ID_COL = 'ZONASTAT'
ZONE_NAME_COL = 'DENOM'

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    # This file is in src/utils/, so go up 2 levels
    return Path(__file__).parent.parent.parent


def get_zones_path() -> Path:
    """Get the standard path to the zones shapefile."""
    return get_project_root() / "data" / "raw" / "zone_statistiche_geo" / "zone_statistiche_geo.shp"


# ============================================================================
# ZONE LOADING FUNCTIONS
# ============================================================================

def load_zones(crs: str = CRS_WGS84, validate: bool = True) -> gpd.GeoDataFrame:
    """
    Load Turin statistical zones with standardized CRS.
    
    Parameters:
    -----------
    crs : str
        Target CRS for the zones. Default is WGS84 (EPSG:4326) for GPS data.
        Options: CRS_WGS84, CRS_UTM32N, CRS_WEB_MERCATOR
    validate : bool
        If True, validate that zone count matches expected (94 zones).
    
    Returns:
    --------
    gpd.GeoDataFrame
        Zones GeoDataFrame with standardized CRS and columns.
    
    Raises:
    -------
    FileNotFoundError
        If the zones shapefile is not found.
    ValueError
        If validate=True and zone count doesn't match expected.
    """
    zones_path = get_zones_path()
    
    if not zones_path.exists():
        raise FileNotFoundError(
            f"Zones shapefile not found at: {zones_path}\n"
            f"Please ensure the file exists in: data/raw/zone_statistiche_geo/"
        )
    
    # Load zones
    zones_gdf = gpd.read_file(zones_path)
    
    # Convert to target CRS
    if zones_gdf.crs is None:
        warnings.warn("Zones shapefile has no CRS defined. Assuming EPSG:3003.")
        zones_gdf = zones_gdf.set_crs(CRS_ORIGINAL)
    
    if str(zones_gdf.crs) != crs:
        zones_gdf = zones_gdf.to_crs(crs)
    
    # Validate zone count
    if validate and len(zones_gdf) != EXPECTED_ZONE_COUNT:
        warnings.warn(
            f"Zone count mismatch: expected {EXPECTED_ZONE_COUNT}, got {len(zones_gdf)}. "
            f"This may cause inconsistencies across analyses."
        )
    
    # Ensure standard columns exist
    if ZONE_ID_COL not in zones_gdf.columns:
        raise ValueError(f"Zone ID column '{ZONE_ID_COL}' not found in shapefile.")
    
    return zones_gdf


def load_zones_with_area(crs: str = CRS_WGS84) -> gpd.GeoDataFrame:
    """
    Load zones with pre-calculated area in square kilometers.
    
    The area is calculated in UTM 32N (EPSG:32632) for accuracy,
    then the GeoDataFrame is returned in the requested CRS.
    
    Parameters:
    -----------
    crs : str
        Target CRS for the zones.
    
    Returns:
    --------
    gpd.GeoDataFrame
        Zones with 'area_sqkm' column added.
    """
    zones_gdf = load_zones(crs=CRS_UTM32N)
    
    # Calculate area in UTM (metric)
    zones_gdf['area_sqkm'] = zones_gdf.geometry.area / 1e6
    
    # Convert to target CRS if different
    if crs != CRS_UTM32N:
        zones_gdf = zones_gdf.to_crs(crs)
    
    return zones_gdf


# ============================================================================
# CRS CONVERSION UTILITIES
# ============================================================================

def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """
    Ensure a GeoDataFrame is in the target CRS.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame.
    target_crs : str
        Target CRS (e.g., "EPSG:4326").
    
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame in the target CRS.
    """
    if gdf.crs is None:
        warnings.warn(f"GeoDataFrame has no CRS. Setting to {target_crs}.")
        return gdf.set_crs(target_crs)
    
    if str(gdf.crs) != target_crs:
        return gdf.to_crs(target_crs)
    
    return gdf


def create_points_gdf(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    crs: str = CRS_WGS84
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from a DataFrame with lat/lon columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with coordinate columns.
    lon_col : str
        Name of the longitude column.
    lat_col : str
        Name of the latitude column.
    crs : str
        CRS for the output GeoDataFrame. Default is WGS84.
    
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with Point geometry.
    """
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs
    )


# ============================================================================
# SPATIAL JOIN UTILITIES
# ============================================================================

def assign_zones(
    points_gdf: gpd.GeoDataFrame,
    zones_gdf: Optional[gpd.GeoDataFrame] = None,
    zone_id_col: str = ZONE_ID_COL
) -> gpd.GeoDataFrame:
    """
    Assign zone IDs to points via spatial join.
    
    Parameters:
    -----------
    points_gdf : gpd.GeoDataFrame
        GeoDataFrame with point geometries.
    zones_gdf : gpd.GeoDataFrame, optional
        Zones GeoDataFrame. If None, will be loaded automatically.
    zone_id_col : str
        Name of the zone ID column to add.
    
    Returns:
    --------
    gpd.GeoDataFrame
        Points with zone ID column added.
    """
    if zones_gdf is None:
        zones_gdf = load_zones(crs=str(points_gdf.crs))
    
    # Ensure same CRS
    zones_gdf = ensure_crs(zones_gdf, str(points_gdf.crs))
    
    # Spatial join
    result = gpd.sjoin(
        points_gdf,
        zones_gdf[[zone_id_col, ZONE_NAME_COL, 'geometry']],
        how='left',
        predicate='within'
    )
    
    # Clean up index column from sjoin
    if 'index_right' in result.columns:
        result = result.drop(columns=['index_right'])
    
    return result


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_zone_consistency(
    gdf: gpd.GeoDataFrame,
    zone_id_col: str = ZONE_ID_COL
) -> Tuple[bool, str]:
    """
    Validate that a GeoDataFrame has the expected number of zones.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to validate.
    zone_id_col : str
        Name of the zone ID column.
    
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, message)
    """
    if zone_id_col not in gdf.columns:
        return False, f"Zone ID column '{zone_id_col}' not found."
    
    unique_zones = gdf[zone_id_col].nunique()
    
    if unique_zones == EXPECTED_ZONE_COUNT:
        return True, f"Zone count OK: {unique_zones} zones."
    else:
        return False, f"Zone count mismatch: expected {EXPECTED_ZONE_COUNT}, got {unique_zones}."


def get_zone_summary(gdf: gpd.GeoDataFrame) -> dict:
    """
    Get a summary of zone information from a GeoDataFrame.
    
    Returns:
    --------
    dict
        Summary including CRS, zone count, bounds, etc.
    """
    return {
        'crs': str(gdf.crs),
        'zone_count': len(gdf),
        'expected_zones': EXPECTED_ZONE_COUNT,
        'is_valid_count': len(gdf) == EXPECTED_ZONE_COUNT,
        'bounds': gdf.total_bounds.tolist(),
        'columns': list(gdf.columns)
    }


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SPATIAL UTILITIES - CRS and Zone Verification")
    print("=" * 70)
    
    try:
        # Test loading zones in different CRS
        print("\n[1] Loading zones in WGS84 (EPSG:4326)...")
        zones_wgs84 = load_zones(crs=CRS_WGS84)
        print(f"    ✓ Loaded {len(zones_wgs84)} zones")
        print(f"    CRS: {zones_wgs84.crs}")
        print(f"    Bounds: {zones_wgs84.total_bounds}")
        
        print("\n[2] Loading zones in UTM 32N (EPSG:32632)...")
        zones_utm = load_zones(crs=CRS_UTM32N)
        print(f"    ✓ Loaded {len(zones_utm)} zones")
        print(f"    CRS: {zones_utm.crs}")
        
        print("\n[3] Loading zones with area calculation...")
        zones_area = load_zones_with_area(crs=CRS_WGS84)
        print(f"    ✓ Total area: {zones_area['area_sqkm'].sum():.2f} km²")
        print(f"    Mean zone area: {zones_area['area_sqkm'].mean():.2f} km²")
        
        print("\n[4] Zone validation...")
        is_valid, msg = validate_zone_consistency(zones_wgs84)
        status = "✓" if is_valid else "✗"
        print(f"    {status} {msg}")
        
        print("\n" + "=" * 70)
        print("All spatial utilities working correctly!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
