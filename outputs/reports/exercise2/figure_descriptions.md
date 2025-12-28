# Turin E-Scooter Shared Mobility Analysis
## Professional Figure Descriptions and Interpretations

**Analysis Period:** Full dataset (2019-2023)
**Total Valid Trips:** 2,509,948
**Zone System:** 94 statistical zones (Zone Statistiche di Torino)

---

## 1. Origin-Destination Heatmaps (Combined All Operators)

### Figure 1.1: `od_heatmap_allday_top30.png`
**Title:** O-D Matrix Heatmap - All Day (Top 30 Zones)

**Description:**
This heatmap displays the trip distribution between the 30 most active zones in Turin, representing the core e-scooter network. The visualization captures 2,509,948 total trips across all three operators (LIME, BIRD, VOI). Color intensity (red gradient) indicates trip volume, with darker shades representing higher demand corridors.

**Key Insights:**
- **Diagonal Pattern:** The prominent diagonal indicates significant intra-zonal trips (10.5% of total), suggesting e-scooters are used for short trips within zones.
- **Off-Diagonal Hotspots:** Strong corridors between central zones (04, 01, 03, 08) indicate inter-zonal commuting patterns, particularly city center connections.
- **Zone Concentration:** The top 30 zones capture the majority of e-scooter activity, with Zone 04 (Piazza San Carlo) being the most active origin (126,229 trips).

### Figure 1.2: `od_heatmap_peak_top30.png`
**Title:** O-D Matrix Heatmap - Peak Hours (07:00-09:00, 17:00-19:00)

**Description:**
This heatmap isolates 752,924 trips (30.0% of total) occurring during morning and evening rush hours. The pattern reveals commuting behavior.

**Key Insights:**
- **Commuter Corridors:** Stronger off-diagonal patterns compared to all-day, indicating directional flows toward employment centers in the morning and residential areas in the evening.
- **Transport Hub Activity:** Zones near Porta Nuova (Zone 10) and Porta Susa (Zone 08) show elevated activity, suggesting last-mile connectivity with public transport.

### Figure 1.3: `od_heatmap_offpeak_top30.png`
**Title:** O-D Matrix Heatmap - Off-Peak Hours

**Description:**
Representing 1,757,024 trips (70.0% of total), this heatmap shows leisure and non-commuting travel patterns.

**Key Insights:**
- **More Diffuse Pattern:** Activity is more evenly distributed across zones.
- **Recreational Zones:** Zones near parks (Valentino Park - Zone 09) and entertainment districts show relatively higher activity.

---

## 2. Flow Maps (Combined All Operators)

### Figure 2.1: `flow_map_allday_ALL.png`
**Title:** All Inter-Zonal E-Scooter Flows - All Day

**Description:**
This comprehensive flow map visualizes ALL 6,435 unique origin-destination corridors in the Turin e-scooter network. Lines connect zone centroids, with color and thickness graduated by trip volume.

**Color Legend (Trips per Corridor):**
- 🔴 **10,000+**: Highest volume corridors (major arterials)
- 🟠 **5,000-10,000**: Very high volume (670 corridors in 1k+ category)
- 🟡 **2,500-5,000**: High volume corridors
- 🟢 **1,000-2,500**: Medium-high volume
- 🔵 **500-1,000**: Medium volume (546 corridors)
- ⚪ **100-500**: Low-medium volume (1657 corridors)
- ⚫ **1-100**: Low volume (3562 corridors)

**Key Insights:**
- **Central Core Dominance:** The densest flow cluster is in the historic center, connecting Zones 01, 03, 04, 08, and 10.
- **Radial Pattern:** Flows radiate from center to peripheral zones (Lingotto, San Donato).
- **North-South Axis:** Strong connectivity along Via Roma and Corso Francia corridors.

### Figure 2.2: `flow_map_allday_improved.png`
**Title:** E-Scooter Flows - All Day (500+ trips)

**Description:**
A filtered view showing only corridors with 500+ trips, highlighting the core network structure without visual clutter from low-volume connections.

---

## 3. Per-Operator O-D Heatmaps

**Note:** These heatmaps are **normalized** within each operator to show their relative trip distribution patterns, not absolute volumes. This reveals each operator's unique market positioning despite LIME's dominant market share.

### Figure 3.1: `od_heatmap_lime_allday.png`
**Operator:** LIME (1,418,185 trips, 56.5% market share)

**Description:**
LIME's normalized O-D matrix reveals their coverage strategy across Turin. As the market leader, LIME shows the most balanced distribution across all zones.

**Key Insights:**
- **Broad Coverage:** Strong presence in both central and peripheral zones.
- **Top Corridors:** Zone 04→01 (7,334 trips), Zone 04→03 (6,472 trips) - city center focus.
- **Lower Intra-zonal Rate (7.8%):** Users travel longer distances on average.

### Figure 3.2: `od_heatmap_bird_allday.png`
**Operator:** BIRD (823,057 trips, 32.8% market share)

**Description:**
BIRD's pattern shows concentration in specific neighborhood clusters, particularly in the southern and eastern parts of Turin.

**Key Insights:**
- **Stronger Diagonal:** Higher intra-zonal rate (14.7%) - used for shorter trips.
- **Southern Focus:** Strong in Zones 53, 56, 57, 61 (Lingotto/Nizza Millefonti area).
- **Top Corridors:** Zone 57→57, 56→56 - local circulation patterns.

### Figure 3.3: `od_heatmap_voi_allday.png`
**Operator:** VOI (268,706 trips, 10.7% market share)

**Description:**
VOI shows the most distinctive pattern with concentrated activity in specific corridors, suggesting a niche market strategy or limited operational zone.

**Key Insights:**
- **Corridor Specialization:** Zone 38→23 is their top corridor (4,143 trips).
- **University Connection:** Strong in zones near Politecnico (Zone 35, 38, 23).
- **Limited Geographic Spread:** Fewer active zone pairs than competitors.

---

## 4. Per-Operator Flow Maps

### Figure 4.1: `flow_map_lime_allday.png`
**Operator:** LIME - All Flows Visualization

**Description:**
This map displays all 5,907 unique corridors served by LIME. The network shows comprehensive citywide coverage.

**Flow Distribution:**
- High volume (1k+): 331 corridors
- Medium volume (100-1k): 1,561 corridors
- Low volume (1-100): 3,932 corridors

**Geographic Pattern:**
- Dense core network in the city center
- Radial extensions to all major neighborhoods
- Strongest flows between transport hubs and commercial areas

### Figure 4.2: `flow_map_bird_allday.png`
**Operator:** BIRD - All Flows Visualization

**Description:**
BIRD's flow map reveals a more clustered pattern with 6,178 corridors. The network shows distinct operational clusters.

**Flow Distribution:**
- High volume (1k+): 68 corridors
- Medium volume (100-1k): 1,751 corridors
- Low volume (1-100): 4,277 corridors

**Geographic Pattern:**
- Concentrated activity in southern Turin (Lingotto district)
- Secondary cluster in the central-eastern area
- Less inter-cluster connectivity compared to LIME

### Figure 4.3: `flow_map_voi_allday.png`
**Operator:** VOI - All Flows Visualization

**Description:**
VOI shows the most concentrated network with 4,385 corridors, reflecting their smaller fleet and focused operational area.

**Flow Distribution:**
- High volume (1k+): 13 corridors only
- Medium volume (100-1k): 600 corridors
- Low volume (1-100): 3,697 corridors

**Geographic Pattern:**
- Strong north-central corridor (Zones 35-38-23-39)
- University area specialization (Politecnico di Torino)
- Minimal presence in southern and western districts

---

## 5. Operator Comparison Analysis

### Figure 5.1: `operator_comparison.png`
**Title:** E-Scooter Operator Comparison - Turin

**Description:**
A multi-panel comparison showing (1) Market share distribution, (2) Trip volume by operator, (3) Intra-zonal vs inter-zonal split, and (4) Zone coverage breadth.

**Key Comparative Insights:**
| Metric | LIME | BIRD | VOI |
|--------|------|------|-----|
| Market Share | 56.5% | 32.8% | 10.7% |
| Total Trips | 1,418,185 | 823,057 | 268,706 |
| Intra-zonal Rate | 7.8% | 14.7% | 11.8% |
| Active Corridors | ~5,800 | ~6,100 | ~4,300 |

**Market Positioning:**
- **LIME:** Market leader with broadest geographic coverage and longest average trip distances.
- **BIRD:** Strong regional presence, particularly in southern Turin, with more localized usage patterns.
- **VOI:** Niche player focused on university/student corridors with concentrated demand.

---

## 6. Data Quality Notes

- **Temporal Coverage:** Multi-year dataset (2019-2023)
- **Spatial Join Success Rate:** 98.7% of trips mapped to zones
- **Peak Hours Definition:** [7, 8, 9, 17, 18, 19] (morning and evening rush)
- **Coordinate System:** EPSG:4326 (WGS84)
- **Zone System:** ISTAT Zone Statistiche (94 zones)
