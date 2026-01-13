
import pandas as pd
import os

# Get the project root directory (TWO levels up from src/data/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)  # src/
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # project root

# Define data paths
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')

print("="*100)
print(" COLUMN INSPECTION - All 3 Operators")
print("="*100)

# ==========================================
# OPERATOR A (LIME)
# ==========================================

print("\n" + "="*100)
print(" OPERATOR A: LIME - Torino_Corse24-25.csv")
print("="*100)

try:
    df_lime = pd.read_csv(
        os.path.join(DATA_RAW, 'lime', 'Torino_Corse24-25.csv'),
        encoding='utf-8',
        low_memory=False
    )

    print(f"\n✅ LIME Records: {len(df_lime):,}")
    print(f"\n🔍 COLUMN NAMES ({len(df_lime.columns)} columns):")
    for i, col in enumerate(df_lime.columns, 1):
        print(f"  {i:2d}. {col}")

    print(f"\nDATA TYPES:")
    for col in df_lime.columns:
        print(f"  {col:40s} â†’ {str(df_lime[col].dtype):15s}")

    print(f"\nFIRST 5 ROWS:")
    print(df_lime.head())

    print(f"\nBasic Stats:")
    print(f"  Shape: {df_lime.shape}")
    print(f"  Memory: {df_lime.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Nulls: {df_lime.isnull().sum().sum():,}")
    
except Exception as e:
    print(f"ERROR: {e}")

# ==========================================
# OPERATOR B (VOI)
# ==========================================

print("\n\n" + "="*100)
print(" OPERATOR B: VOI - Monthly Files")
print("="*100)

voi_base_path = os.path.join(DATA_RAW, 'voi')

# Load first file (January 2024) to see structure
first_voi_file = 'DATINOLEGGI_202401.xlsx'
first_voi_path = os.path.join(voi_base_path, first_voi_file)

try:
    df_voi_sample = pd.read_excel(first_voi_path)
    
    print(f"\nSample file: {first_voi_file}")
    print(f"  Records in this file: {len(df_voi_sample):,}")
    
    print(f"\nCOLUMN NAMES ({len(df_voi_sample.columns)} columns):")
    for i, col in enumerate(df_voi_sample.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\nDATA TYPES:")
    for col in df_voi_sample.columns:
        print(f"  {col:40s} â†’ {str(df_voi_sample[col].dtype):15s}")

    print(f"\nFIRST 5 ROWS:")
    print(df_voi_sample.head())
    
    print(f"\nBasic Stats:")
    print(f"  Shape: {df_voi_sample.shape}")
    print(f"  Memory: {df_voi_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Nulls: {df_voi_sample.isnull().sum().sum():,}")
    
    # Now check all VOI files (skip 202503)
    print(f"\n🔍 Checking all VOI files:")
    voi_files = []
    for year in [2024, 2025]:
        months = range(1, 13) if year == 2024 else range(1, 11)
        for month in months:
            # SKIP MARCH 2025 (202503)
            if year == 2025 and month == 3:
                print(f"  ⚠️ SKIPPED: DATINOLEGGI_202503.xlsx (file not provided)")
                continue
            
            filename = f"DATINOLEGGI_{year}{month:02d}.xlsx"
            filepath = os.path.join(voi_base_path, filename)
            
            if os.path.exists(filepath):
                df = pd.read_excel(filepath)
                print(f"  ✅ {filename:30s} → {len(df):>8,} records")
                voi_files.append(len(df))
            else:
                print(f"  ERROR: {filename:30s} → NOT FOUND")

    if voi_files:
        print(f"\n  TOTAL: {sum(voi_files):,} records across {len(voi_files)} files")
    
except Exception as e:
    print(f"ERROR: {e}")

# ==========================================
# OPERATOR C (BIRD) - FIXED PATH
# ==========================================

print("\n\n" + "="*100)
print(" OPERATOR C: BIRD - Annual Files (CSV Format)")
print("="*100)

# Use relative path based on project structure
bird_base_path = os.path.join(DATA_RAW, 'bird')

# Bird file names (CSV format)
bird_files_names = [
    'Bird Torino - 2024 - Sheet1.csv',
    'Bird Torino - 2025 (fino il 18_11_2025) - Sheet1.csv'
]

print(f"\nðŸ“ Looking for files in: {bird_base_path}")

# First, verify the directory exists
if not os.path.exists(bird_base_path):
    print(f"ERROR: Directory not found!")
    print(f"  Please verify the path is correct")
else:
    print(f"  Directory found")
    
    # List actual files in the directory
    print(f"\n  Files in directory:")
    actual_files = os.listdir(bird_base_path)
    for f in actual_files:
        if os.path.isfile(os.path.join(bird_base_path, f)):
            print(f"    - {f}")

# Now try to load Bird files
for filename in bird_files_names:
    filepath = os.path.join(bird_base_path, filename)
    
    # print(f"\n{'â”€'*100}")
    print(f"FILE: {filename}")
    # print(f"{'â”€'*100}")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at: {filepath}")
        continue

    print(f"File found")

    try:
        # Try to read CSV with different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        used_encoding = None
        
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, encoding=enc, low_memory=False)
                used_encoding = enc
                print(f"Loaded with encoding: {used_encoding}")
                break
            except Exception as enc_error:
                continue
        
        if df is None:
            print(f"ERROR: Could not load file with any encoding")
            continue
        
        print(f"  Records: {len(df):,}")

        print(f"\nCOLUMN NAMES ({len(df.columns)} columns):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")

        print(f"\nDATA TYPES:")
        for col in df.columns:
            print(f"  {col:40s} â†’ {str(df[col].dtype):15s}")

        print(f"\nFIRST 5 ROWS:")
        print(df.head(5))

        print(f"\nBASIC STATS:")
        print(f"  Shape: {df.shape}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"  Nulls: {df.isnull().sum().sum():,}")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

# ==========================================
# SUMMARY
# ==========================================

print("\n\n" + "="*100)
print(" SUMMARY - Ready for Analysis")
print("="*100)
print("""
Next step: Use these column names to update your cleaning code!

For each operator, you now have:
  Exact column names
  Data types
  Sample data
  Record counts

Key observations:
  1. Italian column names (no English translations)
  2. Combined datetime columns (not separate date + time)
  3. Typo in some column names (check LIME LONGITUTIDE vs LONGITUDINE)
  4. Different operators may have different column name formats

Next: Create mapping in cleaning code matching these exact names!
""")

print("="*100)
