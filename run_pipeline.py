#!/usr/bin/env python3
"""
==============================================================================
RUN_PIPELINE.PY - Master Pipeline Controller
==============================================================================

Single entry point to run the complete Turin E-Scooter analysis pipeline.

PIPELINE STAGES:
----------------
Stage 0: Data Cleaning       - Clean and prepare raw data
Stage 1: Temporal Analysis   - Hourly/daily/monthly patterns
Stage 2: OD Matrix Analysis  - Origin-destination flow analysis
Stage 3: Integration Analysis - E-scooter & public transport comparison
Stage 4: Parking Analysis    - Parking duration and patterns
Stage 5: Economic Analysis   - Revenue and fleet economics

USAGE:
------
    # Run complete pipeline
    python run_pipeline.py
    
    # Run specific stages
    python run_pipeline.py --stages 1 2 3
    
    # Run from specific stage
    python run_pipeline.py --from-stage 3
    
    # Run only visualizations
    python run_pipeline.py --viz-only
    
    # Skip visualizations
    python run_pipeline.py --no-viz

Author: Ali Vaezi
Last Updated: December 2025
==============================================================================
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import importlib.util

# ==============================================================================
# CONFIGURATION
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"

# Define pipeline stages and their corresponding scripts
PIPELINE_CONFIG = {
    # Stage 0: Data Cleaning
    0: {
        "name": "Data Cleaning",
        "analysis": "src/data/00_data_cleaning.py",
        "visualization": "src/visualization/00_data_cleaning.py",
        "description": "Clean and preprocess raw e-scooter data"
    },
    # Stage 1: Temporal Analysis
    1: {
        "name": "Temporal Analysis",
        "analysis": "src/analysis/01_temporal_analysis.py",
        "visualization": "src/visualization/01_temporal_dashboard.py",
        "statistics": "src/visualization/01_temporal_statistics.py",
        "description": "Analyze hourly, daily, and monthly usage patterns"
    },
    # Stage 2: OD Matrix Analysis
    2: {
        "name": "OD Matrix Analysis",
        "analysis": "src/analysis/02_od_matrix_analysis.py",
        "visualization": "src/visualization/02_od_spatial_flows.py",
        "statistics": "src/visualization/02_od_statistics.py",
        "description": "Compute origin-destination flow matrices"
    },
    # Stage 3: Integration Analysis
    3: {
        "name": "Integration Analysis",
        "analysis": "src/analysis/03_integration_analysis.py",
        "visualization": "src/visualization/03_integration_maps.py",
        "statistics": "src/visualization/03_integration_statistics.py",
        "description": "Compare e-scooter with public transport"
    },
    # Stage 4: Parking Analysis
    4: {
        "name": "Parking Analysis",
        "analysis": "src/analysis/04_parking_analysis.py",
        "visualization": "src/visualization/04_parking_maps.py",
        "statistics": "src/visualization/04_parking_survival.py",
        "description": "Analyze parking duration and patterns"
    },
    # Stage 5: Economic Analysis
    5: {
        "name": "Economic Analysis",
        "analysis": "src/analysis/05_economic_analysis.py",
        "visualization": "src/visualization/05_economic_maps.py",
        "statistics": "src/visualization/05_economic_sensitivity.py",
        "description": "Revenue and fleet economic analysis"
    }
}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")


def print_stage(stage_num, name, description):
    """Print stage information"""
    print(f"{Colors.BLUE}[Stage {stage_num}]{Colors.ENDC} {Colors.BOLD}{name}{Colors.ENDC}")
    print(f"          {description}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}  ✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}  ✗ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}  ⚠ {message}{Colors.ENDC}")


def get_active_config():
    """Determine which config to use based on file existence"""
    # Check if reorganization has been done
    new_structure_exists = (PROJECT_ROOT / "src" / "analysis").exists()
    
    if new_structure_exists:
        return PIPELINE_CONFIG


def run_script(script_path, description=""):
    """Run a Python script and capture output"""
    full_path = PROJECT_ROOT / script_path
    
    if not full_path.exists():
        print_warning(f"Script not found: {script_path}")
        return False
    
    print(f"\n          Running: {script_path}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print_success(f"Completed in {elapsed:.1f}s")
            # Print last few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')[-3:]
                for line in lines:
                    if line.strip():
                        print(f"          {line}")
            return True
        else:
            print_error(f"Failed with exit code {result.returncode}")
            if result.stderr:
                print(f"{Colors.RED}          Error: {result.stderr[:200]}...{Colors.ENDC}")
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"Script timed out after 1 hour")
        return False
    except Exception as e:
        print_error(f"Exception: {str(e)}")
        return False


# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

def run_stage(stage_num, config, run_viz=True, run_stats=True):
    """Run a single pipeline stage"""
    stage = config.get(stage_num)
    if not stage:
        print_error(f"Unknown stage: {stage_num}")
        return False
    
    print_stage(stage_num, stage["name"], stage["description"])
    print("-" * 60)
    
    success = True
    
    # Run analysis script
    if stage.get("analysis"):
        print(f"\n{Colors.CYAN}[Analysis]{Colors.ENDC}")
        if not run_script(stage["analysis"]):
            success = False
    
    # Run visualization script
    if run_viz and stage.get("visualization"):
        print(f"\n{Colors.CYAN}[Visualization]{Colors.ENDC}")
        if not run_script(stage["visualization"]):
            # Visualization failure is not critical
            print_warning("Visualization had issues but continuing...")
    
    # Run statistical figures
    if run_stats and stage.get("statistics"):
        print(f"\n{Colors.CYAN}[Statistical Figures]{Colors.ENDC}")
        if not run_script(stage["statistics"]):
            print_warning("Statistics visualization had issues but continuing...")
    
    return success


def run_pipeline(stages=None, from_stage=None, viz_only=False, no_viz=False, no_stats=False):
    """Run the complete pipeline or selected stages"""
    
    print_header("TURIN E-SCOOTER SHARED MOBILITY ANALYSIS PIPELINE")
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project Root: {PROJECT_ROOT}")
    
    # Get active configuration
    config = get_active_config()
    
    # Determine which stages to run
    all_stages = sorted(config.keys())
    
    if stages:
        stages_to_run = [s for s in stages if s in all_stages]
    elif from_stage is not None:
        stages_to_run = [s for s in all_stages if s >= from_stage]
    else:
        stages_to_run = all_stages
    
    print(f"\n  Stages to run: {stages_to_run}")
    print(f"  Run visualizations: {not no_viz}")
    print(f"  Run statistics figures: {not no_stats}")
    
    # Pipeline execution
    print_header("EXECUTING PIPELINE")
    
    start_time = time.time()
    results = {}
    
    for stage_num in stages_to_run:
        stage_start = time.time()
        
        if viz_only:
            # Only run visualization scripts
            stage = config[stage_num]
            success = True
            if stage.get("visualization"):
                success = run_script(stage["visualization"])
            if not no_stats and stage.get("statistics"):
                run_script(stage["statistics"])
        else:
            # Run full stage
            success = run_stage(
                stage_num, 
                config, 
                run_viz=not no_viz,
                run_stats=not no_stats
            )
        
        stage_elapsed = time.time() - stage_start
        results[stage_num] = {
            "success": success,
            "time": stage_elapsed
        }
        
        print(f"\n          Stage {stage_num} completed in {stage_elapsed:.1f}s")
        print()
    
    # Summary
    total_time = time.time() - start_time
    
    print_header("PIPELINE SUMMARY")
    
    print(f"  Total execution time: {total_time/60:.1f} minutes")
    print()
    print("  Stage Results:")
    print("  " + "-" * 50)
    
    for stage_num, result in results.items():
        stage_name = config[stage_num]["name"]
        status = f"{Colors.GREEN}✓ SUCCESS{Colors.ENDC}" if result["success"] else f"{Colors.RED}✗ FAILED{Colors.ENDC}"
        print(f"  Stage {stage_num}: {stage_name:<25} {status} ({result['time']:.1f}s)")
    
    print("  " + "-" * 50)
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    if successful == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}  ✓ ALL STAGES COMPLETED SUCCESSFULLY!{Colors.ENDC}")
    else:
        print(f"\n{Colors.YELLOW}  ⚠ {successful}/{total} stages completed successfully{Colors.ENDC}")
    
    print(f"\n  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return successful == total


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Run Turin E-Scooter Shared Mobility Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                  # Run complete pipeline
  python run_pipeline.py --stages 1 2 3   # Run specific stages
  python run_pipeline.py --from-stage 3   # Run from stage 3 onwards
  python run_pipeline.py --viz-only       # Only run visualizations
  python run_pipeline.py --no-viz         # Skip visualizations
  python run_pipeline.py --no-stats       # Skip statistical figures

Stages:
  0: Data Cleaning
  1: Temporal Analysis
  2: OD Matrix Analysis
  3: Integration Analysis (E-Scooter vs Public Transport)
  4: Parking Analysis
  5: Economic Analysis
        """
    )
    
    parser.add_argument(
        "--stages", "-s",
        type=int,
        nargs="+",
        help="Specific stages to run (e.g., --stages 1 2 3)"
    )
    
    parser.add_argument(
        "--from-stage", "-f",
        type=int,
        help="Run pipeline starting from this stage"
    )
    
    parser.add_argument(
        "--viz-only", "-v",
        action="store_true",
        help="Only run visualization scripts (skip analysis)"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization scripts"
    )
    
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip statistical figures generation"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all pipeline stages and exit"
    )
    
    args = parser.parse_args()
    
    # List stages
    if args.list:
        config = get_active_config()
        print("\nPipeline Stages:")
        print("-" * 60)
        for num, stage in sorted(config.items()):
            print(f"  Stage {num}: {stage['name']}")
            print(f"           {stage['description']}")
            if stage.get('analysis'):
                print(f"           Analysis: {stage['analysis']}")
            if stage.get('visualization'):
                print(f"           Viz: {stage['visualization']}")
        return
    
    # Run pipeline
    success = run_pipeline(
        stages=args.stages,
        from_stage=args.from_stage,
        viz_only=args.viz_only,
        no_viz=args.no_viz,
        no_stats=args.no_stats
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
