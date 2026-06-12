"""
Weekly Update Pipeline

Runs all update steps in the correct order after Saturday fight cards:
  1. Scrape latest data from UFC.com / UFCStats
  2. Validate scraped CSVs
  3. Load data to database (upsert)
  4. Recalculate ELO ratings
  5. Re-populate fighter styles + style matchup records
  6. Update adjusted ELOs (inactivity penalties)

Usage:
    python -m ml.weekly_update                  # Full update (scrape + everything)
    python -m ml.weekly_update --skip-scrape    # Skip scrape, just recalculate
    python -m ml.weekly_update --dry-run        # Preview what would happen
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Paths
MODELS_DIR = Path(__file__).parent.parent  # DelphiAIApp/Models
DATA_DIR = MODELS_DIR / 'data'
SCRAPERS_DIR = DATA_DIR / 'scrapers'
OUTPUT_DIR = DATA_DIR / 'output'
ML_DIR = MODELS_DIR / 'ml'


def run_step(step_num, total, name, command, working_dir, dry_run=False):
    """Run a pipeline step and report timing."""
    print(f"\n{'=' * 60}")
    print(f"  STEP {step_num}/{total}: {name}")
    print(f"{'=' * 60}")
    
    if dry_run:
        print(f"  [DRY RUN] Would run: {command}")
        print(f"  Working dir: {working_dir}")
        return True, 0
    
    start = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(working_dir),
            capture_output=False,  # Show output in real-time
            text=True,
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"\n  COMPLETED in {elapsed:.1f}s")
            return True, elapsed
        else:
            print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
            return False, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ERROR: {e} (after {elapsed:.1f}s)")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Weekly update pipeline - refresh all fighter data after fight cards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python -m ml.weekly_update                  # Full update
  python -m ml.weekly_update --skip-scrape    # Skip scrape, recalculate only
  python -m ml.weekly_update --dry-run        # Preview steps
  python -m ml.weekly_update --skip-scrape --skip-elo  # Just reload DB + styles
        '''
    )
    parser.add_argument('--skip-scrape', action='store_true',
                        help='Skip scraping (use existing CSV files)')
    parser.add_argument('--skip-elo', action='store_true',
                        help='Skip ELO recalculation')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview steps without executing')
    parser.add_argument('--incremental', action='store_true',
                        help='Scrape only fighters from recent/upcoming events (~20-30 fighters) '
                             'instead of all 946. Use this for weekly runs; full scrape runs quarterly.')
    
    args = parser.parse_args()

    # Incremental mode: resolve a temp file path for the fighter URL list
    _fighters_tmp = Path("/tmp/delphi_active_fighters.txt")
    if args.incremental and sys.platform == "win32":
        _fighters_tmp = Path(os.environ.get("TEMP", "C:/Temp")) / "delphi_active_fighters.txt"

    print("\n" + "#" * 60)
    print(f"{'DELPHI AI - WEEKLY UPDATE':^60}")
    print(f"{'Started: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^60}")
    print("#" * 60)

    if args.dry_run:
        print("\n  *** DRY RUN MODE - No changes will be made ***")
    if args.incremental:
        print("\n  *** INCREMENTAL MODE - Only scraping recently active fighters ***")

    # Calculate total steps
    steps = []
    step_num = 0

    if not args.skip_scrape:
        if args.incremental:
            # Step 1a: collect the fighter URL list
            _get_fighters_script = Path(__file__).resolve().parents[3] / "scripts" / "get_active_fighters.py"
            step_num += 1
            steps.append((step_num, "Identify Active Fighters",
                           f"python {_get_fighters_script} --output {_fighters_tmp}",
                           MODELS_DIR))
            # Step 1b: incremental scrape
            step_num += 1
            steps.append((step_num, "Scrape Active Fighters (Incremental)",
                           f"python scrape_all.py --fighters-file {_fighters_tmp}",
                           SCRAPERS_DIR))
        else:
            step_num += 1
            steps.append((step_num, "Scrape UFC Data",
                           f"python scrape_all.py",
                           SCRAPERS_DIR))
    
    step_num += 1
    steps.append((step_num, "Validate Data",
                   f"python validate_data.py --fix",
                   DATA_DIR))
    
    step_num += 1
    steps.append((step_num, "Load to Database",
                   f"python load_to_db.py",
                   DATA_DIR))
    
    if not args.skip_elo:
        step_num += 1
        steps.append((step_num, "Recalculate ELO Ratings",
                       f"python features.py",
                       DATA_DIR))
    
    step_num += 1
    steps.append((step_num, "Update Fighter Styles",
                   f"python -m ml.populate_styles",
                   MODELS_DIR))
    
    step_num += 1
    steps.append((step_num, "Update Adjusted ELOs",
                   f"python -m ml.update_adjusted_elos",
                   MODELS_DIR))
    
    total_steps = len(steps)
    
    # Show plan
    print(f"\n  Pipeline ({total_steps} steps):")
    for num, name, cmd, _ in steps:
        print(f"    {num}. {name}")
    
    # Run steps
    results = []
    total_time = 0
    
    for num, name, cmd, working_dir in steps:
        success, elapsed = run_step(num, total_steps, name, cmd, working_dir, dry_run=args.dry_run)
        results.append((name, success, elapsed))
        total_time += elapsed
        
        if not success and not args.dry_run:
            print(f"\n  Pipeline stopped at step {num} due to failure.")
            print(f"  Fix the issue and re-run with --skip-scrape if scraping already completed.")
            break
    
    # Summary
    print("\n" + "#" * 60)
    print(f"{'UPDATE SUMMARY':^60}")
    print("#" * 60)
    
    for name, success, elapsed in results:
        status = "OK" if success else "FAILED"
        if args.dry_run:
            status = "SKIPPED (dry run)"
        print(f"  [{status:>6}] {name} ({elapsed:.1f}s)")
    
    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    all_passed = all(s for _, s, _ in results)
    if all_passed and not args.dry_run:
        print(f"\n  All steps completed successfully!")
        print(f"  Fighter data is up to date as of {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    elif args.dry_run:
        print(f"\n  Dry run complete. Run without --dry-run to execute.")
    else:
        print(f"\n  Some steps failed. Check output above for details.")
        sys.exit(1)
    
    print("#" * 60)


if __name__ == '__main__':
    main()
