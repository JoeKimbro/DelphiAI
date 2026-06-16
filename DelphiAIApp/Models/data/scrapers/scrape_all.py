"""
UFC Data Scraper

Runs the unified ufc_official spider which:
1. Scrapes fighter data from UFC.com
2. For each fighter, also looks them up on UFCStats
3. Merges data from both sources (UFC.com priority, UFCStats fills gaps)
4. Yields complete fighter records with all available data

This ensures NO missing fields - every fighter gets data from both sources.

Usage:
    python scrape_all.py                    # Run full scrape
    python scrape_all.py --stats-only       # Run only UFCStats spider (legacy)
    python scrape_all.py --test             # Run with limited pages for testing
    python scrape_all.py --fresh            # Clear output files before scraping

Requirements:
    pip install scrapy
"""

import argparse
import os
import sys
from pathlib import Path

# Add the scrapers directory to the path
scrapers_dir = Path(__file__).parent
sys.path.insert(0, str(scrapers_dir))

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings


def run_spiders(stats_only=False, test_mode=False, fresh=False, fighters_file=None):
    """
    Run the UFC scrapers.

    Args:
        stats_only: Legacy mode - only run standalone UFCStats spider
        test_mode: Run with limited pages for testing
        fresh: Clear output files before scraping (start fresh)
        fighters_file: Path to a file of UFC.com fighter URLs for incremental scrape.
                       When provided, only those fighters are scraped (skips full listing crawl).
    """
    # Clear output files if fresh mode
    output_dir = scrapers_dir.parent / 'output'
    if fresh and output_dir.exists():
        print("[FRESH] Clearing existing output files...")
        for f in output_dir.glob('*.csv'):
            f.unlink()
            print(f"   Removed {f.name}")
        for f in output_dir.glob('*.json'):
            f.unlink()
            print(f"   Removed {f.name}")
        print()
    
    # Change to the scrapers directory so Scrapy can find settings
    os.chdir(scrapers_dir / 'ufc_scraper')
    
    # Get Scrapy settings
    settings = get_project_settings()
    
    # Apply test mode settings if requested
    if test_mode:
        settings.set('CLOSESPIDER_PAGECOUNT', 10)
        settings.set('LOG_LEVEL', 'DEBUG')
        print("[TEST MODE] Running with limited pages (10 max)")
    
    # Create crawler process
    process = CrawlerProcess(settings)
    
    # Determine which spiders to run
    # The ufc_official spider now handles both UFC.com AND UFCStats lookups
    spiders_to_run = []
    
    if stats_only:
        # Legacy mode: run only the standalone UFCStats spider.
        # WARNING: UFCStats never visits UFC.com, so it cannot populate
        # weight_class, nickname, place_of_birth, or leg_reach. The CSV
        # pipeline preserves previously-scraped values for those fields
        # (see CsvExportPipeline._PRESERVE_IF_BLANK), but if this is the
        # FIRST scrape ever run, those fields will be left blank for every
        # fighter. Prefer the default `ufc_official` spider for routine use.
        print("[WARNING] --stats-only does not collect weight_class/nickname/")
        print("          place_of_birth/leg_reach (UFC.com-only fields).")
        print("          Existing values for those fields are preserved, but")
        print("          this mode cannot fill them in for new fighters.")
        spiders_to_run.append('ufcstats')
    else:
        # Default: run the unified spider that scrapes both sources
        spiders_to_run.append('ufc_official')
    
    if fighters_file:
        print(f"[INCREMENTAL] Using fighter list: {fighters_file}")

    print("=" * 60)
    print("UFC DATA SCRAPER")
    print("=" * 60)
    print(f"Spiders to run: {', '.join(spiders_to_run)}")
    print("=" * 60)
    print()

    # Queue spiders
    for spider_name in spiders_to_run:
        print(f"[QUEUE] Adding spider: {spider_name}")
        spider_kwargs = {}
        if fighters_file and spider_name == 'ufc_official':
            spider_kwargs['fighter_urls_file'] = fighters_file
        process.crawl(spider_name, **spider_kwargs)
    
    # Run all spiders
    print("\n[START] Beginning scrape...\n")
    process.start()
    
    print("\n" + "=" * 60)
    print("[DONE] All spiders completed!")
    print("=" * 60)
    print("\nOutput files:")
    
    output_dir = scrapers_dir.parent / 'output'
    if output_dir.exists():
        for f in output_dir.glob('*.csv'):
            print(f"  - {f.name}")
        for f in output_dir.glob('*.json'):
            print(f"  - {f.name}")
    
    print("\nNext steps:")
    print("  1. Review the CSV files in the output directory")
    print("  2. Run 'python load_to_db.py' to load data into PostgreSQL")


def main():
    parser = argparse.ArgumentParser(
        description='Run UFC data scrapers (UFC.com + UFCStats)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_all.py              # Run both spiders
  python scrape_all.py --ufc-only   # Only scrape UFC.com
  python scrape_all.py --stats-only # Only scrape UFCStats
  python scrape_all.py --test       # Test mode (limited pages)
        """
    )
    
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Legacy mode: run only the standalone UFCStats spider (no UFC.com data)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with limited pages'
    )
    
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Clear existing output files before scraping (start fresh)'
    )

    parser.add_argument(
        '--fighters-file',
        metavar='PATH',
        default=None,
        help='Path to a file of UFC.com fighter URLs (one per line) for incremental scraping. '
             'Skips the full athlete listing crawl.'
    )

    args = parser.parse_args()

    run_spiders(
        stats_only=args.stats_only,
        test_mode=args.test,
        fresh=args.fresh,
        fighters_file=args.fighters_file,
    )


if __name__ == '__main__':
    main()
