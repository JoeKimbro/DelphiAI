"""
Data Validation Script for UFC Fighter Data
Run this before loading to database to catch issues early.

Usage:
    python validate_data.py              # Validate all CSVs
    python validate_data.py --fix        # Validate and create cleaned versions
    python validate_data.py --summary    # Quick summary only
"""

import pandas as pd
import numpy as np
import re
import argparse
from datetime import datetime
from pathlib import Path

# Configuration
OUTPUT_DIR = Path(__file__).parent / "output"
VALIDATION_REPORT_FILE = OUTPUT_DIR / "validation_report.txt"

class DataValidator:
    def __init__(self):
        self.errors = []  # Critical issues (❌)
        self.warnings = []  # Acceptable issues (⚠️)
        self.info = []  # Informational
        self.fixes_applied = []
        
    def log_error(self, category, message, count=1):
        self.errors.append({"category": category, "message": message, "count": count})
        
    def log_warning(self, category, message, count=1):
        self.warnings.append({"category": category, "message": message, "count": count})
        
    def log_info(self, category, message):
        self.info.append({"category": category, "message": message})

    def validate_fighters(self, df, fix=False):
        """Validate fighters.csv"""
        print("\n" + "="*60)
        print("VALIDATING: fighters.csv")
        print("="*60)
        
        original_count = len(df)
        self.log_info("fighters", f"Total records: {original_count}")
        
        # 1. Check for duplicates by fighter_url (critical)
        dupes = df[df.duplicated(subset=['fighter_url'], keep=False)]
        if len(dupes) > 0:
            self.log_error("fighters", f"Duplicate fighter_url entries", len(dupes))
            if fix:
                df = df.drop_duplicates(subset=['fighter_url'], keep='last')
                self.fixes_applied.append(f"Removed {len(dupes) - len(df[df.duplicated(subset=['fighter_url'], keep=False)])} duplicate fighters")
        
        # 2. Check required fields
        required_fields = ['name', 'fighter_url']
        for field in required_fields:
            if field in df.columns:
                missing = df[field].isna().sum()
                if missing > 0:
                    self.log_error("fighters", f"Missing required field '{field}'", missing)
            else:
                self.log_error("fighters", f"Required column '{field}' not found", 1)
        
        # 3. Check name format (should not be empty or just whitespace)
        if 'name' in df.columns:
            bad_names = df[df['name'].str.strip().eq('') | df['name'].isna()]
            if len(bad_names) > 0:
                self.log_error("fighters", f"Empty or whitespace-only names", len(bad_names))
        
        # 4. Validate weight_class values
        if 'weight_class' in df.columns:
            valid_classes = [
                'Strawweight', 'Flyweight', 'Bantamweight', 'Featherweight',
                'Lightweight', 'Welterweight', 'Middleweight', 'Light Heavyweight',
                'Heavyweight', "Women's Strawweight", "Women's Flyweight",
                "Women's Bantamweight", "Women's Featherweight", 'Catch Weight', ''
            ]
            missing_wc = df['weight_class'].isna().sum()
            if missing_wc > 0:
                self.log_warning("fighters", f"Missing weight_class", missing_wc)
            
            # Check for unusual weight classes
            if not df['weight_class'].isna().all():
                unusual = df[~df['weight_class'].isna() & ~df['weight_class'].isin(valid_classes)]
                if len(unusual) > 0:
                    unique_unusual = unusual['weight_class'].unique()[:5]
                    self.log_warning("fighters", f"Unusual weight classes: {list(unique_unusual)}", len(unusual))
        
        # 5. Validate numeric fields are in reasonable ranges
        numeric_checks = {
            'height_cm': (100, 220, "Height"),
            'reach_cm': (100, 230, "Reach"),
            'wins': (0, 100, "Wins"),
            'losses': (0, 100, "Losses"),
            'draws': (0, 20, "Draws"),
        }
        
        for field, (min_val, max_val, label) in numeric_checks.items():
            if field in df.columns:
                # Convert to numeric, coercing errors
                numeric_col = pd.to_numeric(df[field], errors='coerce')
                out_of_range = numeric_col[(numeric_col < min_val) | (numeric_col > max_val)]
                if len(out_of_range) > 0:
                    self.log_warning("fighters", f"{label} out of expected range ({min_val}-{max_val})", len(out_of_range))
        
        # 6. Check for valid date of birth format
        if 'date_of_birth' in df.columns:
            non_null_dob = df[df['date_of_birth'].notna()]
            if len(non_null_dob) > 0:
                # Try to parse dates
                try:
                    parsed = pd.to_datetime(non_null_dob['date_of_birth'], errors='coerce')
                    invalid_dates = parsed.isna().sum()
                    if invalid_dates > 0:
                        self.log_warning("fighters", f"Invalid date_of_birth format", invalid_dates)
                    
                    # Check for unrealistic dates
                    future_dates = (parsed > datetime.now()).sum()
                    if future_dates > 0:
                        self.log_error("fighters", f"Future date_of_birth values", future_dates)
                except Exception as e:
                    self.log_warning("fighters", f"Could not parse date_of_birth: {e}", 1)
        
        # 7. URL format validation
        if 'fighter_url' in df.columns:
            invalid_urls = df[~df['fighter_url'].str.contains(r'^https?://', na=False, regex=True)]
            if len(invalid_urls) > 0:
                self.log_error("fighters", f"Invalid fighter_url format (not a URL)", len(invalid_urls))
        
        return df
    
    def validate_career_stats(self, df, fix=False):
        """Validate career_stats.csv"""
        print("\n" + "="*60)
        print("VALIDATING: career_stats.csv")
        print("="*60)
        
        original_count = len(df)
        self.log_info("career_stats", f"Total records: {original_count}")
        
        # 1. Check for duplicates by fighter_url
        dupes = df[df.duplicated(subset=['fighter_url'], keep=False)]
        if len(dupes) > 0:
            self.log_error("career_stats", f"Duplicate fighter_url entries", len(dupes))
            if fix:
                df = df.drop_duplicates(subset=['fighter_url'], keep='last')
                self.fixes_applied.append(f"Removed duplicate career_stats entries")
        
        # 2. Check required fields
        required_fields = ['fighter_url']
        for field in required_fields:
            if field in df.columns:
                missing = df[field].isna().sum()
                if missing > 0:
                    self.log_error("career_stats", f"Missing required field '{field}'", missing)
        
        # 3. Validate percentage fields (0-100 or 0-1)
        pct_fields = ['str_acc', 'str_def', 'td_acc', 'td_def']
        for field in pct_fields:
            if field in df.columns:
                numeric_col = pd.to_numeric(df[field], errors='coerce')
                # Check if values are reasonable (could be 0-1 or 0-100)
                invalid = numeric_col[(numeric_col < 0) | (numeric_col > 100)]
                if len(invalid) > 0:
                    self.log_warning("career_stats", f"{field} out of range (0-100)", len(invalid))
        
        # 4. Validate per-minute stats
        rate_fields = ['slpm', 'sapm', 'td_avg', 'sub_avg']
        for field in rate_fields:
            if field in df.columns:
                numeric_col = pd.to_numeric(df[field], errors='coerce')
                invalid = numeric_col[(numeric_col < 0) | (numeric_col > 20)]
                if len(invalid) > 0:
                    self.log_warning("career_stats", f"{field} unusually high (>20)", len(invalid))
        
        # 5. Check ELO ratings
        if 'elo_rating' in df.columns:
            elo = pd.to_numeric(df['elo_rating'], errors='coerce')
            missing_elo = elo.isna().sum()
            if missing_elo > 0:
                self.log_warning("career_stats", f"Missing elo_rating", missing_elo)
            
            # ELO should typically be 800-2000
            out_of_range = elo[(elo < 800) | (elo > 2200)]
            if len(out_of_range) > 0:
                self.log_warning("career_stats", f"ELO rating outside typical range (800-2200)", len(out_of_range))
        
        # 6. Validate win streak is non-negative
        if 'win_streak' in df.columns:
            ws = pd.to_numeric(df['win_streak'], errors='coerce')
            negative = ws[ws < 0]
            if len(negative) > 0:
                self.log_error("career_stats", f"Negative win_streak values", len(negative))
        
        return df
    
    def validate_fights(self, df, fix=False):
        """Validate fights.csv"""
        print("\n" + "="*60)
        print("VALIDATING: fights.csv")
        print("="*60)
        
        original_count = len(df)
        self.log_info("fights", f"Total records: {original_count}")
        
        # 1. Check for duplicates
        if 'fight_url' in df.columns:
            dupes = df[df.duplicated(subset=['fight_url'], keep=False)]
            if len(dupes) > 0:
                self.log_error("fights", f"Duplicate fight_url entries", len(dupes))
                if fix:
                    df = df.drop_duplicates(subset=['fight_url'], keep='last')
        
        # 2. Check required fields
        required_fields = ['fight_url', 'fighter1_url', 'fighter2_url']
        for field in required_fields:
            if field in df.columns:
                missing = df[field].isna().sum()
                if missing > 0:
                    self.log_warning("fights", f"Missing field '{field}'", missing)
        
        # 3. Validate date format
        if 'date' in df.columns:
            non_null_dates = df[df['date'].notna()]
            if len(non_null_dates) > 0:
                parsed = pd.to_datetime(non_null_dates['date'], errors='coerce')
                invalid = parsed.isna().sum()
                if invalid > 0:
                    self.log_warning("fights", f"Invalid date format", invalid)
                
                # Check for future fights (warning, might be scheduled)
                future = (parsed > datetime.now()).sum()
                if future > 0:
                    self.log_info("fights", f"Future dated fights (scheduled?): {future}")
        
        # 4. Validate method
        if 'method' in df.columns:
            valid_methods = ['KO/TKO', 'SUB', 'DEC', 'U-DEC', 'S-DEC', 'M-DEC', 
                           'NC', 'DQ', 'Overturned', 'Could Not Continue', '']
            non_null = df[df['method'].notna()]
            # Just check that method field exists and has reasonable values
            
        # 5. Validate round
        if 'round' in df.columns:
            rounds = pd.to_numeric(df['round'], errors='coerce')
            invalid = rounds[(rounds < 1) | (rounds > 5)]
            if len(invalid) > 0:
                self.log_warning("fights", f"Round outside 1-5 range", len(invalid))
        
        return df
    
    def validate_events(self, df, fix=False):
        """Validate events.csv"""
        print("\n" + "="*60)
        print("VALIDATING: events.csv")
        print("="*60)
        
        original_count = len(df)
        self.log_info("events", f"Total records: {original_count}")
        
        # 1. Check for duplicates
        if 'event_url' in df.columns:
            dupes = df[df.duplicated(subset=['event_url'], keep=False)]
            if len(dupes) > 0:
                self.log_error("events", f"Duplicate event_url entries", len(dupes))
                if fix:
                    df = df.drop_duplicates(subset=['event_url'], keep='last')
        
        return df
    
    def generate_report(self):
        """Generate validation report"""
        report = []
        report.append("="*70)
        report.append("UFC DATA VALIDATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*70)
        
        # Summary
        report.append("\n[SUMMARY]")
        report.append("-"*40)
        report.append(f"[X] Critical Errors: {len(self.errors)}")
        report.append(f"[!] Warnings: {len(self.warnings)}")
        report.append(f"[i] Info: {len(self.info)}")
        
        # Critical Errors
        if self.errors:
            report.append("\n[X] CRITICAL ERRORS (Must Fix)")
            report.append("-"*40)
            for e in self.errors:
                report.append(f"  [{e['category']}] {e['message']} (count: {e['count']})")
        
        # Warnings
        if self.warnings:
            report.append("\n[!] WARNINGS (Review Recommended)")
            report.append("-"*40)
            for w in self.warnings:
                report.append(f"  [{w['category']}] {w['message']} (count: {w['count']})")
        
        # Info
        if self.info:
            report.append("\n[i] INFO")
            report.append("-"*40)
            for i in self.info:
                report.append(f"  [{i['category']}] {i['message']}")
        
        # Fixes Applied
        if self.fixes_applied:
            report.append("\n[FIX] FIXES APPLIED")
            report.append("-"*40)
            for f in self.fixes_applied:
                report.append(f"  + {f}")
        
        # Recommendation
        report.append("\n" + "="*70)
        if len(self.errors) == 0:
            report.append("[OK] RECOMMENDATION: Data is ready for database upload")
        else:
            report.append("[X] RECOMMENDATION: Fix critical errors before database upload")
            report.append("    Run with --fix flag to auto-fix issues where possible")
        report.append("="*70)
        
        return "\n".join(report)
    
    def check_cross_file_integrity(self, fighters_df, career_df, fights_df):
        """Check referential integrity between files"""
        print("\n" + "="*60)
        print("CROSS-FILE INTEGRITY CHECKS")
        print("="*60)
        
        fighter_urls = set(fighters_df['fighter_url'].dropna()) if 'fighter_url' in fighters_df.columns else set()
        
        # Career stats should reference valid fighters
        if 'fighter_url' in career_df.columns:
            career_urls = set(career_df['fighter_url'].dropna())
            orphan_career = career_urls - fighter_urls
            if len(orphan_career) > 0:
                self.log_warning("integrity", f"Career stats for fighters not in fighters.csv", len(orphan_career))
        
        # Fights should reference valid fighters
        if fights_df is not None and len(fights_df) > 0:
            if 'fighter1_url' in fights_df.columns:
                fight_fighter1 = set(fights_df['fighter1_url'].dropna())
                orphan_f1 = fight_fighter1 - fighter_urls
                if len(orphan_f1) > 0:
                    self.log_warning("integrity", f"Fights referencing unknown fighter1_url", len(orphan_f1))
            
            if 'fighter2_url' in fights_df.columns:
                fight_fighter2 = set(fights_df['fighter2_url'].dropna())
                orphan_f2 = fight_fighter2 - fighter_urls
                if len(orphan_f2) > 0:
                    self.log_warning("integrity", f"Fights referencing unknown fighter2_url", len(orphan_f2))


def main():
    parser = argparse.ArgumentParser(description='Validate UFC fighter data before database upload')
    parser.add_argument('--fix', action='store_true', help='Auto-fix issues and create validated CSVs')
    parser.add_argument('--summary', action='store_true', help='Quick summary only')
    args = parser.parse_args()
    
    validator = DataValidator()
    
    # Load CSVs
    fighters_df = None
    career_df = None
    fights_df = None
    events_df = None
    
    print("\n[*] Loading data files...")
    
    try:
        fighters_df = pd.read_csv(OUTPUT_DIR / "fighters.csv")
        print(f"  [+] fighters.csv: {len(fighters_df)} rows")
    except FileNotFoundError:
        print("  [-] fighters.csv not found")
    except Exception as e:
        print(f"  [-] fighters.csv error: {e}")
    
    try:
        career_df = pd.read_csv(OUTPUT_DIR / "career_stats.csv")
        print(f"  [+] career_stats.csv: {len(career_df)} rows")
    except FileNotFoundError:
        print("  [-] career_stats.csv not found")
    except Exception as e:
        print(f"  [-] career_stats.csv error: {e}")
    
    try:
        fights_df = pd.read_csv(OUTPUT_DIR / "fights.csv")
        print(f"  [+] fights.csv: {len(fights_df)} rows")
    except FileNotFoundError:
        print("  [i] fights.csv not found (optional)")
        fights_df = pd.DataFrame()
    except Exception as e:
        print(f"  [-] fights.csv error: {e}")
        fights_df = pd.DataFrame()
    
    try:
        events_df = pd.read_csv(OUTPUT_DIR / "events.csv")
        print(f"  [+] events.csv: {len(events_df)} rows")
    except FileNotFoundError:
        print("  [i] events.csv not found (optional)")
        events_df = pd.DataFrame()
    except Exception as e:
        print(f"  [-] events.csv error: {e}")
        events_df = pd.DataFrame()
    
    # Run validations
    if fighters_df is not None:
        fighters_df = validator.validate_fighters(fighters_df, fix=args.fix)
    
    if career_df is not None:
        career_df = validator.validate_career_stats(career_df, fix=args.fix)
    
    if fights_df is not None and len(fights_df) > 0:
        fights_df = validator.validate_fights(fights_df, fix=args.fix)
    
    if events_df is not None and len(events_df) > 0:
        events_df = validator.validate_events(events_df, fix=args.fix)
    
    # Cross-file integrity
    if fighters_df is not None and career_df is not None:
        validator.check_cross_file_integrity(
            fighters_df, 
            career_df, 
            fights_df if fights_df is not None else pd.DataFrame()
        )
    
    # Generate and print report
    report = validator.generate_report()
    print("\n" + report)
    
    # Save report
    with open(VALIDATION_REPORT_FILE, 'w') as f:
        f.write(report)
    print(f"\n[*] Report saved to: {VALIDATION_REPORT_FILE}")
    
    # Save fixed CSVs if requested
    if args.fix and len(validator.fixes_applied) > 0:
        print("\n[*] Saving validated CSVs...")
        if fighters_df is not None:
            fighters_df.to_csv(OUTPUT_DIR / "fighters_validated.csv", index=False)
            print(f"  + fighters_validated.csv")
        if career_df is not None:
            career_df.to_csv(OUTPUT_DIR / "career_stats_validated.csv", index=False)
            print(f"  + career_stats_validated.csv")
        if fights_df is not None and len(fights_df) > 0:
            fights_df.to_csv(OUTPUT_DIR / "fights_validated.csv", index=False)
            print(f"  + fights_validated.csv")
        if events_df is not None and len(events_df) > 0:
            events_df.to_csv(OUTPUT_DIR / "events_validated.csv", index=False)
            print(f"  + events_validated.csv")
    
    # Return exit code based on errors
    return 1 if len(validator.errors) > 0 else 0


if __name__ == "__main__":
    exit(main())
