"""
Test Runner for DelphiAI

Runs all tests and generates a summary report.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all tests with pytest."""
    test_dir = Path(__file__).parent
    
    print("="*70)
    print("DELPHI AI TEST SUITE")
    print("="*70)
    
    # Run pytest with verbose output
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(test_dir),
            "-v",
            "--tb=short",
            "-x",  # Stop on first failure
        ],
        capture_output=False,
    )
    
    return result.returncode


if __name__ == '__main__':
    sys.exit(run_tests())
