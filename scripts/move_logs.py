#!/usr/bin/env python
"""
Script to move log files from logs/ directory to output/ directory.

This script helps with the transition from the old logging setup to the new one
by moving existing log files to the standardized output directory.
"""

import os
import shutil
import argparse
from datetime import datetime


def main():
    """Move log files from logs/ directory to output/ directory."""
    print("\n=== Log File Migration Utility ===")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Move log files from logs/ to output/ directory")
    parser.add_argument(
        "--source-dir",
        default="logs",
        help="Source directory containing log files (default: logs)"
    )
    parser.add_argument(
        "--target-dir",
        default="output",
        help="Target directory to move log files to (default: output)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files"
    )
    args = parser.parse_args()
    
    source_dir = args.source_dir
    target_dir = args.target_dir
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return 1
    
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        if args.dry_run:
            print(f"Would create target directory: {target_dir}")
        else:
            os.makedirs(target_dir, exist_ok=True)
            print(f"Created target directory: {target_dir}")
    
    # Get list of log files in source directory
    log_files = [f for f in os.listdir(source_dir) if f.endswith(".log")]
    
    if not log_files:
        print(f"No log files found in '{source_dir}' directory.")
        return 0
    
    print(f"Found {len(log_files)} log files in '{source_dir}' directory.")
    
    # Move each log file
    moved_count = 0
    for log_file in log_files:
        source_path = os.path.join(source_dir, log_file)
        target_path = os.path.join(target_dir, log_file)
        
        if args.dry_run:
            print(f"Would move: {source_path} -> {target_path}")
            moved_count += 1
        else:
            try:
                shutil.move(source_path, target_path)
                print(f"Moved: {source_path} -> {target_path}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {source_path}: {str(e)}")
    
    action = "Would move" if args.dry_run else "Moved"
    print(f"\n{action} {moved_count} out of {len(log_files)} log files.")
    
    if not args.dry_run and moved_count > 0:
        # Create a README file in the logs directory to explain the move
        readme_path = os.path.join(source_dir, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write(f"""# Log Files Have Moved

Log files have been moved to the '{target_dir}/' directory.

This change was made on {datetime.now().strftime('%Y-%m-%d')} to standardize the output location
for all strategy runners. New log files will be automatically created in the
'{target_dir}/' directory based on the configuration file's output directory setting.

Please update any scripts or tools that expect log files in this directory.
""")
            print(f"Created README in {source_dir}/ directory to explain the move.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 