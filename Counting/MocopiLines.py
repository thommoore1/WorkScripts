#!/usr/bin/env python3
"""
Counts the number of lines in each CSV file found within the
Mocopi/Raw subfolders for a list of participant IDs.

Expected folder structure:
    <BASE_DIR>/<PARTICIPANT_ID>/Mocopi/Raw/**/*.csv

Usage:
    python count_csv_lines.py
    python count_csv_lines.py --base-dir /path/to/Research
"""

import argparse
import os
import sys

DEFAULT_BASE_DIR = "/Users/cibrian/Documents/Github/Research"

DEFAULT_PARTICIPANTS = [
    "P001", "P002", "P003", "P004", "P005", "P006", "P007",
    "P008", "P009", "P0012", "P0014", "P0016",
]


def count_lines(filepath):
    """Count the number of lines in a file, tolerant of encoding issues."""
    count = 0
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
            for _ in f:
                count += 1
    except OSError as e:
        print(f"  ERROR reading {filepath}: {e}", file=sys.stderr)
        return None
    return count


def find_csvs(raw_dir):
    """Recursively find all CSV files under raw_dir."""
    csv_paths = []
    for root, _dirs, files in os.walk(raw_dir):
        for name in files:
            if name.lower().endswith(".csv"):
                csv_paths.append(os.path.join(root, name))
    return sorted(csv_paths)


def format_size(num_bytes):
    """Human-readable file size."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description="Count lines in CSV files under each participant's Mocopi/Raw folder.")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                         help=f"Base directory containing participant folders (default: {DEFAULT_BASE_DIR})")
    parser.add_argument("--participants", nargs="+", default=DEFAULT_PARTICIPANTS,
                         help="List of participant IDs (default: the P001-P0016 list)")
    args = parser.parse_args()

    total_lines = 0
    total_size = 0
    total_files = 0

    for participant in args.participants:
        raw_dir = os.path.join(args.base_dir, participant, "Mocopi", "Raw")
        print(f"\n=== {participant} ===")

        if not os.path.isdir(raw_dir):
            print(f"  Folder not found: {raw_dir}")
            continue

        csv_files = find_csvs(raw_dir)
        if not csv_files:
            print(f"  No CSV files found in {raw_dir}")
            continue

        for filepath in csv_files:
            lines = count_lines(filepath)
            rel = os.path.relpath(filepath, raw_dir)
            if lines is not None:
                size = os.path.getsize(filepath)
                print(f"  {rel}: {lines} lines, {format_size(size)}")
                total_lines += lines
                total_size += size
                total_files += 1

    print("\n=== Totals ===")
    print(f"  Files:       {total_files}")
    print(f"  Total lines: {total_lines}")
    print(f"  Total size:  {format_size(total_size)}")


if __name__ == "__main__":
    main()