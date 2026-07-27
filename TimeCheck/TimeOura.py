"""
analyze_oura_sampling_frequency.py

Goal
----
The pipeline currently assumes Oura Ring heart rate data arrives as one
sample every 5 minutes. This script checks that assumption directly by
looking at the *actual* timestamps in each participant's raw HR files:

  1. Time-gap analysis: for every pair of consecutive timestamps in a file,
     how much time elapsed? This tells us the real sampling interval(s).
  2. Burst analysis: Oura (and many wearables) often write several samples
     with the same or near-identical timestamp, then go quiet for a while.
     This groups consecutive rows into "bursts" (rows whose gap from the
     previous row is below a small threshold) and reports how many data
     points land in each burst, and how far apart bursts are from one
     another.

Outputs
-------
- time_gap_distribution.csv   : every observed gap (seconds) + frequency
- burst_size_distribution.csv : every observed burst size (# points) + frequency
- per_participant_summary.csv : per-participant median/most-common gap & burst size
- gap_histogram.png           : histogram of time gaps between consecutive samples
- burst_size_histogram.png    : histogram of burst sizes
- Printed summary stats to console
"""

import os
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths (mirrors the existing heatmap script) ===
root_path = "/Users/cibrian/Documents/Github/Research"
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/OuraRing/SamplingFrequency")
os.makedirs(output_folder, exist_ok=True)

# === Burst detection threshold ===
# Rows whose gap from the previous row is <= this many seconds are treated
# as belonging to the same "burst" (i.e. written together in one batch).
#
# NOTE: an earlier run of this script (threshold = 2.0s) showed the most
# common sample-to-sample gap is actually ~5 seconds for nearly every
# participant -- i.e. the ring's native recording cadence is ~5s, not the
# assumed 5 minutes. A 2s threshold was splitting a single ~5s-cadence
# recording session into many separate "bursts" of size 1. Raising the
# threshold to comfortably cover a 5s cadence (with some jitter) groups
# those samples into one real burst/session, so burst size now reflects
# how many samples were captured per recording session, and the time
# between bursts reflects genuine gaps/dropouts in coverage.
BURST_GAP_THRESHOLD_SEC = 7.0

# === Find participant folders ===
participant_folders = [
    f for f in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, f)) and f.startswith('P')
]

def get_participant_number(name):
    return int(name[1:])

participant_folders = sorted(participant_folders, key=get_participant_number)


def parse_time(s):
    try:
        return datetime.strptime(s, "%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(s, "%H:%M:%S")


def is_valid_hr(bpm):
    if pd.isna(bpm):
        return False
    return 40 <= bpm <= 200


# === Accumulators (global, across all participants/files) ===
all_gaps_sec = []            # every consecutive-row time gap, in seconds
all_burst_sizes = []         # size (# points) of every detected burst
all_burst_gap_sec = []       # gap between the *start* of consecutive bursts

# Per-participant accumulators for the summary table
participant_gap_lists = {p: [] for p in participant_folders}
participant_burst_size_lists = {p: [] for p in participant_folders}
participant_burst_gap_lists = {p: [] for p in participant_folders}

files_processed = 0
files_skipped_missing_cols = 0

for participant in participant_folders:
    hr_path = os.path.join(root_path, participant, "OuraRing", "HeartRate")
    if not os.path.exists(hr_path):
        continue

    for file in os.listdir(hr_path):
        if not (file.endswith(".csv") and "RAW" not in file):
            continue

        file_path = os.path.join(hr_path, file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Could not read {file_path}: {e}")
            continue

        if 'Time_In_PST' not in df.columns or 'bpm' not in df.columns:
            files_skipped_missing_cols += 1
            continue

        df['TimeObj'] = df['Time_In_PST'].apply(parse_time)
        df['valid_hr'] = df['bpm'].apply(is_valid_hr)

        # Only look at rows with a valid HR sample -- these are the actual
        # "data points" the pipeline consumes downstream.
        df = df[df['valid_hr']].copy()
        if df.empty:
            continue

        df = df.sort_values('TimeObj').reset_index(drop=True)
        files_processed += 1

        # --- Gap analysis: seconds between consecutive samples ---
        gaps = df['TimeObj'].diff().dropna().dt.total_seconds().tolist()
        all_gaps_sec.extend(gaps)
        participant_gap_lists[participant].extend(gaps)

        # --- Burst detection ---
        # Walk through gaps; start a new burst whenever the gap exceeds the
        # threshold, otherwise extend the current burst.
        burst_sizes_this_file = []
        burst_start_times = []
        current_burst_size = 1
        current_burst_start = df.loc[0, 'TimeObj']
        burst_start_times.append(current_burst_start)

        for i, gap in enumerate(gaps, start=1):
            if gap <= BURST_GAP_THRESHOLD_SEC:
                current_burst_size += 1
            else:
                burst_sizes_this_file.append(current_burst_size)
                current_burst_size = 1
                current_burst_start = df.loc[i, 'TimeObj']
                burst_start_times.append(current_burst_start)
        burst_sizes_this_file.append(current_burst_size)

        all_burst_sizes.extend(burst_sizes_this_file)
        participant_burst_size_lists[participant].extend(burst_sizes_this_file)

        # Gap between the start of one burst and the start of the next
        if len(burst_start_times) > 1:
            burst_gaps = [
                (burst_start_times[i] - burst_start_times[i - 1]).total_seconds()
                for i in range(1, len(burst_start_times))
            ]
            all_burst_gap_sec.extend(burst_gaps)
            participant_burst_gap_lists[participant].extend(burst_gaps)

print(f"Files processed: {files_processed}")
print(f"Files skipped (missing columns): {files_skipped_missing_cols}")

if not all_gaps_sec:
    raise SystemExit("No valid HR data found -- check root_path / folder structure.")

# === Build distribution tables ===

# Round gaps to the nearest second for a readable frequency table
gap_rounded = np.round(all_gaps_sec).astype(int)
gap_counts = Counter(gap_rounded)
gap_dist_df = (
    pd.DataFrame(sorted(gap_counts.items()), columns=['gap_seconds', 'count'])
    .assign(pct=lambda d: (d['count'] / d['count'].sum() * 100).round(2))
)
gap_dist_df.to_csv(os.path.join(output_folder, "time_gap_distribution.csv"), index=False)

burst_counts = Counter(all_burst_sizes)
burst_dist_df = (
    pd.DataFrame(sorted(burst_counts.items()), columns=['burst_size_points', 'count'])
    .assign(pct=lambda d: (d['count'] / d['count'].sum() * 100).round(2))
)
burst_dist_df.to_csv(os.path.join(output_folder, "burst_size_distribution.csv"), index=False)

# === Per-participant summary ===
summary_rows = []
for p in participant_folders:
    gaps_p = participant_gap_lists[p]
    bursts_p = participant_burst_size_lists[p]
    burst_gaps_p = participant_burst_gap_lists[p]
    if not gaps_p:
        continue
    gap_mode = Counter(np.round(gaps_p).astype(int)).most_common(1)[0][0]
    burst_mode = Counter(bursts_p).most_common(1)[0][0] if bursts_p else np.nan
    summary_rows.append({
        'Part': p,
        'n_samples': len(gaps_p) + 1,  # +1 because diff() drops the first sample
        'median_gap_sec': np.median(gaps_p),
        'mean_gap_sec': np.mean(gaps_p),
        'most_common_gap_sec': gap_mode,
        'n_bursts': len(bursts_p),
        'median_burst_size': np.median(bursts_p) if bursts_p else np.nan,
        'most_common_burst_size': burst_mode,
        'max_burst_size': max(bursts_p) if bursts_p else np.nan,
        'median_time_btwn_bursts': np.median(burst_gaps_p) if burst_gaps_p else np.nan,
        'mean_time_btwn_bursts': np.mean(burst_gaps_p) if burst_gaps_p else np.nan,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(output_folder, "per_participant_summary.csv"), index=False)

# === Plots ===

# Gap histogram (clip at e.g. 600s / 10 min for readability; note outliers separately)
clip_sec = 600
gaps_clipped = [g for g in all_gaps_sec if g <= clip_sec]
n_outliers = len(all_gaps_sec) - len(gaps_clipped)

plt.figure(figsize=(10, 6))
sns.histplot(gaps_clipped, bins=60, color='steelblue')
plt.title(f'Distribution of Time Gaps Between Consecutive HR Samples\n'
          f'(clipped at {clip_sec}s; {n_outliers} larger gaps not shown)')
plt.xlabel('Gap between consecutive samples (seconds)')
plt.ylabel('Count')
plt.axvline(300, color='red', linestyle='--', label='Assumed 5-min interval (300s)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "gap_histogram.png"), dpi=300)
plt.close()

# Burst size histogram
plt.figure(figsize=(10, 6))
max_burst_display = min(max(all_burst_sizes), 30)
sns.histplot(all_burst_sizes, bins=range(1, max_burst_display + 2), discrete=True, color='seagreen')
plt.title('Distribution of Burst Sizes (# of samples written together)')
plt.xlabel('Number of data points per burst')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "burst_size_histogram.png"), dpi=300)
plt.close()

# Time-between-bursts histogram (clip for readability, same as gap histogram)
if all_burst_gap_sec:
    burst_gaps_clipped = [g for g in all_burst_gap_sec if g <= clip_sec]
    n_burst_gap_outliers = len(all_burst_gap_sec) - len(burst_gaps_clipped)

    plt.figure(figsize=(10, 6))
    sns.histplot(burst_gaps_clipped, bins=60, color='darkorange')
    plt.title(f'Distribution of Time Between Bursts\n'
              f'(clipped at {clip_sec}s; {n_burst_gap_outliers} larger gaps not shown)')
    plt.xlabel('Time between start of consecutive bursts (seconds)')
    plt.ylabel('Count')
    plt.axvline(300, color='red', linestyle='--', label='Assumed 5-min interval (300s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "burst_gap_histogram.png"), dpi=300)
    plt.close()

# === Console summary ===
print("\n=== Time Gap Summary (seconds between consecutive samples) ===")
print(f"Median gap: {np.median(all_gaps_sec):.1f}s")
print(f"Mean gap:   {np.mean(all_gaps_sec):.1f}s")
print(f"Most common gap(s):")
print(gap_dist_df.sort_values('count', ascending=False).head(10).to_string(index=False))

print("\n=== Burst Size Summary (# points arriving together) ===")
print(f"Median burst size: {np.median(all_burst_sizes):.1f}")
print(f"Most common burst size(s):")
print(burst_dist_df.sort_values('count', ascending=False).head(10).to_string(index=False))

if all_burst_gap_sec:
    print("\n=== Gap Between Bursts (seconds) ===")
    print(f"Median: {np.median(all_burst_gap_sec):.1f}s")
    print(f"Mean:   {np.mean(all_burst_gap_sec):.1f}s")

print("\n=== Per-Participant Summary ===")
print(summary_df.to_string(index=False))

print(f"\nAll outputs saved to: {output_folder}")