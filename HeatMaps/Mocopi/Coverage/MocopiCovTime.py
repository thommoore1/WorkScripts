import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# === Paths ===
root_path = "/Users/cibrian/Documents/Github/Research"
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/Mocopi/Coverage")
fileName_coverage = "time_coverage.png"
fileName_csv = "time_coverage_metrics.csv"
os.makedirs(output_folder, exist_ok=True)

# === Find participant folders ===
participant_folders = [
    f for f in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, f)) and f.startswith('P')
]

def get_participant_number(name):
    return int(name[1:])

participant_folders = sorted(participant_folders, key=get_participant_number)

# === Define time bins ===
start_time = datetime.strptime("08:30", "%H:%M")
end_time = datetime.strptime("15:00", "%H:%M")

# 30-min bins for heatmap display
time_bins_30min = []
temp_time = start_time
while temp_time < end_time:
    bin_end = temp_time + timedelta(minutes=30)
    time_bins_30min.append((temp_time.time(), bin_end.time()))
    temp_time = bin_end

# 5-min bins for coverage calculation
time_bins_5min = []
temp_time = start_time
while temp_time < end_time:
    bin_end = temp_time + timedelta(minutes=5)
    time_bins_5min.append((temp_time.time(), bin_end.time()))
    temp_time = bin_end

# === Initialize dataframes ===
heatmap_coverage = pd.DataFrame(
    0.0,
    index=[f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" for start, end in time_bins_30min],
    columns=participant_folders
)

coverage_metrics = {p: {'total_bins': 0, 'covered_bins': 0} for p in participant_folders}

# === Helpers ===
def parse_time(s):
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time: {s}")

# === Process each participant ===
for participant in participant_folders:
    mocopi_path = os.path.join(root_path, participant, "Mocopi", "Labeled")
    if not os.path.exists(mocopi_path):
        print(f"  Skipping {participant}: path not found")
        continue

    # For each 5-min bin, track which days had coverage (1) or not (0)
    participant_5min_coverage = {
        f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}": []
        for start, end in time_bins_5min
    }

    for root_dir, dirs, files in os.walk(mocopi_path):
        for fname in files:
            if not fname.endswith('.csv'):
                continue

            # Extract date robustly
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
            if not date_match:
                continue
            date_str = date_match.group(1)

            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue

            # Skip Fridays
            if file_date.weekday() == 4:
                continue

            file_path = os.path.join(root_dir, fname)
            df = pd.read_csv(file_path)

            if 'Time_In_PST' not in df.columns:
                continue

            df['TimeObj'] = df['Time_In_PST'].apply(parse_time)

            # For each 5-min bin, check if any row falls within it
            for bin_start, bin_end in time_bins_5min:
                interval = f"{bin_start.strftime('%H:%M')}-{bin_end.strftime('%H:%M')}"
                bin_df = df[df['TimeObj'].between(bin_start, bin_end)]
                has_data = not bin_df.empty
                participant_5min_coverage[interval].append(1 if has_data else 0)

    # === Aggregate coverage across days ===
    total_5min_bins = len(time_bins_5min)
    covered_5min_bins = 0

    for interval, coverage_list in participant_5min_coverage.items():
        if coverage_list and sum(coverage_list) > 0:
            covered_5min_bins += 1

    coverage_metrics[participant]['total_bins'] = total_5min_bins
    coverage_metrics[participant]['covered_bins'] = covered_5min_bins
    coverage_metrics[participant]['coverage_pct'] = (
        (covered_5min_bins / total_5min_bins * 100) if total_5min_bins > 0 else 0
    )

    # === Aggregate 5-min coverage into 30-min bins for heatmap ===
    for start_30, end_30 in time_bins_30min:
        interval_30 = f"{start_30.strftime('%H:%M')}-{end_30.strftime('%H:%M')}"

        # Collect all 5-min bins within this 30-min window
        bins_in_30min = []
        temp_time = datetime.combine(datetime.today(), start_30)
        end_time_30 = datetime.combine(datetime.today(), end_30)

        while temp_time < end_time_30:
            bin_end = temp_time + timedelta(minutes=5)
            interval_5 = f"{temp_time.time().strftime('%H:%M')}-{bin_end.time().strftime('%H:%M')}"
            if interval_5 in participant_5min_coverage:
                bins_in_30min.append(interval_5)
            temp_time = bin_end

        # Average coverage % across all 5-min bins in this 30-min window
        all_coverage_values = []
        for interval_5 in bins_in_30min:
            daily_vals = participant_5min_coverage[interval_5]
            if daily_vals:
                avg = (sum(daily_vals) / len(daily_vals)) * 100
                all_coverage_values.append(avg)

        if all_coverage_values:
            heatmap_coverage.loc[interval_30, participant] = np.mean(all_coverage_values)

    print(f"  {participant}: {covered_5min_bins}/{total_5min_bins} bins covered "
          f"({coverage_metrics[participant]['coverage_pct']:.1f}%)")

# === Save coverage metrics to CSV ===
coverage_df = pd.DataFrame(coverage_metrics).T
coverage_df.index.name = 'Participant'
coverage_df = coverage_df.reset_index()
csv_path = os.path.join(output_folder, fileName_csv)
coverage_df.to_csv(csv_path, index=False)
print(f"\nCoverage metrics saved to: {csv_path}")

# === Plot Coverage heatmap ===
mask_coverage = heatmap_coverage == 0.0
annot_coverage = heatmap_coverage.round(1).astype(str)
annot_coverage[mask_coverage] = ""

plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_coverage,
    cmap="viridis_r",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_coverage,
    fmt='s',
    mask=mask_coverage,
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Data Coverage (%)'}
)
plt.title('Mocopi Data Coverage Heatmap (5-min bin resolution, 30-min display)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Time Interval", fontsize=12)

output_file_coverage = os.path.join(output_folder, fileName_coverage)
plt.savefig(output_file_coverage, dpi=300, bbox_inches='tight')
plt.close()
print(f"Coverage heatmap saved to: {output_file_coverage}")

# === Print summary statistics ===
print("\n=== Coverage Summary ===")
print(coverage_df.to_string(index=False))
print(f"\nMean coverage:   {coverage_df['coverage_pct'].mean():.1f}%")
print(f"Median coverage: {coverage_df['coverage_pct'].median():.1f}%")
print(f"Min coverage:    {coverage_df['coverage_pct'].min():.1f}%")
print(f"Max coverage:    {coverage_df['coverage_pct'].max():.1f}%")