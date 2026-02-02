import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# === Paths ===
root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/OuraRing/Coverage")
fileName_hr = "fig7_hr_hour.png"
fileName_coverage = "fig7_coverage_5min.png"
fileName_csv = "coverage_metrics.csv"
os.makedirs(output_folder, exist_ok=True)

# === Find participant folders ===
participant_folders = [
    f for f in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, f)) and f.startswith('P')
]

# Sort numerically
def get_participant_number(name):
    return int(name[1:])  # assumes format "P###"

participant_folders = sorted(participant_folders, key=get_participant_number)

# === Define time bins ===
# 30-min bins for HR averaging (original)
start_time = datetime.strptime("08:30", "%H:%M")
end_time = datetime.strptime("15:00", "%H:%M")
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
# HR heatmap (30-min bins)
heatmap_hr = pd.DataFrame(
    0.0,
    index=[f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" for start, end in time_bins_30min],
    columns=participant_folders
)

# Coverage heatmap (30-min bins for display, but calculated from 5-min bins)
heatmap_coverage = pd.DataFrame(
    0.0,
    index=[f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" for start, end in time_bins_30min],
    columns=participant_folders
)

# Overall coverage metrics (per participant)
coverage_metrics = {p: {'total_bins': 0, 'covered_bins': 0} for p in participant_folders}

# Safe time parser 
def parse_time(s):
    try:
        return datetime.strptime(s, "%H:%M:%S.%f").time()
    except ValueError:
        return datetime.strptime(s, "%H:%M:%S").time()

# Function to check if HR is valid
def is_valid_hr(bpm):
    if pd.isna(bpm):
        return False
    return 40 <= bpm <= 200

# === Process each participant ===
for participant in participant_folders:
    hr_path = os.path.join(root_path, participant, "OuraRing", "HeartRate")
    if not os.path.exists(hr_path):
        continue

    # Track coverage across all valid days for this participant
    participant_5min_coverage = {f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}": []
                                  for start, end in time_bins_5min}
    participant_30min_hr = {f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}": []
                             for start, end in time_bins_30min}

    for file in os.listdir(hr_path):
        if file.endswith(".csv") and "RAW" not in file:
            try:
                date_str = file[-14:-4]
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue

            # Skip Fridays
            if file_date.weekday() == 4:
                continue

            file_path = os.path.join(hr_path, file)
            df = pd.read_csv(file_path)

            if 'Time_In_PST' not in df.columns or 'bpm' not in df.columns:
                continue

            df['TimeObj'] = df['Time_In_PST'].apply(parse_time)
            
            # Filter for valid HR samples
            df['valid_hr'] = df['bpm'].apply(is_valid_hr)
            
            # === Calculate coverage for 5-min bins ===
            for start, end in time_bins_5min:
                bin_df = df[df['TimeObj'].between(start, end)]
                interval = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                
                # Check if bin has at least 1 valid HR sample
                has_valid_sample = bin_df['valid_hr'].any()
                participant_5min_coverage[interval].append(1 if has_valid_sample else 0)
            
            # === Calculate HR averages for 30-min bins ===
            for start, end in time_bins_30min:
                bin_df = df[df['TimeObj'].between(start, end) & df['valid_hr']]
                interval = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                
                if not bin_df.empty:
                    mean_bpm = bin_df['bpm'].mean()
                    participant_30min_hr[interval].append(mean_bpm)
    
    # === Aggregate coverage across days ===
    # For each 5-min bin, calculate % of days that had coverage
    total_5min_bins = len(time_bins_5min)
    covered_5min_bins = 0
    
    for interval, coverage_list in participant_5min_coverage.items():
        if coverage_list:  # If we have data for this bin
            coverage_pct = (sum(coverage_list) / len(coverage_list)) * 100
            # For overall metrics, count as covered if >0% coverage
            if sum(coverage_list) > 0:
                covered_5min_bins += 1
    
    # Store overall coverage metrics
    coverage_metrics[participant]['total_bins'] = total_5min_bins
    coverage_metrics[participant]['covered_bins'] = covered_5min_bins
    coverage_metrics[participant]['coverage_pct'] = (covered_5min_bins / total_5min_bins * 100) if total_5min_bins > 0 else 0
    
    # === Aggregate coverage into 30-min bins for heatmap ===
    for start_30, end_30 in time_bins_30min:
        interval_30 = f"{start_30.strftime('%H:%M')}-{end_30.strftime('%H:%M')}"
        
        # Find all 5-min bins within this 30-min bin
        bins_in_30min = []
        temp_time = datetime.combine(datetime.today(), start_30)
        end_time_30 = datetime.combine(datetime.today(), end_30)
        
        while temp_time < end_time_30:
            bin_end = temp_time + timedelta(minutes=5)
            interval_5 = f"{temp_time.time().strftime('%H:%M')}-{bin_end.time().strftime('%H:%M')}"
            if interval_5 in participant_5min_coverage:
                bins_in_30min.append(interval_5)
            temp_time = bin_end
        
        # Calculate average coverage across 5-min bins within this 30-min window
        all_coverage_values = []
        for interval_5 in bins_in_30min:
            if participant_5min_coverage[interval_5]:
                avg_coverage = (sum(participant_5min_coverage[interval_5]) / 
                               len(participant_5min_coverage[interval_5])) * 100
                all_coverage_values.append(avg_coverage)
        
        if all_coverage_values:
            heatmap_coverage.loc[interval_30, participant] = np.mean(all_coverage_values)
    
    # === Aggregate HR across days ===
    for interval, hr_list in participant_30min_hr.items():
        if hr_list:
            heatmap_hr.loc[interval, participant] = np.mean(hr_list)

# Force 12:00–12:30 bin for P14 and P16 to zero (both HR and coverage)
interval_to_zero = "12:00-12:30"
for p in ["P014", "P016"]:
    if p in heatmap_hr.columns and interval_to_zero in heatmap_hr.index:
        heatmap_hr.loc[interval_to_zero, p] = 0.0
        heatmap_coverage.loc[interval_to_zero, p] = 0.0

# === Save coverage metrics to CSV ===
coverage_df = pd.DataFrame(coverage_metrics).T
coverage_df.index.name = 'Participant'
coverage_df = coverage_df.reset_index()
csv_path = os.path.join(output_folder, fileName_csv)
coverage_df.to_csv(csv_path, index=False)
print(f"Coverage metrics saved to: {csv_path}")

# === Plot HR heatmap ===
mask_hr = heatmap_hr == 0.0
annot_hr = heatmap_hr.round(1).astype(str)
annot_hr[mask_hr] = ""

plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_hr,
    cmap="viridis_r",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_hr,
    fmt='s',
    mask=mask_hr,
    vmin=75,
    vmax=130,
    cbar_kws={'label': 'Average Heart Rate (BPM)'}
)
plt.title('Heart Rate Heatmap (30-min bins)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Time Interval", fontsize=12)

output_file_hr = os.path.join(output_folder, fileName_hr)
plt.savefig(output_file_hr, dpi=300, bbox_inches='tight')
plt.close()
print(f"HR heatmap saved to: {output_file_hr}")

# === Plot Coverage heatmap ===
mask_coverage = heatmap_coverage == 0.0
annot_coverage = heatmap_coverage.round(1).astype(str)
annot_coverage[mask_coverage] = ""

plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_coverage,
    cmap="YlGnBu",  # Yellow-Green-Blue colormap (0% = light, 100% = dark)
    linewidths=0.5,
    linecolor='gray',
    annot=annot_coverage,
    fmt='s',
    mask=mask_coverage,
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Data Coverage (%)'}
)
plt.title('Data Coverage Heatmap (5-min bin resolution)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Time Interval (30-min display)", fontsize=12)

output_file_coverage = os.path.join(output_folder, fileName_coverage)
plt.savefig(output_file_coverage, dpi=300, bbox_inches='tight')
plt.close()
print(f"Coverage heatmap saved to: {output_file_coverage}")

# === Print summary statistics ===
print("\n=== Coverage Summary ===")
print(f"Overall coverage across all participants:")
print(coverage_df.to_string(index=False))
print(f"\nMean coverage: {coverage_df['coverage_pct'].mean():.1f}%")
print(f"Median coverage: {coverage_df['coverage_pct'].median():.1f}%")
print(f"Min coverage: {coverage_df['coverage_pct'].min():.1f}%")
print(f"Max coverage: {coverage_df['coverage_pct'].max():.1f}%")