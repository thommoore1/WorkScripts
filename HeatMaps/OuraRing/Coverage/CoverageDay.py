import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# === Paths ===
root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/OuraRing/Coverage")
fileName_hr = "day_data.png"
fileName_coverage = "day_coverage.png"
fileName_csv = "day_coverage_metrics.csv"
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

# === Define weekdays (Monday=0 to Friday=4) ===
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# === Define 5-min bins ===
start_time = datetime.strptime("08:30", "%H:%M")
end_time = datetime.strptime("15:00", "%H:%M")
time_bins_5min = []
temp_time = start_time
while temp_time < end_time:
    bin_end = temp_time + timedelta(minutes=5)
    time_bins_5min.append((temp_time.time(), bin_end.time()))
    temp_time = bin_end

# === Initialize dataframes ===
# HR heatmap (by weekday)
heatmap_hr = pd.DataFrame(
    0.0,
    index=weekday_names,
    columns=participant_folders
)

# Coverage heatmap (by weekday)
heatmap_coverage = pd.DataFrame(
    0.0,
    index=weekday_names,
    columns=participant_folders
)

# Overall coverage metrics (per participant, aggregated across weekdays)
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

    # Track coverage and HR by weekday
    participant_weekday_5min_coverage = {day: {f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}": []
                                                for start, end in time_bins_5min}
                                         for day in weekday_names}
    participant_weekday_hr = {day: [] for day in weekday_names}

    for file in os.listdir(hr_path):
        if file.endswith(".csv") and "RAW" not in file:
            try:
                date_str = file[-14:-4]
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue

            # Only process Monday-Friday (0-4)
            weekday = file_date.weekday()
            if weekday > 4:  # Skip weekends
                continue
            
            weekday_name = weekday_names[weekday]

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
                participant_weekday_5min_coverage[weekday_name][interval].append(1 if has_valid_sample else 0)
            
            # === Calculate HR averages for the entire day ===
            valid_hr_df = df[df['TimeObj'].between(start_time.time(), end_time.time()) & df['valid_hr']]
            if not valid_hr_df.empty:
                mean_bpm = valid_hr_df['bpm'].mean()
                participant_weekday_hr[weekday_name].append(mean_bpm)
    
    # === Aggregate coverage by weekday ===
    total_5min_bins_all_days = 0
    covered_5min_bins_all_days = 0
    
    for weekday_name in weekday_names:
        # For each 5-min bin on this weekday, calculate % of days that had coverage
        bins_for_this_weekday = []
        
        for interval, coverage_list in participant_weekday_5min_coverage[weekday_name].items():
            if coverage_list:  # If we have data for this bin
                coverage_pct = (sum(coverage_list) / len(coverage_list)) * 100
                bins_for_this_weekday.append(coverage_pct)
                
                # For overall metrics across all weekdays
                total_5min_bins_all_days += 1
                if sum(coverage_list) > 0:
                    covered_5min_bins_all_days += 1
        
        # Average coverage across all 5-min bins for this weekday
        if bins_for_this_weekday:
            heatmap_coverage.loc[weekday_name, participant] = np.mean(bins_for_this_weekday)
        
        # Average HR for this weekday
        hr_list = participant_weekday_hr[weekday_name]
        if hr_list:
            heatmap_hr.loc[weekday_name, participant] = np.mean(hr_list)
    
    # Store overall coverage metrics (aggregated across all weekdays)
    coverage_metrics[participant]['total_bins'] = total_5min_bins_all_days
    coverage_metrics[participant]['covered_bins'] = covered_5min_bins_all_days
    coverage_metrics[participant]['coverage_pct'] = (covered_5min_bins_all_days / total_5min_bins_all_days * 100) if total_5min_bins_all_days > 0 else 0

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

plt.figure(figsize=(14, 6))
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
plt.title('Heart Rate Heatmap by Weekday (8:30 AM - 3:00 PM)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Day of Week", fontsize=12)

output_file_hr = os.path.join(output_folder, fileName_hr)
plt.savefig(output_file_hr, dpi=300, bbox_inches='tight')
plt.close()
print(f"HR heatmap saved to: {output_file_hr}")

# === Plot Coverage heatmap ===
mask_coverage = heatmap_coverage == 0.0
annot_coverage = heatmap_coverage.round(1).astype(str)
annot_coverage[mask_coverage] = ""

plt.figure(figsize=(14, 6))
sns.heatmap(
    heatmap_coverage,
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_coverage,
    fmt='s',
    mask=mask_coverage,
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Data Coverage (%)'}
)
plt.title('Data Coverage Heatmap by Weekday (5-min bin resolution)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Day of Week", fontsize=12)

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