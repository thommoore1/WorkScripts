import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Paths
root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "Heatmaps")
os.makedirs(output_folder, exist_ok=True)

# Find participant folders
participant_folders = [
    f for f in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, f)) and f.startswith('P')
]

# Sort numerically
def get_participant_number(name):
    return int(name[1:])  # assumes format "P###"

participant_folders = sorted(participant_folders, key=get_participant_number)

# Define 30-min time bins from 08:30 to 15:00
start_time = datetime.strptime("08:30", "%H:%M")
end_time = datetime.strptime("15:00", "%H:%M")
time_bins = []
while start_time < end_time:
    bin_end = start_time + timedelta(minutes=30)
    time_bins.append((start_time.time(), bin_end.time()))
    start_time = bin_end

# Initialize heatmap dataframe 
heatmap_data = pd.DataFrame(
    0,
    index=[f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" for start, end in time_bins],
    columns=participant_folders
)

# Safe time parser 
def parse_time(s):
    try:
        return datetime.strptime(s, "%H:%M:%S.%f").time()
    except ValueError:
        return datetime.strptime(s, "%H:%M:%S").time()

# Process each participant 
for participant in participant_folders:
    hr_path = os.path.join(root_path, participant, "OuraRing", "HeartRate")
    if not os.path.exists(hr_path):
        continue

    for file in os.listdir(hr_path):
        # Skip if RAW in filename or if date is Friday
        if file.endswith(".csv") and "RAW" not in file:
            try:
                # Extract date substring before .csv (last 14 chars before '.csv' expected as YYYY-MM-DD)
                date_str = file[-14:-4]
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                # If date parsing fails, skip this file
                continue

            # Skip if the date is a Friday (weekday() == 4)
            if file_date.weekday() == 4:
                continue

            file_path = os.path.join(hr_path, file)
            df = pd.read_csv(file_path)

            if 'Time_In_PST' not in df.columns:
                continue

            df['TimeObj'] = df['Time_In_PST'].apply(parse_time)

            for start, end in time_bins:
                count = df[df['TimeObj'].between(start, end)].shape[0]
                interval = f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
                heatmap_data.loc[interval, participant] += count

# Ensure final column order is numeric
heatmap_data = heatmap_data[participant_folders]

# Force 12:00–12:30 bin for P14 and P16 to zero
interval_to_zero = "12:00-12:30"
for p in ["P014", "P016"]:
    if p in heatmap_data.columns and interval_to_zero in heatmap_data.index:
        heatmap_data.loc[interval_to_zero, p] = 0

# Create mask: True for zeros
mask = heatmap_data == 0

# Only show annotations for non-zero cells
annot_data = heatmap_data.copy().astype(str)
annot_data[mask] = ""

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_data,
    cmap="viridis_r",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_data,
    fmt='s',
    mask=mask,
    vmin=0,
    vmax=400,
    cbar_kws={'label': 'Number of Data Points'}
)

plt.title("OuraRing Heart Rate Data Points\nby 30-Min Time Interval and Participant")
plt.xlabel("Participant")
plt.ylabel("Time Interval")

# Save output
output_file = os.path.join(output_folder, "Participant_HeartRate_DataPoint_Heatmap.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"Heatmap saved to: {output_file}")
