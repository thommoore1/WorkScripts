import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Paths
rootPath = "/Users/tommoore/Documents/GitHub/Research"
outputFolder = os.path.join(rootPath, "Heatmaps/SensorLogger/WatchLocation")
output_file = os.path.join(outputFolder, "WatchLocationTimeStampDataPoints.png")
os.makedirs(outputFolder, exist_ok=True)

# Find participant folders
participant_folders = [
    f for f in os.listdir(rootPath)
    if os.path.isdir(os.path.join(rootPath, f)) and f.startswith('P')
]

# Sort numerically
def getParticipantNumber(name):
    return int(name[1:])  # assumes format "P###"

participant_folders = sorted(participant_folders, key=getParticipantNumber)

# Define 30-min time bins from 08:30 to 15:00
startTime = datetime.strptime("08:30", "%H:%M")
endTime = datetime.strptime("15:00", "%H:%M")
timeBins = []
while startTime < endTime:
    binEnd = startTime + timedelta(minutes=30)
    timeBins.append((startTime.time(), binEnd.time()))
    startTime = binEnd

# Initialize heatmap dataframe 
heatmap_data = pd.DataFrame(
    0,
    index=[f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}" for start, end in timeBins],
    columns=participant_folders
)

# Safe time parser 
def parseTime(s):
    try:
        return datetime.strptime(s, "%H:%M:%S.%f").time()
    except ValueError:
        return datetime.strptime(s, "%H:%M:%S").time()

# Process each participant 
for participant in participant_folders:
    sensorLoggerFolder = os.path.join(rootPath, participant, "SensorLogger")
    if not os.path.exists(sensorLoggerFolder):
        continue

    csv_files = []
    for root, dirs, files in os.walk(sensorLoggerFolder):
        for f in files:
            if f.endswith(".csv") and "TRUE" in f and "WatchLocation" in f:
                csv_files.append(os.path.join(root, f))
    for file in csv_files:
        # Skip if RAW in filename or if date is Friday
        if file.endswith(".csv") and "TRUE" in file:
            try:
                # Extract date substring before .csv (last 14 chars before '.csv' expected as YYYY-MM-DD)
                dateStr = file[-14:-4]
                normalized = dateStr.replace("_", "-")
                fileDate = datetime.strptime(normalized, "%Y-%m-%d").date()
            except Exception:
                # If date parsing fails, skip this file
                continue

            # Skip if the date is a Friday (weekday() == 4)
            if fileDate.weekday() == 4:
                continue

            filePath = os.path.join(sensorLoggerFolder, file)
            df = pd.read_csv(filePath)

            if 'Time_In_PST' not in df.columns:
                continue

            df['TimeObj'] = df['Time_In_PST'].apply(parseTime)
            for start, end in timeBins:
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

# Replace zeros with NaN if you want to show them as empty
heatmap_data.replace(0, np.nan, inplace=True)

# Sort rows by count of non-NaN values (descending)
heatmap_data['non_nan_count'] = heatmap_data.notna().sum(axis=1)
heatmap_data = heatmap_data.sort_values(by='non_nan_count', ascending=False)
heatmap_data = heatmap_data.drop(columns='non_nan_count')

# Create mask: True for zeros
mask = heatmap_data == 0

# Only show annotations for non-zero cells
annot_data = heatmap_data.copy()
annot_data = annot_data.applymap(
    lambda x: "Null" if pd.isna(x) else ("" if x == 0 else f"{int(x)}")
)

# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(
    heatmap_data,
    cmap="viridis_r",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_data,
    fmt="",
    vmin=5,
    vmax=850,
    cbar_kws={'label': 'Number of Data Points'}
)

plt.title("OuraRing Heart Rate Data Points\nby 30-Min Time Interval and Participant")
plt.xlabel("Participant")
plt.ylabel("Time Interval")

# Save output
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f" Heatmap saved to: {output_file}")
