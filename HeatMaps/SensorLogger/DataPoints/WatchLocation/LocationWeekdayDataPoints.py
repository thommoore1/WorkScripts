import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime

rootPath = "/Users/tommoore/Documents/GitHub/Research"
outputFolder = os.path.join(rootPath, "Heatmaps/SensorLogger/WatchLocation")
timestampColumn = "time"
outputFilename = "WatchLocationWeekdayDataPoints.png"
outputPath = os.path.join(outputFolder, outputFilename)

# Create Heatmaps folder if it doesn't exist
os.makedirs(outputFolder, exist_ok=True)

# Get all participant folders
participantFolders = [
    f for f in os.listdir(rootPath)
    if f.startswith("P") and os.path.isdir(os.path.join(rootPath, f))
]

all_data = []

for participant in participantFolders:
    participantNumber = participant

    sensorLoggerFolder = os.path.join(rootPath, participant, "SensorLogger")
    if not os.path.exists(sensorLoggerFolder):
        continue

    csv_files = []
    for root, dirs, files in os.walk(sensorLoggerFolder):
        for f in files:
            if f.endswith(".csv") and "TRUE" in f and "WatchLocation" in f:
                csv_files.append(os.path.join(root, f))

    for file in csv_files:
        # Extract the date string from the filename
        date_str = file[-14:-4]  # Assumes format: YYYY-MM-DD.csv

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

        # Check if the date is a Friday (weekday() == 4 means Friday)
        if fileDate.weekday() == 4:
            continue

        file_path = os.path.join(sensorLoggerFolder, file)
        df = pd.read_csv(file_path)

        file_path = os.path.join(sensorLoggerFolder, file)
        df = pd.read_csv(file_path)
        if timestampColumn not in df.columns:
            continue

        # Convert UNIX time to date
        df['date'] = pd.to_datetime(df[timestampColumn], unit='ns').dt.date
        df['participant'] = participantNumber
        all_data.append(df[['date', 'participant']])

# Combine all data
if not all_data:
    print("No labeled CSV files found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Count rows per date per participant
counts = combined_df.groupby(['date', 'participant']).size().reset_index(name='count')

# Add weekday info
counts['date'] = pd.to_datetime(counts['date'])
counts['weekday'] = counts['date'].dt.weekday  # Monday = 0

# ✅ Filter for Monday (0) to Friday (4) only
counts = counts[counts['weekday'].isin([0, 1, 2, 3])]

# Pivot: rows = weekday, columns = participant, values = sum of rows per weekday
heatmap_data = counts.groupby(['participant', 'weekday'])['count'].sum().reset_index()
heatmap_data = heatmap_data.pivot_table(index='weekday', columns='participant', values='count', fill_value=0)

# Replace zeros with NaN for white background
heatmap_data_masked = heatmap_data.replace(0, np.nan)

# Create annotation data: no text for NaN cells
annot_data = heatmap_data_masked.copy()
annot_data = annot_data.applymap(lambda x: "" if pd.isna(x) else int(x))

# Plot
plt.figure(figsize=(len(heatmap_data.columns) * 0.8, 4))
ax = sns.heatmap(
    heatmap_data_masked,
    cmap='viridis_r',
    linewidths=0.5,
    linecolor='gray',
    cbar=True,
    square=False,
    annot=annot_data,
    fmt="",
    #annot_kws={"size": 8, "color": "black"},
    mask=heatmap_data_masked.isna(),
    vmin=5,
    vmax=100
)

# Fix missing grid lines on edges
ax.set_xlim(-0.5, len(heatmap_data_masked.columns) + 0.5)
ax.set_ylim(len(heatmap_data_masked.index) + 0.5, -0.5)

# Format y-axis: only Monday–Friday
plt.yticks(
    ticks=[0.5 + i for i in range(4)],
    labels=['Mon', 'Tue', 'Wed', 'Thu'],
    rotation=0
)

plt.title('Participant Activity Heatmap (Rows per Day)')
plt.xlabel('Participant')
plt.ylabel('')

# Save image to the Heatmaps folder
plt.tight_layout()
plt.savefig(outputPath)
print(f"Saved heatmap to: {outputPath}")