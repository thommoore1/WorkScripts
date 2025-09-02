import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

rootPath = "/Users/tommoore/Documents/GitHub/Research"
outputFolder = os.path.join(rootPath, "Heatmaps/SensorLogger/All")
timestampColumn = "time"
activityColumn = "class"
heartRateColumn = "seconds_elapsed"
outputFilename = "SensLogActivityHeatMapAll.png"

os.makedirs(outputFolder, exist_ok=True)

# Get all participant folders
participant_folders = [
    f for f in os.listdir(rootPath)
    if f.startswith("P") and os.path.isdir(os.path.join(rootPath, f))
]

all_data = []

for participant in participant_folders:
    participant_number = participant

    sensorLoggerFolder = os.path.join(rootPath, participant, "SensorLogger")

    csv_files = []
    for root, dirs, files in os.walk(sensorLoggerFolder):
        for f in files:
            if f.endswith(".csv") and "TRUE" in f:
                csv_files.append(os.path.join(root, f))


    for file in csv_files:
        dateStr = file[-14:-4]
        normalized = dateStr.replace("_", "-")
        

        try:
            fileDate = datetime.strptime(normalized, "%Y-%m-%d")
        except ValueError:
            # If the date can't be parsed, skip the file
            continue

        # Check if the date is a Friday (weekday() == 4 means Friday)
        if fileDate.weekday() == 4:
            continue

        filePath = os.path.join(heart_rate_folder, file)
        df = pd.read_csv(filePath)

        df['participant'] = participant_number
        all_data.append(df[[activity_column, 'participant']])

# Combine all data
if not all_data:
    print("No labeled CSV files with activities found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Count number of data points per activity per participant
counts = combined_df.groupby(['participant', activity_column]).size().reset_index(name='count')

# Pivot: rows = activity, columns = participant
heatmap_data = counts.pivot_table(index=activity_column, columns='participant', values='count', fill_value=0)

# Replace zeros with NaN if you want to show them as empty
heatmap_data.replace(0, np.nan, inplace=True)

# Sort rows by count of non-NaN values (descending)
heatmap_data['non_nan_count'] = heatmap_data.notna().sum(axis=1)
heatmap_data = heatmap_data.sort_values(by='non_nan_count', ascending=False)
heatmap_data = heatmap_data.drop(columns='non_nan_count')

# Create annotation DataFrame with "Null" for NaN
annot_data = heatmap_data.copy()
annot_data = annot_data.applymap(lambda x: "Null" if pd.isna(x) else f"{int(x)}")

# Plot
plt.figure(figsize=(len(heatmap_data.columns) * 0.8, len(heatmap_data.index) * 0.5))
ax = sns.heatmap(
    heatmap_data,
    cmap='viridis_r',
    linewidths=0.5,
    linecolor='gray',
    cbar=True,
    square=False,
    annot=annot_data,
    fmt="",
    vmin=heatmap_data.min().min(),
    vmax=650
)

plt.title('Number of Data Points per Activity per Participant', color='black', fontsize=14)
plt.xlabel('Participant', color='black')
plt.ylabel('Activity', color='black')
ax.set_ylim(len(heatmap_data.index) + 0.5, -0.5)
ax.set_xlim(-0.5, len(heatmap_data.columns) + 0.5)


# Save image
output_path = os.path.join(output_folder, output_filename)
plt.gcf().set_facecolor('white')
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved activity heatmap to: {output_path}")
