import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/HeatMaps/Mocopi/DataPoints")
timestamp_column = "time"
activity_column = "class"
output_filename = "Mocopi_DP_Class.png"

os.makedirs(output_folder, exist_ok=True)

# Get all participant folders
participant_folders = [
    f for f in os.listdir(root_path)
    if f.startswith("P") and os.path.isdir(os.path.join(root_path, f))
]

all_data = []

for participant in participant_folders:
    participant_number = participant

    heart_rate_folder = os.path.join(root_path, participant, "Mocopi", "Labeled")

    csv_files = [
    os.path.join(root, f)
    for root, dirs, files in os.walk(heart_rate_folder)
    for f in files
    if f.endswith('.csv')
    ]

    for file in csv_files:
        # Extract the date string from the filename
        date_str = file[-14:-4]  # Assumes format: YYYY-MM-DD.csv

        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            # If the date can't be parsed, skip the file
            continue

        # Check if the date is a Friday (weekday() == 4 means Friday)
        if file_date.weekday() == 4:
            continue

        file_path = os.path.join(heart_rate_folder, file)
        df = pd.read_csv(file_path)

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

heatmap_data_masked = heatmap_data.replace(0, np.nan)

# Create annotation DataFrame with "Null" for NaN
annot_data = heatmap_data.copy()
annot_data = annot_data.applymap(lambda x: "Null" if pd.isna(x) else f"{int(x)}")

# Plot
plt.figure(figsize=(len(heatmap_data.columns) * 0.8, len(heatmap_data.index) * 0.5))
ax = sns.heatmap(
    heatmap_data_masked,
    cmap='viridis_r',
    linewidths=0.5,
    linecolor='gray',
    cbar=True,
    square=False,
    annot=annot_data,
    fmt="",
    annot_kws={"fontsize": 7},
    mask=heatmap_data_masked.isna(),
    vmin=heatmap_data.min().min(),
    vmax=heatmap_data.max().max()
)

plt.xlabel('Participant', color='black')
plt.ylabel('', color='black')
ax.set_ylim(len(heatmap_data.index) + 0.5, -0.5)
ax.set_xlim(-0.5, len(heatmap_data.columns) + 0.5)


# Save image
output_path = os.path.join(output_folder, output_filename)
plt.gcf().set_facecolor('white')
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved activity heatmap to: {output_path}")
