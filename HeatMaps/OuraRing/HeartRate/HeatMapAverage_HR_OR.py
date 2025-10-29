import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/OuraRing")
timestamp_column = "time"
activity_column = "class"
heart_rate_column = "bpm"
output_filename = "fig6_hr_activity.png"

os.makedirs(output_folder, exist_ok=True)

# Get all participant folders
participant_folders = [
    f for f in os.listdir(root_path)
    if f.startswith("P") and os.path.isdir(os.path.join(root_path, f))
]

all_data = []

for participant in participant_folders:
    participant_number = participant

    heart_rate_folder = os.path.join(root_path, participant, "OuraRing", "HeartRate")
    if not os.path.exists(heart_rate_folder):
        continue

    csv_files = [
        f for f in os.listdir(heart_rate_folder)
        if f.endswith(".csv") and "RAW" not in f
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

        if timestamp_column not in df.columns or activity_column not in df.columns or heart_rate_column not in df.columns:
            continue

        file_path = os.path.join(heart_rate_folder, file)
        df = pd.read_csv(file_path)
        if timestamp_column not in df.columns or activity_column not in df.columns or heart_rate_column not in df.columns:
            continue

        df['participant'] = participant_number
        all_data.append(df[[activity_column, 'participant', heart_rate_column]])

# Combine all data
if not all_data:
    print("No labeled CSV files with activities and heart rate found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Calculate average heart rate per activity per participant
avg_hr = combined_df.groupby(['participant', activity_column])[heart_rate_column].mean().reset_index(name='avg_heart_rate')

# Pivot: rows = activity, columns = participant
heatmap_data = avg_hr.pivot_table(index=activity_column, columns='participant', values='avg_heart_rate', fill_value=0)

# Replace zeros with NaN
heatmap_data.replace(0, np.nan, inplace=True)

# Sort rows by count of non-NaN values (descending)
heatmap_data['non_nan_count'] = heatmap_data.notna().sum(axis=1)
heatmap_data = heatmap_data.sort_values(by='non_nan_count', ascending=False)
heatmap_data = heatmap_data.drop(columns='non_nan_count')

# Create annotation DataFrame with "Null" for NaN
annot_data = heatmap_data.copy()
annot_data = annot_data.applymap(lambda x: "Null" if pd.isna(x) else f"{x:.1f}")

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
    fmt=""
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
print(f"Saved average heart rate heatmap to: {output_path}")