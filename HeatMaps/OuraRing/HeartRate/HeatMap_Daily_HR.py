import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime

root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/OuraRing")
timestamp_column = "time"
bpm_column = "bpm"
output_filename = "participant_avg_heart_rate_heatmap.png"

# Create Heatmaps folder if it doesn't exist
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

        if timestamp_column not in df.columns or bpm_column not in df.columns:
            continue

        # Convert UNIX time to date
        df['date'] = pd.to_datetime(df[timestamp_column], unit='s').dt.date
        df['participant'] = participant_number
        df = df[['date', 'participant', bpm_column]]
        all_data.append(df)

# Combine all data
if not all_data:
    print("No labeled CSV files found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Add weekday info
combined_df['date'] = pd.to_datetime(combined_df['date'])
combined_df['weekday'] = combined_df['date'].dt.weekday  # Monday = 0

# Filter for Monday–Thursday only
combined_df = combined_df[combined_df['weekday'].isin([0, 1, 2, 3])]

# Calculate average BPM per weekday per participant
avg_bpm = combined_df.groupby(['participant', 'weekday'])[bpm_column].mean().reset_index()

# Pivot: rows = weekday, columns = participant, values = average BPM
heatmap_data = avg_bpm.pivot_table(index='weekday', columns='participant', values=bpm_column, fill_value=0)

# Replace zeros with NaN for white background
heatmap_data_masked = heatmap_data.replace(0, np.nan)

# Create annotation data: no text for NaN cells
annot_data = heatmap_data_masked.copy()
annot_data = annot_data.applymap(lambda x: "" if pd.isna(x) else f"{x:.1f}")

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
    annot_kws={"size": 8, "color": "black"},
    mask=heatmap_data_masked.isna(),
)

# Fix missing grid lines on edges
ax.set_xlim(-0.5, len(heatmap_data_masked.columns) + 0.5)
ax.set_ylim(len(heatmap_data_masked.index) + 0.5, -0.5)

# Format y-axis: only Monday–Thursday
plt.yticks(
    ticks=[0.5 + i for i in range(4)],
    labels=['Mon', 'Tue', 'Wed', 'Thu'],
    rotation=0
)

plt.title('Participant Average Heart Rate Heatmap (BPM)')
plt.xlabel('Participant')
plt.ylabel('')

# Save image to the Heatmaps folder
output_path = os.path.join(output_folder, output_filename)
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved heatmap to: {output_path}")
