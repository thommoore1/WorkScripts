import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from datetime import datetime

root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "Heatmaps")
timestamp_column = "time"
output_filename = "participant_heatmap.png"

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
        file_path = os.path.join(heart_rate_folder, file)
        df = pd.read_csv(file_path)
        if timestamp_column not in df.columns:
            continue

        # Convert UNIX time to date
        df['date'] = pd.to_datetime(df[timestamp_column], unit='s').dt.date
        df['participant'] = participant_number
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

# Pivot: rows = weekday, columns = participant, values = sum of rows per weekday
heatmap_data = counts.groupby(['participant', 'weekday'])['count'].sum().reset_index()
heatmap_data = heatmap_data.pivot_table(index='weekday', columns='participant', values='count', fill_value=0)

# Plot
plt.figure(figsize=(len(heatmap_data.columns) * 0.8, 4))
sns.heatmap(
    heatmap_data,
    cmap='Greens',
    linewidths=0.5,
    linecolor='gray',
    cbar=True
)

# Format y-axis
plt.yticks(
    ticks=[0.5 + i for i in range(7)],
    labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    rotation=0
)

plt.title('Participant Activity Heatmap (Rows per Day)')
plt.xlabel('Participant')
plt.ylabel('')

# Save image to the Heatmaps folder
output_path = os.path.join(output_folder, output_filename)
plt.tight_layout()
plt.savefig(output_path)
print(f"✅ Saved heatmap to: {output_path}")