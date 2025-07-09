import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "Heatmaps")
timestamp_column = "time"
activity_column = "class"
output_filename = "participant_activity_heatmap.png"

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
        if timestamp_column not in df.columns or activity_column not in df.columns:
            continue

        df['participant'] = participant_number
        all_data.append(df[[activity_column, 'participant']])

# Combine all data
if not all_data:
    print("No labeled CSV files with activities found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Count rows per activity per participant
counts = combined_df.groupby(['participant', activity_column]).size().reset_index(name='count')

# Pivot: rows = activity, columns = participant
heatmap_data = counts.pivot_table(index=activity_column, columns='participant', values='count', fill_value=0)

# Plot
plt.figure(figsize=(len(heatmap_data.columns) * 0.8, len(heatmap_data.index) * 0.5))
sns.heatmap(
    heatmap_data,
    cmap='viridis',  # higher contrast colormap
    linewidths=0.5,
    linecolor='gray',
    cbar=True,
    square=False,
    annot=False
)

plt.title('Oura Ring Heart Rate heatmap', color='black', fontsize=14)
plt.xlabel('Participant', color='black')
plt.ylabel('Activity', color='black')

# Save image
output_path = os.path.join(output_folder, output_filename)
plt.gcf().set_facecolor('white')  # set figure background white
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved activity heatmap to: {output_path}")