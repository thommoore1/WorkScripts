import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/BoxPlots")
timestamp_column = "time"
activity_column = "class"
heart_rate_column = "bpm"
output_filename = "HR by Participant.png"

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

    csv_files = [
        f for f in os.listdir(heart_rate_folder)
        if f.endswith(".csv") and "RAW" not in f
    ]

    for file in csv_files:
        # Extract date string
        date_str = file[-14:-4]  # Assumes format: YYYY-MM-DD.csv
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        # Skip Fridays
        if file_date.weekday() == 4:
            continue

        file_path = os.path.join(heart_rate_folder, file)
        df = pd.read_csv(file_path)

        df['class'] = df['class'].replace('Homework Reinforcement/Study Hall', 'HW Reinfor')

        df['participant'] = participant_number
        all_data.append(df[[activity_column, heart_rate_column, 'participant']])


# Combine all data
if not all_data:
    print("No labeled CSV files with activities found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# --- Compute global bounds for consistent y-axis ---
y_min = combined_df[heart_rate_column].min()
y_max = combined_df[heart_rate_column].max()

# --- Replace participant palette with activity palette ---
activities = sorted(combined_df[activity_column].unique())
palette = sns.color_palette("Set3", len(activities))
activity_palette = dict(zip(activities, palette))

# --- FACETED BOX PLOTS (per participant) ---
participants = sorted(combined_df['participant'].unique())
num = len(participants)
cols = 4
rows = int(np.ceil(num / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = axes.flatten()

for i, participant in enumerate(participants):
    ax = axes[i]
    subset = combined_df[combined_df['participant'] == participant]

    sns.boxplot(
        data=subset,
        x=activity_column,
        y=heart_rate_column,
        ax=ax,
        palette=activity_palette,
        order=activities
    )

    ax.set_title(f"Participant {participant}")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), ha='right')


    # --- Set consistent y-axis limits ---
    ax.set_ylim(y_min, y_max)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# Save image
output_path = os.path.join(output_folder, output_filename)
fig.set_facecolor('white')
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved activity box plot to: {output_path}")