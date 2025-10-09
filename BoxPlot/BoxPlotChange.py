import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# === CONFIGURATION ===
root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/BoxPlots")
timestamp_column = "time"
activity_column = "class"
heart_rate_column = "bpm"
output_filename = "activity_participant_heartRate_colorPerActivity.png"

os.makedirs(output_folder, exist_ok=True)

# === FIND PARTICIPANT FOLDERS ===
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
        # Extract date string (assumes format YYYY-MM-DD.csv)
        date_str = file[-14:-4]
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        # Skip Fridays
        if file_date.weekday() == 4:
            continue

        file_path = os.path.join(heart_rate_folder, file)
        df = pd.read_csv(file_path)

        # Shorten long activity names
        df['class'] = df['class'].replace('Homework Reinforcement/Study Hall', 'HW Reinfor')
        df['participant'] = participant_number

        all_data.append(df[[activity_column, heart_rate_column, 'participant']])

# === COMBINE ALL DATA ===
if not all_data:
    print("No labeled CSV files with activities found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# === SETUP ===
activities = sorted(combined_df[activity_column].unique())
participants = sorted(combined_df['participant'].unique())

# Global Y-limits for consistent comparison
y_min = combined_df[heart_rate_column].min()
y_max = combined_df[heart_rate_column].max()

# One unique color per activity
palette = sns.color_palette("Set2", len(activities))
activity_palette = dict(zip(activities, palette))

# === GRID SETUP ===
num = len(activities)
cols = 4
rows = int(np.ceil(num / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
axes = axes.flatten()

# === PLOT PER ACTIVITY ===
for i, activity in enumerate(activities):
    ax = axes[i]
    subset = combined_df[combined_df[activity_column] == activity]

    # Single consistent color per activity
    color = activity_palette[activity]

    sns.boxplot(
        data=subset,
        x="participant",
        y=heart_rate_column,
        ax=ax,
        color=color,
        order=participants,
        showfliers=False
    )

    ax.set_title(f"{activity}")
    ax.set_xlabel("Participant")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(y_min, y_max)

# Remove unused subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# === LEGEND ===
handles = [plt.Rectangle((0, 0), 1, 1, color=activity_palette[a]) for a in activities]
fig.legend(handles, activities, title="Activity", bbox_to_anchor=(1.05, 1), loc="upper left")

# === SAVE ===
output_path = os.path.join(output_folder, output_filename)
fig.set_facecolor('white')
plt.savefig(output_path, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Saved box plots (per activity, color per activity) to: {output_path}")
