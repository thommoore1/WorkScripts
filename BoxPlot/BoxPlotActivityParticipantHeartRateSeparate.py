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
output_filename = "participant_activity_heartRate_separate.png"

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

# Count number of data points per activity per participant
counts = combined_df.groupby(['participant', activity_column]).size().reset_index(name='count')

# --- BOX PLOT ---
plt.figure(figsize=(14, 7))

# High-contrast palette for participants
num_participants = combined_df['participant'].nunique()
palette = sns.color_palette("tab20", num_participants)

# Create a boxplot of counts per participant/activity
# Using the raw combined_df instead of aggregated counts
g = sns.catplot(
    data=combined_df,
    x=activity_column,
    y=heart_rate_column,
    col="participant",         # separate plot per participant
    kind="box",
    col_wrap=4,                # wrap into rows of 4
    height=4, aspect=1.2
)

g.set_titles("Participant {col_name}")
g.set_axis_labels("Activity", "Heart Rate (bpm)")
plt.xticks(rotation=45)
plt.tight_layout()

# Legend
plt.legend(title="Participant", bbox_to_anchor=(1.05, 1), loc="upper left")

# Save image
output_path = os.path.join(output_folder, output_filename)
plt.gcf().set_facecolor('white')
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved activity box plot to: {output_path}")
