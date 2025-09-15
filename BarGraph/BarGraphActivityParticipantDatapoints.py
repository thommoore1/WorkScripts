import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "BarGraphs")
timestamp_column = "time"
activity_column = "class"
heart_rate_column = "bpm"
output_filename = "participant_activity_bargraph.png"

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

        df['participant'] = participant_number
        all_data.append(df[[activity_column, 'participant']])

# Combine all data
if not all_data:
    print("No labeled CSV files with activities found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# Count number of data points per activity per participant
counts = combined_df.groupby(['participant', activity_column]).size().reset_index(name='count')

# --- BAR GRAPH ---
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=counts,
    x="participant",
    y="count",
    hue=activity_column,
    estimator=sum,     # Sum counts within each group
    dodge=True
)

plt.title("Number of Data Points per Participant per Activity", fontsize=14)
plt.xlabel("Participant")
plt.ylabel("Number of Data Points")
plt.xticks(rotation=45)
plt.legend(title="Activity", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.ylim(0, 600)  # focus on the range of interest


# Save image
output_path = os.path.join(output_folder, output_filename)
plt.gcf().set_facecolor('white')
plt.tight_layout()
plt.savefig(output_path)
print(f"Saved activity bar graph to: {output_path}")
