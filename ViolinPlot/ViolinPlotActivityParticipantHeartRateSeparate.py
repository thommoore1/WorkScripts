import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/ViolinPlot")
timestamp_column = "time"
activity_column = "class"
heart_rate_column = "bpm"
output_filename = "HR per activity per participant1.png"

os.makedirs(output_folder, exist_ok=True)

# --- load and combine data (same as before) ---
participant_folders = [
    f for f in os.listdir(root_path)
    if f.startswith("P") and os.path.isdir(os.path.join(root_path, f))
]

all_data = []
for participant in participant_folders:
    participant_number = participant
    heart_rate_folder = os.path.join(root_path, participant, "OuraRing", "HeartRate")
    csv_files = [f for f in os.listdir(heart_rate_folder)
                 if f.endswith(".csv") and "RAW" not in f]

    for file in csv_files:
        date_str = file[-14:-4]
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        if file_date.weekday() == 4:  # skip Fridays
            continue

        file_path = os.path.join(heart_rate_folder, file)
        df = pd.read_csv(file_path)
        df['class'] = df['class'].replace('Homework Reinforcement/Study Hall', 'HW Reinfor')
        df['participant'] = participant_number
        all_data.append(df[[activity_column, heart_rate_column, 'participant']])

if not all_data:
    print("No labeled CSV files with activities found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# --- GRID OF VIOLIN PLOTS IN ONE PNG ---
activities = sorted(combined_df[activity_column].unique())
participants = sorted(combined_df['participant'].unique())

# Color palette for participants
palette = sns.color_palette("Set2", len(participants))
participant_palette = dict(zip(participants, palette))

num = len(activities)
cols = 4
rows = int(np.ceil(num / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axes = axes.flatten()

# Compute global y-axis limits
y_min = combined_df[heart_rate_column].min()
y_max = combined_df[heart_rate_column].max()

for i, activity in enumerate(activities):
    ax = axes[i]
    subset = combined_df[combined_df[activity_column] == activity]

    sns.violinplot(
        data=subset,
        x="participant",
        y=heart_rate_column,
        ax=ax,
        palette=participant_palette,
        order=participants,
        inner="quartile",  # change to None if you want pure violin
        cut=0
    )

    ax.set_title(f"Activity: {activity}")
    ax.set_xlabel("Participant")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.tick_params(axis='x', rotation=45)
    
    # Set consistent y-axis
    ax.set_ylim(y_min, y_max)

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# Save as a single PNG
output_path = os.path.join(output_folder, output_filename)
fig.set_facecolor('white')
plt.savefig(output_path, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Saved combined violin plot PNG to: {output_path}")
