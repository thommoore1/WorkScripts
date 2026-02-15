import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from matplotlib.colors import to_hex
import colorsys

root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/BoxPlots")
activity_column = "class"
heart_rate_column = "bpm"
output_filename = "HR per activity per participant.png"

os.makedirs(output_folder, exist_ok=True)

# --- Function to create gradient colors with controlled lightness ---
def create_gradient(base_color, n_colors=5, min_lightness=0.6, max_lightness=0.9):
    r, g, b = base_color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    gradient = []
    for i in range(n_colors):
        li = min_lightness + (max_lightness - min_lightness) * (i / (n_colors - 1))
        ri, gi, bi = colorsys.hls_to_rgb(h, li, s)
        gradient.append((ri, gi, bi))
    return gradient

# --- Load all participant data ---
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

if not all_data:
    print("No labeled CSV files with activities found.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# --- Compute global y-axis bounds ---
y_min = combined_df[heart_rate_column].min()
y_max = combined_df[heart_rate_column].max()

participants = sorted(combined_df['participant'].unique())
activities = sorted(combined_df[activity_column].unique())

# --- FACETED BOX PLOTS (per activity) ---
num = len(activities)
cols = 3
rows = int(np.ceil(num / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axes = axes.flatten()

# Base colors per activity
base_colors = sns.color_palette("Set2", len(activities))

for i, activity in enumerate(activities):
    ax = axes[i]
    subset = combined_df[combined_df[activity_column] == activity]

    # Gradient across participants
    gradient_palette = create_gradient(base_colors[i], n_colors=len(participants), min_lightness=0.6, max_lightness=0.9)
    activity_color_map = dict(zip(participants, [to_hex(c) for c in gradient_palette]))

    # Draw boxplots per participant
    for j, participant in enumerate(participants):
        participant_data = subset[subset['participant'] == participant]
        sns.boxplot(
            x=[participant]*len(participant_data),
            y=heart_rate_column,
            data=participant_data,
            ax=ax,
            color=activity_color_map[participant],
            order=participants,
            showfliers=False
        )

    ax.set_title(f"{activity}", fontsize=12)
    ax.set_xlabel("Participant", fontsize=10)
    ax.set_ylabel("Heart Rate (bpm)", fontsize=10)
    ax.set_ylim(y_min, y_max)

    # Rotate x-axis labels for readability
    ax.tick_params(axis='x', rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, ha='right')


# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
fig.set_facecolor('white')

# Save figure
output_path = os.path.join(output_folder, output_filename)
plt.savefig(output_path, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Saved activity-faceted gradient box plots to: {output_path}")
