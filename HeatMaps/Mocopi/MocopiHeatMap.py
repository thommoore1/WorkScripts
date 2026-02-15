import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime

# ============ CONFIGURATION ============
root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/Heatmaps")
output_filename = "mocopi_activity_participant_heatmap.png"

os.makedirs(output_folder, exist_ok=True)

# ============ FILE NAME → JOINT MAPPING ============
def extract_joint_from_filename(filename):
    """Return standardized joint name from mocopi filename."""
    name = filename.lower()
    if "head" in name:
        return "head"
    elif "hip" in name:
        return "hip"
    elif "anklel" in name or "ankle_l" in name:
        return "left_ankle"
    elif "ankler" in name or "ankle_r" in name:
        return "right_ankle"
    elif "wristl" in name or "wrist_l" in name:
        return "left_wrist"
    elif "wristr" in name or "wrist_r" in name:
        return "right_wrist"
    else:
        return None


def extract_date_from_filename(filename):
    """Extract date from Mocopi filename (format ...YYYY-MM-DD.csv)."""
    basename = os.path.basename(filename)
    date_part = basename[-14:-4]  # last 10 chars before .csv
    try:
        return datetime.strptime(date_part, "%Y-%m-%d")
    except ValueError:
        return None


# ============ DATA LOADING ============
participant_folders = [
    f for f in os.listdir(root_path)
    if f.startswith("P") and os.path.isdir(os.path.join(root_path, f))
]

print(f"Found participant folders: {participant_folders}")

all_data = []

for participant in participant_folders:
    participant_number = participant
    mocopi_folder = os.path.join(root_path, participant, "Mocopi", "Labeled")
    if not os.path.exists(mocopi_folder):
        print(f"⚠️ Skipping {participant_number}: Mocopi folder not found.")
        continue

    csv_files = glob.glob(os.path.join(mocopi_folder, "**", "*.csv"), recursive=True)
    print(f"  {participant_number}: Found {len(csv_files)} CSV files.")

    for file in csv_files:
        joint = extract_joint_from_filename(os.path.basename(file))
        if joint is None:
            continue

        # Extract and check date — skip if Friday
        file_date = extract_date_from_filename(file)
        if file_date is None:
            print(f"⚠️ Skipping {file}: could not extract date.")
            continue
        if file_date.weekday() == 4:  # 4 = Friday
            print(f"⏭️ Skipping Friday file: {file}")
            continue

        # Read data
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            continue

        if "class" not in df.columns:
            print(f"⚠️ Skipping {file}: no 'class' column found.")
            continue

        # Count rows per activity within this file
        activity_counts = df["class"].value_counts().to_dict()

        for activity, count in activity_counts.items():
            all_data.append({
                "participant": participant_number,
                "activity": activity,
                "joint": joint,
                "value": count
            })

# ============ COMBINE AND SUMMARIZE ============
if not all_data:
    print("❌ No mocopi data found.")
    exit()

combined_df = pd.DataFrame(all_data)
print(f"✅ Combined dataframe shape: {combined_df.shape}")

# Sum across all joints for each participant + activity
summary_df = combined_df.groupby(["participant", "activity"])["value"].sum().reset_index()
print(summary_df.head())

# ============ PIVOT INTO HEATMAP TABLE ============
heatmap_data = summary_df.pivot_table(
    index="activity",
    columns="participant",
    values="value",
    fill_value=0
)

# Replace 0 with NaN so missing cells appear empty
heatmap_data.replace(0, np.nan, inplace=True)

# Sort activities by total number of data points
heatmap_data["total"] = heatmap_data.sum(axis=1)
heatmap_data = heatmap_data.sort_values(by="total", ascending=False)
heatmap_data = heatmap_data.drop(columns="total")

# Annotations (show count or "Null")
annot_data = heatmap_data.copy()
annot_data = annot_data.applymap(lambda x: "Null" if pd.isna(x) else f"{int(x)}")

# ============ PLOT ============
plt.figure(figsize=(len(heatmap_data.columns) * 0.8, len(heatmap_data.index) * 0.5))
ax = sns.heatmap(
    heatmap_data,
    cmap='viridis_r',
    linewidths=0.5,
    linecolor='gray',
    cbar=True,
    square=False,
    annot=annot_data,
    fmt="",
    vmin=heatmap_data.min().min(),
    vmax=heatmap_data.max().max()
)

plt.title('Number of Mocopi Data Points per Activity per Participant (Fridays Skipped)', fontsize=14, color='black')
plt.xlabel('Participant', color='black')
plt.ylabel('Activity', color='black')
ax.set_ylim(len(heatmap_data.index) + 0.5, -0.5)
ax.set_xlim(-0.5, len(heatmap_data.columns) + 0.5)

plt.gcf().set_facecolor('white')
plt.tight_layout()

# ============ SAVE ============
output_path = os.path.join(output_folder, output_filename)
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Saved heatmap to: {output_path}")
