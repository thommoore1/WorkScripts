import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from datetime import datetime

# ============ CONFIGURATION ============
root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/SkeletonPlots")
output_filename = "mocopi_skeleton_datapoints_all_participants.png"
joint_columns = ["head", "hip", "left_wrist", "right_wrist", "left_ankle", "right_ankle"]
os.makedirs(output_folder, exist_ok=True)

print(f"Root path: {root_path}")
print(f"Output folder: {output_folder}")

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
            print(f"⚠️ Skipping file (unknown joint): {file}")
            continue

        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"❌ Error reading {file}: {e}")
            continue

        num_points = len(df)  # number of rows = number of recorded data points
        all_data.append({
            "participant": participant_number,
            "joint": joint,
            "value": num_points
        })
        print(f"   ✅ {participant_number} - {joint}: {num_points} data points")

if not all_data:
    print("❌ No mocopi data found.")
    exit()

combined_df = pd.DataFrame(all_data)
print(f"✅ Combined dataframe shape: {combined_df.shape}")
print(combined_df.head())

# ============ COMPUTE TOTAL DATA POINTS PER JOINT ============
joint_totals = combined_df.groupby("joint")["value"].sum().to_dict()
print("✅ Total data points per joint (across all participants):")
for joint, val in joint_totals.items():
    print(f"   {joint}: {val}")

# ============ SKELETON STRUCTURE ============
coords = {
    "head": (0, 10),
    "neck": (0, 9),
    "left_shoulder": (-1.5, 9),
    "right_shoulder": (1.5, 9),
    "left_elbow": (-2.5, 7.5),
    "right_elbow": (2.5, 7.5),
    "left_wrist": (-3, 6),
    "right_wrist": (3, 6),
    "spine": (0, 7),
    "hip": (0, 5),
    "left_knee": (-1, 2.5),
    "right_knee": (1, 2.5),
    "left_ankle": (-1, 0),
    "right_ankle": (1, 0)
}

connections = [
    ("head", "neck"),
    ("neck", "left_shoulder"), ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
    ("neck", "spine"), ("spine", "hip"),
    ("hip", "left_knee"), ("hip", "right_knee"),
    ("left_knee", "left_ankle"), ("right_knee", "right_ankle")
]

# ============ PLOT FUNCTION ============
def plot_skeleton(ax, joint_values):
    print("🎨 Plotting combined skeleton...")
    max_val = max(joint_values.values()) if joint_values else 1

    # Draw bones
    for a, b in connections:
        if a in coords and b in coords:
            x = [coords[a][0], coords[b][0]]
            y = [coords[a][1], coords[b][1]]
            ax.plot(x, y, color='black', lw=2, zorder=1)



    # Draw joints
    for joint, (x, y) in coords.items():
        val = joint_values.get(joint, 0)
        size = (np.sqrt(val / max_val)) * 3000  # nonlinear scaling
        color = 'red' if joint in joint_values else 'gray'
        ax.scatter(x, y, s=size, color=color, alpha=0.8, zorder=3)


    # Label joints
    for joint, val in joint_values.items():
        if joint in coords:
            x, y = coords[joint]
            ax.text(
                x, y + 0.6, f"{joint}\n{val}",
                ha='center', va='bottom',
                fontsize=9, color='darkred',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),  # keeps text readable
                zorder=5  # draw above circles and lines
        )


    ax.set_title("Total Data Points per Joint (All Participants)", fontsize=14)
    ax.axis('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1, 11) 
    ax.axis('off')

# ============ PLOT AND SAVE ============
fig, ax = plt.subplots(figsize=(6, 10))
plot_skeleton(ax, joint_totals)
fig.tight_layout()
fig.set_facecolor('white')

output_path = os.path.join(output_folder, output_filename)
plt.savefig(output_path, bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"✅ Saved skeleton plot to: {output_path}")
