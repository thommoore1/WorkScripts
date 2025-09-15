import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
root_path = "/Users/tommoore/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "BarGraphs")
output_path = os.path.join(output_folder, "BarGraph_ActivityParticipantDataPoints.png")
os.makedirs(output_folder, exist_ok=True)

# Find participant folders
participant_folders = [
    f for f in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, f)) and f.startswith("P")
]

# Sort numerically
def get_participant_number(name):
    return int(name[1:])  # assumes format "P###"

participant_folders = sorted(participant_folders, key=get_participant_number)

# Collect counts per activity per participant
records = []

for participant in participant_folders:
    participant_path = os.path.join(root_path, participant, "OuraRing", "HeartRate")
    
    # Loop through activity files
    
    print("HELLO\n\n\n\n\n\n")
    for file in os.listdir(participant_path):
        if file.endswith(".csv") and "RAW" not in file:
            print("YOOOO\n\n\n\n\n\n")






            activity_name = os.path.splitext(file)[0]
            
            df = pd.read_csv(os.path.join(participant_path, file))
            
            # Count number of columns (data points)
            count = df.shape[1]
            print("HELP\n\n\n\n\n\n")
            records.append({
                "Participant": participant,
                "Activity": activity_name,
                "Count": count
            })

# Convert to DataFrame
counts_df = pd.DataFrame(records)

print(counts_df.head())
print(counts_df.columns)


# Plot bar graph
plt.figure(figsize=(12, 6))
sns.barplot(
    data=counts_df,
    x="Activity",
    y="Count",
    hue="Participant",
    palette="Spectral"
)

plt.title("Number of Data Points per Participant per Activity", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of Data Points")
plt.xlabel("Activity")
plt.legend(title="Participant", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# Save figure
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Bar graph saved to {output_path}")
