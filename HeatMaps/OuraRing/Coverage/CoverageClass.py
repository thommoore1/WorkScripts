import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import numpy as np

# === Paths ===
root_path = "/Users/tommoore/Documents/GitHub/Research"
schedule_path = os.path.join(root_path, "Schedules")
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/OuraRing/Coverage")

fileName_hr = "class_data.png"
fileName_coverage = "class_coverage.png"
fileName_csv = "class_coverage_metrics.csv"
os.makedirs(output_folder, exist_ok=True)

# === Find participant folders ===
participant_folders = [
    f for f in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, f)) and f.startswith('P')
]

# Sort numerically
def get_participant_number(name):
    return int(name[1:])

participant_folders = sorted(participant_folders, key=get_participant_number)

# === Define 5-min bins ===
start_time = datetime.strptime("08:30", "%H:%M")
end_time = datetime.strptime("15:00", "%H:%M")
time_bins_5min = []
temp_time = start_time
while temp_time < end_time:
    bin_end = temp_time + timedelta(minutes=5)
    time_bins_5min.append((temp_time.time(), bin_end.time()))
    temp_time = bin_end

# Safe time parser 
def parse_time(s):
    try:
        return datetime.strptime(s, "%H:%M:%S.%f").time()
    except ValueError:
        try:
            return datetime.strptime(s, "%H:%M:%S").time()
        except ValueError:
            return datetime.strptime(s, "%H:%M").time()

# Function to check if HR is valid
def is_valid_hr(bpm):
    if pd.isna(bpm):
        return False
    return 40 <= bpm <= 200

# === Classes to exclude ===
EXCLUDED_CLASSES = {'DELETE', 'ELA/History', 'Friday Funday'}

# === Load and parse schedule files ===
schedule_files = {
    'schedData_P(01,02,03,06,07,08,12)_FR.csv': {'participants': ['P001', 'P002', 'P003', 'P006', 'P007', 'P008', 'P012'], 'weekdays': [4]},
    'schedData_P(01,02,03,06,07,08,12)_M-TH.csv': {'participants': ['P001', 'P002', 'P003', 'P006', 'P007', 'P008', 'P012'], 'weekdays': [0, 1, 2, 3]},
    'schedData_P(04,05,09,14,16)_FR.csv': {'participants': ['P004', 'P005', 'P009', 'P014', 'P016'], 'weekdays': [4]},
    'schedData_P(04,05,09,14,16)_M-TH.csv': {'participants': ['P004', 'P005', 'P009', 'P014', 'P016'], 'weekdays': [0, 1, 2, 3]},
    'schedData_P(14,16)TU.csv': {'participants': ['P014', 'P016'], 'weekdays': [1]},
}

# First, let's examine the schedule file structure
print("Examining schedule file structure...")
sample_file = os.path.join(schedule_path, 'schedData_P(01,02,03,06,07,08,12)_M-TH.csv')
if os.path.exists(sample_file):
    sample_df = pd.read_csv(sample_file)
    print(f"Columns in schedule file: {sample_df.columns.tolist()}")
    print(f"First few rows:\n{sample_df.head()}")
else:
    print(f"Sample schedule file not found: {sample_file}")

# Initialize schedule data structure
all_classes = set()
participant_schedules = {p: {wd: [] for wd in range(5)} for p in participant_folders}

# Parse schedule files
for sched_file, info in schedule_files.items():
    file_path = os.path.join(schedule_path, sched_file)
    
    if not os.path.exists(file_path):
        print(f"Warning: Schedule file not found: {file_path}")
        continue
    
    print(f"\nProcessing schedule file: {sched_file}")
    df = pd.read_csv(file_path)
    
    # Try to identify time and class columns
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'start' in col.lower() or 'period' in col.lower()]
    
    if len(time_cols) > 0:
        # Assuming each row is a time slot
        for _, row in df.iterrows():
            # Get start time
            start_time_val = None
            end_time_val = None
            
            for col in df.columns:
                if 'start' in col.lower():
                    start_time_val = row[col]
                elif 'end' in col.lower():
                    end_time_val = row[col]
                elif 'time' in col.lower() and start_time_val is None:
                    start_time_val = row[col]
            
            if pd.isna(start_time_val):
                continue
            
            try:
                start_t = parse_time(str(start_time_val))
                
                # Get end time or assume 50 minutes
                if pd.notna(end_time_val):
                    end_t = parse_time(str(end_time_val))
                else:
                    end_t = (datetime.combine(datetime.today(), start_t) + timedelta(minutes=50)).time()
                
                # Now look for class names in remaining columns
                for col in df.columns:
                    if col not in time_cols and 'period' not in col.lower():
                        class_name = row[col]
                        if pd.notna(class_name) and str(class_name).strip():
                            class_name_clean = str(class_name).strip()
                            
                            # Skip excluded classes
                            if class_name_clean in EXCLUDED_CLASSES:
                                continue
                            
                            all_classes.add(class_name_clean)
                            
                            # Add to all participants in this schedule
                            for participant in info['participants']:
                                if participant not in participant_folders:
                                    continue
                                for weekday in info['weekdays']:
                                    participant_schedules[participant][weekday].append((start_t, end_t, class_name_clean))
            
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue

class_names = sorted(list(all_classes))

if not class_names:
    print("No class labels found in schedule files.")
    print("Please check the schedule file format and update the parsing logic.")
    exit()

print(f"\nFound {len(class_names)} unique classes: {class_names}")
print(f"Excluded classes: {EXCLUDED_CLASSES}")

# === Initialize dataframes ===
heatmap_hr = pd.DataFrame(0.0, index=class_names, columns=participant_folders)
heatmap_coverage = pd.DataFrame(0.0, index=class_names, columns=participant_folders)
coverage_metrics = {p: {'total_bins': 0, 'covered_bins': 0} for p in participant_folders}

# === Process each participant ===
for participant in participant_folders:
    print(f"\nProcessing participant: {participant}")
    hr_path = os.path.join(root_path, participant, "OuraRing", "HeartRate")
    if not os.path.exists(hr_path):
        continue

    # Expected bins for each class (from schedule)
    expected_class_bins = {cls: set() for cls in class_names}
    # Actual bins with HR data for each class
    actual_class_bins = {cls: set() for cls in class_names}
    # HR values for averaging
    participant_class_hr = {cls: [] for cls in class_names}

    # Build expected bins from schedule
    for file in os.listdir(hr_path):
        if file.endswith(".csv") and "RAW" not in file:
            try:
                date_str = file[-14:-4]
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue

            weekday = file_date.weekday()
            
            # Skip Fridays for actual data (but we use Friday schedules)
            if weekday == 4:
                continue
            
            if weekday > 4:  # Skip weekends
                continue
            
            # Get schedule for this weekday
            schedule = participant_schedules[participant][weekday]
            
            # Mark expected bins
            for start_t, end_t, class_name in schedule:
                if class_name not in class_names:
                    continue
                
                for bin_start, bin_end in time_bins_5min:
                    # Check if this bin overlaps with the class period
                    if not (bin_end <= start_t or bin_start >= end_t):
                        bin_key = (file_date, f"{bin_start.strftime('%H:%M')}-{bin_end.strftime('%H:%M')}")
                        expected_class_bins[class_name].add(bin_key)
    
    # Now check which expected bins actually have HR data
    for file in os.listdir(hr_path):
        if file.endswith(".csv") and "RAW" not in file:
            try:
                date_str = file[-14:-4]
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue

            weekday = file_date.weekday()
            
            if weekday == 4 or weekday > 4:  # Skip Fridays and weekends
                continue

            file_path = os.path.join(hr_path, file)
            df = pd.read_csv(file_path)

            if 'Time_In_PST' not in df.columns or 'bpm' not in df.columns or 'class' not in df.columns:
                continue

            df['TimeObj'] = df['Time_In_PST'].apply(parse_time)
            df['valid_hr'] = df['bpm'].apply(is_valid_hr)
            
            # Check each 5-min bin
            for bin_start, bin_end in time_bins_5min:
                bin_df = df[df['TimeObj'].between(bin_start, bin_end)]
                
                if bin_df.empty:
                    continue
                
                interval = f"{bin_start.strftime('%H:%M')}-{bin_end.strftime('%H:%M')}"
                bin_key = (file_date, interval)
                
                # Check each class
                for class_name in class_names:
                    # Is this bin expected for this class?
                    if bin_key in expected_class_bins[class_name]:
                        # Does it have HR data for this class?
                        class_bin_df = bin_df[bin_df['class'] == class_name]
                        if not class_bin_df.empty and class_bin_df['valid_hr'].any():
                            actual_class_bins[class_name].add(bin_key)
                            
                            # Collect HR for averaging
                            valid_hr_df = class_bin_df[class_bin_df['valid_hr']]
                            if not valid_hr_df.empty:
                                participant_class_hr[class_name].extend(valid_hr_df['bpm'].tolist())
    
    # Calculate coverage
    total_5min_bins_all_classes = 0
    covered_5min_bins_all_classes = 0
    
    for class_name in class_names:
        total_bins = len(expected_class_bins[class_name])
        covered_bins = len(actual_class_bins[class_name])
        
        if total_bins > 0:
            coverage_pct = (covered_bins / total_bins) * 100
            heatmap_coverage.loc[class_name, participant] = coverage_pct
            print(f"  {class_name}: {covered_bins}/{total_bins} bins = {coverage_pct:.1f}%")
        
        total_5min_bins_all_classes += total_bins
        covered_5min_bins_all_classes += covered_bins
        
        # Average HR
        if participant_class_hr[class_name]:
            heatmap_hr.loc[class_name, participant] = np.mean(participant_class_hr[class_name])
    
    coverage_metrics[participant]['total_bins'] = total_5min_bins_all_classes
    coverage_metrics[participant]['covered_bins'] = covered_5min_bins_all_classes
    coverage_metrics[participant]['coverage_pct'] = (covered_5min_bins_all_classes / total_5min_bins_all_classes * 100) if total_5min_bins_all_classes > 0 else 0

# Sort classes by total coverage
class_coverage_totals = heatmap_coverage.sum(axis=1).sort_values(ascending=False)
class_order = class_coverage_totals.index.tolist()
heatmap_hr = heatmap_hr.loc[class_order]
heatmap_coverage = heatmap_coverage.loc[class_order]

# === Save coverage metrics ===
coverage_df = pd.DataFrame(coverage_metrics).T
coverage_df.index.name = 'Participant'
coverage_df = coverage_df.reset_index()
csv_path = os.path.join(output_folder, fileName_csv)
coverage_df.to_csv(csv_path, index=False)
print(f"\nCoverage metrics saved to: {csv_path}")

# === Plot HR heatmap ===
mask_hr = heatmap_hr == 0.0
annot_hr = heatmap_hr.round(1).astype(str)
annot_hr[mask_hr] = ""

plt.figure(figsize=(14, len(class_names) * 0.5))
sns.heatmap(
    heatmap_hr,
    cmap="viridis_r",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_hr,
    fmt='s',
    mask=mask_hr,
    vmin=75,
    vmax=130,
    cbar_kws={'label': 'Average Heart Rate (BPM)'}
)
plt.title('Heart Rate Heatmap by Class (8:30 AM - 3:00 PM)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Class", fontsize=12)

output_file_hr = os.path.join(output_folder, fileName_hr)
plt.savefig(output_file_hr, dpi=300, bbox_inches='tight')
plt.close()
print(f"HR heatmap saved to: {output_file_hr}")

# === Plot Coverage heatmap ===
mask_coverage = heatmap_coverage == 0.0
annot_coverage = heatmap_coverage.round(1).astype(str)
annot_coverage[mask_coverage] = ""

plt.figure(figsize=(14, len(class_names) * 0.5))
sns.heatmap(
    heatmap_coverage,
    cmap="YlGnBu",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_coverage,
    fmt='s',
    mask=mask_coverage,
    vmin=0,
    vmax=100,
    cbar_kws={'label': 'Data Coverage (%)'}
)
plt.title('Data Coverage Heatmap by Class (5-min bin resolution)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Class", fontsize=12)

output_file_coverage = os.path.join(output_folder, fileName_coverage)
plt.savefig(output_file_coverage, dpi=300, bbox_inches='tight')
plt.close()
print(f"Coverage heatmap saved to: {output_file_coverage}")

# === Print summary statistics ===
print("\n=== Coverage Summary ===")
print(coverage_df.to_string(index=False))
print(f"\nMean coverage: {coverage_df['coverage_pct'].mean():.1f}%")
print(f"Median coverage: {coverage_df['coverage_pct'].median():.1f}%")
print(f"Min coverage: {coverage_df['coverage_pct'].min():.1f}%")
print(f"Max coverage: {coverage_df['coverage_pct'].max():.1f}%")