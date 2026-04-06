import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import numpy as np

# === Paths ===
root_path = "/Users/cibrian/Documents/GitHub/Research"
schedule_path = os.path.join(root_path, "Schedules")
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/Mocopi/Coverage")

fileName_hr = "class_data.png"
fileName_coverage = "class_coverage.png"
fileName_csv = "class_coverage_metrics.csv"
os.makedirs(output_folder, exist_ok=True)

# === Define 5-min bins ===
start_time = datetime.strptime("08:30", "%H:%M")
end_time   = datetime.strptime("15:00", "%H:%M")
time_bins_5min = []
temp_time = start_time
while temp_time < end_time:
    bin_end = temp_time + timedelta(minutes=5)
    time_bins_5min.append((temp_time.time(), bin_end.time()))
    temp_time = bin_end

# === Helpers ===
def parse_time(s):
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time: {s}")

def is_valid_hr(bpm):
    return not pd.isna(bpm) and 40 <= bpm <= 200

def get_participant_number(name):
    return int(name[1:])

# === Classes to exclude ===
EXCLUDED_CLASSES = {'DELETE', 'ELA/History', 'Friday Funday'}

# === Schedule file definitions ===
schedule_files = {
    'schedData_P(01,02,03,06,07,08,12)_FR.csv':   {'participants': ['P001','P002','P003','P006','P007','P008','P012'], 'weekdays': [4]},
    'schedData_P(01,02,03,06,07,08,12)_M-TH.csv': {'participants': ['P001','P002','P003','P006','P007','P008','P012'], 'weekdays': [0,1,2,3]},
    'schedData_P(04,05,09,14,16)_FR.csv':          {'participants': ['P004','P005','P009','P014','P016'], 'weekdays': [4]},
    'schedData_P(04,05,09,14,16)_M-TH.csv':        {'participants': ['P004','P005','P009','P014','P016'], 'weekdays': [0,1,2,3]},
    'schedData_P(14,16)TU.csv':                    {'participants': ['P014','P016'], 'weekdays': [1]},
}

# === PASS 1 — Build all_data (class + participant only) ===
# This single pass replaces the original scattered file walks.
all_data_rows = []

participant_folders_raw = [
    f for f in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, f)) and f.startswith('P')
]
participant_folders_raw = sorted(participant_folders_raw, key=get_participant_number)

for participant in participant_folders_raw:
    hr_path = os.path.join(root_path, participant, "Mocopi", "Labeled")
    if not os.path.exists(hr_path):
        continue

    for root_dir, dirs, files in os.walk(hr_path):
        for f in files:
            if not f.endswith('.csv'):
                continue
            date_str = f[-14:-4]
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if file_date.weekday() == 4:   # skip Fridays
                continue

            file_path = os.path.join(root_dir, f)
            df = pd.read_csv(file_path)
            if 'class' not in df.columns:
                continue
            df['participant'] = participant
            all_data_rows.append(df[['class', 'participant']])

all_data = pd.concat(all_data_rows, ignore_index=True) if all_data_rows else pd.DataFrame(columns=['class','participant'])

# === Derive participant list and class list from all_data ===
participant_folders = sorted(all_data['participant'].unique().tolist(), key=get_participant_number)

# class_names: only classes that appear in actual data, minus exclusions
class_names = sorted([
    c for c in all_data['class'].dropna().unique()
    if str(c).strip() not in EXCLUDED_CLASSES
])

print(f"Participants with data: {participant_folders}")
print(f"Classes found in data ({len(class_names)}): {class_names}")

if not class_names:
    print("No class labels found in all_data. Check file contents.")
    exit()

# === Parse schedule files to get expected bins ===
# participant_schedules[participant][weekday] = [(start_t, end_t, class_name), ...]
participant_schedules = {p: {wd: [] for wd in range(5)} for p in participant_folders}
all_schedule_classes = set()

for sched_file, info in schedule_files.items():
    file_path = os.path.join(schedule_path, sched_file)
    if not os.path.exists(file_path):
        print(f"Warning: Schedule file not found: {file_path}")
        continue

    df = pd.read_csv(file_path)
    time_cols = [col for col in df.columns if any(k in col.lower() for k in ('time','start','period'))]

    for _, row in df.iterrows():
        start_time_val = end_time_val = None

        for col in df.columns:
            col_l = col.lower()
            if 'start' in col_l:
                start_time_val = row[col]
            elif 'end' in col_l:
                end_time_val = row[col]
            elif 'time' in col_l and start_time_val is None:
                start_time_val = row[col]

        if pd.isna(start_time_val):
            continue

        try:
            start_t = parse_time(str(start_time_val))
            end_t = (
                parse_time(str(end_time_val))
                if pd.notna(end_time_val)
                else (datetime.combine(datetime.today(), start_t) + timedelta(minutes=50)).time()
            )

            for col in df.columns:
                if col in time_cols or 'period' in col.lower():
                    continue
                class_name = str(row[col]).strip() if pd.notna(row[col]) else ''
                if not class_name or class_name in EXCLUDED_CLASSES:
                    continue

                all_schedule_classes.add(class_name)
                for participant in info['participants']:
                    if participant not in participant_folders:
                        continue
                    for weekday in info['weekdays']:
                        participant_schedules[participant][weekday].append((start_t, end_t, class_name))

        except Exception as e:
            print(f"Error parsing schedule row: {e}")
            continue

# === PASS 2 — Single file walk per participant for coverage ===
heatmap_hr       = pd.DataFrame(0.0, index=class_names, columns=participant_folders)
heatmap_coverage = pd.DataFrame(0.0, index=class_names, columns=participant_folders)
coverage_metrics = {}

for participant in participant_folders:
    print(f"\nProcessing: {participant}")
    hr_path = os.path.join(root_path, participant, "Mocopi", "Labeled")
    if not os.path.exists(hr_path):
        continue

    expected_class_bins = {cls: set() for cls in class_names}
    actual_class_bins   = {cls: set() for cls in class_names}

    for root_dir, dirs, files in os.walk(hr_path):
        for fname in files:
            if not fname.endswith('.csv') or 'RAW' in fname:
                continue

            # Bug 2 fix: extract date robustly using regex instead of fixed slice
            import re
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
            if not date_match:
                continue
            date_str = date_match.group(1)

            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except Exception:
                continue

            weekday = file_date.weekday()
            if weekday >= 4:  # skip Fridays and weekends
                continue

            file_path = os.path.join(root_dir, fname)
            df = pd.read_csv(file_path)

            # Bug 1 fix: only require 'Time_In_PST' and 'class' — no bpm in these files
            if not {'Time_In_PST', 'class'}.issubset(df.columns):
                continue

            df['TimeObj'] = df['Time_In_PST'].apply(parse_time)

            # Build expected bins from schedule for this date
            schedule = participant_schedules[participant][weekday]
            for start_t, end_t, class_name in schedule:
                if class_name not in class_names:
                    continue
                for bin_start, bin_end in time_bins_5min:
                    if not (bin_end <= start_t or bin_start >= end_t):
                        bin_key = (file_date, f"{bin_start.strftime('%H:%M')}-{bin_end.strftime('%H:%M')}")
                        expected_class_bins[class_name].add(bin_key)

            # Check actual coverage: a bin is covered if it has any rows for that class
            for bin_start, bin_end in time_bins_5min:
                bin_df = df[df['TimeObj'].between(bin_start, bin_end)]
                if bin_df.empty:
                    continue

                interval = f"{bin_start.strftime('%H:%M')}-{bin_end.strftime('%H:%M')}"
                bin_key  = (file_date, interval)

                for class_name in class_names:
                    if bin_key not in expected_class_bins[class_name]:
                        continue
                    # Covered = at least one row exists for this class in this bin
                    if not bin_df[bin_df['class'] == class_name].empty:
                        actual_class_bins[class_name].add(bin_key)

    # Aggregate coverage per class
    total_bins   = 0
    covered_bins = 0

    for class_name in class_names:
        n_expected = len(expected_class_bins[class_name])
        n_covered  = len(actual_class_bins[class_name])

        total_bins   += n_expected
        covered_bins += n_covered

        if n_expected > 0:
            pct = (n_covered / n_expected) * 100
            heatmap_coverage.loc[class_name, participant] = pct
            print(f"  {class_name}: {n_covered}/{n_expected} bins = {pct:.1f}%")

    coverage_pct = (covered_bins / total_bins * 100) if total_bins > 0 else 0
    coverage_metrics[participant] = {
        'total_bins':   total_bins,
        'covered_bins': covered_bins,
        'coverage_pct': coverage_pct,
    }

# === Sort classes by total coverage ===
class_order      = heatmap_coverage.sum(axis=1).sort_values(ascending=False).index.tolist()
heatmap_hr       = heatmap_hr.loc[class_order]
heatmap_coverage = heatmap_coverage.loc[class_order]

# === Save coverage metrics CSV ===
coverage_df = pd.DataFrame(coverage_metrics).T
coverage_df.index.name = 'Participant'
coverage_df = coverage_df.reset_index()
csv_path = os.path.join(output_folder, fileName_csv)
coverage_df.to_csv(csv_path, index=False)
print(f"\nCoverage metrics saved to: {csv_path}")

# === Plot HR heatmap ===
mask_hr    = heatmap_hr == 0.0
annot_hr   = heatmap_hr.round(1).astype(str)
annot_hr[mask_hr] = ""

plt.figure(figsize=(14, len(class_names) * 0.5))
sns.heatmap(
    heatmap_hr, cmap="viridis_r", linewidths=0.5, linecolor='gray',
    annot=annot_hr, fmt='s', mask=mask_hr, vmin=75, vmax=130,
    cbar_kws={'label': 'Average Heart Rate (BPM)'}
)
plt.title('Heart Rate Heatmap by Class (8:30 AM – 3:00 PM)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Class", fontsize=12)
# plt.savefig(os.path.join(output_folder, fileName_hr), dpi=300, bbox_inches='tight')
# plt.close()

# === Plot Coverage heatmap ===
mask_cov   = heatmap_coverage == 0.0
annot_cov  = heatmap_coverage.round(1).astype(str)
annot_cov[mask_cov] = ""

plt.figure(figsize=(14, len(class_names) * 0.5))
sns.heatmap(
    heatmap_coverage, cmap="viridis_r", linewidths=0.5, linecolor='gray',
    annot=annot_cov, fmt='s', mask=mask_cov, vmin=0, vmax=100,
    cbar_kws={'label': 'Data Coverage (%)'}
)
plt.title('Data Coverage Heatmap by Class (5-min bin resolution)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Class", fontsize=12)
plt.savefig(os.path.join(output_folder, fileName_coverage), dpi=300, bbox_inches='tight')
plt.close()
print(f"Coverage heatmap saved to: {os.path.join(output_folder, fileName_coverage)}")

# === Summary ===
print("\n=== Coverage Summary ===")
print(coverage_df.to_string(index=False))
for stat, fn in [("Mean", "mean"), ("Median", "median"), ("Min", "min"), ("Max", "max")]:
    print(f"{stat} coverage: {getattr(coverage_df['coverage_pct'], fn)():.1f}%")