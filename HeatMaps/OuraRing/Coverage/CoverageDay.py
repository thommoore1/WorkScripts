import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# === Paths ===
root_path      = "/Users/cibrian/Documents/GitHub/Research"
schedules_path = "/Users/cibrian/Documents/GitHub/Research/Schedules"
output_folder  = os.path.join(root_path, "1_visualization/Heatmaps/OuraRing/Coverage")
fileName_hr       = "day_data.png"
fileName_coverage = "day_coverage.png"
fileName_csv      = "day_coverage_metrics.csv"
os.makedirs(output_folder, exist_ok=True)

# === Find participant folders ===
participant_folders = [
    f for f in os.listdir(root_path)
    if os.path.isdir(os.path.join(root_path, f)) and f.startswith('P')
]

def get_participant_number(name):
    return int(name[1:])

participant_folders = sorted(participant_folders, key=get_participant_number)

# === Define weekdays ===
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# === Define 5-min bins across the full day window ===
start_time = datetime.strptime("08:30", "%H:%M")
end_time   = datetime.strptime("15:00", "%H:%M")
time_bins_5min = []
temp_time = start_time
while temp_time < end_time:
    bin_end = temp_time + timedelta(minutes=5)
    time_bins_5min.append((temp_time.time(), bin_end.time()))
    temp_time = bin_end

# ============================================================
# === Schedule loading
# ============================================================

def parse_schedule_csv(filepath):
    """
    Load a schedule CSV with columns: TimeStart, TimeEnd, Class.
    Returns a list of (start_time, end_time) tuples for non-DELETE rows.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    blocks = []
    for _, row in df.iterrows():
        if str(row['Class']).strip().upper() == 'DELETE':
            continue
        try:
            s = datetime.strptime(str(row['TimeStart']).strip(), "%H:%M:%S").time()
            e = datetime.strptime(str(row['TimeEnd']).strip(), "%H:%M:%S").time()
            blocks.append((s, e))
        except Exception as ex:
            print(f"  WARNING: Could not parse schedule row {row.to_dict()}: {ex}")
    return blocks


def extract_participants_from_filename(fname):
    """
    Given a filename like 'schedData_P(01,02,03)_M-TH.csv',
    return a set of zero-padded participant IDs, e.g. {'01','02','03'}.
    """
    match = re.search(r'P\(([^)]+)\)', fname)
    if not match:
        return set()
    ids = [x.strip() for x in match.group(1).split(',')]
    return set(ids)


def build_schedule_map(schedules_dir):
    """
    Parse all schedule files and build a map:
        participant_id (str, zero-padded) ->
            {
              'Monday':    [(start, end), ...],
              'Tuesday':   [...],
              'Wednesday': [...],
              'Thursday':  [...],
              'Friday':    [...],
            }
    Logic:
      - *_M-TH.csv  -> Monday, Wednesday, Thursday
                       (Tuesday uses M-TH schedule unless a TU override exists)
      - *_FR.csv    -> Friday
      - *TU.csv     -> Tuesday override for those participants
    """
    mth_files = {}
    fr_files  = {}
    tu_files  = {}

    for fname in os.listdir(schedules_dir):
        fpath = os.path.join(schedules_dir, fname)
        if not fname.endswith('.csv'):
            continue
        ids = extract_participants_from_filename(fname)
        if not ids:
            continue

        fname_upper = fname.upper()
        if fname_upper.endswith('TU.CSV') or '_TU.' in fname_upper:
            for pid in ids:
                tu_files[pid] = fpath
        elif '_FR' in fname_upper:
            for pid in ids:
                fr_files[pid] = fpath
        elif '_M-TH' in fname_upper:
            for pid in ids:
                mth_files[pid] = fpath

    all_pids = set(mth_files) | set(fr_files) | set(tu_files)

    schedule_map = {}
    for pid in all_pids:
        mth_blocks = parse_schedule_csv(mth_files[pid]) if pid in mth_files else []
        fr_blocks  = parse_schedule_csv(fr_files[pid])  if pid in fr_files  else []
        # Tuesday: use TU override if it exists, otherwise fall back to M-TH
        tu_blocks  = parse_schedule_csv(tu_files[pid])  if pid in tu_files  else mth_blocks

        schedule_map[pid] = {
            'Monday':    mth_blocks,
            'Tuesday':   tu_blocks,
            'Wednesday': mth_blocks,
            'Thursday':  mth_blocks,
            'Friday':    fr_blocks,
        }

    return schedule_map


def get_participant_schedule(participant_name, schedule_map):
    """
    Match a folder name like 'P001' or 'P01' to a schedule entry.
    Tries zero-padded variants: '01', '001', '1'.
    """
    num = int(participant_name[1:])
    for candidate in [f"{num:02d}", f"{num:03d}", str(num)]:
        if candidate in schedule_map:
            return schedule_map[candidate]
    return None


# ============================================================
# === Load all schedules once
# ============================================================
print("Loading schedules...")
schedule_map = build_schedule_map(schedules_path)
print(f"  Loaded schedules for participants: {sorted(schedule_map.keys())}")


# === Helper: check if a 5-min bin overlaps any class block ===
def bin_is_in_class(bin_start, bin_end, class_blocks):
    for cs, ce in class_blocks:
        if bin_start < ce and bin_end > cs:
            return True
    return False


# === Initialize dataframes ===
heatmap_hr       = pd.DataFrame(0.0, index=weekday_names, columns=participant_folders)
heatmap_coverage = pd.DataFrame(0.0, index=weekday_names, columns=participant_folders)
coverage_metrics = {p: {'total_bins': 0, 'covered_bins': 0} for p in participant_folders}

# Safe time parser
def parse_time(s):
    try:
        return datetime.strptime(s, "%H:%M:%S.%f").time()
    except ValueError:
        return datetime.strptime(s, "%H:%M:%S").time()

def is_valid_hr(bpm):
    if pd.isna(bpm):
        return False
    return 40 <= bpm <= 200


# === Process each participant ===
for participant in participant_folders:
    participant_path = os.path.join(root_path, participant)
    hr_path = os.path.join(participant_path, "OuraRing", "HeartRate")
    if not os.path.exists(hr_path):
        continue

    p_schedule = get_participant_schedule(participant, schedule_map)
    if p_schedule is None:
        print(f"  WARNING: No schedule found for {participant}. Skipping.")
        continue

    # Pre-compute which 5-min bins fall within class time per weekday
    class_bins_by_weekday = {}
    for day in weekday_names:
        class_blocks = p_schedule[day]
        class_bins_by_weekday[day] = [
            (bs, be) for bs, be in time_bins_5min
            if bin_is_in_class(bs, be, class_blocks)
        ]

    # Per-weekday tracking structures
    participant_weekday_5min_coverage = {
        day: {
            f"{bs.strftime('%H:%M')}-{be.strftime('%H:%M')}": []
            for bs, be in class_bins_by_weekday[day]
        }
        for day in weekday_names
    }
    participant_weekday_hr = {day: [] for day in weekday_names}

    for file in os.listdir(hr_path):
        if not (file.endswith(".csv") and "RAW" not in file):
            continue
        try:
            date_str = file[-14:-4]
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue

        weekday = file_date.weekday()
        if weekday > 4:
            continue
        weekday_name = weekday_names[weekday]

        if not class_bins_by_weekday[weekday_name]:
            continue  # No class blocks for this day — skip entirely

        file_path = os.path.join(hr_path, file)
        df = pd.read_csv(file_path)
        if 'Time_In_PST' not in df.columns or 'bpm' not in df.columns:
            continue

        df['TimeObj']  = df['Time_In_PST'].apply(parse_time)
        df['valid_hr'] = df['bpm'].apply(is_valid_hr)

        # Coverage: only count bins that fall within class time
        for bs, be in class_bins_by_weekday[weekday_name]:
            bin_df   = df[df['TimeObj'].between(bs, be)]
            interval = f"{bs.strftime('%H:%M')}-{be.strftime('%H:%M')}"
            has_valid_sample = bin_df['valid_hr'].any()
            participant_weekday_5min_coverage[weekday_name][interval].append(
                1 if has_valid_sample else 0
            )

        # HR average: only over class-time windows
        class_blocks = p_schedule[weekday_name]
        valid_hr_rows = df[
            df['valid_hr'] &
            df['TimeObj'].apply(
                lambda t: any(cs <= t <= ce for cs, ce in class_blocks)
            )
        ]
        if not valid_hr_rows.empty:
            participant_weekday_hr[weekday_name].append(valid_hr_rows['bpm'].mean())

    # === Aggregate coverage by weekday ===
    total_5min_bins_all_days   = 0
    covered_5min_bins_all_days = 0

    for weekday_name in weekday_names:
        bins_for_this_weekday = []
        for interval, coverage_list in participant_weekday_5min_coverage[weekday_name].items():
            if coverage_list:
                coverage_pct = (sum(coverage_list) / len(coverage_list)) * 100
                bins_for_this_weekday.append(coverage_pct)
                total_5min_bins_all_days += 1
                if sum(coverage_list) > 0:
                    covered_5min_bins_all_days += 1

        if bins_for_this_weekday:
            heatmap_coverage.loc[weekday_name, participant] = np.mean(bins_for_this_weekday)

        hr_list = participant_weekday_hr[weekday_name]
        if hr_list:
            heatmap_hr.loc[weekday_name, participant] = np.mean(hr_list)

    coverage_metrics[participant]['total_bins']   = total_5min_bins_all_days
    coverage_metrics[participant]['covered_bins'] = covered_5min_bins_all_days
    coverage_metrics[participant]['coverage_pct'] = (
        covered_5min_bins_all_days / total_5min_bins_all_days * 100
    ) if total_5min_bins_all_days > 0 else 0

# === Save coverage metrics to CSV ===
coverage_df = pd.DataFrame(coverage_metrics).T
coverage_df.index.name = 'Participant'
coverage_df = coverage_df.reset_index()
csv_path = os.path.join(output_folder, fileName_csv)
coverage_df.to_csv(csv_path, index=False)
print(f"Coverage metrics saved to: {csv_path}")

# === Plot HR heatmap ===
mask_hr  = heatmap_hr == 0.0
annot_hr = heatmap_hr.round(1).astype(str)
annot_hr[mask_hr] = ""
plt.figure(figsize=(14, 6))
sns.heatmap(
    heatmap_hr,
    cmap="viridis_r",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_hr,
    fmt='s',
    mask=mask_hr,
    vmin=75, vmax=130,
    cbar_kws={'label': 'Average Heart Rate (BPM)'}
)
plt.title('Heart Rate Heatmap by Weekday (Class Times Only)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Day of Week", fontsize=12)
output_file_hr = os.path.join(output_folder, fileName_hr)
plt.savefig(output_file_hr, dpi=300, bbox_inches='tight')
plt.close()
print(f"HR heatmap saved to: {output_file_hr}")

# === Plot Coverage heatmap ===
mask_coverage  = heatmap_coverage == 0.0
annot_coverage = heatmap_coverage.round(1).astype(str)
annot_coverage[mask_coverage] = ""
plt.figure(figsize=(14, 6))
sns.heatmap(
    heatmap_coverage,
    cmap="viridis_r",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_coverage,
    fmt='s',
    mask=mask_coverage,
    vmin=0, vmax=100,
    cbar_kws={'label': 'Data Coverage (%)'}
)
plt.title('Data Coverage Heatmap by Weekday (Class Times Only, 5-min bins)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Day of Week", fontsize=12)
output_file_coverage = os.path.join(output_folder, fileName_coverage)
plt.savefig(output_file_coverage, dpi=300, bbox_inches='tight')
plt.close()
print(f"Coverage heatmap saved to: {output_file_coverage}")

# === Print summary statistics ===
print("\n=== Coverage Summary (Class Times Only) ===")
print(coverage_df.to_string(index=False))
print(f"\nMean coverage:   {coverage_df['coverage_pct'].mean():.1f}%")
print(f"Median coverage: {coverage_df['coverage_pct'].median():.1f}%")
print(f"Min coverage:    {coverage_df['coverage_pct'].min():.1f}%")
print(f"Max coverage:    {coverage_df['coverage_pct'].max():.1f}%")