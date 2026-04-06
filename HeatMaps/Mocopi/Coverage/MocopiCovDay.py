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
output_folder  = os.path.join(root_path, "1_visualization/Heatmaps/Mocopi/Coverage")
fileName_coverage = "day_coverage.png"
fileName_csv      = "day_coverage_metrics.csv"
os.makedirs(output_folder, exist_ok=True)

# === Classes to exclude ===
EXCLUDED_CLASSES = {'DELETE', 'ELA/History', 'Friday Funday'}

# === Find participant folders ===
participant_folders = sorted(
    [f for f in os.listdir(root_path)
     if os.path.isdir(os.path.join(root_path, f)) and f.startswith('P')],
    key=lambda name: int(name[1:])
)

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
# === Schedule loading (reused from Script 1 verbatim)
# ============================================================

def parse_schedule_csv(filepath):
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
    match = re.search(r'P\(([^)]+)\)', fname)
    if not match:
        return set()
    return set(x.strip() for x in match.group(1).split(','))


def build_schedule_map(schedules_dir):
    mth_files, fr_files, tu_files = {}, {}, {}

    for fname in os.listdir(schedules_dir):
        fpath = os.path.join(schedules_dir, fname)
        if not fname.endswith('.csv'):
            continue
        ids = extract_participants_from_filename(fname)
        if not ids:
            continue
        fname_upper = fname.upper()
        if fname_upper.endswith('TU.CSV') or '_TU.' in fname_upper:
            for pid in ids: tu_files[pid] = fpath
        elif '_FR' in fname_upper:
            for pid in ids: fr_files[pid] = fpath
        elif '_M-TH' in fname_upper:
            for pid in ids: mth_files[pid] = fpath

    schedule_map = {}
    for pid in set(mth_files) | set(fr_files) | set(tu_files):
        mth_blocks = parse_schedule_csv(mth_files[pid]) if pid in mth_files else []
        fr_blocks  = parse_schedule_csv(fr_files[pid])  if pid in fr_files  else []
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
    num = int(participant_name[1:])
    for candidate in [f"{num:02d}", f"{num:03d}", str(num)]:
        if candidate in schedule_map:
            return schedule_map[candidate]
    return None


def bin_is_in_class(bin_start, bin_end, class_blocks):
    return any(bin_start < ce and bin_end > cs for cs, ce in class_blocks)


# ============================================================
# === Time parser (supports Mocopi's formats)
# ============================================================

def parse_time(s):
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(str(s), fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse time: {s}")


# ============================================================
# === Load schedules
# ============================================================
print("Loading schedules...")
schedule_map = build_schedule_map(schedules_path)
print(f"  Loaded schedules for participants: {sorted(schedule_map.keys())}")

# ============================================================
# === Main processing loop
# ============================================================
heatmap_coverage = pd.DataFrame(0.0, index=weekday_names, columns=participant_folders)
coverage_metrics = {p: {'total_bins': 0, 'covered_bins': 0} for p in participant_folders}

for participant in participant_folders:
    # ── Mocopi path instead of OuraRing ──────────────────────
    labeled_path = os.path.join(root_path, participant, "Mocopi", "Labeled")
    if not os.path.exists(labeled_path):
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

    # Per-weekday coverage tracking
    participant_weekday_5min_coverage = {
        day: {
            f"{bs.strftime('%H:%M')}-{be.strftime('%H:%M')}": []
            for bs, be in class_bins_by_weekday[day]
        }
        for day in weekday_names
    }

    # Walk all CSV files under Mocopi/Labeled/ (mirrors Script 2's os.walk)
    for root_dir, dirs, files in os.walk(labeled_path):
        for fname in files:
            if not fname.endswith('.csv'):
                continue

            # Robust date extraction (Script 2's regex approach)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
            if not date_match:
                continue
            try:
                file_date = datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue

            weekday = file_date.weekday()
            if weekday > 4:
                continue
            weekday_name = weekday_names[weekday]

            if not class_bins_by_weekday[weekday_name]:
                continue  # no class scheduled this day

            file_path = os.path.join(root_dir, fname)
            df = pd.read_csv(file_path)

            # Mocopi files have 'Time_In_PST' and 'class', NOT 'bpm'
            if not {'Time_In_PST', 'class'}.issubset(df.columns):
                continue

            df['TimeObj'] = df['Time_In_PST'].apply(parse_time)

            # A bin is "covered" if it contains at least one row
            # whose 'class' value is not excluded
            df_valid = df[~df['class'].astype(str).str.strip().isin(EXCLUDED_CLASSES)]

            for bs, be in class_bins_by_weekday[weekday_name]:
                bin_df   = df_valid[df_valid['TimeObj'].between(bs, be)]
                interval = f"{bs.strftime('%H:%M')}-{be.strftime('%H:%M')}"
                participant_weekday_5min_coverage[weekday_name][interval].append(
                    1 if not bin_df.empty else 0
                )

    # === Aggregate coverage per weekday ===
    total_bins_all   = 0
    covered_bins_all = 0

    for day in weekday_names:
        bins_for_day = []
        for interval, cov_list in participant_weekday_5min_coverage[day].items():
            if cov_list:
                pct = (sum(cov_list) / len(cov_list)) * 100
                bins_for_day.append(pct)
                total_bins_all   += 1
                if sum(cov_list) > 0:
                    covered_bins_all += 1

        if bins_for_day:
            heatmap_coverage.loc[day, participant] = np.mean(bins_for_day)

    coverage_metrics[participant]['total_bins']   = total_bins_all
    coverage_metrics[participant]['covered_bins'] = covered_bins_all
    coverage_metrics[participant]['coverage_pct'] = (
        covered_bins_all / total_bins_all * 100
    ) if total_bins_all > 0 else 0

# === Save coverage metrics CSV ===
coverage_df = pd.DataFrame(coverage_metrics).T
coverage_df.index.name = 'Participant'
coverage_df = coverage_df.reset_index()
csv_path = os.path.join(output_folder, fileName_csv)
coverage_df.to_csv(csv_path, index=False)
print(f"Coverage metrics saved to: {csv_path}")

# === Plot Coverage heatmap ===
mask_cov  = heatmap_coverage == 0.0
annot_cov = heatmap_coverage.round(1).astype(str)
annot_cov[mask_cov] = ""

plt.figure(figsize=(14, 6))
sns.heatmap(
    heatmap_coverage,
    cmap="viridis_r",
    linewidths=0.5,
    linecolor='gray',
    annot=annot_cov,
    fmt='s',
    mask=mask_cov,
    vmin=0, vmax=100,
    cbar_kws={'label': 'Data Coverage (%)'}
)
plt.title('Mocopi Data Coverage Heatmap by Weekday (Class Times Only, 5-min bins)', fontsize=14, pad=20)
plt.xlabel("Participant", fontsize=12)
plt.ylabel("Day of Week", fontsize=12)
output_file_coverage = os.path.join(output_folder, fileName_coverage)
plt.savefig(output_file_coverage, dpi=300, bbox_inches='tight')
plt.close()
print(f"Coverage heatmap saved to: {output_file_coverage}")

# === Summary ===
print("\n=== Coverage Summary (Class Times Only) ===")
print(coverage_df.to_string(index=False))
for stat, fn in [("Mean", "mean"), ("Median", "median"), ("Min", "min"), ("Max", "max")]:
    print(f"{stat} coverage: {getattr(coverage_df['coverage_pct'], fn)():.1f}%")