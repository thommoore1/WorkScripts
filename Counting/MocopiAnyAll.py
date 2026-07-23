import pandas as pd
import pytz
import os
import re
import numpy as np

from datetime import datetime, timezone, date
from collections import defaultdict
from pathlib import Path

def convert_to_unix_time(timestamp_str):
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()

def extract_time_only(timestamp_str):
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.time()

def convert_string_to_time(time_string):
    time_obj = datetime.strptime(time_string, "%H:%M:%S").time()
    return time_obj

def get_day_of_week(date_obj):
    return date_obj.strftime("%A")

def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000

def getSensorLocation(fileName):
    mapping = {
        "11CCD": "HeadDeviceOne",
        "132D3": "HeadDeviceTwo",
        "1092A": "HeadDeviceThree",
        "13CF2": "HeadDeviceFour",
        "12144": "HipDeviceOne",
        "114C8": "HipDeviceTwo",
        "10B1F": "HipDeviceThree",
        "1211E": "HipDeviceFour",
        "0E3E9": "WristRDeviceOne",
        "0EE55": "WristRDeviceTwo",
        "12801": "WristRDeviceThree",
        "0EA70": "WristRDeviceFour",
        "14A51": "WristLDeviceOne",
        "134F5": "WristLDeviceTwo",
        "1447A": "WristLDeviceThree",
        "14A53": "WristLDeviceFour",
        "1503C": "AnkleRDeviceOne",
        "13B8F": "AnkleRDeviceTwo",
        "13B06": "AnkleRDeviceThree",
        "158A6": "AnkleRDeviceFour",
        "16E17": "AnkleLDeviceOne",
        "16FB1": "AnkleLDeviceTwo",
        "142A8": "AnkleLDeviceThree",
        "16CA7": "AnkleLDeviceFour",
    }
    for code, label in mapping.items():
        if code in fileName:
            return label
    return "None"


def normalize_sensor(label):
    """Strip trailing device number suffix: AnkleLDeviceOne -> AnkleLDevice
    This collapses the 24 raw device codes down to the 6 physical sensor
    locations: HeadDevice, HipDevice, WristRDevice, WristLDevice,
    AnkleRDevice, AnkleLDevice.
    """
    return re.sub(r'(Device)(One|Two|Three|Four)$', r'\1', label)


# ── Invalid-data detection ──────────────────────────────────────────────────
# Columns that must contain sane numeric values for a row to be considered
# "valid" sensor data.
REQUIRED_NUMERIC_COLS = [
    'Rotation X', 'Rotation Y', 'Rotation Z', 'Rotation W',
    'Acceleration X', 'Acceleration Y', 'Acceleration Z',
]

# Quaternion components must fall within [-1, 1]. Adjust if your rotation
# columns are not normalized quaternion components.
ROTATION_MIN, ROTATION_MAX = -1.0, 1.0

# Acceleration "impossible value" bound, in g's. Mocopi-style IMUs typically
# report in the +/-16g range; tune this to your actual sensor spec.
ACCEL_MAX_G = 16.0


def find_invalid_rows(df, timestamp_col='Old Timestamp'):
    """
    Returns a boolean Series (aligned to df's index) that is True for rows
    where any of Timestamp, Rotation X/Y/Z/W, Acceleration X/Y/Z are either
    not parseable/numeric or fall outside a physically plausible range.
    """
    invalid = pd.Series(False, index=df.index)

    # Timestamp must parse as a valid datetime.
    if timestamp_col in df.columns:
        ts_parsed = pd.to_datetime(
            df[timestamp_col], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce'
        )
        invalid |= ts_parsed.isna()
    else:
        invalid |= True

    # Rotation components: numeric and within [-1, 1].
    for col in ['Rotation X', 'Rotation Y', 'Rotation Z', 'Rotation W']:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            invalid |= vals.isna() | (vals < ROTATION_MIN) | (vals > ROTATION_MAX)
        else:
            invalid |= True  # column missing entirely -> can't validate, treat as invalid

    # Acceleration components: numeric and within +/- ACCEL_MAX_G.
    for col in ['Acceleration X', 'Acceleration Y', 'Acceleration Z']:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            invalid |= vals.isna() | (vals.abs() > ACCEL_MAX_G)
        else:
            invalid |= True

    return invalid
# ─────────────────────────────────────────────────────────────────────────────


# ── 1-second bin classification ─────────────────────────────────────────────
def compute_bin_dict(df, time_col='time', threshold=50, start_sec=None, end_sec=None):
    """
    Breaks [start_sec, end_sec] (inclusive) into 1-second bins (keyed by
    integer unix-epoch second) and returns a dict {bin_second: bool} where
    the bool is True iff the bin contains >= `threshold` datapoints.

    Passing explicit start_sec/end_sec lets "before" and "after" invalid-row
    removal be compared over the exact same bin range, so bins that lose all
    their data due to invalid rows correctly become False rather than
    disappearing from the comparison.
    """
    if start_sec is None or end_sec is None:
        if df.empty:
            return {}
        start_sec = int(df[time_col].min())
        end_sec   = int(df[time_col].max())

    all_bins = np.arange(int(start_sec), int(end_sec) + 1)

    if df.empty:
        counts = pd.Series(0, index=all_bins)
    else:
        counts_per_sec = df.groupby(time_col).size()
        counts = counts_per_sec.reindex(all_bins, fill_value=0)

    return {int(b): bool(c >= threshold) for b, c in counts.items()}


def compute_bins_true_count(filtered_df, time_col='time', threshold=50,
                             start_sec=None, end_sec=None):
    """Convenience wrapper: returns (true_bin_count, total_bin_count)."""
    d = compute_bin_dict(filtered_df, time_col=time_col, threshold=threshold,
                          start_sec=start_sec, end_sec=end_sec)
    return sum(d.values()), len(d)


def merge_bin_dict(target, addition):
    """OR-merge `addition` into `target` in place (True wins on collision)."""
    for k, v in addition.items():
        target[k] = target.get(k, False) or v


def combine_any_all(sensor_bin_dicts):
    """
    Given a dict {sensor_name: {bin_second: bool}} for a fixed set of
    sensors, returns (any_true_count, all_true_count, universe_size,
    n_sensors_present).

    - "any_true": bin is True if at least one sensor is True in that bin.
    - "all_true": bin is True only if every sensor in `sensor_bin_dicts`
      is True in that bin (a sensor with no coverage for a given bin is
      treated as False for that bin).
    """
    sensor_names = list(sensor_bin_dicts.keys())
    n_sensors = len(sensor_names)

    universe = set()
    for d in sensor_bin_dicts.values():
        universe |= set(d.keys())

    any_true = 0
    all_true = 0
    for b in universe:
        vals = [sensor_bin_dicts[s].get(b, False) for s in sensor_names]
        if any(vals):
            any_true += 1
        if n_sensors > 0 and all(vals):
            all_true += 1

    return any_true, all_true, len(universe), n_sensors
# ─────────────────────────────────────────────────────────────────────────────


now = datetime.now()
current_time = now.strftime("%H:%M:%S")

participant_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "12", "14", "16"]
rootPath = "/Users/cibrian/Documents/GitHub/Research"

# ── accumulators for summary stats ───────────────────────────────────────────
# Maps participant → total valid datapoints
participant_totals = {}
# Maps participant → {sensor_label: count}
participant_sensor_counts = {}

# ── accumulators for 1-second bin true-count stats ──────────────────────────
# Maps participant → {sensor_label: true_bin_count}, before/after invalid-data removal
participant_sensor_true_bins_before = {}
participant_sensor_true_bins_after  = {}
# Maps participant → total true bins across all sensors, before/after
participant_true_bins_before_total = {}
participant_true_bins_after_total  = {}

# ── accumulators for the 6-sensor "any"/"all" bin comparison ───────────────
# Maps participant → {normalized_sensor: {bin_second: bool}}, merged across
# every raw device file mapping to that sensor location.
participant_sensor_bin_dicts_before = {}
participant_sensor_bin_dicts_after  = {}
# Maps participant → (any_true, all_true, universe_size, n_sensors_present)
participant_any_all_before = {}
participant_any_all_after  = {}
# ─────────────────────────────────────────────────────────────────────────────

for pNum in participant_numbers:
    print(f"Processing Participant P0{pNum}...")

    rawParentPath     = f"{rootPath}/P0{pNum}/Mocopi/Raw"
    labeledParentPath = f"{rootPath}/P0{pNum}/Mocopi/Labeled"

    directories = [d for d in os.listdir(rawParentPath) if os.path.isdir(os.path.join(rawParentPath, d))]

    grouped_raw_data = defaultdict(list)

    for dir_name in directories:
        folder_path = Path(rawParentPath) / dir_name
        for file in folder_path.iterdir():
            dataFrame = pd.read_csv(file)
            dateTime  = datetime.strptime(dataFrame.iloc[0]['Timestamp'], "%Y-%m-%d %H:%M:%S.%f")
            dateOnly  = dateTime.date().strftime("%Y-%m-%d")
            sensor_label = getSensorLocation(str(file))
            grouped_raw_data[(sensor_label, dateOnly)].append(dataFrame)

    dataFrames   = []
    csvPathList  = []
    sensorLabels = []          # track sensor label per dataframe

    for (sensor_label, dateOnly), dfs in grouped_raw_data.items():
        combined_df = pd.concat(dfs, ignore_index=True).sort_values(by="Timestamp").reset_index(drop=True)
        dataFrames.append(combined_df)
        sensorLabels.append(sensor_label)

        dirPath = os.path.join(labeledParentPath, dateOnly)
        os.makedirs(dirPath, exist_ok=True)

        file_path = os.path.join(dirPath, f"P0{pNum}Mocopi{sensor_label}{dateOnly}.csv")
        csvPathList.append(file_path)

    if pNum in ["04", "05", "09", "14", "16"]:
        scheduleDataFri = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(04,05,09,14,16)_FR.csv")
        scheduleDataOth = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(04,05,09,14,16)_M-TH.csv")
        if pNum in ['14', '16']:
            scheduleDataTu = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(14,16)TU.csv")
    else:
        scheduleDataFri = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,12)_FR.csv")
        scheduleDataOth = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,12)_M-TH.csv")

    zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
    for rawData in dataFrames:
        rawData.insert(0, 'class', "NONE")
        rawData.insert(1, 'Time_In_PST', zero_time)
        rawData.insert(2, 'time', 0.0)

    for i, df in enumerate(dataFrames):
        dt = pd.to_datetime(df['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f")
        df['time']       = dt.astype('int64') // 10**9
        df['Time_In_PST'] = dt.dt.time
        df.rename(columns={'Timestamp': 'Old Timestamp'}, inplace=True)
        dataFrames[i] = df

    for dataFrame in dataFrames:
        DayOfWeek = get_day_of_week(datetime.fromtimestamp(dataFrame.iloc[0]['time']))
        if DayOfWeek == 'Friday':
            scheduleData = scheduleDataFri
        elif DayOfWeek == 'Tuesday' and (pNum == "14" or pNum == "16"):
            scheduleData = scheduleDataTu
        else:
            scheduleData = scheduleDataOth

        scheduleData = scheduleData.copy()
        scheduleData['TimeStart'] = pd.to_datetime(scheduleData['TimeStart'], format="%H:%M:%S").dt.time
        scheduleData['TimeEnd']   = pd.to_datetime(scheduleData['TimeEnd'],   format="%H:%M:%S").dt.time
        scheduleData['TimeStart_sec'] = scheduleData['TimeStart'].apply(time_to_seconds)
        scheduleData['TimeEnd_sec']   = scheduleData['TimeEnd'].apply(time_to_seconds)

        time_values_sec = dataFrame['Time_In_PST'].apply(time_to_seconds)

        intervals = pd.IntervalIndex.from_arrays(
            scheduleData['TimeStart_sec'],
            scheduleData['TimeEnd_sec'],
            closed='right'
        )

        matched_class = np.full(len(time_values_sec), None, dtype=object)
        for i, interval in enumerate(intervals):
            mask = (interval.left < time_values_sec) & (time_values_sec <= interval.right)
            matched_class = np.where(mask, scheduleData.iloc[i]['Class'], matched_class)

        dataFrame['class'] = matched_class

    # ── count valid datapoints AND 1-second-bin true/false stats per sensor ──
    sensor_counts = {}
    sensor_true_bins_before = {}
    sensor_true_bins_after  = {}

    # Per-participant, per-normalized-sensor (6 total) bin dicts, merged
    # (OR'd) across every raw device file that maps to that sensor location.
    sensor_bin_dicts_before = {}   # {normalized_sensor: {bin_second: bool}}
    sensor_bin_dicts_after  = {}

    for i in range(len(dataFrames)):
        dataFrame    = dataFrames[i].copy()
        sensor_label = sensorLabels[i]
        norm_sensor  = normalize_sensor(sensor_label)

        dataFrame['class'] = dataFrame['class'].str.strip()

        # "Class time" = rows where the schedule is not NONE and not DELETE.
        # DELETE rows mean "no class time" and are excluded outright, from
        # both the before- and after- comparisons.
        class_time_df = dataFrame[
            dataFrame['class'].notna()
            & (dataFrame['class'] != 'NONE')
            & (dataFrame['class'] != 'DELETE')
        ].reset_index(drop=True)

        if class_time_df.empty:
            true_before, true_after, total_bins = 0, 0, 0
            invalid_mask = pd.Series(dtype=bool)
            bin_dict_before = {}
            bin_dict_after  = {}
        else:
            start_sec = int(class_time_df['time'].min())
            end_sec   = int(class_time_df['time'].max())

            # --- BEFORE removing invalid data: all class-time rows, valid or not ---
            bin_dict_before = compute_bin_dict(class_time_df, start_sec=start_sec, end_sec=end_sec)
            true_before  = sum(bin_dict_before.values())
            total_bins   = len(bin_dict_before)

            # --- Identify invalid rows: bad Timestamp / Rotation / Acceleration ---
            invalid_mask = find_invalid_rows(class_time_df)
            valid_class_time_df = class_time_df[~invalid_mask].reset_index(drop=True)

            # --- AFTER removing invalid data: same bin range, invalid rows gone ---
            bin_dict_after = compute_bin_dict(valid_class_time_df, start_sec=start_sec, end_sec=end_sec)
            true_after = sum(bin_dict_after.values())

        sensor_counts[sensor_label] = sensor_counts.get(sensor_label, 0) + len(class_time_df) - int(invalid_mask.sum() if len(invalid_mask) else 0)
        sensor_true_bins_before[sensor_label] = sensor_true_bins_before.get(sensor_label, 0) + true_before
        sensor_true_bins_after[sensor_label]  = sensor_true_bins_after.get(sensor_label, 0) + true_after

        # Merge this device's per-bin True/False dict into its normalized
        # (6-sensor-location) bucket for the any/all comparison.
        sensor_bin_dicts_before.setdefault(norm_sensor, {})
        sensor_bin_dicts_after.setdefault(norm_sensor, {})
        merge_bin_dict(sensor_bin_dicts_before[norm_sensor], bin_dict_before)
        merge_bin_dict(sensor_bin_dicts_after[norm_sensor],  bin_dict_after)

        # Save the labeled file with DELETE rows removed (unchanged behavior).
        dataFrame_to_save = dataFrame[dataFrame['class'] != 'DELETE'].reset_index(drop=True)
        dataFrame_to_save.to_csv(csvPathList[i], index=False)
        dataFrames[i] = dataFrame_to_save

    participant_sensor_counts[pNum] = sensor_counts
    participant_totals[pNum]        = sum(sensor_counts.values())

    participant_sensor_true_bins_before[pNum] = sensor_true_bins_before
    participant_sensor_true_bins_after[pNum]  = sensor_true_bins_after
    participant_true_bins_before_total[pNum]  = sum(sensor_true_bins_before.values())
    participant_true_bins_after_total[pNum]   = sum(sensor_true_bins_after.values())

    participant_sensor_bin_dicts_before[pNum] = sensor_bin_dicts_before
    participant_sensor_bin_dicts_after[pNum]  = sensor_bin_dicts_after

    participant_any_all_before[pNum] = combine_any_all(sensor_bin_dicts_before)
    participant_any_all_after[pNum]  = combine_any_all(sensor_bin_dicts_after)
    # ─────────────────────────────────────────────────────────────────────────

# ── print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATAPOINT SUMMARY — VALID CLASS PERIODS ONLY")
print("=" * 60)

# Per-participant totals
print("\nPer-Participant Totals:")
print(f"  {'Participant':<15} {'Total Valid Datapoints':>22}")
print(f"  {'-'*15} {'-'*22}")
for pNum in participant_numbers:
    print(f"  P0{pNum:<13} {participant_totals[pNum]:>22,}")

overall_avg_total = np.mean(list(participant_totals.values()))
print(f"\n  {'Average across participants:':<35} {overall_avg_total:>10,.1f}")

# Collapse per-participant sensor counts into normalized sensor names
normalized_participant_sensor_counts = {}
for pNum, sensor_counts in participant_sensor_counts.items():
    collapsed = defaultdict(int)
    for sensor, count in sensor_counts.items():
        collapsed[normalize_sensor(sensor)] += count
    normalized_participant_sensor_counts[pNum] = collapsed

# Per-sensor averages across participants
print("\nPer-Sensor Averages (across all participants):")
print(f"  {'Sensor':<20} {'Avg Valid Datapoints':>20}")
print(f"  {'-'*20} {'-'*20}")

all_sensors = sorted({
    s
    for counts in normalized_participant_sensor_counts.values()
    for s in counts
})
for sensor in all_sensors:
    counts_for_sensor = [
        normalized_participant_sensor_counts[p][sensor]
        for p in participant_numbers
        if sensor in normalized_participant_sensor_counts[p]
    ]
    avg = np.mean(counts_for_sensor) if counts_for_sensor else 0.0
    print(f"  {sensor:<20} {avg:>20,.1f}")

print("=" * 60)

# ── 1-second bin True-count summary (before vs. after removing invalid data) ──
print("\n" + "=" * 60)
print("1-SECOND BIN SUMMARY — TRUE BINS (>=50 datapoints/bin)")
print("Before/After = before vs. after removing rows with invalid")
print("Timestamp/Rotation/Acceleration values (DELETE rows are always excluded)")
print("=" * 60)

print("\nPer-Participant True-Bin Counts:")
print(f"  {'Participant':<15} {'True Bins (Before)':>20} {'True Bins (After)':>20} {'Diff':>10}")
print(f"  {'-'*15} {'-'*20} {'-'*20} {'-'*10}")
for pNum in participant_numbers:
    before = participant_true_bins_before_total[pNum]
    after  = participant_true_bins_after_total[pNum]
    print(f"  P0{pNum:<13} {before:>20,} {after:>20,} {before - after:>10,}")

avg_before = np.mean(list(participant_true_bins_before_total.values()))
avg_after  = np.mean(list(participant_true_bins_after_total.values()))
print(f"\n  {'Average True Bins Before:':<35} {avg_before:>10,.1f}")
print(f"  {'Average True Bins After:':<35} {avg_after:>10,.1f}")

# Collapse per-participant per-sensor true-bin counts into normalized sensor names
def collapse_true_bins(participant_sensor_true_bins):
    normalized = {}
    for pNum, sensor_counts in participant_sensor_true_bins.items():
        collapsed = defaultdict(int)
        for sensor, count in sensor_counts.items():
            collapsed[normalize_sensor(sensor)] += count
        normalized[pNum] = collapsed
    return normalized

normalized_true_bins_before = collapse_true_bins(participant_sensor_true_bins_before)
normalized_true_bins_after  = collapse_true_bins(participant_sensor_true_bins_after)

print("\nPer-Sensor Average True-Bin Counts (across all participants):")
print(f"  {'Sensor':<20} {'Avg True Bins (Before)':>22} {'Avg True Bins (After)':>22}")
print(f"  {'-'*20} {'-'*22} {'-'*22}")

all_sensors_bins = sorted({
    s
    for counts in normalized_true_bins_before.values()
    for s in counts
})
for sensor in all_sensors_bins:
    before_vals = [
        normalized_true_bins_before[p][sensor]
        for p in participant_numbers
        if sensor in normalized_true_bins_before.get(p, {})
    ]
    after_vals = [
        normalized_true_bins_after[p][sensor]
        for p in participant_numbers
        if sensor in normalized_true_bins_after.get(p, {})
    ]
    avg_b = np.mean(before_vals) if before_vals else 0.0
    avg_a = np.mean(after_vals) if after_vals else 0.0
    print(f"  {sensor:<20} {avg_b:>22,.1f} {avg_a:>22,.1f}")

print("=" * 60)

# ── ANY-sensor-true / ALL-sensors-true bin summary ─────────────────────────
# "Any true": bin counted True if at least 1 of the 6 sensor locations has
#             >=50 valid datapoints in that bin.
# "All true": bin counted True only if every one of the 6 sensor locations
#             has >=50 valid datapoints in that bin.
print("\n" + "=" * 60)
print("CROSS-SENSOR BIN SUMMARY (6 sensor locations: Head, Hip, WristR,")
print("WristL, AnkleR, AnkleL) — ANY-TRUE vs ALL-TRUE, BEFORE vs AFTER")
print("=" * 60)

print("\nPer-Participant Any/All True-Bin Counts:")
header = (f"  {'Participant':<12} {'#Sensors':>9} {'Universe':>9} "
          f"{'Any(Before)':>12} {'Any(After)':>11} "
          f"{'All(Before)':>12} {'All(After)':>11}")
print(header)
print("  " + "-" * (len(header) - 2))

for pNum in participant_numbers:
    any_b, all_b, universe_b, n_sensors_b = participant_any_all_before[pNum]
    any_a, all_a, universe_a, n_sensors_a = participant_any_all_after[pNum]
    # universe/n_sensors are identical before vs after by construction
    print(f"  P0{pNum:<10} {n_sensors_b:>9} {universe_b:>9,} "
          f"{any_b:>12,} {any_a:>11,} {all_b:>12,} {all_a:>11,}")
    if n_sensors_b < 6:
        print(f"    (note: only {n_sensors_b}/6 sensor locations had data for P0{pNum})")

any_before_vals = [participant_any_all_before[p][0] for p in participant_numbers]
any_after_vals  = [participant_any_all_after[p][0]  for p in participant_numbers]
all_before_vals = [participant_any_all_before[p][1] for p in participant_numbers]
all_after_vals  = [participant_any_all_after[p][1]  for p in participant_numbers]

print(f"\n  {'Average Any-True Bins (Before):':<35} {np.mean(any_before_vals):>10,.1f}")
print(f"  {'Average Any-True Bins (After):':<35} {np.mean(any_after_vals):>10,.1f}")
print(f"  {'Average All-True Bins (Before):':<35} {np.mean(all_before_vals):>10,.1f}")
print(f"  {'Average All-True Bins (After):':<35} {np.mean(all_after_vals):>10,.1f}")

print("=" * 60)
