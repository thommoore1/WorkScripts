import pandas as pd
import pytz
import os
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


now = datetime.now()
current_time = now.strftime("%H:%M:%S")

participant_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "12", "14", "16"]
rootPath = "/Users/cibrian/Documents/GitHub/Research"

# ── NEW: accumulators for summary stats ──────────────────────────────────────
# Maps participant → total valid datapoints
participant_totals = {}
# Maps participant → {sensor_label: count}
participant_sensor_counts = {}
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
    sensorLabels = []          # ← NEW: track sensor label per dataframe

    for (sensor_label, dateOnly), dfs in grouped_raw_data.items():
        combined_df = pd.concat(dfs, ignore_index=True).sort_values(by="Timestamp").reset_index(drop=True)
        dataFrames.append(combined_df)
        sensorLabels.append(sensor_label)   # ← NEW

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

    # ── NEW: count valid datapoints per sensor for this participant ───────────
    sensor_counts = {}
    for i in range(len(dataFrames)):
        dataFrame    = dataFrames[i].copy()
        sensor_label = sensorLabels[i]

        dataFrame['class'] = dataFrame['class'].str.strip()
        dataFrame = dataFrame[dataFrame['class'] != 'DELETE'].reset_index(drop=True)

        valid_mask  = dataFrame['class'].notna() & (dataFrame['class'] != 'NONE')
        valid_count = valid_mask.sum()

        sensor_counts[sensor_label] = sensor_counts.get(sensor_label, 0) + int(valid_count)

        dataFrame.to_csv(csvPathList[i], index=False)
        dataFrames[i] = dataFrame

    participant_sensor_counts[pNum] = sensor_counts
    participant_totals[pNum]        = sum(sensor_counts.values())
    # ─────────────────────────────────────────────────────────────────────────

# ── NEW: print summary ────────────────────────────────────────────────────────
import re

def normalize_sensor(label):
    """Strip trailing device number suffix: AnkleLDeviceOne -> AnkleLDevice"""
    return re.sub(r'(Device)(One|Two|Three|Four)$', r'\1', label)

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