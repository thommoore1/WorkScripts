"""
bin_size_sweep.py

Goal
----
The coverage metric (fraction of scheduled-class bins that contain >=1 HR
sample) is sensitive to bin width: if the bin is narrower than a
participant's typical "quiet time between bursts," coverage looks bad even
when the ring is recording normally. This script re-uses the exact
class-labeling / scheduling pipeline from the coverage script, but sweeps
over a range of candidate bin widths and reports coverage % per
participant at each width, so you can see empirically where the numbers
stop being dominated by burstiness and start reflecting real data loss.

Approach
--------
The expensive part of the original script -- loading raw data, splitting
by day, converting timestamps, labeling each row with its scheduled class,
and dropping DELETE rows -- does NOT depend on bin width. So it's done
ONCE per participant. Only the binning + occupancy-check step is repeated
for each candidate bin width.

Outputs
-------
- bin_size_sweep_results.csv   : Participant x bin_width_min coverage_pct matrix
- bin_size_sweep_curve.png     : coverage % vs bin width, one line per participant
                                  + a bold "average across participants" line
- Printed table to console
"""

from datetime import datetime, timezone, timedelta
import os

import pandas as pd
import numpy as np
import pytz
import matplotlib.pyplot as plt

# === Candidate bin widths to test (minutes) ===
CANDIDATE_BIN_MINUTES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

# === Paths ===
root_path = "/Users/cibrian/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/OuraRing/BinSizeSweep")
os.makedirs(output_folder, exist_ok=True)

participant_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "12", "14", "16"]


# === Shared helper functions (same as coverage script) ===

def convert_timestamp_to_pacific(timestamp):
    pacific_tz = pytz.timezone('America/Los_Angeles')
    dt_utc = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_pacific = dt_utc.astimezone(pacific_tz)
    return dt_pacific.time()


def convert_string_to_time(time_string):
    return datetime.strptime(time_string, "%H:%M:%S").time()


def convert_iso_to_pacific_date(timestamp):
    pacific_tz = pytz.timezone('America/Los_Angeles')
    dt_utc = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_pacific = dt_utc.astimezone(pacific_tz)
    return dt_pacific.date()


def convert_iso_to_unix(iso_timestamp):
    dt = datetime.strptime(iso_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def get_day_of_week(date_obj):
    return date_obj.strftime("%A")


def generate_bins(start_time, end_time, bin_minutes):
    """Split (start_time, end_time] into consecutive bin_minutes-wide bins."""
    base_date = datetime(1900, 1, 1)
    dt = datetime.combine(base_date, start_time)
    end_dt = datetime.combine(base_date, end_time)
    bins = []
    while dt < end_dt:
        bin_end = min(dt + timedelta(minutes=bin_minutes), end_dt)
        bins.append((dt.time(), bin_end.time()))
        dt = bin_end
    return bins


# === Step 1: build labeled dfList + schedulePerDay ONCE per participant ===
# (identical logic to the coverage script, minus the CSV-writing /
#  bin-occupancy steps, which happen later per bin width)

participant_data = {}  # pNum -> {'dfList': [...], 'schedulePerDay': [...]}

for pNum in participant_numbers:
    rawDataPath = f"{root_path}/P0{pNum}/OuraRing/HeartRate/P0{pNum}OrHrRAW.csv"
    if not os.path.exists(rawDataPath):
        print(f"Raw data not found for P0{pNum}, skipping...")
        continue

    print(f"Loading & labeling Participant P0{pNum}...")
    rawData = pd.read_csv(rawDataPath)

    if pNum in ["04", "05", "09", "14", "16"]:
        scheduleDataFri = pd.read_csv(f"{root_path}/Schedules/schedData_P(04,05,09,14,16)_FR.csv")
        scheduleDataOth = pd.read_csv(f"{root_path}/Schedules/schedData_P(04,05,09,14,16)_M-TH.csv")
        if pNum in ['14', '16']:
            scheduleDataTu = pd.read_csv(f"{root_path}/Schedules/schedData_P(14,16)TU.csv")
    else:
        scheduleDataFri = pd.read_csv(f"{root_path}/Schedules/schedData_P(01,02,03,06,07,08,12)_FR.csv")
        scheduleDataOth = pd.read_csv(f"{root_path}/Schedules/schedData_P(01,02,03,06,07,08,12)_M-TH.csv")

    zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
    rawData.insert(0, 'class', "NONE")
    rawData.insert(1, 'Time_In_PST', zero_time)
    rawData.insert(2, 'time', 0)

    prevDate = convert_iso_to_pacific_date(rawData.iloc[0]['timestamp'])
    start_idx = 0
    dfList = []
    for idx, row in enumerate(rawData.itertuples()):
        currDate = convert_iso_to_pacific_date(row.timestamp)
        if currDate != prevDate:
            dfList.append(rawData.iloc[start_idx:idx].copy())
            start_idx = idx
            prevDate = currDate
    dfList.append(rawData.iloc[start_idx:].copy())

    for df in dfList:
        df.loc[:, 'time'] = df['timestamp'].apply(convert_iso_to_unix)
        df.loc[:, 'Time_In_PST'] = df['timestamp'].apply(convert_timestamp_to_pacific)
        df.rename(columns={'timestamp': 'Time_In_ISO'}, inplace=True)

    schedulePerDay = []
    for df in dfList:
        DayOfWeek = get_day_of_week(datetime.fromtimestamp(df.iloc[0]['time']))
        if DayOfWeek == 'Friday':
            scheduleData = scheduleDataFri
        elif DayOfWeek == 'Tuesday' and (pNum == "14" or pNum == "16"):
            scheduleData = scheduleDataTu
        else:
            scheduleData = scheduleDataOth
        schedulePerDay.append(scheduleData)

        for row in df.itertuples():
            for schedRow in scheduleData.itertuples():
                timeA = convert_string_to_time(getattr(schedRow, 'TimeStart'))
                timeB = convert_string_to_time(getattr(schedRow, 'TimeEnd'))
                if timeA < df.at[row.Index, 'Time_In_PST'] <= timeB:
                    df.at[row.Index, 'class'] = getattr(schedRow, 'Class')
                    break

    for i in range(len(dfList)):
        df = dfList[i].copy()
        df.loc[:, 'class'] = df['class'].str.strip()
        df = df[df['class'] != 'DELETE'].reset_index(drop=True)
        dfList[i] = df

    participant_data[pNum] = {'dfList': dfList, 'schedulePerDay': schedulePerDay}

print("\nAll participants loaded & labeled.\n")

# === Step 2: sweep bin widths, computing coverage % per participant each time ===

# results[pNum][bin_minutes] = coverage_pct (after HR filter, matching the
# original script's "clean_true_bins" definition)
results = {pNum: {} for pNum in participant_data}

for bin_minutes in CANDIDATE_BIN_MINUTES:
    print(f"Evaluating bin width = {bin_minutes} min ...")
    for pNum, data in participant_data.items():
        dfList = data['dfList']
        schedulePerDay = data['schedulePerDay']

        total_bins = 0
        clean_true_bins = 0

        for i in range(len(dfList)):
            df = dfList[i]
            scheduleData = schedulePerDay[i]

            for schedRow in scheduleData.itertuples():
                classLabel = str(getattr(schedRow, 'Class')).strip()
                if classLabel == 'DELETE':
                    continue

                timeA = convert_string_to_time(getattr(schedRow, 'TimeStart'))
                timeB = convert_string_to_time(getattr(schedRow, 'TimeEnd'))
                bins = generate_bins(timeA, timeB, bin_minutes)
                total_bins += len(bins)

                for (binStart, binEnd) in bins:
                    in_bin = df[(df['Time_In_PST'] > binStart) & (df['Time_In_PST'] <= binEnd)]
                    if len(in_bin) > 0:
                        clean_in_bin = in_bin[in_bin['bpm'].between(40, 180)]
                        if len(clean_in_bin) > 0:
                            clean_true_bins += 1

        coverage_pct = (clean_true_bins / total_bins * 100) if total_bins > 0 else np.nan
        results[pNum][bin_minutes] = coverage_pct

# === Step 3: assemble results table ===

sweep_df = pd.DataFrame(results).T  # rows = participants, cols = bin widths
sweep_df.index.name = 'Participant'
sweep_df.columns = [f"{b}min" for b in CANDIDATE_BIN_MINUTES]
sweep_df['average_across_bin_widths'] = sweep_df.mean(axis=1)

avg_row = sweep_df.drop(columns='average_across_bin_widths').mean(axis=0)
sweep_df.loc['AVERAGE'] = list(avg_row) + [avg_row.mean()]

csv_path = os.path.join(output_folder, "bin_size_sweep_results.csv")
sweep_df.to_csv(csv_path)
print(f"\nSweep results saved to: {csv_path}")
print(sweep_df.round(1).to_string())

# === Step 4: plot coverage % vs bin width ===

plt.figure(figsize=(11, 7))
for pNum in participant_data:
    coverage_series = [results[pNum][b] for b in CANDIDATE_BIN_MINUTES]
    plt.plot(CANDIDATE_BIN_MINUTES, coverage_series, marker='o', alpha=0.5, label=f"P0{pNum}")

avg_series = [np.mean([results[pNum][b] for pNum in participant_data]) for b in CANDIDATE_BIN_MINUTES]
plt.plot(CANDIDATE_BIN_MINUTES, avg_series, marker='o', color='black', linewidth=3, label="Average")

plt.axvline(5, color='red', linestyle='--', alpha=0.6, label="Original 5-min bin")
plt.xlabel("Bin width (minutes)")
plt.ylabel("Coverage (%) -- bins with >=1 valid HR sample")
plt.title("Coverage % as a Function of Bin Width, Per Participant")
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()

plot_path = os.path.join(output_folder, "bin_size_sweep_curve.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Sweep curve plot saved to: {plot_path}")