from datetime import datetime, timezone, timedelta
import pytz
import pandas as pd
import os

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

# --- New: bin generation helper ---
BIN_MINUTES = 5

def generate_bins(start_time, end_time, bin_minutes=BIN_MINUTES):
    """
    Split the half-open interval (start_time, end_time] into consecutive
    bin_minutes-wide bins. Returns a list of (bin_start, bin_end) time tuples.
    The final bin may be shorter than bin_minutes if the interval doesn't
    divide evenly.
    """
    base_date = datetime(1900, 1, 1)
    dt = datetime.combine(base_date, start_time)
    end_dt = datetime.combine(base_date, end_time)
    bins = []
    while dt < end_dt:
        bin_end = min(dt + timedelta(minutes=bin_minutes), end_dt)
        bins.append((dt.time(), bin_end.time()))
        dt = bin_end
    return bins

participant_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "12", "14", "16"]

# --- Tracking totals across participants ---
all_total_bins = []       # total number of 5-min class bins per participant
all_true_bins = []        # bins with >=1 datapoint (before HR filtering)
all_clean_true_bins = []  # bins with >=1 datapoint after HR filtering (40-180 bpm)

for pNum in participant_numbers:
    print(f"Processing Participant P0{pNum}...")

    rawDataPath = f"/Users/cibrian/Documents/GitHub/Research/P0{pNum}/OuraRing/HeartRate/P0{pNum}OrHrRAW.csv"
    if not os.path.exists(rawDataPath):
        print(f"Raw data not found for P0{pNum}, skipping...")
        continue

    rawData = pd.read_csv(rawDataPath)

    if pNum in ["04", "05", "09", "14", "16"]:
        scheduleDataFri = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(04,05,09,14,16)_FR.csv")
        scheduleDataOth = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(04,05,09,14,16)_M-TH.csv")
        if pNum in ['14', '16']:
            scheduleDataTu = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(14,16)TU.csv")
    else:
        scheduleDataFri = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,12)_FR.csv")
        scheduleDataOth = pd.read_csv("/Users/cibrian/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,12)_M-TH.csv")

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

    csvPathList = []

    for df in dfList:
        timestamp = convert_iso_to_pacific_date(df.iloc[0]['timestamp'])
        date_str = timestamp.strftime("%Y-%m-%d")
        file_path = f"/Users/cibrian/Documents/GitHub/Research/P0{pNum}/OuraRing/HeartRate/P0{pNum}OrHrLabeled{date_str}.csv"
        csvPathList.append(file_path)
        with open(file_path, 'w') as f:
            pass

    for df in dfList:
        df.loc[:, 'time'] = df['timestamp'].apply(convert_iso_to_unix)
        df.loc[:, 'Time_In_PST'] = df['timestamp'].apply(convert_timestamp_to_pacific)
        df.rename(columns={'timestamp': 'Time_In_ISO'}, inplace=True)

    # Keep track of which scheduleData applies to each day's df, so we can
    # reuse it later for bin generation without recomputing DayOfWeek logic.
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
        df.to_csv(csvPathList[i], index=False)
        dfList[i] = df

    # --- Bin class time into 5-minute bins and check occupancy ---
    participant_total_bins = 0
    participant_true_bins = 0
    participant_clean_true_bins = 0

    for i in range(len(dfList)):
        df = dfList[i]
        scheduleData = schedulePerDay[i]

        for schedRow in scheduleData.itertuples():
            classLabel = str(getattr(schedRow, 'Class')).strip()
            if classLabel == 'DELETE':
                continue

            timeA = convert_string_to_time(getattr(schedRow, 'TimeStart'))
            timeB = convert_string_to_time(getattr(schedRow, 'TimeEnd'))
            bins = generate_bins(timeA, timeB)
            participant_total_bins += len(bins)

            for (binStart, binEnd) in bins:
                in_bin = df[(df['Time_In_PST'] > binStart) & (df['Time_In_PST'] <= binEnd)]
                if len(in_bin) > 0:
                    participant_true_bins += 1

                    clean_in_bin = in_bin[in_bin['bpm'].between(40, 180)]
                    if len(clean_in_bin) > 0:
                        participant_clean_true_bins += 1

    all_total_bins.append(participant_total_bins)
    all_true_bins.append(participant_true_bins)
    all_clean_true_bins.append(participant_clean_true_bins)

    print(f"  P0{pNum} — total 5-min class bins: {participant_total_bins}")
    print(f"  P0{pNum} — true bins (has data) before HR filter: {participant_true_bins}")
    print(f"  P0{pNum} — true bins (has valid data) after HR filter (40-180 bpm): {participant_clean_true_bins}")
    print(f"Done for Participant P0{pNum}")

print("\nAll participants processed.")
print(f"\n--- Summary across {len(all_true_bins)} participants ---")
print(f"  Avg total 5-min class bins:                         {sum(all_total_bins) / len(all_total_bins):.1f}")
print(f"  Avg true bins before HR filtering:                  {sum(all_true_bins) / len(all_true_bins):.1f}")
print(f"  Avg true bins after HR filtering (40-180 bpm):      {sum(all_clean_true_bins) / len(all_clean_true_bins):.1f}")