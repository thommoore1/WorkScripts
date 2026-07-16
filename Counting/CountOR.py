from datetime import datetime, timezone
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

participant_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "12", "14", "16"]

# --- Tracking totals across participants ---
all_total_counts = []
all_clean_counts = []

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

    for df in dfList:
        DayOfWeek = get_day_of_week(datetime.fromtimestamp(df.iloc[0]['time']))
        if DayOfWeek == 'Friday':
            scheduleData = scheduleDataFri
        elif DayOfWeek == 'Tuesday' and (pNum == "14" or pNum == "16"):
            scheduleData = scheduleDataTu
        else:
            scheduleData = scheduleDataOth

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

    # --- Count datapoints during valid class times ---
    # Combine all daily DataFrames for this participant
    combined = pd.concat(dfList, ignore_index=True)

    # Valid class time = any row not labelled "NONE"
    in_class = combined[combined['class'] != 'NONE']
    total_count = len(in_class)

    # Clean = also within valid HR range
    clean_count = len(in_class[in_class['bpm'].between(40, 180)])

    all_total_counts.append(total_count)
    all_clean_counts.append(clean_count)

    print(f"  P0{pNum} — in-class datapoints: {total_count}, after HR filter (40–180 bpm): {clean_count}")
    print(f"Done for Participant P0{pNum}")

print("\nAll participants processed.")
print(f"\n--- Summary across {len(all_total_counts)} participants ---")
print(f"  Avg datapoints during class time:              {sum(all_total_counts) / len(all_total_counts):.1f}")
print(f"  Avg datapoints after bad-value removal:        {sum(all_clean_counts)  / len(all_clean_counts):.1f}")