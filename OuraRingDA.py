from datetime import datetime, timezone, date
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
    time_obj = datetime.strptime(time_string, "%H:%M:%S").time()
    return time_obj

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

pNum = input("Enter the participant number: ")

rawData = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/P0" + pNum + "/OuraRing/DailyActivity/P0" + pNum + "OrHrRAW.csv")

if pNum == "04" or pNum == "05":
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_Fr.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_M-Th.csv")
else:
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")

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
    file_path = "/Users/tommoore/Documents/GitHub/Research/P0" + pNum + "/OuraRing/HeartRate/P0" + pNum + "OrHrLabeled" + date_str + ".csv"
    csvPathList.append(file_path)
    with open(file_path, 'w') as f:
        pass

for df in dfList:
    df.loc[:, 'time'] = df['timestamp'].apply(convert_iso_to_unix)
    df.loc[:, 'Time_In_PST'] = df['timestamp'].apply(convert_timestamp_to_pacific)
    df.rename(columns={'timestamp': 'Time_In_ISO'}, inplace=True)
    df = df.copy()

for df in dfList:
    DayOfWeek = get_day_of_week(datetime.fromtimestamp(df.iloc[0]['time']))
    if DayOfWeek == 'Friday':
        scheduleData = scheduleDataFri
    else:
        scheduleData = scheduleDataOth

    for row in df.itertuples():
        for schedRow in scheduleData.itertuples():
            timeA = convert_string_to_time(getattr(schedRow, 'TimeStart'))
            timeB = convert_string_to_time(getattr(schedRow, 'TimeEnd'))
            if  timeA < df.at[row.Index, 'Time_In_PST'] <= timeB:
                df.at[row.Index, 'class'] = getattr(schedRow, 'Class')
                break

for i in range(len(dfList)):
    df = dfList[i].copy()
    df.loc[:, 'class'] = df['class'].str.strip()
    df = df[df['class'] != 'DELETE'].reset_index(drop=True)
    df.to_csv(csvPathList[i], index=False)
    dfList[i] = df