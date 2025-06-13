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

pNum = input("Enter the participant number: ")

rawData = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/P0" + pNum + "/OuraRing/HeartRate/P001OrHrRAW.csv")

if pNum == "04" or pNum == "05":
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_Fr.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_M-Th.csv")
else:
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")

zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
rawData.insert(0, 'class', "NONE")
rawData.insert(1, 'Time_In_PST', zero_time)

prevDate = convert_iso_to_pacific_date(rawData.iloc[0]['timestamp'])
start_idx = 0
dfList = []

for idx, row in enumerate(rawData.itertuples()):
    currDate = convert_iso_to_pacific_date(row.timestamp)
    if currDate != prevDate:
        dfList.append(rawData.iloc[start_idx:idx])
        start_idx = idx
        prevDate = currDate
dfList.append(rawData.iloc[start_idx:])

csvPathList = []

print(rawData.head)

for df in dfList:
    if df.empty:
        print("HELLO")
        continue

    print("FDSUIHDFSN")
    print(df.head)
    timestamp = convert_iso_to_pacific_date(df.iloc[0]['timestamp'])
    date_str = timestamp.strftime("%Y-%m-%d")
    input("The date is: " + date_str)
    csvPathList = []
    file_path = "/Users/tommoore/Documents/GitHub/Research/P0" + pNum + "/OuraRing/HeartRate/P0" + pNum + "OrHrLabeled" + date_str + ".csv"
    csvPathList.append(file_path)
    with open(file_path, 'w') as f:
        pass

# Add Unix and PST time
# label with schedule datadf.rename(columns={'old_name': 'new_name'}, inplace=True)

# Save to file