import pandas as pd
import pytz
import os

from datetime import datetime, timezone, date
from collections import defaultdict
from pathlib import Path

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
        "OEA70": "WristRDeviceFour",
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

#Gathering file location stuff
pNum = input("Enter the participant number: ")
rawParentPath = "/Users/tommoore/Documents/GitHub/Research/P0" + pNum + "/Mocopi/Raw"
directories = [d for d in os.listdir(rawParentPath) if os.path.isdir(os.path.join(rawParentPath, d))]
rawDataDFs = []
rawDataCSVNames = []

#Storing different csv in to list
for dir_name in directories:
    folder_path = Path(rawParentPath) / dir_name
    for file in folder_path.iterdir():
        rawDataCSVNames.append(file)
        df = pd.read_csv(file)
        rawDataDFs.append(df)

#Storing schedule path depending on P#
if pNum == "04" or pNum == "05":
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_Fr.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_M-Th.csv")
else:
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")

#adding time and class columns
zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
for rawData in rawDataDFs:
    rawData.insert(0, 'class', "NONE")
    rawData.insert(1, 'Time_In_PST', zero_time)
    rawData.insert(2, 'time', 0)

#Creating locations for saving
labeledParentPath = "/Users/tommoore/Documents/GitHub/Research/P0" + pNum + "/Mocopi/Labeled"
LabeledPathList = []
for rawData, fileName in zip(rawDataDFs, rawDataCSVNames):
    dt = datetime.strptime(rawData.iloc[0]['Timestamp'], "%Y-%m-%d %H:%M:%S.%f")
    dateOnly = dt.date().strftime("%Y-%m-%d")
    dirPath = labeledParentPath + "/" + dateOnly 
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    file_path = "/Users/tommoore/Documents/GitHub/Research/P0" + pNum + "/Mocopi/Labeled/" + dateOnly + "/P0" + pNum + "Mocopi" + getSensorLocation(str(fileName)) + "_" + dateOnly + ".csv"
    LabeledPathList.append(file_path)
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