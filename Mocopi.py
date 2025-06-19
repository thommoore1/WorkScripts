from datetime import datetime, timezone, date
import pytz
import pandas as pd
import os
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
        "11CCD": "HeadDevice1",
        "132D3": "HeadDevice2",
        "1092A": "HeadDevice3",
        "13CF2": "HeadDevice4",
        "12144": "HipDevice1",
        "114C8": "HipDevice2",
        "10B1F": "HipDevice3",
        "1211E": "HipDevice4",
        "OE3E9": "WristRDevice1",
        "OEE55": "WristRDevice2",
        "12801": "WristRDevice3",
        "0EA70": "WristRDevice4",
        "14A51": "WristLDevice1",
        "134F5": "WristLDevice2",
        "1447A": "WristLDevice3",
        "14A53": "WristLDevice4",
        "1503C": "AnkleRDevice1",
        "13B8F": "AnkleRDevice2",
        "13B06": "AnkleRDevice3",
        "158A6": "AnkleRDevice4",
        "16E17": "AnkleLDevice1",
        "16FB1": "AnkleLDevice2",
        "142A8": "AnkleLDevice3",
        "16CA7": "AnkleLDevice4",
    }

    for code, label in mapping.items():
        if code in fileName:
            return label
    return None

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
    dirPath = labeledParentPath + "/date_only"
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    file_path = "/Users/tommoore/Documents/GitHub/Research/P0" + pNum + "/Mocopi/Labeled/" + dateOnly + "/P0" + pNum + "Mocopi" + getSensorLocation(fileName) + dateOnly + ".csv"
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