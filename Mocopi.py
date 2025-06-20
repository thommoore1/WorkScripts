import pandas as pd
import pytz
import os

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

# Just get the time part
current_time = now.strftime("%H:%M:%S")

print("Current Time:", current_time)



# Prompt for participant number
pNum = input("Enter the participant number: ")

#Gathering parent paths
rawParentPath = f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/Mocopi/Raw"
labeledParentPath = f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/Mocopi/Labeled"

# all of the directories in raw data folder
directories = [d for d in os.listdir(rawParentPath) if os.path.isdir(os.path.join(rawParentPath, d))]

# Group raw CSVs by (sensor, date)
grouped_raw_data = defaultdict(list)

for dir_name in directories:
    folder_path = Path(rawParentPath) / dir_name
    for file in folder_path.iterdir():
        dataFrame = pd.read_csv(file)
        dateTime = datetime.strptime(dataFrame.iloc[0]['Timestamp'], "%Y-%m-%d %H:%M:%S.%f")
        dateOnly = dateTime.date().strftime("%Y-%m-%d")
        sensor_label = getSensorLocation(str(file))
        grouped_raw_data[(sensor_label, dateOnly)].append(dataFrame)

# Combine groups & build save paths
dataFrames = []
csvPathList = []
print("combine things")
for (sensor_label, dateOnly), dfs in grouped_raw_data.items():
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(by="Timestamp").reset_index(drop=True)
    dataFrames.append(combined_df)

    dirPath = os.path.join(labeledParentPath, dateOnly)
    os.makedirs(dirPath, exist_ok=True)

    file_path = os.path.join(dirPath, f"P0{pNum}Mocopi{sensor_label}{dateOnly}.csv")
    csvPathList.append(file_path)
print("schedData")
# Load schedule data
if pNum in ["04", "05"]:
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_Fr.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_M-Th.csv")
else:
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")
print("columns")
# Add time & class columns
zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
for rawData in dataFrames:
    rawData.insert(0, 'class', "NONE")
    rawData.insert(1, 'Time_In_PST', zero_time)
    rawData.insert(2, 'time', 0.0)
print("timestamps")
# Process timestamp columns
for i, dataFrame in enumerate(dataFrames):
    dataFrame.loc[:, 'time'] = dataFrame['Timestamp'].apply(convert_to_unix_time).astype('float64')
    dataFrame.loc[:, 'Time_In_PST'] = dataFrame['Timestamp'].apply(extract_time_only)
    dataFrame.rename(columns={'Timestamp': 'Old Timestamp'}, inplace=True)
    dataFrames[i] = dataFrame.copy()

print("schedule")
print(len(dataFrames))
# Label classes using schedule
for dataFrame in dataFrames:
    DayOfWeek = get_day_of_week(datetime.fromtimestamp(dataFrame.iloc[0]['time']))
    scheduleData = scheduleDataFri if DayOfWeek == 'Friday' else scheduleDataOth

    for row in dataFrame.itertuples():
        for schedRow in scheduleData.itertuples():
            timeA = convert_string_to_time(getattr(schedRow, 'TimeStart'))
            timeB = convert_string_to_time(getattr(schedRow, 'TimeEnd'))
            if timeA < dataFrame.at[row.Index, 'Time_In_PST'] <= timeB:
                dataFrame.at[row.Index, 'class'] = getattr(schedRow, 'Class')
                break
print("schedule done")
#clean and save
for i in range(len(dataFrames)):
    dataFrame = dataFrames[i].copy()
    dataFrame.loc[:, 'class'] = dataFrame['class'].str.strip()
    dataFrame = dataFrame[dataFrame['class'] != 'DELETE'].reset_index(drop=True)
    dataFrame.to_csv(csvPathList[i], index=False)
    dataFrames[i] = dataFrame





now = datetime.now()

# Just get the time part
current_time = now.strftime("%H:%M:%S")

print("Current Time:", current_time)