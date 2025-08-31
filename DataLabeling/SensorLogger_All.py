from datetime import datetime, timezone
import pytz
import os

import pandas as pd

def get_day_of_week(date_obj):
    return date_obj.strftime("%A")

def convert_timestamp_to_pacific(timestamp):
    pacific_tz = pytz.timezone('America/Los_Angeles')
    dt_utc = datetime.fromtimestamp(timestamp / 1e9, timezone.utc)
    dt_pacific = dt_utc.astimezone(pacific_tz)
    return dt_pacific.time()

def convert_string_to_time(time_string):
    time_obj = datetime.strptime(time_string, "%H:%M:%S").time()
    return time_obj

participant_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "12", "14", "16"]
rootPath = "/Users/tommoore/Documents/GitHub/Research"

for pNum in participant_numbers:
    print(f"Processing Participant P0{pNum}...")

    dataPath = rootPath + "/P0" + pNum + "/SensorLogger"
    if pNum in ["04", "05", "09", "14", "16"]:
        scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05,09,14,16)_FR.csv")
        scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05,09,14,16)_M-TH.csv")
        if pNum in ['14', '16']:
            scheduleDataTu = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(14,16)TU.csv")
    else:
        scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,12)_FR.csv")
        scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,12)_M-TH.csv")

    subFolders = []
    for root, dirs, files in os.walk(dataPath):
        if not dirs:
            subFolders.append(root)

    
    rawDataPaths = []
    for subFolder in subFolders:
        for filename in os.listdir(subFolder):
            file_path = os.path.join(subFolder, filename)
            if os.path.isfile(file_path) and "LABELED" not in filename:
                rawDataPaths.append(file_path)

    for rawDataPath in rawDataPaths:
        rawData = pd.read_csv(rawDataPath)
        savePath = os.path.dirname(rawDataPath)
        directories = savePath if os.path.isdir(savePath) else os.path.dirname(savePath)
        dirList = directories.split(os.sep)
        #print(rawData.iloc[0]['time'])
        #print(datetime.fromtimestamp(rawData.iloc[0]['time']).date())
        saveLocation = f"{savePath}/P0{pNum}SensorLog{dirList[8]}_{datetime.fromtimestamp(rawData.iloc[0]['time']/1e9).strftime('%Y_%m_%d')}.csv"


        print(saveLocation)
        DayOfWeek = get_day_of_week(datetime.fromtimestamp(rawData.iloc[0]['time'] / 1e9))
        if DayOfWeek == 'Friday':
            scheduleData = scheduleDataFri
        elif DayOfWeek == 'Tuesday' and (pNum == "14" or pNum == "16"):
            scheduleData = scheduleDataTu
        else:
            scheduleData = scheduleDataOth

        zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
        rawData.insert(0, 'class', "NONE")
        rawData.insert(1, 'Time_In_PST', zero_time)

        print("stoop")
        for row in rawData.itertuples():
            print(row)
            rawData.at[row.Index, 'Time_In_PST'] = convert_timestamp_to_pacific(getattr(row, 'time'))
            for schedRow in scheduleData.itertuples():
                timeA = convert_string_to_time(getattr(schedRow, 'TimeStart'))
                timeB = convert_string_to_time(getattr(schedRow, 'TimeEnd'))
                if  timeA < rawData.at[row.Index, 'Time_In_PST'] <= timeB:
                    rawData.at[row.Index, 'class'] = getattr(schedRow, 'Class')
                    break
        rawData = rawData[rawData['class'] != 'DELETE']
        print(rawData.head)

        #rawData.to_csv(saveLocation, index=False)