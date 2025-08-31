from datetime import datetime, timezone, time
import pytz
import os
import pandas as pd
import numpy as np

pacific_tz = pytz.timezone('America/Los_Angeles')

participant_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "12", "14", "16"]
rootPath = "/Users/tommoore/Documents/GitHub/Research"

def convert_timestamp_to_pacific_vectorized(ts_series):
    # ts_series is in nanoseconds
    dt_utc = pd.to_datetime(ts_series, unit='ns', utc=True)
    dt_pacific = dt_utc.dt.tz_convert(pacific_tz)
    return dt_pacific.dt.time

def convert_string_to_time_series(time_series):
    # Convert a series of strings like 'HH:MM:SS' to time objects
    return pd.to_datetime(time_series, format="%H:%M:%S").dt.time

for pNum in participant_numbers:
    print(f"Processing Participant P0{pNum}...")
    dataPath = os.path.join(rootPath, f"P0{pNum}", "SensorLogger")

    # Load schedules once
    if pNum in ["04", "05", "09", "14", "16"]:
        scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05,09,14,16)_FR.csv")
        scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05,09,14,16)_M-TH.csv")
        scheduleDataTu = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(14,16)TU.csv") if pNum in ['14','16'] else None
    else:
        scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,12)_FR.csv")
        scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,12)_M-TH.csv")
        scheduleDataTu = None

    # Find leaf folders
    subFolders = [root for root, dirs, files in os.walk(dataPath) if not dirs]

    # Collect raw data paths
    rawDataPaths = []
    for subFolder in subFolders:
        rawDataPaths.extend([
            os.path.join(subFolder, f)
            for f in os.listdir(subFolder)
            if os.path.isfile(os.path.join(subFolder, f)) and "LABELED" not in f
        ])

    for rawDataPath in rawDataPaths:
        rawData = pd.read_csv(rawDataPath)

        # Convert timestamps to PST vectorized
        rawData['Time_In_PST'] = convert_timestamp_to_pacific_vectorized(rawData['time'])

        # Determine day of week
        day_of_week = pd.to_datetime(rawData['time'].iloc[0], unit='ns').day_name()
        if day_of_week == 'Friday':
            scheduleData = scheduleDataFri
        elif day_of_week == 'Tuesday' and scheduleDataTu is not None:
            scheduleData = scheduleDataTu
        else:
            scheduleData = scheduleDataOth

        # Convert schedule times once
        scheduleData['TimeStart'] = convert_string_to_time_series(scheduleData['TimeStart'])
        scheduleData['TimeEnd'] = convert_string_to_time_series(scheduleData['TimeEnd'])

        # Initialize class column
        rawData['class'] = "NONE"

        # Vectorized assignment
        for _, schedRow in scheduleData.iterrows():
            mask = (rawData['Time_In_PST'] > schedRow['TimeStart']) & (rawData['Time_In_PST'] <= schedRow['TimeEnd'])
            rawData.loc[mask, 'class'] = schedRow['Class']

        # Remove rows marked DELETE
        rawData = rawData[rawData['class'] != 'DELETE']

        savePath = os.path.dirname(rawDataPath)
        dirList = savePath.split(os.sep)
        if not rawData.empty:
            saveLocation = f"{savePath}/P0{pNum}SensorLog_TRUE_{dirList[9]}_{pd.to_datetime(rawData['time'].iloc[0], unit='ns').strftime('%Y_%m_%d')}.csv"
        else:
            date = dirList[8]
            date_obj = datetime.strptime(date, "%b%d")
            formatted_date = date_obj.replace(year=2025).strftime("%m_%d_%Y")
            saveLocation = f"{savePath}/P0{pNum}SensorLog_TRUE_{dirList[9]}_{formatted_date}.csv"
        rawData.to_csv(saveLocation, index=False)
