from datetime import datetime
import pytz
import os
import pandas as pd

pacific_tz = pytz.timezone('America/Los_Angeles')
participant_numbers = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "12", "14", "16"]
rootPath = "/Users/cibrian/Documents/GitHub/Research"
schedPath = os.path.join(rootPath, "Schedules")

def to_time(series):
    return pd.to_datetime(series, format="%H:%M:%S").dt.time

def load_sched(name):
    df = pd.read_csv(os.path.join(schedPath, name))
    df['TimeStart'] = to_time(df['TimeStart'])
    df['TimeEnd'] = to_time(df['TimeEnd'])
    return df

for pNum in participant_numbers:
    print(f"Processing Participant P0{pNum}...")
    dataPath = os.path.join(rootPath, f"P0{pNum}", "SensorLogger")  # verify prefix

    scheduleDataTu = None
    if pNum in ["04", "05", "09", "14", "16"]:
        scheduleDataFri = load_sched("schedData_P(04,05,09,14,16)_FR.csv")
        scheduleDataOth = load_sched("schedData_P(04,05,09,14,16)_M-TH.csv")
        if pNum in ["14", "16"]:
            scheduleDataTu = load_sched("schedData_P(14,16)TU.csv")
    else:
        scheduleDataFri = load_sched("schedData_P(01,02,03,06,07,08,12)_FR.csv")
        scheduleDataOth = load_sched("schedData_P(01,02,03,06,07,08,12)_M-TH.csv")

    subFolders = [root for root, dirs, files in os.walk(dataPath) if not dirs]
    rawDataPaths = [
        os.path.join(sf, f)
        for sf in subFolders
        for f in os.listdir(sf)
        if os.path.isfile(os.path.join(sf, f)) and "LABELED" not in f and "TRUE" not in f
    ]

    for rawDataPath in rawDataPaths:
        rawData = pd.read_csv(rawDataPath)
        savePath = os.path.dirname(rawDataPath)
        dirList = savePath.split(os.sep)

        # Empty file: save with date derived from folder name
        if rawData.empty:
            date_obj = datetime.strptime(dirList[8], "%b%d").replace(year=2025)
            saveLocation = f"{savePath}/P{pNum}SensorLog_TRUE_{dirList[9]}_{date_obj.strftime('%m_%d_%Y')}.csv"
            rawData.to_csv(saveLocation, index=False)
            continue

        # Timestamps -> Pacific
        dt_pacific = pd.to_datetime(rawData['time'], unit='ns', utc=True).dt.tz_convert(pacific_tz)
        rawData['Time_In_PST'] = dt_pacific.dt.time

        first_dt = dt_pacific.iloc[0]
        day_of_week = first_dt.day_name()

        if day_of_week == 'Friday':
            scheduleData = scheduleDataFri
        elif (day_of_week == 'Tuesday' and scheduleDataTu is not None
              and first_dt.date() != datetime(2025, 4, 1).date()):
            scheduleData = scheduleDataTu
        else:
            scheduleData = scheduleDataOth

        rawData['class'] = "NONE"
        for _, schedRow in scheduleData.iterrows():
            mask = (rawData['Time_In_PST'] > schedRow['TimeStart']) & \
                   (rawData['Time_In_PST'] <= schedRow['TimeEnd'])
            rawData.loc[mask, 'class'] = schedRow['Class']

        rawData = rawData[rawData['class'] != 'DELETE']

        date_str = first_dt.strftime('%Y_%m_%d')
        saveLocation = f"{savePath}/P0{pNum}SensorLog_TRUE_{dirList[9]}_{date_str}.csv"
        rawData.to_csv(saveLocation, index=False)