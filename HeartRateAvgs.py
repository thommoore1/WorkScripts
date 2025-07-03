import pandas as pd
import pytz
import os
import datetime

def unixToDayOfWeek(unix_time):
    dt = datetime.datetime.fromtimestamp(unix_time, tz=datetime.timezone.utc)
    return dt.strftime('%A')

rootPath = "/Users/tommoore/Documents/GitHub/Research/"
ouraRingPath = "/OuraRing/HeartRate/"
fileName = "OrHrLabeled"
rawDataName = "OrHrRAW.csv"
savePath = "/Users/tommoore/Documents/GitHub/Research/Averages/OrHrAvgs.csv"

participants = ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P012', 'P014', 'P016']

heartRateAverages = pd.DataFrame({
    'participant': participants,
    'Monday avg': 0.0,
    'Tuesday avg': 0.0,
    'Wednesday avg': 0.0,
    'Thursday avg': 0.0,
    'Friday avg': 0.0,
    'Total avg': 0.0,
})

for idx, participant in enumerate(participants):
    dataPath = rootPath + participant + ouraRingPath
    allBPM = []

    for file in os.listdir(dataPath):
        filePath = os.path.join(dataPath, file)
        if file != participant + rawDataName:
            dataFrame = pd.read_csv(filePath)
            avgBPM = dataFrame['bpm'].mean()
            allBPM.extend(dataFrame['bpm'].tolist())
            dayOfWeek = unixToDayOfWeek(dataFrame['time'].iloc[0])

            if dayOfWeek == 'Monday':
                heartRateAverages.at[idx, 'Monday avg'] = avgBPM
            elif dayOfWeek == 'Tuesday':
                heartRateAverages.at[idx, 'Tuesday avg'] = avgBPM
            elif dayOfWeek == 'Wednesday':
                heartRateAverages.at[idx, 'Wednesday avg'] = avgBPM
            elif dayOfWeek == 'Thursday':
                heartRateAverages.at[idx, 'Thursday avg'] = avgBPM
            elif dayOfWeek == 'Friday':
                heartRateAverages.at[idx, 'Friday avg'] = avgBPM

    nonZeroBPMs = [bpm for bpm in allBPM if bpm != 0]
    heartRateAverages.at[idx, 'Total avg'] = sum(nonZeroBPMs) / len(nonZeroBPMs)

print(heartRateAverages)
heartRateAverages.to_csv(savePath, index=False)