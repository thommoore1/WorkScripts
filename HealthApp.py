import pandas as pd
import pytz
import os
from collections import defaultdict
from datetime import datetime, timezone, date

def convert_date_format(date_str):
    try:
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None
    
def time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000

def get_day_of_week(date_obj):
    return date_obj.strftime("%A")

def filter_dates_for_participant(df, participant_id, date_column):
    allowed_dates = participants_dates.get(participant_id, set())
    converted_dates = df[date_column].astype(str).apply(convert_date_format)
    filtered_df = df[converted_dates.isin(allowed_dates)].reset_index(drop=True)
    
    return filtered_df


participants_dates = {
    "01": {"2025-02-03", "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-07"},
    "02": {"2025-02-03", "2025-02-04", "2025-02-05"},
    "03": {"2025-02-03", "2025-02-04"},
    "04": {"2025-02-10", "2025-02-11", "2025-02-12", "2025-02-13", "2025-02-14"},
    "05": {"2025-02-10", "2025-02-11", "2025-02-12"},
    "06": {"2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"},
    "07": {"2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"},
    "08": {"2025-02-24", "2025-02-25", "2025-02-26", "2025-02-27", "2025-02-28"},
    "09": {"2025-02-03", "2025-02-04", "2025-02-05", "2025-02-06", "2025-02-07"},
    "12": {"2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06", "2025-03-07"},
    "14": {"2025-03-25", "2025-03-26", "2025-03-27", "2025-03-31", "2025-04-01"},
    "16": {"2025-03-25", "2025-03-26", "2025-03-27", "2025-03-31", "2025-04-01"},
}




# Prompt for participant number
pNum = input("Enter the participant number: ")

# Load schedule data
if pNum in ["04", "05"]:
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_Fr.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_M-Th.csv")
else:
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
    scheduleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")

# Gathering parent paths
rawParentPath = f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Raw/export.csv"
labeledActivityParentPath = f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Labeled/ActivitySummary"
labeledRecordParentPath = f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Labeled/ActivitySummary"

# Ensure the Record directory exists
recordDir = f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Labeled/Record"
os.makedirs(recordDir, exist_ok=True)

# Make dataframe of entire csv (will split later)
majorDF = pd.read_csv(rawParentPath, low_memory=False)

# Drop uneccessary colummns
majorDF.drop(columns=["/@locale"], inplace=True)
majorDF.drop(columns=[col for col in majorDF.columns if col.startswith("/Me/")], inplace=True)
majorDF.drop(columns=[col for col in majorDF.columns if col.startswith("/Workout/")], inplace=True)

# Making DF for activity data
activityCols = [col for col in majorDF.columns if col.startswith("/ActivitySummary/")]
activityDF = majorDF[activityCols]

# Making DF for record data
recordCols = [col for col in majorDF.columns if col.startswith("/Record/")]
recordDF = majorDF[recordCols]

# Delete empty rows
activityDF = activityDF.dropna(how='all')
recordDF = recordDF.dropna(how='all')

# Reset index
activityDF = activityDF.reset_index(drop=True)
recordDF = recordDF.reset_index(drop=True)

activityDF = filter_dates_for_participant(activityDF, pNum, "/ActivitySummary/@dateComponents")
recordDF = filter_dates_for_participant(recordDF, pNum, "/Record/@startDate")

#renaming and removing unecessary columns
activityDF = activityDF.rename(columns={'/ActivitySummary/@activeEnergyBurned': 'ActiveEnergyBurned'})
activityDF.drop(columns=["/ActivitySummary/@activeEnergyBurnedGoal"], inplace=True)
activityDF = activityDF.rename(columns={'/ActivitySummary/@activeEnergyBurnedUnit': 'ActiveEnergyBurnedUnit'})
activityDF = activityDF.rename(columns={'/ActivitySummary/@appleExerciseTime': 'ExerciseTime'})
activityDF.drop(columns=["/ActivitySummary/@appleExerciseTimeGoal"], inplace=True)
activityDF.drop(columns=["/ActivitySummary/@appleMoveTime"], inplace=True)
activityDF.drop(columns=["/ActivitySummary/@appleMoveTimeGoal"], inplace=True)
activityDF = activityDF.rename(columns={'/ActivitySummary/@appleStandHours': 'StandOurs'})
activityDF.drop(columns=["/ActivitySummary/@appleStandHoursGoal"], inplace=True)
activityDF = activityDF.rename(columns={'/ActivitySummary/@dateComponents': 'date'})

# Saving activity DF
activityDF.to_csv(f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Labeled/P0{pNum}ActivityLabeled.csv", index=False)

# Adding columns to record DF
zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
recordDF.insert(0, 'class', "NONE")
recordDF.insert(1, 'Time_In_PST', zero_time)
recordDF.insert(2, 'time', 0.0)

# Creating list of the different data frames
recordDF = recordDF.sort_values(by='/Record/@startDate').reset_index(drop=True)
prevDate = convert_date_format(recordDF.iloc[0]['/Record/@startDate'])
start_idx = 0
dfList = []


#Separating based on date
for idx, row in recordDF.iterrows():
    currDate = convert_date_format(row['/Record/@startDate'])
    if currDate != prevDate:
        dfList.append(recordDF.iloc[start_idx:idx].copy())
        start_idx = idx
        prevDate = currDate
dfList.append(recordDF.iloc[start_idx:].copy())

#adding time columns
for i, df in enumerate(dfList):
    dt = pd.to_datetime(df['/Record/@startDate'], errors='coerce')
    dt = dt.dt.tz_convert('US/Pacific')
    df['Time_In_PST'] = dt.dt.time
    df['time'] = dt.dt.floor('s').astype('int64') // 10**9
    dfList[i] = df

for dataFrame in dfList:
    day_of_week = get_day_of_week(datetime.fromtimestamp(dataFrame.iloc[0]['time']))
    schedule = scheduleDataFri if day_of_week == 'Friday' else scheduleDataOth

    schedule = schedule.copy()
    schedule['TimeStart'] = pd.to_datetime(schedule['TimeStart'], format="%H:%M:%S").dt.time
    schedule['TimeEnd']   = pd.to_datetime(schedule['TimeEnd'],   format="%H:%M:%S").dt.time

    schedule['TimeStart_sec'] = schedule['TimeStart'].apply(time_to_seconds)
    schedule['TimeEnd_sec']   = schedule['TimeEnd'].apply(time_to_seconds)

    time_values_sec = dataFrame['Time_In_PST'].apply(time_to_seconds)

    intervals = pd.IntervalIndex.from_arrays(
    schedule['TimeStart_sec'],
    schedule['TimeEnd_sec'],
    closed='right'
    )

    import numpy as np
    matched_class = np.full(len(time_values_sec), None, dtype=object)
    for i, interval in enumerate(intervals):
        mask = (interval.left < time_values_sec) & (time_values_sec <= interval.right)
        matched_class = np.where(mask, schedule.iloc[i]['Class'], matched_class)

    dataFrame['class'] = matched_class

csvPathList = []

for df in dfList:
    date = convert_date_format(df['/Record/@startDate'].iloc[0])
    csvPathList.append(f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Labeled/Record/P0{pNum}HealthAppRecord{date}.csv")

for i in range(len(dfList)):
    dataFrame = dfList[i].copy()
    dataFrame.loc[:, 'class'] = dataFrame['class'].str.strip()
    dataFrame = dataFrame[dataFrame['class'] != 'DELETE'].reset_index(drop=True)
    dataFrame.to_csv(csvPathList[i], index=False)
    dfList[i] = dataFrame