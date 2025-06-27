import pandas as pd
import pytz
import os
from collections import defaultdict
from datetime import datetime, timezone, date

def convert_date_format(date_str):
    try:
        # Parse date, letting pandas infer format
        dt = pd.to_datetime(date_str, errors='coerce')
        if pd.isna(dt):
            return None
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


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

# Gathering parent paths
rawParentPath = f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Raw/export.csv"
labeledActivityParentPath = f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Labeled/ActivitySummary"
labeledRecordParentPath = f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Labeled/ActivitySummary"

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

activityDF.to_csv(f"/Users/tommoore/Documents/GitHub/Research/P0{pNum}/HealthApp/Labeled/P0{pNum}ActivityLabeled.csv", index=False)

zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
recordDF.insert(0, 'class', "NONE")
recordDF.insert(1, 'Time_In_PST', zero_time)
recordDF.insert(2, 'time', 0.0)

zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
recordDF.insert(0, 'class', "NONE")
recordDF.insert(1, 'Time_In_PST', zero_time)
recordDF.insert(2, 'time', 0.0)

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