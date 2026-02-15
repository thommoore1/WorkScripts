from datetime import datetime, timezone
import pytz
import pandas as pd

def convert_timestamp_to_pacific(timestamp):
    pacific_tz = pytz.timezone('America/Los_Angeles')
    dt_utc = datetime.fromtimestamp(timestamp / 1e9, timezone.utc)
    dt_pacific = dt_utc.astimezone(pacific_tz)
    return dt_pacific.time()

def convert_string_to_time(time_string):
    time_obj = datetime.strptime(time_string, "%H:%M:%S").time()
    return time_obj

rawDataPath = input("Enter the file path of the raw data: ")
rawData = pd.read_csv(rawDataPath)

scheduleDataPath = input("Enter the file path of the schedule data: ")
scheduleData = pd.read_csv(scheduleDataPath)

saveLocation = input("Enter file path of where you would like to save to: ")

zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
rawData.insert(0, 'class', "NONE")
rawData.insert(1, 'Time_In_PST', zero_time)

for row in rawData.itertuples():
    rawData.at[row.Index, 'Time_In_PST'] = convert_timestamp_to_pacific(getattr(row, 'time'))
    for schedRow in scheduleData.itertuples():
        timeA = convert_string_to_time(getattr(schedRow, 'TimeStart'))
        timeB = convert_string_to_time(getattr(schedRow, 'TimeEnd'))
        if  timeA < rawData.at[row.Index, 'Time_In_PST'] <= timeB:
            rawData.at[row.Index, 'class'] = getattr(schedRow, 'Class')
            break
rawData = rawData[rawData['class'] != 'DELETE']
print(rawData.head)

rawData.to_csv(saveLocation, index=False)