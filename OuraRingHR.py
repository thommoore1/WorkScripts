from datetime import datetime, timezone
import pytz
import pandas as pd

def convert_timestamp_to_pacific(timestamp):
    pacific_tz = pytz.timezone('America/Los_Angeles')
    dt_utc = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_pacific = dt_utc.astimezone(pacific_tz)
    return dt_pacific.time()

def convert_string_to_time(time_string):
    time_obj = datetime.strptime(time_string, "%H:%M:%S").time()
    return time_obj

rawDataPath = input("Enter the file path of the raw data: ")
rawData = pd.read_csv(rawDataPath)

scheduleDataPath = input("Enter the file path of the schedule data: ")
scheduleData = pd.read_csv(scheduleDataPath)

zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
rawData.insert(0, 'class', "NONE")
rawData.insert(1, 'Time_In_PST', zero_time)