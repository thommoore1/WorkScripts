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

# rawDataPath = input("Enter the file path of the raw data: ")
# rawData = pd.read_csv(rawDataPath)

# scheduleDataPath = input("Enter the file path of the schedule data: ")
# scheduleData = pd.read_csv(scheduleDataPath)

# saveLocation = input("Enter file path of where you would like to save to: ")

# zero_time = datetime(1900, 1, 1, 0, 0, 0).time()
# rawData.insert(0, 'class', "NONE")
# rawData.insert(1, 'Time_In_PST', zero_time)

pNum = input("Enter the participant number: ")

rawData = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/P0" + pNum + "/OuraRing/HeartRate/P001OrHrRAW.csv")

if pNum == "04" OR pNum == "05":
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_Fr.csv")
    scdhuleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(04,05)_M-Th.csv")
else:
    scheduleDataFri = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_FR.csv")
    scdhuleDataOth = pd.read_csv("/Users/tommoore/Documents/GitHub/Research/Schedules/schedData_P(01,02,03,06,07,08,09,12,14,16)_M-TH.csv")

dfList = []