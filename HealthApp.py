import pandas as pd
import pytz
import os
from collections import defaultdict







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

