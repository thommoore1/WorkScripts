"""
config.py
---------
Central, single place to edit paths and settings used by BOTH:
  - e2_coverage_missingness.py
  - e3_activity_hr_effect.py

Edit ROOT_PATH and OUTPUT_FOLDER below. Nothing else needs to change unless
your folder/file naming conventions differ from what's described here.
"""

import os

# ============================================================
# EDIT THESE TWO PATHS
# ============================================================
ROOT_PATH = "/Users/cibrian/Documents/Github/Research"

# All output images (tables + heatmaps) get written here.
# Change this one line to redirect ALL output from both scripts.
OUTPUT_FOLDER = os.path.join(ROOT_PATH, "1_visualization", "Coverage_and_Activity")
# ============================================================

SCHEDULE_FOLDER = os.path.join(ROOT_PATH, "Schedules")

# Timestamps in the HeartRate CSVs are epoch seconds, UTC.
# This is the local timezone they get converted to (DST-aware).
TIMEZONE = "America/Los_Angeles"

# ---- School-day time-of-day window (E2, Table 1) ----
SCHOOL_DAY_START = "08:30:00"
SCHOOL_DAY_END = "15:00:00"
CHUNK_MINUTES = 30

# ---- Valid classroom activity labels ----
# Any other class value (including "DELETE" or blank) is excluded from
# both the E2 activity table and the E3 classroom-time analysis.
VALID_CLASSES = [
    "Cash-out",
    "ELA",
    "History",
    "Homeroom",
    "Math",
    "Social Skills",
    "HW Rein./Study Hall",
    "Lunch",
    "PE",
]

# ---- Oura daily-activity array decoding ----
# ASSUMPTION (confirmed via example data): Oura's class_5_min / met_1_min
# arrays are anchored to a 4:00 AM local "activity day" start, not midnight.
# This was inferred from your example class_5_min string (63 leading zero
# bins before activity starts = 5h15m after 4:00 AM = ~9:15 AM, which lines
# up with the 8:30 school start). Both scripts save a diagnostic plot
# (E3 only) so you can visually double check this before trusting results.
ACTIVITY_ARRAY_START_HOUR = 4

# ---- Oura class_5_min code -> activity-intensity group mapping ----
# Confirmed grouping: {0,1}=sedentary/rest, {2}=light, {3,4}=active
# (0 = non-wear, 1 = rest, 2 = inactive, 3 = low activity, 4 = medium/high)
ACTIVITY_GROUP_MAP = {
    0: "sedentary",
    1: "sedentary",
    2: "light",
    3: "active",
    4: "active",
}

# ---- Schedule weekday-assignment ASSUMPTION ----
# Confirmed: Friday uses the "_FR.csv" schedule; all other weekdays use
# "_M-TH.csv" -- EXCEPT participants P014 and P016, who use the special
# "schedData_P(14,16)TU.csv" file on Tuesdays specifically (not Mon/Wed/Thu).
# This exception was inferred from the filename you gave and has NOT been
# explicitly re-confirmed in the latest round -- please double check this
# is correct (see utils.get_schedule_path).

os.makedirs(OUTPUT_FOLDER, exist_ok=True)