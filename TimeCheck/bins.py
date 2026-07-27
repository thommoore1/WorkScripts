"""
check_expected_bin_parity.py

Goal
----
Participants are grouped by which schedule file they use:
    Group A: P01, P02, P03, P06, P07, P08, P12
    Group B: P04, P05, P09
    Group C: P14, P16

Within a group, everyone follows the same class schedule (same TimeStart/
TimeEnd windows per day-of-week). Specific calendar DATES don't need to
match across participants (different participants may have joined on
different weeks) -- what matters is how many days of EACH WEEKDAY TYPE
(Monday, Tuesday, ... Friday) each participant has data for. If two
participants in the same group have the same count of Mondays, same count
of Tuesdays, etc., their total_bins should be identical (since total_bins
is purely schedule-time-per-weekday x number of days of that weekday
present). If total_bins differs, it should be fully explained by a
difference in weekday-type counts -- i.e. a missing day of some weekday
type. If it's not explained by that, something else is going on.

Approach
--------
  1. For each participant, count how many calendar days of each weekday
     (Mon-Fri) appear in their raw data (a day "counts" if the ring
     produced >=1 row that day, regardless of how much data).
  2. For each GROUP, compute the "bins per single day" for each weekday
     type under that group's schedule (Friday schedule, Tuesday-special
     schedule for P14/P16, otherwise the M-Th schedule).
  3. Build a weekday x participant matrix of day COUNTS (not dates) --
     e.g. how many Mondays, Tuesdays, etc. each participant has.
  4. Reconstruct each participant's expected total_bins as:
         sum over weekdays of (day_count_for_that_weekday x bins_per_day_for_that_weekday)
     and compare this reconstruction to the group -- if everyone's
     reconstructed total matches, all differences are explained by
     differing day-of-week counts (i.e. missing days). If not, flag it.

Outputs
-------
- weekday_count_matrix_<group>.csv   : weekday x participant day-count matrix, per group
- bin_parity_summary.csv             : per participant -- day counts per weekday,
                                         expected total_bins, and any unexplained gap
- Printed report to console, grouped, explaining any discrepancies
"""

from datetime import datetime, timezone, timedelta
import os
from collections import Counter

import pandas as pd
import pytz

# === Bin width to use for this check (should match whatever you use downstream) ===
BIN_MINUTES = 5

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# === Paths ===
root_path = "/Users/cibrian/Documents/GitHub/Research"
output_folder = os.path.join(root_path, "1_visualization/Heatmaps/OuraRing/BinParityCheck")
os.makedirs(output_folder, exist_ok=True)

# === Groups (participants expected to share total_bins, absent missing days) ===
GROUPS = {
    "GroupA_01_02_03_06_07_08_12": ["01", "02", "03", "06", "07", "08", "12"],
    "GroupB_04_05_09": ["04", "05", "09"],
    "GroupC_14_16": ["14", "16"],
}


# === Shared helper functions (same as coverage script) ===

def convert_iso_to_pacific_date(timestamp):
    pacific_tz = pytz.timezone('America/Los_Angeles')
    dt_utc = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")
    dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    dt_pacific = dt_utc.astimezone(pacific_tz)
    return dt_pacific.date()


def convert_string_to_time(time_string):
    return datetime.strptime(time_string, "%H:%M:%S").time()


def get_day_of_week(date_obj):
    return date_obj.strftime("%A")


def generate_bins(start_time, end_time, bin_minutes):
    base_date = datetime(1900, 1, 1)
    dt = datetime.combine(base_date, start_time)
    end_dt = datetime.combine(base_date, end_time)
    bins = []
    while dt < end_dt:
        bin_end = min(dt + timedelta(minutes=bin_minutes), end_dt)
        bins.append((dt.time(), bin_end.time()))
        dt = bin_end
    return bins


def bins_for_schedule(scheduleData, bin_minutes):
    """Total bin count implied by a schedule dataframe (all non-DELETE rows)."""
    total = 0
    for schedRow in scheduleData.itertuples():
        classLabel = str(getattr(schedRow, 'Class')).strip()
        if classLabel == 'DELETE':
            continue
        timeA = convert_string_to_time(getattr(schedRow, 'TimeStart'))
        timeB = convert_string_to_time(getattr(schedRow, 'TimeEnd'))
        total += len(generate_bins(timeA, timeB, bin_minutes))
    return total


# === Step 1: for every participant, count days present PER WEEKDAY ===

def load_participant_weekday_counts(pNum):
    """Returns Counter: weekday_name -> number of calendar days present with that weekday."""
    rawDataPath = f"{root_path}/P0{pNum}/OuraRing/HeartRate/P0{pNum}OrHrRAW.csv"
    if not os.path.exists(rawDataPath):
        print(f"Raw data not found for P0{pNum}, skipping...")
        return Counter()

    rawData = pd.read_csv(rawDataPath)
    dates = rawData['timestamp'].apply(convert_iso_to_pacific_date)
    unique_dates = sorted(dates.unique())

    weekday_counts = Counter()
    for d in unique_dates:
        dow = get_day_of_week(datetime(d.year, d.month, d.day))
        weekday_counts[dow] += 1
    return weekday_counts


# === Step 2: load schedules per group, and compute bins-per-single-day for each weekday ===

def load_group_schedules(group_participants):
    sample_pNum = group_participants[0]
    if sample_pNum in ["04", "05", "09", "14", "16"]:
        scheduleDataFri = pd.read_csv(f"{root_path}/Schedules/schedData_P(04,05,09,14,16)_FR.csv")
        scheduleDataOth = pd.read_csv(f"{root_path}/Schedules/schedData_P(04,05,09,14,16)_M-TH.csv")
        scheduleDataTu = None
        if sample_pNum in ["14", "16"]:
            scheduleDataTu = pd.read_csv(f"{root_path}/Schedules/schedData_P(14,16)TU.csv")
    else:
        scheduleDataFri = pd.read_csv(f"{root_path}/Schedules/schedData_P(01,02,03,06,07,08,12)_FR.csv")
        scheduleDataOth = pd.read_csv(f"{root_path}/Schedules/schedData_P(01,02,03,06,07,08,12)_M-TH.csv")
        scheduleDataTu = None

    return {"Friday": scheduleDataFri, "Tuesday_special": scheduleDataTu, "Other": scheduleDataOth}


def bins_per_day_for_weekday(weekday, schedules, pNum):
    """Mirrors the coverage script's day-of-week -> schedule selection logic."""
    if weekday == "Friday":
        sched = schedules["Friday"]
    elif weekday == "Tuesday" and pNum in ["14", "16"] and schedules["Tuesday_special"] is not None:
        sched = schedules["Tuesday_special"]
    else:
        sched = schedules["Other"]
    return bins_for_schedule(sched, BIN_MINUTES)


# === Main analysis ===

summary_rows = []

for group_name, group_participants in GROUPS.items():
    print(f"\n{'=' * 70}\n{group_name}\n{'=' * 70}")

    schedules = load_group_schedules(group_participants)

    participant_weekday_counts = {}
    for pNum in group_participants:
        participant_weekday_counts[pNum] = load_participant_weekday_counts(pNum)

    # bins-per-single-day for each weekday, under this group's schedule
    # (computed per participant since P14/P16 have a Tuesday-specific
    #  schedule that differs from the rest of their group)
    bins_per_weekday_by_participant = {
        pNum: {wd: bins_per_day_for_weekday(wd, schedules, pNum) for wd in WEEKDAY_ORDER}
        for pNum in group_participants
    }

    # === Weekday count matrix (rows = weekday, cols = participants) ===
    matrix_rows = []
    for wd in WEEKDAY_ORDER:
        row = {"day_of_week": wd}
        for pNum in group_participants:
            row[f"P0{pNum}_day_count"] = participant_weekday_counts[pNum].get(wd, 0)
        matrix_rows.append(row)
    matrix_df = pd.DataFrame(matrix_rows)

    matrix_csv_path = os.path.join(output_folder, f"weekday_count_matrix_{group_name}.csv")
    matrix_df.to_csv(matrix_csv_path, index=False)
    print(f"Weekday count matrix saved to: {matrix_csv_path}")
    print(matrix_df.to_string(index=False))

    # Flag weekdays where day counts differ across the group
    mismatched_weekdays = []
    for wd in WEEKDAY_ORDER:
        counts = [participant_weekday_counts[pNum].get(wd, 0) for pNum in group_participants]
        if len(set(counts)) > 1:
            mismatched_weekdays.append((wd, dict(zip([f"P0{p}" for p in group_participants], counts))))

    if mismatched_weekdays:
        print(f"\n-> Day-of-week COUNT mismatches within {group_name}:")
        for wd, counts in mismatched_weekdays:
            print(f"   {wd}: {counts}")
    else:
        print(f"\n-> All participants in {group_name} have IDENTICAL day-of-week counts. "
              f"No missing days detected for this group.")

    # === Reconstruct expected total_bins per participant from weekday counts ===
    group_expected_totals = {}
    for pNum in group_participants:
        wd_counts = participant_weekday_counts[pNum]
        bins_per_wd = bins_per_weekday_by_participant[pNum]
        expected_total = sum(wd_counts.get(wd, 0) * bins_per_wd[wd] for wd in WEEKDAY_ORDER)
        group_expected_totals[pNum] = expected_total

        summary_rows.append({
            "Group": group_name,
            "Participant": f"P0{pNum}",
            **{f"n_{wd}": wd_counts.get(wd, 0) for wd in WEEKDAY_ORDER},
            "expected_total_bins_from_weekday_counts": expected_total,
        })

    # === Check whether weekday-count differences fully explain any total_bins gap ===
    if len(set(group_expected_totals.values())) == 1:
        common_total = next(iter(group_expected_totals.values()))
        print(f"-> Reconstructed total_bins is IDENTICAL for every participant in {group_name} "
              f"({common_total}). If ACTUAL total_bins differs from this, the cause is NOT "
              f"missing days -- check schedule file assignment, duplicate rows, or bin-generation logic.")
    else:
        print(f"-> Reconstructed total_bins DIFFERS across {group_name} using weekday-count "
              f"logic: {[(f'P0{p}', t) for p, t in group_expected_totals.items()]}")
        print("   This should be fully explained by the day-of-week count mismatches printed above.")
        print("   If a participant's ACTUAL total_bins (from the coverage script) doesn't match "
              "their reconstructed value here, check for:")
        print("   - Duplicate rows / duplicate dates in the raw file (inflating a weekday's count)")
        print("   - A participant using a different schedule file than expected")
        print("   - Timezone/date-boundary mismatches (samples near midnight PT double- or under-counting a day)")

summary_df = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(output_folder, "bin_parity_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nFull summary saved to: {summary_csv_path}")
print(summary_df.to_string(index=False))
