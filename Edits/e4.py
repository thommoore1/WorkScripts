"""
e2_coverage_by_weekday.py
-------------------------
Data coverage by DAY OF THE WEEK (Mon-Fri).

Produces TWO image files in OUTPUT_FOLDER/edit4:
  1. table_weekday_coverage.png   - coverage table by weekday
  2. heatmap_weekday_coverage.png - per-participant coverage % by weekday

Definitions:
  - Expected window: the school day (SCHOOL_DAY_START - SCHOOL_DAY_END from
    config.py, i.e. 8:30-15:00) on every participant-day.
  - "Coverage %" = (5-min bins containing >=1 HR reading) / (expected 5-min
    bins in that window), aggregated across all participants and dates that
    fall on that weekday.
  - "Observations (n)" = total HR rows (5-sec bpm readings) inside the window.
  - "Participants (n)" = distinct participants with at least one reading in
    the window on that weekday.
  - "Participant-days (n)" = distinct (participant, date) files on that weekday.

Run: python3 e2_coverage_by_weekday.py
"""

import os
import numpy as np
import pandas as pd

from config import (
    ROOT_PATH, OUTPUT_FOLDER, TIMEZONE,
    SCHOOL_DAY_START, SCHOOL_DAY_END,
)
import utils as u

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri"]
EDIT_FOLDER = os.path.join(OUTPUT_FOLDER, "edit4")


def compute_weekday_table(hr_all):
    """Coverage table + per-participant pivot for day of the week."""
    school_start = pd.to_datetime(SCHOOL_DAY_START).time()
    school_end = pd.to_datetime(SCHOOL_DAY_END).time()

    rows = []
    per_participant_pct = {d: {} for d in WEEKDAY_ORDER}

    for (participant, file_date), day_df in hr_all.groupby(["participant", "file_date"]):
        weekday = u.weekday_name_for(file_date)
        if weekday not in WEEKDAY_ORDER:  # skips Sat/Sun
            continue

        start_dt = pd.Timestamp.combine(file_date, school_start).tz_localize(TIMEZONE)
        end_dt = pd.Timestamp.combine(file_date, school_end).tz_localize(TIMEZONE)

        window = day_df[(day_df["datetime_local"] >= start_dt) & (day_df["datetime_local"] < end_dt)]
        bin_starts = u.expected_bin_starts(start_dt, end_dt, 5)
        true_bins = u.bins_with_data(window["datetime_local"], bin_starts, 5)
        pct = 100 * true_bins / len(bin_starts) if len(bin_starts) else np.nan

        per_participant_pct[weekday].setdefault(participant, []).append(pct)
        rows.append({
            "Weekday": weekday, "participant": participant, "file_date": file_date,
            "expected": len(bin_starts), "true": true_bins, "obs": len(window),
            "has_data": len(window) > 0,
        })

    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Participants (n) counts only participants with >=1 reading in the window
    participants_with_data = (
        raw[raw["has_data"]].groupby("Weekday")["participant"].nunique()
    )

    summary = raw.groupby("Weekday").agg(
        Participant_days=("file_date", "size"),
        Observations=("obs", "sum"),
        Expected_bins=("expected", "sum"),
        True_bins=("true", "sum"),
    ).reset_index()
    summary["Participants"] = summary["Weekday"].map(participants_with_data).fillna(0).astype(int)
    summary["Coverage %"] = (100 * summary["True_bins"] / summary["Expected_bins"]).round(1)

    summary = summary[[
        "Weekday", "Participants", "Participant_days", "Observations",
        "Expected_bins", "True_bins", "Coverage %",
    ]].rename(columns={
        "Participants": "Participants (n)",
        "Participant_days": "Participant-days (n)",
        "Observations": "Observations (n)",
        "Expected_bins": "Expected 5-min bins",
        "True_bins": "Bins with data",
    })
    summary["_order"] = summary["Weekday"].apply(
        lambda d: WEEKDAY_ORDER.index(d) if d in WEEKDAY_ORDER else 99)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    heat_rows = [
        {"weekday": d, "participant": participant, "pct": np.nanmean(pcts)}
        for d, pmap in per_participant_pct.items()
        for participant, pcts in pmap.items()
    ]
    heat_df = pd.DataFrame(heat_rows)
    pivot = (heat_df.pivot_table(index="weekday", columns="participant", values="pct")
             if not heat_df.empty else pd.DataFrame())
    if not pivot.empty:
        pivot = pivot.reindex([d for d in WEEKDAY_ORDER if d in pivot.index])

    return summary, pivot


def main():
    os.makedirs(EDIT_FOLDER, exist_ok=True)
    print(f"ROOT_PATH     = {ROOT_PATH}")
    print(f"OUTPUT_FOLDER = {EDIT_FOLDER}\n")

    print("Loading HR data for all participants...")
    hr_all = u.load_all_hr(ROOT_PATH)
    if hr_all.empty:
        print("No HR data found anywhere under ROOT_PATH -- check ROOT_PATH in config.py")
        return

    print("\nComputing day-of-week coverage table...")
    weekday_table, weekday_pivot = compute_weekday_table(hr_all)
    if weekday_table.empty:
        print("No weekday coverage could be computed -- no Mon-Fri data found.")
        return

    u.render_table_image(
        weekday_table,
        f"Data Coverage by Day of the Week ({SCHOOL_DAY_START[:5]}-{SCHOOL_DAY_END[:5]})",
        os.path.join(EDIT_FOLDER, "table_weekday_coverage.png"),
    )
    u.render_heatmap_image(
        weekday_pivot,
        "Coverage % by Day of the Week and Participant",
        os.path.join(EDIT_FOLDER, "heatmap_weekday_coverage.png"),
    )

    print("\nDone. Check for weekdays with low coverage or few contributing participants")
    print("before interpreting HR differences between days.")


if __name__ == "__main__":
    main()