"""
e2_coverage_missingness.py
---------------------------
E2: Report data coverage and the number of participants and observations
contributing to every activity and time comparison, so differential
missingness can be assessed before interpreting HR differences.

Produces FOUR image files in OUTPUT_FOLDER (set in config.py):
  1. table_time_chunk_coverage.png   - coverage table by fixed 30-min clock chunk
  2. heatmap_time_chunk_coverage.png - per-participant coverage % by chunk
  3. table_activity_coverage.png     - coverage table by classroom activity
  4. heatmap_activity_coverage.png   - per-participant coverage % by activity

Definitions:
  - "Time chunk" comparison: the school day (8:30-15:00) split into fixed
    30-minute clock windows, combining ALL weekdays (Fri included).
  - "Activity" comparison: each participant-day's classroom schedule
    (from the matching Schedules CSV) defines the expected start/end time
    of each class period -- this is the coverage DENOMINATOR, independent
    of the HR file's own 'class' label (which is still used elsewhere, e.g.
    in the original heatmap script, but not for defining expected windows
    here -- see chat discussion).
  - "Coverage %" = (5-min bins containing >=1 HR reading) / (expected 5-min
    bins in that window), aggregated across all participants and days.
  - "Observations (n)" = total HR rows (5-sec granularity bpm readings)
    falling inside that window across all participants/days.
  - "Participants (n)" = distinct participants with at least one reading
    in that window at all.

Run: python3 e2_coverage_missingness.py
"""

import os
import numpy as np
import pandas as pd

from config import (
    ROOT_PATH, OUTPUT_FOLDER, TIMEZONE, VALID_CLASSES,
    SCHOOL_DAY_START, SCHOOL_DAY_END, CHUNK_MINUTES, SCHEDULE_FOLDER,
)
import utils as u


def build_time_chunks():
    """Return list of (start_time, end_time) tuples covering the school day."""
    ref_date = pd.Timestamp.today().date()
    start = pd.Timestamp.combine(ref_date, pd.to_datetime(SCHOOL_DAY_START).time())
    end = pd.Timestamp.combine(ref_date, pd.to_datetime(SCHOOL_DAY_END).time())
    chunks = []
    t = start
    while t < end:
        t2 = min(t + pd.Timedelta(minutes=CHUNK_MINUTES), end)
        chunks.append((t.time(), t2.time()))
        t = t2
    return chunks


def chunk_label(t1, t2):
    return f"{t1.strftime('%H:%M')}-{t2.strftime('%H:%M')}"


def compute_time_chunk_table(hr_all, chunks):
    """Coverage table + per-participant pivot for the fixed 30-min clock chunks."""
    rows = []
    per_participant_pct = {}

    for t1, t2 in chunks:
        label = chunk_label(t1, t2)
        total_expected = total_true = total_obs = 0
        participants_seen = set()
        per_participant_pct[label] = {}

        for (participant, file_date), day_df in hr_all.groupby(["participant", "file_date"]):
            start_dt = pd.Timestamp.combine(file_date, t1).tz_localize(TIMEZONE)
            end_dt = pd.Timestamp.combine(file_date, t2).tz_localize(TIMEZONE)
            window = day_df[(day_df["datetime_local"] >= start_dt) & (day_df["datetime_local"] < end_dt)]

            bin_starts = u.expected_bin_starts(start_dt, end_dt, 5)
            true_bins = u.bins_with_data(window["datetime_local"], bin_starts, 5)

            total_expected += len(bin_starts)
            total_true += true_bins
            total_obs += len(window)
            if len(window) > 0:
                participants_seen.add(participant)

            pct = 100 * true_bins / len(bin_starts) if len(bin_starts) else np.nan
            per_participant_pct[label].setdefault(participant, []).append(pct)

        coverage_pct = 100 * total_true / total_expected if total_expected else np.nan
        rows.append({
            "Time Chunk": label,
            "Participants (n)": len(participants_seen),
            "Observations (n)": total_obs,
            "Expected 5-min bins": total_expected,
            "Bins with data": total_true,
            "Coverage %": f"{coverage_pct:.1f}" if pd.notna(coverage_pct) else "NA",
        })

    table = pd.DataFrame(rows)

    heat_rows = [
        {"chunk": label, "participant": participant, "pct": np.nanmean(pcts)}
        for label, pmap in per_participant_pct.items()
        for participant, pcts in pmap.items()
    ]
    heat_df = pd.DataFrame(heat_rows)
    pivot = heat_df.pivot_table(index="chunk", columns="participant", values="pct") if not heat_df.empty else pd.DataFrame()
    if not pivot.empty:
        pivot = pivot.reindex([chunk_label(t1, t2) for t1, t2 in chunks])

    return table, pivot


def compute_activity_table(hr_all, schedule_index):
    """Coverage table + per-participant pivot for classroom activities."""
    rows = []
    per_participant_pct = {c: {} for c in VALID_CLASSES}

    for (participant, file_date), day_df in hr_all.groupby(["participant", "file_date"]):
        pnum = u.participant_number(participant)
        weekday = u.weekday_name_for(file_date)
        if weekday in ("Sat", "Sun"):
            continue
        sched_path = u.get_schedule_path(pnum, weekday, schedule_index)
        blocks = u.load_schedule_blocks(sched_path)
        if blocks.empty:
            continue

        for _, block in blocks.iterrows():
            cls = block["Class"]
            start_dt = pd.Timestamp.combine(file_date, block["TimeStart"]).tz_localize(TIMEZONE)
            end_dt = pd.Timestamp.combine(file_date, block["TimeEnd"]).tz_localize(TIMEZONE)
            if end_dt <= start_dt:
                continue

            window = day_df[(day_df["datetime_local"] >= start_dt) & (day_df["datetime_local"] < end_dt)]
            bin_starts = u.expected_bin_starts(start_dt, end_dt, 5)
            true_bins = u.bins_with_data(window["datetime_local"], bin_starts, 5)
            pct = 100 * true_bins / len(bin_starts) if len(bin_starts) else np.nan

            per_participant_pct.setdefault(cls, {}).setdefault(participant, []).append(pct)
            rows.append({"Class": cls, "participant": participant,
                         "expected": len(bin_starts), "true": true_bins, "obs": len(window)})

    raw = pd.DataFrame(rows)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = raw.groupby("Class").agg(
        Participants=("participant", "nunique"),
        Observations=("obs", "sum"),
        Expected_bins=("expected", "sum"),
        True_bins=("true", "sum"),
    ).reset_index()
    summary["Coverage %"] = (100 * summary["True_bins"] / summary["Expected_bins"]).round(1)
    summary = summary.rename(columns={
        "Class": "Activity", "Participants": "Participants (n)",
        "Observations": "Observations (n)", "Expected_bins": "Expected 5-min bins",
        "True_bins": "Bins with data",
    })
    summary["_order"] = summary["Activity"].apply(
        lambda c: VALID_CLASSES.index(c) if c in VALID_CLASSES else 99)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    heat_rows = [
        {"activity": cls, "participant": participant, "pct": np.nanmean(pcts)}
        for cls, pmap in per_participant_pct.items()
        for participant, pcts in pmap.items()
    ]
    heat_df = pd.DataFrame(heat_rows)
    pivot = heat_df.pivot_table(index="activity", columns="participant", values="pct") if not heat_df.empty else pd.DataFrame()
    if not pivot.empty:
        pivot = pivot.reindex([c for c in VALID_CLASSES if c in pivot.index])

    return summary, pivot


def main():
    print(f"ROOT_PATH   = {ROOT_PATH}")
    print(f"OUTPUT_FOLDER = {OUTPUT_FOLDER}\n")

    print("Loading HR data for all participants...")
    hr_all = u.load_all_hr(ROOT_PATH)
    if hr_all.empty:
        print("No HR data found anywhere under ROOT_PATH -- check ROOT_PATH in config.py")
        return

    print("\nBuilding schedule index...")
    schedule_index = u.build_schedule_index(SCHEDULE_FOLDER)
    if not schedule_index:
        print("No schedules found -- activity coverage table will be empty. Check SCHEDULE_FOLDER.")

    print("\nComputing 30-minute time-of-day coverage table (all weekdays combined)...")
    chunks = build_time_chunks()
    chunk_table, chunk_pivot = compute_time_chunk_table(hr_all, chunks)
    u.render_table_image(
        chunk_table,
        "Data Coverage by 30-Minute Time-of-Day Chunk (8:30-15:00, all weekdays)",
        os.path.join(OUTPUT_FOLDER, "table_time_chunk_coverage.png"),
    )
    u.render_heatmap_image(
        chunk_pivot,
        "Coverage % by Time Chunk and Participant",
        os.path.join(OUTPUT_FOLDER, "heatmap_time_chunk_coverage.png"),
    )

    print("\nComputing per-activity (classroom) coverage table...")
    activity_table, activity_pivot = compute_activity_table(hr_all, schedule_index)
    if not activity_table.empty:
        u.render_table_image(
            activity_table,
            "Data Coverage by Classroom Activity",
            os.path.join(OUTPUT_FOLDER, "table_activity_coverage.png"),
        )
        u.render_heatmap_image(
            activity_pivot,
            "Coverage % by Activity and Participant",
            os.path.join(OUTPUT_FOLDER, "heatmap_activity_coverage.png"),
        )
    else:
        print("No activity/schedule coverage could be computed -- check SCHEDULE_FOLDER in config.py")

    print("\nDone. Use these tables/heatmaps to check for classes or time chunks with low")
    print("coverage or few contributing participants before interpreting HR differences")
    print("between them -- low, uneven coverage is a sign that comparison may be biased")
    print("by differential missingness rather than a true underlying difference.")


if __name__ == "__main__":
    main()