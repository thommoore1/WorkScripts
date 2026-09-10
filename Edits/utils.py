"""
utils.py
--------
Shared helper functions used by both e2_coverage_missingness.py and
e3_activity_hr_effect.py. You shouldn't need to edit this file -- edit
config.py instead.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import (
    TIMEZONE, VALID_CLASSES, ACTIVITY_ARRAY_START_HOUR, ACTIVITY_GROUP_MAP,
    SCHEDULE_FOLDER,
)

# ---------------------------------------------------------------------------
# Participant discovery
# ---------------------------------------------------------------------------

def list_participants(root_path):
    """Return sorted list of participant folder names (e.g. 'P001') under root_path."""
    if not os.path.exists(root_path):
        print(f"  [WARN] ROOT_PATH does not exist: {root_path}")
        return []
    return sorted([
        f for f in os.listdir(root_path)
        if f.startswith("P") and os.path.isdir(os.path.join(root_path, f))
    ])


def participant_number(participant_folder):
    """'P001' -> 1"""
    m = re.search(r"P0*(\d+)", participant_folder)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Heart-rate data loading
# ---------------------------------------------------------------------------

def load_participant_hr(root_path, participant):
    """
    Load and concatenate all daily (non-RAW) HeartRate CSVs for one participant.

    Adds columns:
      file_date       - date parsed from the filename (YYYY-MM-DD.csv)
      datetime_utc    - 'time' column (epoch seconds, UTC) converted to a UTC Timestamp
      datetime_local  - converted to TIMEZONE (DST-aware)
      class_clean     - stripped 'class' column value

    Returns an empty DataFrame with the expected columns if nothing is found.
    """
    hr_folder = os.path.join(root_path, participant, "OuraRing", "HeartRate")
    empty_cols = ["participant", "file_date", "time", "class", "bpm",
                  "datetime_utc", "datetime_local", "class_clean"]
    if not os.path.exists(hr_folder):
        return pd.DataFrame(columns=empty_cols)

    csv_files = [f for f in os.listdir(hr_folder) if f.endswith(".csv") and "RAW" not in f]
    frames = []
    for file in csv_files:
        date_str = file[-14:-4]  # assumes filename ends ..._YYYY-MM-DD.csv
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            print(f"  [WARN] Could not parse date from filename, skipping: {file}")
            continue

        path = os.path.join(hr_folder, file)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  [WARN] Could not read {path}: {e}")
            continue

        if not {"time", "class", "bpm"}.issubset(df.columns):
            print(f"  [WARN] Missing expected columns in {path}, skipping")
            continue

        df["participant"] = participant
        df["file_date"] = file_date
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=empty_cols)

    combined = pd.concat(frames, ignore_index=True)
    combined["datetime_utc"] = pd.to_datetime(combined["time"], unit="s", utc=True)
    combined["datetime_local"] = combined["datetime_utc"].dt.tz_convert(TIMEZONE)
    combined["class_clean"] = combined["class"].astype(str).str.strip()
    return combined


def load_all_hr(root_path, participants=None):
    """Load HR data for all (or specified) participants into one DataFrame."""
    participants = participants or list_participants(root_path)
    frames = []
    for p in participants:
        print(f"  Loading HR data: {p}")
        df = load_participant_hr(root_path, p)
        if not df.empty:
            frames.append(df)
        else:
            print(f"    [INFO] No HR data found for {p}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Schedule parsing
# ---------------------------------------------------------------------------

# Matches e.g. "schedData_P(01,02,03,06,07,08,12)_M-TH.csv",
# "schedData_P(04,05,09,14,16)_FR.csv", "schedData_P(14,16)TU.csv"
_SCHED_FILENAME_RE = re.compile(r"schedData_P\(([\d,]+)\)_?(M-TH|FR|TU)\.csv$", re.IGNORECASE)


def build_schedule_index(schedule_folder=SCHEDULE_FOLDER):
    """
    Scan the Schedules folder and build:
        {participant_number(int): {"M-TH": path, "FR": path, "TU": path (optional)}}
    """
    index = {}
    if not os.path.exists(schedule_folder):
        print(f"  [WARN] Schedule folder not found: {schedule_folder}")
        return index

    for fname in os.listdir(schedule_folder):
        m = _SCHED_FILENAME_RE.search(fname)
        if not m:
            continue
        nums = [int(x) for x in m.group(1).split(",")]
        day_type = m.group(2).upper()
        path = os.path.join(schedule_folder, fname)
        for n in nums:
            index.setdefault(n, {})[day_type] = path
    return index


def get_schedule_path(pnum, weekday_name, schedule_index):
    """
    weekday_name: 'Mon','Tue','Wed','Thu','Fri'

    ASSUMPTION (see config.py note): Tuesday uses the special
    schedData_P(14,16)TU.csv file INSTEAD of the normal M-TH schedule, and
    ONLY for participants included in that file. All other participants and
    weekdays follow the plain M-TH or FR schedule. Please double-check this.
    """
    sched = schedule_index.get(pnum, {})
    if weekday_name == "Fri":
        return sched.get("FR")
    if weekday_name == "Tue" and "TU" in sched:
        return sched.get("TU")
    return sched.get("M-TH")


def load_schedule_blocks(path, valid_classes=VALID_CLASSES):
    """
    Parse a schedule CSV (TimeStart,TimeEnd,Class) and return a DataFrame
    with only rows whose Class is one of valid_classes (drops 'DELETE' etc).
    TimeStart/TimeEnd are returned as datetime.time objects.
    """
    if path is None or not os.path.exists(path):
        return pd.DataFrame(columns=["TimeStart", "TimeEnd", "Class"])
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] Could not read schedule {path}: {e}")
        return pd.DataFrame(columns=["TimeStart", "TimeEnd", "Class"])

    df.columns = [c.strip() for c in df.columns]
    if not {"TimeStart", "TimeEnd", "Class"}.issubset(df.columns):
        print(f"  [WARN] Schedule file missing expected columns: {path}")
        return pd.DataFrame(columns=["TimeStart", "TimeEnd", "Class"])

    df["Class"] = df["Class"].astype(str).str.strip()
    df = df[df["Class"].isin(valid_classes)].copy()
    for col in ["TimeStart", "TimeEnd"]:
        df[col] = pd.to_datetime(df[col].astype(str).str.strip(), format="%H:%M:%S",
                                  errors="coerce").dt.time
    df = df.dropna(subset=["TimeStart", "TimeEnd"])
    return df.reset_index(drop=True)


def weekday_name_for(date_obj):
    return date_obj.strftime("%a")  # 'Mon','Tue','Wed','Thu','Fri','Sat','Sun'


# ---------------------------------------------------------------------------
# 5-minute bin coverage helpers (E2)
# ---------------------------------------------------------------------------

def expected_bin_starts(start_dt, end_dt, bin_minutes=5):
    """List of bin-start Timestamps covering [start_dt, end_dt)."""
    bins = []
    t = start_dt
    while t < end_dt:
        bins.append(t)
        t = t + pd.Timedelta(minutes=bin_minutes)
    return bins


def bins_with_data(local_datetimes, bin_starts, bin_minutes=5):
    """
    Count how many of bin_starts contain at least one timestamp from
    local_datetimes. A missing reading = no row at all (per confirmed
    behavior), so "has data" simply means >=1 row falls in that 5-min window.
    """
    if len(bin_starts) == 0:
        return 0
    times = pd.Series(pd.to_datetime(local_datetimes)) if len(local_datetimes) else pd.Series([], dtype="datetime64[ns, UTC]")
    count = 0
    for b in bin_starts:
        b_end = b + pd.Timedelta(minutes=bin_minutes)
        if len(times) and ((times >= b) & (times < b_end)).any():
            count += 1
    return count


# ---------------------------------------------------------------------------
# Daily activity (class_5_min / met_1_min) decoding (E3)
# ---------------------------------------------------------------------------

def _split_activity_string(raw):
    """Handle either a delimited string or a bare concatenated-digit string."""
    s = str(raw).strip()
    if ";" in s:
        return [x for x in s.split(";") if x != ""]
    if "," in s:
        return [x for x in s.split(",") if x != ""]
    return list(s)  # bare digits, e.g. class_5_min "0003332..."


def decode_activity_array(raw_value, day_date, values_are_float=False, bin_minutes=5,
                           tz=TIMEZONE, start_hour=ACTIVITY_ARRAY_START_HOUR):
    """
    Decode an Oura class_5_min / met_1_min style array into a DataFrame with
    columns: datetime_local, value.

    ASSUMPTION: array index 0 corresponds to start_hour:00 local time on
    day_date (Oura's activity-day convention), not midnight. See config.py.
    """
    if pd.isna(raw_value):
        return pd.DataFrame(columns=["datetime_local", "value"])
    tokens = _split_activity_string(raw_value)
    start_dt = pd.Timestamp(day_date, tz=tz).normalize() + pd.Timedelta(hours=start_hour)
    rows = []
    for i, tok in enumerate(tokens):
        try:
            val = float(tok) if values_are_float else int(tok)
        except ValueError:
            continue
        rows.append((start_dt + pd.Timedelta(minutes=bin_minutes * i), val))
    return pd.DataFrame(rows, columns=["datetime_local", "value"])


def load_daily_activity(root_path, participant):
    """
    Load {participant}OrDaLabeled.csv, return raw DataFrame (one row per day).

    IMPORTANT: class_5_min is a bare string of digits with no separators
    (e.g. "0000...33323...0000"). If read with pandas' default type
    inference, a long digit-only column gets silently parsed as a number --
    which strips leading zeros and, because these strings can be 200+
    digits long (far beyond int64/float64 precision), corrupts the data
    entirely. We force it (and met_1_min, for safety) to be read as text.
    """
    path = os.path.join(root_path, participant, "OuraRing", "DailyActivity",
                         f"{participant}OrDaLabeled.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"class_5_min": str, "met_1_min": str})
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return pd.DataFrame()


def activity_group(code):
    """Map an Oura class_5_min integer code to sedentary/light/active."""
    try:
        return ACTIVITY_GROUP_MAP.get(int(code), np.nan)
    except (ValueError, TypeError):
        return np.nan


# ---------------------------------------------------------------------------
# Image rendering: styled tables + heatmaps
# ---------------------------------------------------------------------------

def render_table_image(df, title, output_path, figsize=None):
    """Render a DataFrame as a clean, readable table PNG (not a plain heatmap)."""
    df = df.astype(str)
    n_rows, n_cols = df.shape
    if figsize is None:
        figsize = (max(7, n_cols * 2.0), max(2, 0.55 * n_rows + 1.6))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.7)
    table.auto_set_column_width(col=list(range(n_cols)))

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            table[i, j].set_facecolor("#F2F2F2" if i % 2 == 0 else "white")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved table image: {output_path}")


def render_heatmap_image(pivot_df, title, output_path, cmap="viridis_r", fmt_pct=True):
    """Render a participant x category heatmap. NaN cells are labeled 'Null'."""
    if pivot_df.empty:
        print(f"  [WARN] Nothing to plot for heatmap '{title}', skipping")
        return
    data = pivot_df.copy()
    fmt_func = lambda x: "Null" if pd.isna(x) else (f"{x:.0f}%" if fmt_pct else f"{x:.1f}")
    try:
        annot = data.map(fmt_func)  # pandas >= 2.1
    except AttributeError:
        annot = data.applymap(fmt_func)  # pandas < 2.1

    plt.figure(figsize=(max(6, len(data.columns) * 0.9), max(3, len(data.index) * 0.7)))
    ax = sns.heatmap(
        data, cmap=cmap, linewidths=0.5, linecolor="gray",
        cbar=True, annot=annot, fmt="",
        vmin=0, vmax=100 if fmt_pct else None,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Participant")
    plt.ylabel("")
    plt.tight_layout()
    plt.gcf().set_facecolor("white")
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved heatmap image: {output_path}")