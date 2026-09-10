"""
e3_activity_hr_effect.py
--------------------------
E3: Address movement as a possible explanation for higher HR.

Aligns Oura's per-day class_5_min (5-min activity-intensity code) and
met_1_min (1-min MET value) arrays with classroom-time heart-rate data,
then tests whether HR is higher during more physically active periods and
computes effect sizes.

Produces in OUTPUT_FOLDER (set in config.py):
  1. table_hr_by_activity_level.png   - descriptive stats: n, mean/SD HR by activity level
  2. table_effect_sizes.png           - Cohen's d, ANOVA eta^2, HR-MET correlation,
                                         and (if statsmodels installed) a mixed-effects
                                         model estimate with participant random intercept
  3. heatmap_hr_by_activity_level.png - per-participant mean HR by activity level
  4. diagnostic_alignment_check_<P>_<date>.png
        - ONE sanity-check plot (first participant-day processed) overlaying raw HR
          against the decoded Oura activity code, so you can visually confirm the
          4:00 AM array-start assumption before trusting the results above.

Scope: classroom time only (per confirmed scope), using the existing 'class'
column in the HeartRate CSVs to identify classroom periods.

Run: python3 e3_activity_hr_effect.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from config import ROOT_PATH, OUTPUT_FOLDER, VALID_CLASSES, ACTIVITY_ARRAY_START_HOUR
import utils as u

warnings.filterwarnings("ignore")

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def floor_to_activity_bin(dt, start_hour=ACTIVITY_ARRAY_START_HOUR, bin_minutes=5):
    """Floor a local datetime to the nearest 5-min bin on Oura's 4:00-AM-anchored grid."""
    day_start = dt.normalize() + pd.Timedelta(hours=start_hour)
    if dt < day_start:
        day_start = day_start - pd.Timedelta(days=1)
    delta = dt - day_start
    n = int(delta.total_seconds() // (bin_minutes * 60))
    return day_start + pd.Timedelta(minutes=bin_minutes * n)


def save_alignment_check(participant, day_date, hr_day, class_decoded):
    """Diagnostic plot: raw HR vs. decoded Oura activity code, for visual verification."""
    import matplotlib.pyplot as plt
    if hr_day.empty or class_decoded.empty:
        return
    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(hr_day["datetime_local"], hr_day["bpm"], color="crimson", label="Heart rate (bpm)")
    ax1.set_ylabel("BPM", color="crimson")
    ax2 = ax1.twinx()
    ax2.step(class_decoded["datetime_local"], class_decoded["value"], color="steelblue",
              where="post", label="Oura class_5_min code")
    ax2.set_ylabel("Oura activity code (0-4)", color="steelblue")
    ax1.set_xlim(hr_day["datetime_local"].min(), hr_day["datetime_local"].max())
    plt.title(f"Alignment check: {participant} on {day_date}\n"
              f"(verifies class_5_min array is anchored at {ACTIVITY_ARRAY_START_HOUR}:00 AM local)")
    fig.autofmt_xdate()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER, f"diagnostic_alignment_check_{participant}_{day_date}.png")
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close()
    print(f"  Saved alignment sanity-check plot: {out_path}")
    print("  >>> Please open this and visually confirm HR patterns line up sensibly")
    print("      with the activity codes before trusting the results below. <<<")


def build_bin_level_dataset(root_path):
    """
    For every participant/day, bin classroom-time HR data into 5-minute bins
    aligned to Oura's activity-day grid, and attach the class_5_min
    activity-intensity group + met_1_min mean for that same bin.
    Returns one row per participant-day-bin with columns:
        participant, file_date, bin, bpm_mean, n_readings, classroom,
        activity_code (raw Oura 0-4 code), activity_group, met_mean
    """
    participants = u.list_participants(root_path)
    all_rows = []
    diagnostic_saved = False

    for participant in participants:
        print(f"  Processing {participant}...")
        hr = u.load_participant_hr(root_path, participant)
        if hr.empty:
            print(f"    [INFO] No HR data for {participant}, skipping")
            continue

        daily = u.load_daily_activity(root_path, participant)
        if daily.empty or "class_5_min" not in daily.columns:
            print(f"    [INFO] No usable DailyActivity/class_5_min data for {participant}, skipping")
            continue

        hr = hr[hr["class_clean"].isin(VALID_CLASSES)].copy()
        if hr.empty:
            print(f"    [INFO] No classroom-labeled HR rows for {participant}, skipping")
            continue
        hr["bin"] = hr["datetime_local"].apply(floor_to_activity_bin)

        binned = hr.groupby(["file_date", "bin"]).agg(
            bpm_mean=("bpm", "mean"),
            n_readings=("bpm", "size"),
            classroom=("class_clean", lambda s: s.mode().iat[0] if not s.mode().empty else np.nan),
        ).reset_index()

        for _, day_row in daily.iterrows():
            try:
                day_date = pd.to_datetime(day_row["day"]).date()
            except Exception:
                continue

            day_binned = binned[binned["file_date"] == day_date]
            if day_binned.empty:
                continue

            class_decoded = u.decode_activity_array(
                day_row["class_5_min"], day_date, values_are_float=False, bin_minutes=5)
            if class_decoded.empty:
                continue
            class_decoded["activity_group"] = class_decoded["value"].apply(u.activity_group)

            met_decoded = None
            if "met_1_min" in daily.columns and pd.notna(day_row.get("met_1_min")):
                met_raw = u.decode_activity_array(
                    day_row["met_1_min"], day_date, values_are_float=True, bin_minutes=1)
                if not met_raw.empty:
                    met_raw["bin"] = met_raw["datetime_local"].apply(floor_to_activity_bin)
                    met_decoded = (met_raw.groupby("bin")["value"].mean()
                                   .reset_index().rename(columns={"value": "met_mean"}))

            merged = day_binned.merge(
                class_decoded[["datetime_local", "value", "activity_group"]].rename(
                    columns={"value": "activity_code"}),
                left_on="bin", right_on="datetime_local", how="left",
            ).drop(columns="datetime_local")

            merged = merged.merge(met_decoded, on="bin", how="left") if met_decoded is not None \
                else merged.assign(met_mean=np.nan)

            merged["participant"] = participant
            all_rows.append(merged)

            if not diagnostic_saved:
                save_alignment_check(
                    participant, day_date,
                    hr[hr["file_date"] == day_date],
                    class_decoded,
                )
                diagnostic_saved = True

    if not all_rows:
        return pd.DataFrame()
    result = pd.concat(all_rows, ignore_index=True)
    return result.dropna(subset=["activity_group", "bpm_mean"])


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_sd = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan


def eta_squared_oneway(groups):
    all_vals = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    grand_mean = all_vals.mean()
    ss_total = ((all_vals - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total if ss_total > 0 else np.nan


def main():
    print(f"ROOT_PATH   = {ROOT_PATH}")
    print(f"OUTPUT_FOLDER = {OUTPUT_FOLDER}\n")

    print("Building bin-level classroom HR + activity-intensity dataset...")
    data = build_bin_level_dataset(ROOT_PATH)
    if data.empty:
        print("No overlapping HR + DailyActivity classroom data found.")
        print("Check ROOT_PATH, and that both HeartRate and DailyActivity files exist and parse.")
        return

    print(f"\nTotal classroom 5-min bins with valid activity data: {len(data)}")
    print(f"Participants contributing: {data['participant'].nunique()}")

    # --- Descriptive stats table ---
    desc = data.groupby("activity_group").agg(
        Participants=("participant", "nunique"),
        Bins=("bpm_mean", "size"),
        Mean_BPM=("bpm_mean", "mean"),
        SD_BPM=("bpm_mean", "std"),
    ).reset_index()
    order = {"sedentary": 0, "light": 1, "active": 2}
    desc["_order"] = desc["activity_group"].map(order)
    desc = desc.sort_values("_order").drop(columns="_order")
    desc["Mean_BPM"] = desc["Mean_BPM"].round(1)
    desc["SD_BPM"] = desc["SD_BPM"].round(1)
    desc = desc.rename(columns={
        "activity_group": "Activity Level", "Participants": "Participants (n)",
        "Bins": "Bins (n)", "Mean_BPM": "Mean HR (bpm)", "SD_BPM": "SD HR (bpm)",
    })
    u.render_table_image(
        desc, "Heart Rate by Concurrent Physical-Activity Level (classroom time only)",
        os.path.join(OUTPUT_FOLDER, "table_hr_by_activity_level.png"),
    )

    # --- Effect sizes ---
    groups = {
        g: data.loc[data["activity_group"] == g, "bpm_mean"].values
        for g in ["sedentary", "light", "active"] if g in data["activity_group"].unique()
    }

    effect_rows = []
    if "active" in groups and "sedentary" in groups:
        d = cohens_d(groups["active"], groups["sedentary"])
        t, p = stats.ttest_ind(groups["active"], groups["sedentary"], equal_var=False)
        effect_rows.append({"Comparison": "Active vs Sedentary",
                             "Effect size": f"Cohen's d = {d:.2f}",
                             "Statistic": f"t = {t:.2f}", "p-value": f"{p:.4g}"})

    if len(groups) >= 2:
        eta2 = eta_squared_oneway(list(groups.values()))
        f_stat, p_anova = stats.f_oneway(*groups.values())
        effect_rows.append({"Comparison": "Overall (sedentary / light / active)",
                             "Effect size": f"eta^2 = {eta2:.3f}",
                             "Statistic": f"F = {f_stat:.2f}", "p-value": f"{p_anova:.4g}"})

    if data["met_mean"].notna().sum() > 5:
        valid = data.dropna(subset=["met_mean"])
        r_met, p_met = stats.pearsonr(valid["bpm_mean"], valid["met_mean"])
        effect_rows.append({"Comparison": "HR vs MET (continuous)",
                             "Effect size": f"Pearson r = {r_met:.2f}",
                             "Statistic": "-", "p-value": f"{p_met:.4g}"})

    if HAS_STATSMODELS and len(groups) >= 2:
        try:
            model = smf.mixedlm(
                "bpm_mean ~ C(activity_group, Treatment('sedentary'))",
                data, groups=data["participant"],
            )
            fit = model.fit()
            for name, coef, p in zip(fit.params.index, fit.params.values, fit.pvalues.values):
                if "activity_group" in name:
                    effect_rows.append({
                        "Comparison": f"Mixed model: {name.split('T.')[-1].rstrip(']')} vs sedentary "
                                      f"(participant random intercept)",
                        "Effect size": f"{coef:+.2f} bpm",
                        "Statistic": "-", "p-value": f"{p:.4g}",
                    })
        except Exception as e:
            print(f"  [WARN] Mixed-effects model failed to fit: {e}")
    elif not HAS_STATSMODELS:
        print("\n  [INFO] statsmodels not installed -- skipping the mixed-effects model.")
        print("  Install with: pip install statsmodels --break-system-packages")

    effect_df = pd.DataFrame(effect_rows)
    if not effect_df.empty:
        u.render_table_image(
            effect_df, "Effect Size: Higher HR vs Concurrent Physical Activity (classroom time)",
            os.path.join(OUTPUT_FOLDER, "table_effect_sizes.png"),
        )

    # --- Heatmap: participant x activity level mean HR ---
    pivot = data.pivot_table(index="activity_group", columns="participant",
                              values="bpm_mean", aggfunc="mean")
    pivot = pivot.reindex([g for g in ["sedentary", "light", "active"] if g in pivot.index])
    u.render_heatmap_image(
        pivot, "Mean HR (bpm) by Activity Level and Participant (classroom time)",
        os.path.join(OUTPUT_FOLDER, "heatmap_hr_by_activity_level.png"),
        cmap="magma", fmt_pct=False,
    )

    print("\nDone. IMPORTANT: open the diagnostic_alignment_check_*.png file and confirm")
    print(f"the {ACTIVITY_ARRAY_START_HOUR}:00 AM activity-array start assumption looks right")
    print("(HR bumps should visually line up with higher Oura activity codes) before")
    print("relying on the effect sizes above.")


if __name__ == "__main__":
    main()