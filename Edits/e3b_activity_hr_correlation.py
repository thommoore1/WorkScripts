"""
e3b_activity_hr_correlation.py
--------------------------------
E3 (part b): Produce tables that specifically demonstrate the correlation
between heart rate and physical-activity level during classroom time.

This reuses the same bin-level dataset built by e3_activity_hr_effect.py
(5-minute bins, classroom time only, aligned to Oura's class_5_min /
met_1_min arrays) -- it does not rebuild or duplicate that alignment logic.

Produces in OUTPUT_FOLDER (set in config.py):
  1. table_correlation_overall.png
       Pearson r and Spearman rho between BPM and (a) the raw Oura
       activity-intensity code (0-4) and (b) MET, each with N, p-value,
       and a plain-language effect-size label (Cohen's small/medium/large
       thresholds for r).
  2. table_correlation_by_participant.png
       The same two correlations computed separately per participant, so
       you can see whether the BPM-activity relationship is consistent
       across people or driven by a subset of participants.
  3. table_bpm_by_activity_code.png
       Mean/SD BPM at each raw activity code (0-4), with N bins and N
       participants -- shows the dose-response shape behind the
       correlation number (i.e., is it roughly monotonic).
  4. heatmap_bpm_by_activity_code_participant.png
       Participant x activity-code heatmap of mean BPM, finer-grained
       than the group-level heatmap from e3_activity_hr_effect.py.

Run: python3 e3b_activity_hr_correlation.py
(Run e3_activity_hr_effect.py first, or just run this directly -- it will
build the dataset itself if needed.)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

from config import ROOT_PATH, OUTPUT_FOLDER
import utils as u
from e3_activity_hr_effect import build_bin_level_dataset


def cohen_r_label(r):
    """Plain-language effect-size label for a correlation coefficient (Cohen, 1988)."""
    a = abs(r)
    if pd.isna(r):
        return "n/a"
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "small"
    if a < 0.5:
        return "medium"
    return "large"


def correlate(x, y):
    """Return (pearson_r, pearson_p, spearman_rho, spearman_p, n) for two aligned arrays."""
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan, np.nan, np.nan, n)
    r, p_r = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    return (r, p_r, rho, p_rho, n)


def build_overall_table(data):
    rows = []
    r, p_r, rho, p_rho, n = correlate(data["activity_code"], data["bpm_mean"])
    rows.append({
        "Comparison": "BPM vs Oura activity code (0-4)",
        "N (bins)": n,
        "Pearson r": f"{r:.2f}" if pd.notna(r) else "NA",
        "Pearson p": f"{p_r:.4g}" if pd.notna(p_r) else "NA",
        "Spearman rho": f"{rho:.2f}" if pd.notna(rho) else "NA",
        "Spearman p": f"{p_rho:.4g}" if pd.notna(p_rho) else "NA",
        "Effect size": cohen_r_label(r),
    })

    valid_met = data.dropna(subset=["met_mean"])
    r2, p_r2, rho2, p_rho2, n2 = correlate(valid_met["met_mean"], valid_met["bpm_mean"])
    rows.append({
        "Comparison": "BPM vs MET (continuous)",
        "N (bins)": n2,
        "Pearson r": f"{r2:.2f}" if pd.notna(r2) else "NA",
        "Pearson p": f"{p_r2:.4g}" if pd.notna(p_r2) else "NA",
        "Spearman rho": f"{rho2:.2f}" if pd.notna(rho2) else "NA",
        "Spearman p": f"{p_rho2:.4g}" if pd.notna(p_rho2) else "NA",
        "Effect size": cohen_r_label(r2),
    })
    return pd.DataFrame(rows)


def build_per_participant_table(data):
    rows = []
    for participant, pdf in data.groupby("participant"):
        r, p_r, rho, p_rho, n = correlate(pdf["activity_code"], pdf["bpm_mean"])
        valid_met = pdf.dropna(subset=["met_mean"])
        r_met, p_met, rho_met, p_rho_met, n_met = correlate(valid_met["met_mean"], valid_met["bpm_mean"])
        rows.append({
            "Participant": participant,
            "N (bins)": n,
            "r: BPM vs code": f"{r:.2f}" if pd.notna(r) else "NA",
            "p": f"{p_r:.3g}" if pd.notna(p_r) else "NA",
            "r: BPM vs MET": f"{r_met:.2f}" if pd.notna(r_met) else "NA",
            "p ": f"{p_met:.3g}" if pd.notna(p_met) else "NA",
        })
    df = pd.DataFrame(rows).sort_values("Participant").reset_index(drop=True)
    return df


def build_dose_response_table(data):
    grp = data.groupby("activity_code").agg(
        Participants=("participant", "nunique"),
        Bins=("bpm_mean", "size"),
        Mean_BPM=("bpm_mean", "mean"),
        SD_BPM=("bpm_mean", "std"),
    ).reset_index()
    grp["Mean_BPM"] = grp["Mean_BPM"].round(1)
    grp["SD_BPM"] = grp["SD_BPM"].round(1)
    grp = grp.rename(columns={
        "activity_code": "Oura Activity Code (0-4)", "Participants": "Participants (n)",
        "Bins": "Bins (n)", "Mean_BPM": "Mean HR (bpm)", "SD_BPM": "SD HR (bpm)",
    })
    return grp


def main():
    print(f"ROOT_PATH   = {ROOT_PATH}")
    print(f"OUTPUT_FOLDER = {OUTPUT_FOLDER}\n")

    print("Building bin-level classroom HR + activity dataset (same as E3a)...")
    data = build_bin_level_dataset(ROOT_PATH)
    if data.empty or "activity_code" not in data.columns:
        print("No overlapping HR + DailyActivity classroom data found (or activity_code missing --")
        print("make sure you're using the updated e3_activity_hr_effect.py).")
        return

    print(f"Total classroom 5-min bins: {len(data)}, participants: {data['participant'].nunique()}\n")

    print("Computing overall correlation table...")
    overall = build_overall_table(data)
    u.render_table_image(
        overall, "Correlation Between Heart Rate and Physical-Activity Level (classroom time)",
        os.path.join(OUTPUT_FOLDER, "table_correlation_overall.png"),
    )

    print("Computing per-participant correlation table...")
    per_participant = build_per_participant_table(data)
    u.render_table_image(
        per_participant, "HR-Activity Correlation by Participant (classroom time)",
        os.path.join(OUTPUT_FOLDER, "table_correlation_by_participant.png"),
    )

    print("Computing dose-response (mean HR by activity code) table...")
    dose_response = build_dose_response_table(data)
    u.render_table_image(
        dose_response, "Mean Heart Rate by Oura Activity Intensity Code (classroom time)",
        os.path.join(OUTPUT_FOLDER, "table_bpm_by_activity_code.png"),
    )

    print("Computing participant x activity-code heatmap...")
    pivot = data.pivot_table(index="activity_code", columns="participant",
                              values="bpm_mean", aggfunc="mean")
    u.render_heatmap_image(
        pivot, "Mean HR (bpm) by Activity Code and Participant (classroom time)",
        os.path.join(OUTPUT_FOLDER, "heatmap_bpm_by_activity_code_participant.png"),
        cmap="magma", fmt_pct=False,
    )

    print("\nDone. The per-participant table shows whether the BPM-activity relationship")
    print("is consistent across people; the dose-response table shows whether higher")
    print("activity codes correspond to progressively higher mean HR.")


if __name__ == "__main__":
    main()