"""
Analyze whether higher-movement participants have lower Oura ring bin coverage.

Joins:
  - 1_visualization/Movement/summary_by_participant.csv   (from mocopi_pipeline script)
  - 1_visualization/Bins/bin_summary.csv                  (from oura bin script)

Outputs (CSV + PNG figures + text report) to:
  1_visualization/Movement/BinsMovement/

Run this locally where the Research folder lives (same machine as the two
upstream scripts) -- this Claude session has no access to that filesystem.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# === Paths (match the conventions used by the two upstream scripts) ===
ROOT_PATH = "/Users/cibrian/Documents/GitHub/Research"
MOVEMENT_SUMMARY_CSV = os.path.join(ROOT_PATH, "1_visualization", "Movement", "summary_by_participant.csv")
# Note: bins script used a lowercase "Github" segment in its OUTPUT_DIR -- kept as-is here.
BINS_SUMMARY_CSV = "/Users/cibrian/Documents/Github/Research/1_visualization/Bins/bin_summary.csv"
OUTPUT_DIR = os.path.join(ROOT_PATH, "1_visualization", "Movement", "BinsMovement")

MOVEMENT_METRICS = ["Mean_Intensity", "Mean_Variability", "Mean_Jerk", "Active_Ratio_Pct"]


def load_and_merge():
    if not os.path.exists(MOVEMENT_SUMMARY_CSV):
        raise FileNotFoundError(f"Could not find movement summary at: {MOVEMENT_SUMMARY_CSV}")
    if not os.path.exists(BINS_SUMMARY_CSV):
        raise FileNotFoundError(f"Could not find bin summary at: {BINS_SUMMARY_CSV}")

    movement_df = pd.read_csv(MOVEMENT_SUMMARY_CSV)
    bins_df = pd.read_csv(BINS_SUMMARY_CSV)

    # Drop the synthetic "AVERAGE" row before merging
    bins_df = bins_df[bins_df["participant"] != "AVERAGE"].copy()

    # Both scripts use zero-padded IDs like "P001", "P014" -- normalize just in case
    movement_df["Participant"] = movement_df["Participant"].astype(str).str.upper().str.strip()
    bins_df["participant"] = bins_df["participant"].astype(str).str.upper().str.strip()

    merged = movement_df.merge(
        bins_df, left_on="Participant", right_on="participant", how="inner"
    )

    missing_in_bins = set(movement_df["Participant"]) - set(bins_df["participant"])
    missing_in_movement = set(bins_df["participant"]) - set(movement_df["Participant"])
    if missing_in_bins:
        print(f"[Warning] Participants in movement data but missing from bin data: {sorted(missing_in_bins)}")
    if missing_in_movement:
        print(f"[Warning] Participants in bin data but missing from movement data: {sorted(missing_in_movement)}")

    # Coverage rate is the fairer comparison since total_bins varies by participant schedule
    merged["Completion_Rate"] = merged["true_bins_after_hr_filter"] / merged["total_bins"]
    merged["Completion_Rate_Before_HR"] = merged["true_bins_before_hr_filter"] / merged["total_bins"]

    return merged


def run_correlations(merged):
    """Correlate each movement metric against raw true-bin counts and completion rate."""
    oura_targets = {
        "true_bins_after_hr_filter": "True bins (after HR filter, raw count)",
        "Completion_Rate": "Completion rate (after HR filter / total bins)",
        "true_bins_before_hr_filter": "True bins (before HR filter, raw count)",
    }

    results = []
    for mv_col in MOVEMENT_METRICS:
        for target_col, target_label in oura_targets.items():
            sub = merged[[mv_col, target_col]].dropna()
            if len(sub) < 3:
                continue
            r, p = pearsonr(sub[mv_col], sub[target_col])
            results.append({
                "Movement_Metric": mv_col,
                "Oura_Target": target_col,
                "Oura_Target_Label": target_label,
                "n": len(sub),
                "r": r,
                "p_value": p,
                "direction": "negative (supports hypothesis)" if r < 0 else "positive (against hypothesis)",
            })
    return pd.DataFrame(results)


def make_scatter_plots(merged, results_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    key_target = "Completion_Rate"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, mv_col in zip(axes, MOVEMENT_METRICS):
        sub = merged[[mv_col, key_target, "Participant"]].dropna()
        ax.scatter(sub[mv_col], sub[key_target], color="#3b6ea5", s=60, edgecolor="white", zorder=3)

        for _, row in sub.iterrows():
            ax.annotate(row["Participant"], (row[mv_col], row[key_target]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")

        if len(sub) >= 3:
            r, p = pearsonr(sub[mv_col], sub[key_target])
            z = np.polyfit(sub[mv_col], sub[key_target], 1)
            xs = np.linspace(sub[mv_col].min(), sub[mv_col].max(), 100)
            ax.plot(xs, np.polyval(z, xs), color="#c0392b", linestyle="--", zorder=2)
            ax.set_title(f"{mv_col} vs Oura Completion Rate\nr = {r:.3f}, p = {p:.3g}, n = {len(sub)}")
        else:
            ax.set_title(f"{mv_col} vs Oura Completion Rate (insufficient n)")

        ax.set_xlabel(mv_col)
        ax.set_ylabel("Completion Rate (true bins / total bins)")
        ax.grid(alpha=0.3)

    fig.suptitle("Movement Intensity vs. Oura Ring Bin Completion Rate", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, "movement_vs_completion_scatter.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {fig_path}")

    # Correlation heatmap across all movement metrics x oura targets
    if not results_df.empty:
        pivot = results_df.pivot(index="Movement_Metric", columns="Oura_Target", values="r")
        pivot = pivot.reindex(index=MOVEMENT_METRICS)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        im = ax2.imshow(pivot.values, cmap="RdBu", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(pivot.columns)))
        ax2.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                              color="white" if abs(val) > 0.5 else "black")
        ax2.set_title("Pearson r: Movement Metrics vs Oura Bin Coverage")
        fig2.colorbar(im, ax=ax2, label="Pearson r")
        fig2.tight_layout()
        heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
        fig2.savefig(heatmap_path, dpi=150)
        plt.close(fig2)
        print(f"[Saved] {heatmap_path}")

    # Bar chart ranking participants by intensity vs completion rate side-by-side
    ranked = merged.sort_values("Mean_Intensity", ascending=False)
    fig3, ax3a = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ranked))
    width = 0.4
    ax3a.bar(x - width/2, ranked["Mean_Intensity"], width, color="#3b6ea5", label="Mean Intensity")
    ax3a.set_ylabel("Mean Intensity", color="#3b6ea5")
    ax3a.set_xticks(x)
    ax3a.set_xticklabels(ranked["Participant"], rotation=45)
    ax3b = ax3a.twinx()
    ax3b.bar(x + width/2, ranked["Completion_Rate"], width, color="#e67e22", label="Completion Rate")
    ax3b.set_ylabel("Oura Completion Rate", color="#e67e22")
    fig3.suptitle("Participants Ranked by Movement Intensity vs. Oura Completion Rate")
    fig3.tight_layout()
    bar_path = os.path.join(output_dir, "ranked_intensity_vs_completion.png")
    fig3.savefig(bar_path, dpi=150)
    plt.close(fig3)
    print(f"[Saved] {bar_path}")


def write_report(merged, results_df, output_dir):
    merged_out = merged[[
        "Participant", "Mean_Intensity", "Mean_Variability", "Mean_Jerk", "Active_Ratio_Pct",
        "total_bins", "true_bins_before_hr_filter", "true_bins_after_hr_filter",
        "Completion_Rate_Before_HR", "Completion_Rate",
    ]].sort_values("Mean_Intensity", ascending=False)

    merged_csv_path = os.path.join(output_dir, "movement_vs_bins_merged.csv")
    merged_out.to_csv(merged_csv_path, index=False)

    results_csv_path = os.path.join(output_dir, "movement_vs_bins_correlations.csv")
    results_df.to_csv(results_csv_path, index=False)

    report_path = os.path.join(output_dir, "movement_vs_bins_report.txt")
    with open(report_path, "w") as f:
        f.write("============================================================\n")
        f.write(" MOVEMENT INTENSITY vs. OURA RING BIN COVERAGE\n")
        f.write(" Hypothesis: higher-movement participants have fewer true bins\n")
        f.write("============================================================\n\n")
        f.write(f"Participants included (n={len(merged_out)}): {merged_out['Participant'].tolist()}\n\n")

        f.write("--- Participants ranked by Mean Intensity (highest first) ---\n")
        f.write(merged_out.to_string(index=False))
        f.write("\n\n--- Correlation results ---\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n--- Interpretation guide ---\n")
        f.write("A negative r between a movement metric and true-bin count/completion rate\n")
        f.write("supports the hypothesis (more movement -> less Oura coverage).\n")
        f.write("p < 0.05 is conventionally 'statistically significant', but with a small\n")
        f.write("number of participants these correlations should be read as suggestive,\n")
        f.write("not confirmatory.\n")

    print(f"[Saved] {merged_csv_path}")
    print(f"[Saved] {results_csv_path}")
    print(f"[Saved] {report_path}")


def main():
    merged = load_and_merge()
    print(f"\nMerged {len(merged)} participants across movement + bin data.\n")

    results_df = run_correlations(merged)
    print("--- Correlation summary (Pearson r) ---")
    print(results_df.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    make_scatter_plots(merged, results_df, OUTPUT_DIR)
    write_report(merged, results_df, OUTPUT_DIR)

    # Quick headline check against the hypothesis
    key_row = results_df[
        (results_df["Movement_Metric"] == "Mean_Intensity") &
        (results_df["Oura_Target"] == "Completion_Rate")
    ]
    if not key_row.empty:
        r = key_row.iloc[0]["r"]
        p = key_row.iloc[0]["p_value"]
        verdict = "SUPPORTS" if r < 0 else "DOES NOT SUPPORT"
        print(f"\nHeadline check (Mean_Intensity vs Completion_Rate): r = {r:.3f}, p = {p:.3g}")
        print(f"-> This {verdict} the hypothesis that higher movement means lower Oura coverage.")

    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()