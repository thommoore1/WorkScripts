import os
import re
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# === Paths ===
# Root of the research repo. Participant data lives at ROOT_PATH/P###/Mocopi/Labeled,
# same layout used by the heatmap/coverage script.
ROOT_PATH = "/Users/cibrian/Documents/GitHub/Research"
SUMMARY_DIR = os.path.join(ROOT_PATH, "1_visualization", "Movement")


def get_participant_number(name):
    return int(name[1:])


def clean_and_combine_raw_data(base_directory, participant_id):
    """
    [Phase 1 - Jaime's Logic]
    Dynamically scans the directory for YYYY-MM-DD subfolders, maps sensors 
    from raw file names, and merges everything into a chronological master CSV.
    """
    print(f"\n[{participant_id} - Step 1/3] Combining raw daily files...")
    
    # Dynamically find any subfolders matching the YYYY-MM-DD format
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    dates = [d for d in os.listdir(base_directory) if date_pattern.match(d) and os.path.isdir(os.path.join(base_directory, d))]
    dates.sort()
    
    if not dates:
        print(f"      [Error] No date subfolders (YYYY-MM-DD) found inside {base_directory}.")
        return None
        
    print(f"      Found date subfolders: {dates}")
    combined_list = []

    for date in dates:
        day_folder = os.path.join(base_directory, date)
        for file in os.listdir(day_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(day_folder, file)
                
                # Jaime's exact naming parser to find the sensor placement
                parts = file.split("Mocopi")
                if len(parts) > 1:
                    sensor = parts[1].split("Device")[0]
                else:
                    sensor = "Unknown"
                    
                try:
                    temp_df = pd.read_csv(file_path)
                    temp_df['Sensor'] = sensor
                    temp_df['Date'] = date
                    combined_list.append(temp_df)
                except Exception as e:
                    print(f"      Error reading {file}: {e}")

    if not combined_list:
        print(f"      [Error] No raw CSV files found inside the date folders for {participant_id}.")
        return None

    combined = pd.concat(combined_list, ignore_index=True)
    
    # Jaime's timeseries sorting logic
    combined = combined.sort_values(['Sensor', 'Date', 'Time_In_PST']).reset_index(drop=True)
    
    # Jaime's exact raw Jerk and Acceleration Magnitude calculations
    print(f"[{participant_id} - Step 2/3] Computing raw kinematics (Acc Magnitude, Jerk)...")
    combined['Jerk_X'] = combined.groupby(['Sensor', 'Date'])['Acceleration X'].diff()
    combined['Jerk_Y'] = combined.groupby(['Sensor', 'Date'])['Acceleration Y'].diff()
    combined['Jerk_Z'] = combined.groupby(['Sensor', 'Date'])['Acceleration Z'].diff()

    combined['Jerk_Mag'] = np.sqrt(
        combined['Jerk_X']**2 +
        combined['Jerk_Y']**2 +
        combined['Jerk_Z']**2
    )

    combined['Acc_Mag'] = np.sqrt(
        combined['Acceleration X']**2 +
        combined['Acceleration Y']**2 +
        combined['Acceleration Z']**2
    )

    # Clean up temporary component columns to save runtime memory
    combined = combined.drop(columns=['Jerk_X', 'Jerk_Y', 'Jerk_Z'])
    
    # Save master raw combined file
    output_combined_path = os.path.join(base_directory, f"{participant_id}_dates_combined.csv")
    combined.to_csv(output_combined_path, index=False)
    print(f"      Master combined dataset saved to: {output_combined_path}")
    return combined


def generate_epoch_features(combined_df, base_directory, participant_id, movement_threshold=1.15):
    """
    [Phase 2 - Your Epoch Logic]
    Aggregates the sub-second signals into clean 1-minute windowed averages,
    variabilities, jerk profiles, and active-vs-sedentary categorizations.
    """
    print(f"[{participant_id} - Step 3/3] Epoching raw data to 1-minute blocks...")
    
    # Combine recording date with clock time so epoch timestamps use the actual school day.
    combined_df['Timestamp'] = pd.to_datetime(
        combined_df['Date'].astype(str) + ' ' + combined_df['Time_In_PST'].astype(str),
        errors='coerce',
    )
    combined_df['Epoch_1Min'] = combined_df['Timestamp'].dt.floor('1min')
    
    # Calculate window metrics (including standard deviation for variability)
    epoch_df = combined_df.groupby(['Sensor', 'Date', 'class', 'Epoch_1Min']).agg(
        Intensity=('Acc_Mag', 'mean'),
        Variability=('Acc_Mag', 'std'),  # Movement variability
        Jerk=('Jerk_Mag', 'mean')        # Windowed average Jerk
    ).reset_index()
    
    # Classify active vs low-movement window ratios
    epoch_df['Is_Active'] = (epoch_df['Intensity'] > movement_threshold).astype(int)
    epoch_df['Participant'] = participant_id
    
    # Extract Hour of Day for Time of Day stratification
    epoch_df['Hour'] = epoch_df['Epoch_1Min'].dt.hour
    epoch_df['Time_of_Day'] = pd.cut(
        epoch_df['Hour'],
        bins=[0, 11, 14, 24],
        labels=['Morning', 'Midday', 'Afternoon'],
        right=False
    )
    
    # Save the windowed features CSV
    output_features_path = os.path.join(base_directory, f"{participant_id}_epoch_kinematics.csv")
    epoch_df.to_csv(output_features_path, index=False)
    print(f"      Completed kinematic features saved to: {output_features_path}")
    return epoch_df


def print_results(epoch_df, participant_id):
    """
    Generates command-line printouts of comparisons and sensor correlations
    for a single participant (unchanged from the original per-participant view).
    """
    print(f"\n========================================================")
    print(f" RESULTS ANALYSIS FOR {participant_id}")
    print(f"========================================================")
    
    dimensions = {
        'Sensor Placement': 'Sensor',
        'Classroom Context': 'class',
        'Time of Day': 'Time_of_Day'
    }
    
    for label, col in dimensions.items():
        print(f"\n--- Stratification by {label} ---")
        summary = epoch_df.groupby(col).agg(
            Mean_Intensity=('Intensity', 'mean'),
            SD_Intensity=('Intensity', 'std'),
            Mean_Variability=('Variability', 'mean'),
            SD_Variability=('Variability', 'std'),
            Mean_Jerk=('Jerk', 'mean'),
            SD_Jerk=('Jerk', 'std'),
            Active_Ratio=('Is_Active', 'mean')
        ).reset_index()
        
        summary['Active_Ratio_Pct'] = (summary['Active_Ratio'] * 100).round(2)
        print(summary.to_string(index=False))

    # Calculate and output Cross-Body Correlations
    print(f"\n--- Cross-Body Sensor Correlations ({participant_id}) ---")
    pivot_df = epoch_df.pivot_table(
        index=['Participant', 'Date', 'class', 'Epoch_1Min'],
        columns='Sensor',
        values='Intensity'
    ).dropna()
    
    if pivot_df.empty:
        print("Warning: Insufficient overlapping epochs to run sensor correlations.")
        return
        
    corr_matrix = pivot_df.corr(method='pearson')
    print("\nPearson Correlation Matrix (r):")
    print(corr_matrix.round(3))
    
    cols = corr_matrix.columns
    print("\nPairwise Cross-Body Correlations (p-values):")
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1, s2 = cols[i], cols[j]
            r_val, p_val = pearsonr(pivot_df[s1], pivot_df[s2])
            print(f"  {s1} vs {s2}: r = {r_val:.3f} (p = {p_val:.3e})")


def stratified_summary(all_epochs_df, group_col):
    """Same aggregation as print_results, but across every participant at once."""
    summary = all_epochs_df.groupby(group_col).agg(
        Mean_Intensity=('Intensity', 'mean'),
        SD_Intensity=('Intensity', 'std'),
        Mean_Variability=('Variability', 'mean'),
        SD_Variability=('Variability', 'std'),
        Mean_Jerk=('Jerk', 'mean'),
        SD_Jerk=('Jerk', 'std'),
        Active_Ratio=('Is_Active', 'mean')
    ).reset_index()
    summary['Active_Ratio_Pct'] = (summary['Active_Ratio'] * 100).round(2)
    return summary


def cross_body_correlations(all_epochs_df):
    """Pearson correlation matrix + pairwise stats, pooled across all participants."""
    pivot_df = all_epochs_df.pivot_table(
        index=['Participant', 'Date', 'class', 'Epoch_1Min'],
        columns='Sensor',
        values='Intensity'
    ).dropna()

    if pivot_df.empty:
        return None, []

    corr_matrix = pivot_df.corr(method='pearson')
    cols = corr_matrix.columns
    pairwise = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1, s2 = cols[i], cols[j]
            r_val, p_val = pearsonr(pivot_df[s1], pivot_df[s2])
            pairwise.append({'Sensor_1': s1, 'Sensor_2': s2, 'r': r_val, 'p_value': p_val})

    return corr_matrix, pairwise


def write_group_summary(all_epochs_df, summary_dir):
    """
    Builds and writes the cross-participant summary artifacts:
      - one combined epoch-level CSV
      - stratified summaries by sensor / class / time of day / participant
      - a pooled cross-body correlation matrix + pairwise stats
      - a human-readable text report
    """
    os.makedirs(summary_dir, exist_ok=True)

    # 1. Combined epoch-level dataset across all participants
    combined_path = os.path.join(summary_dir, "all_participants_epoch_kinematics.csv")
    all_epochs_df.to_csv(combined_path, index=False)

    # 2. Stratified summaries
    dimensions = {
        'Sensor': 'summary_by_sensor.csv',
        'class': 'summary_by_class.csv',
        'Time_of_Day': 'summary_by_time_of_day.csv',
        'Participant': 'summary_by_participant.csv',
    }
    summaries = {}
    for col, filename in dimensions.items():
        summary = stratified_summary(all_epochs_df, col)
        summaries[col] = summary
        summary.to_csv(os.path.join(summary_dir, filename), index=False)

    # 3. Pooled cross-body correlations
    corr_matrix, pairwise = cross_body_correlations(all_epochs_df)
    if corr_matrix is not None:
        corr_matrix.to_csv(os.path.join(summary_dir, "cross_body_correlation_matrix.csv"))
        pd.DataFrame(pairwise).to_csv(os.path.join(summary_dir, "cross_body_pairwise_correlations.csv"), index=False)

    # 4. Human-readable text report
    report_path = os.path.join(summary_dir, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("========================================================\n")
        f.write(" CROSS-PARTICIPANT MOCOPI SUMMARY\n")
        f.write("========================================================\n")
        f.write(f"Participants included: {sorted(all_epochs_df['Participant'].unique().tolist())}\n")

        for col in ['Sensor', 'class', 'Time_of_Day', 'Participant']:
            f.write(f"\n--- Stratification by {col} ---\n")
            f.write(summaries[col].to_string(index=False))
            f.write("\n")

        f.write("\n--- Pooled Cross-Body Sensor Correlations (all participants) ---\n")
        if corr_matrix is not None:
            f.write("\nPearson Correlation Matrix (r):\n")
            f.write(corr_matrix.round(3).to_string())
            f.write("\n\nPairwise Cross-Body Correlations (p-values):\n")
            for row in pairwise:
                f.write(f"  {row['Sensor_1']} vs {row['Sensor_2']}: r = {row['r']:.3f} (p = {row['p_value']:.3e})\n")
        else:
            f.write("Warning: Insufficient overlapping epochs to run sensor correlations.\n")

    print(f"\n[Summary] Cross-participant summary written to: {summary_dir}")
    print(f"[Summary]   - {combined_path}")
    for filename in dimensions.values():
        print(f"[Summary]   - {os.path.join(summary_dir, filename)}")
    if corr_matrix is not None:
        print(f"[Summary]   - {os.path.join(summary_dir, 'cross_body_correlation_matrix.csv')}")
        print(f"[Summary]   - {os.path.join(summary_dir, 'cross_body_pairwise_correlations.csv')}")
    print(f"[Summary]   - {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run complete MOCOPI Processing and Feature Analysis pipeline.")
    parser.add_argument(
        "--participant",
        type=str,
        default="all",
        help="Specific participant ID to process (e.g. 'P002', 'P014'), or 'all' (default) to run every participant found."
    )
    args = parser.parse_args()

    if not os.path.exists(ROOT_PATH):
        print(f"[Fatal Error] Could not find research root folder: {ROOT_PATH}")
        return

    print(f"Using root directory: {ROOT_PATH}")

    # Identify participants by scanning ROOT_PATH for P### folders that actually
    # have a Mocopi/Labeled directory (same discovery approach as the heatmap script).
    candidate_participants = sorted(
        (d for d in os.listdir(ROOT_PATH)
         if os.path.isdir(os.path.join(ROOT_PATH, d)) and re.match(r'^P\d+$', d)),
        key=get_participant_number
    )

    all_participants = []
    for p in candidate_participants:
        labeled_path = os.path.join(ROOT_PATH, p, "Mocopi", "Labeled")
        if os.path.exists(labeled_path):
            all_participants.append(p)
        else:
            print(f"[Info] Skipping {p}: no Mocopi/Labeled folder found at {labeled_path}")

    if args.participant.lower() == "all":
        participants = all_participants
    else:
        participants = [args.participant] if args.participant in all_participants else []
        if not participants:
            print(f"[Fatal Error] Participant '{args.participant}' not found (or missing Mocopi/Labeled data) under {ROOT_PATH}.")
            return

    print(f"Found {len(participants)} participant directories to process:")
    print(participants)

    all_epoch_frames = []

    for p_id in participants:
        folder_path = os.path.join(ROOT_PATH, p_id, "Mocopi", "Labeled")
        if not os.path.exists(folder_path):
            print(f"\n[Warning] Participant folder does not exist at: '{folder_path}'. Skipping.")
            continue

        print(f"\n========================================================")
        print(f" RUNNING FULL PIPELINE FOR PARTICIPANT: {p_id}")
        print(f"========================================================")

        # Step 1 & 2: Clean and merge raw timeseries (Jaime's pipeline)
        combined_data = clean_and_combine_raw_data(folder_path, p_id)

        if combined_data is not None:
            # Step 3: Run windowed feature aggregation (Your pipeline)
            epoch_data = generate_epoch_features(combined_data, folder_path, p_id)
            # Per-participant console output (unchanged behavior)
            print_results(epoch_data, p_id)
            all_epoch_frames.append(epoch_data)

    # Cross-participant summary, written out to the shared Movement folder
    if all_epoch_frames:
        all_epochs_df = pd.concat(all_epoch_frames, ignore_index=True)
        write_group_summary(all_epochs_df, SUMMARY_DIR)
    else:
        print("\n[Warning] No participants produced usable data — skipping cross-participant summary.")


if __name__ == "__main__":
    main()