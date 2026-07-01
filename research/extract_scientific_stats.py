import os
import re
import pandas as pd
import numpy as np

def main():
    print("Starting scientific stats extraction...")
    
    # 1. Paths definition
    bf16_dir = r"C:\Aditya_Data\Personal\ResearchThesis\research\outputs\real_traces_bf16_ladder\qwen2p5_7b"
    fourbit_dir = r"C:\Aditya_Data\Personal\ResearchThesis\research\outputs\real_traces_l4_qwen_7b_4bit"
    
    if not os.path.exists(bf16_dir):
        print(f"Error: bf16 directory not found at {bf16_dir}")
        return
    if not os.path.exists(fourbit_dir):
        print(f"Error: 4bit directory not found at {fourbit_dir}")
        return
        
    print(f"Using bf16 directory: {bf16_dir}")
    print(f"Using 4bit directory: {fourbit_dir}")
    
    bf16_steps_path = os.path.join(bf16_dir, "trace_steps.csv")
    fourbit_steps_path = os.path.join(fourbit_dir, "trace_steps.csv")
    
    # 2. Extract step-by-step stats (steps 1 to 10)
    print("Loading step-by-step trace files...")
    df_bf16 = pd.read_csv(bf16_steps_path)
    df_4bit = pd.read_csv(fourbit_steps_path)
    
    # Filter for steps 1 to 10
    df_bf16 = df_bf16[(df_bf16['step'] >= 1) & (df_bf16['step'] <= 10)]
    df_4bit = df_4bit[(df_4bit['step'] >= 1) & (df_4bit['step'] <= 10)]
    
    # Check what columns we have and if they are numeric
    cols_to_extract = ['correct', 'entropy_mean', 'hidden_l2_shift']
    for col in cols_to_extract:
        df_bf16[col] = pd.to_numeric(df_bf16[col], errors='coerce')
        df_4bit[col] = pd.to_numeric(df_4bit[col], errors='coerce')
        
    # Group by step and aggregate
    bf16_grouped = df_bf16.groupby('step')[cols_to_extract].mean().reset_index()
    fourbit_grouped = df_4bit.groupby('step')[cols_to_extract].mean().reset_index()
    
    # Merge for comparison
    comparison_df = pd.merge(bf16_grouped, fourbit_grouped, on='step', suffixes=('_bf16', '_4bit'))
    
    # Rename columns for clarity
    comparison_df.rename(columns={
        'correct_bf16': 'correctness_q_t_bf16',
        'correct_4bit': 'correctness_q_t_4bit',
        'entropy_mean_bf16': 'mean_entropy_bf16',
        'entropy_mean_4bit': 'mean_entropy_4bit',
        'hidden_l2_shift_bf16': 'hidden_l2_drift_bf16',
        'hidden_l2_shift_4bit': 'hidden_l2_drift_4bit'
    }, inplace=True)
    
    # Save step stats
    step_stats_out_path = r"C:\Aditya_Data\Personal\ResearchThesis\research\outputs\qwen_bf16_vs_4bit_step_stats.csv"
    comparison_df.to_csv(step_stats_out_path, index=False)
    print(f"Saved step stats to {step_stats_out_path}")
    
    # Print the step stats table
    print("\n=========================================================================")
    print("STEP-BY-STEP COMPARISON: QWEN 2.5 7B BF16 vs 4BIT")
    print("=========================================================================")
    print(comparison_df.to_string(index=False, formatters={
        'correctness_q_t_bf16': '{:.4f}'.format,
        'correctness_q_t_4bit': '{:.4f}'.format,
        'mean_entropy_bf16': '{:.4f}'.format,
        'mean_entropy_4bit': '{:.4f}'.format,
        'hidden_l2_drift_bf16': '{:.4f}'.format,
        'hidden_l2_drift_4bit': '{:.4f}'.format
    }))
    print("=========================================================================\n")
    
    # 3. Temperature specific analysis for bf16
    detector_path = os.path.join(bf16_dir, "detector_comparison_by_run.csv")
    if not os.path.exists(detector_path):
        print(f"Error: Could not find {detector_path}")
        return
        
    print(f"Loading detector comparison file: {detector_path}")
    df_det = pd.read_csv(detector_path)
    
    def parse_temp(run_id):
        match = re.search(r'temp(\d+\.\d+)', str(run_id))
        if match:
            val = float(match.group(1))
            if abs(val - 0.1) < 0.05:
                return 0.1
            elif abs(val - 0.6) < 0.05:
                return 0.6
            elif abs(val - 1.0) < 0.05:
                return 1.0
            return val
        return None
        
    # Extract temperature
    df_det['temperature'] = df_det['run_id'].apply(parse_temp)
    
    # Filter for valid temperatures and drop rows without temperature
    df_det = df_det[df_det['temperature'].isin([0.1, 0.6, 1.0])]
    
    # Filter detectors
    df_det_filtered = df_det[df_det['detector'].isin(['hazard_drift', 'never_stop'])].copy()
    
    # Pivot utility and stop_step with pivot_table to handle any duplicate rows robustly
    pivoted_util = df_det_filtered.pivot_table(index=['run_id', 'temperature'], columns='detector', values='stop_utility', aggfunc='mean').reset_index()
    pivoted_step = df_det_filtered.pivot_table(index=['run_id', 'temperature'], columns='detector', values='stop_step', aggfunc='mean').reset_index()
    
    pivoted = pd.merge(pivoted_util, pivoted_step, on=['run_id', 'temperature'], suffixes=('_util', '_step'))
    
    # Calculate win rates and mean stopping step
    temp_records = []
    for temp in [0.1, 0.6, 1.0]:
        df_temp = pivoted[pivoted['temperature'] == temp]
        if df_temp.empty:
            print(f"Warning: No runs found for temperature {temp}")
            continue
            
        # Drop NaNs for utility comparison
        df_temp_clean = df_temp.dropna(subset=['hazard_drift_util', 'never_stop_util'])
        
        total_runs = len(df_temp_clean)
        # Strict win: hazard_drift_util > never_stop_util
        strict_wins = (df_temp_clean['hazard_drift_util'] > df_temp_clean['never_stop_util']).sum()
        strict_win_rate = strict_wins / total_runs if total_runs > 0 else 0.0
        
        # Non-strict win: hazard_drift_util >= never_stop_util
        non_strict_wins = (df_temp_clean['hazard_drift_util'] >= df_temp_clean['never_stop_util']).sum()
        non_strict_win_rate = non_strict_wins / total_runs if total_runs > 0 else 0.0
        
        # Mean stopping step of hazard_drift
        mean_stop_step_hd = df_temp_clean['hazard_drift_step'].mean()
        
        # Mean stop utilities
        mean_util_hd = df_temp_clean['hazard_drift_util'].mean()
        mean_util_ns = df_temp_clean['never_stop_util'].mean()
        
        temp_records.append({
            'temperature': temp,
            'total_runs': total_runs,
            'strict_win_rate': strict_win_rate,
            'non_strict_win_rate': non_strict_win_rate,
            'mean_stopping_step_hazard_drift': mean_stop_step_hd,
            'mean_utility_hazard_drift': mean_util_hd,
            'mean_utility_never_stop': mean_util_ns
        })
        
    temp_df = pd.DataFrame(temp_records)
    
    # Save temperature stats
    temp_stats_out_path = r"C:\Aditya_Data\Personal\ResearchThesis\research\outputs\qwen_temperature_stats.csv"
    temp_df.to_csv(temp_stats_out_path, index=False)
    print(f"Saved temperature stats to {temp_stats_out_path}")
    
    # Print the temperature stats table
    print("\n=========================================================================")
    print("TEMPERATURE-SPECIFIC ANALYSIS: QWEN 2.5 7B BF16")
    print("=========================================================================")
    print(temp_df.to_string(index=False, formatters={
        'strict_win_rate': '{:.4%}'.format,
        'non_strict_win_rate': '{:.4%}'.format,
        'mean_stopping_step_hazard_drift': '{:.4f}'.format,
        'mean_utility_hazard_drift': '{:.4f}'.format,
        'mean_utility_never_stop': '{:.4f}'.format
    }))
    print("=========================================================================\n")

if __name__ == "__main__":
    main()
