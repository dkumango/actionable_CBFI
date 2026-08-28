"""
========================================================================================
Aggregation Script: result_benchmark/aggregate_benchmark_table3.py
========================================================================================
Reads benchmark CSV results from result_benchmark/ and aggregates metrics grouped by:
  (Model Architecture, Method)
Filtering Methods:
  - Actionable CBFI (Proposed)
  - Untargeted Causal (Karimi 2021)

Model Architectures:
  - SVM
  - RandomForest
  - XGBoost
  - [Overall All Models]

Subsets Included in Output Table:
  - [TOTAL] (All benchmark instances for each Model Architecture)
  - [BOTH-SUCCESS MEAN COMPARISON] (Instances where both A-CBFI and Untargeted succeeded)
  - [DIVERGENT PRESCRIPTIONS] (Instances where A-CBFI and Untargeted prescribed different levers)

Saves results to: result_benchmark/aggregate_benchmark_table3.csv
========================================================================================
"""

import os
import ast
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def safe_to_csv(df: pd.DataFrame, filepath: str):
    """Safely saves DataFrame to CSV, handling file lock/permission errors gracefully."""
    try:
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n[Success] Table saved to '{filepath}'")
    except PermissionError:
        alt_path = filepath.replace('.csv', '_new.csv')
        try:
            df.to_csv(alt_path, index=False, encoding='utf-8-sig')
            print(f"\n[Warning] File '{filepath}' is currently locked. Saved to '{alt_path}' instead.")
        except Exception as e:
            print(f"\n[Error] Could not save CSV file '{filepath}': {e}")

def get_col(df: pd.DataFrame, names: list):
    """Helper to retrieve the first matching non-empty column from a list of candidate names."""
    for n in names:
        if n in df.columns and not df[n].isna().all():
            return df[n]
    return None

def parse_l_active(val):
    """Parses stringified Intervention_Nodes list to compute L_active = |V_do|."""
    if pd.isna(val): return 0
    try:
        nodes = ast.literal_eval(val) if isinstance(val, str) else val
        return len(nodes)
    except Exception:
        return 0

def compute_metrics(acbfi_sub: pd.DataFrame, method_sub: pd.DataFrame, method_name: str) -> dict:
    """Helper function to compute mean metrics and Wilcoxon p-values against Actionable CBFI."""
    if method_sub.empty:
        return {
            'Time (s)': "", 'Evaluations': "", 'Success Rate': "",
            'Recourse Cost': "", 'RCR': "", 'L_active': "", 'Sparsity (L0)': "",
            'R_SCM': "", 'R_0': "", 'D_M': "",
            'Lactive p-value (vs A-CBFI)': "", 'Cost p-value (vs A-CBFI)': "",
            'RCR p-value (vs A-CBFI)': "", 'R_SCM p-value (vs A-CBFI)': ""
        }

    time_series = get_col(method_sub, ['Time (s)'])
    eval_series = get_col(method_sub, ['Evaluations'])
    succ_series = get_col(method_sub, ['Success'])
    cost_series = get_col(method_sub, ['Recourse Cost (Pure MAD)', 'Recourse Cost'])
    acbfi_cost_series = get_col(acbfi_sub, ['Recourse Cost (Pure MAD)', 'Recourse Cost'])
    rcr_series = get_col(method_sub, ['RCR (%)'])
    lactive_series = get_col(method_sub, ['L_active'])
    l0_series = get_col(method_sub, ['Sparsity (L0)', 'Active Levers'])
    r_scm_series = get_col(method_sub, ['R_SCM', 'Causal Plausibility MSE'])
    r_0_series = get_col(method_sub, ['R_0'])
    d_m_series = get_col(method_sub, ['D_M'])

    time_val = f"{time_series.mean():.4f}" if time_series is not None else ""
    eval_val = f"{eval_series.mean():.1f}" if eval_series is not None else ""
    succ_val = f"{succ_series.mean()*100:.1f}%" if succ_series is not None else ""
    cost_val = f"{cost_series.mean():.4f}" if cost_series is not None else ""

    if rcr_series is not None:
        rcr_val = f"{rcr_series.mean():.2f}%"
    elif cost_series is not None and acbfi_cost_series is not None and acbfi_cost_series.mean() > 0:
        rcr_val = f"{cost_series.mean() / acbfi_cost_series.mean():.4f}"
    else:
        rcr_val = "1.0000"

    lactive_val = f"{lactive_series.mean():.4f}" if lactive_series is not None else ""
    l0_val = f"{l0_series.mean():.4f}" if l0_series is not None else ""
    r_scm_val = f"{r_scm_series.mean():.4f}" if r_scm_series is not None else ""
    r_0_val = f"{r_0_series.mean():.4f}" if r_0_series is not None else ""
    d_m_val = f"{d_m_series.mean():.4f}" if d_m_series is not None else ""

    if method_name == 'Actionable CBFI':
        lactive_pval_str = "ref"
        cost_pval_str = "ref"
        rcr_pval_str = "ref"
        r_scm_pval_str = "ref"
    else:
        merge_cols = ['Dataset', 'Model_Clean', 'Instance']
        acbfi_cols = merge_cols.copy()
        method_cols = merge_cols.copy()

        if lactive_series is not None: acbfi_cols.append(lactive_series.name); method_cols.append(lactive_series.name)
        elif l0_series is not None: acbfi_cols.append(l0_series.name); method_cols.append(l0_series.name)
        if cost_series is not None: acbfi_cols.append(cost_series.name); method_cols.append(cost_series.name)
        if rcr_series is not None: acbfi_cols.append(rcr_series.name); method_cols.append(rcr_series.name)
        if r_scm_series is not None: acbfi_cols.append(r_scm_series.name); method_cols.append(r_scm_series.name)

        merged = pd.merge(
            acbfi_sub[acbfi_cols],
            method_sub[method_cols],
            on=merge_cols,
            suffixes=('_acbfi', '_method')
        )

        def calc_wilcoxon(series_name):
            col_acbfi = f"{series_name}_acbfi"
            col_method = f"{series_name}_method"
            if col_acbfi in merged.columns and col_method in merged.columns:
                mask = ~merged[col_acbfi].isna() & ~merged[col_method].isna()
                if mask.sum() > 5:
                    try:
                        _, p = wilcoxon(merged.loc[mask, col_method], merged.loc[mask, col_acbfi], zero_method='pratt')
                        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                        return f"{p:.2e} ({stars})"
                    except Exception:
                        return "1.00e+00 (ns)"
            return ""

        target_l_col = lactive_series.name if lactive_series is not None else (l0_series.name if l0_series is not None else "")
        lactive_pval_str = calc_wilcoxon(target_l_col) if target_l_col else ""
        cost_pval_str = calc_wilcoxon(cost_series.name) if cost_series is not None else ""
        rcr_pval_str = calc_wilcoxon(rcr_series.name) if rcr_series is not None else ""
        r_scm_pval_str = calc_wilcoxon(r_scm_series.name) if r_scm_series is not None else ""

    res = {
        'Time (s)': time_val,
        'Evaluations': eval_val,
        'Success Rate': succ_val,
        'Recourse Cost': cost_val,
        'RCR': rcr_val,
        'L_active': lactive_val,
        'Sparsity (L0)': l0_val,
    }
    if r_scm_series is not None: res['R_SCM'] = r_scm_val
    if r_0_series is not None: res['R_0'] = r_0_val
    if d_m_series is not None: res['D_M'] = d_m_val

    res['Lactive p-value (vs A-CBFI)'] = lactive_pval_str
    res['Cost p-value (vs A-CBFI)'] = cost_pval_str
    if rcr_series is not None: res['RCR p-value (vs A-CBFI)'] = rcr_pval_str
    if r_scm_series is not None: res['R_SCM p-value (vs A-CBFI)'] = r_scm_pval_str

    return res

def run_table3_aggregation():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load all extended_benchmark_*.csv files
    file_list = [
        'extended_benchmark_adult.csv',
        'extended_benchmark_financial_loan.csv',
        'extended_benchmark_german_credit.csv',
        'extended_benchmark_breast_cancer.csv',
        'extended_benchmark_diabetes.csv',
        'extended_benchmark_medical_insurance.csv'
    ]

    # Filter Methods: ONLY Actionable CBFI and Untargeted Causal
    method_order = [
        'Actionable CBFI',
        'Untargeted Causal'
    ]

    def map_method_name(name):
        if 'Actionable CBFI' in name: return 'Actionable CBFI'
        elif 'Untargeted Causal' in name: return 'Untargeted Causal'
        return None

    model_order = ['SVM', 'RandomForest', 'XGBoost']

    def map_model_name(name):
        if 'SVM' in name: return 'SVM'
        elif 'RandomForest' in name: return 'RandomForest'
        elif 'XGB' in name: return 'XGBoost'
        return name

    all_dfs = []

    for fname in file_list:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f"[Warning] File not found: {fpath}")
            continue

        df = pd.read_csv(fpath)
        if 'Intervention_Nodes' in df.columns:
            df['L_active'] = df['Intervention_Nodes'].apply(parse_l_active)
        elif 'Active Levers' in df.columns:
            df['L_active'] = df['Active Levers']
        else:
            df['L_active'] = df['Sparsity (L0)']

        df['Method_Clean'] = df['Method'].apply(map_method_name)
        df['Model_Clean'] = df['Model_Architecture'].apply(map_model_name)
        
        # Filter for the two target methods only
        df = df[df['Method_Clean'].isin(method_order)].copy()
        all_dfs.append(df)

    if not all_dfs:
        print("[Error] No benchmark CSV files found to aggregate.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)

    # -------------------------------------------------------------------------------------
    # Identify Both-Success and Divergent Subsets
    # -------------------------------------------------------------------------------------
    both_success_tuples = set()
    divergent_tuples = set()

    for (dataset, model), df_group in full_df.groupby(['Dataset', 'Model_Clean']):
        success_pivot = df_group.pivot_table(index='Instance', columns='Method_Clean', values='Success', aggfunc='first')
        req = ['Actionable CBFI', 'Untargeted Causal']
        if all(c in success_pivot.columns for c in req):
            both_insts = success_pivot.index[(success_pivot[req[0]] == True) & (success_pivot[req[1]] == True)]
            for inst in both_insts:
                both_success_tuples.add((dataset, model, inst))

        action_pivot = df_group.pivot_table(index='Instance', columns='Method_Clean', values='Intervention_Nodes' if 'Intervention_Nodes' in df_group.columns else 'Sparsity (L0)', aggfunc='first')
        if all(c in action_pivot.columns for c in req):
            div_insts = action_pivot.index[(action_pivot.index.isin(both_insts)) & (action_pivot[req[0]] != action_pivot[req[1]])]
            for inst in div_insts:
                divergent_tuples.add((dataset, model, inst))

    full_df['is_both'] = full_df.apply(lambda r: (r['Dataset'], r['Model_Clean'], r['Instance']) in both_success_tuples, axis=1)
    full_df['is_div'] = full_df.apply(lambda r: (r['Dataset'], r['Model_Clean'], r['Instance']) in divergent_tuples, axis=1)

    full_df_both = full_df[full_df['is_both']].copy()
    full_df_div = full_df[full_df['is_div']].copy()

    def get_n_unique(df, model=None):
        sub = df if model is None else df[df['Model_Clean'] == model]
        return len(sub[['Dataset', 'Instance']].drop_duplicates())

    empty_metrics = {
        'Time (s)': "", 'Evaluations': "", 'Success Rate': "", 'Recourse Cost': "", 'RCR': "",
        'L_active': "", 'Sparsity (L0)': "", 'R_SCM': "", 'R_0': "", 'D_M': "",
        'Lactive p-value (vs A-CBFI)': "", 'Cost p-value (vs A-CBFI)': "",
        'RCR p-value (vs A-CBFI)': "", 'R_SCM p-value (vs A-CBFI)': ""
    }

    rows = []

    # Helper function to generate rows for a given dataset subset
    def build_subset_rows(df_subset, subset_title):
        rows.append({'Model Architecture': f"[{subset_title}]", 'Method': '', **empty_metrics})
        
        # Per Model Architecture
        for model in model_order:
            first_model = True
            n_samples = get_n_unique(df_subset, model)
            model_header = f"{model} (N={n_samples})"
            
            sub_model = df_subset[df_subset['Model_Clean'] == model]
            acbfi_sub = sub_model[sub_model['Method_Clean'] == 'Actionable CBFI']

            for method in method_order:
                method_sub = sub_model[sub_model['Method_Clean'] == method]
                m = compute_metrics(acbfi_sub, method_sub, method)
                model_str = model_header if first_model else ""
                rows.append({
                    'Model Architecture': model_str,
                    'Method': method,
                    **m
                })
                first_model = False

        # Overall All Models
        first_overall = True
        n_all_samples = get_n_unique(df_subset)
        overall_header = f"Overall Models (N={n_all_samples})"
        acbfi_all = df_subset[df_subset['Method_Clean'] == 'Actionable CBFI']

        for method in method_order:
            method_sub = df_subset[df_subset['Method_Clean'] == method]
            m = compute_metrics(acbfi_all, method_sub, method)
            model_str = overall_header if first_overall else ""
            rows.append({
                'Model Architecture': model_str,
                'Method': method,
                **m
            })
            first_overall = False

    # Build 3 Subsets: [TOTAL], [BOTH-SUCCESS MEAN COMPARISON], [DIVERGENT PRESCRIPTIONS]
    build_subset_rows(full_df, "TOTAL BENCHMARK BY MODEL ARCHITECTURE")
    build_subset_rows(full_df_both, "BOTH-SUCCESS SUBSET BY MODEL ARCHITECTURE")
    build_subset_rows(full_df_div, "DIVERGENT PRESCRIPTIONS SUBSET BY MODEL ARCHITECTURE")

    df_out = pd.DataFrame(rows)
    print("\n" + "="*160)
    print("      AGGREGATED BENCHMARK BY MODEL ARCHITECTURE & METHOD (A-CBFI & UNTARGETED CAUSAL ONLY)")
    print("="*160)
    print(df_out.to_string(index=False))
    
    target_csv = os.path.join(base_dir, 'aggregate_benchmark_table3.csv')
    safe_to_csv(df_out, target_csv)
    print("\n")


if __name__ == '__main__':
    run_table3_aggregation()
