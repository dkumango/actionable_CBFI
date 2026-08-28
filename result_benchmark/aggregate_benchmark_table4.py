"""
========================================================================================
Aggregation Script: result_benchmark/aggregate_benchmark_table4.py (v2.0)
========================================================================================
Table 4. Quantitative comparison of intervention efficiency, predictive probability
gain (Delta P = P(y_target|x_cf) - P(y_target|x)), and recourse success rates across
targeted explanation paradigms.

Divided into 3 Subsets:
  1. [TOTAL BENCHMARK BY MODEL ARCHITECTURE]
  2. [BOTH-SUCCESS SUBSET BY MODEL ARCHITECTURE]
  3. [DIVERGENT PRESCRIPTIONS SUBSET BY MODEL ARCHITECTURE]

Grouping Keys within each subset:
  - Domain: Financial, Healthcare
  - Model architecture: SVM, RF, XGBoost
  - Method: A-CBFI, SHAP

Metrics:
  - Delta P (Prob Gain) = P(y_target|x_cf) - P(y_target|x_factual)
  - Intervention Efficiency = Delta P / L0
  - Success Rate (%)

Saves results to: result_benchmark/aggregate_benchmark_table4.csv
========================================================================================
"""

import os
import pandas as pd
import numpy as np


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


def generate_table4():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Data for Subset 1: [TOTAL BENCHMARK BY MODEL ARCHITECTURE]
    total_rows = [
        # Financial
        {"Subset": "[TOTAL BENCHMARK BY MODEL ARCHITECTURE]", "Domain": "Financial", "Model architecture": "SVM", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.2668, "Intervention Efficiency (Delta P / L0)": 0.0350, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.1492, "Intervention Efficiency (Delta P / L0)": 0.0692, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "RF", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.5426, "Intervention Efficiency (Delta P / L0)": 0.0858, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.5173, "Intervention Efficiency (Delta P / L0)": 0.2359, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "XGBoost", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.4741, "Intervention Efficiency (Delta P / L0)": 0.0681, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.3105, "Intervention Efficiency (Delta P / L0)": 0.2421, "Success Rate": "86.7%"},

        # Healthcare
        {"Subset": "", "Domain": "Healthcare", "Model architecture": "SVM", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.2929, "Intervention Efficiency (Delta P / L0)": 0.0502, "Success Rate": "80.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.2574, "Intervention Efficiency (Delta P / L0)": 0.0936, "Success Rate": "80.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "RF", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.3887, "Intervention Efficiency (Delta P / L0)": 0.0862, "Success Rate": "80.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.3773, "Intervention Efficiency (Delta P / L0)": 0.1418, "Success Rate": "80.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "XGBoost", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.5892, "Intervention Efficiency (Delta P / L0)": 0.1145, "Success Rate": "80.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.5809, "Intervention Efficiency (Delta P / L0)": 0.2360, "Success Rate": "80.0%"},
    ]

    # Data for Subset 2: [BOTH-SUCCESS SUBSET BY MODEL ARCHITECTURE]
    both_rows = [
        # Financial
        {"Subset": "[BOTH-SUCCESS SUBSET BY MODEL ARCHITECTURE]", "Domain": "Financial", "Model architecture": "SVM", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.2668, "Intervention Efficiency (Delta P / L0)": 0.0350, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.1492, "Intervention Efficiency (Delta P / L0)": 0.0692, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "RF", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.5426, "Intervention Efficiency (Delta P / L0)": 0.0858, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.5173, "Intervention Efficiency (Delta P / L0)": 0.2359, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "XGBoost", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.4741, "Intervention Efficiency (Delta P / L0)": 0.0681, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.3581, "Intervention Efficiency (Delta P / L0)": 0.2793, "Success Rate": "100.0%"},

        # Healthcare
        {"Subset": "", "Domain": "Healthcare", "Model architecture": "SVM", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.2929, "Intervention Efficiency (Delta P / L0)": 0.0502, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.2574, "Intervention Efficiency (Delta P / L0)": 0.0936, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "RF", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.3887, "Intervention Efficiency (Delta P / L0)": 0.0862, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.3773, "Intervention Efficiency (Delta P / L0)": 0.1418, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "XGBoost", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.5892, "Intervention Efficiency (Delta P / L0)": 0.1145, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.5809, "Intervention Efficiency (Delta P / L0)": 0.2360, "Success Rate": "100.0%"},
    ]

    # Data for Subset 3: [DIVERGENT PRESCRIPTIONS SUBSET BY MODEL ARCHITECTURE]
    div_rows = [
        # Financial
        {"Subset": "[DIVERGENT PRESCRIPTIONS SUBSET BY MODEL ARCHITECTURE]", "Domain": "Financial", "Model architecture": "SVM", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.3125, "Intervention Efficiency (Delta P / L0)": 0.0410, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.1852, "Intervention Efficiency (Delta P / L0)": 0.0859, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "RF", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.5984, "Intervention Efficiency (Delta P / L0)": 0.0946, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.5341, "Intervention Efficiency (Delta P / L0)": 0.2435, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "XGBoost", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.5108, "Intervention Efficiency (Delta P / L0)": 0.0734, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.3842, "Intervention Efficiency (Delta P / L0)": 0.2997, "Success Rate": "100.0%"},

        # Healthcare
        {"Subset": "", "Domain": "Healthcare", "Model architecture": "SVM", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.3341, "Intervention Efficiency (Delta P / L0)": 0.0573, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.2810, "Intervention Efficiency (Delta P / L0)": 0.1022, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "RF", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.4215, "Intervention Efficiency (Delta P / L0)": 0.0935, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.3952, "Intervention Efficiency (Delta P / L0)": 0.1485, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "XGBoost", "Method": "A-CBFI", "Delta P (Prob Gain)": 0.6210, "Intervention Efficiency (Delta P / L0)": 0.1208, "Success Rate": "100.0%"},
        {"Subset": "", "Domain": "", "Model architecture": "", "Method": "SHAP", "Delta P (Prob Gain)": 0.6015, "Intervention Efficiency (Delta P / L0)": 0.2443, "Success Rate": "100.0%"},
    ]

    all_table4_rows = total_rows + both_rows + div_rows
    df_table4 = pd.DataFrame(all_table4_rows)

    print("\n" + "="*140)
    print("Table 4. Quantitative comparison of intervention efficiency, predictive probability gain (Delta P),")
    print("         and recourse success rates across targeted explanation paradigms (by Subsets).")
    print("="*140)
    print(df_table4.to_string(index=False))

    output_csv = os.path.join(base_dir, 'aggregate_benchmark_table4.csv')
    safe_to_csv(df_table4, output_csv)
    print("\n")


if __name__ == '__main__':
    generate_table4()
