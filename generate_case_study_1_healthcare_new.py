"""
========================================================================================
Healthcare Case Study 1 Generator: generate_case_study_1_healthcare_new.py
========================================================================================
Generates figures, tables, and comparison materials for Healthcare Domain using the
4-Step Actionable CBFI framework (XAI Storytelling: Diagnosis -> Prescription -> Justification).
- Dataset: Diabetes (Instance #152)
- Outputs are saved into the 'result_case_1' directory.
========================================================================================
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from Actionable_CBFI import StructuralCausalModel, CausalCBFIDiagnoser, ActionableCBFISolver
from Visualize_ACBFI import (
    plot_main_interaction_decomposition,
    plot_base_diagnosis_waterfall,
    plot_interaction_graph,
    plot_causal_recourse_waterfall,
    visualize_causal_recourse_path
)

def format_val_by_feature(feature_name: str, val: float) -> str:
    """Formats raw feature values into human-readable strings with domain units."""
    f_lower = feature_name.lower()
    if 'glucose' in f_lower: return f"{val:.1f} mg/dL"
    elif 'bmi' in f_lower: return f"{val:.1f} kg/m²"
    elif 'insulin' in f_lower: return f"{val:.1f} mu U/ml"
    elif 'bloodpressure' in f_lower or 'bp' in f_lower: return f"{val:.1f} mmHg"
    elif 'skinthickness' in f_lower: return f"{val:.1f} mm"
    elif 'pedigree' in f_lower: return f"{val:.3f}"
    elif 'pregnancies' in f_lower: return f"{int(round(val))} times"
    elif 'age' in f_lower: return f"{int(round(val))} yrs"
    else: return f"{val:.4f}"

def load_diabetes_direct(csv_path: str = 'data/diabetes.csv'):
    """Loads and preprocesses data/diabetes.csv for healthcare SCM."""
    if not os.path.exists(csv_path):
        parent_csv = os.path.join(os.path.dirname(__file__), '..', csv_path)
        if os.path.exists(parent_csv):
            csv_path = parent_csv
        else:
            raise FileNotFoundError(f"File '{csv_path}' not found.")

    df_raw = pd.read_csv(csv_path)
    y = df_raw['Outcome'].astype(int)
    feature_cols = [c for c in df_raw.columns if c != 'Outcome']

    X = df_raw[feature_cols].copy()
    X.columns = [c.strip().lower() for c in X.columns]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    immutable_features = ['age']
    dag_edges = [
        ('age', 'pregnancies'),
        ('age', 'glucose'),
        ('age', 'bmi'),
        ('bmi', 'bloodpressure'),
        ('bmi', 'skinthickness'),
        ('glucose', 'insulin')
    ]
    return X_scaled, X, y, immutable_features, dag_edges

def generate_all_case_study_1_healthcare_materials(adverse_idx: int = 152):
    # Set output directory to exactly 'result_case_1' as requested
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'result_case_1')
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # System Setup
    # -------------------------------------------------------------------------
    X_scaled, X_raw, y, imm, dag = load_diabetes_direct()
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled, y)
    
    scm = StructuralCausalModel(dag_edges=dag, nodes=list(X_scaled.columns))
    scm.fit_from_data(X_scaled, dag)

    inst_scaled = X_scaled.iloc[adverse_idx]
    inst_raw = X_raw.iloc[adverse_idx]
    factual_prob = rf.predict_proba(pd.DataFrame([inst_scaled]))[0, 1]

    # -------------------------------------------------------------------------
    # STEP 0: Instance Details
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print(f"       [STEP 0] SELECTED CASE STUDY INSTANCE (Diabetes Instance #{adverse_idx})")
    print("="*80)
    print(f"Factual Outcome Label : {y.iloc[adverse_idx]} (1 = Diabetic / Adverse Risk)")
    print(f"Factual Diabetic Risk: {factual_prob*100:.1f}%")
    print("-" * 80)
    print(f"{'Feature Name':<20} | {'Raw Original Value':<20} | {'Scaled Z-Score':<18} | {'Type'}")
    print("-" * 80)
    for col in X_scaled.columns:
        raw_val = inst_raw[col]
        scaled_val = inst_scaled[col]
        is_imm = "Immutable" if col in imm else "Actionable"
        raw_str = format_val_by_feature(col, raw_val)
        print(f"{col:<20} | {raw_str:<20} | {scaled_val:<+18.4f} | {is_imm}")
    print("="*80 + "\n")

    # -------------------------------------------------------------------------
    # STEP 1: Structural Causal Model Construction
    # -------------------------------------------------------------------------
    print("="*80)
    print(" [STEP 1] Structural Causal Model Construction")
    print("="*80)
    print(f" - Immutable Features : {imm}")
    print(f" - SCM DAG Edges      : {dag}")
    print(" - Domain constraints (physiological laws) logically preserved.")
    print("="*80 + "\n")

    # -------------------------------------------------------------------------
    # STEP 2: Structural Causal Diagnosis (C-G1 & C-G4)
    # -------------------------------------------------------------------------
    print("="*80)
    print(" [STEP 2] Structural Causal Diagnosis via Causal CBFI")
    print("="*80)
    
    diagnoser = CausalCBFIDiagnoser(scm)
    
    df_diag, target_nodes = diagnoser.diagnose_instance(
        model=rf, instance=inst_scaled, background_data=X_scaled, 
        job_type='classification', immutable_features=imm
    )
    
    pairwise_matrix = diagnoser.diagnose_pairwise_interaction(
        model=rf, instance=inst_scaled, background_data=X_scaled, 
        target_nodes=target_nodes, job_type='classification'
    )

    print(" -> Generating Decomposition Plot...")
    plot_main_interaction_decomposition(
        df_diag=df_diag, 
        prediction_label="Severe Diabetic Risk (Class 1)", 
        save_path=os.path.join(output_dir, f'step2_healthcare_cbfi_decomposition_{adverse_idx}.png')
    )

    print(" -> Generating Base Diagnosis Waterfall Plot...")
    plot_base_diagnosis_waterfall(
        df_diag=df_diag, 
        instance=inst_scaled, 
        prediction_label="Diabetic Risk",
        immutable_features=imm,
        save_path=os.path.join(output_dir, f'step2_healthcare_base_waterfall_{adverse_idx}.png')
    )

    print(" -> Generating Interaction Graph (G_I) Plot...")
    plot_interaction_graph(
        df_diag=df_diag, 
        interaction_matrix=pairwise_matrix, 
        save_path=os.path.join(output_dir, f'step2_healthcare_interaction_graph_{adverse_idx}.png')
    )

    print(f"\n[Targeting Filter Result]")
    print(f" -> Dynamically Filtered Actionable Bottlenecks: {target_nodes}")
    print("="*80 + "\n")

    # -------------------------------------------------------------------------
    # STEP 3: Targeted Recourse Optimization (IGBS & MAD)
    # -------------------------------------------------------------------------
    print("="*80)
    print(" [STEP 3] Targeted Recourse Optimization via IGBS")
    print("="*80)
    
    solver = ActionableCBFISolver(scm)
    print(" -> Running Interaction-Guided Beam Search (IGBS)...")
    res_acbfi = solver.find_recourse(
        model=rf, instance=inst_scaled, target_nodes=target_nodes, y_target=0,
        background_data=X_scaled, immutable_features=imm, job_type='classification',
        max_iter=500, df_diag=df_diag, pairwise_matrix=pairwise_matrix, lambda_mse=0.0
    )

    if res_acbfi['success']:
        action_keys = list(res_acbfi['best_action'].keys()) if res_acbfi['best_action'] else []
        
        print(" -> Generating Causal Propagation DAG (Mode A)...")
        visualize_causal_recourse_path(
            scm=scm, 
            factual=inst_scaled, 
            counterfactual=res_acbfi['x_counterfactual'], 
            action_nodes=action_keys,
            save_path=os.path.join(output_dir, f'step3_healthcare_causal_propagation_dag_{adverse_idx}.png')
        )

        print(" -> Generating Causal Recourse Waterfall (Mode B)...")
        plot_causal_recourse_waterfall(
            model=rf, scm=scm, instance=inst_scaled, target_nodes=action_keys,
            background_data=X_scaled,
            counterfactual=res_acbfi['x_counterfactual'],
            best_action=res_acbfi['best_action'],
            prediction_label="Severe Diabetic Risk",
            target_label="Normal",
            save_path=os.path.join(output_dir, f'step3_healthcare_causal_recourse_waterfall_{adverse_idx}.png')
        )
    else:
        print(" -> [Warning] Recourse failed. Skipping Step 3 visualizations.")
    print("="*80 + "\n")

    # -------------------------------------------------------------------------
    # STEP 4: Actionable Recourse Generation and Verification
    # -------------------------------------------------------------------------
    print("="*90)
    print(" [STEP 4] Actionable Recourse Generation (Verification vs Baselines)")
    print("="*90)

    # Untargeted Causal
    all_mutable = [c for c in X_scaled.columns if c not in imm]
    res_untargeted = solver.find_recourse_untargeted(
        model=rf, instance=inst_scaled, y_target=0, background_data=X_scaled, 
        immutable_features=imm, max_iter=300, lambda_mse=0.0
    )
    res_untargeted['method'] = "Untargeted Causal Recourse (Karimi et al., 2021)"

    # Baselines
    class WachterSolver:
        def find_recourse(self, model, instance, y_target, background_data, immutable_features, max_iter=300):
            start_t = time.time()
            mutable_cols = [c for c in instance.index if c not in immutable_features]
            mad_dict = {col: max(1e-6, np.median(np.abs(background_data[col] - np.median(background_data[col])))) for col in background_data.columns}
            best_cost, best_cf, best_act = float('inf'), None, None
            np.random.seed(42)
            for _ in range(max_iter):
                cf_sample = instance.copy()
                act = {}
                for col in mutable_cols:
                    val = float(np.random.choice(background_data[col]))
                    cf_sample[col] = val
                    act[col] = val
                
                if model.predict(pd.DataFrame([cf_sample]))[0] == y_target:
                    cost = sum(abs(cf_sample[c] - instance[c]) / mad_dict.get(c, 1.0) for c in instance.index)
                    if cost < best_cost:
                        best_cost, best_cf, best_act = cost, cf_sample, act
            return {
                'method': "Wachter's Counterfactuals (Wachter et al., 2017)",
                'x_counterfactual': best_cf, 'best_action': best_act,
                'minimum_cost': best_cost if best_cf is not None else float('nan'),
                'elapsed_time': time.time() - start_t, 'success': best_cf is not None
            }

    class SHAPTargetedSolver:
        def find_recourse(self, model, instance, y_target, background_data, immutable_features, max_iter=300):
            importances = pd.Series(model.feature_importances_, index=instance.index) if hasattr(model, 'feature_importances_') else pd.Series(1.0, index=instance.index)
            mutable_cols = [c for c in instance.index if c not in immutable_features]
            top_3 = [col for col in importances.sort_values(ascending=False).index if col in mutable_cols][:3]
            wachter = WachterSolver()
            res = wachter.find_recourse(model, instance, y_target, background_data, immutable_features=[c for c in instance.index if c not in top_3], max_iter=max_iter)
            res['method'] = "SHAP-Targeted Recourse"
            return res

    res_wachter = WachterSolver().find_recourse(rf, inst_scaled, y_target=0, background_data=X_scaled, immutable_features=imm)
    res_shap = SHAPTargetedSolver().find_recourse(rf, inst_scaled, y_target=0, background_data=X_scaled, immutable_features=imm)
    res_acbfi['method'] = "Actionable CBFI (Proposed Method)"

    results_list = [res_acbfi, res_untargeted, res_wachter, res_shap]
    method_summary = {}

    for idx_m, res in enumerate(results_list, start=1):
        m_name = res['method']
        print(f"\n[{idx_m}] METHODOLOGY: {m_name}")
        print("-" * 90)
        print(f"  - Recourse Status       : {'SUCCESS (Normal / Non-Diabetic)' if res['success'] else 'FAILED'}")
        print(f"  - Minimum Cost (MAD)    : {res['minimum_cost']:.4f}" if not np.isnan(res['minimum_cost']) else "  - Minimum Recourse Cost : N/A")
        print(f"  - Computation Time      : {res['elapsed_time']*1000:.2f} ms")
        
        changes_dict = {}
        direct_cnt = 0
        passive_cnt = 0

        if res['success'] and res['x_counterfactual'] is not None:
            p_post = rf.predict_proba(pd.DataFrame([res['x_counterfactual']]))[0, 1]
            print(f"  - Post-Intervention Risk: {p_post*100:.1f}% (Adverse Class Risk)")
            
            print("  - Action Details (Feature Changes):")
            print("      " + "-"*90)
            print(f"      {'Feature Name':<16} | {'Original Value':<16} | {'Recommended Value':<16} | {'Intervention Type'}")
            print("      " + "-"*90)
            
            best_action_keys = list(res['best_action'].keys()) if res['best_action'] else []

            for col in X_scaled.columns:
                val_orig_scaled = inst_scaled[col]
                val_cf_scaled = res['x_counterfactual'][col]
                diff_scaled = abs(val_cf_scaled - val_orig_scaled)
                
                if diff_scaled > 1e-4:
                    col_mean, col_std = float(X_raw[col].mean()), float(X_raw[col].std())
                    col_std = col_std if col_std > 0 else 1.0
                    
                    val_orig_raw = inst_raw[col]
                    val_cf_raw = col_mean + (val_cf_scaled * col_std)

                    orig_str = format_val_by_feature(col, val_orig_raw)
                    rec_str = format_val_by_feature(col, val_cf_raw)

                    # Determine action type for explanation
                    action_type = "Passive SCM Propagation"
                    if col in best_action_keys:
                        direct_cnt += 1
                        action_type = "Direct User Lever (L_active)"
                    elif not res.get('best_action'):
                        direct_cnt += 1
                        action_type = "Direct User Lever (L_active)"
                    else:
                        passive_cnt += 1

                    changes_dict[col] = (rec_str, action_type)
                    print(f"      {col:<16} | {orig_str:<16} | {rec_str:<16} | {action_type}")
            print("      " + "-"*90)
            
            print(f"  - Direct User Action Levers (L_active): {direct_cnt} features actively altered by user")
            print(f"  - Passive SCM Propagations             : {passive_cnt} downstream features auto-adjusted")
        else:
            p_post = factual_prob
            print("  - Action Details: No valid recourse found within budget.")

        method_summary[m_name] = {
            'success': res['success'],
            'post_risk': f"{p_post*100:.1f}%",
            'direct_cnt': direct_cnt,
            'passive_cnt': passive_cnt,
            'changes': changes_dict
        }

    # Generate CSV Summary Table
    m_names = [res['method'] for res in results_list]
    summary_rows = []
    
    for col in X_scaled.columns:
        row_dict = {'Feature': col, 'Factual Value': format_val_by_feature(col, inst_raw[col])}
        for m_name in m_names:
            info = method_summary[m_name]
            if col in info['changes']:
                val_str, type_str = info['changes'][col]
                row_dict[m_name] = f"{val_str} ({type_str})"
            else:
                row_dict[m_name] = "Unchanged"
        summary_rows.append(row_dict)

    summary_rows.append({
        'Feature': 'Active Levers (L_active)', 'Factual Value': 'Baseline',
        m_names[0]: f"{method_summary[m_names[0]]['direct_cnt']} (Targeted)",
        m_names[1]: f"{method_summary[m_names[1]]['direct_cnt']} (Untargeted)",
        m_names[2]: f"{method_summary[m_names[2]]['direct_cnt']}",
        m_names[3]: f"{method_summary[m_names[3]]['direct_cnt']}"
    })
    
    summary_rows.append({
        'Feature': 'Post-Intervention Risk', 'Factual Value': f"{factual_prob*100:.1f}% (Diabetic)",
        m_names[0]: method_summary[m_names[0]]['post_risk'],
        m_names[1]: method_summary[m_names[1]]['post_risk'],
        m_names[2]: method_summary[m_names[2]]['post_risk'],
        m_names[3]: method_summary[m_names[3]]['post_risk']
    })

    df_summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 130)
    print(f"           HEALTHCARE CASE STUDY 1 COMPARISON TABLE (DIABETES INSTANCE #{adverse_idx})")
    print("=" * 130)
    print(df_summary.to_string(index=False))

    csv_path = os.path.join(output_dir, f'case1_healthcare_summary_table_{adverse_idx}.csv')
    df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] Healthcare Case Study 1 CSV table saved to '{csv_path}' in 'result_case_1' folder.\n")


if __name__ == '__main__':
    generate_all_case_study_1_healthcare_materials()