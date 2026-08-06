"""
========================================================================================
Benchmark Comparison Extension Suite: test_benchmark_comparison_extension.py
========================================================================================
Extends the original benchmark suite to support academic publication requirements:
1. Multi-Architecture Validation: RandomForest, XGBoost (XGBClassifier), and SVM (SVC).
2. Domain Diversification: Supports Financial Loan, German Credit, and Medical Insurance.
3. Statistical Significance Testing: Applies the Wilcoxon signed-rank test (p-values) 
   to validate that A-CBFI's improvements over baselines across N=100 instances are 
   statistically significant and non-accidental.

Baseline Methodologies Evaluated:
- Actionable CBFI (Proposed): SCM + Causal CBFI Diagnosis -> Targeted Search
- Untargeted Causal Recourse (Karimi et al., 2021): SCM -> All Mutable Features Search
- Wachter's Counterfactuals (Wachter et al., 2017): Non-causal Distance Optimization
- SHAP-Targeted Recourse: SHAP Top-K Feature Selection -> Non-causal Search
========================================================================================
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Try importing XGBoost for multi-architecture validation
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[Warning] XGBoost not installed. Falling back to RandomForest and SVC only.")

from Actionable_CBFI import (
    StructuralCausalModel,
    ActionableCBFISolver,
    CausalCBFIDiagnoser,
    compute_search_efficiency,
    compute_sparsity,
    compute_causal_plausibility
)

# =========================================================================================
# 1. Baseline Implementations (Wachter's CE & SHAP-Targeted CE)
# =========================================================================================

class WachterCounterfactual:
    """
    Baseline 1: Wachter's Counterfactual Explanations (Wachter et al., 2017)
    Non-causal optimization solving min_x Cost(x, x_cf) s.t. M(x_cf) == y_target
    """
    def find_recourse(
        self, 
        model, 
        instance: pd.Series, 
        y_target: int, 
        background_data: pd.DataFrame, 
        immutable_features: list, 
        max_iter: int = 500
    ) -> dict:
        start_time = time.time()
        mutable_cols = [c for c in instance.index if c not in immutable_features]

        mad_dict = {}
        for col in background_data.columns:
            mad_val = np.median(np.abs(background_data[col] - np.median(background_data[col])))
            mad_dict[col] = mad_val if mad_val > 1e-6 else 1.0

        best_cost = float('inf')
        best_x_cf = None
        best_action = None
        evaluations = 0

        n_samples_per_feat = max(5, int(np.power(max_iter, 1.0 / max(1, len(mutable_cols)))))
        grids = {}
        for col in mutable_cols:
            grids[col] = np.linspace(background_data[col].min(), background_data[col].max(), n_samples_per_feat)

        keys = list(grids.keys())
        values = list(grids.values())

        total_combos = 1
        for v in values:
            total_combos *= len(v)

        np.random.seed(42)
        if total_combos > max_iter:
            combos_to_eval = [tuple(float(np.random.choice(grids[k])) for k in keys) for _ in range(max_iter)]
        else:
            from itertools import product
            combos_to_eval = list(product(*values))

        cf_list = []
        actions_list = []

        for combo in combos_to_eval:
            evaluations += 1
            x_cf = instance.copy().astype(float)
            action = {}
            for k, v in zip(keys, combo):
                x_cf[k] = float(v)
                action[k] = float(v)
            cf_list.append(x_cf)
            actions_list.append(action)

        if cf_list:
            df_cf_batch = pd.DataFrame(cf_list)
            preds_batch = model.predict(df_cf_batch)

            for i in range(len(preds_batch)):
                pred_y = preds_batch[i]
                if pred_y == y_target:
                    x_cf = cf_list[i]
                    action = actions_list[i]
                    cost = sum(abs(x_cf[c] - instance[c]) / mad_dict.get(c, 1.0) for c in instance.index)
                    if cost < best_cost:
                        best_cost = cost
                        best_x_cf = x_cf
                        best_action = action

        elapsed_time = time.time() - start_time
        return {
            'method': "Wachter's CE",
            'x_counterfactual': best_x_cf,
            'best_action': best_action,
            'minimum_cost': best_cost if best_x_cf is not None else float('nan'),
            'evaluations': evaluations,
            'elapsed_time': elapsed_time,
            'success': best_x_cf is not None
        }


class SHAPTargetedRecourse:
    """
    Baseline 2: SHAP-Targeted Recourse
    Uses additive importance to pick Top-K features, then performs non-causal search.
    """
    def find_recourse(
        self, 
        model, 
        instance: pd.Series, 
        y_target: int, 
        background_data: pd.DataFrame, 
        immutable_features: list, 
        top_k: int = 3,
        max_iter: int = 500
    ) -> dict:
        mutable_cols = [c for c in instance.index if c not in immutable_features]

        # Extract feature importances (or fallback to uniform coefficients for SVM)
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=instance.index)
        elif hasattr(model, 'coef_'):
            importances = pd.Series(np.abs(model.coef_[0]), index=instance.index)
        else:
            importances = pd.Series(1.0, index=instance.index)

        top_mutable = [col for col in importances.sort_values(ascending=False).index if col in mutable_cols][:top_k]

        wachter = WachterCounterfactual()
        res = wachter.find_recourse(
            model, instance, y_target, background_data, 
            immutable_features=[c for c in instance.index if c not in top_mutable], 
            max_iter=max_iter
        )
        res['method'] = "SHAP-Targeted CE"
        return res


# =========================================================================================
# 2. Domain-Diversified Dataset Loaders
# =========================================================================================



def normalize_feature_name(name: str) -> str:
    """
    Normalizes feature names by handling lowercasing, whitespace, word-order variations (e.g. radius_mean vs mean_radius),
    and common dataset column synonyms.
    """
    s = name.lower().strip().replace(' ', '_')
    synonyms = {
        'credit_amount': 'amount',
        'duration': 'months_loan_duration',
        'present_residence': 'residence_history',
    }
    s = synonyms.get(s, s)
    parts = set(s.split('_'))
    return "_".join(sorted(parts))


def load_domain_scm_configs(config_path: str = 'DOMAIN_SCM_CONFIG.txt') -> dict:
    """
    Loads domain-specific SCM configurations (immutable_features and dag_edges) from DOMAIN_SCM_CONFIG.txt.
    """
    if not os.path.exists(config_path):
        print(f"[Warning] SCM config file '{config_path}' not found.")
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    scope = {}
    try:
        exec(content, scope)
        return scope.get('DOMAIN_SCM_CONFIGS', {})
    except Exception as e:
        print(f"[Warning] Failed to parse SCM config file '{config_path}': {e}")
        return {}


def load_dataset_by_name(dataset_name: str = 'financial_loan', config_path: str = 'DOMAIN_SCM_CONFIG.txt'):
    """
    Loads and preprocesses diverse benchmark datasets for multi-domain validation.
    Reads DAG edges and immutable features from DOMAIN_SCM_CONFIG.txt.
    Returns: X_scaled (DataFrame), y (Series), immutable_features (list), dag_edges (list)
    """
    encoder = OrdinalEncoder()
    scaler = StandardScaler()

    if dataset_name == 'financial_loan':
        csv_path = 'data/financial_loan.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")
        
        df = pd.read_csv(csv_path, encoding='latin1')
        df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
        y = (df['loan_status'] == 'Charged Off').astype(int)
        
        feature_cols = ['annual_income', 'dti', 'int_rate', 'loan_amount', 'installment', 'term', 'emp_length', 'home_ownership', 'total_acc']
        actual_cols = list(dict.fromkeys([c for c in feature_cols if c in df.columns]))
        X = df[actual_cols].copy()
        
        cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
        if cat_cols: X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
        X.columns = [c.strip().lower() for c in X.columns]
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    elif dataset_name == 'adult' or dataset_name == 'adult_income':
        csv_path = 'data/adult.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")
        df = pd.read_csv(csv_path)
        
        y = (df['income'].astype(str).str.contains('>50K')).astype(int)
        feature_cols = ['age', 'workclass', 'education_num', 'marital_status', 'occupation', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week']
        actual_cols = [c for c in feature_cols if c in df.columns]
        X = df[actual_cols].copy()
        
        cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
        if cat_cols: X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
        X.columns = [c.strip().lower() for c in X.columns]
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    elif dataset_name == 'german_credit':
        csv_path = 'data/german_credit.csv'
        if not os.path.exists(csv_path):
            csv_path = 'data/loan_approval.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"German Credit dataset file not found.")
        df = pd.read_csv(csv_path)
        
        # Target default: 2 -> 1 (Bad Credit Risk), 1 -> 0 (Good Credit)
        y = (df['default'] == 2).astype(int)
        X = df.drop(columns=['default'], errors='ignore').copy()
        
        cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
        if cat_cols: X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
        X.columns = [c.strip().lower() for c in X.columns]
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    elif dataset_name == 'medical_insurance':
        csv_path = 'data/insurance.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")
        df = pd.read_csv(csv_path)
        
        y = (df['charges'] > df['charges'].median()).astype(int)
        X = df.drop(columns=['charges'], errors='ignore').copy()
        
        cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
        if cat_cols: X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
        X.columns = [c.strip().lower() for c in X.columns]
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    elif dataset_name == 'diabetes':
        csv_path = 'data/diabetes.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")
        df = pd.read_csv(csv_path)
        
        y = df['Outcome'].astype(int)
        X = df.drop(columns=['Outcome'], errors='ignore').copy()
        
        cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
        if cat_cols: X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
        X.columns = [c.strip().lower() for c in X.columns]
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    elif dataset_name == 'breast_cancer':
        csv_path = 'data/breast_cancer.csv'
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' not found.")
        df = pd.read_csv(csv_path)
        
        # Target 0: Malignant -> High risk (1), Target 1: Benign -> Normal (0)
        y = (df['target'] == 0).astype(int)
        # Select first 10 mean feature attributes for clean modeling
        mean_cols = [c for c in df.columns if 'mean' in c][:10]
        X = df[mean_cols].copy() if mean_cols else df.drop(columns=['target']).iloc[:, :10].copy()
        
        cat_cols = X.select_dtypes(include=['object', 'str']).columns.tolist()
        if cat_cols: X[cat_cols] = encoder.fit_transform(X[cat_cols].astype(str))
        X.columns = [c.strip().lower().replace(' ', '_') for c in X.columns]
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: ['financial_loan', 'adult', 'german_credit', 'medical_insurance', 'diabetes', 'breast_cancer']")

    # -------------------------------------------------------------------------
    # Apply DOMAIN_SCM_CONFIG.txt for DAG Edges & Immutable Features
    # -------------------------------------------------------------------------
    domain_configs = load_domain_scm_configs(config_path)
    key_mapping = {
        'financial_loan': 'financial_loan',
        'german_credit': 'german_credit',
        'adult': 'adult_income',
        'adult_income': 'adult_income',
        'medical_insurance': 'medical_insurance',
        'diabetes': 'diabetes',
        'breast_cancer': 'breast_cancer'
    }
    config_key = key_mapping.get(dataset_name, dataset_name)
    dataset_config = domain_configs.get(config_key, {})

    raw_immutable = dataset_config.get('immutable_features', [])
    raw_dag_edges = dataset_config.get('dag_edges', [])

    # Map features using normalized names
    col_norm_map = {normalize_feature_name(c): c for c in X_scaled.columns}

    immutable_feats = []
    for f in raw_immutable:
        norm_f = normalize_feature_name(f)
        if norm_f in col_norm_map:
            immutable_feats.append(col_norm_map[norm_f])

    dag_edges = []
    for src, dst in raw_dag_edges:
        norm_src = normalize_feature_name(src)
        norm_dst = normalize_feature_name(dst)
        if norm_src in col_norm_map and norm_dst in col_norm_map:
            dag_edges.append((col_norm_map[norm_src], col_norm_map[norm_dst]))

    if not dag_edges:
        print(f"[Warning] No valid matching DAG edges found for '{dataset_name}' in SCM Config. Falling back to auto causal discovery (dag_edges=None).")
        dag_edges = None
    else:
        print(f"[Config Loaded] Loaded {len(dag_edges)} DAG edges and {len(immutable_feats)} immutable features from '{config_path}' for dataset '{dataset_name}'.")

    return X_scaled, y, immutable_feats, dag_edges




# =========================================================================================
# 3. Statistical Significance Evaluator
# =========================================================================================

def evaluate_statistical_significance(df_bm: pd.DataFrame, prop_method: str = "Actionable CBFI (Proposed)"):
    """
    Performs Wilcoxon signed-rank test comparing A-CBFI against baselines across N=100 instances.
    Returns formatted p-value strings with significance stars (* p<0.05, ** p<0.01, *** p<0.001).
    """
    methods = [m for m in df_bm['Method'].unique() if m != prop_method]
    metrics = ['Evaluations', 'Time (s)', 'Sparsity (L0)', 'Recourse Cost']
    
    stats_summary = []
    
    df_prop = df_bm[df_bm['Method'] == prop_method].sort_values('Instance')
    
    for method in methods:
        df_base = df_bm[df_bm['Method'] == method].sort_values('Instance')
        row_dict = {'Method': method}
        
        for metric in metrics:
            val_prop = df_prop[metric].values
            val_base = df_base[metric].values
            
            # Remove NaNs for fair paired test
            valid_mask = ~np.isnan(val_prop) & ~np.isnan(val_base)
            if np.sum(valid_mask) > 5:
                try:
                    stat, p_val = wilcoxon(val_prop[valid_mask], val_base[valid_mask], zero_method='pratt')
                except Exception:
                    p_val = 1.0
                
                # Assign significance stars
                if p_val < 0.001: stars = "***"
                elif p_val < 0.01: stars = "**"
                elif p_val < 0.05: stars = "*"
                else: stars = "ns"
                
                row_dict[f"{metric} (p-val)"] = f"{p_val:.2e} ({stars})"
            else:
                row_dict[f"{metric} (p-val)"] = "N/A"
                
        stats_summary.append(row_dict)
        
    return pd.DataFrame(stats_summary)


# =========================================================================================
# 4. Multi-Architecture & Multi-Domain Benchmark Orchestrator
# =========================================================================================

def run_extended_benchmark(dataset_name: str = 'financial_loan', n_test_instances: int = 100):
    print(f"\n=============================================================================")
    print(f"STARTING EXTENDED BENCHMARK: Dataset = '{dataset_name}' | N = {n_test_instances}")
    print(f"=============================================================================")

    X, y, immutable_feats, dag_edges = load_dataset_by_name(dataset_name)
    print(f"Dataset Shape: {X.shape} | Immutable Features: {immutable_feats}")

    # 1. Initialize Multi-Architecture Models
    models_dict = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42)
    }
    if HAS_XGBOOST:
        models_dict['XGBoost'] = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)

    # 2. Build and Fit SCM
    scm = StructuralCausalModel(dag_edges=dag_edges, nodes=list(X.columns))
    scm.fit_from_data(X, dag_edges)

    # Select target instances requiring recourse (Adverse label = 1 -> Desired target = 0)
    adverse_indices = np.where(y.to_numpy() == 1)[0][:n_test_instances]

    solver_acbfi = ActionableCBFISolver(scm)
    diagnoser = CausalCBFIDiagnoser(scm)
    solver_wachter = WachterCounterfactual()
    solver_shap = SHAPTargetedRecourse()

    all_records = []

    for model_name, model in models_dict.items():
        print(f"\n---> Training Model Architecture: [{model_name}]...")
        model.fit(X, y)

        for idx_num, inst_idx in enumerate(adverse_indices):
            instance = X.iloc[inst_idx]
            if (idx_num + 1) % 20 == 0 or idx_num == 0:
                print(f"     Evaluating Instance {idx_num + 1}/{len(adverse_indices)}...")

            # --- Method 1: Actionable CBFI (Proposed) ---
            df_diag, target_nodes = diagnoser.diagnose_instance(
                model=model, instance=instance, background_data=X, job_type='classification', immutable_features=immutable_feats
            )
            res_acbfi = solver_acbfi.find_recourse(
                model=model, instance=instance, target_nodes=target_nodes, y_target=0, background_data=X, immutable_features=immutable_feats
            )
            res_acbfi['method'] = "Actionable CBFI (Proposed)"

            # --- Method 2: Untargeted Causal Recourse (Karimi et al., 2021) ---
            res_causal_untargeted = solver_acbfi.find_recourse_untargeted(
                model=model, instance=instance, y_target=0, background_data=X, immutable_features=immutable_feats
            )
            res_causal_untargeted['method'] = "Untargeted Causal (Karimi 2021)"

            # --- Method 3: Wachter's CE (Wachter et al., 2017) ---
            res_wachter = solver_wachter.find_recourse(
                model=model, instance=instance, y_target=0, background_data=X, immutable_features=immutable_feats
            )

            # --- Method 4: SHAP-Targeted CE ---
            res_shap = solver_shap.find_recourse(
                model=model, instance=instance, y_target=0, background_data=X, immutable_features=immutable_feats
            )

            for m_res in [res_acbfi, res_causal_untargeted, res_wachter, res_shap]:
                x_cf = m_res.get('x_counterfactual')
                if x_cf is not None:
                    l0_cnt = compute_sparsity(instance, x_cf)['L0 Modified Count']
                    plausibility = compute_causal_plausibility(x_cf, scm)
                    cost = m_res.get('minimum_cost', float('nan'))
                else:
                    l0_cnt, plausibility, cost = float('nan'), float('nan'), float('nan')

                all_records.append({
                    'Dataset': dataset_name,
                    'Model_Architecture': model_name,
                    'Instance': idx_num + 1,
                    'Method': m_res['method'],
                    'Evaluations': m_res.get('evaluations', 0),
                    'Time (s)': m_res.get('elapsed_time', 0.0),
                    'Recourse Cost': cost,
                    'Sparsity (L0)': l0_cnt,
                    'Causal Plausibility MSE': plausibility,
                    'Success': m_res.get('success', False)
                })

    df_results = pd.DataFrame(all_records)
    output_csv = f'extended_benchmark_{dataset_name}.csv'
    df_results.to_csv(output_csv, index=False)
    print(f"\n[Completed] Full benchmark logs saved to '{output_csv}'.")

    # =========================================================================================
    # 5. Display Summaries & Statistical Significance Reports
    # =========================================================================================
    for model_name in models_dict.keys():
        print(f"\n=============================================================================")
        print(f" SUMMARY & STATISTICAL SIGNIFICANCE: [{model_name}] (N={len(adverse_indices)})")
        print(f"=============================================================================")
        
        df_sub = df_results[df_results['Model_Architecture'] == model_name]
        
        # 1. Mean performance summary
        summary_df = df_sub.groupby('Method').agg({
            'Evaluations': 'mean',
            'Time (s)': 'mean',
            'Recourse Cost': 'mean',
            'Sparsity (L0)': 'mean',
            'Causal Plausibility MSE': 'mean',
            'Success': 'mean'
        }).reset_index()
        print("\n[Mean Benchmark Performance Table]:")
        print(summary_df.to_string(index=False))

        # 2. Statistical Significance Testing (Wilcoxon Signed-Rank Test)
        print("\n[Wilcoxon Signed-Rank Test vs. Actionable CBFI (p-values & significance)]: ")
        print(" (* p<0.05, ** p<0.01, *** p<0.001, ns = not significant)")
        df_stats = evaluate_statistical_significance(df_sub, prop_method="Actionable CBFI (Proposed)")
        print(df_stats.to_string(index=False))

    # =========================================================================================
    # 6. Generate Multi-Architecture Comparison Charts
    # =========================================================================================
    plt.close('all')
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, model_name in enumerate(models_dict.keys()):
        df_sub = df_results[df_results['Model_Architecture'] == model_name]
        summary_df = df_sub.groupby('Method')['Time (s)'].mean().reset_index()
        
        axes[idx].bar(summary_df['Method'], summary_df['Time (s)'], color=colors)
        axes[idx].set_title(f'Execution Time: {model_name}', fontweight='bold')
        axes[idx].set_ylabel('Seconds (Lower = Faster)')
        axes[idx].tick_params(axis='x', rotation=20)
        axes[idx].grid(axis='y', linestyle='--', alpha=0.5)

    plt.suptitle(f'Search Efficiency across Machine Learning Architectures ({dataset_name})', fontsize=15, fontweight='bold')
    plt.tight_layout()
    chart_filename = f'multi_architecture_chart_{dataset_name}.png'
    plt.savefig(chart_filename, bbox_inches='tight')
    plt.close('all')
    print(f"\n[Visualization] Multi-architecture comparison chart saved to '{chart_filename}'.")


def run_all_datasets_benchmark(n_test_instances: int = 100):
    """
    Executes extended benchmarks across all 6 supported domain datasets.
    Saves individual dataset results into separate CSV and PNG chart files.
    """
    datasets = ['financial_loan', 'adult', 'german_credit', 'medical_insurance', 'diabetes', 'breast_cancer']
    print(f"=============================================================================")
    print(f"   STARTING MULTI-DOMAIN BENCHMARK ACROSS ALL {len(datasets)} DATASETS (N={n_test_instances})")
    print(f"=============================================================================")
    
    for ds in datasets:
        try:
            run_extended_benchmark(dataset_name=ds, n_test_instances=n_test_instances)
        except Exception as e:
            print(f"[Error] Failed running benchmark for '{ds}': {e}")


#########################################################
ds_target = 'all'
n_inst = 100

if ds_target == 'all':
    run_all_datasets_benchmark(n_test_instances=n_inst)
else:
    run_extended_benchmark(dataset_name=ds_target, n_test_instances=n_inst)
#########################################################

# if __name__ == '__main__':
#    # Usage: python test_benchmark_comparison_extension.py [dataset_name] [n_test_instances]
#    # Example 1: python test_benchmark_comparison_extension.py adult 50
#    # Example 2: python test_benchmark_comparison_extension.py all 100
    
#    ds_target = 'financial_loan'
#    n_inst = 100

#    if len(sys.argv) > 1:
#        ds_target = sys.argv[1]
#    if len(sys.argv) > 2:
#        try:
#            n_inst = int(sys.argv[2])
#        except ValueError:
#            n_inst = 100

#    if ds_target == 'all':
#        run_all_datasets_benchmark(n_test_instances=n_inst)
#    else:
#        run_extended_benchmark(dataset_name=ds_target, n_test_instances=n_inst)