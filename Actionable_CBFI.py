"""
========================================================================================
Actionable CBFI (A-CBFI): Bridging Structural Decomposition and Causal Counterfactual Recourse
========================================================================================
This module extends Localized CBFI into a Diagnosis-Prescription integrated XAI framework.

Key Components:
1. StructuralCausalModel (SCM): DAG modeling and do-calculus causal forward propagation.
2. CausalCBFIDiagnoser: Diagnosis phase computing C-G1 (Causal Main Effect) and 
   C-G4 (Causal Interaction) via SCM do-interventions to identify bottleneck targets (T).
3. ActionableCBFISolver: Targeted Counterfactual Search phase constraining search space 
   to actionable target nodes (T) and optimizing minimum cost recourse actions do(T = a).
4. Evaluation Metrics: Search Efficiency, Sparsity (L0-norm), and Causal Plausibility.
5. Legacy Compatibility & Visualizations: Retains all original functions from 
   local_cbfi_clean_20260410.py and adds causal recourse path network graphs.
========================================================================================
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Union, Optional, Callable

# =========================================================================================
# 1. Structural Causal Model (SCM)
# =========================================================================================

class StructuralCausalModel:
    """
    Represents a Structural Causal Model (SCM) specified by a Directed Acyclic Graph (DAG)
    and structural causal functions for descendant nodes.
    """
    def __init__(self, dag_edges: List[Tuple[str, str]] = None, nodes: List[str] = None):
        self.graph = nx.DiGraph()
        if nodes:
            self.graph.add_nodes_from(nodes)
        if dag_edges:
            self.graph.add_edges_from(dag_edges)
        
        self.structural_equations: Dict[str, Callable] = {}
        self.residual_noises: Dict[str, float] = {}
        self._topological_order: List[str] = []
        self._update_topological_order()

    def _update_topological_order(self):
        if len(self.graph.nodes) > 0:
            if not nx.is_directed_acyclic_graph(self.graph):
                raise ValueError("Graph must be a Directed Acyclic Graph (DAG).")
            self._topological_order = list(nx.topological_sort(self.graph))
        else:
            self._topological_order = []

    def set_nodes_and_edges(self, nodes: List[str], dag_edges: List[Tuple[str, str]]):
        self.graph.clear()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(dag_edges)
        self._update_topological_order()

    def add_structural_equation(self, child_node: str, equation_fn: Callable):
        """
        Assigns a custom structural equation x_child = f(parents_dict, noise).
        equation_fn signature: fn(parents_values: dict, noise: float = 0.0) -> float
        """
        if child_node not in self.graph.nodes:
            self.graph.add_node(child_node)
            self._update_topological_order()
        self.structural_equations[child_node] = equation_fn

    @staticmethod
    def auto_discover_dag(df: pd.DataFrame, threshold: float = 0.15, max_parents: int = 3) -> List[Tuple[str, str]]:
        """
        Automatically estimates a Directed Acyclic Graph (DAG) structure from observational data.
        Uses feature correlation topology and acyclic DAG constraints.
        """
        nodes = list(df.columns)
        corr_matrix = df.corr().abs().fillna(0.0)
        variances = df.var().fillna(1.0)
        
        # Sort nodes by variance (exogenous / root nodes typically higher variance)
        sorted_nodes = list(variances.sort_values(ascending=False).index)
        
        edges = []
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        
        for i in range(len(sorted_nodes)):
            parent = sorted_nodes[i]
            for j in range(i + 1, len(sorted_nodes)):
                child = sorted_nodes[j]
                if corr_matrix.loc[parent, child] >= threshold:
                    if G.in_degree(child) < max_parents:
                        G.add_edge(parent, child)
                        if nx.is_directed_acyclic_graph(G):
                            edges.append((parent, child))
                        else:
                            G.remove_edge(parent, child)
        return edges

    def fit_from_data(self, df: pd.DataFrame, dag_edges: Optional[List[Tuple[str, str]]] = None, regressor_factory: Optional[Callable] = None):
        """
        Automatically fits structural causal models given observational data and a DAG structure.
        If dag_edges is None or empty, automatically estimates DAG structure using Auto Causal Discovery!
        Supports both linear regression and non-linear regressors (e.g. RandomForestRegressor, GAM, XGBoost).
        """
        if dag_edges is None or len(dag_edges) == 0:
            print("[Auto Causal Discovery] No DAG provided. Automatically estimating Causal DAG from data...")
            dag_edges = self.auto_discover_dag(df)
            print(f"[Auto Causal Discovery] Estimated DAG Edges ({len(dag_edges)}): {dag_edges}")

        nodes = list(df.columns)
        self.set_nodes_and_edges(nodes, dag_edges)

        for node in self._topological_order:
            parents = list(self.graph.predecessors(node))
            if len(parents) > 0:
                X_parent = df[parents].values
                y_child = df[node].values
                
                if regressor_factory is not None:
                    model = regressor_factory()
                else:
                    model = LinearRegression()
                
                model.fit(X_parent, y_child)
                
                # Define closure for structural equation
                def make_eq(m=model, p_list=parents):
                    def eq_fn(p_dict: dict, noise: float = 0.0) -> float:
                        p_vals = np.array([[p_dict[p] for p in p_list]])
                        pred = m.predict(p_vals)[0]
                        return float(pred + noise)
                    return eq_fn

                self.structural_equations[node] = make_eq()

    def get_parents(self, node: str) -> List[str]:
        return list(self.graph.predecessors(node))

    def get_descendants(self, node: str) -> List[str]:
        return list(nx.descendants(self.graph, node)) if node in self.graph else []

    def forward_pass(self, instance: pd.Series, interventions: Dict[str, float]) -> pd.Series:
        """
        Executes Pearl's do-calculus intervention do(X = v) and propagates structural 
        changes downstream along the DAG in topological order.
        """
        x_cf = instance.copy().astype(float)
        
        # Determine topological execution order
        all_nodes = [col for col in instance.index if col in self._topological_order]
        # Include any extra nodes not in topology at the end
        remaining = [col for col in instance.index if col not in all_nodes]
        eval_order = all_nodes + remaining

        for node in eval_order:
            if node in interventions:
                x_cf[node] = float(interventions[node])
            elif node in self.structural_equations and node in self.graph.nodes:
                parents = list(self.graph.predecessors(node))
                if parents:
                    parents_dict = {p: x_cf[p] for p in parents}
                    
                    # Compute factual exogenous noise residual: noise = x_factual[node] - f(factual_parents)
                    factual_parents_dict = {p: instance[p] for p in parents}
                    try:
                        f_factual_pred = self.structural_equations[node](factual_parents_dict, noise=0.0)
                        noise = instance[node] - f_factual_pred
                    except Exception:
                        noise = 0.0
                    
                    # Propagate counterfactual parent values
                    x_cf[node] = float(self.structural_equations[node](parents_dict, noise=noise))

        return x_cf


# =========================================================================================
# 2. Causal CBFI Diagnosis Phase
# =========================================================================================

class CausalCBFIDiagnoser:
    """
    Diagnoses prediction bottlenecks by evaluating Causal Main Effects (C-G1)
    and Causal Interaction Effects (C-G4) using Pearl's do-calculus on an SCM.
    """
    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm

    def diagnose_instance(
        self,
        model,
        instance: pd.Series,
        background_data: pd.DataFrame,
        actual_y: Optional[float] = None,
        job_type: str = 'classification',
        n_samples: int = 50,
        immutable_features: Optional[List[str]] = None,
        use_proba: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Computes C-G1 and C-G4 scores using SCM do-interventions across features.
        Identifies actionable target feature set T.
        Supports both hard label flips and continuous probability delta shifts (use_proba=True).
        """
        if immutable_features is None:
            immutable_features = []

        features = list(instance.index)
        instance_df = pd.DataFrame([instance])
        
        has_proba = hasattr(model, 'predict_proba') and use_proba

        if job_type == 'classification':
            factual_pred = model.predict(instance_df)[0]
            if has_proba:
                factual_prob = model.predict_proba(instance_df)[0, factual_pred]
            else:
                factual_prob = 1.0
        else:
            factual_pred = model.predict(instance_df)[0]
            if actual_y is None:
                actual_y = factual_pred
            factual_error = abs(factual_pred - actual_y)

        results = []

        for feat in features:
            if feat in background_data.columns:
                interv_pool = background_data[feat].sample(n=min(n_samples, len(background_data)), random_state=42).values
            else:
                interv_pool = np.linspace(instance[feat] * 0.5, instance[feat] * 1.5, n_samples)

            cg1_acc = 0.0

            for val in interv_pool:
                x_do_feat = self.scm.forward_pass(instance, {feat: val})
                pred_do_feat = model.predict(pd.DataFrame([x_do_feat]))[0]

                if job_type == 'classification':
                    if has_proba:
                        prob_do_feat = model.predict_proba(pd.DataFrame([x_do_feat]))[0, factual_pred]
                        # Preserve signed probability drop delta (allows negative G1 when intervention increases risk)
                        cg1_acc += (factual_prob - prob_do_feat)
                    else:
                        if pred_do_feat != factual_pred:
                            cg1_acc += 1.0
                else:
                    err_do = abs(pred_do_feat - actual_y)
                    cg1_acc += (factual_error - err_do)

            cg1_score = cg1_acc / len(interv_pool)

            # Compute C-G4 interaction with other features
            other_feats = [f for f in features if f != feat]
            cg4_scores = []
            
            for other_f in other_feats[:3]:
                if other_f in background_data.columns:
                    other_pool = background_data[other_f].sample(n=min(n_samples, len(background_data)), random_state=42).values
                else:
                    other_pool = np.linspace(instance[other_f] * 0.5, instance[other_f] * 1.5, n_samples)
                
                cg4_pair_acc = 0.0
                for v1, v2 in zip(interv_pool[:len(other_pool)], other_pool):
                    x_do_joint = self.scm.forward_pass(instance, {feat: v1, other_f: v2})
                    pred_joint = model.predict(pd.DataFrame([x_do_joint]))[0]

                    x_do_f2 = self.scm.forward_pass(instance, {other_f: v2})
                    pred_f2 = model.predict(pd.DataFrame([x_do_f2]))[0]

                    if job_type == 'classification':
                        if has_proba:
                            prob_joint = model.predict_proba(pd.DataFrame([x_do_joint]))[0, factual_pred]
                            prob_do_f1 = model.predict_proba(pd.DataFrame([self.scm.forward_pass(instance, {feat: v1})]))[0, factual_pred]
                            prob_do_f2 = model.predict_proba(pd.DataFrame([x_do_f2]))[0, factual_pred]
                            
                            joint_drop = (factual_prob - prob_joint)
                            indiv_drop = (factual_prob - prob_do_f1) + (factual_prob - prob_do_f2)
                            cg4_pair_acc += max(0.0, joint_drop - indiv_drop)
                        else:
                            if (pred_joint != factual_pred) and (pred_do_feat == factual_pred) and (pred_f2 == factual_pred):
                                cg4_pair_acc += 1.0
                    else:
                        err_joint = abs(pred_joint - actual_y)
                        err_f2 = abs(pred_f2 - actual_y)
                        joint_reduction = (factual_error - err_joint)
                        indiv_reduction = (factual_error - err_do) + (factual_error - err_f2)
                        cg4_pair_acc += max(0.0, joint_reduction - indiv_reduction)
                
                cg4_scores.append(cg4_pair_acc / len(other_pool))

            cg4_score = np.mean(cg4_scores) if cg4_scores else 0.0
            is_actionable = feat not in immutable_features

            results.append({
                'Feature': feat,
                'C-G1 (Main Effect)': cg1_score,
                'C-G4 (Interaction)': cg4_score,
                'Total Causal Importance': cg1_score + cg4_score,
                'Actionable': is_actionable
            })

        df_diag = pd.DataFrame(results).set_index('Feature')
        df_diag = df_diag.sort_values(by='Total Causal Importance', ascending=False)

        # Identify target set T: top actionable features with positive importance
        target_nodes = [
            f for f in df_diag.index 
            if df_diag.loc[f, 'Actionable'] and df_diag.loc[f, 'Total Causal Importance'] >= 0.0
        ]
        
        if not target_nodes:
            # Fallback to any mutable feature if no high importance feature found
            target_nodes = [f for f in features if f not in immutable_features][:2]

        return df_diag, target_nodes


# =========================================================================================
# 3. Targeted Counterfactual Search & Actionable Recourse Phase
# =========================================================================================

class ActionableCBFISolver:
    """
    Performs targeted counterfactual recourse search by constraining search space
    to diagnosed actionable target nodes T and evaluating minimum cost actions via SCM.
    """
    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm

    def _calculate_cost(
        self,
        instance: pd.Series,
        x_cf: pd.Series,
        mad_dict: Optional[Dict[str, float]] = None,
        cost_weights: Optional[Dict[str, float]] = None,
        lambda_mse: float = 0.0
    ) -> float:
        cost = 0.0
        for col in instance.index:
            diff = abs(x_cf[col] - instance[col])
            mad = mad_dict.get(col, 1.0) if mad_dict else 1.0
            if mad == 0 or np.isnan(mad):
                mad = 1.0
            weight = cost_weights.get(col, 1.0) if cost_weights else 1.0
            cost += weight * (diff / mad)

        # Improvement 2: SCM Plausibility Regularization
        if lambda_mse > 0:
            mse_scm = compute_causal_plausibility(x_cf, self.scm)
            cost += lambda_mse * mse_scm

        return float(cost)

    def find_recourse(
        self,
        model,
        instance: pd.Series,
        target_nodes: List[str],
        y_target: Union[int, float],
        background_data: pd.DataFrame,
        immutable_features: Optional[List[str]] = None,
        job_type: str = 'classification',
        cost_weights: Optional[Dict[str, float]] = None,
        max_iter: int = 500,
        lambda_mse: float = 0.5,
        enable_adaptive_relaxation: bool = True,
        max_target_expansion: int = 2
    ) -> Dict:
        """
        Executes Step 3: Next-Gen Targeted Counterfactual Search with:
        1. Adaptive Target Node Relaxation (Hybrid Elastic Targeting)
        2. SCM Plausibility Regularized Optimization (Curvature Penalty)
        3. Multi-Hop Dynamic Feedback Loop (Dynamic Re-routing)
        """
        start_time = time.time()
        if immutable_features is None:
            immutable_features = []

        # Filter target nodes against immutable features
        valid_targets = [t for t in target_nodes if t not in immutable_features]
        if not valid_targets:
            raise ValueError("No actionable (mutable) target nodes provided.")

        # Compute MAD for scaling
        mad_dict = {}
        for col in background_data.columns:
            mad_val = np.median(np.abs(background_data[col] - np.median(background_data[col])))
            mad_dict[col] = mad_val if mad_val > 1e-6 else 1.0

        # Sub-routine to evaluate search over a specific candidate target set
        def _evaluate_target_set(current_targets, budget):
            best_c = float('inf')
            best_act = None
            best_cf = None
            evals = 0

            n_samples_per_target = max(5, int(np.power(budget, 1.0 / max(1, len(current_targets)))))
            target_grids = {}
            for t in current_targets:
                min_val = background_data[t].min()
                max_val = background_data[t].max()
                target_grids[t] = np.linspace(min_val, max_val, n_samples_per_target)

            grid_keys = list(target_grids.keys())
            grid_values = list(target_grids.values())

            total_combos = 1
            for v in grid_values:
                total_combos *= len(v)

            np.random.seed(42)
            if total_combos > budget:
                combos_to_eval = []
                for _ in range(budget):
                    combo = tuple(float(np.random.choice(target_grids[k])) for k in grid_keys)
                    combos_to_eval.append(combo)
            else:
                from itertools import product
                combos_to_eval = list(product(*grid_values))

            cf_list = []
            actions_list = []

            for combo in combos_to_eval:
                evals += 1
                action = {k: float(v) for k, v in zip(grid_keys, combo)}
                x_cf = self.scm.forward_pass(instance, action)
                cf_list.append(x_cf)
                actions_list.append(action)

            if cf_list:
                df_cf_batch = pd.DataFrame(cf_list)
                preds_batch = model.predict(df_cf_batch)

                for i in range(len(preds_batch)):
                    pred_y = preds_batch[i]
                    x_cf = cf_list[i]
                    action = actions_list[i]

                    success = False
                    if job_type == 'classification':
                        if pred_y == y_target:
                            success = True
                    else:
                        if abs(pred_y - y_target) < abs(model.predict(pd.DataFrame([instance]))[0] - y_target) * 0.2:
                            success = True

                    if success:
                        # Evaluate regularized cost (incorporating SCM Plausibility MSE)
                        reg_cost = self._calculate_cost(instance, x_cf, mad_dict, cost_weights, lambda_mse=lambda_mse)
                        if reg_cost < best_c:
                            best_c = reg_cost
                            best_act = action
                            best_cf = x_cf

            return best_c, best_act, best_cf, evals, current_targets

        # Phase 1: Search using primary diagnosed target nodes
        budget_p1 = max_iter // 2 if enable_adaptive_relaxation else max_iter
        best_cost, best_action, best_x_cf, evaluations, evaluated_targets = _evaluate_target_set(valid_targets, budget_p1)

        # Phase 2: Improvement 1 & 3 - Multi-Hop Dynamic Feedback Loop (Adaptive Target Node Relaxation)
        # Triggered if Phase 1 found no recourse OR if best cost is high (high intervention load)
        if enable_adaptive_relaxation and (best_action is None or best_cost > 8.0):
            expanded_targets = set(valid_targets)
            if hasattr(self.scm, 'dag') and self.scm.dag is not None:
                for node in valid_targets:
                    # Explore 1-hop parents and children in DAG
                    for p, c in self.scm.dag.edges():
                        if c == node and p not in immutable_features:
                            expanded_targets.add(p)
                        elif p == node and c not in immutable_features:
                            expanded_targets.add(c)

            added_nodes = [n for n in expanded_targets if n not in valid_targets][:max_target_expansion]
            relaxed_targets = valid_targets + added_nodes

            if len(relaxed_targets) > len(valid_targets):
                budget_p2 = max_iter - budget_p1
                c_p2, act_p2, cf_p2, evals_p2, targets_p2 = _evaluate_target_set(relaxed_targets, budget_p2)
                evaluations += evals_p2

                if act_p2 is not None and (best_action is None or c_p2 < best_cost):
                    best_cost = c_p2
                    best_action = act_p2
                    best_x_cf = cf_p2
                    evaluated_targets = targets_p2

        elapsed_time = time.time() - start_time

        # Calculate pure unregularized MAD cost for standard benchmark reporting
        clean_mad_cost = self._calculate_cost(instance, best_x_cf, mad_dict, cost_weights, lambda_mse=0.0) if best_x_cf is not None else None

        return {
            'best_action': best_action,
            'x_counterfactual': best_x_cf,
            'minimum_cost': clean_mad_cost,
            'target_nodes': evaluated_targets,
            'evaluations': evaluations,
            'elapsed_time': elapsed_time,
            'success': best_action is not None
        }

    def find_recourse_untargeted(
        self,
        model,
        instance: pd.Series,
        y_target: Union[int, float],
        background_data: pd.DataFrame,
        immutable_features: Optional[List[str]] = None,
        job_type: str = 'classification',
        max_iter: int = 500
    ) -> Dict:
        """
        Executes untargeted baseline search across ALL mutable features for comparison.
        """
        start_time = time.time()
        if immutable_features is None:
            immutable_features = []

        all_mutable = [f for f in instance.index if f not in immutable_features]
        return self.find_recourse(
            model=model,
            instance=instance,
            target_nodes=all_mutable,
            y_target=y_target,
            background_data=background_data,
            immutable_features=immutable_features,
            job_type=job_type,
            max_iter=max_iter
        )


# =========================================================================================
# 4. Evaluation Metrics
# =========================================================================================

def compute_search_efficiency(targeted_res: Dict, untargeted_res: Dict) -> Dict[str, float]:
    """
    Computes Search Efficiency comparison metrics (iterations & speedup ratio).
    """
    t_evals = targeted_res.get('evaluations', 1)
    u_evals = untargeted_res.get('evaluations', 1)
    
    t_time = targeted_res.get('elapsed_time', 1e-4)
    u_time = untargeted_res.get('elapsed_time', 1e-4)

    eval_reduction_ratio = float(u_evals) / max(1, t_evals)
    speedup_ratio = float(u_time) / max(1e-6, t_time)

    return {
        'Targeted Evaluations': t_evals,
        'Untargeted Evaluations': u_evals,
        'Evaluation Reduction Factor': eval_reduction_ratio,
        'Targeted Time (s)': t_time,
        'Untargeted Time (s)': u_time,
        'Speedup Ratio': speedup_ratio
    }


def compute_sparsity(factual_instance: pd.Series, counterfactual_instance: pd.Series, tol: float = 1e-5) -> Dict[str, float]:
    """
    Computes L0 Sparsity metrics (number of modified features).
    """
    diffs = np.abs(factual_instance - counterfactual_instance)
    modified_mask = diffs > tol
    
    l0_count = int(np.sum(modified_mask))
    sparsity_ratio = float(l0_count) / len(factual_instance)

    return {
        'L0 Modified Count': l0_count,
        'Total Features': len(factual_instance),
        'Sparsity Ratio': sparsity_ratio,
        'Modified Features': list(factual_instance.index[modified_mask])
    }


def compute_causal_plausibility(counterfactual_instance: pd.Series, scm: StructuralCausalModel) -> float:
    """
    Computes Causal Plausibility score: mean squared error of counterfactual values 
    relative to structural causal equations in the SCM. Lower is more plausible.
    """
    total_residual = 0.0
    evaluated_nodes = 0

    for node in scm.graph.nodes:
        parents = scm.get_parents(node)
        if parents and node in scm.structural_equations:
            parents_dict = {p: counterfactual_instance[p] for p in parents if p in counterfactual_instance}
            if len(parents_dict) == len(parents):
                try:
                    expected_val = scm.structural_equations[node](parents_dict, noise=0.0)
                    actual_val = counterfactual_instance[node]
                    total_residual += (actual_val - expected_val) ** 2
                    evaluated_nodes += 1
                except Exception:
                    pass

    return float(total_residual / max(1, evaluated_nodes))


# =========================================================================================
# 5. Actionable CBFI End-to-End Orchestrator
# =========================================================================================

class ActionableCBFI:
    """
    High-level API for Actionable CBFI: Diagnosis -> Prescription -> Evaluation -> Report.
    """
    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm
        self.diagnoser = CausalCBFIDiagnoser(scm)
        self.solver = ActionableCBFISolver(scm)

    def explain_and_prescribe(
        self,
        model,
        instance: pd.Series,
        background_data: pd.DataFrame,
        y_desired: Union[int, float],
        actual_y: Optional[float] = None,
        job_type: str = 'classification',
        immutable_features: Optional[List[str]] = None,
        cost_weights: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Runs full Actionable CBFI pipeline.
        """
        if immutable_features is None:
            immutable_features = []

        # 1. Step 1 & 2: Diagnosis Phase
        df_diag, target_nodes = self.diagnoser.diagnose_instance(
            model=model,
            instance=instance,
            background_data=background_data,
            actual_y=actual_y,
            job_type=job_type,
            immutable_features=immutable_features
        )

        # 2. Step 3: Prescription Phase (Targeted CE Search)
        recourse_res = self.solver.find_recourse(
            model=model,
            instance=instance,
            target_nodes=target_nodes,
            y_target=y_desired,
            background_data=background_data,
            immutable_features=immutable_features,
            job_type=job_type,
            cost_weights=cost_weights
        )

        # 3. Untargeted Search for Benchmark Comparison
        untargeted_res = self.solver.find_recourse_untargeted(
            model=model,
            instance=instance,
            y_target=y_desired,
            background_data=background_data,
            immutable_features=immutable_features,
            job_type=job_type
        )

        # 4. Step 4: Metrics Computation
        eff_metrics = compute_search_efficiency(recourse_res, untargeted_res)
        
        if recourse_res['x_counterfactual'] is not None:
            sparsity_metrics = compute_sparsity(instance, recourse_res['x_counterfactual'])
            plausibility_score = compute_causal_plausibility(recourse_res['x_counterfactual'], self.scm)
        else:
            sparsity_metrics = {}
            plausibility_score = float('nan')

        # 5. Generate Text Report
        text_report = self._generate_report(instance, target_nodes, recourse_res, y_desired)

        return {
            'diagnosis_table': df_diag,
            'target_nodes': target_nodes,
            'recourse_action': recourse_res['best_action'],
            'counterfactual_instance': recourse_res['x_counterfactual'],
            'minimum_cost': recourse_res['minimum_cost'],
            'efficiency_metrics': eff_metrics,
            'sparsity_metrics': sparsity_metrics,
            'causal_plausibility': plausibility_score,
            'text_report': text_report
        }

    def _generate_report(self, instance: pd.Series, target_nodes: List[str], recourse_res: Dict, y_desired: Union[int, float]) -> str:
        report = []
        report.append("==========================================================")
        report.append("            Actionable CBFI Recourse Report               ")
        report.append("==========================================================")
        report.append(f"Target Desired Outcome: {y_desired}")
        report.append(f"Diagnosed Bottleneck Target Nodes (T): {target_nodes}")
        
        if recourse_res['success']:
            report.append("\n[Prescribed Actions (do-interventions)]:")
            for feat, val in recourse_res['best_action'].items():
                orig_val = instance[feat]
                report.append(f"  - do({feat}): {orig_val:.3f} -> {val:.3f}")
            
            report.append(f"\nRecourse Cost: {recourse_res['minimum_cost']:.4f}")
            
            report.append("\n[Downstream Causal Propagation Effects]:")
            x_cf = recourse_res['x_counterfactual']
            for col in instance.index:
                if col not in recourse_res['best_action'] and abs(x_cf[col] - instance[col]) > 1e-4:
                    report.append(f"  * {col}: {instance[col]:.3f} -> {x_cf[col]:.3f} (via SCM)")
        else:
            report.append("\nNo valid recourse action found within specified action bounds.")
        
        report.append("==========================================================")
        return "\n".join(report)


# =========================================================================================
# 6. Legacy Functions & Visualizations (from local_cbfi_clean_20260410.py)
# =========================================================================================

def _get_conditional_samples(instance, background_data, target_feature, n_neighbors=None, use_conditional=False):
    """
    Extracts a sample pool for feature neutralization.
    If use_conditional is False or n_neighbors is None, returns the full background dataset 
    (marginal sampling as specified in local_CBFI.pdf Eq 2-3).
    Otherwise, extracts a localized neighborhood sample pool based on KNN.
    """
    if not use_conditional or n_neighbors is None or n_neighbors >= len(background_data):
        return background_data

    corr_matrix = background_data.corr()
    relevant_features = corr_matrix[target_feature].abs().sort_values(ascending=False)
    cond_features = relevant_features.index[1:3].tolist()
    
    k = max(2, min(n_neighbors, len(background_data)))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(background_data[cond_features])
    
    query_instance = pd.DataFrame([instance[cond_features]])
    dist, indices = nn.kneighbors(query_instance)
    return background_data.iloc[indices[0]]


def explain_local_cbfi_classification_conditional(
    model, 
    instance, 
    background_data, 
    target_feature, 
    n_samples=100, 
    use_conditional=False, 
    n_neighbors=None,
    random_state=42
):
    """
    Decomposes classification predictions into mutually exclusive structural groups (G1 to G4).
    """
    instance_df = pd.DataFrame([instance])
    target_label = model.predict(instance_df)[0]
    conditional_pool = _get_conditional_samples(instance, background_data, target_feature, n_neighbors=n_neighbors, use_conditional=use_conditional)
    
    other_features = [f for f in instance.index if f != target_feature]
    counts = {'G1': 0, 'G2': 0, 'G3': 0, 'G4': 0}

    sampled_pool = conditional_pool.sample(n=n_samples, replace=True, random_state=random_state)

    for i in range(n_samples):
        random_sample = sampled_pool.iloc[i]

        ds_fx = instance.copy().astype(float)
        for f in other_features: ds_fx[f] = random_sample[f]
            
        ds_fx_minus = instance.copy().astype(float)
        ds_fx_minus[target_feature] = random_sample[target_feature]
        
        pred_fx = model.predict(pd.DataFrame([ds_fx]))[0]
        pred_fx_minus = model.predict(pd.DataFrame([ds_fx_minus]))[0]
        
        is_fx_correct = (pred_fx == target_label)
        is_fx_minus_correct = (pred_fx_minus == target_label)
        
        if is_fx_correct and not is_fx_minus_correct: counts['G1'] += 1
        elif not is_fx_correct and is_fx_minus_correct: counts['G2'] += 1
        elif is_fx_correct and is_fx_minus_correct: counts['G3'] += 1
        elif not is_fx_correct and not is_fx_minus_correct: counts['G4'] += 1

    ratios = {k: v / n_samples for k, v in counts.items()}
    return ratios, ratios['G1'] + ratios['G4']


def explain_local_cbfi_regression_conditional(
    model, 
    instance, 
    background_data, 
    target_feature, 
    actual, 
    pred=None, 
    n_samples=100,
    use_conditional=False,
    n_neighbors=None,
    random_state=42
):
    """
    Decomposes regression errors into G1, G2, G3, and G4 components.
    """
    pred = model.predict(pd.DataFrame([instance]))[0]
    diff_u = abs(pred - actual)
    conditional_pool = _get_conditional_samples(instance, background_data, target_feature, n_neighbors=n_neighbors, use_conditional=use_conditional)
    other_features = [f for f in instance.index if f != target_feature]
    
    sums = {'G1': 0.0, 'G2': 0.0, 'G3': 0.0, 'G4': 0.0}
    eps = 1e-9

    sampled_pool = conditional_pool.sample(n=n_samples, replace=True, random_state=random_state)

    for i in range(n_samples):
        random_sample = sampled_pool.iloc[i]

        ds_fx = instance.copy().astype(float)
        for f in other_features: ds_fx[f] = random_sample[f]
            
        ds_fx_minus = instance.copy().astype(float)
        ds_fx_minus[target_feature] = random_sample[target_feature]
        
        pred_x = model.predict(pd.DataFrame([ds_fx]))[0]
        pred_minus = model.predict(pd.DataFrame([ds_fx_minus]))[0]
        
        diff_x = abs(pred_x - actual)
        diff_minus = abs(pred_minus - actual)
        contribution = diff_minus - diff_u
        
        if abs(diff_minus - diff_u) < eps:
            sums['G3'] += contribution
        elif diff_minus > diff_u:
            if diff_x > diff_u: sums['G4'] += contribution
            else: sums['G1'] += contribution
        elif diff_u > diff_minus:
            if diff_x > diff_u: sums['G1'] += contribution
            else: sums['G4'] += contribution
                
    ratios = {
        'G1': sums['G1'] / n_samples,
        'G2': 0.0,
        'G3': sums['G3'] / n_samples,
        'G4': sums['G4'] / n_samples
    }
    importance = ratios['G1'] + ratios['G4']
    return ratios, importance


def get_cbfi_table(
    model, 
    instance, 
    background_data, 
    actual=None, 
    pred=None, 
    n_samples=100, 
    job_type='classification',
    use_conditional=False,
    n_neighbors=None
):
    """
    Computes Localized CBFI metrics for all features of an instance.
    """
    results = []
    features = instance.index

    for f in features:
        if job_type == 'classification':
            ratios, _ = explain_local_cbfi_classification_conditional(
                model, instance, background_data, f, n_samples=n_samples, use_conditional=use_conditional, n_neighbors=n_neighbors
            )
        else:
            ratios, _ = explain_local_cbfi_regression_conditional(
                model, instance, background_data, f, actual, pred, n_samples=n_samples, use_conditional=use_conditional, n_neighbors=n_neighbors
            )

        results.append({
            'Feature': f,
            'Power (G1)': ratios['G1'],
            'Others (G2)': ratios['G2'],
            'Common (G3)': ratios['G3'],
            'Interact (G4)': ratios['G4']
        })
    
    df_cbfi_table = pd.DataFrame(results).set_index('Feature')
    df_cbfi_table['Total'] = df_cbfi_table['Power (G1)'] + df_cbfi_table['Interact (G4)']
    df_cbfi_table = df_cbfi_table.sort_values(by='Total', ascending=True)
    return df_cbfi_table


def _local_pairwise_interaction_regression(model, instance, background_data, feat_x, n_neighbors=50, n_samples=500, random_state=42):
    features = instance.index
    instance_df = pd.DataFrame([instance])
    target_y = model.predict(instance_df)[0]
    interaction_results = []

    def get_diff_vectorized(fixed_feats, sampled_pool):
        diffs = []
        for i in range(len(sampled_pool)):
            random_sample = sampled_pool.iloc[i]
            temp_ds = instance.copy().astype(float)
            for f in features:
                if f not in fixed_feats:
                    temp_ds[f] = random_sample[f]
            pred = model.predict(pd.DataFrame([temp_ds]))[0]
            diffs.append(abs(pred - target_y))
        return np.array(diffs)

    for feat_y in features:
        if feat_x == feat_y: continue
        corr_matrix = background_data.corr()
        relevant = corr_matrix[[feat_x, feat_y]].abs().mean(axis=1).sort_values(ascending=False)
        cond_cols = [c for c in relevant.index if c not in [feat_x, feat_y]][:2]
        
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(background_data)))
        nn.fit(background_data[cond_cols])
        indices = nn.kneighbors(pd.DataFrame([instance[cond_cols]]))[1][0]
        pool = background_data.iloc[indices]
        sampled_pool = pool.sample(n=n_samples, replace=True, random_state=random_state)
        
        diff_x = get_diff_vectorized([feat_x], sampled_pool)
        diff_y = get_diff_vectorized([feat_y], sampled_pool)
        diff_xy = get_diff_vectorized([feat_x, feat_y], sampled_pool)
        
        int_val = (np.mean(diff_x - diff_xy) + np.mean(diff_y - diff_xy)) / 2
        interaction_results.append({'Feature_Y': feat_y, 'Interaction': int_val})
        
    return pd.DataFrame(interaction_results).set_index('Feature_Y').sort_values(by='Interaction', ascending=False)


def _local_pairwise_interaction_classification(model, instance, background_data, feat_x, n_neighbors=50, n_samples=500, random_state=42):
    features = instance.index
    instance_df = pd.DataFrame([instance])
    target_label = model.predict(instance_df)[0]
    interaction_results = []
    
    for feat_y in features:
        if feat_x == feat_y: continue
        corr_matrix = background_data.corr()
        relevant = corr_matrix[[feat_x, feat_y]].abs().mean(axis=1).sort_values(ascending=False)
        cond_cols = [c for c in relevant.index if c not in [feat_x, feat_y]][:2]
        
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(background_data)))
        nn.fit(background_data[cond_cols])
        indices = nn.kneighbors(pd.DataFrame([instance[cond_cols]]))[1][0]
        pool = background_data.iloc[indices]
        sampled_pool = pool.sample(n=n_samples, replace=True, random_state=random_state)
        
        g4_count = 0
        for i in range(n_samples):
            random_sample = sampled_pool.iloc[i]
            ds_xy, ds_x, ds_y = instance.copy().astype(float), instance.copy().astype(float), instance.copy().astype(float)
            for f in features:
                if f not in [feat_x, feat_y]: ds_xy[f] = random_sample[f]
                if f != feat_x: ds_x[f] = random_sample[f]
                if f != feat_y: ds_y[f] = random_sample[f]
            
            pred_xy = model.predict(pd.DataFrame([ds_xy]))[0]
            pred_x = model.predict(pd.DataFrame([ds_x]))[0]
            pred_y = model.predict(pd.DataFrame([ds_y]))[0]
            
            if (pred_xy == target_label) and (pred_x != target_label) and (pred_y != target_label):
                g4_count += 1
        
        interaction_results.append({'Feature_Y': feat_y, 'Interaction': g4_count / n_samples})
        
    return pd.DataFrame(interaction_results).set_index('Feature_Y').sort_values(by='Interaction', ascending=False)


def get_local_pairwise_interaction(model, instance, background_data, feat_x, n_samples=100, job_type='classification'):
    if job_type == 'classification':
        return _local_pairwise_interaction_classification(model, instance, background_data, feat_x, n_samples=n_samples)
    else:
        return _local_pairwise_interaction_regression(model, instance, background_data, feat_x, n_samples=n_samples)


def generate_all_interactions(model, instance, background_data, n_samples=100, job_type='classification'):
    features = instance.index.tolist()
    interaction_list = []
    feature_pairs = list(combinations(features, 2))

    for feat_x, feat_y in feature_pairs:
        res_df = get_local_pairwise_interaction(model, instance, background_data, feat_x, n_samples, job_type)
        int_val = res_df.loc[feat_y, 'Interaction']
        interaction_list.append({'Feature_X': feat_x, 'Feature_Y': feat_y, 'Interaction': int_val})
        
    return pd.DataFrame(interaction_list)


# --- Visualization Routines ---

def visualize_feature_contribution(df_res, sample_instance, actual_y, pred_y, scaler=None):
    if scaler is not None:
        original_instance = pd.Series(scaler.inverse_transform(sample_instance.values.reshape(1, -1))[0], index=sample_instance.index)
    else:
        original_instance = sample_instance    

    plt.close('all')
    features = df_res.index
    values = original_instance[features].values
    y_labels = [f"{f}: {v}" for f, v in zip(features, values)]
    
    g1_vals = df_res['Power (G1)'].values
    g4_vals = df_res['Interact (G4)'].values

    plt.figure(figsize=(12, 7))
    for i in range(len(features)):
        g1 = g1_vals[i]
        g4 = g4_vals[i]
        plt.barh(y_labels[i], g1, color='#00bfc4', label='Main effect (G1)' if i==0 else "", alpha=1.0)
        if np.sign(g1) == np.sign(g4) or g1 == 0:
            plt.barh(y_labels[i], g4, left=g1, color='#f8766d', label='Interaction (G4)' if i==0 else "", alpha=1.0)
        else:
            plt.barh(y_labels[i], g4, color='#f8766d', label='Interaction (G4)' if i==0 else "", alpha=1.0)

    all_points = np.concatenate([g1_vals, g4_vals, g1_vals + g4_vals, [0]])
    min_x, max_x = np.min(all_points), np.max(all_points)
    x_range = max_x - min_x
    buffer = x_range * 0.1 if x_range > 0 else 0.5
    
    plt.xlim(min_x - buffer, max_x + buffer)
    plt.axvline(0, color='black', linewidth=1.2)
    plt.title(f"Local Feature Importance\n(Target: Actual={actual_y}, Predict={pred_y})")
    plt.xlabel("Local Importance Value (G1 + G4)")
    plt.legend(loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_local_pairwise_interaction(df_interact_input, sample_instance, actual_y, pred_y, scaler=None):
    if scaler is not None:
        original_instance = pd.Series(scaler.inverse_transform(sample_instance.values.reshape(1, -1))[0], index=sample_instance.index)
    else:
        original_instance = sample_instance    

    df_interact = df_interact_input.copy()
    target_feature = original_instance.index.difference(df_interact.index)[0]
    df_interact.index = df_interact.index.astype(str) + ": " + original_instance[df_interact.index].astype(str)

    plt.close('all')
    plt.figure(figsize=(10, 6))
    colors = ['#f8766d' if x > 0 else '#00bfc4' for x in df_interact['Interaction']]
    df_interact['Interaction'].plot(kind='barh', color=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f"Interaction Strength with [{target_feature}: {original_instance[target_feature]}]\n(Target: Actual={actual_y}, Predict={pred_y})")
    plt.xlabel("Interaction Value")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def visualize_feature_interaction_graph(importance_data, interaction_df, threshold=1.0):
    plt.close('all')
    G = nx.Graph()
    
    if isinstance(importance_data, pd.Series):
        for feat, val in importance_data.items(): G.add_node(feat, importance=val)
    else:
        col = 'Total' if 'Total' in importance_data.columns else importance_data.columns[0]
        for feat, row in importance_data.iterrows(): G.add_node(feat, importance=row[col])
            
    for _, row in interaction_df.iterrows():
        u, v, weight = row['Feature_X'], row['Feature_Y'], row['Interaction']
        if abs(weight) > threshold:
            G.add_edge(u, v, weight=weight)

    pos = nx.spring_layout(G, k=1.5, seed=42)
    plt.figure(figsize=(14, 10))
    ax = plt.gca()

    node_sizes = [max(800, G.nodes[n]['importance'] * 3500) for n in G.nodes]
    node_colors = [G.nodes[n]['importance'] for n in G.nodes]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.YlOrRd, edgecolors='gray', linewidths=0.5, ax=ax)
    
    node_labels = {n: f"{n}\n({G.nodes[n]['importance']:.2f})" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold', ax=ax)

    edges = G.edges(data=True)
    if len(edges) > 0:
        edge_widths = [((abs(d['weight']) - threshold) * 8) + 1 for u, v, d in edges]
        edge_colors = ['#D3D3D3' if d['weight'] > 0 else '#FF8C00' for u, v, d in edges]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.5, ax=ax)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='#0000FF', font_weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

    plt.title(f"Case-Based Feature Interaction Graph (Threshold > {threshold})", fontsize=16, fontweight='bold', pad=30)
    cbar = plt.colorbar(nodes, ax=ax, shrink=0.8)
    cbar.set_label('Feature Importance', rotation=270, labelpad=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


class visualize_draggable_interaction_graph:
    def __init__(self, importance_data, interaction_df, threshold=0.0):
        self.importance_data = importance_data
        self.interaction_df = interaction_df
        self.threshold = threshold
        self.selected_node = None
        
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.G = nx.Graph()
        if isinstance(self.importance_data, pd.Series):
            for feat, val in self.importance_data.items(): self.G.add_node(feat, importance=val)
        else:
            col = 'Total' if 'Total' in self.importance_data.columns else self.importance_data.columns[0]
            for feat, row in self.importance_data.iterrows(): self.G.add_node(feat, importance=row[col])

        for _, row in self.interaction_df.iterrows():
            u, v, weight = row['Feature_X'], row['Feature_Y'], row['Interaction']
            if abs(weight) > self.threshold:
                self.G.add_edge(u, v, weight=weight)

        self.pos = nx.spring_layout(self.G, k=1.5)
        self.update_plot()
        plt.show()

    def update_plot(self):
        self.ax.clear()
        importances = [self.G.nodes[n]['importance'] for n in self.G.nodes]
        max_imp = max(importances) if importances and max(importances) > 0 else 1
        node_sizes = [max(1000, (self.G.nodes[n]['importance'] / max_imp) * 7000) for n in self.G.nodes]
        
        nodes = nx.draw_networkx_nodes(self.G, self.pos, node_size=node_sizes, node_color=importances, cmap=plt.cm.YlOrRd, edgecolors='gray', linewidths=2, ax=self.ax)
        node_labels = {n: f"{n}\n({self.G.nodes[n]['importance']:.2f})" for n in self.G.nodes}
        nx.draw_networkx_labels(self.G, self.pos, labels=node_labels, font_size=10, font_weight='bold', ax=self.ax)
        
        edges = self.G.edges(data=True)
        if len(edges) > 0:
            weights = [abs(d['weight']) for u, v, d in edges]
            max_w = max(weights) if max(weights) > 0 else 1
            widths = [(w / max_w) * 10 + 1 for w in weights]
            colors = ['#D3D3D3' if d['weight'] > 0 else '#FF8C00' for u, v, d in edges]
            nx.draw_networkx_edges(self.G, self.pos, width=widths, edge_color=colors, alpha=0.4, ax=self.ax)
            edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in edges}
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=8)
        
        self.ax.set_title(f"Feature Interaction Graph (Threshold > {self.threshold})")
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax: return
        for node, (x, y) in self.pos.items():
            if np.hypot(x - event.xdata, y - event.ydata) < 0.1:
                self.selected_node = node
                break

    def on_release(self, event):
        self.selected_node = None

    def on_motion(self, event):
        if self.selected_node is not None and event.inaxes == self.ax:
            self.pos[self.selected_node] = (event.xdata, event.ydata)
            self.update_plot()


def visualize_causal_recourse_path(scm: StructuralCausalModel, factual: pd.Series, counterfactual: pd.Series, action_nodes: List[str]):
    """
    Visualizes the SCM DAG highlighting direct intervention nodes and downstream propagated features.
    """
    plt.close('all')
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    pos = nx.spring_layout(scm.graph, seed=42)
    node_colors = []
    
    for n in scm.graph.nodes:
        if n in action_nodes:
            node_colors.append('#ff7f0e') # Orange for direct action
        elif abs(counterfactual[n] - factual[n]) > 1e-4:
            node_colors.append('#2ca02c') # Green for downstream propagated
        else:
            node_colors.append('#1f77b4') # Blue for unchanged

    nx.draw_networkx_nodes(scm.graph, pos, node_color=node_colors, node_size=1500, ax=ax)
    
    labels = {}
    for n in scm.graph.nodes:
        diff = counterfactual[n] - factual[n]
        labels[n] = f"{n}\n({factual[n]:.2f} -> {counterfactual[n]:.2f})"
        
    nx.draw_networkx_labels(scm.graph, pos, labels=labels, font_size=9, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(scm.graph, pos, arrowstyle='->', arrowsize=20, edge_color='gray', ax=ax)

    plt.title("Actionable CBFI: Causal Recourse Propagation Path\n(Orange: Direct Action, Green: Downstream Causal Change)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_main_interaction_decomposition(
    df_diag: pd.DataFrame,
    instance_id: Optional[str] = None,
    prediction_label: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10.5, 6.5),
    show_plot: bool = False
) -> plt.Figure:
    """
    Visualizes the Causal Main Effect (C-G1) and Causal Interaction Effect (C-G4) 
    decomposition produced by Actionable-CBFI diagnosis.
    
    Produces a publication-ready stacked horizontal bar chart breaking down:
    Prediction -> Main/Interaction Effect Decomposition -> Causal Diagnosis -> Target Set T.

    Parameters:
    -----------
    df_diag : pd.DataFrame
        DataFrame returned by CausalCBFIDiagnoser.diagnose_instance() with columns:
        ['C-G1 (Main Effect)', 'C-G4 (Interaction)', 'Total Causal Importance', 'Actionable']
    instance_id : str, optional
        ID or index label of the evaluated instance.
    prediction_label : str, optional
        Factual prediction result (e.g. 'Rejection (High Risk)').
    save_path : str, optional
        Path to save the generated figure file.
    figsize : tuple, default (10.5, 6.5)
        Dimensions of matplotlib figure.
    show_plot : bool, default False
        Whether to invoke plt.show().

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure dataset is sorted by Total Causal Importance in ascending order for horizontal bar chart (top = most important)
    df_plot = df_diag.sort_values(by='Total Causal Importance', ascending=True).copy()

    features = list(df_plot.index)
    cg1_values = df_plot['C-G1 (Main Effect)'].values
    cg4_values = df_plot['C-G4 (Interaction)'].values

    y_pos = np.arange(len(features))
    bar_height = 0.55

    # Calculate min and max bounds for X-axis
    min_val = float(min(np.min(cg1_values), 0.0))
    max_val = float(max(np.max(cg1_values + cg4_values), 0.1))

    # Draw vertical reference line at 0.0
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.5, zorder=4)

    # Plot Stacked Horizontal Bars ensuring C-G1 is ALWAYS blue (#1f77b4) regardless of sign
    for i, (cg1, cg4) in enumerate(zip(cg1_values, cg4_values)):
        if cg1 < 0:
            # Negative C-G1: Blue bar extending leftwards from 0 to cg1
            ax.barh(i, cg1, height=bar_height, left=0, color='#1f77b4', edgecolor='black', alpha=0.85, zorder=3, label='C-G1: Causal Main Effect' if i==0 else "")
            # Positive C-G4 synergy: Orange bar extending rightwards from 0 to (cg1 + cg4)
            total_net = cg1 + cg4
            if total_net > 0:
                ax.barh(i, total_net, height=bar_height, left=0, color='#ff7f0e', edgecolor='black', alpha=0.85, zorder=2, label='C-G4: Causal Interaction Effect' if i==0 else "")
        else:
            # Positive C-G1: Blue bar extending rightwards from 0 to cg1
            ax.barh(i, cg1, height=bar_height, left=0, color='#1f77b4', edgecolor='black', alpha=0.85, zorder=2, label='C-G1: Causal Main Effect' if i==0 else "")
            
            if cg4 < 0:
                # Negative C-G4 (Interference/Antagonism): Orange/Hatched bar extending LEFTWARDS from cg1 back towards (cg1 + cg4)
                # Rendered with zorder=3 and narrower height so BOTH blue main effect and orange interference are visible!
                ax.barh(i, cg4, height=bar_height*0.70, left=cg1, color='#ff7f0e', edgecolor='black', hatch='//', alpha=0.90, zorder=3, label='C-G4: Negative Interaction Interference' if i==0 else "")
            else:
                # Positive C-G4 (Synergy): Orange bar extending rightwards from cg1 to (cg1 + cg4)
                ax.barh(i, cg4, height=bar_height, left=cg1, color='#ff7f0e', edgecolor='black', alpha=0.85, zorder=2, label='C-G4: Causal Interaction Effect' if i==0 else "")

    # Format Y-axis Labels:
    # Put ★ for immutable features (no (Immutable) text), clean feature name for actionable
    y_labels = []
    for feat in features:
        is_actionable = df_plot.loc[feat, 'Actionable']
        if not is_actionable:
            y_labels.append(f"★ {feat}")
        else:
            y_labels.append(feat)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight='bold')

    # Format Value Annotations:
    for i, (cg1, cg4) in enumerate(zip(cg1_values, cg4_values)):
        total = cg1 + cg4
        if abs(total) > 0.0001:
            text_x = total + (max_val * 0.015) if total >= 0 else 0.015
            val_str = f"{total:+.4f}" if cg1 < 0 else f"{total:.4f}"
            ax.text(text_x, i, val_str, va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Causal Importance Score (Probability Delta Shift via SCM Do-Interventions)', fontsize=12, fontweight='bold')
    
    # Format Title: Add 2nd line showing Prediction Result ONLY (no Instance ID)
    main_title = "Actionable-CBFI: Main/Interaction Effect Decomposition & Causal Diagnosis"
    
    if prediction_label is not None:
        title_str = f"{main_title}\nFactual Prediction: {prediction_label}"
    else:
        title_str = f"{main_title}\nFactual Prediction: Rejection (High Risk / Class 1)"

    ax.set_title(title_str, fontsize=13, fontweight='bold', pad=15)

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    x_min_bound = min_val * 1.35 if min_val < 0 else 0.0
    x_max_bound = max_val * 1.20
    ax.set_xlim(x_min_bound, x_max_bound)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Main/Interaction Effect Decomposition plot saved to '{save_path}'.")

    if show_plot:
        plt.show()

    return fig


def plot_causal_recourse_waterfall(
    model,
    scm: StructuralCausalModel,
    instance: pd.Series,
    target_nodes: List[str],
    background_data: pd.DataFrame,
    immutable_features: Optional[List[str]] = None,
    prediction_label: str = "Loan Denied (Class 1)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = False
) -> plt.Figure:
    """
    Visualizes the step-by-step Causal Recourse Waterfall Chart.
    Traces how factual prediction probability drops step-by-step as each diagnosed 
    bottleneck target feature is intervened on via SCM do-calculus.

    Parameters:
    -----------
    model : trained ML model (RandomForest, XGBoost, etc.)
    scm : StructuralCausalModel
    instance : pd.Series factual instance
    target_nodes : list of diagnosed target feature names
    background_data : pd.DataFrame
    prediction_label : str, default 'Loan Denied (Class 1)'
    save_path : str, optional
    figsize : tuple, default (10, 6)
    show_plot : bool, default False

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    instance_df = pd.DataFrame([instance])
    has_proba = hasattr(model, 'predict_proba')
    factual_pred = model.predict(instance_df)[0]
    
    if has_proba:
        factual_prob = model.predict_proba(instance_df)[0, factual_pred]
    else:
        factual_prob = 1.0

    steps = ['Factual\n(Rejection)']
    probs = [factual_prob]
    deltas = [0.0]

    current_interventions = {}
    current_x = instance.copy()

    for node in target_nodes:
        # Find optimal counterfactual value from background data
        if node in background_data.columns:
            median_val = float(background_data[node].median())
        else:
            median_val = float(instance[node] * 0.7)

        current_interventions[node] = median_val
        x_do = scm.forward_pass(instance, current_interventions)
        
        if has_proba:
            p_do = model.predict_proba(pd.DataFrame([x_do]))[0, factual_pred]
        else:
            p_do = 0.0 if model.predict(pd.DataFrame([x_do]))[0] != factual_pred else 1.0

        drop = probs[-1] - p_do
        steps.append(f"do({node})")
        probs.append(p_do)
        deltas.append(-drop)

    # Final Counterfactual step
    steps.append('Counterfactual\n(Approval 🎯)')
    probs.append(probs[-1])
    deltas.append(0.0)

    # Plot Waterfall Bars
    n_steps = len(steps)
    bar_width = 0.5

    for i in range(n_steps):
        if i == 0:
            # Baseline Factual (Red)
            ax.bar(i, probs[i], width=bar_width, color='#d62728', edgecolor='black', alpha=0.85)
            ax.text(i, probs[i] + 0.02, f"{probs[i]*100:.1f}%", ha='center', fontsize=11, fontweight='bold', color='#d62728')
        elif i == n_steps - 1:
            # Final Counterfactual (Green)
            ax.bar(i, probs[i], width=bar_width, color='#2ca02c', edgecolor='black', alpha=0.85)
            ax.text(i, probs[i] + 0.02, f"{probs[i]*100:.1f}%", ha='center', fontsize=11, fontweight='bold', color='#2ca02c')
        else:
            # Step Drop (Blue)
            bottom_val = probs[i]
            height_val = probs[i-1] - probs[i]
            ax.bar(i, height_val, bottom=bottom_val, width=bar_width, color='#1f77b4', edgecolor='black', alpha=0.85)
            ax.text(i, bottom_val + (height_val / 2.0), f"-{height_val*100:.1f}%p", ha='center', va='center', fontsize=10, fontweight='bold', color='white')

            # Draw connecting step lines
            ax.plot([i - 0.5, i + 0.5], [probs[i-1], probs[i-1]], color='gray', linestyle='--', linewidth=1.2)

    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(steps, fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Prediction Probability of Adverse Class ({prediction_label})', fontsize=12, fontweight='bold')
    ax.set_title(f'Actionable-CBFI: Causal Recourse Waterfall Chart\nFactual Prediction: {prediction_label} -> Target Approval', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, max(probs) * 1.18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Causal Recourse Waterfall plot saved to '{save_path}'.")

    if show_plot:
        plt.show()

    return fig


def plot_causal_diagnosis_matrix(
    df_diag: pd.DataFrame,
    prediction_label: str = "Credit Denied (High Risk / Class 1)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 7),
    show_plot: bool = False
) -> plt.Figure:
    """
    Pure Diagnosis Visualization Tool 1: Causal Main & Interaction Heatmap Matrix.
    
    Displays a 2D Heatmap Matrix where:
    - Diagonal elements (Xi, Xi): C-G1 Causal Main Effect
    - Off-diagonal elements (Xi, Xj): C-G4 Causal Interaction Synergy
    - Rightmost annotation column: Total Causal Importance & Actionability Tag

    Parameters:
    -----------
    df_diag : pd.DataFrame
        DataFrame returned by CausalCBFIDiagnoser.diagnose_instance()
    prediction_label : str
    save_path : str, optional
    figsize : tuple, default (9, 7)
    show_plot : bool, default False

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    # Sort features by total causal importance in descending order
    df_plot = df_diag.sort_values(by='Total Causal Importance', ascending=False).copy()
    features = list(df_plot.index[:8])  # Top 8 features for clear matrix

    n_feats = len(features)
    matrix = np.zeros((n_feats, n_feats))

    for i, f1 in enumerate(features):
        cg1 = df_plot.loc[f1, 'C-G1 (Main Effect)']
        cg4 = df_plot.loc[f1, 'C-G4 (Interaction)']
        
        # Diagonal: C-G1 Main Effect
        matrix[i, i] = cg1
        
        # Off-Diagonal: Distribute C-G4 interaction to neighbors
        for j, f2 in enumerate(features):
            if i != j:
                matrix[i, j] = cg4 / (n_feats - 1)

    # Plot Heatmap
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    # Annotate numeric values inside matrix cells
    for i in range(n_feats):
        for j in range(n_feats):
            val = matrix[i, j]
            if val > 0.0001:
                font_weight = 'bold' if i == j else 'normal'
                color_text = 'white' if val > (matrix.max() * 0.6) else 'black'
                ax.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=9.5, fontweight=font_weight, color=color_text)

    # Y-axis Labels
    y_labels = [f"★ {f}" if not df_plot.loc[f, 'Actionable'] else f"  {f}" for f in features]
    ax.set_xticks(range(n_feats))
    ax.set_yticks(range(n_feats))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10.5, fontweight='bold')
    ax.set_yticklabels(y_labels, fontsize=10.5, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Causal Bottleneck Score (SCM Probability Shift)', fontsize=11, fontweight='bold')

    main_title = "Actionable-CBFI: Pure Causal Diagnosis Matrix (C-G1 Main & C-G4 Interaction)"
    subtitle = f"Factual Prediction: {prediction_label} | (Diagonal: Main Effect, Off-Diagonal: Interaction Synergy)"
    ax.set_title(f"{main_title}\n{subtitle}", fontsize=12, fontweight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Pure Causal Diagnosis Matrix plot saved to '{save_path}'.")

    if show_plot:
        plt.show()

    return fig

