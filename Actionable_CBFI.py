"""
========================================================================================
Actionable CBFI (A-CBFI): Bridging Structural Decomposition and Causal Counterfactual Recourse
========================================================================================
This module extends Localized CBFI into a Diagnosis-Prescription integrated XAI framework.

Key Upgrades in this Version (Interaction-Aware Revision):
1. Level 1 (Node Diagnosis): Existing node-level target selection.
2. Level 2 (Interaction Diagnosis): Pairwise C_G4(X, Y) synergistic matrix calculation.
3. Level 3 (Causal Search): Interaction-Guided Beam Search using Heuristic Score (S) 
   to determine branching priority, while maintaining pure MAD-scaled objective costs.
4. Visualization: Generates Interaction Graph (G_I) for structural bottleneck mapping.
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
        if child_node not in self.graph.nodes:
            self.graph.add_node(child_node)
            self._update_topological_order()
        self.structural_equations[child_node] = equation_fn

    @staticmethod
    def auto_discover_dag(df: pd.DataFrame, threshold: float = 0.15, max_parents: int = 3) -> List[Tuple[str, str]]:
        nodes = list(df.columns)
        corr_matrix = df.corr().abs().fillna(0.0)
        variances = df.var().fillna(1.0)
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
        if dag_edges is None or len(dag_edges) == 0:
            dag_edges = self.auto_discover_dag(df)

        nodes = list(df.columns)
        self.set_nodes_and_edges(nodes, dag_edges)

        for node in self._topological_order:
            parents = list(self.graph.predecessors(node))
            if len(parents) > 0:
                X_parent = df[parents].values
                y_child = df[node].values
                
                model = regressor_factory() if regressor_factory is not None else LinearRegression()
                model.fit(X_parent, y_child)
                
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
        if hasattr(self, 'graph') and node in self.graph.nodes:
            return list(nx.descendants(self.graph, node))
        return []

    def forward_pass(self, instance: pd.Series, interventions: Dict[str, float]) -> pd.Series:
        x_cf = instance.copy().astype(float)
        
        all_nodes = [col for col in instance.index if col in self._topological_order]
        remaining = [col for col in instance.index if col not in all_nodes]
        eval_order = all_nodes + remaining

        for node in eval_order:
            if node in interventions:
                x_cf[node] = float(interventions[node])
            elif node in self.structural_equations and node in self.graph.nodes:
                parents = list(self.graph.predecessors(node))
                if parents:
                    parents_dict = {p: x_cf[p] for p in parents}
                    factual_parents_dict = {p: instance[p] for p in parents}
                    try:
                        f_factual_pred = self.structural_equations[node](factual_parents_dict, noise=0.0)
                        noise = instance[node] - f_factual_pred
                    except Exception:
                        noise = 0.0
                    x_cf[node] = float(self.structural_equations[node](parents_dict, noise=noise))

        return x_cf

# =========================================================================================
# 2. Causal CBFI Diagnosis Phase (Level 1 & Level 2 Diagnosis)
# =========================================================================================

class CausalCBFIDiagnoser:
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
        """Level 1 (Node Diagnosis): Determines which variables matter |C_G1(Xi)| + |C_G4(Xi)|"""
        if immutable_features is None: immutable_features = []
        features = list(instance.index)
        instance_df = pd.DataFrame([instance])
        has_proba = hasattr(model, 'predict_proba') and use_proba

        if job_type == 'classification':
            factual_pred = model.predict(instance_df)[0]
            factual_prob = model.predict_proba(instance_df)[0, factual_pred] if has_proba else 1.0
        else:
            factual_pred = model.predict(instance_df)[0]
            if actual_y is None: actual_y = factual_pred
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
                        cg1_acc += (factual_prob - prob_do_feat)
                    else:
                        if pred_do_feat != factual_pred: cg1_acc += 1.0
                else:
                    cg1_acc += (factual_error - abs(pred_do_feat - actual_y))

            cg1_score = cg1_acc / len(interv_pool)

            other_feats = [f for f in features if f != feat]
            cg4_scores = []
            
            for other_f in other_feats[:3]: # Fast heuristic for node-level G4
                other_pool = background_data[other_f].sample(n=min(n_samples, len(background_data)), random_state=42).values if other_f in background_data.columns else np.linspace(instance[other_f] * 0.5, instance[other_f] * 1.5, n_samples)
                
                cg4_pair_acc = 0.0
                for v1, v2 in zip(interv_pool[:len(other_pool)], other_pool):
                    x_do_joint = self.scm.forward_pass(instance, {feat: v1, other_f: v2})
                    pred_joint = model.predict(pd.DataFrame([x_do_joint]))[0]
                    x_do_f2 = self.scm.forward_pass(instance, {other_f: v2})

                    if job_type == 'classification':
                        if has_proba:
                            prob_joint = model.predict_proba(pd.DataFrame([x_do_joint]))[0, factual_pred]
                            prob_do_f1 = model.predict_proba(pd.DataFrame([self.scm.forward_pass(instance, {feat: v1})]))[0, factual_pred]
                            prob_do_f2 = model.predict_proba(pd.DataFrame([x_do_f2]))[0, factual_pred]
                            
                            joint_drop = (factual_prob - prob_joint)
                            indiv_drop = (factual_prob - prob_do_f1) + (factual_prob - prob_do_f2)
                            cg4_pair_acc += max(0.0, joint_drop - indiv_drop)
                    else:
                        joint_reduction = (factual_error - abs(pred_joint - actual_y))
                        indiv_reduction = (factual_error - abs(model.predict(pd.DataFrame([self.scm.forward_pass(instance, {feat: v1})]))[0] - actual_y)) + (factual_error - abs(model.predict(pd.DataFrame([x_do_f2]))[0] - actual_y))
                        cg4_pair_acc += max(0.0, joint_reduction - indiv_reduction)
                
                cg4_scores.append(cg4_pair_acc / len(other_pool))

            cg4_score = np.mean(cg4_scores) if cg4_scores else 0.0
            total_magnitude = np.abs(cg1_score) + np.abs(cg4_score)

            results.append({
                'Feature': feat,
                'C-G1 (Main Effect)': cg1_score,
                'C-G4 (Interaction)': cg4_score,
                'Total Causal Importance': total_magnitude,
                'Actionable': feat not in immutable_features
            })

        df_diag = pd.DataFrame(results).set_index('Feature').sort_values(by='Total Causal Importance', ascending=False)
        df_actionable = df_diag[df_diag['Actionable']]
        
        if not df_actionable.empty:
            threshold = np.percentile(df_actionable['Total Causal Importance'], 25)
            target_nodes_candidates = df_actionable[df_actionable['Total Causal Importance'] >= threshold].index.tolist()
            # Top-K capping for ultra-high dimensional safety
            max_allowed_targets = 15
            if len(target_nodes_candidates) > max_allowed_targets:
                target_nodes = df_actionable.head(max_allowed_targets).index.tolist()
            else:
                target_nodes = target_nodes_candidates
        else:
            target_nodes = [f for f in features if f not in immutable_features][:2]

        return df_diag, target_nodes

    def diagnose_pairwise_interaction(
        self,
        model,
        instance: pd.Series,
        background_data: pd.DataFrame,
        target_nodes: List[str],
        actual_y: Optional[float] = None,
        job_type: str = 'classification',
        n_samples: int = 30,
        use_proba: bool = True
    ) -> pd.DataFrame:
        """
        Level 2 (Interaction Diagnosis):
        Estimate the signed pairwise C_G4(X_i, X_j) interaction matrix.

        Important design choice:
        - The signed interaction is preserved; negative values are NOT clipped.
        - Joint values for (X_i, X_j) are sampled from the same observational
          background rows so their empirical dependence is retained.
        - The returned matrix is diagnostic/search-guidance information only;
          it is not used to discount the final recourse cost.
        """
        n = len(target_nodes)
        pairwise_matrix = pd.DataFrame(0.0, index=target_nodes, columns=target_nodes)
        if n < 2 or len(background_data) == 0:
            return pairwise_matrix

        has_proba = hasattr(model, 'predict_proba') and use_proba
        factual_df = pd.DataFrame([instance])
        factual_pred = model.predict(factual_df)[0]
        factual_prob = (
            model.predict_proba(factual_df)[0, int(factual_pred)]
            if has_proba else 1.0
        )

        sample_n = min(n_samples, len(background_data))
        sampled_background = background_data.sample(
            n=sample_n, random_state=42, replace=False
        )

        for i, f1 in enumerate(target_nodes):
            for j in range(i + 1, n):
                f2 = target_nodes[j]
                cg4_values = []

                # Use paired observational samples (v1, v2) from the same
                # background row rather than independently sampling each feature.
                for _, row in sampled_background[[f1, f2]].iterrows():
                    v1, v2 = row[f1], row[f2]

                    x_joint = self.scm.forward_pass(instance, {f1: v1, f2: v2})
                    x_f1 = self.scm.forward_pass(instance, {f1: v1})
                    x_f2 = self.scm.forward_pass(instance, {f2: v2})

                    if job_type == 'classification':
                        if not has_proba:
                            continue

                        p_joint = model.predict_proba(pd.DataFrame([x_joint]))[0, int(factual_pred)]
                        p_f1 = model.predict_proba(pd.DataFrame([x_f1]))[0, int(factual_pred)]
                        p_f2 = model.predict_proba(pd.DataFrame([x_f2]))[0, int(factual_pred)]

                        # Signed excess joint effect:
                        #   C_G4(X_i,X_j) = joint effect - additive individual effects.
                        joint_effect = factual_prob - p_joint
                        individual_effect = (factual_prob - p_f1) + (factual_prob - p_f2)
                        cg4_values.append(joint_effect - individual_effect)

                    else:
                        actual = actual_y if actual_y is not None else factual_pred
                        factual_error = abs(factual_pred - actual)
                        pred_joint = model.predict(pd.DataFrame([x_joint]))[0]
                        pred_f1 = model.predict(pd.DataFrame([x_f1]))[0]
                        pred_f2 = model.predict(pd.DataFrame([x_f2]))[0]

                        joint_effect = factual_error - abs(pred_joint - actual)
                        individual_effect = (
                            factual_error - abs(pred_f1 - actual)
                        ) + (factual_error - abs(pred_f2 - actual))
                        cg4_values.append(joint_effect - individual_effect)

                score = float(np.mean(cg4_values)) if cg4_values else 0.0
                pairwise_matrix.loc[f1, f2] = score
                pairwise_matrix.loc[f2, f1] = score

        return pairwise_matrix

# =========================================================================================
# 3. Targeted Counterfactual Search & Actionable Recourse Phase (Level 3 Causal Search)
# =========================================================================================

class ActionableCBFISolver:
    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm

    def _calculate_pure_cost(self, instance: pd.Series, x_cf: pd.Series, mad_dict: Dict[str, float]) -> float:
        """Pure MAD-scaled L1 Cost calculation (No interaction discounting here)."""
        cost = 0.0
        for col in instance.index:
            diff = abs(x_cf[col] - instance[col])
            mad = mad_dict.get(col, 1.0)
            if mad == 0 or np.isnan(mad): mad = 1.0
            cost += (diff / mad)
        return float(cost)

    def _discretize_feature_space(self, background_data: pd.DataFrame, target_nodes: List[str], bins: int = 5) -> Dict[str, np.ndarray]:
        grids = {}
        for t in target_nodes:
            if background_data[t].nunique() <= bins:
                grids[t] = background_data[t].unique()
            else:
                grids[t] = np.unique(np.quantile(background_data[t].dropna(), np.linspace(0, 1, bins)))
        return grids

    def find_recourse(
        self, model, instance: pd.Series, target_nodes: List[str], y_target: Union[int, float],
        background_data: pd.DataFrame, immutable_features: Optional[List[str]] = None,
        job_type: str = 'classification', max_iter: int = 500, lambda_mse: float = 0.5,
        df_diag: Optional[pd.DataFrame] = None, pairwise_matrix: Optional[pd.DataFrame] = None
    ) -> Dict:
        start_time = time.time()
        if immutable_features is None: immutable_features = []
        valid_targets = [t for t in target_nodes if t not in immutable_features]
        if not valid_targets:
            raise ValueError("No actionable (mutable) target nodes provided.")

        mad_dict = {}
        for col in background_data.columns:
            median = background_data[col].median()
            mad = (background_data[col] - median).abs().median()
            if mad < 1e-3:
                std = background_data[col].std()
                if std > 1e-3: mad_dict[col] = float(std)
                else:
                    v_range = background_data[col].max() - background_data[col].min()
                    mad_dict[col] = float(v_range) if v_range > 0 else 1.0
            else:
                mad_dict[col] = float(mad)

        # Interaction-Guided Beam Search Evaluator
        def _evaluate_interaction_guided_beam_search(current_targets, budget):
            """
            Interaction-guided discretized beam search.

            C_G4(X_i, X_j) is used ONLY to prioritize which branches are
            expanded/evaluated first. It does not alter the MAD-scaled
            intervention cost or the final recourse objective.
            """
            beam_width = 5
            max_depth = min(len(current_targets), 5)
            grids = self._discretize_feature_space(background_data, current_targets, bins=6)

            # Normalize node-level and pairwise interaction magnitudes so that
            # their scales are comparable. No arbitrary lambda weighting is used.
            self_scores = {}
            if df_diag is not None:
                for t in current_targets:
                    if t in df_diag.index:
                        self_scores[t] = abs(float(df_diag.loc[t, 'C-G4 (Interaction)']))
                    else:
                        self_scores[t] = 0.0
            else:
                self_scores = {t: 0.0 for t in current_targets}

            self_scale = max(self_scores.values(), default=0.0)

            pair_abs_scale = 0.0
            if pairwise_matrix is not None and not pairwise_matrix.empty:
                valid_vals = []
                for i, f1 in enumerate(current_targets):
                    for f2 in current_targets[i + 1:]:
                        if f1 in pairwise_matrix.index and f2 in pairwise_matrix.columns:
                            valid_vals.append(abs(float(pairwise_matrix.loc[f1, f2])))
                pair_abs_scale = max(valid_vals, default=0.0)

            def _interaction_priority(candidate, active_set):
                """State-dependent priority S(X_j | A), used only for search order."""
                self_norm = (self_scores.get(candidate, 0.0) / self_scale) if self_scale > 0 else 0.0

                pair_norm = 0.0
                if active_set and pairwise_matrix is not None and pair_abs_scale > 0:
                    pair_values = []
                    for a in active_set:
                        if a in pairwise_matrix.index and candidate in pairwise_matrix.columns:
                            pair_values.append(abs(float(pairwise_matrix.loc[a, candidate])))
                    if pair_values:
                        pair_norm = max(pair_values) / pair_abs_scale

                # Equal, normalized contribution of node- and edge-level diagnosis.
                return 0.5 * self_norm + 0.5 * pair_norm

            # Each beam stores the actual objective score separately from the
            # interaction-guidance score.
            active_beams = [{
                'actions': {},
                'cost': 0.0,
                'search_score': 0.0,
                'interaction_priority': 0.0
            }]
            completed_paths = []
            evals = 0

            factual_pred = model.predict(pd.DataFrame([instance]))[0]
            factual_prob_target = None
            if job_type == 'classification' and hasattr(model, 'predict_proba'):
                factual_prob_target = model.predict_proba(pd.DataFrame([instance]))[0, int(y_target)]

            for depth in range(max_depth):
                next_beams = []

                for beam in active_beams:
                    active_set = list(beam['actions'].keys())
                    available_targets = [t for t in current_targets if t not in active_set]

                    # First order the candidate variables using C_G4(X) and
                    # state-dependent |C_G4(X_i,X_j)|. This is the only role of
                    # interaction diagnosis in the optimization engine.
                    candidate_targets = sorted(
                        available_targets,
                        key=lambda t: _interaction_priority(t, active_set),
                        reverse=True
                    )

                    for t in candidate_targets:
                        interaction_priority = _interaction_priority(t, active_set)

                        # Evaluate highly diagnosed branches first.
                        for val in grids[t]:
                            if evals >= budget:
                                break

                            evals += 1
                            new_actions = beam['actions'].copy()
                            new_actions[t] = float(val)

                            # Causal evaluation through SCM.
                            x_cf = self.scm.forward_pass(instance, new_actions)
                            df_cf = pd.DataFrame([x_cf])
                            pred_y = model.predict(df_cf)[0]

                            success = False
                            if job_type == 'classification' and pred_y == y_target:
                                success = True
                            elif job_type == 'regression':
                                factual_error = abs(float(factual_pred) - float(y_target))
                                success = abs(float(pred_y) - float(y_target)) < factual_error * 0.2

                            # Pure MAD-scaled intervention cost. Interaction scores
                            # NEVER discount or modify this quantity.
                            cost = self._calculate_pure_cost(instance, x_cf, mad_dict)
                            if lambda_mse > 0:
                                cost += lambda_mse * compute_causal_plausibility(x_cf, self.scm)

                            # Search objective used for beam survival. This remains
                            # independent of C_G4; interaction only determines the
                            # order in which candidate branches are explored.
                            search_score = cost
                            if not success and job_type == 'classification' and hasattr(model, 'predict_proba'):
                                prob = model.predict_proba(df_cf)[0, int(y_target)]
                                search_score += 5.0 * (-np.log(prob + 1e-9))

                            new_beam = {
                                'actions': new_actions,
                                'x_cf': x_cf,
                                'cost': cost,
                                'search_score': search_score,
                                'interaction_priority': interaction_priority
                            }

                            if success:
                                completed_paths.append(new_beam)
                            else:
                                next_beams.append(new_beam)

                        if evals >= budget:
                            break
                    if evals >= budget:
                        break

                if evals >= budget or not next_beams:
                    break

                # Beam survival is determined by the actual search objective.
                # Interaction priority is only a tie-breaker, preventing the
                # diagnostic score from becoming a hidden cost discount.
                next_beams.sort(
                    key=lambda x: (x['search_score'], -x['interaction_priority'])
                )
                active_beams = next_beams[:beam_width]

            # Final recourse is selected strictly by the actual intervention cost.
            best_c, best_act, best_cf = float('inf'), None, None
            for path in completed_paths:
                if path['cost'] < best_c:
                    best_c = path['cost']
                    best_act = path['actions']
                    best_cf = path['x_cf']

            return best_c, best_act, best_cf, evals, current_targets

        best_cost, best_action, best_x_cf, evaluations, evaluated_targets = _evaluate_interaction_guided_beam_search(valid_targets, max_iter)

        return {
            'best_action': best_action,
            'x_counterfactual': best_x_cf,
            'minimum_cost': best_cost if best_x_cf is not None else None,
            'target_nodes': evaluated_targets,
            'evaluations': evaluations,
            'elapsed_time': time.time() - start_time,
            'success': best_action is not None
        }

    # def find_recourse_untargeted(self, model, instance, y_target, background_data, immutable_features=None, job_type='classification', max_iter=500):
    #     # Untargeted search has no diagnosis info (df_diag=None, pairwise_matrix=None)
    #     all_mutable = [f for f in instance.index if f not in (immutable_features or [])]
    #     return self.find_recourse(model, instance, all_mutable, y_target, background_data, immutable_features, job_type, max_iter, df_diag=None, pairwise_matrix=None)
    def find_recourse_untargeted(self, model, instance, y_target, background_data, immutable_features=None, job_type='classification', max_iter=500, lambda_mse=0.0):
        # Untargeted search has no diagnosis info (df_diag=None, pairwise_matrix=None)
        all_mutable = [f for f in instance.index if f not in (immutable_features or [])]
        return self.find_recourse(model, instance, all_mutable, y_target, background_data, immutable_features, job_type, max_iter, lambda_mse=lambda_mse, df_diag=None, pairwise_matrix=None)

# =========================================================================================
# 4. Evaluation Metrics
# =========================================================================================

def compute_search_efficiency(targeted_res: Dict, untargeted_res: Dict) -> Dict[str, float]:
    t_evals, u_evals = targeted_res.get('evaluations', 1), untargeted_res.get('evaluations', 1)
    t_time, u_time = targeted_res.get('elapsed_time', 1e-4), untargeted_res.get('elapsed_time', 1e-4)
    return {
        'Targeted Evaluations': t_evals,
        'Untargeted Evaluations': u_evals,
        'Evaluation Reduction Factor': float(u_evals) / max(1, t_evals),
        'Targeted Time (s)': t_time,
        'Untargeted Time (s)': u_time,
        'Speedup Ratio': float(u_time) / max(1e-6, t_time)
    }

def compute_sparsity(factual_instance: pd.Series, counterfactual_instance: pd.Series, tol: float = 1e-5) -> Dict:
    modified_mask = np.abs(factual_instance - counterfactual_instance) > tol
    l0_count = int(np.sum(modified_mask))
    return {
        'L0 Modified Count': l0_count,
        'Total Features': len(factual_instance),
        'Sparsity Ratio': float(l0_count) / len(factual_instance),
        'Modified Features': list(factual_instance.index[modified_mask])
    }

def compute_causal_plausibility(counterfactual_instance: pd.Series, scm: StructuralCausalModel) -> float:
    total_residual, evaluated_nodes = 0.0, 0
    for node in scm.graph.nodes:
        parents = scm.get_parents(node)
        if parents and node in scm.structural_equations:
            parents_dict = {p: counterfactual_instance[p] for p in parents if p in counterfactual_instance}
            if len(parents_dict) == len(parents):
                try:
                    total_residual += (counterfactual_instance[node] - scm.structural_equations[node](parents_dict, noise=0.0)) ** 2
                    evaluated_nodes += 1
                except Exception: pass
    return float(total_residual / max(1, evaluated_nodes))

# =========================================================================================
# 5. Actionable CBFI End-to-End Orchestrator
# =========================================================================================

class ActionableCBFI:
    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm
        self.diagnoser = CausalCBFIDiagnoser(scm)
        self.solver = ActionableCBFISolver(scm)

    def explain_and_prescribe(self, model, instance, background_data, y_desired, actual_y=None, job_type='classification', immutable_features=None, max_iter=500) -> Dict:
        # 1. Node Diagnosis
        df_diag, target_nodes = self.diagnoser.diagnose_instance(model, instance, background_data, actual_y, job_type, immutable_features=immutable_features)
        
        # 2. Interaction Diagnosis (Pairwise)
        pairwise_matrix = self.diagnoser.diagnose_pairwise_interaction(model, instance, background_data, target_nodes, actual_y, job_type)
        
        # 3. Causal Search (Interaction-Guided)
        recourse_res = self.solver.find_recourse(
            model, instance, target_nodes, y_desired, background_data, 
            immutable_features, job_type, max_iter=max_iter, 
            df_diag=df_diag, pairwise_matrix=pairwise_matrix
        )
        
        return {
            'diagnosis_table': df_diag,
            'pairwise_interaction_matrix': pairwise_matrix,
            'target_nodes': target_nodes,
            'recourse_action': recourse_res['best_action'],
            'counterfactual_instance': recourse_res['x_counterfactual'],
            'minimum_cost': recourse_res['minimum_cost'],
            'sparsity_metrics': compute_sparsity(instance, recourse_res['x_counterfactual']) if recourse_res['x_counterfactual'] is not None else {},
            'causal_plausibility': compute_causal_plausibility(recourse_res['x_counterfactual'], self.scm) if recourse_res['x_counterfactual'] is not None else float('nan'),
            'evaluations': recourse_res['evaluations'],
            'elapsed_time': recourse_res['elapsed_time'],
            'success': recourse_res['success']
        }

# =========================================================================================
# 6. Visualization Routines (Including New Interaction Graph G_I)
# =========================================================================================

def plot_interaction_graph_g_I(
    df_diag: pd.DataFrame, 
    pairwise_matrix: pd.DataFrame, 
    threshold: float = 0.01,
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Visualizes the Interaction Graph (G_I = (T, E_I)) mapping synergistic structural bottlenecks.
    """
    plt.close('all')
    G = nx.Graph()
    
    # 1. Add Nodes (Target Features) and their Importance (Node Size)
    for feat in pairwise_matrix.index:
        if feat in df_diag.index:
            importance = df_diag.loc[feat, 'Total Causal Importance']
            G.add_node(feat, importance=importance)
            
    # 2. Add Edges (Pairwise Interactions C_G4(X, Y))
    for i, f1 in enumerate(pairwise_matrix.index):
        for j in range(i + 1, len(pairwise_matrix.columns)):
            f2 = pairwise_matrix.columns[j]
            weight = pairwise_matrix.loc[f1, f2]
            if abs(weight) > threshold:
                G.add_edge(f1, f2, weight=weight)

    pos = nx.spring_layout(G, k=1.2, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw nodes
    importances = [G.nodes[n]['importance'] for n in G.nodes]
    node_sizes = [max(800, imp * 5000) for imp in importances]
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=importances, cmap=plt.cm.YlOrRd, edgecolors='gray', linewidths=1.5, ax=ax)
    
    # Draw node labels
    node_labels = {n: f"{n}\n({G.nodes[n]['importance']:.2f})" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight='bold', ax=ax)

    # Draw edges and edge labels
    edges = G.edges(data=True)
    if len(edges) > 0:
        weights = [d['weight'] for u, v, d in edges]
        max_w = max(abs(w) for w in weights) if weights else 1.0
        edge_widths = [(abs(w) / max_w) * 8 + 1 for w in weights]
        edge_colors = ['#FF4500' if w > 0 else '#1E90FF' for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6, ax=ax)
        
        edge_labels_dict = {(u, v): f"{d['weight']:.3f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_size=9, font_color='#8B0000', font_weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1), ax=ax)

    # Aesthetics
    plt.title("A-CBFI Interaction Graph $\mathcal{G}_I$\n(Nodes: C_G1 + C_G4 Importance | Edges: Pairwise Synergy C_G4(X, Y))", fontsize=14, fontweight='bold', pad=20)
    cbar = plt.colorbar(nodes, ax=ax, shrink=0.7)
    cbar.set_label('Node Importance (Magnitude)', rotation=270, labelpad=15)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

    return fig

# (Other existing visualization functions like plot_main_interaction_decomposition, plot_causal_recourse_waterfall remain unchanged...)