"""
========================================================================================
Actionable-CBFI Visualization Module: Visualize_ACBFI.py
========================================================================================
Contains all publication-ready visualization routines and interactive plotting tools
for Actionable-CBFI (Causal Bottleneck Feature Importance & Actionable Recourse):

1. plot_main_interaction_decomposition: Stacked horizontal bar chart (C-G1 & C-G4)
2. plot_causal_recourse_waterfall: Step-by-step probability drop waterfall chart
3. plot_base_diagnosis_waterfall: Pure Factual prediction rejection risk feature contribution waterfall plot
4. plot_causal_diagnosis_matrix: 2D Causal Main/Interaction heatmap matrix
5. plot_interaction_graph: Causal interaction network graph (G_I)
6. visualize_feature_contribution: Local feature importance bar chart
7. visualize_local_pairwise_interaction: Pairwise interaction strength chart
8. visualize_feature_interaction_graph: Case-based interaction graph
9. visualize_draggable_interaction_graph: Interactive draggable graph layout
10. visualize_causal_recourse_path: SCM DAG recourse propagation path
========================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, List, Tuple, Dict, Union
from itertools import combinations
from sklearn.neighbors import NearestNeighbors

from Actionable_CBFI import StructuralCausalModel


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
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    df_plot = df_diag.sort_values(by='Total Causal Importance', ascending=True).copy()

    features = list(df_plot.index)
    cg1_values = df_plot['C-G1 (Main Effect)'].values
    cg4_values = df_plot['C-G4 (Interaction)'].values

    y_pos = np.arange(len(features))
    bar_height = 0.55

    min_val = float(min(np.min(cg1_values), 0.0))
    max_val = float(max(np.max(cg1_values + cg4_values), 0.1))

    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.5, zorder=4)

    for i, (cg1, cg4) in enumerate(zip(cg1_values, cg4_values)):
        if cg1 < 0:
            ax.barh(i, cg1, height=bar_height, left=0, color='#1f77b4', edgecolor='black', alpha=0.85, zorder=3, label='C-G1: Causal Main Effect' if i==0 else "")
            total_net = cg1 + cg4
            if total_net > 0:
                ax.barh(i, total_net, height=bar_height, left=0, color='#ff7f0e', edgecolor='black', alpha=0.85, zorder=2, label='C-G4: Causal Interaction Effect' if i==0 else "")
        else:
            ax.barh(i, cg1, height=bar_height, left=0, color='#1f77b4', edgecolor='black', alpha=0.85, zorder=2, label='C-G1: Causal Main Effect' if i==0 else "")
            if cg4 < 0:
                ax.barh(i, cg4, height=bar_height*0.70, left=cg1, color='#ff7f0e', edgecolor='black', hatch='//', alpha=0.90, zorder=3, label='C-G4: Negative Interaction Interference' if i==0 else "")
            else:
                ax.barh(i, cg4, height=bar_height, left=cg1, color='#ff7f0e', edgecolor='black', alpha=0.85, zorder=2, label='C-G4: Causal Interaction Effect' if i==0 else "")

    y_labels = []
    for feat in features:
        is_actionable = df_plot.loc[feat, 'Actionable']
        if not is_actionable:
            y_labels.append(f"★ {feat}")
        else:
            y_labels.append(feat)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight='bold')

    for i, (cg1, cg4) in enumerate(zip(cg1_values, cg4_values)):
        total = cg1 + cg4
        if abs(total) > 0.0001:
            text_x = total + (max_val * 0.015) if total >= 0 else 0.015
            val_str = f"{total:+.4f}" if cg1 < 0 else f"{total:.4f}"
            ax.text(text_x, i, val_str, va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Causal Importance Score (Probability Delta Shift via SCM Do-Interventions)', fontsize=12, fontweight='bold')
    
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
    counterfactual: Optional[pd.Series] = None,
    best_action: Optional[Dict[str, float]] = None,
    immutable_features: Optional[List[str]] = None,
    prediction_label: str = "Loan Denied (Class 1)",
    target_label: str = "Loan Approval (Class 0)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_plot: bool = False
) -> plt.Figure:
    """
    Visualizes the step-by-step Causal Recourse Waterfall Chart.
    Traces how factual prediction probability drops step-by-step as each diagnosed 
    bottleneck target feature is intervened on via SCM do-calculus.
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    instance_df = pd.DataFrame([instance])
    has_proba = hasattr(model, 'predict_proba')
    factual_pred = model.predict(instance_df)[0]
    
    if has_proba:
        factual_prob = float(model.predict_proba(instance_df)[0, factual_pred])
    else:
        factual_prob = 1.0

    steps = ['Factual']
    probs = [factual_prob]
    deltas = [0.0]

    current_interventions = {}

    for node in target_nodes:
        if best_action is not None and isinstance(best_action, dict) and node in best_action:
            val = float(best_action[node])
        elif counterfactual is not None and isinstance(counterfactual, pd.Series) and node in counterfactual.index:
            val = float(counterfactual[node])
        elif node in background_data.columns:
            val = float(background_data[node].median())
        else:
            val = float(instance[node] * 0.7)

        current_interventions[node] = val
        x_do = scm.forward_pass(instance, current_interventions)
        
        if has_proba:
            p_do = float(model.predict_proba(pd.DataFrame([x_do]))[0, factual_pred])
        else:
            p_do = 0.0 if model.predict(pd.DataFrame([x_do]))[0] != factual_pred else 1.0

        drop = probs[-1] - p_do
        steps.append(f"do({node})")
        probs.append(p_do)
        deltas.append(-drop)

    # Use exact counterfactual probability if counterfactual vector is provided
    if counterfactual is not None and has_proba:
        p_cf_exact = float(model.predict_proba(pd.DataFrame([counterfactual]))[0, factual_pred])
        steps.append('Counterfactual')
        probs.append(p_cf_exact)
        deltas.append(0.0)
    else:
        steps.append('Counterfactual')
        probs.append(probs[-1])
        deltas.append(0.0)

    n_steps = len(steps)
    bar_width = 0.5

    for i in range(n_steps):
        if i == 0:
            ax.bar(i, probs[i], width=bar_width, color='#d62728', edgecolor='black', alpha=0.85)
            ax.text(i, probs[i] + 0.02, f"{probs[i]*100:.1f}%", ha='center', fontsize=11, fontweight='bold', color='#d62728')
        elif i == n_steps - 1:
            ax.bar(i, probs[i], width=bar_width, color='#2ca02c', edgecolor='black', alpha=0.85)
            ax.text(i, probs[i] + 0.02, f"{probs[i]*100:.1f}%", ha='center', fontsize=11, fontweight='bold', color='#2ca02c')
        else:
            bottom_val = min(probs[i-1], probs[i])
            height_val = abs(probs[i-1] - probs[i])
            ax.bar(i, height_val, bottom=bottom_val, width=bar_width, color='#1f77b4', edgecolor='black', alpha=0.85)
            sign_str = "-" if probs[i-1] >= probs[i] else "+"
            ax.text(i, bottom_val + (height_val / 2.0), f"{sign_str}{height_val*100:.1f}%p", ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            ax.plot([i - 0.5, i + 0.5], [probs[i-1], probs[i-1]], color='gray', linestyle='--', linewidth=1.2)

    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(steps, fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Prediction Probability of Adverse Class', fontsize=12, fontweight='bold')
    ax.set_title(f'Actionable-CBFI: Causal Recourse Waterfall Chart\nFactual Prediction: {prediction_label} -> {target_label}', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, max(probs) * 1.18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Causal Recourse Waterfall plot saved to '{save_path}'.")

    if show_plot:
        plt.show()

    return fig


def plot_base_diagnosis_waterfall(
    df_diag: Optional[pd.DataFrame] = None,
    instance: Optional[pd.Series] = None,
    model = None,
    scm: Optional[StructuralCausalModel] = None,
    background_data: Optional[pd.DataFrame] = None,
    base_value: Optional[float] = None,
    prediction_label: Optional[str] = None,
    factual_prob: Optional[float] = None,
    immutable_features: Optional[List[str]] = None,
    max_display: int = 10,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10.5, 7.0),
    show_plot: bool = False
) -> plt.Figure:
    """
    Renders a Pure Factual Prediction Rejection Risk Waterfall Plot (SHAP-style).
    
    Explains purely how much each feature's value in a factual instance contributed
    to increasing (+ risk factor) or decreasing (- protective factor) the model's
    factual prediction rejection risk probability, starting from the baseline
    population risk E[f(X)] up to the final factual risk f(x).

    Parameters:
    -----------
    df_diag : pd.DataFrame, optional
        Diagnosis DataFrame produced by CausalCBFIDiagnoser.
    instance : pd.Series, optional
        Feature values for the target instance to display in 'Feature = Value' format.
    model : object, optional
        ML model for prediction.
    scm : StructuralCausalModel, optional
        Fitted SCM model.
    background_data : pd.DataFrame, optional
        Background data for baseline risk estimation.
    base_value : float, optional
        Baseline expected rejection risk E[f(X)]. If None, estimated from background data or default.
    prediction_label : str, optional
        Text label for factual prediction outcome (e.g. "Rejection Risk / Class 1").
    factual_prob : float, optional
        Factual prediction rejection probability / output score f(x).
    max_display : int, default 10
        Maximum number of feature contributions to display individually.
    save_path : str, optional
        Path to save the generated figure.
    figsize : Tuple[int, int], default (10.5, 7.0)
        Figure size.
    show_plot : bool, default False
        Whether to display the plot immediately.

    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    plt.close('all')

    # 1. Obtain df_diag if not provided directly
    if df_diag is None:
        if model is not None and scm is not None and instance is not None and background_data is not None:
            from Actionable_CBFI import CausalCBFIDiagnoser
            diagnoser = CausalCBFIDiagnoser(scm)
            df_diag, _ = diagnoser.diagnose_instance(model, instance, background_data)
        else:
            raise ValueError("Either 'df_diag' or ('model', 'scm', 'instance', 'background_data') must be provided.")

    # 2. Extract feature contributions (C-G1 Main Effect & C-G4 Interaction Synergy)
    features_list = list(df_diag.index)
    feat_contrib_tuples = []

    has_cg1_cg4 = ('C-G1 (Main Effect)' in df_diag.columns and 'C-G4 (Interaction)' in df_diag.columns)
    
    for feat in features_list:
        if has_cg1_cg4:
            cg1 = float(df_diag.loc[feat, 'C-G1 (Main Effect)'])
            cg4 = float(df_diag.loc[feat, 'C-G4 (Interaction)'])
            total = cg1 + cg4
        elif 'Total Causal Importance' in df_diag.columns:
            cg1 = float(df_diag.loc[feat, 'Total Causal Importance'])
            cg4 = 0.0
            total = cg1
        else:
            numeric_cols = df_diag.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                cg1 = float(df_diag.loc[feat, numeric_cols[0]])
                cg4 = 0.0
                total = cg1
            else:
                raise ValueError("df_diag does not contain numeric importance columns.")
        feat_contrib_tuples.append((feat, cg1, cg4, total))

    # 3. Sort features by absolute total contribution magnitude descending
    feat_contrib_tuples.sort(key=lambda x: abs(x[3]), reverse=True)

    # 4. Handle max_display capping
    if len(feat_contrib_tuples) > max_display:
        top_tuples = feat_contrib_tuples[:max_display - 1]
        other_tuples = feat_contrib_tuples[max_display - 1:]
        other_cg1 = sum(p[1] for p in other_tuples)
        other_cg4 = sum(p[2] for p in other_tuples)
        other_total = sum(p[3] for p in other_tuples)
        display_tuples = top_tuples + [(f"+ {len(other_tuples)} other features", other_cg1, other_cg4, other_total)]
    else:
        display_tuples = feat_contrib_tuples

    # SHAP waterfall layout puts smallest magnitude/other at bottom (y=0) and largest at top (y=N-1)
    display_tuples.reverse()

    # 5. Determine base_value (E[f(X)]) and factual_prob (f(x))
    if base_value is None:
        if model is not None and background_data is not None and hasattr(model, 'predict_proba'):
            try:
                base_value = float(model.predict_proba(background_data)[:, 1].mean())
            except Exception:
                base_value = 0.35
        else:
            base_value = 0.35

    total_shift = sum(p[3] for p in display_tuples)
    if factual_prob is None:
        if model is not None and instance is not None and hasattr(model, 'predict_proba'):
            try:
                factual_prob = float(model.predict_proba(pd.DataFrame([instance]))[0, 1])
            except Exception:
                factual_prob = base_value + total_shift
        else:
            factual_prob = base_value + total_shift

    # 6. Plotting Base Diagnosis Waterfall Chart
    fig, ax = plt.subplots(figsize=figsize)
    n_display = len(display_tuples)
    y_pos = np.arange(n_display)
    bar_height = 0.55

    # Format y-tick labels (Feature = Value), adding '★ ' prefix for Immutable Features
    y_labels = []
    for feat, cg1, cg4, total in display_tuples:
        is_imm = False
        if immutable_features is not None and feat in immutable_features:
            is_imm = True
        elif df_diag is not None and 'Actionable' in df_diag.columns and feat in df_diag.index:
            is_imm = not bool(df_diag.loc[feat, 'Actionable'])

        prefix = "★ " if is_imm else ""

        if instance is not None and feat in instance.index:
            val = instance[feat]
            if isinstance(val, (float, np.floating)):
                val_str = f"{val:,.2f}" if abs(val) >= 1000 else f"{val:.4g}"
            else:
                val_str = str(val)
            y_labels.append(f"{prefix}{feat} = {val_str}")
        else:
            y_labels.append(f"{prefix}{feat}")

    # Track running total for waterfall steps
    running_x = base_value
    x_positions = [running_x]

    # Plot horizontal bars for C-G1 (Main Effect) and C-G4 (Interaction Effect)
    for i, (feat, cg1, cg4, total) in enumerate(display_tuples):
        start_x = running_x

        # Step A: C_G1 bar (Main Effect)
        if abs(cg1) > 1e-6:
            left1 = min(start_x, start_x + cg1)
            width1 = abs(cg1)
            # Increase risk -> Red (#ff0051), Decrease risk -> Blue (#008bfb)
            color1 = '#ff0051' if cg1 >= 0 else '#008bfb'
            ax.barh(i, width1, left=left1, height=bar_height, color=color1, edgecolor='black', alpha=0.85, zorder=3)

        # Step B: C_G4 bar (Interaction Effect)
        mid_x = start_x + cg1
        if abs(cg4) > 1e-6:
            left2 = min(mid_x, mid_x + cg4)
            width2 = abs(cg4)
            # Increase risk -> Orange (#ff7f0e), Decrease risk -> Sky Blue (#00bfff)
            color2 = '#ff7f0e' if cg4 >= 0 else '#00bfff'
            ax.barh(i, width2, left=left2, height=bar_height, color=color2, edgecolor='black', hatch='//' if cg4 < 0 else None, alpha=0.90, zorder=4)

        end_x = start_x + cg1 + cg4
        running_x = end_x
        x_positions.append(running_x)

        # Connect step i to i+1 with dotted line
        if i > 0:
            prev_x = x_positions[i]
            ax.plot([prev_x, prev_x], [i - 1 + bar_height/2, i - bar_height/2], color='gray', linestyle='--', linewidth=1.2, zorder=2)

        # Annotate text on/near bar
        if abs(cg4) > 1e-4:
            val_str = f"{total:+.4f} (G1:{cg1:+.3f}, G4:{cg4:+.3f})"
        else:
            val_str = f"{total:+.4f}" if abs(total) < 1.0 else f"{total:+.2f}"

        # Determine alignment and placement
        extreme_x_max = max(start_x, mid_x, end_x)
        extreme_x_min = min(start_x, mid_x, end_x)
        if total >= 0:
            text_x = extreme_x_max + 0.005
            ha = 'left'
            color_text = '#d9381e' if cg4 > 0.01 else '#ff0051'
        else:
            text_x = extreme_x_min - 0.005
            ha = 'right'
            color_text = '#008bfb'

        ax.text(text_x, i, val_str, va='center', ha=ha, fontsize=9.5, fontweight='bold', color=color_text)

    top_x = x_positions[-1]

    # Reference lines for base value E[f(X)] and final factual risk f(x)
    ax.axvline(base_value, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    ax.axvline(top_x, color='gray', linestyle=':', linewidth=1.2, zorder=1)

    # Labels for baseline E[f(X)] and final f(x)
    ax.text(base_value, -0.7, f"Base Risk E[f(X)] = {base_value:.3f}", ha='center', va='top', fontsize=11, fontweight='bold', color='gray')
    ax.text(top_x, n_display - 0.3 + bar_height, f"Factual Risk f(x) = {top_x:.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ff0051' if top_x >= base_value else '#008bfb')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight='bold')
    ax.set_xlabel(r'Factual Rejection Risk Probability $P(y = \text{Adverse Class})$', fontsize=12, fontweight='bold')

    main_title = "Factual Prediction Rejection Risk Waterfall (Base Diagnosis)"
    if prediction_label is not None:
        title_str = f"{main_title}\nC-G1 Main vs. C-G4 Interaction Breakdown for {prediction_label}"
    else:
        title_str = f"{main_title}\nC-G1 Main vs. C-G4 Interaction Breakdown for Factual Rejection Risk"

    ax.set_title(title_str, fontsize=13, fontweight='bold', pad=25)
    ax.grid(True, axis='x', linestyle='--', alpha=0.4, zorder=0)

    # Set x-limits with margin
    all_x = x_positions + [base_value, top_x]
    min_x_val = min(all_x)
    max_x_val = max(all_x)
    x_range = max_x_val - min_x_val
    margin = max(x_range * 0.18, 0.06)
    ax.set_xlim(min_x_val - margin, max_x_val + margin)
    ax.set_ylim(-1.0, n_display + 0.5)

    # Legend specifying C_G1 & C_G4 colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff0051', edgecolor='black', label=r'$C_{G1}$ Main Effect (+ Risk Factor)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label=r'$C_{G4}$ Interaction Synergy (+ Risk Factor)'),
        Patch(facecolor='#008bfb', edgecolor='black', label=r'$C_{G1}$ Main Effect (- Protective Factor)'),
        Patch(facecolor='#00bfff', edgecolor='black', label=r'$C_{G4}$ Interaction Interference (- Protective Factor)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5, framealpha=0.95)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Base Diagnosis Waterfall plot saved to '{save_path}'.")

    if show_plot:
        plt.show()

    return fig


# Aliases for backwards compatibility & convenience
plot_causal_diagnosis_waterfall = plot_base_diagnosis_waterfall
plot_shap_style_waterfall = plot_base_diagnosis_waterfall


def plot_causal_diagnosis_matrix(
    df_diag: pd.DataFrame,
    prediction_label: str = "Credit Denied (High Risk / Class 1)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 7),
    show_plot: bool = False
) -> plt.Figure:
    """
    Pure Diagnosis Visualization Tool: Causal Main & Interaction Heatmap Matrix.
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    df_plot = df_diag.sort_values(by='Total Causal Importance', ascending=False).copy()
    features = list(df_plot.index[:8])

    n_feats = len(features)
    matrix = np.zeros((n_feats, n_feats))

    for i, f1 in enumerate(features):
        cg1 = df_plot.loc[f1, 'C-G1 (Main Effect)']
        cg4 = df_plot.loc[f1, 'C-G4 (Interaction)']
        matrix[i, i] = cg1
        for j, f2 in enumerate(features):
            if i != j:
                matrix[i, j] = cg4 / (n_feats - 1)

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    for i in range(n_feats):
        for j in range(n_feats):
            val = matrix[i, j]
            if val > 0.0001:
                font_weight = 'bold' if i == j else 'normal'
                color_text = 'white' if val > (matrix.max() * 0.6) else 'black'
                ax.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=9.5, fontweight=font_weight, color=color_text)

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


def plot_interaction_graph(
    df_diag: pd.DataFrame,
    interaction_matrix: Optional[pd.DataFrame] = None,
    threshold: float = 0.05,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (9.5, 8.0),
    show_plot: bool = False
) -> plt.Figure:
    """
    Renders NetworkX graph visualization G_I of feature interaction topology.
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)

    G = nx.Graph()
    for feat, row in df_diag.iterrows():
        total_imp = row['Total Causal Importance']
        is_actionable = row['Actionable']
        G.add_node(feat, importance=total_imp, actionable=is_actionable)

    if interaction_matrix is not None:
        for u in interaction_matrix.index:
            for v in interaction_matrix.columns:
                if u != v and u in G.nodes and v in G.nodes:
                    w = interaction_matrix.loc[u, v]
                    if abs(w) >= threshold and not G.has_edge(u, v):
                        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G, k=1.8, seed=42)

    importances = [G.nodes[n]['importance'] for n in G.nodes]
    max_imp = max(abs(i) for i in importances) if importances else 1.0
    node_sizes = [max(800, (abs(G.nodes[n]['importance']) / max_imp) * 3500) for n in G.nodes]
    node_colors = [G.nodes[n]['importance'] for n in G.nodes]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.YlOrRd, edgecolors='gray', linewidths=1.5, ax=ax)

    labels = {n: f"★ {n}" if not G.nodes[n]['actionable'] else n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', ax=ax)

    edges = G.edges(data=True)
    if len(edges) > 0:
        weights = [d['weight'] for u, v, d in edges]
        max_w = max(abs(w) for w in weights) if weights else 1.0
        edge_widths = [(abs(w) / max_w) * 8 + 1 for w in weights]
        edge_colors = ['#FF4500' if w > 0 else '#1E90FF' for w in weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6, ax=ax)
        edge_labels_dict = {(u, v): f"{d['weight']:.3f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_dict, font_size=9, font_color='#8B0000', font_weight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1), ax=ax)

    plt.title(r"A-CBFI Interaction Graph $\mathcal{G}_I$" + "\n(Nodes: C_G1 + C_G4 Importance | Edges: Pairwise Synergy C_G4(X, Y))", fontsize=14, fontweight='bold', pad=20)
    cbar = plt.colorbar(nodes, ax=ax, shrink=0.7)
    cbar.set_label('Node Importance (Magnitude)', rotation=270, labelpad=15)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Interaction Graph (G_I) plot saved to '{save_path}'.")
        
    if show_plot:
        plt.show()

    return fig


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


def visualize_causal_recourse_path(
    scm: StructuralCausalModel, 
    factual: pd.Series, 
    counterfactual: pd.Series, 
    action_nodes: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = False
) -> plt.Figure:
    """
    Visualizes the SCM DAG highlighting direct intervention nodes and downstream propagated features.
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 7))

    pos = nx.spring_layout(scm.graph, seed=42)
    node_colors = []
    
    for n in scm.graph.nodes:
        if n in action_nodes:
            node_colors.append('#ff7f0e') # Direct Action (Orange)
        elif abs(counterfactual[n] - factual[n]) > 1e-4:
            node_colors.append('#2ca02c') # Passive Ripple (Green)
        else:
            node_colors.append('#1f77b4') # Unchanged (Blue)

    nx.draw_networkx_nodes(scm.graph, pos, node_color=node_colors, node_size=1500, edgecolors='black', ax=ax)
    
    labels = {n: f"{n}\n({factual[n]:.2f} -> {counterfactual[n]:.2f})" for n in scm.graph.nodes}
    nx.draw_networkx_labels(scm.graph, pos, labels=labels, font_size=9, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(scm.graph, pos, arrowstyle='->', arrowsize=20, edge_color='gray', ax=ax)

    plt.title("Actionable CBFI: Causal Recourse Propagation Path\n(Orange: Direct Action L_active, Green: Downstream Causal Change)", fontsize=13, fontweight='bold', pad=15)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Causal Recourse Propagation Path (DAG) plot saved to '{save_path}'.")

    if show_plot:
        plt.show()

    return fig