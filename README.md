# Actionable CBFI (A-CBFI)

![Python Version](https://img.shields.io/badge/python-3.14.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Paper Status](https://img.shields.io/badge/paper-submitted-yellow.svg)

**Actionable Case-Based Feature Importance (A-CBFI)** is a diagnosis-prescription integrated framework for tabular machine learning that unifies localized structural explanation with causal counterfactual recourse. 

This repository provides the official implementation of the A-CBFI framework, which builds upon the foundational [Localized-CBFI](https://github.com/dkumango/Local_CBFI). By mathematically separating the active user intervention space ($L_{active}$) from the total downstream manifold modifications, A-CBFI shifts the algorithmic recourse paradigm from exhaustive search to targeted, root-cause prescription.

## 📖 Overview

Existing attribution-based explanations (e.g., SHAP, LIME) are fundamentally descriptive and often fail to provide actionable guidance for users receiving adverse predictions. Furthermore, purely optimization-based counterfactual methods (e.g., Wachter's CE, DiCE) ignore causal dependencies or enforce impractical "diffuse shifts" across numerous features, severely increasing cognitive burden.

**A-CBFI** overcomes these limitations by:
1. **Discovering Synergistic Bottlenecks:** It isolates critical high-order synergistic interactions (C_G4) that amplify adverse predictions or suppress positive outcomes.
2. **Targeted Causal Intervention:** It leverages Structural Causal Models (SCMs) to constrain the counterfactual search space exclusively to the diagnosed root causes.
3. **Actionable Prescription:** It outputs clear, minimal-effort directives for end-users while allowing the SCM to automatically handle downstream topological ripple effects.

## ✨ Key Features & Contributions

* **Cognitive Burden Compression via "Surgical Push":** A-CBFI compresses the active human intervention burden down to an average of 1.85 levers—a 75.3% reduction compared to exhaustive baselines—while maintaining global cost parity.
* **Overcoming the Additive Fallacy:** Unlike additive-guided methods (e.g., SHAP-Targeted CE) that fail in complex non-linear boundaries like XGBoost, A-CBFI successfully navigates non-linear manifolds by targeting true synergistic drivers.
* **Causal Validity:** Grounded in Pearl's do-calculus, A-CBFI preserves real-world chronological and domain constraints, achieving a >99.8% relative causal convergence rate across physically feasible instances.

## ⚙️ Installation

```bash
# Clone the repository
git clone [https://github.com/dkumango/Actionable_CBFI.git](https://github.com/dkumango/Actionable_CBFI.git)
cd Actionable_CBFI

# Install dependencies (requires Python 3.14.0+)
pip install -r requirements.txt
```
*Core dependencies include `pandas`, `numpy`, `scikit-learn`, `networkx`, and `matplotlib`.*

## 🚀 Quick Start

The core causal propagation engine is implemented in `Actionable_CBFI.py`. Below is a minimal example of diagnosing an adverse prediction and generating targeted recourse.

```python
import pandas as pd
from Actionable_CBFI import ACBFI_Engine, SCM_Model
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data & Train Target Model
data = pd.read_csv('data/financial_loan.csv')
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Build SCM with Domain Priors (LLM-Guided DAG)
scm = SCM_Model(
    immutable_features=['annual_income', 'emp_length'],
    dag_edges=[('annual_income', 'dti'), ('loan_amount', 'installment')]
)
scm.fit_from_data(X_train) # Fit structural equations

# 3. Initialize A-CBFI Engine
explainer = ACBFI_Engine(predictive_model=model, scm=scm)

# 4. Diagnose and Prescribe Recourse for a Rejected Instance
factual_instance = X_test.iloc[0]
diagnosis = explainer.diagnose_bottlenecks(factual_instance, target_class=0)

# The engine isolates synergistic targets (C_G4 > 0)
print(f"Diagnosed Bottlenecks (Target Set T): {diagnosis.target_features}")

# Generate actionable recourse restricted to Target Set T
recourse = explainer.generate_recourse(factual_instance, target_set=diagnosis.target_features)
print(recourse.actionable_prescription())
```

## 📊 Benchmarking Suite

To reproduce the multi-architecture experiments across Financial and Healthcare domains, run the automated evaluation suite:

```bash
python test_benchmark_comparison.py --dataset financial_loan --model xgboost
```
*This suite compares A-CBFI against `Untargeted Causal`, `Wachter's CE`, and `SHAP-Targeted CE` evaluating Recourse Cost, Sparsity ($L_0$), Active Levers ($L_{active}$), and Causal Plausibility MSE.*

## 📁 Repository Structure

* `Actionable_CBFI.py`: Core implementation of the diagnosis and SCM forward propagation engine.
* `test_benchmark_comparison.py`: Automated benchmarking and evaluation script.
* `data/`: Sample preprocessing scripts for the 6 benchmark datasets (Financial Loan, German Credit, Adult Income, Medical Insurance, Pima Diabetes, Breast Cancer).
* `configs/`: Domain-specific SCM DAG topological specifications.

## 📝 Citation

If you find this framework useful in your research, please consider citing our paper:

```bibtex
@article{under review
}
```

## 📬 Contact
For any questions, discussions, or collaboration inquiries regarding Explainable AI (XAI) or Tabular Deep Learning, please open an issue or contact the author.
