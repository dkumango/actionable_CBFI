# Actionable CBFI (A-CBFI)

![Python Version](https://img.shields.io/badge/python-3.14.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Paper Status](https://img.shields.io/badge/paper-submitted-yellow.svg)

**Actionable Case-Based Feature Importance (A-CBFI)** is a diagnosis-prescription integrated framework for tabular machine learning that unifies localized structural explanation with causal counterfactual recourse. 

This repository provides the official implementation of the A-CBFI framework, which builds upon the foundational [Localized-CBFI](https://github.com/dkumango/Local_CBFI). By mathematically separating the active user intervention space ($L_{active}$) from the total downstream manifold modifications, A-CBFI shifts the algorithmic recourse paradigm from exhaustive search to targeted, root-cause prescription.

## 📖 Overview

Existing attribution-based explanations (e.g., SHAP, LIME) are fundamentally descriptive and often fail to provide actionable guidance for users receiving adverse predictions. Furthermore, purely optimization-based counterfactual methods (e.g., Wachter's CE, DiCE) ignore causal dependencies or enforce impractical "diffuse shifts" across numerous features, severely increasing cognitive burden.

**A-CBFI** overcomes these limitations by:
1. **Discovering Synergistic Bottlenecks & Suppressive Locks:** It explicitly isolates critical high-order synergistic interactions ($C_{G4} > 0$) that amplify adverse predictions and releases suppressive structural locks ($C_{G4} < 0$).
2. **Targeted Causal Intervention via IGBS:** It leverages Structural Causal Models (SCMs) to constrain the counterfactual search space exclusively to the diagnosed root causes, utilizing Interaction-Guided Beam Search (IGBS) to prioritize structurally informative branches.
3. **Actionable Prescription:** It outputs clear, highly targeted directives for end-users while allowing the SCM to automatically handle downstream topological ripple effects.

## ✨ Key Features & Contributions

* **Cognitive Burden Compression via Surgical Intervention:** A-CBFI compresses the active human intervention burden down to an average of 1.72 levers—a 76.9% reduction compared to exhaustive baselines. By concentrating over 98.3% of the intervention effort strictly on the diagnosed root causes, it ensures highly targeted interventions while maintaining global cost parity.
* **Overcoming the Additive Fallacy:** Unlike additive-guided methods (e.g., SHAP-Targeted CE) that ignore higher-order feature synergies, leading to suboptimal predictive momentum and diffuse intervention effort in complex non-linear boundaries (e.g., XGBoost), A-CBFI successfully navigates non-linear manifolds by targeting true structural bottlenecks.
* **Causal Validity:** Grounded in Pearl's do-calculus, A-CBFI preserves real-world chronological and domain constraints, achieving a 100% relative causal convergence rate across all causally feasible instances.

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

# The engine isolates synergistic and suppressive targets (|C_G4| > 0)
print(f"Diagnosed Bottlenecks (Target Set T): {diagnosis.target_features}")

# Generate actionable recourse restricted to Target Set T using IGBS
recourse = explainer.generate_recourse(factual_instance, target_set=diagnosis.target_features)
print(recourse.actionable_prescription())
```

## 📊 Benchmarking Suite

To reproduce the multi-architecture experiments across Financial and Healthcare domains, run the automated evaluation suite:

```bash
python test_benchmark_comparison.py --dataset financial_loan --model xgboost
```
*This suite compares A-CBFI against `Untargeted Causal`, `Wachter's CE`, and `SHAP-Targeted CE` evaluating Recourse Cost, Sparsity ($L_0$), Active Levers ($L_{active}$), Recourse Concentration Ratio (RCR), and Causal Plausibility MSE ($R_{SCM}$).*

## 📁 Repository Structure

* `Actionable_CBFI.py`: Core implementation of the diagnosis and SCM forward propagation engine via IGBS.
* `Visualize_ACBFI.py`: Implementation of the various plots related with A-CBFI.   
* `test_benchmark_comparison.py`: Automated benchmarking and evaluation script.
* `DOMAIN_SCM_CONFIG.txt`: Domain-specific SCM DAG topological specifications for benchmark.
* `generate_case_study_1_healthcare.py`: Performs experiment for Case Study (1) in the paper
* `generate_case_study_2_finance.py`: Performs experiment for Case Study (2) in the paper
* `data/`: Sample preprocessing scripts for the 6 benchmark datasets (Financial Loan, German Credit, Adult Income, Medical Insurance, Pima Diabetes, Breast Cancer).
* `result_benchmark/`: Result of benchmark test
* `result_case_1/`: Result of generate_case_study_1_healthcare.py
* `result_case_2/`: Result of generate_case_study_2_finance.py

## 📝 Citation

If you find this framework useful in your research, please consider citing our paper:

```bibtex
@article{under review
}
```

## 📬 Contact
For any questions, discussions, or collaboration inquiries regarding Explainable AI (XAI) or Tabular Deep Learning, please open an issue or contact the author.
