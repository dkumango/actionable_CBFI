# Actionable CBFI (A-CBFI)

![Python Version](https://img.shields.io/badge/python-3.14.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Paper Status](https://img.shields.io/badge/paper-submitted-yellow.svg)

**Actionable Case-Based Feature Importance (A-CBFI)** is a diagnosis-prescription integrated framework for tabular machine learning that unifies localized structural explanation with causal counterfactual recourse[cite: 2]. 

This repository provides the official implementation of the A-CBFI framework, which builds upon the foundational [Localized-CBFI](https://github.com/dkumango/Local_CBFI). By mathematically separating the active user intervention space ($L_{active}$) from the total downstream manifold modifications, A-CBFI shifts the algorithmic recourse paradigm from exhaustive search to targeted, root-cause prescription[cite: 2].

## 📖 Overview

Existing attribution-based explanations (e.g., SHAP, LIME) are fundamentally descriptive and often fail to provide actionable guidance for users receiving adverse predictions. Furthermore, purely optimization-based counterfactual methods (e.g., Wachter's CE, DiCE) ignore causal dependencies or enforce impractical "diffuse shifts" across numerous features, severely increasing cognitive burden.

**A-CBFI** overcomes these limitations by:
1. **Discovering Synergistic Bottlenecks & Suppressive Locks:** It explicitly isolates critical high-order synergistic interactions ($C_{G4} > 0$) that amplify adverse predictions and releases suppressive structural locks ($C_{G4} < 0$)[cite: 2].
2. **Targeted Causal Intervention via IGBS:** It leverages Structural Causal Models (SCMs) to constrain the counterfactual search space exclusively to the diagnosed root causes, utilizing Interaction-Guided Beam Search (IGBS) to prioritize structurally informative branches[cite: 2].
3. **Actionable Prescription:** It outputs clear, highly targeted directives for end-users while allowing the SCM to automatically handle downstream topological ripple effects[cite: 2].

## ✨ Key Features & Contributions

* **Cognitive Burden Compression via Surgical Intervention:** A-CBFI compresses the active human intervention burden down to an average of 1.72 levers—a 76.9% reduction compared to exhaustive baselines[cite: 2]. By concentrating over 98.3% of the intervention effort strictly on the diagnosed root causes, it ensures highly targeted interventions while maintaining global cost parity[cite: 2].
* **Overcoming the Additive Fallacy:** Unlike additive-guided methods (e.g., SHAP-Targeted CE) that ignore higher-order feature synergies, leading to suboptimal predictive momentum and diffuse intervention effort in complex non-linear boundaries (e.g., XGBoost), A-CBFI successfully navigates non-linear manifolds by targeting true structural bottlenecks[cite: 2].
* **Causal Validity:** Grounded in Pearl's do-calculus, A-CBFI preserves real-world chronological and domain constraints, achieving a 100% relative causal convergence rate across all causally feasible instances[cite: 2].

## ⚙️ Installation

```bash
# Clone the repository
git clone [https://github.com/dkumango/Actionable_CBFI.git](https://github.com/dkumango/Actionable_CBFI.git)
cd Actionable_CBFI

# Install dependencies (requires Python 3.14.0+)
pip install -r requirements.txt

Core dependencies include pandas, numpy, scikit-learn, networkx, and matplotlib.🚀 Quick StartThe core causal propagation engine is implemented in Actionable_CBFI.py. Below is a minimal example of diagnosing an adverse prediction and generating targeted recourse.Pythonimport pandas as pd
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
📊 Benchmarking SuiteTo reproduce the multi-architecture experiments across Financial and Healthcare domains, run the automated evaluation suite:Bashpython test_benchmark_comparison.py --dataset financial_loan --model xgboost
This suite compares A-CBFI against Untargeted Causal, Wachter's CE, and SHAP-Targeted CE evaluating Recourse Cost, Sparsity ($L_0$), Active Levers ($L_{active}$), Recourse Concentration Ratio (RCR), and Causal Plausibility MSE ($R_{SCM}$)[cite: 2].📁 Repository StructureActionable_CBFI.py: Core implementation of the diagnosis and SCM forward propagation engine via IGBS.test_benchmark_comparison.py: Automated benchmarking and evaluation script.data/: Sample preprocessing scripts for the 6 benchmark datasets (Financial Loan, German Credit, Adult Income, Medical Insurance, Pima Diabetes, Breast Cancer).configs/: Domain-specific SCM DAG topological specifications.📝 CitationIf you find this framework useful in your research, please consider citing our paper:코드 스니펫@article{under review
}
📬 ContactFor any questions, discussions, or collaboration inquiries regarding Explainable AI (XAI) or Tabular Deep Learning, please open an issue or contact the author.
