"""
Corrected real-data protocol for the lever-vs-cause study.

Differences from the data preparation of the preliminary version (IGBS, arXiv:2608.27821v1)
(see exp_code/CHANGELOG.md):
- stratified 80:20 train/test split; models, scaler and SCM are fitted on train only
- queries are drawn at random from adverse TEST predictions (not the first 100 rows)
- actionability corrected: DPF, race, sex, marital status, credit history, int_rate,
  installment, region ... are not directly intervenable (they still change via the SCM)
- DAG edges are restricted to columns that exist (exercise_hours is absent from insurance)
- large datasets (financial_loan, adult) are capped at MAX_TRAIN training rows so that
  RBF-SVM training and exact searches stay tractable
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "synthetic"))
from synth_lib import SCM  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "original")
MAX_TRAIN = 8000

# name -> (non-intervenable features, DAG edges); names are the lower-cased column names
SPECS = {
    "financial_loan": (
        ["annual_income", "emp_length", "int_rate", "installment"],
        [("annual_income", "dti"), ("annual_income", "loan_amount"), ("emp_length", "int_rate"),
         ("loan_amount", "installment"), ("term", "installment"), ("loan_amount", "int_rate"),
         ("dti", "int_rate")],
    ),
    "german_credit": (
        ["age", "residence_history", "personal_status", "foreign_worker", "credit_history",
         "dependents"],
        [("age", "existing_credits"), ("age", "amount"), ("months_loan_duration", "amount"),
         ("amount", "installment_rate"), ("existing_credits", "amount")],
    ),
    "adult": (
        ["age", "education_num", "race", "sex", "marital_status"],
        [("age", "education_num"), ("age", "hours_per_week"), ("age", "capital_gain"),
         ("education_num", "hours_per_week"), ("education_num", "capital_gain"),
         ("hours_per_week", "capital_gain")],
    ),
    "medical_insurance": (
        ["age", "sex", "children", "region"],
        [("age", "bmi"), ("smoker", "bmi")],
    ),
    "diabetes": (
        ["age", "pregnancies", "diabetespedigreefunction"],
        [("age", "pregnancies"), ("age", "bloodpressure"), ("age", "glucose"),
         ("bmi", "bloodpressure"), ("bmi", "skinthickness"), ("bmi", "insulin"),
         ("glucose", "insulin")],
    ),
    "breast_cancer": (
        [],
        [("mean_radius", "mean_perimeter"), ("mean_radius", "mean_area"),
         ("mean_perimeter", "mean_area"), ("mean_radius", "mean_compactness"),
         ("mean_texture", "mean_smoothness")],
    ),
    # ---- E0b datasets (data/<name>/) -------------------------------------------------------
    # CDC Diabetes Health Indicators (BRFSS 2015, UCI 891). Directly actionable: BMI and
    # lifestyle behaviours only. Smoker = "ever smoked 100 cigarettes" (history, immutable).
    # Clinical conditions (HighBP, HighChol), self-rated health and screening behaviour
    # (CholCheck) are NOT directly intervenable; they respond through the SCM.
    "cdc_diabetes": (
        ["sex", "age", "education", "income", "smoker", "stroke", "heartdiseaseorattack",
         "anyhealthcare", "nodocbccost", "diffwalk", "genhlth", "menthlth", "physhlth",
         "highbp", "highchol", "cholcheck"],
        [("age", "highbp"), ("age", "highchol"), ("bmi", "highbp"), ("bmi", "highchol"),
         ("physactivity", "bmi"), ("hvyalcoholconsump", "highbp"), ("smoker", "highbp"),
         ("education", "income"), ("income", "physactivity"),
         ("bmi", "genhlth"), ("physactivity", "genhlth"), ("highbp", "genhlth"),
         ("age", "genhlth"), ("genhlth", "physhlth"), ("bmi", "diffwalk"), ("age", "diffwalk"),
         ("highbp", "heartdiseaseorattack"), ("highchol", "heartdiseaseorattack"),
         ("age", "heartdiseaseorattack"), ("smoker", "heartdiseaseorattack")],
    ),
    # FICO HELOC (OpenML 45553, cleaned). Directly actionable: current balances/burden,
    # utilisation, recent inquiries and new trades. Credit history, delinquency, file age and
    # the bureau risk score (ExternalRiskEstimate, a consequence of the others) are not.
    "heloc": (
        ["externalriskestimate", "msinceoldesttradeopen", "msincemostrecenttradeopen",
         "averageminfile", "numsatisfactorytrades", "numtrades60ever2derogpubrec",
         "numtrades90ever2derogpubrec", "percenttradesneverdelq", "msincemostrecentdelq",
         "maxdelq2publicreclast12m", "maxdelqever", "numtotaltrades", "percentinstalltrades",
         "msincemostrecentinqexcl7days", "numinqlast6mexcl7days", "percenttradeswbalance"],
        [("numrevolvingtradeswbalance", "percenttradeswbalance"),
         ("numinstalltradeswbalance", "percenttradeswbalance"),
         ("numtotaltrades", "percenttradeswbalance"),
         ("numinqlast6m", "numinqlast6mexcl7days"),
         ("numtradesopeninlast12m", "numinqlast6m"),
         ("netfractionrevolvingburden", "numbank2natltradeswhighutilization"),
         ("netfractionrevolvingburden", "externalriskestimate"),
         ("numinqlast6m", "externalriskestimate"),
         ("numbank2natltradeswhighutilization", "externalriskestimate"),
         ("percenttradesneverdelq", "externalriskestimate"),
         ("maxdelqever", "externalriskestimate"),
         ("averageminfile", "externalriskestimate")],
    ),
    # ---- Added datasets (2026-10-26, data/<name>/) ----------------------------------------
    # Taiwan credit card default (UCI 350). Repayment status (pay_*) and past bills are
    # history; the current bill and recent repayment amounts are actionable. Paying more
    # lowers the next bill and improves the latest repayment status (mediated lever).
    "taiwan_default": (
        ["limit_bal", "sex", "education", "marriage", "age", "pay_0", "pay_2", "pay_3",
         "bill_amt2"],
        [("age", "limit_bal"), ("education", "limit_bal"), ("limit_bal", "bill_amt1"),
         ("bill_amt2", "bill_amt1"), ("pay_amt2", "bill_amt1"), ("pay_amt1", "pay_0"),
         ("bill_amt1", "pay_0"), ("pay_2", "pay_0"), ("pay_3", "pay_2")],
    ),
    # Give Me Some Credit (OpenML 46929). Delinquency counts, age, dependents and real-estate
    # loans are not directly actionable; utilisation, debt ratio, income, open lines are.
    "gmsc": (
        ["age", "numberoftime30_59dayspastduenotworse", "numberoftimes90dayslate",
         "numberoftime60_89dayspastduenotworse", "numberofdependents",
         "numberrealestateloansorlines"],
        [("age", "monthlyincome"), ("age", "numberofopencreditlinesandloans"),
         ("monthlyincome", "debtratio"), ("numberofdependents", "debtratio"),
         ("numberofopencreditlinesandloans", "revolvingutilizationofunsecuredlines"),
         ("numberrealestateloansorlines", "debtratio")],
    ),
    # Student Performance, Portuguese course (UCI 320); adverse = final grade G3 < 10.
    # G1/G2 are dropped (leakage). Background, past failures and health are not actionable.
    "student_perf": (
        ["school", "sex", "age", "address", "famsize", "pstatus", "medu", "fedu", "failures",
         "health", "famrel", "romantic", "higher", "traveltime", "freetime"],
        [("freetime", "goout"), ("goout", "walc"), ("goout", "dalc"), ("walc", "absences"),
         ("dalc", "absences"), ("dalc", "health"), ("walc", "health"), ("medu", "higher"),
         ("higher", "studytime"), ("paid", "studytime"), ("address", "traveltime"),
         ("traveltime", "studytime")],
    ),
    # Predict Students' Dropout and Academic Success (UCI 697); adverse = Dropout.
    # 1st-semester results, background and macro-economy are not actionable; 2nd-semester
    # approvals/grades are outcomes that respond only through the SCM.
    "student_dropout": (
        ["marital_status", "course", "daytime", "prev_grade", "admission_grade", "displaced",
         "special_needs", "gender", "age_enroll", "international", "cu1_enrolled",
         "cu1_evaluations", "cu1_approved", "cu1_grade", "cu2_approved", "cu2_grade",
         "unemployment", "inflation", "gdp"],
        [("debtor", "tuition_up_to_date"), ("scholarship", "tuition_up_to_date"),
         ("tuition_up_to_date", "cu2_enrolled"), ("cu2_enrolled", "cu2_evaluations"),
         ("cu2_evaluations", "cu2_approved"), ("cu1_approved", "cu2_approved"),
         ("cu2_approved", "cu2_grade"), ("cu1_grade", "cu2_grade")],
    ),
}
EXT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DROPOUT_COLS = {
    "Marital Status": "marital_status", "Course": "course",
    "Daytime/evening attendance": "daytime", "Previous qualification (grade)": "prev_grade",
    "Admission grade": "admission_grade", "Displaced": "displaced",
    "Educational special needs": "special_needs", "Debtor": "debtor",
    "Tuition fees up to date": "tuition_up_to_date", "Gender": "gender",
    "Scholarship holder": "scholarship", "Age at enrollment": "age_enroll",
    "International": "international",
    "Curricular units 1st sem (enrolled)": "cu1_enrolled",
    "Curricular units 1st sem (evaluations)": "cu1_evaluations",
    "Curricular units 1st sem (approved)": "cu1_approved",
    "Curricular units 1st sem (grade)": "cu1_grade",
    "Curricular units 2nd sem (enrolled)": "cu2_enrolled",
    "Curricular units 2nd sem (evaluations)": "cu2_evaluations",
    "Curricular units 2nd sem (approved)": "cu2_approved",
    "Curricular units 2nd sem (grade)": "cu2_grade",
    "Unemployment rate": "unemployment", "Inflation rate": "inflation", "GDP": "gdp",
}
TAIWAN_COLS = {"X1": "limit_bal", "X2": "sex", "X3": "education", "X4": "marriage",
               "X5": "age", "X6": "pay_0", "X7": "pay_2", "X8": "pay_3", "X12": "bill_amt1",
               "X13": "bill_amt2", "X18": "pay_amt1", "X19": "pay_amt2", "X20": "pay_amt3"}

# Allowed change direction of directly actionable features (+1 increase only, -1 decrease
# only) and plausibility bounds in RAW units. Applied to every intervention (diagnosis and
# search) through ConstrainedSCM and to the non-causal baselines' candidate values.
DIRECTIONS = {
    "heloc": {"numtradesopeninlast12m": -1, "numinqlast6m": -1,
              "netfractionrevolvingburden": -1, "netfractioninstallburden": -1,
              "numrevolvingtradeswbalance": -1, "numinstalltradeswbalance": -1,
              "numbank2natltradeswhighutilization": -1},
    "cdc_diabetes": {"bmi": -1, "physactivity": +1, "fruits": +1, "veggies": +1,
                     "hvyalcoholconsump": -1},
    "taiwan_default": {"bill_amt1": -1, "pay_amt1": +1, "pay_amt2": +1, "pay_amt3": +1},
    "gmsc": {"revolvingutilizationofunsecuredlines": -1, "debtratio": -1,
             "monthlyincome": +1, "numberofopencreditlinesandloans": -1},
    "student_perf": {"studytime": +1, "absences": -1, "goout": -1, "dalc": -1, "walc": -1,
                     "paid": +1, "internet": +1, "schoolsup": +1, "famsup": +1},
    "student_dropout": {"debtor": -1, "tuition_up_to_date": +1, "scholarship": +1,
                        "cu2_enrolled": +1, "cu2_evaluations": +1},
}
BOUNDS = {
    "cdc_diabetes": {"bmi": (18.5, 100.0)},     # do not prescribe an underweight BMI
}


class ConstrainedSCM(SCM):
    """SCM whose do() clips intervention values to the allowed direction (relative to the
    factual x) and to plausibility bounds; non-intervened nodes are unaffected."""

    def __init__(self, d, parents, eqs, directions=None, bounds=None):
        super().__init__(d, parents, eqs)
        self.directions = directions or {}
        self.bounds = bounds or {}

    def clip(self, x, V):
        V = V.copy()
        for j, s in self.directions.items():
            V[:, j] = np.maximum(V[:, j], x[j]) if s > 0 else np.minimum(V[:, j], x[j])
        for j, (lo, hi) in self.bounds.items():
            V[:, j] = np.clip(V[:, j], min(lo, x[j]), max(hi, x[j]))
        return V

    def allowed(self, x, j, values):
        """Grid values that are a strict move in the allowed direction and within bounds."""
        v = np.asarray(values, float)
        s = self.directions.get(j, 0)
        if s > 0:
            v = v[v > x[j] + 1e-9]
        elif s < 0:
            v = v[v < x[j] - 1e-9]
        if j in self.bounds:
            lo, hi = self.bounds[j]
            v = v[(v >= lo - 1e-9) & (v <= hi + 1e-9)]
        return v

    def do(self, x, V, M):
        return super().do(x, self.clip(x, V) if (self.directions or self.bounds) else V, M)


def _raw(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    p = lambda f: os.path.join(DATA_DIR, f)
    if name == "financial_loan":
        df = pd.read_csv(p("financial_loan.csv"), encoding="latin1")
        df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]
        y = (df["loan_status"] == "Charged Off").astype(int)
        X = df[["annual_income", "dti", "int_rate", "loan_amount", "installment", "term",
                "emp_length", "home_ownership", "total_acc"]]
    elif name == "adult":
        df = pd.read_csv(p("adult.csv"))
        y = (~df["income"].astype(str).str.contains(">50K")).astype(int)
        X = df[["age", "workclass", "education_num", "marital_status", "occupation", "race",
                "sex", "capital_gain", "capital_loss", "hours_per_week"]]
    elif name == "german_credit":
        df = pd.read_csv(p("german_credit.csv"))
        y = (df["default"] == 2).astype(int)
        X = df.drop(columns=["default"])
    elif name == "medical_insurance":
        df = pd.read_csv(p("insurance.csv"))
        y = (df["charges"] > df["charges"].median()).astype(int)
        X = df.drop(columns=["charges"])
    elif name == "diabetes":
        df = pd.read_csv(p("diabetes.csv"))
        y = df["Outcome"].astype(int)
        X = df.drop(columns=["Outcome"])
    elif name == "breast_cancer":
        df = pd.read_csv(p("breast_cancer.csv"))
        y = (df["target"] == 0).astype(int)
        X = df[[c for c in df.columns if c.startswith("mean ")][:10]]
    elif name == "cdc_diabetes":
        df = pd.read_csv(os.path.join(EXT_DIR, "cdc", "cdc_diabetes.csv"))
        y = df["Diabetes_binary"].astype(int)
        X = df.drop(columns=["ID", "Diabetes_binary"])
    elif name == "heloc":
        from scipy.io import arff
        raw, _ = arff.loadarff(os.path.join(EXT_DIR, "heloc", "FICO-HELOC-cleaned.arff"))
        df = pd.DataFrame(raw)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].str.decode("utf-8")
        y = (df["RiskPerformance"] == "Bad").astype(int)
        X = df.drop(columns=["RiskPerformance"]).apply(pd.to_numeric, errors="coerce")
        X = X.replace([-9, -8, -7], np.nan)
        X = X.fillna(X.median())
    elif name == "taiwan_default":
        df = pd.read_csv(os.path.join(EXT_DIR, "taiwan", "taiwan_default.csv"))
        y = df["Y"].astype(int)
        X = df[list(TAIWAN_COLS)].rename(columns=TAIWAN_COLS).astype(float)
        # heavy-tailed money amounts: (signed) log1p, otherwise the MAD-scaled cost explodes
        for c in ["limit_bal", "bill_amt1", "bill_amt2", "pay_amt1", "pay_amt2", "pay_amt3"]:
            X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    elif name == "gmsc":
        from scipy.io import arff
        raw, _ = arff.loadarff(os.path.join(EXT_DIR, "gmsc", "GiveMeSomeCredit.arff"))
        df = pd.DataFrame(raw)
        tgt = df["FinancialDistressNextTwoYears"].str.decode("utf-8")
        y = (tgt == "Yes").astype(int)
        X = df.drop(columns=["FinancialDistressNextTwoYears"]).astype(float)
        X.columns = [c.replace("-", "_") for c in X.columns]
        X = X.fillna(X.median())
        for c in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]:
            X[c] = X[c].clip(X[c].quantile(0.01), X[c].quantile(0.99))   # heavy outliers
            X[c] = np.log1p(X[c].clip(lower=0))      # heavy tails: MAD would be ~0 otherwise
    elif name == "student_perf":
        df = pd.read_csv(os.path.join(EXT_DIR, "student_perf", "student_perf.csv"))
        y = (pd.to_numeric(df["G3"], errors="coerce") < 10).astype(int)
        X = df.drop(columns=["G1", "G2", "G3", "Mjob", "Fjob", "reason", "guardian",
                             "nursery"])
    elif name == "student_dropout":
        df = pd.read_csv(os.path.join(EXT_DIR, "student_dropout", "student_dropout.csv"),
                         sep=None, engine="python", encoding="utf-8-sig")
        y = (df["Target"] == "Dropout").astype(int)
        X = df[list(DROPOUT_COLS)].rename(columns=DROPOUT_COLS)
    else:
        raise ValueError(name)
    X = X.copy()
    X.columns = [c.strip().lower().replace(" ", "_") for c in X.columns]
    cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    if cat:
        X[cat] = OrdinalEncoder().fit_transform(X[cat].astype(str))
    return X.astype(float).reset_index(drop=True), y.reset_index(drop=True)


@dataclass
class RealTask:
    name: str
    features: List[str]          # topologically ordered
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    actionable: List[int]
    scm: SCM
    edges: List[Tuple[str, str]]
    mu: Optional[np.ndarray] = None      # train mean / sd in the (pre-standardisation) units,
    sd: Optional[np.ndarray] = None      # used to report recourse in original units
    n_raw: int = 0


def load_task(name: str, seed: int = 0) -> RealTask:
    X, y = _raw(name)
    immutable, edges = SPECS[name]
    missing = [f for e in edges for f in e if f not in X.columns] + \
              [f for f in immutable if f not in X.columns]
    assert not missing, f"{name}: unknown columns {missing}"

    # topological order that keeps the original column order where possible
    G = nx.DiGraph(); G.add_nodes_from(X.columns); G.add_edges_from(edges)
    pos = {c: i for i, c in enumerate(X.columns)}
    order = list(nx.lexicographical_topological_sort(G, key=lambda c: pos[c]))
    X = X[order]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    if len(Xtr) > MAX_TRAIN:
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=MAX_TRAIN, stratify=ytr,
                                          random_state=seed)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1.0)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    idx = {c: i for i, c in enumerate(order)}
    parents, eqs = {}, {}
    for c in order:
        ps = [idx[p] for p in G.predecessors(c)]
        if not ps:
            continue
        lr = LinearRegression().fit(Xtr.iloc[:, ps].values, Xtr[c].values)
        parents[idx[c]] = ps
        eqs[idx[c]] = (lambda A, ps=ps, w=lr.coef_, b=lr.intercept_: A[:, ps] @ w + b)
    directions = {idx[c]: v for c, v in DIRECTIONS.get(name, {}).items()}
    bounds = {idx[c]: ((lo - mu[c]) / sd[c], (hi - mu[c]) / sd[c])
              for c, (lo, hi) in BOUNDS.get(name, {}).items()}
    scm = ConstrainedSCM(len(order), parents, eqs, directions, bounds)
    actionable = [idx[c] for c in order if c not in immutable]
    return RealTask(name, order, Xtr.values, ytr.values, Xte.values, yte.values,
                    actionable, scm, edges, mu=mu[order].values, sd=sd[order].values,
                    n_raw=len(X))


# =========================================================================================
# Alternative SCMs over the SAME feature order (for the DAG sensitivity study, E4)
# =========================================================================================

class PermutedSCM:
    """Wraps a ConstrainedSCM fitted in topological order `perm` so that it can be used with
    feature vectors in the task's canonical order (models are trained on that order)."""

    def __init__(self, inner, perm):
        self.inner = inner
        self.perm = np.asarray(perm)              # canonical index of inner position p
        self.inv = np.argsort(self.perm)          # inner position of canonical index j
        self.d = inner.d
        self.directions = {int(self.perm[p]): s for p, s in inner.directions.items()}
        self.bounds = {int(self.perm[p]): b for p, b in inner.bounds.items()}

    def do(self, x, V, M):
        out = self.inner.do(x[self.perm], V[:, self.perm], M[:, self.perm])
        return out[:, self.inv]

    def clip(self, x, V):
        return self.inner.clip(x[self.perm], V[:, self.perm])[:, self.inv]

    def allowed(self, x, j, values):
        return self.inner.allowed(x[self.perm], int(self.inv[j]), values)

    def abduct_rows(self, X):
        return self.inner.abduct_rows(X[:, self.perm])[:, self.inv]

    def generate(self, U, V, M):
        return self.inner.generate(U[:, self.perm], V[:, self.perm], M[:, self.perm])[:, self.inv]


def build_scm(task: "RealTask", edges, regressor: str = "linear"):
    """SCM over task.features (canonical order) with the given named edges.
    regressor: 'linear' (LinearRegression) or 'gbr' (GradientBoostingRegressor)."""
    from sklearn.ensemble import GradientBoostingRegressor
    F = task.features
    G = nx.DiGraph(); G.add_nodes_from(F); G.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(G), "edges must form a DAG"
    pos = {c: i for i, c in enumerate(F)}
    order = list(nx.lexicographical_topological_sort(G, key=lambda c: pos[c]))
    perm = [pos[c] for c in order]
    Xp = task.X_train[:, perm]
    ipos = {c: i for i, c in enumerate(order)}
    parents, eqs = {}, {}
    for c in order:
        ps = [ipos[p] for p in G.predecessors(c)]
        if not ps:
            continue
        j = ipos[c]
        if regressor == "gbr":
            reg = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=0)
            reg.fit(Xp[:, ps], Xp[:, j])
            eqs[j] = (lambda A, ps=ps, r=reg: r.predict(A[:, ps]))
        else:
            lr = LinearRegression().fit(Xp[:, ps], Xp[:, j])
            eqs[j] = (lambda A, ps=ps, w=lr.coef_, b=lr.intercept_: A[:, ps] @ w + b)
        parents[j] = ps
    base = task.scm
    directions = {ipos[F[j]]: s for j, s in getattr(base, "directions", {}).items()}
    bounds = {ipos[F[j]]: b for j, b in getattr(base, "bounds", {}).items()}
    inner = ConstrainedSCM(len(order), parents, eqs, directions, bounds)
    return PermutedSCM(inner, perm)
