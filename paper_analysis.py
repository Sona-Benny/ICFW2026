"""Analysis behind the paper "Sombor-Type Indices in Predicting the Combustion
Enthalpy of Organic Compounds".

Reproduces every analysis of the reference article (Saviñon-Flores et al.,
ACS Omega 2025, 10, 41828) with Sombor-type indices as predictors, and writes
all figures, tables and a results.json (numbers quoted in the text) to paper/.

Expensive steps are cached in paper/cache/, so a rerun only redraws figures.
Delete the cache folder to recompute everything.
"""
import json
import math
import pickle
import warnings
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    LassoLars,
    LinearRegression,
    OrthogonalMatchingPursuit,
    Ridge,
)
from sklearn.metrics import (
    davies_bouldin_score,
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
DATA = BASE / "ALL.csv"
OUT = BASE / "paper"
REFERENCE = OUT / "data" / "reference_All.csv"  # reference article's data set (graph descriptors)
FIG = OUT / "figures"
TAB = OUT / "tables"
CACHE = OUT / "cache"

SEED = 42
TARGET = "Enthalpy reported by Gharagheizi"
SOMBOR = ["SO", "SOred", "SOavg", "mSO", "ESO", "EU", "N", "HSO"]
SOMBOR_HD = [f"{c} (HD)" for c in SOMBOR]  # same indices on the hydrogen-depleted graph
REF_INDICES = ["Degree", "Bet", "Eigen", "Closs", "Estrada", "Wiener", "Gutman"]
ENTHALPY_LABEL = r"$-\Delta_c H^\circ$ (kJ mol$^{-1}$)"

# reference article Table 4: experimental and Benson-method combustion enthalpies (kJ/mol)
BENSON_TABLE = [
    ("Undecanoic acid", 6736.5, 6727.07),
    ("Dodecanoic acid", 7423.7, 7377.00),
    ("Tridecanoic acid", 8024.2, 8026.93),
    ("Tetradecanoic acid", 8676.7, 8676.86),
    ("Pentadecanoic acid", 9327.7, 9326.79),
    ("Succinic acid", 1491.0, 1504.23),
    ("Sebacic acid", 5425.0, 5403.81),
    ("Pimelic acid", 3460.2, 3454.02),
    ("Trimethylsuccinic acid", 3450.75, 3460.98),
    ("Triethylsuccinic acid", 5441.3, 5421.81),
]
EXAMPLE_MOLECULES = [
    "benzene",
    "propylbenzene",
    "tetradecane",
    "heptadecylbenzene",
    "pentacosylbenzene",
    "nonacosylcyclopentane",
    "tritriacontylbenzene",
    "pentatriacontylbenzene",
]

# validated palette (dataviz skill): categorical slot 1/2, blue sequential ramp, chrome
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRIDC = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#c3c2b7"
SURFACE = "#ffffff"
BLUE_RAMP = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec",
             350: "#5598e7", 400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab",
             600: "#184f95", 650: "#104281", 700: "#0d366b"}
ORDINAL5 = [BLUE_RAMP[s] for s in (250, 350, 450, 550, 650)]  # passes --ordinal checks
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", [BLUE_RAMP[s] for s in range(100, 701, 50)])
SEQ_CMAP_MARKS = LinearSegmentedColormap.from_list("seq_blue_marks", [BLUE_RAMP[s] for s in range(250, 701, 50)])

plt.rcParams.update({
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 9.5,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRIDC,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ----------------------------------------------------------------------------- helpers

def cached(name, fn):
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    value = fn()
    path.write_bytes(pickle.dumps(value))
    return value


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def all_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "RMSLE": rmsle(y_true, y_pred),
    }


RMSLE_SCORER = make_scorer(rmsle, greater_is_better=False)
CV_SCORERS = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "MAPE": "neg_mean_absolute_percentage_error",
    "RMSE": "neg_root_mean_squared_error",
    "RMSLE": RMSLE_SCORER,
}
METRICS = list(CV_SCORERS)


def make_pipe(estimator):
    # z-score scaling inside the pipeline, so each CV fold is scaled on its own training part
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def panel_label(ax, text):
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, ha="right", va="bottom")


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


# ----------------------------------------------------------------------------- data

def sombor_indices(mol):
    n = mol.GetNumAtoms()
    m = mol.GetNumBonds()
    avg = 2 * m / n
    deg = [a.GetDegree() for a in mol.GetAtoms()]
    vals = np.zeros(8)
    for b in mol.GetBonds():
        du = deg[b.GetBeginAtomIdx()]
        dv = deg[b.GetEndAtomIdx()]
        root = math.sqrt(du * du + dv * dv)
        vals += [
            root,
            math.sqrt((du - 1) ** 2 + (dv - 1) ** 2),
            math.sqrt((du - avg) ** 2 + (dv - avg) ** 2),
            1 / root,
            (du + dv) * root,
            math.sqrt(du * du + dv * dv + du * dv),
            math.sqrt(du + dv),
            root / min(du, dv),
        ]
    return vals


CC_DOUBLE = Chem.MolFromSmarts("[C]=[C]")
CC_TRIPLE = Chem.MolFromSmarts("C#C")
COOH = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
AMINE_N = Chem.MolFromSmarts("[NX3;!$(N=*);!$(N#*);!$(N-[!#6;!#1]);!$(N-[#6]=[!#6])]")
HALOGENS = {"F", "Cl", "Br", "I"}

FAMILY_ORDER = [
    "Aromatic hydrocarbons",
    "Acyclic alkanes",
    "Cycloalkanes",
    "Acyclic alkenes",
    "Cycloalkenes",
    "Alkynes",
    "Amines",
    "Carboxylic acids (1 COOH)",
    "Carboxylic acids (2 COOH)",
]
CLASS_ORDER = [
    "Aliphatic hydrocarbons",
    "Aromatic hydrocarbons",
    "O-containing",
    "N-containing",
    "N- and O-containing",
    "S-containing",
    "Halogenated",
    "Other elements",
]


def family(mol):
    """Narrow compound families used for the family-specific QSPR models."""
    el = {a.GetSymbol() for a in mol.GetAtoms()}
    ring = mol.GetRingInfo().NumRings() > 0
    aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
    double = mol.HasSubstructMatch(CC_DOUBLE)
    triple = mol.HasSubstructMatch(CC_TRIPLE)
    if el == {"C"}:
        if aromatic:
            return "Aromatic hydrocarbons"
        if triple:
            return "Alkynes"
        if double:
            return "Cycloalkenes" if ring else "Acyclic alkenes"
        return "Cycloalkanes" if ring else "Acyclic alkanes"
    if el == {"C", "N"} and not ring and not (double or triple):
        n_nitrogen = sum(a.GetSymbol() == "N" for a in mol.GetAtoms())
        if len(mol.GetSubstructMatches(AMINE_N)) == n_nitrogen:
            return "Amines"
    if el == {"C", "O"} and not ring and not (double or triple):
        k = len(mol.GetSubstructMatches(COOH))
        n_oxygen = sum(a.GetSymbol() == "O" for a in mol.GetAtoms())
        if k in (1, 2) and n_oxygen == 2 * k:
            return f"Carboxylic acids ({k} COOH)"
    return "Other"


def broad_class(mol):
    """Broad chemical classes used for the supplementary per-class plots."""
    el = {a.GetSymbol() for a in mol.GetAtoms()}
    if el <= {"C", "H"}:
        aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
        return "Aromatic hydrocarbons" if aromatic else "Aliphatic hydrocarbons"
    if el & HALOGENS:
        return "Halogenated"
    if "S" in el:
        return "S-containing"
    if el <= {"C", "H", "N", "O"}:
        if "N" in el and "O" in el:
            return "N- and O-containing"
        return "N-containing" if "N" in el else "O-containing"
    return "Other elements"


def load_data():
    df = pd.read_csv(DATA)
    df["SMILES"] = df["SMILES"].str.strip()  # a few SMILES carry a trailing non-breaking space
    ref = pd.read_csv(REFERENCE)
    assert (ref["Name"].values == df["Name"].values).all(), "row order differs from reference"
    for col in REF_INDICES:
        df[col] = ref[col].values
    mols = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
    hd = np.array([sombor_indices(Chem.RemoveHs(m)) for m in mols])
    for i, col in enumerate(SOMBOR_HD):
        df[col] = hd[:, i]
    check = np.array([sombor_indices(Chem.AddHs(m)) for m in mols])
    assert np.allclose(check, df[SOMBOR].values), "H-explicit indices differ from ALL.csv"
    df["canonical"] = [Chem.MolToSmiles(m) for m in mols]
    df["family"] = [family(m) for m in mols]
    df["class"] = [broad_class(m) for m in mols]
    df["n_atoms_H"] = [Chem.AddHs(m).GetNumAtoms() for m in mols]
    return df, mols


def rdkit_descriptors(mols):
    rows = [Descriptors.CalcMolDescriptors(m, missingVal=np.nan) for m in mols]
    X = pd.DataFrame(rows).astype(float)
    n_all = X.shape[1]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().all()]
    X = X.loc[:, X.std() > 0]
    X = X.loc[:, X.abs().max() < 1e12]  # drops Ipc-type columns with astronomically large values
    return X, n_all


# ----------------------------------------------------------------------------- models

def model_zoo():
    return {
        "Linear regression": LinearRegression(),
        "Ridge": Ridge(random_state=SEED),
        "Lasso": Lasso(random_state=SEED, max_iter=50000),
        "Elastic net": ElasticNet(random_state=SEED, max_iter=50000),
        "Lasso least angle": LassoLars(random_state=SEED),
        "Orthogonal matching pursuit": OrthogonalMatchingPursuit(),
        "Bayesian ridge": BayesianRidge(),
        "Huber": HuberRegressor(max_iter=2000),
        "K-nearest neighbors": KNeighborsRegressor(),
        "Decision tree": DecisionTreeRegressor(random_state=SEED),
        "Random forest": RandomForestRegressor(random_state=SEED),
        "Extra trees": ExtraTreesRegressor(random_state=SEED),
        "AdaBoost": AdaBoostRegressor(random_state=SEED),
        "Gradient boosting": GradientBoostingRegressor(random_state=SEED),
        "LightGBM": LGBMRegressor(random_state=SEED, verbose=-1, n_jobs=1),
        "XGBoost": XGBRegressor(random_state=SEED, n_jobs=1),
        "Multilayer perceptron": MLPRegressor(random_state=SEED, max_iter=2000),
        "Support vector regression": SVR(),
        "Dummy (mean)": DummyRegressor(),
    }


RF_SPACE = {
    "model__n_estimators": randint(100, 500),
    "model__max_depth": [None, 5, 10, 15, 20, 30],
    "model__min_samples_split": randint(2, 11),
    "model__min_samples_leaf": randint(1, 6),
    "model__max_features": ["sqrt", "log2", 0.5, 1.0],
}
RF_SPACE_WIDE = {**RF_SPACE, "model__max_features": ["sqrt", "log2", 0.2, 0.3, 0.5]}
SEARCH_SPACES = {
    "Random forest": RF_SPACE,
    "Extra trees": RF_SPACE,
    "Gradient boosting": {
        "model__n_estimators": randint(100, 800),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": randint(2, 7),
        "model__subsample": uniform(0.6, 0.4),
    },
    "LightGBM": {
        "model__n_estimators": randint(100, 1000),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__num_leaves": randint(8, 64),
        "model__min_child_samples": randint(5, 40),
        "model__subsample": uniform(0.6, 0.4),
        "model__subsample_freq": [1],
        "model__colsample_bytree": uniform(0.5, 0.5),
    },
    "XGBoost": {
        "model__n_estimators": randint(100, 1000),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": randint(2, 9),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__min_child_weight": randint(1, 10),
    },
}


def compare_models(X, y):
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    rows = []
    for name, est in model_zoo().items():
        res = cross_validate(make_pipe(est), X, y, cv=cv, scoring=CV_SCORERS, n_jobs=-1)
        row = {"Model": name}
        for metric in METRICS:
            vals = res[f"test_{metric}"]
            if metric != "R2":
                vals = -vals
            row[f"{metric} mean"] = float(vals.mean())
            row[f"{metric} sd"] = float(vals.std())
        rows.append(row)
        print(f"    {name:28s} R2 {row['R2 mean']:.4f}  RMSLE {row['RMSLE mean']:.4f}")
    return pd.DataFrame(rows).set_index("Model").sort_values("RMSLE mean")


def tune(model_name, X, y, space=None, n_iter=30):
    search = RandomizedSearchCV(
        make_pipe(model_zoo()[model_name]),
        space or SEARCH_SPACES[model_name],
        n_iter=n_iter,
        scoring=RMSLE_SCORER,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X, y)
    return {
        "estimator": search.best_estimator_,
        "params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "cv_rmsle": float(-search.best_score_),
    }


# ----------------------------------------------------------------------------- figure 1 & 2

def fig_workflow(n):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    boxes = {
        "data": (2, 44, "Data set", f"{n} organic compounds\nName · SMILES · " + r"$-\Delta_cH^\circ$"),
        "graph": (27, 44, "Molecular graphs", "RDKit, explicit hydrogens\nG = (V, E), degrees $d_u$"),
        "index": (52, 44, "Sombor-type indices", "SO, SO$_{red}$, SO$_{avg}$, mSO\nESO, EU, N, HSO"),
        "eda": (77, 44, "Exploratory analysis", "Pearson correlations\nsingle-index regressions, VIF"),
        "ml": (77, 23, "Supervised learning", "70/30 split, z-scores\n19 regressors, 10-fold CV"),
        "tune": (52, 23, "Random forest", "random search, 5-fold CV\nparity, residuals, SHAP"),
        "bench": (27, 23, "Benchmarks", "reference indices, RDKit\nAD, Y-scrambling"),
        "family": (2, 23, "Family QSPR models", "9 compound families\nnested CV (5 × 3)"),
        "cluster": (52, 2, "Clustering", "K-Means, DBSCAN\nHDBSCAN, silhouette"),
    }
    w, h = 21, 15
    for key, (x, y, title, body) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    linewidth=0.8, edgecolor=AXIS, facecolor="#f4f8fe"))
        ax.add_patch(FancyBboxPatch((x, y + h - 1.2), w, 1.2, boxstyle="square,pad=0",
                                    linewidth=0, facecolor=BLUE))
        ax.text(x + w / 2, y + h - 4, title, ha="center", va="center", fontsize=8,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + 5.5, body, ha="center", va="center", fontsize=6.4, color=INK2,
                linespacing=1.4)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
                                     color=INK2, shrinkA=0, shrinkB=0))

    for a, b in [("data", "graph"), ("graph", "index"), ("index", "eda")]:
        xa, ya = boxes[a][0], boxes[a][1]
        xb = boxes[b][0]
        arrow((xa + w + 0.9, ya + h / 2), (xb - 0.9, ya + h / 2))
    arrow((boxes["eda"][0] + w / 2, 43.1), (boxes["ml"][0] + w / 2, 23 + h + 0.9))
    for a, b in [("ml", "tune"), ("tune", "bench"), ("bench", "family")]:
        xa, ya = boxes[a][0], boxes[a][1]
        xb = boxes[b][0]
        arrow((xa - 0.9, ya + h / 2), (xb + w + 0.9, ya + h / 2))
    arrow((boxes["tune"][0] + w / 2, 22.1), (boxes["cluster"][0] + w / 2, 2 + h + 0.9))
    save(fig, "fig01_workflow.png")


def fig_graph_construction():
    smiles = "CC(=O)O"
    heavy = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(heavy)
    AllChem.Compute2DCoords(mol)
    n = mol.GetNumAtoms()
    deg = [a.GetDegree() for a in mol.GetAtoms()]
    labels = [f"{a.GetSymbol()}{a.GetIdx() + 1}" for a in mol.GetAtoms()]

    drawer = rdMolDraw2D.MolDraw2DCairo(420, 320)
    drawer.drawOptions().clearBackground = False
    drawer.drawOptions().minFontSize = 22
    AllChem.Compute2DCoords(heavy)
    drawer.DrawMolecule(heavy)
    drawer.FinishDrawing()
    img_path = CACHE / "acetic_acid.png"
    img_path.write_bytes(drawer.GetDrawingText())

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(plt.imread(img_path))
    ax.axis("off")
    ax.set_title("Acetic acid, SMILES CC(=O)O", fontsize=8.5)
    panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    G = nx.Graph()
    conf = mol.GetConformer()
    pos = {}
    for a in mol.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        pos[a.GetIdx()] = (p.x, p.y)
        G.add_node(a.GetIdx())
    for b in mol.GetBonds():
        G.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=AXIS, width=1.6)
    heavy_nodes = [i for i in G if mol.GetAtomWithIdx(i).GetSymbol() != "H"]
    h_nodes = [i for i in G if i not in heavy_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=heavy_nodes, node_color=BLUE, node_size=520, ax=ax,
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=h_nodes, node_color=BLUE_RAMP[250], node_size=330, ax=ax,
                           edgecolors="white", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, {i: f"{labels[i]}\nd={deg[i]}" for i in heavy_nodes}, ax=ax,
                            font_size=6, font_color="white")
    nx.draw_networkx_labels(G, pos, {i: labels[i] for i in h_nodes}, ax=ax, font_size=6, font_color=INK)
    edge_labels = {(u, v): f"({deg[u]},{deg[v]})" for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=6, font_color=INK2,
                                 bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"})
    ax.set_title(f"Molecular graph G: n = {n}, m = {mol.GetNumBonds()}", fontsize=8.5)
    ax.axis("off")
    ax.margins(0.12)
    panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 0])
    A = Chem.GetAdjacencyMatrix(mol)
    ax.imshow(A, cmap=LinearSegmentedColormap.from_list("adj", ["#f4f4f2", BLUE]), vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(A[i, j]), ha="center", va="center", fontsize=6.5,
                    color="white" if A[i, j] else MUTED)
    ax.set_xticks(range(n), labels, fontsize=6.5, rotation=90)
    ax.set_yticks(range(n), labels, fontsize=6.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Adjacency matrix A", fontsize=8.5)
    panel_label(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    vals = sombor_indices(mol)
    rows = [[name, f"{v:.4f}"] for name, v in zip(
        ["SO", "SO$_{red}$", "SO$_{avg}$", "mSO", "ESO", "EU", "N", "HSO"], vals)]
    table = ax.table(cellText=rows, colLabels=["Index", "Value"], loc="center", cellLoc="center",
                     colWidths=[0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRIDC)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor("#f4f8fe")
    ax.set_title("Sombor-type indices of G", fontsize=8.5)
    panel_label(ax, "(d)")
    save(fig, "fig02_graph_construction.png")
    return {"smiles": smiles, "n": n, "m": mol.GetNumBonds(), "values": dict(zip(SOMBOR, vals))}


# ----------------------------------------------------------------------------- exploratory

def pretty(col):
    return {"SOred": "SO$_{red}$", "SOavg": "SO$_{avg}$"}.get(col, col)


def corr_heatmap(ax, corr, labels, vmin):
    im = ax.imshow(corr.values, cmap=SEQ_CMAP, vmin=vmin, vmax=1)
    k = len(labels)
    mid = vmin + (1 - vmin) * 0.55
    for i in range(k):
        for j in range(k):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.2,
                    color="white" if v > mid else INK)
    ax.set_xticks(range(k), labels, rotation=45, ha="right")
    ax.set_yticks(range(k), labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def exploratory(df):
    y = df[TARGET].values
    cols = SOMBOR + [TARGET]
    corr = df[cols].corr()
    labels = [pretty(c) for c in SOMBOR] + [r"$-\Delta_cH^\circ$"]
    vmin = math.floor(corr.values.min() * 100) / 100

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = corr_heatmap(ax, corr, labels, vmin)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson r", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "fig03_correlation_matrix.png")

    # supplementary: H-explicit vs H-depleted indices and reference descriptors
    cols_all = SOMBOR + SOMBOR_HD + REF_INDICES + [TARGET]
    corr_all = df[cols_all].corr()
    labels_all = [pretty(c) for c in SOMBOR] + [pretty(c.split(" ")[0]) + " (HD)" for c in SOMBOR_HD] \
        + REF_INDICES + [r"$-\Delta_cH^\circ$"]
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.imshow(corr_all.values, cmap=LinearSegmentedColormap.from_list(
        "div", [ORANGE, "#f0efec", BLUE]), vmin=-1, vmax=1)
    k = len(labels_all)
    for i in range(k):
        for j in range(k):
            v = corr_all.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if abs(v) > 0.7 else INK)
    ax.set_xticks(range(k), labels_all, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(k), labels_all, fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Pearson r", color=INK2)
    cb.outline.set_visible(False)
    save(fig, "figS2_correlation_all_descriptors.png")

    # single-index linear regressions
    fits = {}
    for col in SOMBOR + SOMBOR_HD + REF_INDICES:
        x = df[col].values
        lin = stats.linregress(x, y)
        quad = np.polyfit(x, y, 2)
        fits[col] = {
            "slope": float(lin.slope),
            "intercept": float(lin.intercept),
            "R2": float(lin.rvalue ** 2),
            "r": float(lin.rvalue),
            "R2_quadratic": float(r2_score(y, np.polyval(quad, x))),
        }

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.0), sharey=True)
    for ax, col in zip(axes.flat, SOMBOR):
        x = df[col].values
        f = fits[col]
        ax.scatter(x, y, s=3, color=BLUE, alpha=0.35, linewidths=0, rasterized=True)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, f["slope"] * xs + f["intercept"], color=INK, linewidth=1)
        ax.text(0.04, 0.95, f"R² = {f['R2']:.4f}", transform=ax.transAxes, va="top", fontsize=7,
                color=INK)
        ax.set_xlabel(pretty(col))
        ax.tick_params(labelsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(ENTHALPY_LABEL)
    fig.tight_layout()
    save(fig, "fig04_single_index_regressions.png")

    # multiple linear regression on the three indices most correlated with the target
    top3 = sorted(SOMBOR, key=lambda c: -fits[c]["R2"])[:3]
    mlr = LinearRegression().fit(df[top3].values, y)
    mlr_r2 = float(r2_score(y, mlr.predict(df[top3].values)))

    # variance inflation factors of the eight indices
    Z = StandardScaler().fit_transform(df[SOMBOR].values)
    vif = {}
    for j, col in enumerate(SOMBOR):
        others = np.delete(Z, j, axis=1)
        r2 = r2_score(Z[:, j], LinearRegression().fit(others, Z[:, j]).predict(others))
        vif[col] = float(1 / max(1 - r2, 1e-12))

    # distributions (supplementary)
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 5.6))
    for ax, col in zip(axes.flat, [TARGET] + SOMBOR):
        ax.hist(df[col].values, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
        ax.set_xlabel(ENTHALPY_LABEL if col == TARGET else pretty(col))
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("compounds")
    fig.tight_layout()
    save(fig, "figS1_distributions.png")

    return {
        "corr_with_target": {c: float(corr.loc[c, TARGET]) for c in SOMBOR},
        "corr_min_between_indices": float(corr.loc[SOMBOR, SOMBOR].values.min()),
        "fits": fits,
        "mlr": {"features": top3, "coef": dict(zip(top3, mlr.coef_.tolist())),
                "intercept": float(mlr.intercept_), "R2": mlr_r2},
        "vif": vif,
        "corr_all": corr_all.round(4).to_dict(),
    }


def fig_examples(df, fits, top3):
    low = df["Name"].str.lower().str.strip()
    found = [name for name in EXAMPLE_MOLECULES if (low == name).any()]
    rows = [df.loc[low == name].iloc[0] for name in found]
    y = df[TARGET].values
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    values = {}
    for ax, col, letter in zip(axes, top3, "abc"):
        x = df[col].values
        f = fits[col]
        ax.scatter(x, y, s=3, color=NEUTRAL, alpha=0.6, linewidths=0, rasterized=True)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, f["slope"] * xs + f["intercept"], color=INK2, linewidth=0.9)
        xe = [r[col] for r in rows]
        ye = [r[TARGET] for r in rows]
        ax.scatter(xe, ye, s=30, color=BLUE, edgecolors="white", linewidths=1.2, zorder=3)
        for i, (xi, yi) in enumerate(zip(xe, ye), 1):
            ax.annotate(str(i), (xi, yi), xytext=(-7, 4), textcoords="offset points", fontsize=7,
                        color=INK, fontweight="bold", ha="right")
        ax.text(0.04, 0.95, f"R² = {f['R2']:.4f}", transform=ax.transAxes, va="top", fontsize=7.5,
                color=INK)
        ax.set_xlabel(pretty(col))
        panel_label(ax, f"({letter})")
        values[col] = dict(zip(found, [float(v) for v in xe]))
    axes[0].set_ylabel(ENTHALPY_LABEL)
    key = "   ".join(f"{i} {name}" for i, name in enumerate(found, 1))
    fig.text(0.5, -0.04, key, ha="center", fontsize=6.8, color=INK2, wrap=True)
    fig.tight_layout()
    save(fig, "fig05_top_indices_examples.png")
    return {"molecules": found, "values": values,
            "enthalpy": dict(zip(found, [float(r[TARGET]) for r in rows]))}


def fig_class_small_multiples(df, fits, col, name):
    y_all = df[TARGET].values
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.9), sharex=True, sharey=True)
    out = {}
    for ax, cls in zip(axes.flat, CLASS_ORDER):
        mask = (df["class"] == cls).values
        ax.scatter(df[col].values[~mask], y_all[~mask], s=2, color="#ecebe7", linewidths=0, rasterized=True)
        ax.scatter(df[col].values[mask], y_all[mask], s=3, color=BLUE, alpha=0.6, linewidths=0,
                   rasterized=True)
        if mask.sum() > 2:
            lin = stats.linregress(df[col].values[mask], y_all[mask])
            out[cls] = {"n": int(mask.sum()), "R2": float(lin.rvalue ** 2)}
            ax.text(0.04, 0.95, f"n = {mask.sum()}\nR² = {lin.rvalue ** 2:.4f}", transform=ax.transAxes,
                    va="top", fontsize=6.5, color=INK)
        ax.set_title(cls, fontsize=7.5)
        ax.tick_params(labelsize=6)
    for ax in axes[1]:
        ax.set_xlabel(pretty(col), fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(ENTHALPY_LABEL, fontsize=7.5)
    fig.tight_layout()
    save(fig, name)
    return out


# ----------------------------------------------------------------------------- main model figures

def parity_panel(ax, y, p, colour, title, letter):
    lo, hi = 0, max(y.max(), p.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.9)
    ax.scatter(y, p, s=6, color=colour, alpha=0.55, linewidths=0, rasterized=True)
    m = all_metrics(y, p)
    r2_fit = stats.linregress(y, p).rvalue ** 2
    ax.text(0.04, 0.96,
            f"r² (fit) = {r2_fit:.4f}\nR² = {m['R2']:.4f}\nMAE = {m['MAE']:.1f} kJ mol⁻¹\n"
            f"RMSE = {m['RMSE']:.1f} kJ mol⁻¹\nMAPE = {m['MAPE']:.4f}\nRMSLE = {m['RMSLE']:.4f}",
            transform=ax.transAxes, va="top", fontsize=6.8, color=INK, linespacing=1.35)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"actual $-\Delta_cH^\circ$ (kJ mol$^{-1}$)")
    ax.set_ylabel(r"predicted $-\Delta_cH^\circ$ (kJ mol$^{-1}$)")
    ax.set_title(title)
    panel_label(ax, letter)
    return float(r2_fit)


def fig_parity(y_tr, p_tr, y_te, p_te, name):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7))
    r_tr = parity_panel(axes[0], y_tr, p_tr, BLUE, "Training set", "(a)")
    r_te = parity_panel(axes[1], y_te, p_te, ORANGE, "Test set", "(b)")
    fig.tight_layout()
    save(fig, name)
    return {"r2_fit_train": r_tr, "r2_fit_test": r_te}


def fig_residuals(y_tr, p_tr, y_te, p_te):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (y, p, colour, title, letter) in zip(axes[:2], [
            (y_tr, p_tr, BLUE, "Training set", "(a)"), (y_te, p_te, ORANGE, "Test set", "(b)")]):
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.scatter(p, y - p, s=5, color=colour, alpha=0.55, linewidths=0, rasterized=True)
        ax.set_xlabel(r"predicted $-\Delta_cH^\circ$ (kJ mol$^{-1}$)")
        ax.set_ylabel("residual (kJ mol$^{-1}$)")
        ax.set_title(title)
        panel_label(ax, letter)
    ax = axes[2]
    bins = np.linspace(-3000, 3000, 61)
    ax.hist(np.clip(y_tr - p_tr, -3000, 3000), bins=bins, density=True, histtype="step", color=BLUE,
            linewidth=1.5, label="training")
    ax.hist(np.clip(y_te - p_te, -3000, 3000), bins=bins, density=True, histtype="step", color=ORANGE,
            linewidth=1.5, label="test")
    ax.set_xlabel("residual (kJ mol$^{-1}$)")
    ax.set_ylabel("density")
    ax.set_title("Residual distribution")
    ax.legend(loc="upper right", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    panel_label(ax, "(c)")
    fig.tight_layout()
    save(fig, "fig09_residuals.png")


def tuning_curves(est, X_tr, y_tr):
    sizes, tr_s, va_s = learning_curve(
        clone(est), X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=SEED),
        train_sizes=np.linspace(0.1, 1.0, 10), scoring="neg_root_mean_squared_error", n_jobs=-1)
    depths = [2, 3, 4, 6, 8, 10, 12, 15, 20, 30]
    vtr, vva = validation_curve(
        clone(est), X_tr, y_tr, param_name="model__max_depth", param_range=depths,
        cv=KFold(5, shuffle=True, random_state=SEED), scoring="neg_root_mean_squared_error", n_jobs=-1)
    return {"sizes": sizes, "tr": -tr_s, "va": -va_s, "depths": depths, "vtr": -vtr, "vva": -vva}


def fig_tuning_curves(c):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, xs, a, b, xlabel, title, letter in [
        (axes[0], c["sizes"], c["tr"], c["va"], "training examples", "Learning curve", "(a)"),
        (axes[1], c["depths"], c["vtr"], c["vva"], "max_depth", "Validation curve", "(b)"),
    ]:
        for scores, colour, label in [(a, BLUE, "training"), (b, ORANGE, "validation (5-fold)")]:
            mean, sd = scores.mean(axis=1), scores.std(axis=1)
            ax.fill_between(xs, mean - sd, mean + sd, color=colour, alpha=0.15, linewidth=0)
            ax.plot(xs, mean, color=colour, linewidth=1.5, marker="o", markersize=3.5, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("RMSE (kJ mol$^{-1}$)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        panel_label(ax, letter)
    fig.tight_layout()
    save(fig, "fig07_learning_validation_curves.png")
    return {
        "learning_sizes": c["sizes"], "learning_train_rmse": c["tr"].mean(axis=1),
        "learning_valid_rmse": c["va"].mean(axis=1), "depths": c["depths"],
        "valid_curve_train_rmse": c["vtr"].mean(axis=1), "valid_curve_valid_rmse": c["vva"].mean(axis=1),
    }


def shap_analysis(est, X_te, features):
    scaler = est.named_steps["scaler"]
    model = est.named_steps["model"]
    Xs = scaler.transform(X_te)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Xs, check_additivity=False)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=features).sort_values(ascending=False)

    order = list(mean_abs.index)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    rng = np.random.default_rng(SEED)
    for row, col in enumerate(order):
        j = features.index(col)
        sv = values[:, j]
        fv = X_te[:, j]
        # beeswarm-like vertical jitter scaled by local point density
        hist, edges = np.histogram(sv, bins=60)
        idx = np.clip(np.digitize(sv, edges) - 1, 0, len(hist) - 1)
        spread = 0.38 * hist[idx] / hist.max()
        jitter = rng.uniform(-1, 1, len(sv)) * spread
        lo, hi = np.percentile(fv, [5, 95])
        sc = ax.scatter(sv, row + jitter, c=np.clip((fv - lo) / (hi - lo + 1e-12), 0, 1), cmap=SEQ_CMAP_MARKS,
                        s=4, linewidths=0, rasterized=True)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_yticks(range(len(order)), [pretty(c) for c in order])
    ax.set_xlabel("SHAP value (kJ mol$^{-1}$)")
    ax.set_title("SHAP summary (test set)")
    ax.grid(axis="y", visible=False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("index value", color=INK2)
    cb.outline.set_visible(False)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.barh(range(len(order)), mean_abs[order].values, color=BLUE, height=0.62)
    ax.set_yticks(range(len(order)), [pretty(c) for c in order])
    ax.set_xlabel("mean |SHAP| (kJ mol$^{-1}$)")
    ax.set_title("Global importance")
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(mean_abs[order].values):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=6.5, color=INK2)
    ax.set_xlim(0, mean_abs.max() * 1.22)
    panel_label(ax, "(b)")
    fig.tight_layout()
    save(fig, "fig06_shap.png")
    impurity = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return {"mean_abs_shap": mean_abs.to_dict(), "impurity_importance": impurity.to_dict()}


# ----------------------------------------------------------------------------- families

def family_models(df, predictor):
    """Family-specific QSPR models on one Sombor-type index.

    A structure listed twice keeps its later entry: the data set's second entries carry the
    experimental values quoted by the reference article (e.g. its Table 4 acids).
    """
    huber = make_pipe(HuberRegressor(max_iter=5000))
    huber_grid = {"model__alpha": [1e-7, 1e-4, 1e-2, 0.1, 0.7, 1.0],
                  "model__epsilon": [1.1, 1.2, 1.35, 1.5, 1.8]}
    et = ExtraTreesRegressor(n_estimators=100, random_state=SEED)
    et_grid = {"min_samples_leaf": [1, 2, 4], "max_depth": [None, 10]}
    plan = {
        "Aromatic hydrocarbons": ("Extra trees", et, et_grid),
        "Acyclic alkanes": ("Huber", huber, huber_grid),
        "Cycloalkanes": ("Huber", huber, huber_grid),
        "Acyclic alkenes": ("Huber", huber, huber_grid),
        "Cycloalkenes": ("Huber", huber, huber_grid),
        "Alkynes": ("Huber", huber, huber_grid),
        "Amines": ("Linear regression", None, None),
        "Carboxylic acids (1 COOH)": ("Linear regression", None, None),
        "Carboxylic acids (2 COOH)": ("Linear regression", None, None),
    }
    results = {}
    for fam, (model_name, est, grid) in plan.items():
        sub = df[df["family"] == fam].drop_duplicates("canonical", keep="last")
        X = sub[[predictor]].values
        y = sub[TARGET].values
        if est is None:
            pred = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
            fit = LinearRegression().fit(X, y)
            results[fam] = {
                "model": model_name, "n": len(y), "validation": "leave-one-out",
                "metrics": all_metrics(y, pred), "fit_metrics": all_metrics(y, fit.predict(X)),
                "slope": float(fit.coef_[0]), "intercept": float(fit.intercept_),
                "y": y, "pred": pred, "names": sub["Name"].tolist(), "canonical": sub["canonical"].tolist(),
            }
            continue
        outer = KFold(5, shuffle=True, random_state=SEED)
        pred = np.empty_like(y)
        fold_rows, params = [], []
        for tr, te in outer.split(X):
            search = GridSearchCV(clone(est), grid, cv=KFold(3, shuffle=True, random_state=SEED),
                                  scoring="neg_mean_absolute_error", n_jobs=-1)
            search.fit(X[tr], y[tr])
            pred[te] = search.predict(X[te])
            fold_rows.append(all_metrics(y[te], pred[te]))
            params.append({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        folds = pd.DataFrame(fold_rows)
        results[fam] = {
            "model": model_name, "n": len(y), "validation": "nested CV (5 outer × 3 inner)",
            "metrics_mean": folds.mean().to_dict(), "metrics_sd": folds.std(ddof=0).to_dict(),
            "metrics": all_metrics(y, pred), "params": params,
            "y": y, "pred": pred, "names": sub["Name"].tolist(), "canonical": sub["canonical"].tolist(),
        }
    return results


def fig_families(results, predictor):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.2))
    for ax, fam, letter in zip(axes.flat, FAMILY_ORDER, "abcdefghi"):
        r = results[fam]
        y, p = r["y"], r["pred"]
        lo = min(y.min(), p.min())
        hi = max(y.max(), p.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, linewidth=0.8)
        ax.scatter(y, p, s=10, color=BLUE, alpha=0.7, edgecolors="white", linewidths=0.4)
        m = r["metrics"]
        text = f"n = {r['n']}\nR² = {m['R2']:.4f}\nMAE = {m['MAE']:.1f}\nRMSE = {m['RMSE']:.1f}"
        if "slope" in r:
            text += f"\ny = {r['slope']:.3f}x {'+' if r['intercept'] >= 0 else '−'} {abs(r['intercept']):.1f}"
        ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", fontsize=6.2, color=INK, linespacing=1.3)
        ax.set_title(f"{fam}\n{r['model']}", fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        panel_label(ax, f"({letter})")
    for ax in axes[2]:
        ax.set_xlabel("actual (kJ mol$^{-1}$)", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted (kJ mol$^{-1}$)", fontsize=7.5)
    fig.suptitle(f"Predictor: {pretty(predictor)} index; out-of-fold predictions", x=0.01, ha="left",
                 fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "fig10_family_models.png")


def benson_comparison(df, fam_results):
    rows = []
    for name, exp, benson in BENSON_TABLE:
        mol = Chem.MolFromSmiles(df.loc[df["Name"].str.lower().str.strip() == name.lower(), "SMILES"].iloc[0])
        can = Chem.MolToSmiles(mol)
        fam = family(mol)
        r = fam_results[fam]
        pred = float(r["pred"][r["canonical"].index(can)])
        rows.append({
            "Compound": name, "Exp": exp, "Benson": benson, "Delta Benson": benson - exp,
            "Predicted (LOO)": pred, "Delta predicted": pred - exp,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- RDKit descriptors

def fig_rdkit_correlations(desc, df):
    corr = []
    for d in desc.columns:
        for s in SOMBOR:
            r = np.corrcoef(desc[d].values, df[s].values)[0, 1]
            corr.append((d, s, r))
    corr = pd.DataFrame(corr, columns=["descriptor", "index", "r"])
    corr["abs"] = corr["r"].abs()
    top = corr.sort_values("abs", ascending=False).head(30).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 6.4))
    ypos = np.arange(len(top))
    ax.hlines(ypos, top["r"].min() - 0.002, top["r"], color=GRIDC, linewidth=1)
    ax.scatter(top["r"], ypos, s=34, color=BLUE, edgecolors="white", linewidths=1, zorder=3)
    ax.set_yticks(ypos, [f"{d}  –  {pretty(s)}" for d, s in zip(top["descriptor"], top["index"])], fontsize=7)
    for yv, rv in zip(ypos, top["r"]):
        ax.text(rv + 0.0004, yv, f"{rv:.4f}", va="center", fontsize=6.3, color=INK2)
    ax.set_xlabel("Pearson r")
    ax.set_xlim(top["r"].min() - 0.002, 1.0015)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig12_rdkit_index_correlations.png")
    per_index = {s: corr[corr["index"] == s].sort_values("abs", ascending=False).head(5)[["descriptor", "r"]]
                 .values.tolist() for s in SOMBOR}
    return {"top30": top.iloc[::-1][["descriptor", "index", "r"]].values.tolist(), "per_index_top5": per_index,
            "table": corr}


# ----------------------------------------------------------------------------- applicability domain

def williams_and_scrambling(est, X_tr, y_tr, X_te, y_te, scramble):
    scaler = StandardScaler().fit(X_tr)
    Ztr = np.column_stack([np.ones(len(X_tr)), scaler.transform(X_tr)])
    Zte = np.column_stack([np.ones(len(X_te)), scaler.transform(X_te)])
    inv = np.linalg.pinv(Ztr.T @ Ztr)
    h_tr = np.einsum("ij,jk,ik->i", Ztr, inv, Ztr)
    h_te = np.einsum("ij,jk,ik->i", Zte, inv, Zte)
    h_star = 3 * (X_tr.shape[1] + 1) / len(X_tr)
    p_cv = cross_val_predict(clone(est), X_tr, y_tr, cv=KFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    sd = np.std(y_tr - p_cv)
    r_tr = (y_tr - p_cv) / sd
    r_te = (y_te - est.predict(X_te)) / sd

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.scatter(h_tr, r_tr, s=5, color=BLUE, alpha=0.5, linewidths=0, label="training (5-fold CV)", rasterized=True)
    ax.scatter(h_te, r_te, s=5, color=ORANGE, alpha=0.6, linewidths=0, label="test", rasterized=True)
    for yv in (-3, 3):
        ax.axhline(yv, color=INK2, linewidth=0.8)
    ax.axvline(h_star, color=INK2, linewidth=0.8)
    ax.text(h_star, ax.get_ylim()[1], f" h* = {h_star:.4f}", va="top", fontsize=7, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("leverage h (log scale)")
    ax.set_ylabel("standardized residual")
    ax.set_title("Williams plot")
    ax.legend(loc="lower left", fontsize=7, markerscale=2)
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.hist(scramble["test_r2"], bins=8, color=NEUTRAL, edgecolor="white", linewidth=0.5,
            label="scrambled targets")
    ax.axvline(scramble["true_test_r2"], color=BLUE, linewidth=2, label="real model")
    ax.set_xlim(-0.4, 1.05)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper center", fontsize=7)
    ax.set_xlabel("test R²")
    ax.set_ylabel("count")
    ax.set_title(f"Y-scrambling ({len(scramble['test_r2'])} runs)")
    ax.grid(axis="x", visible=False)
    panel_label(ax, "(b)")
    fig.tight_layout()
    save(fig, "fig14_applicability_domain.png")
    return {
        "h_star": h_star,
        "train_outside_h": int((h_tr > h_star).sum()), "test_outside_h": int((h_te > h_star).sum()),
        "train_outlier_resid": int((np.abs(r_tr) > 3).sum()), "test_outlier_resid": int((np.abs(r_te) > 3).sum()),
        "test_in_domain_frac": float(((h_te <= h_star) & (np.abs(r_te) <= 3)).mean()),
    }


def y_scrambling(est, X_tr, y_tr, X_te, y_te, runs=20):
    rng = np.random.default_rng(SEED)
    test_r2, train_r2 = [], []
    for _ in range(runs):
        ys = rng.permutation(y_tr)
        m = clone(est).fit(X_tr, ys)
        train_r2.append(r2_score(ys, m.predict(X_tr)))
        test_r2.append(r2_score(y_te, m.predict(X_te)))
    return {"test_r2": np.array(test_r2), "train_r2": np.array(train_r2)}


# ----------------------------------------------------------------------------- clustering

def clustering(df, features):
    cols = [TARGET] + features
    Z = MinMaxScaler().fit_transform(df[cols].values)
    ks = np.arange(1, 16)
    inertia = [KMeans(k, init="k-means++", n_init=10, random_state=SEED).fit(Z).inertia_ for k in ks]
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    yv = (np.array(inertia) - min(inertia)) / (max(inertia) - min(inertia))
    k_opt = int(ks[np.argmax((1 - x) - yv)])  # elbow: farthest point below the chord
    sil_k = {}
    db_k = {}
    for k in range(2, 13):
        lab = KMeans(k, init="k-means++", n_init=10, random_state=SEED).fit_predict(Z)
        sil_k[k] = float(silhouette_score(Z, lab))
        db_k[k] = float(davies_bouldin_score(Z, lab))

    km = KMeans(k_opt, init="k-means++", n_init=10, random_state=SEED).fit(Z)
    raw = km.labels_
    order = pd.Series(df[TARGET].values).groupby(raw).mean().sort_values().index.tolist()
    relabel = {old: new for new, old in enumerate(order)}
    labels = np.array([relabel[v] for v in raw])
    centers = km.cluster_centers_[order]

    comparison = {"K-Means": {"clusters": k_opt, "silhouette": float(silhouette_score(Z, labels)),
                              "davies_bouldin": float(davies_bouldin_score(Z, labels)), "noise": 0}}
    for name, algo in [("DBSCAN", DBSCAN(eps=0.1, min_samples=20)), ("HDBSCAN", HDBSCAN(min_cluster_size=50))]:
        lab = algo.fit_predict(Z)
        keep = lab >= 0
        n_cl = len(set(lab[keep]))
        entry = {"clusters": n_cl, "noise": int((~keep).sum())}
        if n_cl >= 2:
            entry["silhouette"] = float(silhouette_score(Z[keep], lab[keep]))
            entry["davies_bouldin"] = float(davies_bouldin_score(Z[keep], lab[keep]))
        comparison[name] = entry

    reps = {}
    for c in range(k_opt):
        idx = np.where(labels == c)[0]
        d = np.linalg.norm(Z[idx] - centers[c], axis=1)
        reps[c] = df["Name"].values[idx[np.argsort(d)[:3]]].tolist()

    return {"features": features, "Z": Z, "ks": ks, "inertia": inertia, "k_opt": k_opt,
            "silhouette_by_k": sil_k, "db_by_k": db_k, "labels": labels,
            "sil_samples": silhouette_samples(Z, labels), "comparison": comparison, "representatives": reps}


def cluster_colours(k):
    if k <= 5:
        return ORDINAL5[:k] if k == 5 else [ORDINAL5[i] for i in np.round(np.linspace(0, 4, k)).astype(int)]
    return [SEQ_CMAP_MARKS(v) for v in np.linspace(0, 1, k)]


def fig_elbow(cl):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.plot(cl["ks"], cl["inertia"], color=BLUE, linewidth=1.5, marker="o", markersize=4)
    i = list(cl["ks"]).index(cl["k_opt"])
    ax.scatter([cl["k_opt"]], [cl["inertia"][i]], s=70, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=3)
    ax.annotate(f"elbow, k = {cl['k_opt']}", (cl["k_opt"], cl["inertia"][i]), xytext=(12, 18),
                textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("sum of squared distances")
    ax.set_title("Elbow method (K-Means)")
    panel_label(ax, "(a)")
    ax = axes[1]
    ks = list(cl["silhouette_by_k"])
    ax.plot(ks, [cl["silhouette_by_k"][k] for k in ks], color=BLUE, linewidth=1.5, marker="o", markersize=4,
            label="silhouette (higher is better)")
    ax.plot(ks, [cl["db_by_k"][k] for k in ks], color=ORANGE, linewidth=1.5, marker="o", markersize=4,
            label="Davies–Bouldin (lower is better)")
    ax.axvline(cl["k_opt"], color=AXIS, linewidth=0.8)
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("score")
    ax.set_title("Cluster quality by k")
    ax.legend(fontsize=7)
    panel_label(ax, "(b)")
    fig.tight_layout()
    save(fig, "fig15_elbow.png")


def fig_silhouette(cl):
    k = cl["k_opt"]
    colours = cluster_colours(k)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    y0 = 10
    for c in range(k):
        s = np.sort(cl["sil_samples"][cl["labels"] == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(s)), 0, s, color=colours[c], linewidth=0)
        ax.text(-0.04, y0 + len(s) / 2, f"C{c + 1}", ha="right", va="center", fontsize=7.5, color=INK)
        y0 += len(s) + 25
    avg = cl["comparison"]["K-Means"]["silhouette"]
    ax.axvline(avg, color=INK, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(avg, y0, f" mean = {avg:.3f}", fontsize=7.5, color=INK, va="bottom")
    ax.set_xlabel("silhouette coefficient")
    ax.set_yticks([])
    ax.set_xlim(min(-0.1, cl["sil_samples"].min() - 0.02), 1)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig16_silhouette.png")


def cluster_scatter(ax, x, y, labels, colours, zoom=None):
    k = len(colours)
    for c in range(k):
        m = labels == c
        if zoom is not None and c not in zoom:
            continue
        ax.scatter(x[m], y[m], s=5, color=colours[c], alpha=0.75, linewidths=0, rasterized=True)
    for c in range(k):
        if zoom is not None and c not in zoom:
            continue
        m = labels == c
        ax.text(np.median(x[m]), np.median(y[m]), f"C{c + 1}", fontsize=7, fontweight="bold", color=INK,
                ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": AXIS, "lw": 0.5, "alpha": 0.9})


def fig_cluster_spaces(df, cl):
    feats = cl["features"]
    labels = cl["labels"]
    k = cl["k_opt"]
    colours = cluster_colours(k)
    zoom = list(range(max(2, (k + 1) // 2)))
    pairs = [(feats[0], feats[1]), (feats[0], feats[2]), (feats[1], feats[2])]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    for j, (a, b) in enumerate(pairs):
        for i, z in enumerate([None, zoom]):
            ax = axes[i, j]
            cluster_scatter(ax, df[a].values, df[b].values, labels, colours, z)
            ax.set_xlabel(pretty(a))
            ax.set_ylabel(pretty(b))
            ax.tick_params(labelsize=6.5)
            panel_label(ax, f"({'abcdef'[i * 3 + j]})")
            if z is not None:
                ax.set_title(f"zoom: C1–C{len(zoom)}", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig17_cluster_index_space.png")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    for j, a in enumerate(feats):
        for i, z in enumerate([None, zoom]):
            ax = axes[i, j]
            cluster_scatter(ax, df[a].values, df[TARGET].values, labels, colours, z)
            ax.set_xlabel(pretty(a))
            ax.set_ylabel(ENTHALPY_LABEL if j == 0 else "")
            ax.tick_params(labelsize=6.5)
            panel_label(ax, f"({'abcdef'[i * 3 + j]})")
            if z is not None:
                ax.set_title(f"zoom: C1–C{len(zoom)}", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig18_cluster_enthalpy.png")
    return zoom


def cluster_tables(df, cl, desc, top5_desc):
    labels = cl["labels"]
    rows = []
    for c in range(cl["k_opt"]):
        sub = df[labels == c]
        for col in [TARGET] + cl["features"]:
            v = sub[col].values
            n = len(v)
            mean, sd = v.mean(), v.std(ddof=1)
            se = sd / math.sqrt(n)
            ci95 = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
            ci99 = stats.t.interval(0.99, n - 1, loc=mean, scale=se)
            rows.append({"Cluster": f"C{c + 1}", "Variable": "-ΔcH° (kJ/mol)" if col == TARGET else col,
                         "n": n, "Mean": mean, "SD": sd, "Min": v.min(), "Max": v.max(),
                         "95% CI": f"{ci95[0]:.2f} – {ci95[1]:.2f}", "99% CI": f"{ci99[0]:.2f} – {ci99[1]:.2f}"})
    s4 = pd.DataFrame(rows)
    rows = []
    for c in range(cl["k_opt"]):
        sub = desc[labels == c]
        row = {"Cluster": f"C{c + 1}", "n": int((labels == c).sum())}
        for d in top5_desc:
            row[d] = f"{sub[d].mean():.2f} ± {sub[d].std(ddof=1):.2f}"
        row["Representative compounds"] = ", ".join(cl["representatives"][c])
        rows.append(row)
    s5 = pd.DataFrame(rows)
    return s4, s5


# ----------------------------------------------------------------------------- feature-set comparison

def fig_feature_sets(bench):
    labels = list(bench.index)[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, metric, letter, fmt in [(axes[0], "RMSE", "(a)", "{:.1f}"), (axes[1], "RMSLE", "(b)", "{:.4f}")]:
        vals = bench.loc[labels, metric].values
        ax.barh(range(len(labels)), vals, color=BLUE, height=0.6)
        for i, v in enumerate(vals):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=6.8, color=INK2)
        ax.set_xlim(0, vals.max() * 1.22)
        ax.set_xlabel(f"test {metric}" + (" (kJ mol$^{-1}$)" if metric == "RMSE" else ""))
        ax.grid(axis="y", visible=False)
        panel_label(ax, letter)
    axes[0].set_yticks(range(len(labels)), labels, fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig13_feature_set_comparison.png")


# ----------------------------------------------------------------------------- main

def main():
    for d in (FIG, TAB, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    R = {}

    print("Data")
    df, mols = cached("data", load_data)
    desc, n_desc_all = cached("rdkit_descriptors", lambda: rdkit_descriptors(mols))
    y = df[TARGET].values
    dup = df.groupby("canonical")[TARGET].agg(["count", "min", "max"])
    dup = dup[dup["count"] > 1]
    R["data"] = {
        "n": len(df), "unique_structures": int(df["canonical"].nunique()),
        "duplicate_rows": int(len(df) - df["canonical"].nunique()),
        "duplicate_groups": len(dup),
        "duplicate_groups_discrepant": int(((dup["max"] - dup["min"]) > 1).sum()),
        "duplicate_max_diff": float((dup["max"] - dup["min"]).max()),
        "target": df[TARGET].describe().to_dict(),
        "families": df["family"].value_counts().to_dict(),
        "families_unique": df.groupby("family")["canonical"].nunique().to_dict(),
        "classes": df["class"].value_counts().to_dict(),
        "n_rdkit_all": n_desc_all, "n_rdkit_used": desc.shape[1],
        "indices_describe": df[SOMBOR].describe().T.to_dict(orient="index"),
    }

    print("Figures 1-5, S1-S2")
    fig_workflow(len(df))
    R["example_graph"] = fig_graph_construction()
    R["eda"] = exploratory(df)
    top3_r = R["eda"]["mlr"]["features"]
    R["examples"] = fig_examples(df, R["eda"]["fits"], top3_r)

    print("Train/test split")
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.3, random_state=SEED)
    y_tr, y_te = y[train_idx], y[test_idx]
    feature_sets = {
        "Sombor-type, H-explicit (8)": SOMBOR,
        "Sombor-type, H-depleted (8)": SOMBOR_HD,
        "Sombor-type, both graphs (16)": SOMBOR + SOMBOR_HD,
        "Reference graph descriptors (7)": REF_INDICES,
        "Sombor-type + reference (15)": SOMBOR + REF_INDICES,
        f"RDKit descriptors ({desc.shape[1]})": list(desc.columns),
    }
    full = pd.concat([df.reset_index(drop=True), desc.reset_index(drop=True)], axis=1)

    def X_of(cols):
        return full[cols].values

    print("Model comparison (10-fold CV)")
    cmp_sombor = cached("cmp_sombor", lambda: compare_models(X_of(SOMBOR)[train_idx], y_tr))
    print("  RDKit descriptors")
    cmp_rdkit = cached("cmp_rdkit", lambda: compare_models(X_of(list(desc.columns))[train_idx], y_tr))
    print("  reference graph descriptors")
    cmp_ref = cached("cmp_ref", lambda: compare_models(X_of(REF_INDICES)[train_idx], y_tr))
    cmp_sombor.to_csv(TAB / "tableS2_model_comparison_sombor.csv")
    cmp_rdkit.to_csv(TAB / "tableS3_model_comparison_rdkit.csv")
    cmp_ref.to_csv(TAB / "tableS3b_model_comparison_reference.csv")
    R["cv_sombor"] = cmp_sombor.to_dict(orient="index")
    R["cv_rdkit"] = cmp_rdkit.to_dict(orient="index")
    R["cv_ref"] = cmp_ref.to_dict(orient="index")

    print("Random forest tuning on Sombor-type indices")
    X_tr, X_te = X_of(SOMBOR)[train_idx], X_of(SOMBOR)[test_idx]
    rf = cached("tune_rf_sombor", lambda: tune("Random forest", X_tr, y_tr))
    best = rf["estimator"]
    p_tr, p_te = best.predict(X_tr), best.predict(X_te)
    R["rf_sombor"] = {"params": rf["params"], "cv_rmsle": rf["cv_rmsle"],
                      "train": all_metrics(y_tr, p_tr), "test": all_metrics(y_te, p_te)}
    R["rf_sombor"].update(fig_parity(y_tr, p_tr, y_te, p_te, "fig08_parity_sombor.png"))
    fig_residuals(y_tr, p_tr, y_te, p_te)
    R["curves"] = fig_tuning_curves(cached("curves", lambda: tuning_curves(best, X_tr, y_tr)))
    print("SHAP")
    R["shap"] = shap_analysis(best, X_te, SOMBOR)
    top3_shap = list(R["shap"]["mean_abs_shap"])[:3]

    print("Best CV model on Sombor-type indices")
    best_cv = cmp_sombor.index[0]
    if best_cv in SEARCH_SPACES and best_cv != "Random forest":
        tuned = cached(f"tune_{best_cv}", lambda: tune(best_cv, X_tr, y_tr, n_iter=40))
        pb_tr, pb_te = tuned["estimator"].predict(X_tr), tuned["estimator"].predict(X_te)
        R["best_cv_model"] = {"name": best_cv, "params": tuned["params"], "cv_rmsle": tuned["cv_rmsle"],
                              "train": all_metrics(y_tr, pb_tr), "test": all_metrics(y_te, pb_te)}
    else:
        R["best_cv_model"] = {"name": best_cv}

    print("Feature-set benchmark (tuned random forest per set)")
    bench_rows = {}
    bench_models = {}
    for label, cols in feature_sets.items():
        is_rdkit = label.startswith("RDKit")
        key = "tune_rf_" + "".join(ch for ch in label if ch.isalnum())
        tuned = cached(key, lambda: tune("Random forest", X_of(cols)[train_idx], y_tr,
                                         space=RF_SPACE_WIDE if is_rdkit else None,
                                         n_iter=20 if is_rdkit else 30))
        bench_models[label] = tuned
        pt = tuned["estimator"].predict(X_of(cols)[test_idx])
        bench_rows[label] = {**all_metrics(y_te, pt), "n_features": len(cols), "params": tuned["params"]}
        print(f"  {label:36s} test R2 {bench_rows[label]['R2']:.4f}  RMSE {bench_rows[label]['RMSE']:.1f}")
    bench = pd.DataFrame(bench_rows).T
    bench[METRICS].astype(float).to_csv(TAB / "table_feature_set_benchmark.csv")
    R["benchmark"] = {k: {m: v[m] for m in METRICS + ["n_features", "params"]} for k, v in bench_rows.items()}
    fig_feature_sets(bench[METRICS].astype(float))

    rd_label = f"RDKit descriptors ({desc.shape[1]})"
    rd = bench_models[rd_label]["estimator"]
    Xr_tr, Xr_te = X_of(list(desc.columns))[train_idx], X_of(list(desc.columns))[test_idx]
    pr_tr, pr_te = rd.predict(Xr_tr), rd.predict(Xr_te)
    R["rf_rdkit"] = {"params": bench_models[rd_label]["params"],
                     "train": all_metrics(y_tr, pr_tr), "test": all_metrics(y_te, pr_te)}
    R["rf_rdkit"].update(fig_parity(y_tr, pr_tr, y_te, pr_te, "fig11_parity_rdkit.png"))
    rd_imp = pd.Series(rd.named_steps["model"].feature_importances_, index=desc.columns).sort_values(ascending=False)
    R["rf_rdkit"]["top_importance"] = rd_imp.head(10).to_dict()

    print("Robustness: structure-grouped split, Y-scrambling, applicability domain")

    def grouped():
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
        tr, te = next(gss.split(df, groups=df["canonical"]))
        m = clone(best).fit(X_of(SOMBOR)[tr], y[tr])
        return {"n_train": len(tr), "n_test": len(te), "test": all_metrics(y[te], m.predict(X_of(SOMBOR)[te]))}

    R["grouped_split"] = cached("grouped_split", grouped)
    scramble = cached("y_scramble", lambda: y_scrambling(best, X_tr, y_tr, X_te, y_te))
    scramble["true_test_r2"] = R["rf_sombor"]["test"]["R2"]
    R["y_scrambling"] = {"test_r2_mean": float(scramble["test_r2"].mean()),
                         "test_r2_sd": float(scramble["test_r2"].std()),
                         "train_r2_mean": float(scramble["train_r2"].mean()),
                         "runs": len(scramble["test_r2"])}
    R["applicability"] = williams_and_scrambling(best, X_tr, y_tr, X_te, y_te, scramble)

    print("Family-specific models")
    family_predictor = top3_r[0]
    fam = cached("families", lambda: family_models(df, family_predictor))
    fig_families(fam, family_predictor)
    fam_rows = []
    for name in FAMILY_ORDER:
        r = fam[name]
        row = {"Family": name, "n": r["n"], "Model": r["model"], "Validation": r["validation"]}
        if "metrics_mean" in r:
            for m in METRICS:
                row[m] = f"{r['metrics_mean'][m]:.4g} ± {r['metrics_sd'][m]:.2g}"
        else:
            for m in METRICS:
                row[m] = f"{r['metrics'][m]:.4g}"
            row["Equation"] = f"{r['slope']:.4f} x + {r['intercept']:.2f}"
        fam_rows.append(row)
    pd.DataFrame(fam_rows).to_csv(TAB / "table3_family_models.csv", index=False)
    R["families"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("y", "pred", "names", "canonical")}
                     for k, v in fam.items()}
    R["family_predictor"] = family_predictor
    fam_r2 = {}
    for name in FAMILY_ORDER:
        sub = df[df["family"] == name].drop_duplicates("canonical", keep="last")
        fam_r2[name] = {c: float(stats.linregress(sub[c], sub[TARGET]).rvalue ** 2)
                        if sub[c].nunique() > 1 else float("nan") for c in SOMBOR + SOMBOR_HD}
    pd.DataFrame(fam_r2).T.to_csv(TAB / "tableS6_family_single_index_r2.csv")
    R["family_single_index_r2"] = fam_r2

    benson = benson_comparison(df, fam)
    benson.to_csv(TAB / "table4_benson_comparison.csv", index=False)
    R["benson"] = benson.to_dict(orient="records")
    R["benson_mae"] = {"benson": float(benson["Delta Benson"].abs().mean()),
                       "predicted": float(benson["Delta predicted"].abs().mean())}

    print("RDKit correlations")
    rc = fig_rdkit_correlations(desc, df)
    R["rdkit_corr"] = {"top30": rc["top30"], "per_index_top5": rc["per_index_top5"]}

    print("Supplementary per-class plots")
    R["class_fits"] = {c: fig_class_small_multiples(df, R["eda"]["fits"], c, f"figS{3 + i}_classes_{c}.png")
                       for i, c in enumerate(top3_r)}

    print("Clustering")
    cl = cached("clustering", lambda: clustering(df, top3_shap))
    fig_elbow(cl)
    fig_silhouette(cl)
    zoom = fig_cluster_spaces(df, cl)
    corr_tab = rc["table"]
    top5_desc = (corr_tab[corr_tab["index"].isin(top3_shap)].groupby("descriptor")["abs"].max()
                 .sort_values(ascending=False).head(5).index.tolist())
    s4, s5 = cluster_tables(df, cl, desc, top5_desc)
    s4.to_csv(TAB / "tableS4_cluster_statistics.csv", index=False)
    s5.to_csv(TAB / "tableS5_cluster_descriptors.csv", index=False)
    R["clustering"] = {
        "features": top3_shap, "k_opt": cl["k_opt"], "comparison": cl["comparison"],
        "silhouette_by_k": cl["silhouette_by_k"], "db_by_k": cl["db_by_k"],
        "sizes": pd.Series(cl["labels"]).value_counts().sort_index().tolist(),
        "mean_enthalpy": pd.Series(y).groupby(cl["labels"]).mean().tolist(),
        "representatives": cl["representatives"], "top5_descriptors": top5_desc, "zoom": zoom,
        "descriptor_means": {d: pd.Series(desc[d].values).groupby(cl["labels"]).mean().tolist() for d in top5_desc},
    }

    (OUT / "results.json").write_text(json.dumps(to_jsonable(R), indent=1, default=str), encoding="utf-8")
    print("done: results.json, figures/, tables/")


if __name__ == "__main__":
    main()
