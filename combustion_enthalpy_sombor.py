"""Step 4: run the step-by-step procedure in Step3_Prompt.md.

Predicts the combustion enthalpy from the Sombor-type indices in ALL.csv,
then benchmarks against the topological indices of the reference article.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

BASE = Path("c:/Users/joel.k/Documents/Sona")
DATA = BASE / "ALL.csv"
REFERENCE = Path("c:/Users/joel.k/sona_work/raw/All.csv")  # reference article dataset, used only in stage 9
FIG = BASE / "figures"
SEED = 42
TARGET = "Enthalpy reported by Gharagheizi"
SOMBOR = ["SO", "SOred", "SOavg", "mSO", "ESO", "EU", "N", "HSO"]
REF_INDICES = ["Degree", "Bet", "Eigen", "Closs", "Estrada", "Wiener", "Gutman"]

# validated light-mode palette (dataviz skill): slot 1 blue, slot 2 orange
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e6e3"
SURFACE = "#fcfcfb"


def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def all_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
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


def candidate_models():
    return {
        "RandomForest": RandomForestRegressor(random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
        "ExtraTrees": ExtraTreesRegressor(random_state=SEED),
        "DecisionTree": DecisionTreeRegressor(random_state=SEED),
        "Ridge": Ridge(),
        "SVR": SVR(),
    }


def make_pipe(estimator):
    # z-score scaling is fit inside each fold, so CV does not leak information
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


TREE_FOREST_SPACE = {
    "model__n_estimators": randint(100, 500),
    "model__max_depth": [None, 5, 10, 15, 20, 30],
    "model__min_samples_split": randint(2, 11),
    "model__min_samples_leaf": randint(1, 6),
    "model__max_features": ["sqrt", "log2", 0.5, 1.0],
}
PARAM_SPACES = {
    "RandomForest": TREE_FOREST_SPACE,
    "ExtraTrees": TREE_FOREST_SPACE,
    "GradientBoosting": {
        "model__n_estimators": randint(100, 500),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": randint(2, 6),
        "model__subsample": uniform(0.6, 0.4),
    },
    "DecisionTree": {
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__min_samples_split": randint(2, 11),
        "model__min_samples_leaf": randint(1, 6),
    },
    "Ridge": {"model__alpha": loguniform(1e-3, 1e2)},
    "SVR": {
        "model__C": loguniform(1e-1, 1e3),
        "model__gamma": loguniform(1e-3, 1e0),
        "model__epsilon": loguniform(1e0, 1e3),  # epsilon is in target units (kJ/mol scale)
    },
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def stage1_load():
    df = pd.read_csv(DATA)
    print(f"Stage 1: {df.shape[0]} rows, {df.shape[1]} columns")
    assert df[SOMBOR + [TARGET]].notna().all().all(), "missing values found"
    assert (df[TARGET] > 0).all(), "RMSLE needs a strictly positive target"
    print(df[SOMBOR + [TARGET]].describe().T.round(3).to_string())
    return df


def stage2_correlations(df):
    target_corr = df[SOMBOR].corrwith(df[TARGET])
    print("\nStage 2: Pearson r of each index with the target")
    print(target_corr.round(4).to_string())
    index_corr = df[SOMBOR].corr()
    pairs = [
        (a, b, round(index_corr.loc[a, b], 3))
        for i, a in enumerate(SOMBOR)
        for b in SOMBOR[i + 1 :]
        if abs(index_corr.loc[a, b]) > 0.95
    ]
    print(f"index pairs with |r| > 0.95: {len(pairs)} of {len(SOMBOR) * (len(SOMBOR) - 1) // 2}")
    print("  (indices are largely size-driven, so multicollinearity is expected)")


def stage3_split(df):
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.3, random_state=SEED)
    print(f"\nStage 3: train {len(train_idx)} rows, test {len(test_idx)} rows (70/30, seed {SEED})")
    return train_idx, test_idx


def stage5_compare(X_tr, y_tr):
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    rows = []
    for name, est in candidate_models().items():
        res = cross_validate(make_pipe(est), X_tr, y_tr, cv=cv, scoring=CV_SCORERS)
        row = {"Model": name}
        for metric in CV_SCORERS:
            vals = res[f"test_{metric}"]
            if metric != "R2":
                vals = -vals
            row[f"{metric} mean"] = vals.mean()
            row[f"{metric} sd"] = vals.std()
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Model")
    print("\nStage 5: 10-fold CV on the training set (Sombor-type features)")
    print(table.round(4).to_string())
    best = table["RMSLE mean"].idxmin()
    print(f"Selected model (lowest mean CV RMSLE): {best}")
    return best


def stage6_tune(model_name, X_tr, y_tr):
    search = RandomizedSearchCV(
        make_pipe(candidate_models()[model_name]),
        PARAM_SPACES[model_name],
        n_iter=30,
        scoring=RMSLE_SCORER,
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    print(f"\nStage 6: randomized search (30 iters, 5-fold CV) for {model_name}")
    print("best params:", search.best_params_)
    print(f"best CV RMSLE: {-search.best_score_:.4f}")
    sizes, tr_scores, va_scores = learning_curve(
        clone(search.best_estimator_),
        X_tr,
        y_tr,
        cv=5,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring=RMSLE_SCORER,
        n_jobs=-1,
    )
    curve = (sizes, -tr_scores.mean(axis=1), -va_scores.mean(axis=1))
    return search.best_estimator_, curve


def plot_learning_curve(model_name, curve, path):
    sizes, tr, va = curve
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    ax.plot(sizes, tr, color=BLUE, linewidth=1.5, marker="o", markersize=4, label="training")
    ax.plot(sizes, va, color=ORANGE, linewidth=1.5, marker="o", markersize=4, label="validation (5-fold)")
    ax.set_xlabel("training set size", color=INK_MUTED)
    ax.set_ylabel("RMSLE", color=INK_MUTED)
    ax.set_title(f"Learning curve: {model_name}", color=INK, loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_parity_residual(y_tr, p_tr, y_te, p_te, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), facecolor=SURFACE)
    for col, (name, y, p, colour) in enumerate([("train", y_tr, p_tr, BLUE), ("test", y_te, p_te, ORANGE)]):
        ax = axes[0, col]
        ax.scatter(y, p, s=10, color=colour, alpha=0.6, linewidths=0)
        lims = [min(y.min(), p.min()), max(y.max(), p.max())]
        ax.plot(lims, lims, color=INK_MUTED, linewidth=0.8, linestyle="-")
        ax.set_title(f"Parity: {name}", color=INK, loc="left", fontsize=10)
        ax.set_xlabel("actual enthalpy", color=INK_MUTED)
        ax.set_ylabel("predicted enthalpy", color=INK_MUTED)
        style_axes(ax)

        ax = axes[1, col]
        ax.scatter(p, y - p, s=10, color=colour, alpha=0.6, linewidths=0)
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.set_title(f"Residuals: {name}", color=INK, loc="left", fontsize=10)
        ax.set_xlabel("predicted enthalpy", color=INK_MUTED)
        ax.set_ylabel("actual - predicted", color=INK_MUTED)
        style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_permutation(imp, path):
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
    imp = imp.sort_values()
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    ax.set_xlabel("increase in test RMSLE when shuffled", color=INK_MUTED)
    ax.set_title("Permutation importance (test set)", color=INK, loc="left", fontsize=10)
    style_axes(ax)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stage7_evaluate(model, X_tr, y_tr, X_te, y_te):
    p_tr = model.predict(X_tr)
    p_te = model.predict(X_te)
    table = pd.DataFrame({"train": all_metrics(y_tr, p_tr), "test": all_metrics(y_te, p_te)})
    print("\nStage 7: final model metrics")
    print(table.round(4).to_string())
    plot_parity_residual(y_tr, p_tr, y_te, p_te, FIG / "parity_residual.png")
    return table


def stage8_importance(model, features, X_te, y_te):
    print("\nStage 8: feature importance")
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        imp = pd.Series(est.feature_importances_, index=features).sort_values(ascending=False)
        print("impurity importance:")
        print(imp.round(4).to_string())
    perm = permutation_importance(
        model, X_te, y_te, scoring=RMSLE_SCORER, n_repeats=20, random_state=SEED, n_jobs=-1
    )
    perm_imp = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
    print("permutation importance on test set (higher = more important):")
    print(perm_imp.round(4).to_string())
    plot_permutation(perm_imp, FIG / "permutation_importance.png")


def stage9_benchmark(model_name, full, train_idx, test_idx):
    feature_sets = {
        "Sombor-type (8)": SOMBOR,
        "Reference topological (7)": REF_INDICES,
        "Combined (15)": SOMBOR + REF_INDICES,
    }
    y_tr = full[TARGET].values[train_idx]
    y_te = full[TARGET].values[test_idx]
    rows = {}
    print("\nStage 9: benchmark (same split, same procedure, tuned per feature set)")
    for label, features in feature_sets.items():
        X = full[features].values
        model, _ = stage6_tune(model_name, X[train_idx], y_tr)
        rows[label] = all_metrics(y_te, model.predict(X[test_idx]))
    table = pd.DataFrame(rows).T
    print(table.round(4).to_string())
    return table


def main():
    FIG.mkdir(exist_ok=True)
    df = stage1_load()
    stage2_correlations(df)
    train_idx, test_idx = stage3_split(df)

    X = df[SOMBOR].values
    y = df[TARGET].values
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    model_name = stage5_compare(X_tr, y_tr)
    best, curve = stage6_tune(model_name, X_tr, y_tr)
    plot_learning_curve(model_name, curve, FIG / "learning_curve.png")

    metrics = stage7_evaluate(best, X_tr, y_tr, X_te, y_te)
    stage8_importance(best, SOMBOR, X_te, y_te)

    # reference dataset rows are in the same order as ALL.csv (checked by name)
    ref = pd.read_csv(REFERENCE)
    assert (ref["Name"].values == df["Name"].values).all(), "row order differs from reference"
    full = df.copy()
    for col in REF_INDICES:
        full[col] = ref[col].values
    bench = stage9_benchmark(model_name, full, train_idx, test_idx)

    summary = pd.concat(
        [metrics["test"].rename("Sombor-type model test").to_frame().T, bench],
        axis=0,
    )
    summary.to_csv(BASE / "results_summary.csv")
    print("\nStage 10: results written to results_summary.csv")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
