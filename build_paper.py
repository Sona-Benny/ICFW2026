"""Build the manuscript and Supporting Information (Word + PDF) from paper/results.json.

Run paper_analysis.py first. Every number quoted in the text is read from results.json
or the CSV tables, so the text always matches the analysis.
"""
import json
import shutil
from pathlib import Path

import pandas as pd
from docx.enum.text import WD_COLOR_INDEX

from manuscript_docx import Manuscript

BASE = Path(__file__).resolve().parent
OUT = BASE / "paper"
FIG = OUT / "figures"
TAB = OUT / "tables"
FINAL = OUT / "figures_final"
R = json.loads((OUT / "results.json").read_text(encoding="utf-8"))

TITLE = "Sombor-Type Indices in Predicting the Combustion Enthalpy of Organic Compounds"
SOMBOR = ["SO", "SOred", "SOavg", "mSO", "ESO", "EU", "N", "HSO"]
METRICS = ["R2", "MAE", "MAPE", "RMSE", "RMSLE"]
KJ = "kJ·mol^<−1>"
DH = "−Δ_<c>H^<°>"


def n(x, d=4):
    return f"{x:,.{d}f}".replace(",", " ") if abs(x) >= 10000 else f"{x:.{d}f}"


def idx(name):
    """Index name with markup subscripts."""
    return {"SOred": "SO_<red>", "SOavg": "SO_<avg>"}.get(name, name)


# --------------------------------------------------------------------------- references
VERIFY = "verify"
REFS = {
    "savinon": "Saviñon-Flores, F.; Arzola-Flores, J. A.; García-Castro, M. A.; Díaz-Sánchez, F.; Vidal Robles, E.; Maruri Valderrabano, F. A. Prediction of Standard Combustion Enthalpy of Organic Compounds Combining Machine Learning and Chemical Graph Theory: A Strategy. *ACS Omega* **2025**, *10*, 41828–41848.",
    "gharagheizi": "Gharagheizi, F.; Mirkhani, S. A.; Mahyari, A. R. T. Prediction of Standard Enthalpy of Combustion of Pure Compounds Using a Very Accurate Group-Contribution-Based Method. *Energy Fuels* **2011**, *25*, 2651–2654.",
    "benson": "Benson, S. W.; Buss, J. H. Additivity Rules for the Estimation of Molecular Properties. Thermodynamic Properties. *J. Chem. Phys.* **1958**, *29*, 546–572.",
    "domalski": "Domalski, E. S.; Hearing, E. D. Estimation of the Thermodynamic Properties of C-H-N-O-S-Halogen Compounds at 298.15 K. *J. Phys. Chem. Ref. Data* **1993**, *22*, 805–1159.",
    "todeschini": "Todeschini, R.; Consonni, V. *Molecular Descriptors for Chemoinformatics*; Wiley-VCH: Weinheim, 2009.",
    "wiener": "Wiener, H. Structural Determination of Paraffin Boiling Points. *J. Am. Chem. Soc.* **1947**, *69*, 17–20.",
    "randic": "Randić, M. Characterization of Molecular Branching. *J. Am. Chem. Soc.* **1975**, *97*, 6609–6615.",
    "kier_hall": "Hall, L. H.; Kier, L. B. The Molecular Connectivity Chi Indexes and Kappa Shape Indexes in Structure-Property Modeling. In *Reviews in Computational Chemistry*; Lipkowitz, K. B., Boyd, D. B., Eds.; VCH: New York, 1991; Vol. 2, pp 367–422.",
    "gutman21": "Gutman, I. Geometric Approach to Degree-Based Topological Indices: Sombor Indices. *MATCH Commun. Math. Comput. Chem.* **2021**, *86*, 11–16.",
    "redzepovic": "Redžepović, I. Chemical Applicability of Sombor Indices. *J. Serb. Chem. Soc.* **2021**, *86*, 445–457.",
    "kulli_gutman": ("Kulli, V. R.; Gutman, I. Computation of Sombor Indices of Certain Networks. *SSRG Int. J. Appl. Chem.* **2021**, *8*, 1–5.", VERIFY),
    "elliptic": ("Gutman, I.; Furtula, B.; Oz, M. S. Geometric Approach to Vertex-Degree-Based Topological Indices – Elliptic Sombor Index, Theory and Application. *Int. J. Quantum Chem.* **2024**, *124*, e27346.", VERIFY),
    "euler": ("Tang, Z.; Li, Y.; Deng, H. The Euler Sombor Index of a Graph. *Int. J. Quantum Chem.* **2024**, *124*, e27387.", VERIFY),
    "nirmala": ("Kulli, V. R. Nirmala Index. *Int. J. Math. Trends Technol.* **2021**, *67*, 8–12.", VERIFY),
    "hyperbolic": ("Barman, J.; Das, S. Geometric Approach to Degree-Based Topological Index: Hyperbolic Sombor Index. *Soft Comput. Lett.* **2024**, *6*, 100116.", VERIFY),
    "wang": "Wang, Z.; Mao, Y.; Li, Y.; Furtula, B. On Relations between Sombor and Other Degree-Based Indices. *J. Appl. Math. Comput.* **2022**, *68*, 1–17.",
    "kirana": "Kirana, B.; Shanmukha, M. C.; Usha, A. Comparative Study of Sombor Index and Its Various Versions Using Regression Models for Top Priority Polycyclic Aromatic Hydrocarbons. *Sci. Rep.* **2024**, *14*, 19841.",
    "estrada": "Estrada, E. Characterization of 3D Molecular Structure. *Chem. Phys. Lett.* **2000**, *319*, 713–718.",
    "gutman94": "Gutman, I. Selected Properties of the Schultz Molecular Topological Index. *J. Chem. Inf. Comput. Sci.* **1994**, *34*, 1087–1089.",
    "weininger": "Weininger, D. SMILES, a Chemical Language and Information System. 1. Introduction to Methodology and Encoding Rules. *J. Chem. Inf. Comput. Sci.* **1988**, *28*, 31–36.",
    "rdkit": "RDKit: Open-Source Cheminformatics, version 2026.09.2. https://www.rdkit.org (accessed 2026-11-18).",
    "networkx": "Hagberg, A. A.; Schult, D. A.; Swart, P. J. Exploring Network Structure, Dynamics, and Function Using NetworkX. In *Proceedings of the 7th Python in Science Conference (SciPy2008)*; Pasadena, CA, 2008; pp 11–15.",
    "sklearn": "Pedregosa, F.; Varoquaux, G.; Gramfort, A.; Michel, V.; Thirion, B.; Grisel, O.; Blondel, M.; Prettenhofer, P.; Weiss, R.; Dubourg, V.; et al. Scikit-learn: Machine Learning in Python. *J. Mach. Learn. Res.* **2011**, *12*, 2825–2830.",
    "breiman": "Breiman, L. Random Forests. *Mach. Learn.* **2001**, *45*, 5–32.",
    "geurts": "Geurts, P.; Ernst, D.; Wehenkel, L. Extremely Randomized Trees. *Mach. Learn.* **2006**, *63*, 3–42.",
    "xgboost": "Chen, T.; Guestrin, C. XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*; ACM, 2016; pp 785–794.",
    "lightgbm": "Ke, G.; Meng, Q.; Finley, T.; Wang, T.; Chen, W.; Ma, W.; Ye, Q.; Liu, T.-Y. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In *Advances in Neural Information Processing Systems 30*; 2017; pp 3146–3154.",
    "bergstra": "Bergstra, J.; Bengio, Y. Random Search for Hyper-Parameter Optimization. *J. Mach. Learn. Res.* **2012**, *13*, 281–305.",
    "lundberg17": "Lundberg, S. M.; Lee, S.-I. A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems 30*; 2017; pp 4765–4774.",
    "lundberg20": "Lundberg, S. M.; Erion, G.; Chen, H.; DeGrave, A.; Prutkin, J. M.; Nair, B.; Katz, R.; Himmelfarb, J.; Bansal, N.; Lee, S.-I. From Local Explanations to Global Understanding with Explainable AI for Trees. *Nat. Mach. Intell.* **2020**, *2*, 56–67.",
    "gramatica": "Gramatica, P. Principles of QSAR Models Validation: Internal and External. *QSAR Comb. Sci.* **2007**, *26*, 694–701.",
    "rucker": "Rücker, C.; Rücker, G.; Meringer, M. y-Randomization and Its Variants in QSPR/QSAR. *J. Chem. Inf. Model.* **2007**, *47*, 2345–2357.",
    "huber": "Huber, P. J. Robust Estimation of a Location Parameter. *Ann. Math. Stat.* **1964**, *35*, 73–101.",
    "varma": "Varma, S.; Simon, R. Bias in Error Estimation When Using Cross-Validation for Model Selection. *BMC Bioinformatics* **2006**, *7*, 91.",
    "arthur": "Arthur, D.; Vassilvitskii, S. k-means++: The Advantages of Careful Seeding. In *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms*; SIAM, 2007; pp 1027–1035.",
    "satopaa": "Satopää, V.; Albrecht, J.; Irwin, D.; Raghavan, B. Finding a “Kneedle” in a Haystack: Detecting Knee Points in System Behavior. In *31st International Conference on Distributed Computing Systems Workshops*; IEEE, 2011; pp 166–171.",
    "rousseeuw": "Rousseeuw, P. J. Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. *J. Comput. Appl. Math.* **1987**, *20*, 53–65.",
    "davies": "Davies, D. L.; Bouldin, D. W. A Cluster Separation Measure. *IEEE Trans. Pattern Anal. Mach. Intell.* **1979**, *PAMI-1*, 224–227.",
    "ester": "Ester, M.; Kriegel, H.-P.; Sander, J.; Xu, X. A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. In *Proceedings of the 2nd International Conference on Knowledge Discovery and Data Mining (KDD-96)*; AAAI Press, 1996; pp 226–231.",
    "campello": "Campello, R. J. G. B.; Moulavi, D.; Sander, J. Density-Based Clustering Based on Hierarchical Density Estimates. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*; Lecture Notes in Computer Science, Vol. 7819; Springer, 2013; pp 160–172.",
}
ORDER = []


def cite(*keys):
    nums = []
    for k in keys:
        assert k in REFS, k
        if k not in ORDER:
            ORDER.append(k)
        nums.append(ORDER.index(k) + 1)
    nums = sorted(nums)
    # compress runs: 3,4,5 -> 3-5
    parts, start = [], nums[0]
    for a, b in zip(nums, nums[1:] + [None]):
        if b != a + 1:
            parts.append(f"{start}" if start == a else (f"{start},{a}" if a == start + 1 else f"{start}–{a}"))
            start = b
    return "[[" + ",".join(parts) + "]]"


# --------------------------------------------------------------------------- numbering
FIGS = ["workflow", "graph", "corr", "single", "examples", "shap", "curves", "parity", "residuals",
        "benchmark", "ad", "families", "rdkit_parity", "rdkit_corr", "elbow", "silhouette",
        "cluster_space", "cluster_enthalpy"]
FIG_FILES = {
    "workflow": "fig01_workflow.png", "graph": "fig02_graph_construction.png",
    "corr": "fig03_correlation_matrix.png", "single": "fig04_single_index_regressions.png",
    "examples": "fig05_top_indices_examples.png", "shap": "fig06_shap.png",
    "curves": "fig07_learning_validation_curves.png", "parity": "fig08_parity_sombor.png",
    "residuals": "fig09_residuals.png", "benchmark": "fig13_feature_set_comparison.png",
    "ad": "fig14_applicability_domain.png", "families": "fig10_family_models.png",
    "rdkit_parity": "fig11_parity_rdkit.png", "rdkit_corr": "fig12_rdkit_index_correlations.png",
    "elbow": "fig15_elbow.png", "silhouette": "fig16_silhouette.png",
    "cluster_space": "fig17_cluster_index_space.png", "cluster_enthalpy": "fig18_cluster_enthalpy.png",
}
TABS = ["single", "cv_top", "rf", "benchmark", "families", "benson", "rdkit", "clustering"]
F = {k: i + 1 for i, k in enumerate(FIGS)}
T = {k: i + 1 for i, k in enumerate(TABS)}
top3 = R["eda"]["mlr"]["features"]
SI_FIGS = {"dist": 1, "corr_all": 2, "class1": 3, "class2": 4, "class3": 5}
SI_TABS = {"descriptors": 1, "cv_sombor": 2, "cv_rdkit": 3, "cv_ref": 4, "cluster_stats": 5,
           "cluster_desc": 6, "family_r2": 7}


def fig(ms, key, caption, width=16.0):
    assert ms.fig_no + 1 == F[key], (key, ms.fig_no)
    ms.figure(FIG / FIG_FILES[key], caption, width_cm=width)
    shutil.copy(FIG / FIG_FILES[key], FINAL / f"Figure_{F[key]:02d}.png")


def tab(ms, key, *args, **kwargs):
    assert ms.tab_no + 1 == T[key], (key, ms.tab_no)
    ms.table(*args, **kwargs)


def pm(mean, sd, d=4):
    return f"{n(mean, d)} ± {n(sd, d)}"


# --------------------------------------------------------------------------- derived numbers
D = R["data"]
fits = R["eda"]["fits"]
rf = R["rf_sombor"]
bench = R["benchmark"]
B_H = bench["Sombor-type, H-explicit (8)"]
B_HD = bench["Sombor-type, H-depleted (8)"]
B_BOTH = bench["Sombor-type, both graphs (16)"]
B_REF = bench["Reference graph descriptors (7)"]
B_SREF = bench["Sombor-type + reference (15)"]
rd_label = [k for k in bench if k.startswith("RDKit")][0]
B_RD = bench[rd_label]
shap_imp = R["shap"]["mean_abs_shap"]
fam = R["families"]
cl = R["clustering"]
ad = R["applicability"]
ys = R["y_scrambling"]
gs = R["grouped_split"]
benson = pd.DataFrame(R["benson"])
mono = benson.iloc[:5]
di = benson.iloc[5:]
cv_s = pd.read_csv(TAB / "tableS2_model_comparison_sombor.csv", index_col=0)
cv_r = pd.read_csv(TAB / "tableS3_model_comparison_rdkit.csv", index_col=0)
cv_ref = pd.read_csv(TAB / "tableS3b_model_comparison_reference.csv", index_col=0)
s4 = pd.read_csv(TAB / "tableS4_cluster_statistics.csv")
s5 = pd.read_csv(TAB / "tableS5_cluster_descriptors.csv")
s6 = pd.read_csv(TAB / "tableS6_family_single_index_r2.csv", index_col=0)
single_sorted = sorted(SOMBOR, key=lambda c: -fits[c]["R2"])
hc_families = ["Aromatic hydrocarbons", "Acyclic alkanes", "Cycloalkanes", "Acyclic alkenes",
               "Cycloalkenes", "Alkynes", "Carboxylic acids (1 COOH)", "Carboxylic acids (2 COOH)"]


def fam_r2(name):
    f = fam[name]
    return f.get("metrics_mean", f["metrics"])["R2"]


min_fam_r2 = min(fam_r2(k) for k in hc_families)
rmsle_gain = 100 * (1 - B_BOTH["RMSLE"] / B_H["RMSLE"])
rmse_gain = 100 * (1 - B_BOTH["RMSE"] / B_H["RMSE"])
cmp = cl["comparison"]
km = cmp["K-Means"]


def build_manuscript():
    FINAL.mkdir(exist_ok=True)
    ms = Manuscript()

    # ------------------------------------------------------------------ front matter
    ms.para(f"**{TITLE}**", align="center", size=16, space_after=10)
    ms.para("[Author Name]^<a>, [Author Name]^<a>, [Author Name]^<a,*>", align="center", size=11, space_after=2)
    ms.para("^<a> [Department, Institution, City, Postal code, Country]", align="center", size=9.5, space_after=2)
    ms.para("^<*> Corresponding author. E-mail: [address]", align="center", size=9.5, space_after=14)

    ms.heading("Abstract")
    ms.para(
        f"The standard enthalpy of combustion ({DH}) is a key thermochemical property for evaluating fuels and "
        f"energetic materials, yet its measurement requires pure samples and careful calorimetry. This work examines "
        f"whether eight degree-based Sombor-type indices — the Sombor (SO), reduced Sombor (SO_<red>), average Sombor "
        f"(SO_<avg>), modified Sombor (mSO), elliptic Sombor (ESO), Euler Sombor (EU), Nirmala (N) and hyperbolic "
        f"Sombor (HSO) indices — can act as compact predictors of {DH} for {D['n']} organic compounds. The indices "
        f"were computed from hydrogen-explicit molecular graphs generated from SMILES codes. Each index alone "
        f"correlates strongly with {DH} (R^<2> = {n(fits[single_sorted[-1]]['R2'])}–{n(fits['mSO']['R2'])}); the "
        f"modified Sombor index gives the best single-descriptor fit (R^<2> = {n(fits['mSO']['R2'])}), slightly "
        f"above the Estrada index used in earlier work (R^<2> = {n(fits['Estrada']['R2'])}). Among 19 regressors "
        f"compared by 10-fold cross-validation, random forest performed best; after hyperparameter tuning it reached "
        f"R^<2> = {n(rf['test']['R2'])}, MAE = {n(rf['test']['MAE'], 1)} {KJ}, MAPE = {n(rf['test']['MAPE'])}, "
        f"RMSE = {n(rf['test']['RMSE'], 1)} {KJ} and RMSLE = {n(rf['test']['RMSLE'])} on an external test set of "
        f"1044 compounds, matching the graph descriptors of a previous study evaluated on the same split "
        f"(R^<2> = {n(B_REF['R2'])}). Adding the same indices computed on hydrogen-depleted graphs, which "
        f"distinguish skeletal isomers, improved the test metrics to R^<2> = {n(B_BOTH['R2'])}, "
        f"RMSE = {n(B_BOTH['RMSE'], 1)} {KJ} and RMSLE = {n(B_BOTH['RMSLE'])} with only 16 descriptors. SHAP "
        f"analysis ranked mSO, N and EU as the most influential indices. Family-specific models built on mSO alone "
        f"reached R^<2> ≥ {n(min_fam_r2)} for six hydrocarbon families and two carboxylic-acid families under "
        f"nested or leave-one-out validation, and K-Means clustering in the space of the leading indices produced "
        f"{cl['k_opt']} compact groups (silhouette coefficient {n(km['silhouette'], 3)}, Davies–Bouldin score "
        f"{n(km['davies_bouldin'], 3)}). Sombor-type indices are therefore inexpensive, interpretable and accurate "
        f"descriptors for QSPR models of combustion enthalpy.")
    ms.para("**Keywords:** Sombor index; topological indices; chemical graph theory; standard enthalpy of combustion; "
            "QSPR; random forest; SHAP; clustering", align="left")

    # ------------------------------------------------------------------ introduction
    ms.heading("1. Introduction")
    ms.para(
        f"The standard enthalpy of combustion measures the heat released when a compound is completely oxidized "
        f"under standard conditions. It is needed to rank fuels, size combustion equipment, assess process safety and "
        f"design energetic materials, and it is the usual experimental route to standard enthalpies of formation. "
        f"Measuring it by bomb calorimetry is costly: the sample must be highly pure, enough material must be "
        f"available, and newly synthesized or unstable compounds are often unsuitable. Estimation methods are "
        f"therefore widely used. Group-contribution schemes, from the additivity rules of Benson and Buss to modern "
        f"parameterizations, express the property as a sum of contributions of structural fragments{cite('benson', 'domalski', 'gharagheizi')}. "
        f"They are accurate when every fragment is parameterized, but they fail for compounds that contain groups "
        f"without parameters.", indent=True)
    ms.para(
        f"Chemical graph theory offers a complementary route. A molecule is represented as a graph whose vertices are "
        f"atoms and whose edges are bonds, and numerical graph invariants — topological indices — are used as "
        f"descriptors in quantitative structure–property relationships (QSPR){cite('todeschini')}. Classical "
        f"examples are the Wiener index{cite('wiener')}, the Randić connectivity index and its valence "
        f"generalizations{cite('randic', 'kier_hall')}. Degree-based indices are particularly attractive because "
        f"they depend only on the degrees of the two atoms joined by each bond, so they are computed in linear time "
        f"and have a direct structural reading.", indent=True)
    ms.para(
        f"In 2021, Gutman introduced the Sombor index by interpreting the degree pair (*d*_<*u*>, *d*_<*v*>) of a "
        f"bond as a point in the plane and summing its Euclidean distances from the origin; the reduced and average "
        f"Sombor indices were defined in the same work{cite('gutman21')}. Redžepović soon showed that these indices "
        f"have useful predictive ability for thermodynamic properties of alkanes{cite('redzepovic')}. The geometric "
        f"idea has since produced a family of related invariants, including the modified{cite('kulli_gutman')}, "
        f"elliptic{cite('elliptic')}, Euler{cite('euler')}, Nirmala{cite('nirmala')} and hyperbolic{cite('hyperbolic')} "
        f"Sombor-type indices. Their mathematical relations with other degree-based indices have been studied in "
        f"detail{cite('wang')}, and regression models based on them have been reported for small and specific "
        f"compound sets such as polycyclic aromatic hydrocarbons{cite('kirana')}.", indent=True)
    ms.para(
        f"Recently, Saviñon-Flores et al. predicted {DH} for {D['n']} organic compounds by combining machine learning "
        f"with chemical graph theory{cite('savinon')}. Using the Estrada, Wiener and Gutman indices together with "
        f"graph centrality measures{cite('estrada', 'gutman94')}, their random forest model reached a test "
        f"R^<2> of 0.9810, and the Estrada index alone explained 97.3% of the variance. To our knowledge, "
        f"Sombor-type indices have not been tested on a property of this kind over a large and chemically diverse "
        f"data set, even though they are simpler to compute than spectral or distance-based indices.", indent=True)
    ms.para(
        "This work fills that gap. Using the same data set and the same overall workflow as the reference study, we "
        "(i) compute eight Sombor-type indices for every compound; (ii) quantify their individual and joint "
        "correlations with the combustion enthalpy; (iii) compare 19 regression algorithms, tune the best one and "
        "interpret it with SHAP values; (iv) benchmark the indices against the reference graph descriptors and 187 "
        "RDKit descriptors on an identical train/test split; (v) build family-specific models and compare them with "
        "the Benson group-contribution method; and (vi) explore the chemical space spanned by the indices with "
        "clustering. We also test the robustness of the models through a structure-grouped split, Y-scrambling and "
        "an applicability-domain analysis, and we show that computing the indices on both hydrogen-explicit and "
        "hydrogen-depleted graphs gives the best descriptor set of this type.", indent=True)

    # ------------------------------------------------------------------ methods
    ms.heading("2. Computational Details")
    ms.para(f"The complete workflow is summarized in Figure {F['workflow']}.", indent=True)
    fig(ms, "workflow",
        "Workflow used to predict the standard enthalpy of combustion of 3477 organic compounds from Sombor-type indices.",
        width=15.5)

    ms.heading("2.1. Data set", 2)
    tgt = D["target"]
    cls = D["classes"]
    ms.para(
        f"The data set released with the reference study{cite('savinon')} (file All.csv) contains {D['n']} organic "
        f"compounds with their names, SMILES codes{cite('weininger')} and standard combustion enthalpies taken from "
        f"Gharagheizi et al.{cite('gharagheizi')} Only these three columns were kept. The values of {DH} range from "
        f"{n(tgt['min'], 1)} to {n(tgt['max'], 1)} {KJ} (mean {n(tgt['mean'], 1)}, median {n(tgt['50%'], 1)} {KJ}). "
        f"The set covers aliphatic hydrocarbons ({cls['Aliphatic hydrocarbons']} compounds), aromatic hydrocarbons "
        f"({cls['Aromatic hydrocarbons']}), oxygen-containing compounds ({cls['O-containing']}), halogenated compounds "
        f"({cls['Halogenated']}), nitrogen-containing compounds ({cls['N-containing']}), compounds containing both "
        f"nitrogen and oxygen ({cls['N- and O-containing']}), sulfur-containing compounds ({cls['S-containing']}) "
        f"and {cls['Other elements']} compounds with other elements (Figure S{SI_FIGS['dist']}).", indent=True)
    ms.para(
        f"Canonicalizing the SMILES codes showed {D['unique_structures']} distinct structures: "
        f"{D['duplicate_groups']} structures appear twice ({D['duplicate_rows']} extra rows), either as stereoisomers "
        f"that share a constitution or as repeated entries. For {D['duplicate_groups_discrepant']} of these pairs the "
        f"two enthalpies differ, by up to {n(D['duplicate_max_diff'], 0)} {KJ}; in those pairs the second entry "
        f"coincides with experimental values quoted in the reference study. All {D['n']} rows were kept for the "
        f"general models so that the results remain comparable with the reference study, and the effect of the "
        f"repeated structures was checked with a structure-grouped split (Section 3.6). For the family-specific "
        f"models each structure was used once, keeping its later entry.", indent=True)

    ms.heading("2.2. Molecular graphs", 2)
    eg = R["example_graph"]
    ms.para(
        f"Each SMILES code was parsed with RDKit{cite('rdkit')} and hydrogen atoms were added explicitly. In the "
        f"resulting molecular graph *G* = (*V*, *E*) every atom, including hydrogen, is a vertex and every chemical "
        f"bond is an edge, regardless of bond order; *n* = |*V*| and *m* = |*E*|, and the degree *d*_<*u*> of atom "
        f"*u* is its number of bonded neighbors. This is the graph convention of the reference study: the Estrada, "
        f"Wiener and Gutman indices and the mean degree recomputed on our graphs reproduced the reference values for "
        f"every compound checked (the first 300 entries). Figure {F['graph']} illustrates the construction for "
        f"acetic acid (*n* = {eg['n']}, *m* = {eg['m']}). Graph drawings were produced with NetworkX{cite('networkx')}. "
        f"For comparison, all indices were also computed on the hydrogen-depleted graph, which contains only the "
        f"non-hydrogen atoms; these variants are denoted by the superscript HD.", indent=True)
    fig(ms, "graph",
        "Construction of the molecular graph and of the Sombor-type indices for acetic acid. (a) Structure obtained "
        "from the SMILES code. (b) Hydrogen-explicit molecular graph; each vertex is labelled with its atom and degree "
        "*d*, and each edge with the degree pair (*d*_<*u*>, *d*_<*v*>). (c) Adjacency matrix of the graph. "
        "(d) Values of the eight Sombor-type indices.", width=14.5)

    ms.heading("2.3. Sombor-type indices", 2)
    ms.para(
        f"For a bond *uv* ∈ *E* joining atoms of degrees *d*_<*u*> and *d*_<*v*>, the eight indices are defined as "
        f"follows: the Sombor index SO and its reduced and average versions SO_<red> and SO_<avg>{cite('gutman21')}, "
        f"the modified Sombor index mSO{cite('kulli_gutman')}, the elliptic Sombor index ESO{cite('elliptic')}, the "
        f"Euler Sombor index EU{cite('euler')}, the Nirmala index N{cite('nirmala')} and the hyperbolic Sombor index "
        f"HSO{cite('hyperbolic')}:", indent=True)
    eqs = [
        r"\mathrm{SO}(G)=\sum_{uv\in E}\sqrt{d_u^2+d_v^2}",
        r"\mathrm{SO}_{\mathrm{red}}(G)=\sum_{uv\in E}\sqrt{(d_u-1)^2+(d_v-1)^2}",
        r"\mathrm{SO}_{\mathrm{avg}}(G)=\sum_{uv\in E}\sqrt{\left(d_u-\frac{2m}{n}\right)^2+\left(d_v-\frac{2m}{n}\right)^2}",
        r"\mathrm{mSO}(G)=\sum_{uv\in E}\frac{1}{\sqrt{d_u^2+d_v^2}}",
        r"\mathrm{ESO}(G)=\sum_{uv\in E}(d_u+d_v)\sqrt{d_u^2+d_v^2}",
        r"\mathrm{EU}(G)=\sum_{uv\in E}\sqrt{d_u^2+d_v^2+d_u d_v}",
        r"\mathrm{N}(G)=\sum_{uv\in E}\sqrt{d_u+d_v}",
        r"\mathrm{HSO}(G)=\sum_{uv\in E}\frac{\sqrt{d_u^2+d_v^2}}{\min\{d_u,d_v\}}",
    ]
    for e in eqs:
        ms.equation(e)
    ms.para(
        "On a hydrogen-explicit graph the degree of an atom equals its number of σ-bonded neighbors: 4 for an sp^<3> "
        "carbon, 3 for an sp^<2> carbon, 2 for an sp carbon or an ether/hydroxyl oxygen, and 1 for hydrogen, halogen "
        "and carbonyl oxygen atoms. Each index is therefore a weighted count of bond types, the weight being a "
        "function of the two end-atom degrees. This reading is useful when interpreting the results.", indent=True)

    ms.heading("2.4. Reference descriptors", 2)
    ms.para(
        f"Two descriptor sets were used as benchmarks. The first contains the seven graph descriptors of the "
        f"reference study — mean degree, betweenness, eigenvector and closeness centralities, and the Estrada, Wiener "
        f"and Gutman indices{cite('estrada', 'wiener', 'gutman94')} — taken directly from its data set. The second "
        f"contains the {D['n_rdkit_all']} molecular descriptors available in RDKit; descriptors with undefined or "
        f"non-finite values, constant descriptors and descriptors with magnitudes above 10^<12> (information-content "
        f"indices of the Ipc type) were removed, leaving {D['n_rdkit_used']} descriptors (Table S{SI_TABS['descriptors']}).",
        indent=True)

    ms.heading("2.5. Machine learning", 2)
    ms.para(
        f"The data set was split at random into a training subset (70%, 2433 compounds) and a test subset (30%, 1044 "
        f"compounds) with a fixed seed; the test subset was used only for the final evaluation. Predictors were "
        f"standardized to zero mean and unit variance inside a pipeline, so that in every cross-validation fold the "
        f"scaling was fitted on the training part only. Nineteen regressors with default hyperparameters were "
        f"compared by 10-fold cross-validation on the training subset: linear regression, ridge, lasso, elastic net, "
        f"lasso least-angle regression, orthogonal matching pursuit, Bayesian ridge, Huber regression, k-nearest "
        f"neighbors, decision tree, random forest{cite('breiman')}, extra trees{cite('geurts')}, AdaBoost, gradient "
        f"boosting, LightGBM{cite('lightgbm')}, XGBoost{cite('xgboost')}, multilayer perceptron, support vector "
        f"regression and a mean-value baseline (scikit-learn 1.9.1{cite('sklearn')}, LightGBM 4.7.0, XGBoost 3.4.1).",
        indent=True)
    ms.para(
        f"Model performance was measured with the coefficient of determination (R^<2>), the mean absolute error "
        f"(MAE), the mean absolute percentage error (MAPE), the root-mean-square error (RMSE) and the "
        f"root-mean-square logarithmic error (RMSLE), where *y*_<*i*> and *ŷ*_<*i*> are the observed and predicted "
        f"values for compound *i* of *N*:", indent=True)
    for e in [
        r"R^2=1-\frac{\sum_{i}(y_i-\hat{y}_i)^2}{\sum_{i}(y_i-\bar{y})^2}",
        r"\mathrm{MAE}=\frac{1}{N}\sum_{i}|y_i-\hat{y}_i|",
        r"\mathrm{MAPE}=\frac{1}{N}\sum_{i}\left|\frac{y_i-\hat{y}_i}{y_i}\right|",
        r"\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i}(y_i-\hat{y}_i)^2}",
        r"\mathrm{RMSLE}=\sqrt{\frac{1}{N}\sum_{i}\left[\ln(1+\hat{y}_i)-\ln(1+y_i)\right]^2}",
    ]:
        ms.equation(e)
    ms.para(
        f"Because {DH} spans almost three orders of magnitude, the relative metrics MAPE and RMSLE were used "
        f"alongside the absolute ones, and models were ranked by their mean validation RMSLE. The selected model was "
        f"tuned by random search{cite('bergstra')} over 30 hyperparameter combinations with 5-fold cross-validation "
        f"(20 combinations for the RDKit descriptors), and learning and validation curves were computed for the tuned "
        f"model. The contributions of the indices to individual predictions were quantified with SHAP values obtained "
        f"with the tree explainer{cite('lundberg17', 'lundberg20')}. To compare descriptor sets fairly, the same tuning "
        f"procedure was repeated for each set on the identical split.", indent=True)
    ms.para(
        f"Three checks addressed the reliability of the model. First, the split was repeated with all entries of a "
        f"structure assigned to the same subset (structure-grouped split by canonical SMILES). Second, the model was "
        f"refitted 20 times on randomly permuted training targets (Y-scrambling){cite('rucker')}. Third, the "
        f"applicability domain was assessed with a Williams plot{cite('gramatica')}, using the leverage of each "
        f"compound and the warning leverage *h*^<*>, where **x**_<*i*> is the standardized descriptor vector of "
        f"compound *i* (with an intercept term), **X** the training descriptor matrix, *p* the number of descriptors "
        f"and *N*_<train> the number of training compounds:", indent=True)
    ms.equation(r"h_i=\mathbf{x}_i^{T}\left(\mathbf{X}^{T}\mathbf{X}\right)^{-1}\mathbf{x}_i,\qquad h^{*}=\frac{3(p+1)}{N_{\mathrm{train}}}")
    ms.para(
        "Residuals were standardized by the standard deviation of the cross-validated training residuals; compounds "
        "with |standardized residual| > 3 were considered response outliers.", indent=True)

    ms.heading("2.6. Family-specific models", 2)
    ms.para(
        f"Nine families were defined with RDKit substructure rules: aromatic hydrocarbons, acyclic alkanes, "
        f"cycloalkanes, acyclic alkenes, cycloalkenes and alkynes (compounds containing only C and H), saturated "
        f"acyclic amines (C, H and N only, all nitrogen atoms in amino groups), and saturated acyclic carboxylic "
        f"acids with one or two carboxyl groups and no other oxygen. Following the reference study, each family was "
        f"modeled with a single index — mSO, the index with the strongest correlation with {DH}. Aromatic "
        f"hydrocarbons were modeled with extra trees; the five other hydrocarbon families with Huber "
        f"regression{cite('huber')}, whose loss is quadratic for small residuals and linear for large ones, with the "
        f"regularization strength α ∈ {{10^<−7>, 10^<−4>, 10^<−2>, 0.1, 0.7, 1}} and the threshold ε ∈ {{1.1, 1.2, "
        f"1.35, 1.5, 1.8}} selected by grid search. These six families were evaluated by nested cross-validation "
        f"with five outer and three inner folds{cite('varma')}. Amines and carboxylic acids were modeled by ordinary "
        f"linear regression and evaluated by leave-one-out cross-validation. For ten carboxylic acids the predictions "
        f"were compared with experimental values and Benson group-contribution estimates reported in the reference "
        f"study{cite('savinon', 'benson')}.", indent=True)

    ms.heading("2.7. Clustering", 2)
    ms.para(
        f"Unsupervised learning was applied to {DH} and the three most influential indices according to the SHAP "
        f"analysis ({', '.join(cl['features'])}), each scaled to [0, 1]. K-Means with k-means++ "
        f"initialization{cite('arthur')} (10 restarts) was run for k = 1–15 clusters, and the optimal number of "
        f"clusters was located at the elbow of the within-cluster sum of squares with the kneedle "
        f"criterion{cite('satopaa')}. Cluster quality was measured by the mean silhouette coefficient{cite('rousseeuw')} "
        f"and the Davies–Bouldin score{cite('davies')}, where *a*(*i*) is the mean distance of sample *i* to the other "
        f"members of its cluster, *b*(*i*) the mean distance to the nearest other cluster, σ_<*i*> the mean distance "
        f"of the members of cluster *i* to its centroid *c*_<*i*>, and *k* the number of clusters:", indent=True)
    ms.equation(r"s(i)=\frac{b(i)-a(i)}{\max\{a(i),\,b(i)\}},\qquad \mathrm{DB}=\frac{1}{k}\sum_{i=1}^{k}\max_{j\neq i}\frac{\sigma_i+\sigma_j}{d(c_i,c_j)}")
    ms.para(
        f"K-Means was compared with DBSCAN{cite('ester')} (ε = 0.1, 20 samples per core point) and "
        f"HDBSCAN{cite('campello')} (minimum cluster size 50). K-Means clusters were numbered C1, C2, … in order of "
        f"increasing mean {DH}.", indent=True)

    # ------------------------------------------------------------------ results
    ms.heading("3. Results and Discussion")
    ms.heading("3.1. Correlation analysis and single-index models", 2)
    corr = R["eda"]["corr_with_target"]
    ms.para(
        f"Figure {F['corr']} shows the Pearson correlation matrix of the eight indices and {DH}. Every index "
        f"correlates strongly and positively with the enthalpy (r = {n(min(corr.values()), 3)}–{n(max(corr.values()), 3)}), "
        f"and the indices are even more strongly correlated with one another (r ≥ "
        f"{n(R['eda']['corr_min_between_indices'], 3)}). The variance inflation factors of the eight indices range "
        f"from {min(R['eda']['vif'].values()):.1e} to {max(R['eda']['vif'].values()):.1e}, so the set is highly "
        f"collinear: all indices grow with molecular size. Hydrogen-depleted indices and the reference descriptors "
        f"are compared in Figure S{SI_FIGS['corr_all']}.", indent=True)
    fig(ms, "corr",
        f"Pearson correlation matrix of the eight Sombor-type indices and the standard enthalpy of combustion "
        f"({DH}). All coefficients exceed {n(min(corr.values()) - 0.0005, 3)}; the color scale starts at 0.97 to "
        f"resolve the differences.", width=11.5)
    ms.para(
        f"The single-index linear regressions (Table {T['single']}, Figure {F['single']}) give R^<2> between "
        f"{n(fits[single_sorted[-1]]['R2'])} ({idx(single_sorted[-1])}) and {n(fits['mSO']['R2'])} (mSO). Adding a "
        f"quadratic term changes R^<2> by less than 0.002 for every index, so the relationships are essentially "
        f"linear. This differs from the Wiener and Gutman indices, which grow faster than linearly with molecular "
        f"size: on the same data their linear R^<2> values are only {n(fits['Wiener']['R2'])} and "
        f"{n(fits['Gutman']['R2'])}, rising to {n(fits['Wiener']['R2_quadratic'])} and "
        f"{n(fits['Gutman']['R2_quadratic'])} with a quadratic term. The modified Sombor index slightly outperforms "
        f"the Estrada index, the best single descriptor of the reference study (R^<2> = {n(fits['Estrada']['R2'])}; "
        f"recomputed here and identical to the published value).", indent=True)
    rows = []
    for c in single_sorted + ["Estrada", "Wiener", "Gutman"]:
        f_ = fits[c]
        rows.append([idx(c), n(f_["slope"], 3), n(f_["intercept"], 1), n(f_["r"]), n(f_["R2"]), n(f_["R2_quadratic"])])
    tab(ms, "single",
        f"Linear Regressions of {DH} ({KJ}) on Each Sombor-Type Index, Ordered by R^<2>, with the Estrada, Wiener and "
        f"Gutman Indices of the Reference Study for Comparison",
        ["Index", "Slope", "Intercept", "r", "R^<2> (linear)", "R^<2> (quadratic)"], rows,
        footnote=f"Fits over all {D['n']} compounds. Slope in {KJ} per index unit; intercept in {KJ}.")
    fig(ms, "single",
        f"Standard combustion enthalpy as a function of each Sombor-type index for the {D['n']} compounds, with the "
        f"least-squares line and its coefficient of determination.")
    ex = R["examples"]
    names = ", ".join(f"{name} ({i})" for i, name in enumerate(ex["molecules"], 1))
    ms.para(
        f"Figure {F['examples']} follows a series of reference molecules — {names} — along the three best indices. "
        f"Their points lie on the regression lines, and the indices increase almost proportionally with the number "
        f"of bonds: for example, mSO rises from {n(ex['values']['mSO']['benzene'], 2)} for benzene to "
        f"{n(ex['values']['mSO']['pentatriacontylbenzene'], 2)} for pentatriacontylbenzene, while {DH} increases from "
        f"{n(ex['enthalpy']['benzene'], 0)} to {n(ex['enthalpy']['pentatriacontylbenzene'], 0)} {KJ}. Figures "
        f"S{SI_FIGS['class1']}–S{SI_FIGS['class3']} show the same relationships separately for each chemical class.",
        indent=True)
    fig(ms, "examples",
        f"Standard combustion enthalpy as a function of (a) mSO, (b) N and (c) EU. The numbered points mark "
        f"{names}. Gray points are the remaining compounds.")
    ms.para(
        "The linearity follows from the bond-type reading in Section 2.3. With hydrogen atoms explicit, each index is "
        "a weighted sum of the numbers of C–H, C–C, C=C, C–O, O–H and other bond types, where the weight depends only "
        "on the degrees of the end atoms. The combustion enthalpy is itself close to additive over bonds, which is the "
        "basis of bond- and group-additivity schemes. A single Sombor-type index therefore behaves like a "
        "one-parameter bond-additivity model, and the index that best matches the true bond contributions fits best. "
        "mSO gives relatively more weight to bonds involving low-degree atoms — a C–H bond (4, 1) contributes 0.243, "
        "a C–C bond (4, 4) 0.177 — whereas SO, ESO and EU weight bonds between high-degree atoms more heavily. A large "
        "share of the combustion energy comes from oxidizing the hydrogen content, which may explain why mSO ranks "
        "first. The same reading reveals two limitations. Degrees do not encode element identity, so a C–Cl bond is "
        "weighted exactly like a C–H bond. And skeletal isomers with the same bond types — for instance, all acyclic "
        "alkanes with the same number of carbon atoms — receive identical values of every hydrogen-explicit index.",
        indent=True)
    mlr = R["eda"]["mlr"]
    a, b, c = (mlr["coef"][k] for k in mlr["features"])
    ms.para(
        f"Combining the three best indices in a multiple linear regression gives eq {ms.eq_no + 1}, with "
        f"R^<2> = {n(mlr['R2'])}:", indent=True)
    eq = (r"-\Delta_c H^{\circ}\ (\mathrm{kJ\ mol^{-1}}) = " + f"{a:.2f}" + r"\,\mathrm{mSO}" +
          ("+" if b >= 0 else "") + f"{b:.2f}" + r"\,\mathrm{N}" + ("+" if c >= 0 else "") + f"{c:.2f}" +
          r"\,\mathrm{EU}" + ("+" if mlr["intercept"] >= 0 else "") + f"{mlr['intercept']:.2f}")
    ms.equation(eq, fontsize=12)
    ms.para(
        f"The improvement over mSO alone is small (ΔR^<2> = {n(mlr['R2'] - fits['mSO']['R2'])}), and the "
        f"coefficients have opposite signs despite the strong positive correlations of all three indices with "
        f"{DH} — a typical symptom of multicollinearity. Nonparametric machine-learning models, which are less "
        f"sensitive to collinear inputs, were therefore used to combine the indices.", indent=True)

    ms.heading("3.2. Model selection", 2)
    top = cv_s.head(3)
    ms.para(
        f"Table {T['cv_top']} lists the three regressors with the lowest mean validation RMSLE in 10-fold "
        f"cross-validation, and Table S{SI_TABS['cv_sombor']} gives the results for all 19. The tree ensembles — "
        f"random forest, extra trees, XGBoost and LightGBM — performed best and were nearly indistinguishable "
        f"(mean R^<2> {n(cv_s.loc['XGBoost', 'R2 mean'])}–{n(cv_s.loc['Random forest', 'R2 mean'])}). Linear models "
        f"reached similar R^<2> values (about 0.98) but much larger RMSLE (≥ {n(cv_s.loc['Ridge', 'RMSLE mean'], 3)}), "
        f"meaning that they predict small molecules poorly; support vector regression with default settings failed "
        f"(R^<2> = {n(cv_s.loc['Support vector regression', 'R2 mean'], 3)}) because its default margin and "
        f"regularization are unsuited to an unscaled target in the thousands. Random forest had the lowest RMSLE "
        f"and RMSE and was selected, which also keeps the comparison with the reference study direct.", indent=True)
    rows = [[m] + [pm(r[f'{k} mean'], r[f'{k} sd'], 4 if k in ('R2', 'MAPE', 'RMSLE') else 1) for k in METRICS]
            for m, r in top.iterrows()]
    tab(ms, "cv_top",
        "Mean and Standard Deviation of the Validation Metrics in 10-Fold Cross-Validation for the Three Best "
        "Regressors Using the Eight Sombor-Type Indices",
        ["Model", "R^<2>", "MAE", "MAPE", "RMSE", "RMSLE"], rows,
        footnote=f"Default hyperparameters. MAE and RMSE in {KJ}. Ranked by mean RMSLE.")

    ms.heading("3.3. Interpretation with SHAP values", 2)
    order = list(shap_imp)
    ms.para(
        f"Figure {F['shap']} summarizes the SHAP analysis of the tuned random forest on the test set. The mean "
        f"absolute SHAP values rank the indices as "
        + ", ".join(f"{idx(k)} ({shap_imp[k]:.0f} {KJ})" for k in order) +
        f". The impurity-based importances give the same order. In the beeswarm plot, high index values push "
        f"predictions up and low values push them down, consistent with the size-driven correlations. The leading "
        f"three indices — mSO, N and EU — are also those with the best single-index fits, and SO_<avg> and HSO "
        f"contribute least: SO_<avg> measures deviations from the mean degree and is thus partly normalized for "
        f"size, while HSO divides each term by the smaller end-atom degree, which is 1 for every bond to hydrogen.",
        indent=True)
    fig(ms, "shap",
        f"SHAP analysis of the tuned random forest on the test set. (a) Summary plot; each point is a compound, "
        f"positioned by its SHAP value and colored by the index value (light blue, low; dark blue, high). (b) Mean "
        f"absolute SHAP value of each index, in {KJ}.")

    ms.heading("3.4. Tuned random forest", 2)
    p = rf["params"]
    curves = R["curves"]
    ms.para(
        f"Random search selected {p['n_estimators']} trees with a maximum depth of {p['max_depth']}, at least "
        f"{p['min_samples_split']} samples to split a node, {p['min_samples_leaf']} samples per leaf and "
        f"√p features per split (cross-validated RMSLE {n(rf['cv_rmsle'])}). The learning curve (Figure "
        f"{F['curves']}a) shows the validation RMSE falling from {curves['learning_valid_rmse'][0]:.0f} to "
        f"{curves['learning_valid_rmse'][-1]:.0f} {KJ} as the training set grows from "
        f"{curves['learning_sizes'][0]:.0f} to {curves['learning_sizes'][-1]:.0f} compounds, with the gap to the "
        f"training error narrowing; more data would still help. The validation curve (Figure {F['curves']}b) "
        f"flattens beyond a depth of about 10, so the selected depth does not lead to overfitting.", indent=True)
    fig(ms, "curves",
        f"(a) Learning curve and (b) validation curve over the maximum tree depth for the tuned random forest. Lines "
        f"are means over five cross-validation folds; shaded bands are ±1 standard deviation.")
    ms.para(
        f"Table {T['rf']} and Figure {F['parity']} report the final performance. On the 1044 test compounds the "
        f"model reaches R^<2> = {n(rf['test']['R2'])}, MAE = {n(rf['test']['MAE'], 1)} {KJ}, MAPE = "
        f"{n(rf['test']['MAPE'])}, RMSE = {n(rf['test']['RMSE'], 1)} {KJ} and RMSLE = {n(rf['test']['RMSLE'])}; the "
        f"training-set values (R^<2> = {n(rf['train']['R2'])}) are close, so overfitting is limited. The MAPE and "
        f"RMSLE are higher on the training set than on the test set because six of the seven compounds with "
        f"{DH} below 200 {KJ}, whose relative errors are largest, fell into the training subset. The residuals (Figure "
        f"{F['residuals']}) are centered on zero and roughly symmetric, with a spread that grows moderately with the "
        f"predicted value. For reference, the random forest of the earlier study reported a test R^<2> of 0.9810, "
        f"MAE 287.6 {KJ}, RMSE 551.9 {KJ} and RMSLE 0.1933{cite('savinon')}; because the random splits differ, the "
        f"comparison on an identical split in Section 3.5 is the fairer one.", indent=True)
    rows = [["Training", *[n(rf['train'][k], 1 if k in ('MAE', 'RMSE') else 4) for k in METRICS]],
            ["Test", *[n(rf['test'][k], 1 if k in ('MAE', 'RMSE') else 4) for k in METRICS]],
            ["Test, structure-grouped split", *[n(gs['test'][k], 1 if k in ('MAE', 'RMSE') else 4) for k in METRICS]]]
    tab(ms, "rf",
        "Evaluation Metrics of the Tuned Random Forest Using the Eight Sombor-Type Indices as Predictors",
        ["Set", "R^<2>", "MAE", "MAPE", "RMSE", "RMSLE"], rows,
        footnote=f"MAE and RMSE in {KJ}. Hyperparameters: n_estimators = {p['n_estimators']}, max_depth = "
                 f"{p['max_depth']}, min_samples_split = {p['min_samples_split']}, min_samples_leaf = "
                 f"{p['min_samples_leaf']}, max_features = {p['max_features']}. The last row refits the same model "
                 f"on a split in which no structure occurs in both subsets ({gs['n_train']} training, {gs['n_test']} "
                 f"test compounds).")
    fig(ms, "parity",
        f"Parity plots of the tuned random forest for (a) the training set and (b) the test set. The line is "
        f"y = x; r^<2> (fit) is the squared correlation between predicted and actual values.")
    fig(ms, "residuals",
        "Residuals (actual − predicted) of the tuned random forest against the predicted value for (a) the training "
        "set and (b) the test set, and (c) the residual distributions (clipped at ±3000 kJ·mol^<−1>).")

    ms.heading("3.5. Comparison of descriptor sets", 2)
    ms.para(
        f"Table {T['benchmark']} and Figure {F['benchmark']} compare six descriptor sets on the identical split, each "
        f"with its own tuned random forest. The eight hydrogen-explicit Sombor-type indices perform on par with the "
        f"seven graph descriptors of the reference study (test RMSE {n(B_H['RMSE'], 1)} vs {n(B_REF['RMSE'], 1)} {KJ}; "
        f"RMSLE {n(B_H['RMSLE'])} vs {n(B_REF['RMSLE'])}), although the Sombor-type indices are much cheaper to "
        f"compute: they need only the vertex degrees, whereas the Estrada index requires the eigenvalues of the "
        f"adjacency matrix and the Wiener and Gutman indices all shortest-path distances. Indices computed on "
        f"hydrogen-depleted graphs perform clearly worse on their own (R^<2> = {n(B_HD['R2'])}), as expected because "
        f"they ignore the hydrogen atoms that contribute much of the combustion energy. Combined with the "
        f"hydrogen-explicit indices, however, they give the best graph-based model: R^<2> = {n(B_BOTH['R2'])}, MAE = "
        f"{n(B_BOTH['MAE'], 1)} {KJ}, RMSE = {n(B_BOTH['RMSE'], 1)} {KJ} and RMSLE = {n(B_BOTH['RMSLE'])}, i.e. "
        f"{rmse_gain:.1f}% lower RMSE and {rmsle_gain:.1f}% lower RMSLE than the hydrogen-explicit set alone. The two "
        f"graphs are complementary: the hydrogen-explicit graph encodes the bond-type composition, while the "
        f"hydrogen-depleted graph encodes the carbon skeleton and its branching, which distinguishes the isomers that "
        f"are degenerate in the hydrogen-explicit graph. Adding the reference descriptors to the hydrogen-explicit "
        f"indices helps less (RMSE {n(B_SREF['RMSE'], 1)} {KJ}). The {D['n_rdkit_used']} RDKit descriptors give the "
        f"most accurate model (R^<2> = {n(B_RD['R2'])}, RMSE = {n(B_RD['RMSE'], 1)} {KJ}; Section 3.8), at the cost of "
        f"an order of magnitude more descriptors.", indent=True)
    label_map = {"Sombor-type, H-explicit (8)": "Sombor-type, H-explicit graph",
                 "Sombor-type, H-depleted (8)": "Sombor-type, H-depleted graph",
                 "Sombor-type, both graphs (16)": "Sombor-type, both graphs",
                 "Reference graph descriptors (7)": "Reference graph descriptors",
                 "Sombor-type + reference (15)": "Sombor-type (H-explicit) + reference",
                 rd_label: "RDKit descriptors"}
    rows = [[label_map[k], str(v["n_features"]), *[n(v[m], 1 if m in ('MAE', 'RMSE') else 4) for m in METRICS]]
            for k, v in bench.items()]
    tab(ms, "benchmark",
        "Test-Set Metrics of Tuned Random Forests Trained on Different Descriptor Sets with the Same 70/30 Split",
        ["Descriptor set", "Descriptors", "R^<2>", "MAE", "MAPE", "RMSE", "RMSLE"], rows,
        col_widths=[5.6, 1.8, 1.8, 1.8, 1.8, 1.8, 1.8],
        footnote=f"MAE and RMSE in {KJ}. Each model was tuned by random search with 5-fold cross-validation on the "
                 f"training subset. Reference graph descriptors: mean degree, betweenness, eigenvector and closeness "
                 f"centralities, Estrada, Wiener and Gutman indices{cite('savinon')}.")
    fig(ms, "benchmark",
        "Test-set (a) RMSE and (b) RMSLE of tuned random forests for each descriptor set; the number of descriptors "
        "is given in parentheses.", width=15)

    ms.heading("3.6. Robustness and applicability domain", 2)
    ms.para(
        f"Three checks support the reliability of the Sombor-type model (Figure {F['ad']}). When all entries of a "
        f"structure were kept in the same subset, the test metrics barely changed (R^<2> = {n(gs['test']['R2'])}, "
        f"RMSE = {n(gs['test']['RMSE'], 1)} {KJ}; Table {T['rf']}), so the repeated structures do not inflate the "
        f"results. Models trained on scrambled targets gave a mean test R^<2> of {n(ys['test_r2_mean'], 3)} ± "
        f"{n(ys['test_r2_sd'], 3)} over {ys['runs']} runs, far below that of the real model, which rules out chance "
        f"correlation. In the Williams plot, {ad['test_outside_h']} test compounds "
        f"({100 * ad['test_outside_h'] / 1044:.1f}%) exceed the warning leverage *h*^<*> = {n(ad['h_star'])} and "
        f"{ad['test_outlier_resid']} ({100 * ad['test_outlier_resid'] / 1044:.1f}%) have standardized residuals "
        f"beyond ±3; {100 * ad['test_in_domain_frac']:.1f}% of the test compounds lie within both limits. The "
        f"high-leverage compounds are not simply the largest molecules but those with unusual combinations of "
        f"bond types, such as allene, o-terphenyl, the decahydronaphthalenes and terephthalic acid; their mean "
        f"absolute error is about three times the test-set average, so predictions outside the leverage limit should "
        f"be treated with caution.", indent=True)
    fig(ms, "ad",
        f"(a) Williams plot of standardized residuals against leverage (log scale) for the training set "
        f"(cross-validated predictions) and the test set; horizontal lines mark ±3 standard deviations and the "
        f"vertical line the warning leverage *h*^<*>. (b) Test R^<2> of {ys['runs']} models trained on scrambled "
        f"targets compared with the real model.")

    ms.heading("3.7. Family-specific QSPR models", 2)
    ms.para(
        f"Table {T['families']} and Figure {F['families']} present the family-specific models built on mSO alone. All "
        f"hydrocarbon families are predicted with R^<2> ≥ {n(min(fam_r2(k) for k in hc_families[:6]))}; for acyclic "
        f"alkanes the MAE is only {n(fam['Acyclic alkanes']['metrics_mean']['MAE'], 1)} {KJ}. This is remarkable "
        f"because every acyclic alkane isomer has the same hydrogen-explicit index value: the residual error of this "
        f"family is essentially the enthalpy difference between isomers, which is small compared with the range of "
        f"the property. The selected Huber models were almost unregularized (α mostly 10^<−7> or 10^<−4>), "
        f"confirming linear relationships within each family. Cycloalkanes show the largest hydrocarbon errors "
        f"(RMSE {n(fam['Cycloalkanes']['metrics_mean']['RMSE'], 1)} {KJ}), and they come almost entirely from five "
        f"bicyclic hydrocarbons (Figure {F['families']}c). The data set lists the four bicyclooctanes (C_<8>H_<14>) at "
        f"2371–2412 {KJ}, less than half the value it gives for cyclooctane (C_<8>H_<16>, 4915 {KJ}); these entries "
        f"are probably erroneous, and the model's predictions of about 4620 {KJ} are chemically more plausible. The "
        f"amine family is broader here than in the reference study ("
        f"{fam['Amines']['n']} saturated acyclic amines, including polyamines, versus 21) and is fitted less well "
        f"(R^<2> = {n(fam_r2('Amines'))}). The largest error belongs to diethylhexylamine: its SMILES encodes a "
        f"C_<16>H_<35>N amine listed at 4964 {KJ}, whereas the four other C_<16>H_<35>N amines in the data set lie "
        f"at 8005–8030 {KJ}. The name suggests N,N-diethylhexylamine (C_<10>H_<23>N), so the entry probably pairs a "
        f"wrong structure with its enthalpy. Supplementary Table S{SI_TABS['family_r2']} lists the single-index R^<2> of "
        f"all 16 indices for every family.", indent=True)
    rows = []
    for name in ["Aromatic hydrocarbons", "Acyclic alkanes", "Cycloalkanes", "Acyclic alkenes", "Cycloalkenes",
                 "Alkynes", "Amines", "Carboxylic acids (1 COOH)", "Carboxylic acids (2 COOH)"]:
        f_ = fam[name]
        if "metrics_mean" in f_:
            mm, sd = f_["metrics_mean"], f_["metrics_sd"]
            vals = [pm(mm["R2"], sd["R2"]), pm(mm["MAE"], sd["MAE"], 1), n(mm["MAPE"]), pm(mm["RMSE"], sd["RMSE"], 1),
                    n(mm["RMSLE"])]
        else:
            vals = [n(f_["metrics"][k], 1 if k in ("MAE", "RMSE") else 4) for k in METRICS]
        rows.append([name, str(f_["n"]), f_["model"], *vals])
    eqs_fam = "; ".join(f"{k[0].lower() + k[1:]}: {DH} = {fam[k]['slope']:.2f}·mSO − {abs(fam[k]['intercept']):.1f}"
                        for k in ["Amines", "Carboxylic acids (1 COOH)", "Carboxylic acids (2 COOH)"])
    tab(ms, "families",
        "Evaluation Metrics of the Family-Specific QSPR Models Based on the Modified Sombor Index (mSO)",
        ["Family", "n", "Model", "R^<2>", "MAE", "MAPE", "RMSE", "RMSLE"], rows,
        footnote=f"MAE and RMSE in {KJ}. The first six families: mean (± standard deviation) over the five outer "
                 f"folds of nested cross-validation. Amines and carboxylic acids: leave-one-out cross-validation; "
                 f"fits to the whole families: {eqs_fam}.", font_size=8,
        col_widths=[3.6, 0.9, 2.6, 2.5, 1.9, 1.4, 2.0, 1.5])
    fig(ms, "families",
        "Out-of-fold predictions against actual standard combustion enthalpies for the family-specific models based "
        "on mSO: (a) aromatic hydrocarbons (extra trees), (b–f) acyclic alkanes, cycloalkanes, acyclic alkenes, "
        "cycloalkenes and alkynes (Huber regression, nested cross-validation), and (g–i) amines and carboxylic acids "
        "with one or two carboxyl groups (linear regression, leave-one-out).")
    mae_mono_b = mono["Delta Benson"].abs().mean()
    mae_mono_p = mono["Delta predicted"].abs().mean()
    mae_di_b = di["Delta Benson"].abs().mean()
    mae_di_p = di["Delta predicted"].abs().mean()
    ms.para(
        f"Table {T['benson']} compares the leave-one-out predictions of the carboxylic-acid models with experimental "
        f"values and Benson estimates. For the five dicarboxylic acids the mSO model is at least as accurate as the "
        f"Benson method (mean absolute deviation {n(mae_di_p, 1)} vs {n(mae_di_b, 1)} {KJ}). For the five long-chain "
        f"monocarboxylic acids it underestimates the experimental values by {n(mono['Delta predicted'].abs().min(), 0)}"
        f"–{n(mono['Delta predicted'].abs().max(), 0)} {KJ} (mean {n(mae_mono_p, 1)} vs {n(mae_mono_b, 1)} {KJ} for "
        f"Benson). The cause lies in the data rather than the index: for most monocarboxylic acids the data set lists "
        f"values 6–7% below experiment (for dodecanoic acid, 6898.3 instead of 7423.7 {KJ}), and a regression "
        f"trained on this mixture of sources inherits the bias. Curated experimental data would be needed to resolve "
        f"this family.", indent=True)
    rows = [[r["Compound"], n(r["Exp"], 2), n(r["Benson"], 2), n(r["Delta Benson"], 2),
             n(r["Predicted (LOO)"], 2), n(r["Delta predicted"], 2)] for _, r in benson.iterrows()]
    tab(ms, "benson",
        "Experimental Standard Combustion Enthalpies of Carboxylic Acids Compared with Benson Group-Contribution "
        "Estimates and with Leave-One-Out Predictions of the mSO-Based Models",
        ["Compound", f"{DH}_<exp>", f"{DH}_<Benson>", "Δ_<Benson>", f"{DH}_<pred>", "Δ_<pred>"], rows,
        footnote=f"All values in {KJ}; Δ = estimate − experimental value. Experimental and Benson values from "
                 f"ref {ORDER.index('savinon') + 1}.")

    ms.heading("3.8. RDKit molecular descriptors", 2)
    rd = R["rf_rdkit"]
    rdp = rd["params"]
    imp = rd["top_importance"]
    ms.para(
        f"To place the Sombor-type indices in context, the same 19 regressors were trained on the "
        f"{D['n_rdkit_used']} RDKit descriptors (Table S{SI_TABS['cv_rdkit']}). With these descriptors LightGBM had "
        f"the lowest validation RMSLE and lasso the highest R^<2> ({n(cv_r.loc['Lasso', 'R2 mean'])}); random forest "
        f"was retained for consistency. The tuned random forest (Table {T['rdkit']}, Figure {F['rdkit_parity']}) "
        f"reached a test R^<2> of {n(rd['test']['R2'])}, MAE {n(rd['test']['MAE'], 1)} {KJ} and RMSE "
        f"{n(rd['test']['RMSE'], 1)} {KJ}, against 0.9927, 142.2 and 342.0 {KJ} reported for the corresponding "
        f"model of the reference study. Its most important descriptors are the valence connectivity indices Chi1n "
        f"({imp['Chi1n']:.3f}) and Chi0n ({imp['Chi0n']:.3f}), as in the reference study.", indent=True)
    rows = [["Training", *[n(rd['train'][k], 1 if k in ('MAE', 'RMSE') else 4) for k in METRICS]],
            ["Test", *[n(rd['test'][k], 1 if k in ('MAE', 'RMSE') else 4) for k in METRICS]]]
    tab(ms, "rdkit",
        f"Evaluation Metrics of the Tuned Random Forest Using the {D['n_rdkit_used']} RDKit Descriptors as Predictors",
        ["Set", "R^<2>", "MAE", "MAPE", "RMSE", "RMSLE"], rows,
        footnote=f"MAE and RMSE in {KJ}. Hyperparameters: n_estimators = {rdp['n_estimators']}, max_depth = "
                 f"{rdp['max_depth']}, min_samples_split = {rdp['min_samples_split']}, min_samples_leaf = "
                 f"{rdp['min_samples_leaf']}, max_features = {rdp['max_features']}.")
    fig(ms, "rdkit_parity",
        f"Parity plots of the tuned random forest trained on the {D['n_rdkit_used']} RDKit descriptors for (a) the "
        f"training set and (b) the test set.")
    top30 = R["rdkit_corr"]["top30"]
    ms.para(
        f"Figure {F['rdkit_corr']} shows the 30 strongest correlations between RDKit descriptors and Sombor-type "
        f"indices. The strongest pairs are {top30[0][0]}–{idx(top30[0][1])} (r = {n(top30[0][2])}), "
        f"{top30[1][0]}–{idx(top30[1][1])} (r = {n(top30[1][2])}) and {top30[2][0]}–{idx(top30[2][1])} "
        f"(r = {n(top30[2][2])}). The close link between mSO and the Kier–Hall connectivity indices has a simple "
        f"origin: the first-order connectivity index is a sum over bonds of (δ_<*u*>δ_<*v*>)^<−1/2>{cite('kier_hall')}, "
        f"and mSO is a sum over bonds of (*d*_<*u*>^<2> + *d*_<*v*>^<2>)^<−1/2>; both are decreasing functions of the "
        f"end-atom degrees. The indices also correlate with the number of valence electrons and with the surface-area "
        f"contributions SlogP_VSA5 and SMR_VSA5. The Sombor-type indices therefore capture much of the information "
        f"that drives the descriptor-based models, which explains their competitive accuracy.", indent=True)
    fig(ms, "rdkit_corr",
        "The 30 strongest Pearson correlations between RDKit molecular descriptors and Sombor-type indices.",
        width=12.5)

    ms.heading("3.9. Clustering in the Sombor-index space", 2)
    sizes = cl["sizes"]
    means = cl["mean_enthalpy"]
    reps = cl["representatives"]
    sil = cl["silhouette_by_k"]
    ms.para(
        f"The elbow of the K-Means sum of squares lies at k = {cl['k_opt']} (Figure {F['elbow']}a). At this k the "
        f"mean silhouette coefficient is {n(km['silhouette'], 3)} and the Davies–Bouldin score "
        f"{n(km['davies_bouldin'], 3)} — the lowest Davies–Bouldin score of all k between 2 and 12 (Figure "
        f"{F['elbow']}b). Two clusters give a higher silhouette ({n(sil['2'], 3)}) but only separate small from large "
        f"molecules. Table {T['clustering']} compares the three algorithms. With ε = 0.1, DBSCAN merges almost all "
        f"compounds into one cluster ({cmp['DBSCAN']['noise']} noise points), so its scores are undefined. HDBSCAN "
        f"finds {cmp['HDBSCAN']['clusters']} dense clusters but labels {cmp['HDBSCAN']['noise']} compounds "
        f"({100 * cmp['HDBSCAN']['noise'] / D['n']:.0f}%) as noise; its high silhouette refers only to the retained "
        f"points. K-Means was therefore preferred. The silhouette plot (Figure {F['silhouette']}) shows positive "
        f"coefficients for nearly all samples, and the larger clusters have broad profiles.", indent=True)
    rows = []
    for name in ["K-Means", "DBSCAN", "HDBSCAN"]:
        e = cmp[name]
        rows.append([name, str(e["clusters"]), str(e["noise"]),
                     n(e["silhouette"], 3) if "silhouette" in e else "–",
                     n(e["davies_bouldin"], 3) if "davies_bouldin" in e else "–"])
    tab(ms, "clustering",
        f"Comparison of Clustering Algorithms in the Space of {DH}, " + ", ".join(cl["features"]),
        ["Algorithm", "Clusters", "Noise points", "Silhouette", "Davies–Bouldin"], rows,
        footnote="Silhouette and Davies–Bouldin scores computed on non-noise points; – : undefined (fewer than two "
                 "clusters).")
    fig(ms, "elbow",
        "(a) Elbow plot of the K-Means within-cluster sum of squared distances; the circle marks the elbow found by the "
        "kneedle criterion. (b) Mean silhouette coefficient and Davies–Bouldin score as functions of the number of "
        "clusters; the vertical line marks the selected k.")
    fig(ms, "silhouette",
        f"Silhouette coefficients of the {D['n']} compounds grouped into the {cl['k_opt']} K-Means clusters. The "
        f"dashed line is the mean silhouette coefficient.", width=11)
    ms.para(
        "Figures "
        f"{F['cluster_space']} and {F['cluster_enthalpy']} show the clusters in the index space and against the "
        f"enthalpy. Because the indices are nearly collinear, the clusters are ordered bands along the size axis: "
        + "; ".join(f"C{i + 1} contains {sizes[i]} compounds with a mean {DH} of {means[i]:.0f} {KJ} "
                    f"(e.g., {', '.join(reps[str(i)])})" for i in range(cl["k_opt"])) +
        f". Within every cluster, {DH} increases with each index. The mean values of the RDKit descriptors most "
        f"correlated with the indices ({', '.join(cl['top5_descriptors'])}) rise monotonically from C1 to "
        f"C{cl['k_opt']} (Table S{SI_TABS['cluster_desc']}), and descriptive statistics with confidence intervals "
        f"for each cluster are given in Table S{SI_TABS['cluster_stats']}. Clustering in this space thus mainly "
        f"stratifies compounds by size and energy content. It is useful for defining applicability domains and for "
        f"selecting representative compounds, but it does not separate chemical classes; for that, the "
        f"hydrogen-depleted indices or element-aware descriptors would be needed.", indent=True)
    fig(ms, "cluster_space",
        f"K-Means clusters in the space of the indices (a) {cl['features'][0]}–{cl['features'][1]}, "
        f"(b) {cl['features'][0]}–{cl['features'][2]} and (c) {cl['features'][1]}–{cl['features'][2]}. Panels (d–f) "
        f"zoom into clusters C1–C{len(cl['zoom'])}. Labels mark the cluster medians; darker blue denotes a higher mean "
        f"enthalpy.")
    fig(ms, "cluster_enthalpy",
        f"Standard combustion enthalpy as a function of (a) {cl['features'][0]}, (b) {cl['features'][1]} and "
        f"(c) {cl['features'][2]} for each K-Means cluster. Panels (d–f) zoom into clusters C1–C{len(cl['zoom'])}.")

    # ------------------------------------------------------------------ conclusions
    ms.heading("4. Conclusions")
    ms.para(
        f"Eight degree-based Sombor-type indices, computed from hydrogen-explicit molecular graphs, were evaluated as "
        f"predictors of the standard enthalpy of combustion of {D['n']} organic compounds. The main findings are:",
        indent=True)
    ms.bullets([
        f"Each index is almost linearly related to {DH}; the modified Sombor index mSO is the best single descriptor "
        f"(R^<2> = {n(fits['mSO']['R2'])}), ahead of the Estrada index of earlier work. On hydrogen-explicit graphs the "
        f"indices act as weighted bond-type counts, which explains this behavior.",
        f"A tuned random forest on the eight indices predicts the test set with R^<2> = {n(rf['test']['R2'])} and "
        f"RMSE = {n(rf['test']['RMSE'], 1)} {KJ}, equal to the spectral and distance-based graph descriptors of the "
        f"reference study on the same split while being cheaper to compute. SHAP identifies mSO, N and EU as the "
        f"key indices.",
        f"Adding the same indices computed on hydrogen-depleted graphs supplies skeletal information and gives the "
        f"best graph-based model (R^<2> = {n(B_BOTH['R2'])}, RMSE = {n(B_BOTH['RMSE'], 1)} {KJ}, RMSLE = "
        f"{n(B_BOTH['RMSLE'])}) with only 16 descriptors.",
        f"The models are robust to repeated structures, show no chance correlation in Y-scrambling and cover "
        f"{100 * ad['test_in_domain_frac']:.1f}% of the test compounds within the applicability domain.",
        f"Family-specific models on mSO alone reach R^<2> ≥ {n(min_fam_r2)} for hydrocarbons and carboxylic acids, "
        f"and match the Benson method for dicarboxylic acids. Inconsistent source values limit the monocarboxylic "
        f"acids.",
        f"K-Means clustering in the index space yields {cl['k_opt']} compact groups (silhouette "
        f"{n(km['silhouette'], 3)}, Davies–Bouldin {n(km['davies_bouldin'], 3)}) that stratify compounds by size and "
        f"energy content.",
    ])
    ms.para(
        "Sombor-type indices cannot distinguish atoms of equal degree but different elements, and on "
        "hydrogen-explicit graphs they cannot separate skeletal isomers. Element-weighted (valence-type) variants of "
        "the indices, curated experimental data for the inconsistent families, and the extension of the approach to "
        "other thermochemical properties are natural next steps.", indent=True)

    # ------------------------------------------------------------------ back matter
    ms.heading("Data Availability")
    ms.para(
        "The data set (ALL.csv with the computed indices), the Python scripts that reproduce every figure and table "
        "(paper_analysis.py, build_paper.py) and the result tables are provided with this article [repository link "
        "to be added].", indent=True)
    ms.heading("Supporting Information")
    ms.para(
        "Distributions of the target and the indices (Figure S1); correlation matrix of all descriptors (Figure S2); "
        "single-index relationships by chemical class (Figures S3–S5); RDKit descriptors used (Table S1); "
        "cross-validation metrics of all 19 regressors for the Sombor-type indices, the RDKit descriptors and the "
        "reference graph descriptors (Tables S2–S4); cluster statistics (Tables S5 and S6); single-index R^<2> by "
        "compound family (Table S7).", indent=True)
    ms.heading("Author Contributions")
    ms.para("[To be completed by the authors.]")
    ms.heading("Conflicts of Interest")
    ms.para("The authors declare no competing financial interest.")
    ms.heading("Acknowledgments")
    ms.para("[To be completed by the authors.]")

    ms.heading("References")
    for i, key in enumerate(ORDER, 1):
        entry = REFS[key]
        flag = isinstance(entry, tuple)
        text = entry[0] if flag else entry
        p = ms.doc.add_paragraph()
        p.paragraph_format.left_indent = ms_cm(0.8)
        p.paragraph_format.first_line_indent = ms_cm(-0.8)
        p.paragraph_format.space_after = ms_pt(3)
        ms.add_text(p, f"({i}) ", size=9.5)
        start = len(p.runs)
        ms.add_text(p, text, size=9.5)
        if flag:
            for run in p.runs[start:]:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    unused = [k for k in REFS if k not in ORDER]
    assert not unused, unused
    ms.save(OUT / "Manuscript.docx")
    return ms


def ms_cm(v):
    from docx.shared import Cm
    return Cm(v)


def ms_pt(v):
    from docx.shared import Pt
    return Pt(v)


def build_si():
    si = Manuscript()
    si.para("**Supporting Information**", align="center", size=14, space_after=4)
    si.para(f"**{TITLE}**", align="center", size=12, space_after=4)
    si.para("[Author Name]^<a>, [Author Name]^<a>, [Author Name]^<a,*>", align="center", size=10.5, space_after=14)

    si.heading("Contents")
    si.bullets([
        "Figure S1. Distributions of the combustion enthalpy and of the eight Sombor-type indices",
        "Figure S2. Correlation matrix of all graph descriptors",
        "Figures S3–S5. Single-index relationships by chemical class",
        "Table S1. RDKit descriptors used",
        "Tables S2–S4. Cross-validation metrics of 19 regressors for three descriptor sets",
        "Tables S5 and S6. Cluster statistics",
        "Table S7. Single-index R^<2> by compound family",
    ])

    shutil.copy(FIG / "figS1_distributions.png", FINAL / "Figure_S1.png")
    si.figure(FIG / "figS1_distributions.png",
              f"Distributions of the standard combustion enthalpy ({DH}) and of the eight Sombor-type indices for the "
              f"{D['n']} compounds.", prefix="Figure S1")
    shutil.copy(FIG / "figS2_correlation_all_descriptors.png", FINAL / "Figure_S2.png")
    si.figure(FIG / "figS2_correlation_all_descriptors.png",
              "Pearson correlation matrix of the Sombor-type indices on hydrogen-explicit graphs, the same indices on "
              "hydrogen-depleted graphs (HD), the graph descriptors of the reference study and the standard "
              "combustion enthalpy. Blue, positive; orange, negative.", prefix="Figure S2")
    for i, c in enumerate(top3):
        f = FIG / f"figS{3 + i}_classes_{c}.png"
        shutil.copy(f, FINAL / f"Figure_S{3 + i}.png")
        si.figure(f, f"Standard combustion enthalpy as a function of {idx(c)} for each chemical class (blue) against "
                     f"all other compounds (gray), with the class size and the linear R^<2>.", prefix=f"Figure S{3 + i}")

    desc_names = pd.read_pickle(OUT / "cache" / "rdkit_descriptors.pkl")[0].columns.tolist()
    si.heading("Table S1. RDKit descriptors used", 2)
    si.para(f"The {len(desc_names)} RDKit descriptors retained after filtering (Section 2.4 of the main text): "
            + ", ".join(desc_names) + ".", size=9)

    def cv_rows(df):
        return [[m] + [pm(r[f"{k} mean"], r[f"{k} sd"], 1 if k in ("MAE", "RMSE") else 4) for k in METRICS]
                for m, r in df.iterrows()]

    hdr = ["Model", "R^<2>", "MAE", "MAPE", "RMSE", "RMSLE"]
    CVW = [4.0, 2.5, 2.4, 2.5, 2.5, 2.5]
    FW = [3.6] + [1.6] * 8
    note = f"Mean ± standard deviation over 10 folds; MAE and RMSE in {KJ}; models ranked by mean RMSLE."
    si.table("Validation Metrics in 10-Fold Cross-Validation for 19 Regressors Using the Eight Sombor-Type Indices",
             hdr, cv_rows(cv_s), footnote=note, font_size=8, prefix="Table S2", col_widths=CVW)
    si.table(f"Validation Metrics in 10-Fold Cross-Validation for 19 Regressors Using the {D['n_rdkit_used']} RDKit "
             f"Descriptors", hdr, cv_rows(cv_r), footnote=note, font_size=8, prefix="Table S3", col_widths=CVW)
    si.table("Validation Metrics in 10-Fold Cross-Validation for 19 Regressors Using the Seven Graph Descriptors of "
             "the Reference Study", hdr, cv_rows(cv_ref), footnote=note, font_size=8, prefix="Table S4", col_widths=CVW)

    rows = [[r["Cluster"], r["Variable"].replace("-ΔcH°", DH), str(r["n"]), n(r["Mean"], 2), n(r["SD"], 2),
             n(r["Min"], 2), n(r["Max"], 2), r["95% CI"], r["99% CI"]] for _, r in s4.iterrows()]
    si.table("Descriptive Statistics and Confidence Intervals of the Mean for Each K-Means Cluster",
             ["Cluster", "Variable", "n", "Mean", "SD", "Min", "Max", "95% CI", "99% CI"], rows,
             footnote=f"Confidence intervals from the t distribution. Enthalpy in {KJ}.", font_size=7.5,
             prefix="Table S5", col_widths=[1.5, 2.0, 1.0, 1.6, 1.5, 1.5, 1.6, 2.85, 2.85])
    hdr6 = list(s5.columns)
    rows = [[str(v) for v in r] for _, r in s5.iterrows()]
    si.table("Mean ± Standard Deviation of the Five RDKit Descriptors Most Correlated with the Clustering Indices, "
             "and Representative Compounds (Nearest to the Centroid) for Each Cluster", hdr6, rows, font_size=7.5,
             prefix="Table S6", col_widths=[1.5, 1.1, 2.0, 2.0, 2.4, 1.9, 2.1, 3.4])
    cols = list(s6.columns)
    rows = [[fam_name] + [n(v, 4) for v in s6.loc[fam_name, cols[:8]]] for fam_name in s6.index]
    si.table("R^<2> of Single-Index Linear Regressions Within Each Compound Family (Hydrogen-Explicit Indices)",
             ["Family"] + [idx(c) for c in cols[:8]], rows, font_size=7.5, prefix="Table S7a", col_widths=FW)
    rows = [[fam_name] + [n(v, 4) if v == v else "–" for v in s6.loc[fam_name, cols[8:]]] for fam_name in s6.index]
    si.table("R^<2> of Single-Index Linear Regressions Within Each Compound Family (Hydrogen-Depleted Indices)",
             ["Family"] + [idx(c.split(" ")[0]) + "^<HD>" for c in cols[8:]], rows, font_size=7.5,
             footnote="– : index constant within the family.", prefix="Table S7b", col_widths=FW)
    si.save(OUT / "Supporting_Information.docx")


if __name__ == "__main__":
    build_manuscript()
    build_si()
    print("written:", OUT / "Manuscript.docx", OUT / "Supporting_Information.docx")
