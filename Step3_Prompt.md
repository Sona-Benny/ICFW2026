# Step-by-step PROMPT: Predicting Standard Combustion Enthalpy with Sombor-type Indices

Context: `ALL.csv` (3477 organic molecules) has columns `Name`, `SMILES`, `Enthalpy reported by Gharagheizi` (target, positive, no missing values), and eight Sombor-type indices: `SO`, `SOred`, `SOavg`, `mSO`, `ESO`, `EU`, `N`, `HSO`. The indices were computed on the molecular graph with explicit hydrogens (n = atoms, m = bonds), which is the same graph convention as the reference dataset.

Follow the reference article's workflow (Saviñon-Flores et al., ACS Omega 2025) exactly, but use the Sombor-type indices as predictors.

1. **Load and check the data.** Read `ALL.csv`. Confirm 3477 rows, no missing values in the eight index columns or the target, and that the target is strictly positive (needed for RMSLE). Print summary statistics.

2. **Exploratory analysis.** Compute the Pearson correlation of every index with the target and between indices. Report the strongest correlations and any multicollinearity (|r| > 0.95).

3. **Train/test split.** Split 70% training / 30% test with a fixed random seed (42). Do not touch the test set until step 7.

4. **Scaling.** Apply z-score standardization (zero mean, unit variance) to the predictors. Fit the scaler on the training subset only, then apply it to the test subset. Use a pipeline so cross-validation does not leak information.

5. **Model comparison with 10-fold CV on the training set.** Evaluate these regressors with default hyperparameters: Random Forest (baseline), Gradient Boosting, Extra Trees, Decision Tree, Ridge, and Support Vector Regression. Report mean ± standard deviation of R², MAE, MAPE, RMSE, and RMSLE across the folds. Pick the best model by validation RMSLE and R².

6. **Hyperparameter tuning.** For the selected model, run a randomized grid search with 5-fold cross-validation on the training set. Report the best parameters and the best CV score. Save the learning curve (training vs. validation score by training-set size).

7. **Final evaluation.** Refit the tuned model on the full training subset. Evaluate once on the 30% test set using R², MAE, MAPE, RMSE, and RMSLE. Produce parity plots (predicted vs. actual) and residual plots for both the training and test sets.

8. **Feature importance.** Report impurity-based feature importance and permutation importance on the test set. State which Sombor-type indices matter most.

9. **Benchmark against the reference predictors.** Using the same split and the same procedure, train the same final model on the reference article's topological indices (Degree, Betweenness, Closeness, Eigenvector centrality, Estrada, Wiener, Gutman) taken from the reference dataset in the same row order, and compare test metrics with the Sombor-type model. Then train on the combined feature set (Sombor + reference) and compare again.

10. **Report.** Give a results table with the test metrics for each feature set, the chosen model and its hyperparameters, and the top features. Save the results to `results_summary.csv` and the plots to a `figures/` folder.

Print the output of each stage so it can be checked before moving to the next one.
