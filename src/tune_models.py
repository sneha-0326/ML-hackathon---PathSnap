import time
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from lightgbm import LGBMClassifier

from train_models import load_data, build_valid_edge_sets, evaluate_observation_level

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_tuning(n_trials: int = 25):
    print("Loading datasets for Bayesian Hyperparameter Optimisation...")
    train_df, test_df, X_train, y_train, groups_train, X_test, y_test, feature_cols = load_data()
    test_edge_sets = build_valid_edge_sets("student_test", test_df["trip_id"].values)

    # Load untuned baseline metrics from model_comparison.csv
    comp_df = pd.read_csv("reports/model_comparison.csv")
    untuned_row = comp_df[comp_df["Model"] == "LightGBM"].iloc[0]

    gkf = GroupKFold(n_splits=5)
    # Precompute splits for speed and exact reproducibility
    splits = list(gkf.split(X_train, y_train, groups_train))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "max_depth": trial.suggest_int("max_depth", 4, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 120),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        fold_f1s = []
        for tr_idx, val_idx in splits:
            clf = LGBMClassifier(**params)
            clf.fit(X_train[tr_idx], y_train[tr_idx])
            preds = clf.predict(X_train[val_idx])
            fold_f1s.append(f1_score(y_train[val_idx], preds, average="macro"))

        mean_f1 = float(np.mean(fold_f1s))
        print(f"  Trial {trial.number + 1:02d}/{n_trials:02d} -> CV Macro F1 = {mean_f1:.4f}")
        return mean_f1

    t0 = time.perf_counter()
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    tuning_time_s = time.perf_counter() - t0

    print("\n>>> BAYESIAN OPTIMISATION COMPLETE <<<")
    print(f"Total Trials       : {n_trials}")
    print(f"Tuning Time        : {tuning_time_s:.1f}s")
    print(f"Best CV Macro F1   : {study.best_value:.4f} (Untuned: {untuned_row['CV_Macro_F1']:.4f})")
    print(f"Best Hyperparams   : {study.best_params}")

    # Save full optimisation trial history (required by Assessment & Plagiarism Policy)
    trials_df = study.trials_dataframe()
    trials_df.to_csv("reports/bayesian_tuning_trials.csv", index=False)

    # Plot optimisation history for Section 4 of the report
    plt.figure(figsize=(8, 4.2))
    plt.plot(trials_df["number"] + 1, trials_df["value"], marker="o", alpha=0.65, label="Trial CV Macro F1")
    plt.plot(trials_df["number"] + 1, trials_df["value"].cummax(), color="#16a34a", linewidth=2.2, label="Best So Far")
    plt.axhline(untuned_row["CV_Macro_F1"], color="#dc2626", linestyle="--", label="Untuned Baseline")
    plt.title("Optuna Bayesian Hyperparameter Optimisation History (LightGBM)")
    plt.xlabel("Trial Number")
    plt.ylabel("5-Fold GroupCV Macro F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/bayesian_optimization_history.png", dpi=200)
    plt.close()

    # Refit best tuned model on all 80 training trips
    best_model = LGBMClassifier(**study.best_params, random_state=42, n_jobs=-1, verbose=-1)
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, "models/LightGBM_tuned.joblib")

    # Evaluate tuned model on 10 student-testing trips
    t0 = time.perf_counter()
    test_probs = best_model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    infer_time = time.perf_counter() - t0

    tuned_test_f1 = float(f1_score(y_test, test_preds, average="macro"))
    tuned_logloss = float(log_loss(y_test, test_probs))
    tuned_top1, tuned_cont = evaluate_observation_level(test_df, test_probs, test_edge_sets)

    comparison = pd.DataFrame([
        {
            "Stage": "Untuned LightGBM",
            "CV_Macro_F1": untuned_row["CV_Macro_F1"],
            "Student_Test_Top1_Acc": untuned_row["Top1_Accuracy"],
            "Student_Test_Macro_F1": untuned_row["Student_Test_Macro_F1"],
            "Route_Continuity": untuned_row["Route_Continuity"],
            "Log_Loss": untuned_row["Log_Loss"],
            "Inference_Time_s": untuned_row["Inference_Time_s"],
        },
        {
            "Stage": "Bayesian-Tuned LightGBM",
            "CV_Macro_F1": round(study.best_value, 4),
            "Student_Test_Top1_Acc": round(tuned_top1 * 100, 2),
            "Student_Test_Macro_F1": round(tuned_test_f1, 4),
            "Route_Continuity": round(tuned_cont * 100, 2),
            "Log_Loss": round(tuned_logloss, 4),
            "Inference_Time_s": round(infer_time, 4),
        },
    ])
    comparison.to_csv("reports/untuned_vs_tuned_comparison.csv", index=False)
    print("\n>>> UNTUNED VS. TUNED COMPARISON (Saved to reports/untuned_vs_tuned_comparison.csv) <<<")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    run_bayesian_tuning(n_trials=25)