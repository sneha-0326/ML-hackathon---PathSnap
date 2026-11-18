"""
FAST Bayesian hyperparameter optimisation (Optuna TPE) for LightGBM - two stages.

Stage 1 (cheap proxy search): random half of the 80 training trips, 3-fold
         GroupKFold, parallel folds, capped model size. TPE explores here.
Stage 2 (full-fidelity check): top-K trials + the untuned baseline are re-scored
         with the FULL protocol (all 80 trips, 5-fold GroupKFold). The best
         finalist becomes the tuned model.

Only the 80 training trips are used for the search; the 10 student-test trips are
used for the final tuned-vs-untuned comparison. Evaluator data is never touched.

Outputs:
  reports/bayesian_tuning_trials.csv          stage-1 trial log
  reports/bayesian_finalists_full_cv.csv      stage-2 full 5-fold results
  reports/best_params.json
  reports/untuned_vs_tuned_comparison.csv
  reports/optuna_study.joblib
  reports/figures/bayesian_optimization_history.png
  reports/figures/bayesian_param_importance.png
  reports/figures/untuned_vs_tuned.png
  models/LightGBM_tuned.joblib
"""
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from train_models import load_data, build_valid_edge_sets, evaluate_observation_level

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ----------------------------- settings ---------------------------------
SEED = 42
CPU = os.cpu_count() or 4
N_TRIALS = 30          # cheap trials, so we can afford more
SEARCH_TRIP_FRAC = 0.5  # stage 1: fraction of training trips used
SEARCH_FOLDS = 3        # stage 1 folds
FULL_FOLDS = 5          # stage 2 folds (matches the baseline protocol)
TOP_K = 3               # finalists re-scored at full fidelity

Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

# Untuned config from train_models.py (enqueued as trial 0 as an anchor)
BASELINE = {
    "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
    "reg_alpha": 1e-8, "reg_lambda": 1e-8,
}


# ----------------------------- helpers -----------------------------------
def cv_score(params, X, y, splits):
    """Folds run in parallel threads (LightGBM releases the GIL); threads are
    split between folds so the CPU is not oversubscribed."""
    n_par = min(len(splits), CPU)
    threads = max(1, CPU // n_par)

    def one(tr, va):
        clf = LGBMClassifier(**params, subsample_freq=1, random_state=SEED,
                             n_jobs=threads, verbose=-1)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:, 1]
        return (f1_score(y[va], (p >= 0.5).astype(int), average="macro"),
                log_loss(y[va], p))

    out = Parallel(n_jobs=n_par, backend="threading")(delayed(one)(tr, va) for tr, va in splits)
    f1s, lls = zip(*out)
    return float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(lls))


def evaluate_on_student_test(model, test_df, X_test, y_test, edge_sets, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        probs = model.predict_proba(X_test)[:, 1]
        times.append(time.perf_counter() - t0)
    preds = (probs >= 0.5).astype(int)
    top1, cont = evaluate_observation_level(test_df, probs, edge_sets)
    return {
        "Student_Test_Top1_Acc": round(top1 * 100, 2),
        "Student_Test_Macro_F1": round(float(f1_score(y_test, preds, average="macro")), 4),
        "Route_Continuity": round(cont * 100, 2),
        "Log_Loss": round(float(log_loss(y_test, probs)), 4),
        "Inference_Time_s": round(float(np.median(times)), 4),
    }


# ----------------------------- main --------------------------------------
def run(n_trials: int = N_TRIALS):
    print(f"Loading datasets... (CPU cores: {CPU})")
    train_df, test_df, X_train, y_train, groups_train, X_test, y_test, _ = load_data()
    test_edge_sets = build_valid_edge_sets("student_test", test_df["trip_id"].values)

    # ---- Stage-1 data: random subset of TRIPS (never split a trip) ----
    rng = np.random.RandomState(SEED)
    all_trips = np.unique(groups_train)
    n_keep = max(SEARCH_FOLDS * 2, int(len(all_trips) * SEARCH_TRIP_FRAC))
    keep_trips = rng.choice(all_trips, size=n_keep, replace=False)
    mask = np.isin(groups_train, keep_trips)
    Xs, ys, gs = X_train[mask], y_train[mask], groups_train[mask]
    search_splits = list(GroupKFold(n_splits=SEARCH_FOLDS).split(Xs, ys, gs))
    full_splits = list(GroupKFold(n_splits=FULL_FOLDS).split(X_train, y_train, groups_train))
    print(f"Stage 1 data: {n_keep}/{len(all_trips)} trips, {mask.sum():,} rows, {SEARCH_FOLDS}-fold")

    # ------------------------- Stage 1: TPE search -------------------------
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        t0 = time.perf_counter()
        f1, f1_std, ll = cv_score(params, Xs, ys, search_splits)
        trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("cv_f1_std", f1_std)
        print(f"  Trial {trial.number:02d}  F1 = {f1:.4f}  logloss = {ll:.4f}  "
              f"({time.perf_counter() - t0:.0f}s)")
        return f1

    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name="lightgbm_pathsnap_fast")
    study.enqueue_trial(BASELINE)

    print(f"\nStage 1: Bayesian search, {n_trials} trials")
    t_search = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    search_time = time.perf_counter() - t_search
    print(f"Stage 1 done in {search_time:.0f}s | best proxy F1 = {study.best_value:.4f}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv("reports/bayesian_tuning_trials.csv", index=False)
    joblib.dump(study, "reports/optuna_study.joblib")

    # ----------------- Stage 2: full-fidelity re-scoring -------------------
    ranked = sorted([t for t in study.trials if t.number > 0 and t.value is not None],
                    key=lambda t: t.value, reverse=True)[:TOP_K]
    candidates = [("Untuned baseline", BASELINE)] + \
                 [(f"Trial {t.number}", t.params) for t in ranked]

    print(f"\nStage 2: full {FULL_FOLDS}-fold CV on all {len(all_trips)} trips "
          f"for baseline + top {len(ranked)} trials")
    t_full = time.perf_counter()
    rows = []
    for name, p in candidates:
        t0 = time.perf_counter()
        f1, std, ll = cv_score(p, X_train, y_train, full_splits)
        rows.append({"Candidate": name, "CV_Macro_F1": round(f1, 4), "CV_F1_Std": round(std, 4),
                     "CV_Log_Loss": round(ll, 4), "Params": json.dumps(p)})
        print(f"  {name:18s} CV F1 = {f1:.4f} (+/- {std:.4f})  logloss = {ll:.4f}  "
              f"({time.perf_counter() - t0:.0f}s)")
    full_time = time.perf_counter() - t_full
    fin_df = pd.DataFrame(rows)
    fin_df.to_csv("reports/bayesian_finalists_full_cv.csv", index=False)

    tuned_rows = fin_df.iloc[1:]
    best_row = tuned_rows.sort_values(["CV_Macro_F1", "CV_Log_Loss"],
                                      ascending=[False, True]).iloc[0]
    best_params = json.loads(best_row["Params"])
    untuned_cv = float(fin_df.iloc[0]["CV_Macro_F1"])
    print(f"\nSelected: {best_row['Candidate']} | CV F1 {best_row['CV_Macro_F1']:.4f} "
          f"vs untuned {untuned_cv:.4f}")
    print(f"Best params: {best_params}")

    total_time = search_time + full_time
    with open("reports/best_params.json", "w") as f:
        json.dump({
            "best_params": best_params, "selected": best_row["Candidate"],
            "full_cv_macro_f1": float(best_row["CV_Macro_F1"]), "untuned_full_cv_macro_f1": untuned_cv,
            "stage1": {"n_trials": n_trials, "trip_fraction": SEARCH_TRIP_FRAC,
                       "folds": SEARCH_FOLDS, "time_s": round(search_time, 1)},
            "stage2": {"top_k": TOP_K, "folds": FULL_FOLDS, "time_s": round(full_time, 1)},
            "total_tuning_time_s": round(total_time, 1), "seed": SEED,
            "sampler": "TPESampler(n_startup_trials=8, multivariate=True)",
        }, f, indent=2)

    # --- plots: history + importance ---
    done = trials_df[trials_df["state"] == "COMPLETE"]
    plt.figure(figsize=(8, 4.2))
    plt.scatter(done["number"], done["value"], alpha=0.7, label="Trial proxy CV Macro F1")
    plt.plot(done["number"], done["value"].cummax(), color="#16a34a", lw=2.2, label="Best so far")
    plt.axhline(float(done[done["number"] == 0]["value"].iloc[0]), color="#dc2626", ls="--",
                label="Baseline config (trial 0)")
    plt.title(f"Optuna TPE History - LightGBM (stage 1: {int(SEARCH_TRIP_FRAC*100)}% trips, {SEARCH_FOLDS}-fold)")
    plt.xlabel("Trial number")
    plt.ylabel("GroupCV Macro F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/bayesian_optimization_history.png", dpi=200)
    plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(7, 4))
        plt.barh(list(imp)[::-1], list(imp.values())[::-1], color="#2563eb")
        plt.title("Hyperparameter Importance (fANOVA)")
        plt.tight_layout()
        plt.savefig("reports/figures/bayesian_param_importance.png", dpi=200)
        plt.close()
    except Exception as e:
        print(f"(Skipped importance plot: {e})")

    # ----------- refit on all 80 trips + student-test comparison -----------
    t0 = time.perf_counter()
    tuned = LGBMClassifier(**best_params, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)
    tuned.fit(X_train, y_train)
    tuned_fit_time = time.perf_counter() - t0
    joblib.dump(tuned, "models/LightGBM_tuned.joblib")
    untuned = joblib.load("models/LightGBM.joblib")

    r_un = evaluate_on_student_test(untuned, test_df, X_test, y_test, test_edge_sets)
    r_tu = evaluate_on_student_test(tuned, test_df, X_test, y_test, test_edge_sets)
    comp = pd.DataFrame([
        {"Stage": "Untuned LightGBM", "CV_Macro_F1": untuned_cv, **r_un},
        {"Stage": "Bayesian-Tuned LightGBM", "CV_Macro_F1": float(best_row["CV_Macro_F1"]), **r_tu},
    ])
    delta = {"Stage": "Delta (tuned - untuned)"}
    for c in comp.columns[1:]:
        delta[c] = round(comp[c].iloc[1] - comp[c].iloc[0], 4)
    comp = pd.concat([comp, pd.DataFrame([delta])], ignore_index=True)
    comp["Tuning_Time_s"] = [None, round(total_time, 1), None]
    comp["Final_Fit_Time_s"] = [None, round(tuned_fit_time, 2), None]
    comp.to_csv("reports/untuned_vs_tuned_comparison.csv", index=False)
    print("\n>>> UNTUNED VS TUNED (CV = full 5-fold, 80 trips; rest = student-test) <<<")
    print(comp.to_string(index=False))

    d = comp.iloc[2]
    wins = [d["CV_Macro_F1"] > 0, d["Student_Test_Top1_Acc"] > 0, d["Student_Test_Macro_F1"] > 0,
            d["Route_Continuity"] > 0, d["Log_Loss"] < 0]
    print(f"\nTuned better on {sum(wins)}/5 quality metrics (CV F1, Top-1, test F1, continuity, log loss).")
    print("Student-test is the unbiased check: the selected CV score is a max over candidates.")

    two = comp.iloc[:2]
    metrics = ["CV_Macro_F1", "Student_Test_Macro_F1", "Student_Test_Top1_Acc", "Route_Continuity", "Log_Loss"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.8))
    for ax, m in zip(axes, metrics):
        ax.bar(["Untuned", "Tuned"], two[m], color=["#dc2626", "#16a34a"])
        ax.set_title(m.replace("_", " "), fontsize=9)
        lo, hi = two[m].min(), two[m].max()
        pad = max((hi - lo) * 1.5, abs(hi) * 0.002)
        ax.set_ylim(lo - pad, hi + pad)
    plt.suptitle("Untuned vs Bayesian-Tuned LightGBM (y-axes zoomed)", y=1.02)
    plt.tight_layout()
    plt.savefig("reports/figures/untuned_vs_tuned.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nTotal tuning time (stage 1 + 2): {total_time:.0f}s")


if __name__ == "__main__":
    run()