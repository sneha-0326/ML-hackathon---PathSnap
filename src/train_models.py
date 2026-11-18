import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, log_loss
from lightgbm import LGBMClassifier

BASE_DATA_DIR = Path("Student folder") / "Student folder"
Path("models").mkdir(exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)


def load_data():
    train_df = pd.read_parquet("data/processed/train_candidates.parquet")
    test_df = pd.read_parquet("data/processed/student_test_candidates.parquet")
    scaler_bundle = joblib.load("data/processed/feature_scaler.joblib")
    scaler = scaler_bundle["scaler"]
    feature_cols = scaler_bundle["feature_cols"]

    X_train = scaler.transform(train_df[feature_cols])
    y_train = train_df["target"].values
    groups_train = train_df["trip_id"].values

    X_test = scaler.transform(test_df[feature_cols])
    y_test = test_df["target"].values

    return train_df, test_df, X_train, y_train, groups_train, X_test, y_test, feature_cols


def build_valid_edge_sets(partition: str, trip_ids: np.ndarray) -> dict[str, set[tuple[int, int]]]:
    """Builds valid directed arc-to-arc transitions (arc_u -> arc_v where arc_u.to_node == arc_v.from_node)."""
    edge_sets = {}
    for tid in np.unique(trip_ids):
        arcs_path = BASE_DATA_DIR / partition / tid / f"{tid}.arcs"
        arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node", "to_node"])
        by_from = arcs.groupby("from_node").indices
        valid_pairs = set()
        for u_idx, to_n in enumerate(arcs["to_node"].values):
            valid_pairs.add((u_idx, u_idx))  # staying on same arc is continuous
            if to_n in by_from:
                for v_idx in by_from[to_n]:
                    valid_pairs.add((u_idx, int(v_idx)))
        edge_sets[tid] = valid_pairs
    return edge_sets


def evaluate_observation_level(df: pd.DataFrame, probs: np.ndarray, edge_sets: dict) -> tuple[float, float]:
    """
    Computes:
      1. Observation-level Top-1 Accuracy (does argmax probability candidate match true target==1?)
      2. Route Continuity (% of consecutive GPS transitions that are physically connected in the road graph)
    """
    eval_df = df[["trip_id", "obs_idx", "arc_id", "target"]].copy()
    eval_df["prob"] = probs

    # Pick highest-probability candidate per (trip_id, obs_idx)
    idx_max = eval_df.groupby(["trip_id", "obs_idx"])["prob"].idxmax()
    top1 = eval_df.loc[idx_max].sort_values(["trip_id", "obs_idx"])

    top1_acc = float((top1["target"] == 1).mean())

    # Route continuity across consecutive observations within each trip
    total_transitions = 0
    valid_transitions = 0
    for tid, grp in top1.groupby("trip_id"):
        pred_arcs = grp["arc_id"].values
        valid_pairs = edge_sets[tid]
        for i in range(len(pred_arcs) - 1):
            total_transitions += 1
            if (int(pred_arcs[i]), int(pred_arcs[i + 1])) in valid_pairs:
                valid_transitions += 1

    continuity = (valid_transitions / total_transitions) if total_transitions > 0 else 0.0
    return top1_acc, continuity


def get_models():
    """Five distinct models across 5 algorithm families as required by guidelines."""
    return {
        "Logistic_Regression": (
            "Linear / Probabilistic",
            "C=1.0, class_weight='balanced', max_iter=500",
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42),
        ),
        "Random_Forest": (
            "Bagging Ensemble",
            "n_estimators=150, max_depth=14, min_samples_leaf=10",
            RandomForestClassifier(
                n_estimators=150, max_depth=14, min_samples_leaf=10, n_jobs=-1, random_state=42
            ),
        ),
        "Extra_Trees": (
            "Randomized Tree Ensemble",
            "n_estimators=150, max_depth=16, min_samples_leaf=8",
            ExtraTreesClassifier(
                n_estimators=80, max_depth=16,bootstrap=True,max_samples=0.25, min_samples_leaf=8, n_jobs=-1, random_state=42
            ),
        ),
        "LightGBM": (
            "Gradient Boosting",
            "n_estimators=300, lr=0.05, num_leaves=31",
            LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
            ),
        ),
        "MLP_Neural_Net": (
            "Neural Network (MLP)",
            "hidden=(64, 32), relu, adam, early_stopping=True",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=30,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ),
        ),
    }


def run_experiments():
    print("Loading processed candidate datasets...")
    train_df, test_df, X_train, y_train, groups_train, X_test, y_test, feature_cols = load_data()
    test_edge_sets = build_valid_edge_sets("student_test", test_df["trip_id"].values)

    gkf = GroupKFold(n_splits=5)
    models = get_models()
    results = []

    for name, (family, config_str, model) in models.items():
        print(f"\n{'='*55}\nTraining & Evaluating: {name} ({family})\n{'='*55}")

        # 1. 5-Fold Group-Aware Cross-Validation on 80 training trips
        cv_f1_scores = []
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train), 1):
            # Clone fresh estimator per fold
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train[tr_idx], y_train[tr_idx])
            val_preds = fold_model.predict(X_train[val_idx])
            cv_f1_scores.append(f1_score(y_train[val_idx], val_preds, average="macro"))

        cv_macro_f1 = float(np.mean(cv_f1_scores))
        cv_macro_f1_std = float(np.std(cv_f1_scores))
        print(f"  5-Fold GroupCV Macro F1 : {cv_macro_f1:.4f} (+/- {cv_macro_f1_std:.4f})")

        # 2. Full fit on all 80 training trips
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # Save MLP loss curve if available (required for Section 3.3)
        if hasattr(model, "loss_curve_"):
            plt.figure(figsize=(6, 4))
            plt.plot(model.loss_curve_, marker="o", color="#2563eb", label="Training Loss")
            plt.title("MLP Neural Network Training Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Log Loss")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("reports/figures/mlp_loss_curve.png", dpi=200)
            plt.close()

        # 3. Evaluate on 10 Student-Testing trips
        t0 = time.perf_counter()
        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        infer_time = time.perf_counter() - t0

        test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
        test_logloss = float(log_loss(y_test, test_probs))
        top1_acc, continuity = evaluate_observation_level(test_df, test_probs, test_edge_sets)

        print(f"  Student-Test Macro F1   : {test_macro_f1:.4f}")
        print(f"  Student-Test Top-1 Acc  : {top1_acc*100:.2f}%")
        print(f"  Route Continuity        : {continuity*100:.2f}%")
        print(f"  Log Loss                : {test_logloss:.4f}")
        print(f"  Train / Inference Time  : {train_time:.2f}s / {infer_time:.4f}s")

        joblib.dump(model, f"models/{name}.joblib")

        results.append({
            "Model": name,
            "Algorithm_Family": family,
            "Key_Configuration": config_str,
            "CV_Macro_F1": round(cv_macro_f1, 4),
            "CV_Macro_F1_Std": round(cv_macro_f1_std, 4),
            "Top1_Accuracy": round(top1_acc * 100, 2),
            "Route_Continuity": round(continuity * 100, 2),
            "Student_Test_Macro_F1": round(test_macro_f1, 4),
            "Log_Loss": round(test_logloss, 4),
            "Train_Time_s": round(train_time, 2),
            "Inference_Time_s": round(infer_time, 4),
        })

    comp_df = pd.DataFrame(results)
    comp_df.to_csv("reports/model_comparison.csv", index=False)
    print("\n>>> FINAL MODEL COMPARISON TABLE (Saved to reports/model_comparison.csv) <<<")
    print(comp_df.to_string(index=False))

    # Generate compulsory Section 3.3 comparison charts
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x_labels = comp_df["Model"]

    axes[0].bar(x_labels, comp_df["CV_Macro_F1"], yerr=comp_df["CV_Macro_F1_Std"], color="#2563eb", capsize=4)
    axes[0].set_title("5-Fold GroupCV Macro F1-Score")
    axes[0].set_ylabel("Macro F1")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(x_labels, comp_df["Top1_Accuracy"], color="#16a34a")
    axes[1].set_title("Student-Test Top-1 Accuracy (%)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(x_labels, comp_df["Log_Loss"], color="#dc2626")
    axes[2].set_title("Student-Test Log Loss (Lower is Better)")
    axes[2].set_ylabel("Log Loss")
    axes[2].tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison_charts.png", dpi=200)
    print("Saved model comparison charts -> reports/figures/model_comparison_charts.png")


if __name__ == "__main__":
    run_experiments()