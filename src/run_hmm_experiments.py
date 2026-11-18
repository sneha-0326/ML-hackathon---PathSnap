"""
Tunes the HMM/Viterbi decoder and compares classifier-only vs classifier + HMM.

Run:  python run_hmm_experiments.py                  (uses LightGBM_tuned probs if present)
      python run_hmm_experiments.py LightGBM         (force the untuned model's probs)

Data rules followed:
  * decoder parameters are tuned ONLY on training trips, using OUT-OF-FOLD probabilities
  * student-test trips are used for the final comparison
  * evaluator data is never touched; targets / .route are used for scoring only,
    never inside the decoder

Outputs:
  reports/hmm_comparison.csv       overall classifier-only vs HMM (both partitions)
  reports/hmm_per_trip.csv         per-trip student-test results (failure analysis)
  reports/hmm_tuning_trials.csv    decoder tuning log
  models/hmm_config.json           final decoder params (used by the API)
  reports/figures/hmm_comparison.png
"""
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from hmm_viterbi import (DecoderParams, RoadNetwork, chosen_arcs, classifier_only,
                         continuity_counts, dedupe, edit_distance, expand_route,
                         prepare_trip, save_config, viterbi)

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DATA_DIR = Path("Student folder") / "Student folder"
PROBS_DIR = Path("data/processed/probs")
MAX_HOPS = 4
SEED = 42
N_TUNE_TRIPS = 25      # training trips used for decoder tuning (speed)
N_TRIALS = 40

Path("reports/figures").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)


def resolve_tag():
    if len(sys.argv) > 1:
        return sys.argv[1]
    if (PROBS_DIR / "train_oof_probs_LightGBM_tuned.parquet").exists():
        return "LightGBM_tuned"
    print("NOTE: tuned-model probs not found, falling back to untuned LightGBM.")
    return "LightGBM"


def find_file(tid, ext):
    hits = list(BASE_DATA_DIR.glob(f"*/{tid}/{tid}.{ext}"))
    return hits[0] if hits else None


def build_trips(df, trip_ids, desc):
    trips = {}
    t0 = time.perf_counter()
    for k, tid in enumerate(trip_ids, 1):
        g = df[df["trip_id"] == tid]
        arcs_path = find_file(tid, "arcs")
        if arcs_path is None:
            print(f"  [skip] no .arcs file for {tid}")
            continue
        net = RoadNetwork(arcs_path, MAX_HOPS)
        prep = prepare_trip(g, net)
        route_path = find_file(tid, "route")
        ref = np.loadtxt(route_path, dtype=int, ndmin=1).tolist() if route_path else None
        trips[tid] = (net, prep, ref)
    print(f"Prepared {len(trips)} {desc} trips in {time.perf_counter() - t0:.0f}s")
    return trips


def seq_metrics(net, arcs, ref):
    route, breaks = expand_route(net, arcs)
    out = {"breaks": breaks, "P": np.nan, "R": np.nan, "F1": np.nan, "edit": np.nan}
    if ref is not None and route:
        ps, rs = set(route), set(ref)
        tp = len(ps & rs)
        P, R = tp / len(ps), tp / len(rs)
        out.update(P=P, R=R, F1=(2 * P * R / (P + R) if P + R else 0.0))
        ref_d = dedupe(ref)
        out["edit"] = edit_distance(route, ref_d) / max(len(route), len(ref_d))
    return out


def evaluate(trips, params):
    rows = []
    for tid, (net, prep, ref) in trips.items():
        t0 = time.perf_counter()
        idx_h = viterbi(prep, params, MAX_HOPS)
        decode_s = time.perf_counter() - t0
        idx_c = classifier_only(prep)
        r = {"trip_id": tid, "n_obs": len(prep.arcs),
             "miss_obs": int(sum(t.max() == 0 for t in prep.tgts)), "hmm_decode_s": decode_s}
        for name, idx in (("clf", idx_c), ("hmm", idx_h)):
            arcs = chosen_arcs(prep, idx)
            r[f"{name}_correct"] = int(sum(prep.tgts[t][i] == 1 for t, i in enumerate(idx)))
            r[f"{name}_valid"], r["n_trans"] = continuity_counts(net, arcs)
            sm = seq_metrics(net, arcs, ref)
            for k, v in sm.items():
                r[f"{name}_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(df, partition):
    out = []
    for name, label in (("clf", "Classifier-only"), ("hmm", "Classifier + HMM/Viterbi")):
        out.append({
            "Partition": partition, "Method": label, "Trips": len(df),
            "Pointwise_Top1_Acc_%": round(100 * df[f"{name}_correct"].sum() / df["n_obs"].sum(), 2),
            "Route_Continuity_%": round(100 * df[f"{name}_valid"].sum() / max(df["n_trans"].sum(), 1), 2),
            "Candidate_Miss_Rate_%": round(100 * df["miss_obs"].sum() / df["n_obs"].sum(), 2),
            "Seq_Arc_Precision": round(df[f"{name}_P"].mean(), 4),
            "Seq_Arc_Recall": round(df[f"{name}_R"].mean(), 4),
            "Seq_Arc_F1": round(df[f"{name}_F1"].mean(), 4),
            "Seq_Norm_Edit_Dist": round(df[f"{name}_edit"].mean(), 4),
            "Route_Breaks_per_trip": round(df[f"{name}_breaks"].mean(), 2),
            "HMM_Decode_s_per_trip": round(df["hmm_decode_s"].mean(), 3) if name == "hmm" else None,
        })
    return out


def main():
    tag = resolve_tag()
    print(f"Using classifier probabilities: {tag}")
    train_p = pd.read_parquet(PROBS_DIR / f"train_oof_probs_{tag}.parquet")
    test_p = pd.read_parquet(PROBS_DIR / f"student_test_probs_{tag}.parquet")

    rng = np.random.RandomState(SEED)
    train_ids = np.unique(train_p["trip_id"])
    tune_ids = rng.choice(train_ids, size=min(N_TUNE_TRIPS, len(train_ids)), replace=False)

    tune_trips = build_trips(train_p, tune_ids, "training (tuning subset)")
    test_trips = build_trips(test_p, np.unique(test_p["trip_id"]), "student-test")

    # ---------------- decoder tuning on training OOF probs ----------------
    def total_correct(params):
        c = n = 0
        for net, prep, _ in tune_trips.values():
            idx = viterbi(prep, params, MAX_HOPS)
            c += sum(prep.tgts[t][i] == 1 for t, i in enumerate(idx))
            n += len(prep.arcs)
        return c / n

    def objective(trial):
        beta = trial.suggest_float("beta", 0.1, 8.0, log=True)
        extra = trial.suggest_float("unreach_extra", 0.5, 30.0)
        params = DecoderParams(
            beta=beta,
            unreach=beta * MAX_HOPS + extra,     # unreachable always costs more than max-hop reachable
            gap_reset=trial.suggest_int("gap_reset", 3, 30),
            normalize=trial.suggest_categorical("normalize", [True, False]),
        )
        acc = total_correct(params)
        print(f"  trial {trial.number:02d}  acc = {acc:.4f}")
        return acc

    print(f"\nTuning decoder: {N_TRIALS} trials on {len(tune_trips)} training trips (OOF probs)")
    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({"beta": 1.5, "unreach_extra": 6.0, "gap_reset": 10, "normalize": True})
    study.optimize(objective, n_trials=N_TRIALS)
    tune_time = time.perf_counter() - t0
    study.trials_dataframe().to_csv("reports/hmm_tuning_trials.csv", index=False)

    bp = study.best_params
    best = DecoderParams(beta=bp["beta"], unreach=bp["beta"] * MAX_HOPS + bp["unreach_extra"],
                         gap_reset=bp["gap_reset"], normalize=bp["normalize"])
    print(f"\nBest decoder params: {best}  (train-OOF acc {study.best_value:.4f}, {tune_time:.0f}s)")
    save_config("models/hmm_config.json", best, MAX_HOPS,
                {"probs_source": tag, "tuned_on": "training OOF probabilities only",
                 "n_tune_trips": len(tune_trips), "n_trials": N_TRIALS, "seed": SEED})

    # ---------------- final comparison ----------------
    tune_df = evaluate(tune_trips, best)
    test_df = evaluate(test_trips, best)
    comp = pd.DataFrame(summarise(tune_df, "Training OOF (tuning subset)")
                        + summarise(test_df, "Student-test"))
    comp.to_csv("reports/hmm_comparison.csv", index=False)
    test_df.to_csv("reports/hmm_per_trip.csv", index=False)
    print("\n>>> CLASSIFIER-ONLY vs CLASSIFIER + HMM <<<")
    print(comp.to_string(index=False))

    # failure analysis
    test_df["acc_delta_pts"] = 100 * (test_df["hmm_correct"] - test_df["clf_correct"]) / test_df["n_obs"]
    bad = test_df[test_df["acc_delta_pts"] < 0]
    print(f"\nStudent-test trips where HMM was WORSE than classifier-only: {len(bad)}/{len(test_df)}")
    if len(bad):
        print(bad[["trip_id", "n_obs", "miss_obs", "acc_delta_pts"]].to_string(index=False))

    # ---------------- plot ----------------
    t = test_df
    x = np.arange(len(t))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].bar(x - 0.2, 100 * t["clf_correct"] / t["n_obs"], 0.4, label="Classifier-only", color="#dc2626")
    ax[0].bar(x + 0.2, 100 * t["hmm_correct"] / t["n_obs"], 0.4, label="Classifier + HMM", color="#16a34a")
    ax[0].set_title("Student-test: pointwise top-1 accuracy per trip (%)")
    ax[1].bar(x - 0.2, 100 * t["clf_valid"] / t["n_trans"].clip(lower=1), 0.4, color="#dc2626")
    ax[1].bar(x + 0.2, 100 * t["hmm_valid"] / t["n_trans"].clip(lower=1), 0.4, color="#16a34a")
    ax[1].set_title("Student-test: route continuity per trip (%)")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([s[-6:] for s in t["trip_id"]], rotation=45)
        a.grid(True, axis="y", alpha=0.3)
    ax[0].legend()
    plt.tight_layout()
    plt.savefig("reports/figures/hmm_comparison.png", dpi=200)
    plt.close()
    print("\nSaved reports/hmm_comparison.csv, hmm_per_trip.csv, hmm_tuning_trials.csv, "
          "models/hmm_config.json, reports/figures/hmm_comparison.png")


if __name__ == "__main__":
    main()
