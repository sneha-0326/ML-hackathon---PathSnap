"""
One-shot evaluation on the evaluator-test partition. NOTHING is fitted or tuned here:
the scaler, classifier and HMM config are loaded as frozen at tag `pre-evaluator`.

Usage:
  python evaluate_final.py "<path to evaluator_test folder>" [model_tag]
  (model_tag default: LightGBM_tuned)

Prerequisite: data/processed/evaluator_test_candidates.parquet, produced by running your
existing candidate/feature script on the evaluator trips (same columns as the
student-test parquet; the scaler must be LOADED, not refitted).

The .route files / target column are used for SCORING only, never as model inputs.

Outputs:
  reports/final_evaluator_results.csv      overall classifier-only vs HMM
  reports/final_evaluator_per_trip.csv     per-trip results
  reports/final_evaluator_run_info.json    model tag, config, git commit, timestamp
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss

import run_hmm_experiments as rh
from hmm_viterbi import load_config

CAND_PATH = Path("data/processed/evaluator_test_candidates.parquet")
STUDENT_PATH = Path("data/processed/student_test_candidates.parquet")


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python evaluate_final.py "<evaluator_test folder>" [model_tag]')
    eval_dir = Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "LightGBM_tuned"
    if not eval_dir.exists():
        sys.exit(f"Folder not found: {eval_dir}")
    if not CAND_PATH.exists():
        sys.exit(f"Missing {CAND_PATH}. Run your candidate/feature script on the evaluator trips first.")

    # ---- load frozen artifacts ----
    df = pd.read_parquet(CAND_PATH)
    bundle = joblib.load("data/processed/feature_scaler.joblib")
    scaler, feature_cols = bundle["scaler"], bundle["feature_cols"]
    model = joblib.load(f"models/{tag}.joblib")
    params, max_hops = load_config("models/hmm_config.json")

    # ---- sanity checks on the new data ----
    try:
        import pyarrow.parquet as pq
        ref_cols = set(pq.read_schema(STUDENT_PATH).names)
        if set(df.columns) != ref_cols:
            print("WARNING: column mismatch vs student-test parquet:",
                  sorted(set(df.columns) ^ ref_cols))
    except Exception:
        pass
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        sys.exit(f"Missing feature columns: {missing}")
    n_trips = df["trip_id"].nunique()
    print(f"Evaluator candidates: {len(df):,} rows, {n_trips} trips")
    if n_trips != 10:
        print(f"WARNING: expected 10 evaluator trips, found {n_trips}")
    if df[feature_cols].isna().any().any():
        print("WARNING: NaNs in feature columns")

    # ---- classifier probabilities (transform only, no fitting) ----
    X = scaler.transform(df[feature_cols])
    t0 = time.perf_counter()
    probs = model.predict_proba(X)[:, 1]
    infer_s = time.perf_counter() - t0
    pdf = df[["trip_id", "obs_idx", "arc_id", "target"]].copy()
    pdf["prob"] = probs

    # ---- HMM decoding + scoring (targets / .route used only for scoring) ----
    rh.MAX_HOPS = max_hops
    rh.find_file = lambda tid, ext: next(eval_dir.rglob(f"{tid}.{ext}"), None)
    trips = rh.build_trips(pdf, np.unique(pdf["trip_id"]), "evaluator-test")
    per_trip = rh.evaluate(trips, params)
    overall = pd.DataFrame(rh.summarise(per_trip, "Evaluator-test"))

    # candidate-level metrics (for the report's model tables)
    y = df["target"].to_numpy()
    cand_f1 = f1_score(y, (probs >= 0.5).astype(int), average="macro")
    cand_ll = log_loss(y, probs)
    overall["Candidate_Macro_F1"] = round(float(cand_f1), 4)
    overall["Candidate_Log_Loss"] = round(float(cand_ll), 4)
    overall["Classifier_Inference_s"] = round(infer_s, 3)

    overall.to_csv("reports/final_evaluator_results.csv", index=False)
    per_trip.to_csv("reports/final_evaluator_per_trip.csv", index=False)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    with open("reports/final_evaluator_run_info.json", "w") as f:
        json.dump({"model_tag": tag, "hmm_params": params.to_dict(), "max_hops": max_hops,
                   "n_trips": int(n_trips), "git_commit": commit,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    print("\n>>> EVALUATOR-TEST RESULTS (final, one-shot) <<<")
    print(overall.to_string(index=False))
    print("\nPer-trip pointwise top-1 (%):")
    show = per_trip[["trip_id", "n_obs"]].copy()
    show["classifier"] = (100 * per_trip["clf_correct"] / per_trip["n_obs"]).round(2)
    show["hmm"] = (100 * per_trip["hmm_correct"] / per_trip["n_obs"]).round(2)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
