"""
Builds data/processed/evaluator_test_candidates.parquet from the evaluator trips.

Usage:
  python build_evaluator_candidates.py "<path to evaluator_test folder>"
  (add --allow-partial only if you knowingly accept missing/corrupt trips)

* Reuses extract_trip_candidates() from features.py with the SAME settings as training
  (radius 50 m, 12 candidates) so the features are produced identically.
* Does NOT fit or touch the scaler, and does NOT touch the train / student-test parquets.
* The `target` column is built from the .route file for SCORING only. It is never a model
  input (FEATURE_COLS does not include it). If a trip has no .route, target = -1.
"""
import sys
import time
from pathlib import Path

import pandas as pd

from features import FEATURE_COLS, extract_trip_candidates

RADIUS_M = 50.0        # must match the training run
MAX_CANDIDATES = 12    # must match the training run
OUT_PATH = Path("data/processed/evaluator_test_candidates.parquet")
STUDENT_PATH = Path("data/processed/student_test_candidates.parquet")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    allow_partial = "--allow-partial" in sys.argv
    if not args:
        sys.exit('Usage: python build_evaluator_candidates.py "<evaluator_test folder>"')
    eval_dir = Path(args[0])
    if not eval_dir.is_dir():
        sys.exit(f"Folder not found: {eval_dir}")

    trip_dirs = sorted(d for d in eval_dir.iterdir() if d.is_dir())
    print(f"Found {len(trip_dirs)} trip folders in {eval_dir}")

    dfs, problems = [], []
    total_obs = total_missed = 0
    t0 = time.perf_counter()
    for i, trip_dir in enumerate(trip_dirs, 1):
        tid = trip_dir.name
        missing = [f"{tid}.{e}" for e in ("track", "nodes", "arcs")
                   if not (trip_dir / f"{tid}.{e}").exists()]
        if missing:
            problems.append(f"{tid}: missing files {missing} (files found: "
                            f"{[f.name for f in trip_dir.iterdir()]})")
            continue
        df, missed = extract_trip_candidates(trip_dir, RADIUS_M, MAX_CANDIDATES)
        if df.empty:
            problems.append(f"{tid}: produced no candidates (corrupt/empty trip)")
            continue
        if not (trip_dir / f"{tid}.route").exists():
            print(f"  [note] {tid}: no .route file, target will be -1 (cannot be scored)")
        total_obs += df["obs_idx"].nunique()
        total_missed += missed
        dfs.append(df)
        print(f"  [{i}/{len(trip_dirs)}] {tid}: {len(df):,} candidate rows")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        if not allow_partial:
            sys.exit("\nAborting so no trip is silently dropped. Fix the data, or re-run with "
                     "--allow-partial and state the missing trips in the report.")

    full = pd.concat(dfs, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUT_PATH, index=False)

    print(f"\n>>> EVALUATOR-TEST CANDIDATES SAVED -> {OUT_PATH} <<<")
    print(f"Trips                : {full['trip_id'].nunique()}")
    print(f"Candidate rows       : {len(full):,}")
    print(f"GPS observations     : {total_obs:,}")
    print(f"Candidate miss rate  : {100 * total_missed / total_obs:.2f}%  (compare with student-test)")
    print(f"Rows with target = -1: {(full['target'] == -1).sum():,}")
    print(f"NaNs in features     : {int(full[FEATURE_COLS].isna().sum().sum())}")
    print(f"Time                 : {time.perf_counter() - t0:.0f}s")

    if STUDENT_PATH.exists():
        ref_cols = list(pd.read_parquet(STUDENT_PATH).columns)
        same = list(full.columns) == ref_cols
        print(f"Columns match student-test parquet: {same}")
        if not same:
            print("  evaluator:", list(full.columns))
            print("  student  :", ref_cols)


if __name__ == "__main__":
    main()
