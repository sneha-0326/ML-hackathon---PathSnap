"""
Exports per-candidate probabilities so the HMM/Viterbi stage can be developed
independently of model training / tuning.

Run:  python export_probs.py                      (uses models/LightGBM.joblib)
      python export_probs.py LightGBM_tuned       (later, once tuning is done)

Outputs (same row order as the candidate parquet files):
  data/processed/probs/train_oof_probs_<model>.parquet
      Out-of-fold probabilities for the 80 training trips (5-fold GroupKFold).
      Use these to tune decoder parameters: probabilities from a model fitted on
      the same trips would be over-confident and give misleading decoder settings.
  data/processed/probs/student_test_probs_<model>.parquet
      Probabilities for the 10 student-test trips from the fully-fitted model.

Columns: trip_id, obs_idx, arc_id, target, prob
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from train_models import load_data

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "LightGBM"
OUT = Path("data/processed/probs")
OUT.mkdir(parents=True, exist_ok=True)

train_df, test_df, X_train, y_train, groups_train, X_test, y_test, _ = load_data()
model = joblib.load(f"models/{MODEL_NAME}.joblib")
cols = ["trip_id", "obs_idx", "arc_id", "target"]

# Student-test probabilities from the final fitted model
test_out = test_df[cols].copy()
test_out["prob"] = model.predict_proba(X_test)[:, 1]
test_out.to_parquet(OUT / f"student_test_probs_{MODEL_NAME}.parquet", index=False)

# Out-of-fold probabilities on training trips (same config, refit per fold)
oof = np.zeros(len(train_df))
for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X_train, y_train, groups_train), 1):
    m = clone(model).fit(X_train[tr], y_train[tr])
    oof[va] = m.predict_proba(X_train[va])[:, 1]
    print(f"fold {k}/5 done")
train_out = train_df[cols].copy()
train_out["prob"] = oof
train_out.to_parquet(OUT / f"train_oof_probs_{MODEL_NAME}.parquet", index=False)

print("Saved:", *sorted(p.name for p in OUT.glob(f"*{MODEL_NAME}*")), sep="\n  ")
