"""
HMM + Viterbi route reconstruction for PathSnap.

Model
-----
* Hidden state at observation t : one of the candidate arcs generated for that
  GPS observation (state space changes with t; candidate -> arc_id is preserved).
* Emission  : log of the classifier probability of that candidate. With
  `normalize=True` the probabilities are first normalised across the candidates of
  the same observation (so emissions compete within an observation).
* Transition u -> v between consecutive observations: based on the shortest path
  (in number of road segments / "hops") from arc u to arc v in the trip's own
  road graph, found by BFS limited to `max_hops`:
        logT = -beta * hops            if v reachable from u within max_hops
        logT = -unreach                otherwise (soft penalty, NOT -inf, so the
                                       decoder never dead-ends)
  hops = 0 means staying on the same arc; hops = 1 means moving to a successor arc
  (u.to_node == v.from_node).
* Gaps / missing candidates: if obs_idx jumps by more than `gap_reset` between two
  consecutive observations that have candidates, the chain is restarted: no
  transition preference is applied for that step.
* Decoding: Viterbi  delta_t(v) = max_u [delta_{t-1}(u) + logT(u,v)] + logE_t(v)
  with back-pointers and back-tracking.

IMPORTANT: arc_id is a per-trip row index of that trip's .arcs file, so every
network here is built from one trip's own .arcs file.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

UNREACH_CODE = 127  # int8 sentinel in hop matrices
EPS = 1e-6


# ----------------------------------------------------------------------------
@dataclass
class DecoderParams:
    beta: float = 1.5        # cost per road segment traversed between observations
    unreach: float = 12.0    # soft penalty for an unreachable transition
    gap_reset: int = 10      # obs_idx jump above which the chain restarts
    normalize: bool = True   # normalise probabilities across candidates of an obs

    def to_dict(self):
        return asdict(self)


def save_config(path, params: DecoderParams, max_hops: int, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"params": params.to_dict(), "max_hops": max_hops, **(extra or {})}, f, indent=2)


def load_config(path):
    cfg = json.load(open(path))
    return DecoderParams(**cfg["params"]), int(cfg["max_hops"])


# ----------------------------------------------------------------------------
class RoadNetwork:
    """Directed arc graph of ONE trip: u -> v if arc_u.to_node == arc_v.from_node."""

    def __init__(self, arcs_path, max_hops: int = 4):
        arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["f", "t"])
        self.frm = arcs["f"].to_numpy()
        self.to = arcs["t"].to_numpy()
        self.by_from = dict(arcs.groupby("f").indices)
        self.max_hops = max_hops
        self._cache: dict[int, tuple[dict, dict]] = {}

    def _bfs(self, u: int):
        hit = self._cache.get(u)
        if hit is None:
            dist, parent = {u: 0}, {}
            frontier = [u]
            for h in range(1, self.max_hops + 1):
                nxt = []
                for a in frontier:
                    for b in self.by_from.get(self.to[a], ()):
                        b = int(b)
                        if b not in dist:
                            dist[b] = h
                            parent[b] = a
                            nxt.append(b)
                frontier = nxt
                if not frontier:
                    break
            hit = self._cache[u] = (dist, parent)
        return hit

    def hops(self, u: int, v: int):
        return self._bfs(u)[0].get(v)

    def path(self, u: int, v: int):
        """Arc sequence u ... v (inclusive) via BFS parents, or None if unreachable."""
        dist, parent = self._bfs(u)
        if v not in dist:
            return None
        seq = [v]
        while seq[-1] != u:
            seq.append(parent[seq[-1]])
        return seq[::-1]


def hop_matrix(net: RoadNetwork, U: list, V: list) -> np.ndarray:
    H = np.full((len(U), len(V)), UNREACH_CODE, dtype=np.int8)
    for i, u in enumerate(U):
        dist = net._bfs(u)[0]
        H[i] = [dist.get(v, UNREACH_CODE) for v in V]
    return H


# ----------------------------------------------------------------------------
@dataclass
class Prepared:
    """Parameter-independent per-trip data (so decoder tuning is fast)."""
    obs: np.ndarray      # obs_idx of each step
    arcs: list           # per step: candidate arc ids (np.int64 array)
    probs: list          # per step: classifier probabilities
    tgts: list           # per step: target labels (evaluation only; never used to decode)
    logE_raw: list
    logE_norm: list
    H: list              # H[t]: hop matrix step t-1 -> t (H[0] is None)
    dt: np.ndarray       # obs_idx jump between step t-1 and t


def prepare_trip(df: pd.DataFrame, net: RoadNetwork, with_targets: bool = True) -> Prepared:
    """df columns: obs_idx, arc_id, prob (+ target for evaluation)."""
    df = df.sort_values("obs_idx", kind="stable")
    obs = df["obs_idx"].to_numpy()
    arc = df["arc_id"].to_numpy().astype(np.int64)
    prob = df["prob"].to_numpy(dtype=float)
    tgt = df["target"].to_numpy() if with_targets and "target" in df else np.zeros(len(df), int)

    cuts = np.flatnonzero(np.diff(obs)) + 1
    obs_u = obs[np.r_[0, cuts]]
    arcs = np.split(arc, cuts)
    probs = np.split(prob, cuts)
    tgts = np.split(tgt, cuts)
    logE_raw = [np.log(np.clip(p, EPS, 1.0)) for p in probs]
    logE_norm = [np.log(np.clip(p / max(p.sum(), EPS), EPS, 1.0)) for p in probs]
    H = [None] + [hop_matrix(net, arcs[t - 1].tolist(), arcs[t].tolist()) for t in range(1, len(arcs))]
    dt = np.r_[0, np.diff(obs_u)]
    return Prepared(obs_u, arcs, probs, tgts, logE_raw, logE_norm, H, dt)


# ----------------------------------------------------------------------------
def viterbi(prep: Prepared, p: DecoderParams, max_hops: int) -> np.ndarray:
    """Returns, for every step, the index of the chosen candidate."""
    T = len(prep.arcs)
    logE = prep.logE_norm if p.normalize else prep.logE_raw
    delta = logE[0].copy()
    back = [None] * T
    for t in range(1, T):
        if prep.dt[t] > p.gap_reset:                     # GPS gap / missing candidates: restart
            trans = np.zeros((len(delta), len(logE[t])))
        else:
            H = prep.H[t]
            trans = np.where(H <= max_hops, -p.beta * H.astype(float), -p.unreach)
        s = delta[:, None] + trans
        bp = s.argmax(axis=0)
        delta = s[bp, np.arange(s.shape[1])] + logE[t]
        back[t] = bp
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t][path[t]]
    return path


def classifier_only(prep: Prepared) -> np.ndarray:
    return np.array([int(np.argmax(p)) for p in prep.probs], dtype=np.int64)


def chosen_arcs(prep: Prepared, idx: np.ndarray) -> np.ndarray:
    return np.array([prep.arcs[t][i] for t, i in enumerate(idx)], dtype=np.int64)


# ----------------------------------------------------------------------------
def dedupe(seq):
    return [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]


def expand_route(net: RoadNetwork, arcs) -> tuple[list, int]:
    """Deduplicate consecutive arcs and fill in intermediate arcs between jumps
    using shortest paths. Returns (route, number_of_breaks), where a break is a
    jump that is not reachable within max_hops (disconnected)."""
    seq = dedupe([int(a) for a in arcs])
    if not seq:
        return [], 0
    route, breaks = [seq[0]], 0
    for u, v in zip(seq[:-1], seq[1:]):
        p = net.path(u, v)
        if p is None:
            breaks += 1
            route.append(v)
        else:
            route.extend(p[1:])
    return route, breaks


def continuity_counts(net: RoadNetwork, arcs: np.ndarray) -> tuple[int, int]:
    """Same definition as train_models.py: consecutive predictions are valid if the
    arc is the same or the next arc starts where the previous one ends."""
    if len(arcs) < 2:
        return 0, 0
    u, v = arcs[:-1], arcs[1:]
    ok = (u == v) | (net.to[u] == net.frm[v])
    return int(ok.sum()), len(u)


def edit_distance(a, b, max_cells: int = 4_000_000):
    if len(a) * len(b) > max_cells:
        return float("nan")
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


# ----------------------------------------------------------------------------
# Convenience entry point for the FastAPI backend (no labels used)
_NET_CACHE: dict[str, RoadNetwork] = {}


def decode_trip_df(df_trip: pd.DataFrame, arcs_path, params: DecoderParams, max_hops: int):
    """df_trip: obs_idx, arc_id, prob for ONE trip. Returns per-observation predicted arcs
    and the expanded continuous route."""
    key = f"{arcs_path}|{max_hops}"
    net = _NET_CACHE.get(key) or _NET_CACHE.setdefault(key, RoadNetwork(arcs_path, max_hops))
    prep = prepare_trip(df_trip, net, with_targets=False)
    idx = viterbi(prep, params, max_hops)
    pred = chosen_arcs(prep, idx)
    route, breaks = expand_route(net, pred)
    return {"obs_idx": prep.obs.tolist(), "pred_arc": pred.tolist(),
            "route": route, "route_breaks": breaks}
