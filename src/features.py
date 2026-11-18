from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

BASE_DATA_DIR = Path(r"..\Student folder\Student folder")
# EPSG:4326 (WGS84 lon/lat) -> EPSG:32632 (UTM Zone 32N - Darmstadt, Germany, in meters)
TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

FEATURE_COLS = [
    "perp_dist_m",
    "dist_to_from_node_m",
    "dist_to_to_node_m",
    "arc_length_m",
    "proj_ratio_alpha",
    "is_interior_proj",
    "heading_diff_deg",
    "cos_heading_diff",
    "speed_mps",
    "dt_s",
    "dist_rank",
    "dist_diff_from_min_m",
    "dist_ratio_to_min",
]


def point_to_segments_geometry(px: float, py: float, x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray):
    """Vectorized exact perpendicular distance and projection ratio alpha onto segments (x1,y1)->(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    seg_len = np.sqrt(seg_len_sq)

    safe_len_sq = np.where(seg_len_sq > 1e-9, seg_len_sq, 1.0)
    alpha_raw = ((px - x1) * dx + (py - y1) * dy) / safe_len_sq
    alpha_clamped = np.clip(alpha_raw, 0.0, 1.0)

    proj_x = x1 + alpha_clamped * dx
    proj_y = y1 + alpha_clamped * dy
    perp_dist = np.hypot(px - proj_x, py - proj_y)

    dist_from = np.hypot(px - x1, py - y1)
    dist_to = np.hypot(px - x2, py - y2)
    is_interior = ((alpha_raw >= 0.0) & (alpha_raw <= 1.0)).astype(np.float32)

    return perp_dist, dist_from, dist_to, seg_len, alpha_clamped, is_interior


def align_track_to_route(track_x: np.ndarray, track_y: np.ndarray, route_arcs: np.ndarray,
                         x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """
    Aligns N GPS points to an ordered route sequence of M arcs using monotonic Dynamic Programming.
    Ensures ground-truth arc indices advance non-decreasingly along the reference route.
    """
    N = len(track_x)
    M = len(route_arcs)
    rx1, ry1, rx2, ry2 = x1[route_arcs], y1[route_arcs], x2[route_arcs], y2[route_arcs]

    cost = np.zeros((N, M), dtype=np.float64)
    for t in range(N):
        d, _, _, _, _, _ = point_to_segments_geometry(track_x[t], track_y[t], rx1, ry1, rx2, ry2)
        cost[t] = d

    dp = np.full((N, M), np.inf, dtype=np.float64)
    backptr = np.zeros((N, M), dtype=np.int32)
    dp[0] = cost[0]

    for t in range(1, N):
        prefix_min = np.minimum.accumulate(dp[t - 1])
        best_prev = np.zeros(M, dtype=np.int32)
        curr_best_idx = 0
        for m in range(M):
            if dp[t - 1, m] < dp[t - 1, curr_best_idx]:
                curr_best_idx = m
            best_prev[m] = curr_best_idx

        dp[t] = cost[t] + prefix_min
        backptr[t] = best_prev

    aligned_steps = np.zeros(N, dtype=np.int32)
    aligned_steps[-1] = int(np.argmin(dp[-1]))
    for t in range(N - 2, -1, -1):
        aligned_steps[t] = backptr[t + 1, aligned_steps[t + 1]]

    return route_arcs[aligned_steps]


def extract_trip_candidates(trip_dir: Path, radius_m: float = 50.0, max_candidates: int = 12) -> tuple[pd.DataFrame, int]:
    """
    Extracts candidate road arcs and spatial/kinematic features for a single trip.
    Handles non-finite coordinates safely without shifting 0-based arc_id indices.
    """
    trip_id = trip_dir.name
    track = pd.read_csv(trip_dir / f"{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    nodes = pd.read_csv(trip_dir / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(trip_dir / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
    route_file = trip_dir / f"{trip_id}.route"
    has_route = route_file.exists()
    route = pd.read_csv(route_file, sep=r"\s+", header=None, names=["arc_id"]) if has_route else None

    track["lon"] = pd.to_numeric(track["lon"], errors="coerce").interpolate(limit_direction="both")
    track["lat"] = pd.to_numeric(track["lat"], errors="coerce").interpolate(limit_direction="both")
    track["timestamp"] = pd.to_numeric(track["timestamp"], errors="coerce").interpolate(limit_direction="both")

    tx, ty = TRANSFORMER.transform(track["lon"].values, track["lat"].values)
    nx_m, ny_m = TRANSFORMER.transform(nodes["lon"].values, nodes["lat"].values)

    invalid_nodes = ~np.isfinite(nx_m) | ~np.isfinite(ny_m)
    if invalid_nodes.any():
        print(f"  [Data Quality Warning] Trip {trip_id}: {invalid_nodes.sum()} invalid node(s) detected and masked.")
        nx_m = np.where(np.isfinite(nx_m), nx_m, 1e12)
        ny_m = np.where(np.isfinite(ny_m), ny_m, 1e12)

    x1 = nx_m[arcs["from_node"].values]
    y1 = ny_m[arcs["from_node"].values]
    x2 = nx_m[arcs["to_node"].values]
    y2 = ny_m[arcs["to_node"].values]

    arc_heading = np.arctan2(y2 - y1, x2 - x1)

    true_arc_per_obs = None
    if has_route and len(route) > 0:
        true_arc_per_obs = align_track_to_route(tx, ty, route["arc_id"].values, x1, y1, x2, y2)

    mid_x, mid_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    sample_pts = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([mid_x, mid_y]),
        np.column_stack([x2, y2]),
    ])
    sample_arc_ids = np.tile(np.arange(len(arcs)), 3)
    kdtree = cKDTree(sample_pts)

    dt = np.diff(track["timestamp"].values, prepend=track["timestamp"].values[0])
    dt[0] = 1.0
    dt = np.maximum(dt, 1e-3)

    dx_gps = np.diff(tx, prepend=tx[0])
    dy_gps = np.diff(ty, prepend=ty[0])
    step_dist = np.hypot(dx_gps, dy_gps)
    speed_mps = step_dist / dt
    speed_mps[0] = speed_mps[1] if len(speed_mps) > 1 else 0.0

    veh_dx = np.gradient(tx)
    veh_dy = np.gradient(ty)
    veh_heading = np.arctan2(veh_dy, veh_dx)

    rows = []
    missed_obs = 0

    for t in range(len(track)):
        if not (np.isfinite(tx[t]) and np.isfinite(ty[t])):
            continue

        nearby_sample_idxs = kdtree.query_ball_point([tx[t], ty[t]], r=radius_m + 150.0)
        if len(nearby_sample_idxs) == 0:
            _, nearest_idx = kdtree.query([tx[t], ty[t]], k=min(max_candidates, len(sample_pts)))
            cand_arcs = np.unique(sample_arc_ids[np.atleast_1d(nearest_idx)])
        else:
            cand_arcs = np.unique(sample_arc_ids[nearby_sample_idxs])

        perp_d, d_from, d_to, seg_len, alpha, is_int = point_to_segments_geometry(
            tx[t], ty[t], x1[cand_arcs], y1[cand_arcs], x2[cand_arcs], y2[cand_arcs]
        )

        keep_mask = perp_d <= radius_m
        if keep_mask.sum() < 3:
            order = np.argsort(perp_d)[:max_candidates]
        else:
            order = np.where(keep_mask)[0]
            order = order[np.argsort(perp_d[order])[:max_candidates]]

        cand_arcs = cand_arcs[order]
        perp_d = perp_d[order]
        d_from = d_from[order]
        d_to = d_to[order]
        seg_len = seg_len[order]
        alpha = alpha[order]
        is_int = is_int[order]

        true_arc = int(true_arc_per_obs[t]) if true_arc_per_obs is not None else -1
        if true_arc != -1 and true_arc not in cand_arcs:
            missed_obs += 1

        raw_diff = np.abs(arc_heading[cand_arcs] - veh_heading[t])
        heading_diff_rad = np.minimum(raw_diff, 2 * np.pi - raw_diff)
        heading_diff_deg = np.degrees(heading_diff_rad)
        cos_heading = np.cos(heading_diff_rad)

        min_d = perp_d.min()
        dist_rank = np.arange(1, len(cand_arcs) + 1)
        dist_diff = perp_d - min_d
        dist_ratio = perp_d / (min_d + 1.0)

        for k in range(len(cand_arcs)):
            rows.append({
                "trip_id": trip_id,
                "obs_idx": t,
                "timestamp": track["timestamp"].iloc[t],
                "arc_id": int(cand_arcs[k]),
                "perp_dist_m": float(perp_d[k]),
                "dist_to_from_node_m": float(d_from[k]),
                "dist_to_to_node_m": float(d_to[k]),
                "arc_length_m": float(seg_len[k]),
                "proj_ratio_alpha": float(alpha[k]),
                "is_interior_proj": float(is_int[k]),
                "heading_diff_deg": float(heading_diff_deg[k]),
                "cos_heading_diff": float(cos_heading[k]),
                "speed_mps": float(speed_mps[t]),
                "dt_s": float(dt[t]),
                "dist_rank": int(dist_rank[k]),
                "dist_diff_from_min_m": float(dist_diff[k]),
                "dist_ratio_to_min": float(dist_ratio[k]),
                "target": int(cand_arcs[k] == true_arc) if true_arc != -1 else -1,
            })

    return pd.DataFrame(rows), missed_obs

def build_datasets(radius_m: float = 50.0, max_candidates: int = 12):
    for partition in ["train", "student_test"]:
        part_dir = BASE_DATA_DIR / partition
        trip_dirs = sorted([d for d in part_dir.iterdir() if d.is_dir()])

        all_dfs = []
        total_obs = 0
        total_missed = 0

        for idx, trip_dir in enumerate(trip_dirs, 1):
            df_trip, missed = extract_trip_candidates(trip_dir, radius_m=radius_m, max_candidates=max_candidates)
            
            # --- FIX: Skip fully corrupt trips that return empty DataFrames ---
            if df_trip.empty:
                print(f"  [Warning] Trip {trip_dir.name} is entirely corrupt or empty. Skipping.")
                continue
            
            n_obs = df_trip["obs_idx"].nunique()
            total_obs += n_obs
            total_missed += missed
            all_dfs.append(df_trip)
            
            if idx % 10 == 0 or idx == len(trip_dirs):
                print(f"[{partition}] Processed {idx}/{len(trip_dirs)} trips...")

        full_df = pd.concat(all_dfs, ignore_index=True)
        miss_rate = (total_missed / total_obs) * 100.0
        print(f"\n>>> {partition.upper()} SUMMARY <<<")
        print(f"Total candidate rows : {len(full_df):,}")
        print(f"Total GPS obs        : {total_obs:,}")
        print(f"Candidate Miss Rate  : {miss_rate:.2f}% ({total_missed}/{total_obs} obs)")
        print(f"Positive class ratio : {full_df['target'].mean()*100:.2f}%\n")

        full_df.to_parquet(f"data/processed/{partition}_candidates.parquet", index=False)

    # Fit StandardScaler strictly on TRAIN features (zero leakage)
    train_df = pd.read_parquet("data/processed/train_candidates.parquet")
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS])
    joblib.dump({"scaler": scaler, "feature_cols": FEATURE_COLS}, "data/processed/feature_scaler.joblib")
    print("Saved fitted StandardScaler -> data/processed/feature_scaler.joblib")


if __name__ == "__main__":
    build_datasets(radius_m=50.0, max_candidates=12)