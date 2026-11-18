import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

# Update this path if you move the Student folder
BASE_DATA_DIR = Path(r"..\Student folder\Student folder")
PARTITIONS = ["train", "student_test"]


def find_file_by_ext(trip_dir: Path, ext: str):
    """Finds a file ending with the given extension (e.g., '.track' or 'track')."""
    matches = [f for f in trip_dir.iterdir() if f.is_file() and (f.suffix == ext or f.name.endswith(ext))]
    return matches[0] if matches else None


def audit_partition(partition_name: str):
    part_dir = BASE_DATA_DIR / partition_name
    trip_dirs = sorted([d for d in part_dir.iterdir() if d.is_dir()])
    print(f"\n{'='*60}\nPARTITION: {partition_name.upper()} ({len(trip_dirs)} trips)\n{'='*60}")

    records = []
    sample_printed = False

    for trip_dir in trip_dirs:
        trip_id = trip_dir.name
        files_in_dir = [f.name for f in trip_dir.iterdir()]

        track_f = find_file_by_ext(trip_dir, "track")
        nodes_f = find_file_by_ext(trip_dir, "nodes")
        arcs_f = find_file_by_ext(trip_dir, "arcs")
        route_f = find_file_by_ext(trip_dir, "route")

        # Load headerless whitespace-separated files
        track = pd.read_csv(track_f, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"]) if track_f else None
        nodes = pd.read_csv(nodes_f, sep=r"\s+", header=None, names=["lon", "lat"]) if nodes_f else None
        arcs = pd.read_csv(arcs_f, sep=r"\s+", header=None, names=["from_node", "to_node"]) if arcs_f else None
        route = pd.read_csv(route_f, sep=r"\s+", header=None, names=["arc_id"]) if route_f else None

        # Print raw sample from the very first trip so we can inspect exact structure
        if not sample_printed and track is not None:
            print(f"\n[SAMPLE TRIP: {trip_id}] Files found: {files_in_dir}")
            print("\n--- .track (first 5 rows) ---")
            print(track.head())
            print("\n--- .nodes (first 5 rows) ---")
            print(nodes.head())
            print("\n--- .arcs (first 5 rows) ---")
            print(arcs.head())
            if route is not None:
                print("\n--- .route (first 10 rows) ---")
                print(route.head(10))
            sample_printed = True

        # Check timestamps & gaps
        dt = track["timestamp"].diff().dropna()
        non_monotonic = (dt <= 0).sum()
        gaps_gt_1s = (dt > 1.0).sum()
        max_gap = dt.max() if len(dt) > 0 else 0

        # Check graph connectivity
        G = nx.DiGraph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from(zip(arcs["from_node"], arcs["to_node"]))
        isolated_nodes = nx.number_of_isolates(G)
        weak_components = nx.number_weakly_connected_components(G)

        # Check route validity
        has_route = route is not None
        route_len = len(route) if has_route else 0
        invalid_arc_ids = ((route["arc_id"] < 0) | (route["arc_id"] >= len(arcs))).sum() if has_route else 0

        records.append({
            "trip_id": trip_id,
            "n_gps": len(track),
            "n_nodes": len(nodes),
            "n_arcs": len(arcs),
            "has_route": has_route,
            "n_route": route_len,
            "track_nan": track.isna().sum().sum(),
            "lon_min": track["lon"].min(),
            "lon_max": track["lon"].max(),
            "lat_min": track["lat"].min(),
            "lat_max": track["lat"].max(),
            "non_monotonic_ts": non_monotonic,
            "gaps_gt_1s": gaps_gt_1s,
            "max_gap_s": max_gap,
            "isolated_nodes": isolated_nodes,
            "weak_components": weak_components,
            "invalid_arc_ids": invalid_arc_ids,
        })

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(f"reports/{partition_name}_eda_summary.csv", index=False)

    print(f"\n--- {partition_name.upper()} AGGREGATE STATS ---")
    print(f"Total GPS observations : {summary_df['n_gps'].sum()}")
    print(f"Avg GPS points / trip  : {summary_df['n_gps'].mean():.1f} (min={summary_df['n_gps'].min()}, max={summary_df['n_gps'].max()})")
    print(f"Avg Nodes / trip       : {summary_df['n_nodes'].mean():.1f}")
    print(f"Avg Arcs / trip        : {summary_df['n_arcs'].mean():.1f}")
    print(f"Trips with .route file : {summary_df['has_route'].sum()} / {len(summary_df)}")
    print(f"Total NaNs in .track   : {summary_df['track_nan'].sum()}")
    print(f"Non-monotonic ts count : {summary_df['non_monotonic_ts'].sum()}")
    print(f"Trips with gaps > 1s   : {(summary_df['gaps_gt_1s'] > 0).sum()} (max gap = {summary_df['max_gap_s'].max()}s)")
    print(f"Lon range              : [{summary_df['lon_min'].min():.5f}, {summary_df['lon_max'].max():.5f}]")
    print(f"Lat range              : [{summary_df['lat_min'].min():.5f}, {summary_df['lat_max'].max():.5f}]")
    print(f"Invalid route arc_ids  : {summary_df['invalid_arc_ids'].sum()}")
    return summary_df


if __name__ == "__main__":
    train_summary = audit_partition("train")
    test_summary = audit_partition("student_test")