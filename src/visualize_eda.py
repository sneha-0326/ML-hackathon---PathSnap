from pathlib import Path
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt

BASE_DATA_DIR = Path(r"..\Student folder\Student folder")


def create_trip_folium_map(partition: str, trip_id: str, out_html: str):
    trip_dir = BASE_DATA_DIR / partition / trip_id
    track = pd.read_csv(trip_dir / f"{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    nodes = pd.read_csv(trip_dir / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(trip_dir / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
    route_path = trip_dir / f"{trip_id}.route"
    route = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"]) if route_path.exists() else None

    center_lat, center_lon = track["lat"].mean(), track["lon"].mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="CartoDB Voyager",
    )
    # 1. Nearby Road Network Layer (arcs within trajectory bounding box + buffer)
    pad = 0.002
    lat_min, lat_max = track["lat"].min() - pad, track["lat"].max() + pad
    lon_min, lon_max = track["lon"].min() - pad, track["lon"].max() + pad

    u_coords = nodes.iloc[arcs["from_node"].values].reset_index(drop=True)
    v_coords = nodes.iloc[arcs["to_node"].values].reset_index(drop=True)

    mask = (
        u_coords["lat"].between(lat_min, lat_max) & u_coords["lon"].between(lon_min, lon_max)
    ) | (
        v_coords["lat"].between(lat_min, lat_max) & v_coords["lon"].between(lon_min, lon_max)
    )
    local_idx = np.where(mask)[0]

    road_layer = folium.FeatureGroup(name="Road Network (Local Arcs)", show=True)
    # Multi-polyline list renders ~20x faster than individual PolyLine objects
    road_segments = [
        [
            [u_coords.loc[idx, "lat"], u_coords.loc[idx, "lon"]],
            [v_coords.loc[idx, "lat"], v_coords.loc[idx, "lon"]],
        ]
        for idx in local_idx
    ]
    if road_segments:
        folium.PolyLine(road_segments, color="#888888", weight=2, opacity=0.45).add_to(road_layer)
    road_layer.add_to(m)

    # 2. Ground-Truth Reference Route Layer
    if route is not None and len(route) > 0:
        route_layer = folium.FeatureGroup(name="Reference Route (.route)", show=True)
        route_segments = []
        for arc_id in route["arc_id"].values:
            u = nodes.iloc[arcs.loc[arc_id, "from_node"]]
            v = nodes.iloc[arcs.loc[arc_id, "to_node"]]
            route_segments.append([[u["lat"], u["lon"]], [v["lat"], v["lon"]]])
        folium.PolyLine(route_segments, color="#16a34a", weight=5, opacity=0.85, tooltip="Reference Route").add_to(route_layer)
        route_layer.add_to(m)

    # 3. Raw Noisy GPS Observations Layer
    gps_layer = folium.FeatureGroup(name="Raw GPS Points (.track)", show=True)
    for idx, row in track.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2.5,
            color="#2563eb",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"Obs {idx} | t={row['timestamp']}s",
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # Add Legend & Layer Control
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; background: white;
                padding: 10px 14px; border: 1px solid #ccc; border-radius: 6px; font-size: 13px;">
      <b>PathSnap Legend</b><br>
      <span style="color:#2563eb;">&#9679;</span> Raw GPS Observations<br>
      <span style="color:#16a34a;">&#9473;</span> Ground-Truth Route<br>
      <span style="color:#888888;">&#9473;</span> Road Network Arcs
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    m.save(out_html)
    print(f"Saved interactive Folium map -> {out_html}")


def plot_eda_charts():
    train_df = pd.read_csv("reports/train_eda_summary.csv")
    test_df = pd.read_csv("reports/student_test_eda_summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(train_df["n_gps"], bins=20, color="#2563eb", alpha=0.8, edgecolor="black", label="Train (80)")
    axes[0].hist(test_df["n_gps"], bins=10, color="#f97316", alpha=0.8, edgecolor="black", label="Student Test (10)")
    axes[0].set_title("GPS Observations per Trip")
    axes[0].set_xlabel("Number of GPS Points")
    axes[0].set_ylabel("Trip Count")
    axes[0].legend()

    axes[1].scatter(train_df["n_gps"], train_df["n_route"], color="#16a34a", alpha=0.75, label="Train (80)")
    axes[1].scatter(test_df["n_gps"], test_df["n_route"], color="#dc2626", marker="s", label="Student Test (10)")
    axes[1].set_title("GPS Points vs. Traversed Road Arcs (.route)")
    axes[1].set_xlabel("GPS Points (.track rows)")
    axes[1].set_ylabel("Reference Route Arcs (.route rows)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("reports/figures/eda_distributions.png", dpi=200)
    print("Saved EDA distributions plot -> reports/figures/eda_distributions.png")


if __name__ == "__main__":
    create_trip_folium_map("train", "00000001", "reports/maps/train_00000001_eda.html")
    create_trip_folium_map("student_test", "00000000", "reports/maps/student_test_00000000_eda.html")
    plot_eda_charts()