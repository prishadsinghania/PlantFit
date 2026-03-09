#!/usr/bin/env python3
"""
Plot Bondville management zones as a rectangular grid (county-style map).
Reads bondville_management_zones.csv, aggregates points into grid cells by majority zone,
and plots with pcolormesh for clear rectangular zones.
"""
import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "bondville_management_zones.csv"
OUTPUT_PNG = Path(__file__).resolve().parent / "bondville_management_zones_map.png"
OUTPUT_HTML = Path(__file__).resolve().parent / "bondville_management_zones_map.html"

# define color scheme
ZONE_COLORS = {
    "Low Yield Zone": "#D73027",
    "Medium Yield Zone": "#FEE08B",
    "High Yield Zone": "#1A9850",
}

# Zone order for encoding (0=Low, 1=Medium, 2=High) and NaN for no data
ZONE_ORDER = ["Low Yield Zone", "Medium Yield Zone", "High Yield Zone"]

# Grid resolution for lat and lon
GRID_LAT = 120
GRID_LON = 180


def get_extent(path: Path) -> tuple[float, float, float, float]:
    """Get lat_min, lat_max, lon_min, lon_max by scanning chunks."""
    lat_min, lat_max = 1e9, -1e9
    lon_min, lon_max = 1e9, -1e9
    for i, chunk in enumerate(pd.read_csv(path, chunksize=500_000)):
        lat_min = min(lat_min, chunk["lat"].min())
        lat_max = max(lat_max, chunk["lat"].max())
        lon_min = min(lon_min, chunk["lon"].min())
        lon_max = max(lon_max, chunk["lon"].max())
        if i >= 14:
            break
    return lat_min, lat_max, lon_min, lon_max


def build_zone_grid(path: Path, lat_edges: np.ndarray, lon_edges: np.ndarray) -> np.ndarray:
    """
    Stream CSV and count zone per grid cell; return 2D array of majority zone index (0/1/2).
    """
    nlat, nlon = len(lat_edges) - 1, len(lon_edges) - 1
    nz = len(ZONE_ORDER)
    zone_to_idx = {z: i for i, z in enumerate(ZONE_ORDER)}
    counts = np.zeros((nlat, nlon, nz), dtype=np.int64)

    for chunk in pd.read_csv(path, chunksize=400_000):
        lat = chunk["lat"].values
        lon = chunk["lon"].values
        zone = chunk["zone"].values
        lat_idx = np.searchsorted(lat_edges[1:], lat, side="left")
        lon_idx = np.searchsorted(lon_edges[1:], lon, side="left")
        lat_idx = np.clip(lat_idx, 0, nlat - 1)
        lon_idx = np.clip(lon_idx, 0, nlon - 1)
        for i in range(len(lat)):
            z = zone[i]
            if z in zone_to_idx:
                k = zone_to_idx[z]
                counts[lat_idx[i], lon_idx[i], k] += 1

    out = np.full((nlat, nlon), np.nan)
    total = counts.sum(axis=2)
    mask = total > 0
    out[mask] = np.argmax(counts, axis=2)[mask]
    return out


def main():
    print("Getting extent...")
    lat_min, lat_max, lon_min, lon_max = get_extent(CSV_PATH)
    lat_min -= 0.002
    lat_max += 0.002
    lon_min -= 0.002
    lon_max += 0.002
    print(f"  Lat [{lat_min:.4f}, {lat_max:.4f}], Lon [{lon_min:.4f}, {lon_max:.4f}]")

    lat_edges = np.linspace(lat_min, lat_max, GRID_LAT + 1)
    lon_edges = np.linspace(lon_min, lon_max, GRID_LON + 1)

    print("Building rectangular zone grid (streaming CSV)...")
    Z = build_zone_grid(CSV_PATH, lat_edges, lon_edges)

    Z_masked = np.ma.masked_invalid(Z)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # colormap: 0=Low, 1=Medium, 2=High
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([ZONE_COLORS[z] for z in ZONE_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, 3.5, 1), cmap.N)

    pc = ax.pcolormesh(
        lon_edges,
        lat_edges,
        Z_masked,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="none",
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Bondville management zones (rectangular grid)", fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    # Legend
    patches = [mpatches.Patch(color=ZONE_COLORS[z], label=z) for z in ZONE_ORDER]
    ax.legend(handles=patches, loc="upper left", framealpha=0.95, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {OUTPUT_PNG}")

    try:
        import folium
        try:
            from folium.raster_layers import ImageOverlay
        except ImportError:
            from folium import ImageOverlay
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.pcolormesh(lon_edges, lat_edges, Z_masked, cmap=cmap, norm=norm, shading="flat")
        ax2.set_xlim(lon_min, lon_max)
        ax2.set_ylim(lat_min, lat_max)
        ax2.axis("off")
        overlay_png = CSV_PATH.parent / "bondville_zone_overlay.png"
        plt.savefig(overlay_png, format="png", bbox_inches="tight", pad_inches=0, dpi=120, facecolor="white")
        plt.close(fig2)
        m = folium.Map(
            location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
            zoom_start=9,
            tiles="CartoDB positron",
        )
        ImageOverlay(
            str(overlay_png),
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            opacity=0.8,
        ).add_to(m)
        m.save(str(OUTPUT_HTML))
        print(f"Saved: {OUTPUT_HTML}")
    except Exception as e:
        print("Skipping HTML map:", e)


if __name__ == "__main__":
    main()
