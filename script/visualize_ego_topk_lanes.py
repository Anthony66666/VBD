#!/usr/bin/env python3
"""
Visualize ego top-k nearest lanes for one processed scenario.

Input file format: scenario_*.pkl produced by script/extract_data.py.
"""
import argparse
import glob
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _transform_polylines_to_ego(polylines: np.ndarray, ego_state: np.ndarray) -> np.ndarray:
    """
    polylines: [L, P, 5] -> [x, y, heading, tl_state, lane_type]
    ego_state: [3] -> [x, y, yaw]
    """
    out = polylines.copy()
    x = polylines[..., 0]
    y = polylines[..., 1]
    h = polylines[..., 2]

    dx = x - ego_state[0]
    dy = y - ego_state[1]
    c = np.cos(-ego_state[2])
    s = np.sin(-ego_state[2])
    out[..., 0] = dx * c - dy * s
    out[..., 1] = dx * s + dy * c
    out[..., 2] = _wrap_to_pi(h - ego_state[2])
    return out


def _select_topk_lanes(map_xy: np.ndarray, map_mask: np.ndarray, topk: int):
    """
    map_xy: [L, P, 2] in ego frame
    map_mask: [L, P] bool
    """
    dist = np.linalg.norm(map_xy, axis=-1)  # [L, P]
    dist = np.where(map_mask, dist, np.inf)
    lane_min_dist = np.min(dist, axis=1)  # [L]
    valid_lane = np.isfinite(lane_min_dist)

    valid_idx = np.where(valid_lane)[0]
    if valid_idx.size == 0:
        return np.array([], dtype=np.int64), lane_min_dist

    sorted_valid = valid_idx[np.argsort(lane_min_dist[valid_idx])]
    return sorted_valid[:topk], lane_min_dist


def _find_scenario_file(data_dir: str, scenario_id: str | None, index: int):
    files = sorted(glob.glob(os.path.join(data_dir, "scenario_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No scenario_*.pkl found in {data_dir}")

    if scenario_id is not None:
        exact = os.path.join(data_dir, f"scenario_{scenario_id}.pkl")
        if not os.path.exists(exact):
            raise FileNotFoundError(f"Scenario not found: {exact}")
        return exact

    if index < 0 or index >= len(files):
        raise IndexError(f"index={index} out of range [0, {len(files)-1}]")
    return files[index]


def main():
    parser = argparse.ArgumentParser(description="Visualize ego top-k lanes from processed scenario")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing scenario_*.pkl")
    parser.add_argument("--index", type=int, default=0, help="Scenario file index when --scenario_id is not set")
    parser.add_argument("--scenario_id", type=str, default=None, help="Use exact scenario id, e.g. abc123")
    parser.add_argument("--topk", type=int, default=16, help="Top-k nearest lanes for ego")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    args = parser.parse_args()

    scenario_file = _find_scenario_file(args.data_dir, args.scenario_id, args.index)
    with open(scenario_file, "rb") as f:
        data = pickle.load(f)

    required = ["agents_history", "polylines", "polylines_valid"]
    for k in required:
        if k not in data:
            raise KeyError(f"Missing key '{k}' in {scenario_file}")

    agents_history = np.asarray(data["agents_history"])  # [A, T, 8]
    polylines = np.asarray(data["polylines"])  # [L, P, 5]
    polylines_valid = np.asarray(data["polylines_valid"])  # [L]

    agents_id = data.get("agents_id", None)
    sdc_id = data.get("sdc_id", None)

    ego_idx = 0
    if agents_id is not None and sdc_id is not None:
        agents_id = np.asarray(agents_id).reshape(-1)
        sdc_id = int(np.asarray(sdc_id).reshape(-1)[0])
        hit = np.where(agents_id == sdc_id)[0]
        if hit.size > 0:
            ego_idx = int(hit[0])

    ego_state = agents_history[ego_idx, -1, :3]  # [x, y, yaw]
    polylines_local = _transform_polylines_to_ego(polylines, ego_state)

    point_valid = np.abs(polylines[..., :2]).sum(axis=-1) > 0
    map_mask = (polylines_valid[:, None] > 0) & point_valid  # [L, P]
    map_xy = polylines_local[..., :2]

    topk_idx, lane_min_dist = _select_topk_lanes(map_xy, map_mask, args.topk)

    scenario_name = Path(scenario_file).stem.replace("scenario_", "")
    if args.output is None:
        out_dir = os.path.join("outputs", "lane_topk_vis")
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"topk_{args.topk}_{scenario_name}.png")
    else:
        output = args.output
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Draw all valid lanes (gray)
    L = polylines.shape[0]
    for i in range(L):
        valid = map_mask[i]
        if not np.any(valid):
            continue
        pts = map_xy[i, valid]
        ax.plot(pts[:, 0], pts[:, 1], ".", color="#b0b0b0", markersize=1.0, alpha=0.45)

    # Draw top-k lanes (colored)
    cmap = plt.cm.get_cmap("tab20", max(1, len(topk_idx)))
    for rank, lane_i in enumerate(topk_idx):
        valid = map_mask[lane_i]
        pts = map_xy[lane_i, valid]
        color = cmap(rank)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=2.2, alpha=0.95)
        ax.plot(pts[:, 0], pts[:, 1], ".", color=color, markersize=2.0, alpha=0.95)
        if pts.shape[0] > 0:
            ax.text(pts[0, 0], pts[0, 1], str(rank + 1), color=color, fontsize=8, weight="bold")

    ax.scatter([0.0], [0.0], s=40, c="black", marker="x", label="ego")
    ax.set_title(f"Ego Top-{args.topk} Lanes | scenario={scenario_name} | ego_idx={ego_idx}")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"Scenario file: {scenario_file}")
    print(f"Ego index: {ego_idx}, ego_state[x,y,yaw]: {ego_state.tolist()}")
    print(f"Saved figure: {output}")
    print(f"Top-{len(topk_idx)} lanes (lane_idx, min_dist_to_ego_m):")
    for lane_i in topk_idx:
        print(f"  {int(lane_i):4d}  {float(lane_min_dist[lane_i]):8.3f}")


if __name__ == "__main__":
    main()

