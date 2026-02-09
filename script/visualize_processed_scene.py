"""
Visualize a processed (ego-frame, normalized) scene produced by MapGlowWrapper.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# Add VBD to path
vbd_root = Path(__file__).parent.parent.absolute()
if str(vbd_root) not in sys.path:
    sys.path.insert(0, str(vbd_root))

from vbd.data.dataset import WaymaxDataset
from vbd.model.MapGlowWrapper import MapGlowWrapper
from torch.utils.data import DataLoader


def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def plot_scene(inputs, pos_scale, max_agents=8, out_path=None, title=None, show=False, color_by_lane_type=True):
    """Plot map + agent history/future in ego frame (unnormalized)."""
    # Map data: [B, L, P, 6] (normalized)
    map_data = inputs["map_data"][0].detach().cpu()  # [L, P, 6]
    map_mask = inputs["map_mask"][0].detach().cpu()  # [L, P]

    # History/Future: [B, C, T, V] normalized
    history = inputs["history_data"][0].detach().cpu()  # [C, T_hist, V]
    target = inputs["input"][0].detach().cpu()  # [C, T, V]
    vehicle_mask = inputs["target_vehicle_mask"][0].detach().cpu()  # [V]
    timestep_mask = inputs["timestep_mask"][0].detach().cpu()  # [T, V]

    # Unnormalize positions
    map_xy = map_data[..., :2] * pos_scale
    history_xy = history[:2].permute(1, 2, 0) * pos_scale  # [T_hist, V, 2]
    target_xy = target[:2].permute(1, 2, 0) * pos_scale    # [T, V, 2]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot map points (optionally colored by lane_type)
    L, P = map_xy.shape[:2]
    lane_norm = mcolors.Normalize(vmin=0, vmax=20)
    for i in range(L):
        valid = map_mask[i]
        if not valid.any():
            continue
        pts = map_xy[i][valid]
        if color_by_lane_type:
            lane_type_norm = map_data[i][valid][:, 4]
            lane_type = torch.round(lane_type_norm * 20.0).int().clamp(min=0, max=20)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=lane_type.numpy(),
                cmap="tab20",
                norm=lane_norm,
                s=4,
                alpha=0.9,
            )
        else:
            ax.plot(pts[:, 0], pts[:, 1], ".", color="#999999", markersize=1)

    # Plot ego origin
    ax.scatter([0], [0], c="red", s=30, label="ego (0,0)")

    # Plot a subset of agents
    agent_indices = torch.where(vehicle_mask)[0].tolist()[:max_agents]
    for idx in agent_indices:
        hist = history_xy[:, idx]
        fut = target_xy[:, idx]

        # Mask invalid history points (padding zeros)
        hist_valid = (hist.abs().sum(dim=-1) > 0)
        hist_plot = hist.clone()
        hist_plot[~hist_valid] = float("nan")

        # Mask invalid future points using timestep_mask
        fut_valid = timestep_mask[:, idx].bool()
        fut_plot = fut.clone()
        fut_plot[~fut_valid] = float("nan")

        ax.plot(hist_plot[:, 0], hist_plot[:, 1], color="#139dec", linewidth=1.5)
        ax.plot(fut_plot[:, 0], fut_plot[:, 1], color="#0eff2e", linewidth=1.5)

        if hist_valid.any():
            last_valid = torch.where(hist_valid)[0][-1].item()
            ax.scatter(hist_plot[last_valid, 0], hist_plot[last_valid, 1], c="#aeb7bd", s=10)

    ax.set_aspect("equal", "box")
    ax.set_title(title or "Processed scene in ego frame (meters)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)

    if color_by_lane_type:
        mappable = plt.cm.ScalarMappable(cmap="tab20", norm=lane_norm)
        mappable.set_array([0, 20])
        cbar = plt.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("lane_type (approx)")

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    if show and not out_path:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize processed MapGlow inputs")
    parser.add_argument("-cfg", "--cfg", type=str, default="config/MapGlow.yaml")
    parser.add_argument("--index", type=int, default=0, help="Sample index")
    parser.add_argument("--num", type=int, default=10, help="Number of scenes to plot")
    parser.add_argument("--max_agents", type=int, default=32, help="Max agents to plot")
    parser.add_argument("--out", type=str, default=None, help="Output image path (single)")
    parser.add_argument("--out_dir", type=str, default="outputs/visuals", help="Output directory for multiple scenes")
    parser.add_argument("--show", action="store_true", help="Show plots instead of saving")
    parser.add_argument("--no_lane_color", action="store_true", help="Do not color lanes by type")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    data_dir = cfg.get("val_data_path") or cfg.get("train_data_path")
    if data_dir is None:
        raise ValueError("train_data_path/val_data_path not set in config")

    dataset = WaymaxDataset(data_dir=data_dir, anchor_path=cfg["anchor_path"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MapGlowWrapper(cfg=cfg)

    if args.num <= 1:
        # Single scene
        batch = None
        for i, b in enumerate(loader):
            if i == args.index:
                batch = b
                break
        if batch is None:
            raise IndexError("Index out of range for dataset")
        inputs = model.prepare_glow_inputs(batch)
        plot_scene(
            inputs,
            pos_scale=model.pos_scale.item(),
            max_agents=args.max_agents,
            out_path=args.out,
            title=f"Scene index {args.index} (ego frame)",
            show=args.show,
            color_by_lane_type=not args.no_lane_color,
        )
        return

    # Multiple scenes
    os.makedirs(args.out_dir, exist_ok=True)
    start = args.index
    end = args.index + args.num

    for i, batch in enumerate(loader):
        if i < start:
            continue
        if i >= end:
            break
        inputs = model.prepare_glow_inputs(batch)
        out_path = os.path.join(args.out_dir, f"scene_{i:06d}.png")
        plot_scene(
            inputs,
            pos_scale=model.pos_scale.item(),
            max_agents=args.max_agents,
            out_path=out_path,
            title=f"Scene index {i} (ego frame)",
            show=args.show,
            color_by_lane_type=not args.no_lane_color,
        )


if __name__ == "__main__":
    main()
