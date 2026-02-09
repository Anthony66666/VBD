"""
Training script for MapGlow model using VBD's data pipeline.
Based on VBD's train.py structure - fully compatible with VBD infrastructure.
"""
import sys
import os
from pathlib import Path

# Add VBD to path
vbd_root = Path(__file__).parent.parent.absolute()
if str(vbd_root) not in sys.path:
    sys.path.insert(0, str(vbd_root))

import torch
import yaml
import datetime
import argparse
import pickle
import glob
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("MPLBACKEND", "Agg")

# Set TF and JAX to CPU only (with error handling for environment issues)
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
except Exception as e:
    print(f"Warning: Could not configure TensorFlow: {e}")

try:
    import jax
    jax.config.update("jax_platform_name", "cpu")
except Exception as e:
    print(f"Warning: Could not configure JAX: {e}")

from vbd.data.dataset import WaymaxDataset
from vbd.model.MapGlowWrapper import MapGlowWrapper
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data._utils.collate import default_collate

from matplotlib import pyplot as plt
import matplotlib as mpl


def load_config(file_path):
    """Load configuration from YAML file."""
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def _batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def _inspect_dataset_schema(dataset, name="dataset"):
    """
    Inspect one sample schema for mask-related fields and print risk hints.
    """
    required = [
        "agents_history",
        "agents_future",
        "agents_interested",
        "polylines",
        "polylines_valid",
        "traffic_light_points",
    ]
    preferred_masks = [
        "agents_history_valid",
        "agents_future_valid",
        "polylines_point_valid",
        "traffic_light_valid",
        "sdc_id",
        "agents_id",
    ]
    out = {
        "ok": True,
        "missing_required": [],
        "missing_preferred": [],
        "has": set(),
    }
    try:
        sample = dataset[0]
        keys = set(sample.keys()) if isinstance(sample, dict) else set()
        out["has"] = keys
        out["missing_required"] = [k for k in required if k not in keys]
        out["missing_preferred"] = [k for k in preferred_masks if k not in keys]
        if out["missing_required"]:
            out["ok"] = False
        print(f"[Schema:{name}] keys={sorted(list(keys))}", flush=True)
        if out["missing_required"]:
            print(f"[Schema:{name}] Missing REQUIRED keys: {out['missing_required']}", flush=True)
        if out["missing_preferred"]:
            print(
                f"[Schema:{name}] Missing preferred mask/ego keys: {out['missing_preferred']}",
                flush=True,
            )
            if "sdc_id" in out["missing_preferred"]:
                print(
                    f"[Schema:{name}] WARNING: missing sdc_id -> wrapper falls back to agent index 0 as ego frame. "
                    "This can significantly hurt training quality.",
                    flush=True,
                )
    except Exception as e:
        out["ok"] = False
        print(f"[Schema:{name}] Failed to inspect sample: {e}", flush=True)
    return out


def _set_simulator_state_timestep(simulator_state, timestep):
    """Best-effort set simulator state's current timestep."""
    if simulator_state is None or timestep is None:
        return simulator_state

    try:
        import numpy as np
        timestep = int(timestep)
        max_t = None
        if hasattr(simulator_state, "log_trajectory") and hasattr(simulator_state.log_trajectory, "valid"):
            valid = np.asarray(simulator_state.log_trajectory.valid)
            if valid.ndim >= 2:
                max_t = valid.shape[1] - 1
        if max_t is not None:
            timestep = max(0, min(timestep, int(max_t)))

        if hasattr(simulator_state, "replace"):
            return simulator_state.replace(timestep=np.int32(timestep))
        simulator_state.timestep = np.int32(timestep)
    except Exception:
        pass
    return simulator_state


def _load_simulator_state_from_dataset(dataset, sample_index=0, timestep=None):
    """Load scenario_raw (SimulatorState) from processed dataset pkl if available."""
    data_list = getattr(dataset, "data_list", None)
    if data_list is None or len(data_list) == 0:
        return None
    if sample_index < 0 or sample_index >= len(data_list):
        return None

    sample_path = data_list[sample_index]
    try:
        with open(sample_path, "rb") as f:
            sample = pickle.load(f)
    except Exception:
        return None

    if not isinstance(sample, dict):
        return None

    simulator_state = None
    if "scenario_raw" in sample:
        simulator_state = sample["scenario_raw"]
    elif "scenario" in sample:
        simulator_state = sample["scenario"]

    required_fields = (
        "log_trajectory",
        "sim_trajectory",
        "roadgraph_points",
        "log_traffic_light",
        "object_metadata",
    )
    if simulator_state is None or not all(hasattr(simulator_state, k) for k in required_fields):
        return None

    return _set_simulator_state_timestep(simulator_state, timestep)


def _normalize_scenario_id(scenario_id):
    if scenario_id is None:
        return None
    try:
        import numpy as np
        if isinstance(scenario_id, np.ndarray):
            if scenario_id.size == 1:
                scenario_id = scenario_id.reshape(-1)[0]
        if isinstance(scenario_id, bytes):
            scenario_id = scenario_id.decode("utf-8")
    except Exception:
        pass
    return str(scenario_id)


def _get_scenario_id_from_dataset_sample(dataset, sample_index=0):
    data_list = getattr(dataset, "data_list", None)
    if data_list is None or len(data_list) == 0:
        return None
    if sample_index < 0 or sample_index >= len(data_list):
        return None

    sample_path = data_list[sample_index]
    scenario_id = None
    try:
        with open(sample_path, "rb") as f:
            sample = pickle.load(f)
        if isinstance(sample, dict):
            scenario_id = sample.get("scenario_id", None)
    except Exception:
        scenario_id = None

    if scenario_id is None:
        stem = Path(sample_path).stem
        prefix = "scenario_"
        if stem.startswith(prefix):
            scenario_id = stem[len(prefix):]
    return _normalize_scenario_id(scenario_id)


def _load_simulator_state_from_raw_tfrecord(
    raw_data_path,
    target_scenario_id,
    timestep=None,
    max_num_objects=128,
    max_scan=None,
):
    """
    Load SimulatorState from original TFRecord directory by scenario_id.
    This is a fallback when processed pkl does not contain scenario_raw.
    """
    if raw_data_path is None or target_scenario_id is None:
        return None
    if not os.path.exists(raw_data_path):
        return None

    try:
        from waymax.config import DatasetConfig, DataFormat
        from vbd.data.waymax_utils import create_iter
    except Exception:
        return None

    resolved_path = raw_data_path
    if os.path.isdir(raw_data_path):
        shard_files = sorted(glob.glob(os.path.join(raw_data_path, "*.tfrecord-*")))
        if len(shard_files) > 0:
            m = re.match(r"(.+)-\d+-of-(\d+)$", shard_files[0])
            if m is not None:
                prefix, n_shards = m.group(1), int(m.group(2))
                resolved_path = f"{prefix}@{n_shards}"
            else:
                resolved_path = shard_files[0]
        else:
            single_files = sorted(glob.glob(os.path.join(raw_data_path, "*.tfrecord")))
            if len(single_files) > 0:
                resolved_path = single_files[0]

    try:
        cfg = DatasetConfig(
            path=resolved_path,
            max_num_objects=max_num_objects,
            repeat=1,
            max_num_rg_points=30000,
            data_format=DataFormat.TFRECORD,
            deterministic=True,
        )
        for i, (sid, scenario) in enumerate(create_iter(cfg)):
            if _normalize_scenario_id(sid) == target_scenario_id:
                return _set_simulator_state_timestep(scenario, timestep)
            if max_scan is not None and (i + 1) >= int(max_scan):
                break
    except Exception as e:
        print(
            f"[Sampling] Raw TFRecord lookup failed: path={resolved_path}, reason={e}",
            flush=True,
        )
        return None
    return None


def _plot_mapglow_sample(
    batch,
    pred_traj,
    save_path,
    max_agents=8,
    use_waymax_vis=True,
    simulator_state=None,
):
    """
    Plot global-frame map + history + GT future + sampled future.
    Prefer Waymax visualization if available, fallback to plain matplotlib.
    """
    import numpy as np

    b = 0
    agents_history = batch["agents_history"][b].detach().cpu().numpy()  # [A, T_hist, 8]
    agents_future = batch["agents_future"][b].detach().cpu().numpy()    # [A, T_fut, 5]
    agents_interested = batch["agents_interested"][b].detach().cpu().numpy()
    polylines = batch["polylines"][b].detach().cpu().numpy()
    polylines_valid = batch["polylines_valid"][b].detach().cpu().numpy()
    polylines_point_valid = None
    if "polylines_point_valid" in batch:
        polylines_point_valid = batch["polylines_point_valid"][b].detach().cpu().numpy().astype(bool)
    traffic_light_points = None
    traffic_light_valid = None
    if "traffic_light_points" in batch:
        traffic_light_points = batch["traffic_light_points"][b].detach().cpu().numpy()
        if "traffic_light_valid" in batch:
            traffic_light_valid = batch["traffic_light_valid"][b].detach().cpu().numpy().astype(bool)
        else:
            traffic_light_valid = (np.abs(traffic_light_points[..., :2]).sum(axis=-1) > 0)
    pred = pred_traj[b].detach().cpu().numpy()  # [A, T, 3], global or local depending on caller
    A_pred = pred.shape[0]
    agents_history = agents_history[:A_pred]
    agents_future = agents_future[:A_pred]
    agents_interested = agents_interested[:A_pred]

    if "agents_future_valid" in batch:
        future_valid = batch["agents_future_valid"][b].detach().cpu().numpy().astype(bool)[:A_pred]
    else:
        future_valid = (np.abs(agents_future[..., :2]).sum(axis=-1) > 0)
    pred_len = pred.shape[1]
    pred_valid = future_valid[:, 1:1 + pred_len]
    if pred_valid.shape[1] < pred_len:
        pad = np.zeros((A_pred, pred_len - pred_valid.shape[1]), dtype=bool)
        pred_valid = np.concatenate([pred_valid, pad], axis=1)

    interested_mask = agents_interested > 0
    has_valid_pred = pred_valid.any(axis=1)
    has_valid_any = future_valid.any(axis=1)
    has_valid_current = np.abs(agents_history[:, -1, :2]).sum(axis=-1) > 0
    valid_agent_mask = has_valid_any | has_valid_current
    draw_idx = np.where(valid_agent_mask)[0]
    if draw_idx.size == 0:
        draw_idx = np.where(has_valid_pred)[0]
    if draw_idx.size == 0:
        draw_idx = np.where(interested_mask)[0]
    if draw_idx.size == 0:
        draw_idx = np.arange(min(max_agents, pred.shape[0]))
    else:
        draw_idx = draw_idx[:max_agents]

    def draw_agent_box(ax, cx, cy, yaw, length, width, face_color, edge_color, z=6, alpha=0.9, lw=0.9):
        if (not np.isfinite(cx)) or (not np.isfinite(cy)) or (not np.isfinite(yaw)):
            return
        if (not np.isfinite(length)) or (not np.isfinite(width)):
            return
        length = float(max(length, 0.5))
        width = float(max(width, 0.3))
        rect = plt.Rectangle(
            (float(cx) - length / 2.0, float(cy) - width / 2.0),
            length,
            width,
            linewidth=lw,
            edgecolor=edge_color,
            facecolor=face_color,
            alpha=alpha,
            zorder=z,
            transform=mpl.transforms.Affine2D().rotate_around(float(cx), float(cy), float(yaw)) + ax.transData,
        )
        ax.add_patch(rect)

    def draw_traffic_lights(ax, tl_points, tl_valid, to_xy_fn=None, z=4):
        if tl_points is None or tl_valid is None:
            return
        if tl_points.ndim != 2 or tl_points.shape[0] == 0:
            return
        # Waymo-like state ids:
        # 0 unknown, 1/4 stop(red), 2/5 caution(yellow), 3/6 go(green), 7/8 flashing
        color_map = {
            1: "#d62728", 4: "#d62728",
            2: "#ffbf00", 5: "#ffbf00",
            3: "#2ca02c", 6: "#2ca02c",
            7: "#ff7f0e", 8: "#ff7f0e",
        }
        for i in range(tl_points.shape[0]):
            if not bool(tl_valid[i]):
                continue
            x, y = float(tl_points[i, 0]), float(tl_points[i, 1])
            if (not np.isfinite(x)) or (not np.isfinite(y)):
                continue
            if to_xy_fn is not None:
                p = to_xy_fn(np.array([[x, y]], dtype=np.float32))[0]
                x, y = float(p[0]), float(p[1])
            state = int(round(float(tl_points[i, 2]))) if tl_points.shape[1] > 2 else 0
            c = color_map.get(state, "#555555")
            ax.scatter(x, y, s=28, c=c, edgecolors="#111111", linewidths=0.6, marker="o", zorder=z, alpha=0.95)

    # Simple local-frame visualization path: draw sampled trajectories directly.
    if not use_waymax_vis:
        # Build ego-local transform for GT/map/history overlays.
        ego_idx = None
        if "agents_id" in batch and "sdc_id" in batch:
            try:
                agents_id = batch["agents_id"][b].detach().cpu().numpy()[:A_pred]
                sdc_id = int(np.asarray(batch["sdc_id"][b].detach().cpu().numpy()).reshape(-1)[0])
                matches = np.where(agents_id == sdc_id)[0]
                if matches.size > 0:
                    ego_idx = int(matches[0])
            except Exception:
                ego_idx = None
        # Match MapGlowWrapper._get_ego_states fallback: use first slot when sdc is unavailable.
        if ego_idx is None:
            ego_idx = 0

        ego_state = agents_history[ego_idx, -1, :3].copy()
        if np.abs(ego_state[:2]).sum() <= 1e-6:
            for i in range(agents_history.shape[0]):
                cand = agents_history[i, -1, :3]
                if np.abs(cand[:2]).sum() > 1e-6:
                    ego_state = cand.copy()
                    break

        ego_x, ego_y, ego_yaw = float(ego_state[0]), float(ego_state[1]), float(ego_state[2])
        cos_y = np.cos(ego_yaw)
        sin_y = np.sin(ego_yaw)

        def to_local_xy(xy):
            dx = xy[..., 0] - ego_x
            dy = xy[..., 1] - ego_y
            lx = dx * cos_y + dy * sin_y
            ly = -dx * sin_y + dy * cos_y
            return np.stack([lx, ly], axis=-1)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Map polylines (gray) in local frame.
        for i in range(polylines.shape[0]):
            if polylines_valid[i] == 0:
                continue
            line_xy = polylines[i, :, :2]
            if polylines_point_valid is not None:
                valid_pt = polylines_point_valid[i]
            else:
                valid_pt = np.abs(line_xy).sum(axis=-1) > 0
            line_xy = line_xy[valid_pt]
            if line_xy.shape[0] > 1:
                line_local = to_local_xy(line_xy)
                ax.plot(line_local[:, 0], line_local[:, 1], ".", markersize=1, alpha=0.45, color="#999999")
        draw_traffic_lights(ax, traffic_light_points, traffic_light_valid, to_xy_fn=to_local_xy, z=4)

        # History (blue) + GT (green) + Pred (red).
        hist_valid = None
        if "agents_history_valid" in batch:
            hist_valid = batch["agents_history_valid"][b].detach().cpu().numpy().astype(bool)[:A_pred]
        for i in draw_idx:
            hist_xy = agents_history[i, :, :2].copy()
            if hist_valid is not None:
                hv = hist_valid[i]
            else:
                hv = np.abs(hist_xy).sum(axis=-1) > 0
            hist_xy = hist_xy[hv]
            if hist_xy.shape[0] > 1:
                hist_local = to_local_xy(hist_xy)
                ax.plot(hist_local[:, 0], hist_local[:, 1], color="#1f77b4", alpha=0.9, linewidth=1.2)

            gt_xy = agents_future[i, 1:1 + pred.shape[1], :2].copy()
            gt_mask = future_valid[i, 1:1 + pred.shape[1]]
            gt_local = to_local_xy(gt_xy)
            gt_local[~gt_mask] = np.nan
            if np.isfinite(gt_local).any():
                ax.plot(gt_local[:, 0], gt_local[:, 1], color="#2ca02c", alpha=0.9, linewidth=1.4)

        for i in draw_idx:
            pr = pred[i, :, :2].copy()
            pr[~pred_valid[i]] = np.nan
            if np.isfinite(pr).any():
                ax.plot(pr[:, 0], pr[:, 1], color="#d62728", alpha=0.9, linewidth=1.6)

        # Agent boxes at current timestep (local frame), using length/width.
        marker_idx = set(draw_idx.tolist())
        marker_idx.add(int(ego_idx))
        for i in marker_idx:
            cur_xy = agents_history[i, -1, :2].copy()
            if np.abs(cur_xy).sum() <= 1e-6:
                continue
            cur_local = to_local_xy(cur_xy.reshape(1, 2))[0]
            cur_yaw = float(agents_history[i, -1, 2])
            local_yaw = np.arctan2(np.sin(cur_yaw - ego_yaw), np.cos(cur_yaw - ego_yaw))
            agent_len = float(agents_history[i, -1, 5]) if agents_history.shape[-1] > 5 else 4.5
            agent_wid = float(agents_history[i, -1, 6]) if agents_history.shape[-1] > 6 else 1.8
            if int(i) == int(ego_idx):
                draw_agent_box(
                    ax,
                    cur_local[0],
                    cur_local[1],
                    local_yaw,
                    agent_len,
                    agent_wid,
                    face_color="#8a2be2",
                    edge_color="#000000",
                    z=7,
                )
            else:
                is_interested = bool(interested_mask[i])
                draw_agent_box(
                    ax,
                    cur_local[0],
                    cur_local[1],
                    local_yaw,
                    agent_len,
                    agent_wid,
                    face_color="#1f77b4" if is_interested else "#444444",
                    edge_color="#000000",
                    z=6,
                )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title("MapGlow sample (local): interested box=blue, other valid box=gray, ego=purple")
        ax.axis("off")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    if use_waymax_vis and simulator_state is not None:
        try:
            from vbd.waymax_visualization.plotting import plot_state

            traj_preds = pred[draw_idx, :, :2].copy()  # plotting.py expects [N, T, D], D>=2
            for j, i in enumerate(draw_idx):
                traj_preds[j, ~pred_valid[i]] = np.nan

            center_xy = None
            for i in draw_idx:
                if np.abs(agents_history[i, -1, :2]).sum() > 0:
                    center_xy = agents_history[i, -1, :2]
                    break

            fig, ax = plot_state(
                current_state=simulator_state,
                log_traj=True,
                traj_preds=traj_preds,
                past_traj_length=max(0, agents_history.shape[1] - 1),
                dx=90,
                center_xy=center_xy,
                tick_off=True,
                return_ax=True,
                traj_color="#d62728",
            )

            # Overlay GT future for direct comparison (green).
            for i in draw_idx:
                gt_xy = agents_future[i, 1:1 + pred.shape[1], :2].copy()
                gt_mask = future_valid[i, 1:1 + pred.shape[1]]
                gt_xy[~gt_mask] = np.nan
                if np.isfinite(gt_xy).any():
                    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="#2ca02c", alpha=0.9, linewidth=1.6)

            ax.set_title("MapGlow sample: GT(green) vs Pred(red)")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return
        except Exception as e:
            print(f"[Sampling] plot_state failed, fallback to basic visualization: {e}", flush=True)

    vis_config = None
    ax = None
    fig = None
    waymax_ok = False
    if use_waymax_vis:
        try:
            from vbd.waymax_visualization import utils as viz_utils
            from vbd.waymax_visualization import viz as waymax_viz
            from waymax import datatypes

            vis_config = viz_utils.VizConfig()
            fig, ax = viz_utils.init_fig_ax(vis_config)
            waymax_ok = True

            traj_full = np.concatenate([agents_history[..., :5], agents_future[..., :5]], axis=1)
            x = traj_full[..., 0]
            y = traj_full[..., 1]
            yaw = traj_full[..., 2]
            vel_x = traj_full[..., 3]
            vel_y = traj_full[..., 4]
            last_dims = agents_history[:, -1, 5:7]
            length = np.repeat(last_dims[:, 0:1], traj_full.shape[1], axis=1)
            width = np.repeat(last_dims[:, 1:2], traj_full.shape[1], axis=1)
            valid = np.concatenate(
                [np.abs(agents_history[..., :2]).sum(axis=-1) > 0, future_valid], axis=1
            )

            traj = datatypes.Trajectory(
                x=x,
                y=y,
                yaw=yaw,
                vel_x=vel_x,
                vel_y=vel_y,
                length=length,
                width=width,
                valid=valid,
            )

            is_controlled = agents_interested > 0
            time_idx = agents_history.shape[1] - 1
            waymax_viz.plot_trajectory(
                ax,
                traj,
                is_controlled=is_controlled,
                time_idx=time_idx,
                indices=np.arange(traj.num_objects),
            )
        except Exception:
            waymax_ok = False

    if not waymax_ok:
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(polylines.shape[0]):
            if polylines_valid[i] == 0:
                continue
            line = polylines[i]
            line = line[np.abs(line[:, :2]).sum(axis=-1) > 0]
            if line.shape[0] > 1:
                ax.plot(line[:, 0], line[:, 1], ".", markersize=1, alpha=0.5, color="#999999")
        draw_traffic_lights(ax, traffic_light_points, traffic_light_valid, to_xy_fn=None, z=4)

        for i in draw_idx:
            hist = agents_history[i]
            hist = hist[np.abs(hist[:, :2]).sum(axis=-1) > 0]
            if hist.shape[0] > 1:
                ax.plot(hist[:, 0], hist[:, 1], color="#1f77b4", alpha=0.9, linewidth=1.2)

            gt = agents_future[i]
            gt_mask = future_valid[i]
            # agents_future includes current step at index 0; predictions are future-only.
            gt = gt[1:1 + pred.shape[1]]
            gt_mask = gt_mask[1:1 + pred.shape[1]]
            gt_plot = gt.copy()
            gt_plot[~gt_mask] = np.nan
            if np.isfinite(gt_plot[:, :2]).any():
                ax.plot(gt_plot[:, 0], gt_plot[:, 1], color="#2ca02c", alpha=0.9, linewidth=1.4)

        # Current agent boxes in global frame fallback.
        if draw_idx.size > 0:
            ego_idx_fallback = int(draw_idx[0])
            if "agents_id" in batch and "sdc_id" in batch:
                try:
                    agents_id = batch["agents_id"][b].detach().cpu().numpy()[:A_pred]
                    sdc_id = int(np.asarray(batch["sdc_id"][b].detach().cpu().numpy()).reshape(-1)[0])
                    matches = np.where(agents_id == sdc_id)[0]
                    if matches.size > 0:
                        ego_idx_fallback = int(matches[0])
                except Exception:
                    pass
            marker_idx = set(draw_idx.tolist())
            marker_idx.add(ego_idx_fallback)
            for i in marker_idx:
                cur = agents_history[i, -1]
                cur_xy = cur[:2]
                if np.abs(cur_xy).sum() <= 1e-6:
                    continue
                cur_yaw = float(cur[2])
                agent_len = float(cur[5]) if agents_history.shape[-1] > 5 else 4.5
                agent_wid = float(cur[6]) if agents_history.shape[-1] > 6 else 1.8
                if int(i) == int(ego_idx_fallback):
                    draw_agent_box(
                        ax, cur_xy[0], cur_xy[1], cur_yaw, agent_len, agent_wid,
                        face_color="#8a2be2", edge_color="#000000", z=7
                    )
                else:
                    is_interested = bool(interested_mask[i])
                    draw_agent_box(
                        ax, cur_xy[0], cur_xy[1], cur_yaw, agent_len, agent_wid,
                        face_color="#1f77b4" if is_interested else "#444444",
                        edge_color="#000000",
                        z=6
                    )

    # Predictions (always red), masked by valid future timesteps.
    for i in draw_idx:
        pr = pred[i, :, :2].copy()
        pr[~pred_valid[i]] = np.nan
        if np.isfinite(pr).any():
            ax.plot(pr[:, 0], pr[:, 1], color="#d62728", alpha=0.9, linewidth=1.6)

    ax.set_title("MapGlow sample: interested box=blue, other valid box=gray, ego=purple")
    ax.axis("off")

    if waymax_ok:
        from vbd.waymax_visualization import utils as viz_utils
        center_xy = None
        for i in range(min(max_agents, agents_history.shape[0])):
            if agents_interested[i] <= 0:
                continue
            if np.abs(agents_history[i, -1, :2]).sum() > 0:
                center_xy = agents_history[i, -1, :2]
                break
        if center_xy is not None:
            viz_utils.center_at_xy(ax, center_xy, vis_config)
        img = viz_utils.img_from_fig(fig)
        viz_utils.save_img_as_png(img, save_path)
    else:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _log_wandb_image(trainer, image_path, step):
    loggers = []
    if getattr(trainer, "loggers", None):
        loggers = trainer.loggers
    elif getattr(trainer, "logger", None) is not None:
        loggers = [trainer.logger]

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            try:
                import wandb
                logger.experiment.log(
                    {"train/sample_vis": wandb.Image(image_path), "global_step": step},
                    step=step,
                )
            except Exception as e:
                print(f"[Sampling] Wandb image logging failed: {e}")
            break


class MapGlowSamplingCallback(pl.Callback):
    def __init__(
        self,
        val_loader,
        output_path,
        every_n_steps=2000,
        max_agents=8,
        n_samples=1,
        temperature=None,
        pos_scale=100.0,
        use_waymax_vis=True,
        raw_data_path=None,
        max_raw_scan=50000,
        max_num_objects=128,
        rotate_scene=True,
    ):
        super().__init__()
        self.val_loader = val_loader
        self.output_path = output_path
        self.every_n_steps = int(every_n_steps)
        self.max_agents = int(max_agents)
        self.n_samples = int(n_samples)
        self.temperature = temperature
        self.pos_scale = float(pos_scale)
        self.use_waymax_vis = bool(use_waymax_vis)
        self.raw_data_path = raw_data_path
        self.max_raw_scan = None if max_raw_scan is None else int(max_raw_scan)
        self.max_num_objects = int(max_num_objects)
        self.rotate_scene = bool(rotate_scene)
        self._last_sample_step = 0
        self._sample_index = -1
        self._sim_state_cache = {}
        self._sim_state_miss = set()
        self._sim_state_warned = False

    def _next_sample_index(self):
        dataset = getattr(self.val_loader, "dataset", None)
        if dataset is None:
            return None
        n = len(dataset)
        if n <= 0:
            return None
        if not self.rotate_scene:
            if self._sample_index < 0:
                self._sample_index = 0
            return self._sample_index
        self._sample_index = (self._sample_index + 1) % n
        return self._sample_index

    def _get_batch(self, sample_index=None, fallback_batch=None):
        dataset = getattr(self.val_loader, "dataset", None)
        if dataset is not None and sample_index is not None:
            try:
                sample = dataset[sample_index]
                # Dataset __getitem__ already returns tensor dict; use default_collate
                # to avoid numpy round-trip and dtype/object issues.
                return default_collate([sample])
            except Exception as e:
                print(f"[Sampling] Failed to load val sample index={sample_index}, fallback to dataloader: {e}", flush=True)

        # Fallback: first batch from val_loader.
        try:
            return next(iter(self.val_loader))
        except Exception:
            # If val loader is empty/missing, fallback to current training batch.
            if fallback_batch is not None:
                return fallback_batch
            raise RuntimeError(
                "Sampling callback cannot get batch: val_loader is empty and no training batch fallback is available."
            )

    def _get_simulator_state(self, sample_batch, sample_index=None):
        if sample_index is not None:
            if sample_index in self._sim_state_cache:
                return self._sim_state_cache[sample_index]
            if sample_index in self._sim_state_miss:
                return None

        dataset = getattr(self.val_loader, "dataset", None)
        timestep = None
        if isinstance(sample_batch, dict) and "agents_history" in sample_batch:
            timestep = int(sample_batch["agents_history"].shape[2] - 1)

        simulator_state = _load_simulator_state_from_dataset(
            dataset=dataset,
            sample_index=0 if sample_index is None else sample_index,
            timestep=timestep,
        )
        if simulator_state is None and self.raw_data_path:
            scenario_id = _get_scenario_id_from_dataset_sample(dataset, sample_index=0 if sample_index is None else sample_index)
            if scenario_id is not None:
                print(
                    f"[Sampling] scenario_raw missing in pkl, searching raw TFRecord by scenario_id={scenario_id} (idx={sample_index}) ...",
                    flush=True,
                )
                simulator_state = _load_simulator_state_from_raw_tfrecord(
                    raw_data_path=self.raw_data_path,
                    target_scenario_id=scenario_id,
                    timestep=timestep,
                    max_num_objects=self.max_num_objects,
                    max_scan=self.max_raw_scan,
                )
                if simulator_state is not None:
                    print("[Sampling] Loaded SimulatorState from raw TFRecord for plotting.py visualization.", flush=True)
        if simulator_state is None and not self._sim_state_warned:
            print(
                "[Sampling] scenario_raw not found in validation pkl, cannot use plotting.py::plot_state. "
                "Fallback to tensor-based visualization. To enable full Waymax plotting, set --sample_raw_data_path "
                "or extract data with --save_raw.",
                flush=True,
            )
            self._sim_state_warned = True
        if sample_index is not None:
            if simulator_state is None:
                self._sim_state_miss.add(sample_index)
            else:
                self._sim_state_cache[sample_index] = simulator_state
        return simulator_state

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0:
            return
        # Trigger every N optimizer steps, not only exact multiples.
        if (step - self._last_sample_step) < self.every_n_steps:
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        sample_index = self._next_sample_index()
        sample_batch = self._get_batch(sample_index=sample_index, fallback_batch=batch)
        sample_batch = _batch_to_device(sample_batch, pl_module.device)
        simulator_state = None
        if self.use_waymax_vis:
            simulator_state = self._get_simulator_state(sample_batch, sample_index=sample_index)

        pl_module.eval()
        try:
            with torch.no_grad():
                pred = pl_module.sample(
                    sample_batch,
                    n_samples=max(self.n_samples, 1),
                    temperature=self.temperature,
                    return_global=bool(self.use_waymax_vis),
                )  # [B, n, A, T, 3]
                pred_global = pred[:, 0]  # [B, A, T, 3]

            save_path = os.path.join(
                self.output_path, f"sample_step_{step:08d}.png"
            )
            _plot_mapglow_sample(
                sample_batch,
                pred_global,
                save_path=save_path,
                max_agents=self.max_agents,
                use_waymax_vis=self.use_waymax_vis,
                simulator_state=simulator_state,
            )
            if sample_index is not None:
                print(f"[Sampling] Visualization sample index={sample_index}", flush=True)
            print(f"[Sampling] Saved sample visualization: {save_path}", flush=True)
            _log_wandb_image(trainer, save_path, step)
            self._last_sample_step = step
        except Exception as e:
            print(f"[Sampling] Failed at step {trainer.global_step}: {e}", flush=True)
        finally:
            pl_module.train()


def build_logger(cfg, output_path, model_name):
    """Build logger with robust wandb fallback."""
    use_wandb = cfg.get("use_wandb", False)
    if not use_wandb:
        return CSVLogger(output_path, name="MapGlow", version=1, flush_logs_every_n_steps=100)

    # Prefer explicit team entity. Keep backward compatibility with old "username".
    wandb_entity = (
        cfg.get("wandb_entity")
        or cfg.get("entity")
        or cfg.get("team")
        or cfg.get("username")
    )

    wandb_kwargs = dict(
        name=cfg.get("wandb_name", model_name),
        project=cfg.get("project", "MapGlow"),
        log_model=False,
        dir=output_path,
    )
    if wandb_entity:
        wandb_kwargs["entity"] = wandb_entity
    if cfg.get("wandb_mode") is not None:
        wandb_kwargs["mode"] = cfg.get("wandb_mode")

    logger = WandbLogger(**wandb_kwargs)
    try:
        # Force wandb.init here so permission errors are caught early.
        _ = logger.experiment
        print(f"  Wandb enabled: project={wandb_kwargs['project']}, entity={wandb_kwargs.get('entity', '<default>')}")
        return logger
    except Exception as e:
        err = str(e)
        print("\nWarning: Wandb init failed, fallback to CSV logger.")
        print(f"  Reason: {err}")
        if "Personal entities are disabled" in err or "PERMISSION_ERROR" in err:
            print("  Hint: set a team entity via --wandb_entity <team_name> or config 'wandb_entity'.")
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish(exit_code=1)
        except Exception:
            pass
        return CSVLogger(output_path, name="MapGlow", version=1, flush_logs_every_n_steps=100)


def train(cfg):
    """Main training function - same structure as VBD's train()."""
    print("=" * 60)
    print("Starting MapGlow Training with VBD Pipeline")
    print("=" * 60)
    
    pl.seed_everything(cfg["seed"])
    torch.set_float32_matmul_precision("high")

    use_val = bool(cfg.get("use_val", True))

    if use_val and cfg.get("val_data_path") in (None, "", "null"):
        cfg["val_data_path"] = cfg["train_data_path"]
        print(
            f"[Config] val_data_path is empty, fallback to train_data_path={cfg['val_data_path']}",
            flush=True,
        )
    
    # Create datasets (same as VBD)
    print("\nLoading datasets...")
    print(f"[Config] Effective batch_size={cfg['batch_size']}, num_workers={cfg['num_workers']}", flush=True)
    train_dataset = WaymaxDataset(
        data_dir=cfg["train_data_path"],
    )

    val_dataset = None
    if use_val:
        val_dataset = WaymaxDataset(
            data_dir=cfg["val_data_path"],
        )

    print(f"  Train samples: {len(train_dataset)}")
    if use_val:
        print(f"  Val samples: {len(val_dataset)}")
    else:
        print("  Validation disabled (use_val=false)")

    train_schema = _inspect_dataset_schema(train_dataset, name="train")
    val_schema = None
    if use_val and val_dataset is not None:
        val_schema = _inspect_dataset_schema(val_dataset, name="val")
        if "sdc_id" in train_schema.get("missing_preferred", []) and "sdc_id" not in val_schema.get("missing_preferred", []):
            print(
                "[Schema] WARNING: train split has no sdc_id but val split has sdc_id. "
                "Ego-frame definition is inconsistent between train/val.",
                flush=True,
            )
    
    # Create dataloaders (same as VBD)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["num_workers"],
        shuffle=True
    )
    
    val_loader = None
    if use_val:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["batch_size"],
            pin_memory=True,
            num_workers=cfg["num_workers"],
            shuffle=False
        )
    
    # Setup output directory (same as VBD)
    output_root = cfg.get("log_dir", "output")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"{cfg['model_name']}_{timestamp}"
    output_path = f"{output_root}/{model_name}"
    print(f"\nOutput directory: {output_path}")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Save config (same as VBD)
    with open(f"{output_path}/config.yaml", "w") as file:
        yaml.dump(cfg, file)
    
    # Build model
    num_gpus = torch.cuda.device_count()
    print(f"\nTotal GPUs: {num_gpus}")
    
    model = MapGlowWrapper(cfg=cfg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint if specified (same logic as VBD)
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"\nLoading weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    
    # Setup logger
    logger = build_logger(cfg, output_path, model_name)
    
    if use_val:
        ckpt_cb = ModelCheckpoint(
            dirpath=output_path,
            save_top_k=cfg.get("save_top_k", 20),
            save_weights_only=False,
            monitor="val/loss",
            mode="min",
            filename="epoch={epoch:02d}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )
    else:
        # No validation metric: checkpoint each epoch + save last.
        ckpt_cb = ModelCheckpoint(
            dirpath=output_path,
            save_top_k=-1,
            save_last=True,
            save_weights_only=False,
            filename="epoch={epoch:02d}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        )

    callbacks = [
        ckpt_cb,
        LearningRateMonitor(logging_interval="step"),
    ]

    sample_every_n_steps = int(cfg.get("sample_every_n_steps", 0) or 0)
    if sample_every_n_steps > 0:
        if (not use_val) and bool(cfg.get("sample_use_waymax_vis", True)):
            print(
                "[Config] use_val=false and sample_use_waymax_vis=true: sampling source is train_loader, "
                "please ensure sample_raw_data_path matches train split; otherwise raw scenario lookup may fail.",
                flush=True,
            )
        callbacks.append(
            MapGlowSamplingCallback(
                val_loader=val_loader if val_loader is not None else train_loader,
                output_path=output_path,
                every_n_steps=sample_every_n_steps,
                max_agents=cfg.get("sample_vis_max_agents", 8),
                n_samples=cfg.get("sample_n_samples", 1),
                temperature=cfg.get("sample_temperature", None),
                pos_scale=model.pos_scale.item(),
                use_waymax_vis=cfg.get("sample_use_waymax_vis", True),
                raw_data_path=cfg.get("sample_raw_data_path", None),
                max_raw_scan=cfg.get("sample_raw_max_scan", 50000),
                max_num_objects=cfg.get("agents_len", 128),
                rotate_scene=cfg.get("sample_rotate_scene", True),
            )
        )
        print(f"Sampling callback enabled: every {sample_every_n_steps} steps")

    # Setup trainer (same as VBD)
    trainer = pl.Trainer(
        num_nodes=cfg.get("num_nodes", 1),
        max_epochs=cfg["epochs"],
        devices=cfg.get("num_gpus", -1),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy=DDPStrategy() if num_gpus > 1 else "auto",
        enable_progress_bar=True,
        logger=logger,
        enable_model_summary=True,
        detect_anomaly=False,
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=0,
        precision=cfg.get("precision", "32-true"),
        log_every_n_steps=cfg.get("log_every_n_steps", 100),
        callbacks=callbacks,
    )
    
    print("\nStarting training...")
    print("=" * 60)
    
    trainer.fit(
        model,
        train_loader,
        val_loader if use_val else None,
        ckpt_path=cfg.get("init_from")
    )
    
    print("\nTraining completed!")


def build_parser():
    """Build argument parser - same structure as VBD."""
    parser = argparse.ArgumentParser(description="MapGlow Training with VBD Pipeline")
    parser.add_argument("-cfg", "--cfg", type=str, default="config/MapGlow.yaml",
                        help="Path to config file")
    
    # Data paths
    parser.add_argument("--train_data_path", type=str, default=None,
                        help="Training data directory")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Validation data directory")
    
    # Model name and logging
    parser.add_argument("-name", "--model_name", type=str, default=None,
                        help="Model name for logging")
    parser.add_argument("-log", "--log_dir", type=str, default=None,
                        help="Log directory")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb team/entity name (recommended, avoids personal-entity permission issues)")
    parser.add_argument("--wandb_mode", type=str, default=None,
                        help="Wandb mode: online/offline/disabled")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Custom wandb run name")
    
    # Training parameters
    parser.add_argument("-bs", "--batch_size", type=int, default=None,
                        help="Batch size")
    parser.add_argument("-lr", "--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--sample_every_n_steps", type=int, default=None,
                        help="Sampling/visualization interval in training steps (0 to disable)")
    parser.add_argument("--sample_vis_max_agents", type=int, default=None,
                        help="Maximum number of agents to draw in sample visualization")
    parser.add_argument("--sample_n_samples", type=int, default=None,
                        help="Number of stochastic samples to draw each visualization step")
    parser.add_argument("--sample_temperature", type=float, default=None,
                        help="Sampling temperature for visualization")
    parser.add_argument("--sample_use_waymax_vis", type=int, default=None,
                        help="Use Waymax visualization for sampling images (1=true, 0=false)")
    parser.add_argument("--sample_raw_data_path", type=str, default=None,
                        help="Optional raw Waymo TFRecord directory for plotting.py fallback (matched by scenario_id)")
    parser.add_argument("--sample_raw_max_scan", type=int, default=None,
                        help="Maximum number of raw scenarios to scan when finding scenario_id in TFRecords")
    parser.add_argument("--sample_rotate_scene", type=int, default=None,
                        help="Rotate visualization scenario across val dataset (1=true, 0=false)")
    parser.add_argument("--use_val", type=int, default=None,
                        help="Whether to run validation each epoch (1=true, 0=false)")
    
    # MapGlow specific parameters
    parser.add_argument("--n_flow", type=int, default=None,
                        help="Number of flows per block")
    parser.add_argument("--n_block", type=int, default=None,
                        help="Number of blocks")
    parser.add_argument("--condition_dim", type=int, default=None,
                        help="Condition dimension")
    parser.add_argument("--affine", action="store_true",
                        help="Use affine coupling")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature")
    
    # Hardware (same as VBD)
    parser.add_argument("-nN", "--num_nodes", type=int, default=1,
                        help="Number of nodes")
    parser.add_argument("-nG", "--num_gpus", type=int, default=-1,
                        help="Number of GPUs (-1 for all)")
    
    # Resume training (same as VBD)
    parser.add_argument("-init", "--init_from", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("-ckpt", "--ckpt_path", type=str, default=None,
                        help="Load model weights from checkpoint")
    
    return parser


def load_cfg(args):
    """Load config and override with command line arguments."""
    cfg = load_config(args.cfg)
    
    # Override config from args
    for key, value in vars(args).items():
        if key == "cfg":
            continue
        elif value is not None:
            cfg[key] = value
    
    return cfg


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_cfg(args)
    
    train(cfg)
