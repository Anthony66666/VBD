"""
Training script for Flow Matching model using VBD's data pipeline.
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
from vbd.model.FlowMatching import FlowMatching
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy


def _batch_to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def _plot_trajectories(batch, pred_trajs, save_path, max_agents=8):
    import numpy as np
    from vbd.waymax_visualization import utils as viz_utils
    try:
        from vbd.waymax_visualization import viz as waymax_viz
        from waymax import datatypes
        use_waymax_viz = True
    except Exception:
        use_waymax_viz = False

    b = 0
    agents_history = batch["agents_history"][b].detach().cpu().numpy()
    agents_future = batch["agents_future"][b].detach().cpu().numpy()
    agents_interested = batch["agents_interested"][b].detach().cpu().numpy()
    polylines = batch["polylines"][b].detach().cpu().numpy()
    polylines_valid = batch["polylines_valid"][b].detach().cpu().numpy()

    pred = pred_trajs[b].detach().cpu().numpy()

    vis_config = viz_utils.VizConfig()
    fig, ax = viz_utils.init_fig_ax(vis_config)

    # Plot map polylines
    for i in range(polylines.shape[0]):
        if polylines_valid[i] == 0:
            continue
        line = polylines[i]
        line = line[line[:, 0] != 0]
        if line.shape[0] > 1:
            ax.plot(line[:, 0], line[:, 1], ".", markersize=1, alpha=0.5, color="#999999")

    # Plot agent histories and futures
    num_agents = min(max_agents, agents_history.shape[0])
    if use_waymax_viz:
        try:
            traj_full = np.concatenate([agents_history[..., :5], agents_future[..., :5]], axis=1)
            x = traj_full[..., 0]
            y = traj_full[..., 1]
            yaw = traj_full[..., 2]
            vel_x = traj_full[..., 3]
            vel_y = traj_full[..., 4]

            last_dims = agents_history[:, -1, 5:7]
            length = np.repeat(last_dims[:, 0:1], traj_full.shape[1], axis=1)
            width = np.repeat(last_dims[:, 1:2], traj_full.shape[1], axis=1)

            valid = (x != 0)
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

            is_controlled = agents_interested[: num_agents] > 0
            time_idx = agents_history.shape[1] - 1
            waymax_viz.plot_trajectory(
                ax,
                traj,
                is_controlled=is_controlled,
                time_idx=time_idx,
                indices=np.arange(traj.num_objects),
            )
        except Exception:
            use_waymax_viz = False

    if not use_waymax_viz:
        for i in range(num_agents):
            if agents_interested[i] <= 0:
                continue
            hist = agents_history[i]
            hist = hist[hist[:, 0] != 0]
            if hist.shape[0] > 1:
                ax.plot(hist[:, 0], hist[:, 1], color="#1f77b4", alpha=0.8, linewidth=1)

            gt = agents_future[i]
            gt = gt[gt[:, 0] != 0]
            if gt.shape[0] > 1:
                ax.plot(gt[:, 0], gt[:, 1], color="#2ca02c", alpha=0.9, linewidth=1.5)

    # Plot predictions (always as red curves)
    for i in range(num_agents):
        if agents_interested[i] <= 0:
            continue
        pr = pred[i]
        if pr.shape[0] > 1:
            ax.plot(pr[:, 0], pr[:, 1], color="#d62728", alpha=0.9, linewidth=1.5)

    ax.set_title("FlowMatching: history(blue) gt(green) pred(red)")
    ax.axis("off")

    # Center view on first valid agent if possible
    center_xy = None
    for i in range(min(max_agents, agents_history.shape[0])):
        if agents_interested[i] <= 0:
            continue
        if agents_history[i, -1, 0] != 0:
            center_xy = agents_history[i, -1, :2]
            break
    if center_xy is not None:
        viz_utils.center_at_xy(ax, center_xy, vis_config)

    img = viz_utils.img_from_fig(fig)
    viz_utils.save_img_as_png(img, save_path)


class FlowMatchVisualizationCallback(pl.Callback):
    def __init__(self, val_loader, output_path, every_n_epochs=1, max_agents=8, num_steps=50):
        super().__init__()
        self.val_loader = val_loader
        self.output_path = output_path
        self.every_n_epochs = every_n_epochs
        self.max_agents = max_agents
        self.num_steps = num_steps
        self._cached_batch = None

    def _get_batch(self):
        if self._cached_batch is None:
            self._cached_batch = next(iter(self.val_loader))
        return self._cached_batch

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if self.every_n_epochs <= 0 or (epoch % self.every_n_epochs) != 0:
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        batch = self._get_batch()
        batch = _batch_to_device(batch, pl_module.device)

        pl_module.eval()
        try:
            with torch.no_grad():
                pred = pl_module.sample_actions(
                    batch,
                    num_steps=max(int(self.num_steps), 2),
                )

            save_path = os.path.join(self.output_path, f"vis_epoch_{epoch:03d}.png")
            print(f"[Visualization] Saving sample to: {save_path}", flush=True)
            _plot_trajectories(batch, pred["trajs"], save_path, max_agents=self.max_agents)
        except Exception as e:
            import traceback
            print(f"[Visualization] Failed to save sample: {e}", flush=True)
            traceback.print_exc()
        pl_module.train()


def load_config(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def train(cfg):
    print("=" * 60)
    print("Starting Flow Matching Training with VBD Pipeline")
    print("=" * 60)

    pl.seed_everything(cfg["seed"])
    torch.set_float32_matmul_precision("high")

    # Validate data paths
    if cfg["train_data_path"] is None or not os.path.exists(cfg["train_data_path"]):
        raise ValueError(
            f"train_data_path not set or does not exist: {cfg.get('train_data_path')}\n"
            "Please set it in config file or via command line: --train_data_path /path/to/train/data"
        )
    if cfg["val_data_path"] is None or not os.path.exists(cfg["val_data_path"]):
        raise ValueError(
            f"val_data_path not set or does not exist: {cfg.get('val_data_path')}\n"
            "Please set it in config file or via command line: --val_data_path /path/to/val/data"
        )

    print("\nLoading datasets...")
    print(f"  Train data: {cfg['train_data_path']}")
    print(f"  Val data: {cfg['val_data_path']}")
    max_samples = cfg.get("max_samples", None)
    if max_samples:
        print(f"LIMITED MODE: Using max {max_samples} samples for testing")
    
    print("  Scanning training files...", flush=True)
    import time
    start_time = time.time()
    train_dataset = WaymaxDataset(
        data_dir=cfg["train_data_path"],
        anchor_path=cfg["anchor_path"],
        max_samples=max_samples,
    )
    train_time = time.time() - start_time
    print(f"  ✓ Train samples: {len(train_dataset)} (scanned in {train_time:.2f}s)")

    print("  Scanning validation files...", flush=True)
    start_time = time.time()
    val_dataset = WaymaxDataset(
        data_dir=cfg["val_data_path"],
        anchor_path=cfg["anchor_path"],
        max_samples=max_samples // 5 if max_samples else None,
    )
    val_time = time.time() - start_time
    print(f"  ✓ Val samples: {len(val_dataset)} (scanned in {val_time:.2f}s)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["num_workers"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["num_workers"],
        shuffle=False,
    )

    output_root = cfg.get("log_dir", "output")
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"{cfg['model_name']}_{timestamp}"
    output_path = f"{output_root}/{model_name}"
    print(f"\nOutput directory: {output_path}")

    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/config.yaml", "w") as file:
        yaml.dump(cfg, file)

    num_gpus = torch.cuda.device_count()
    print(f"\nTotal GPUs: {num_gpus}")

    model = FlowMatching(cfg=cfg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"\nLoading weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)

    if not cfg.get("train_encoder", True):
        encoder_path = cfg.get("encoder_ckpt", None)
        if encoder_path is not None and os.path.exists(encoder_path):
            print(f"\nLoading encoder weights from: {encoder_path}")
            state_dict = torch.load(encoder_path, map_location=torch.device("cpu"))
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict, strict=False)
        else:
            cfg["train_encoder"] = True
            print("Warning: Encoder path not provided, training encoder from scratch")

    use_wandb = cfg.get("use_wandb", False)
    if use_wandb:
        logger = WandbLogger(
            name=model_name,
            project=cfg.get("project", "FlowMatching"),
            entity=cfg.get("username"),
            log_model=False,
            dir=output_path,
        )
    else:
        logger = CSVLogger(output_path, name="FlowMatching", version=1, flush_logs_every_n_steps=100)

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
        precision=cfg.get("precision", "bf16-mixed"),
        log_every_n_steps=cfg.get("log_every_n_steps", 100),
        callbacks=[
            ModelCheckpoint(
                dirpath=output_path,
                save_top_k=cfg.get("save_top_k", 20),
                save_weights_only=False,
                monitor="val/loss",
                mode="min",
                filename="epoch={epoch:02d}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval="step"),
            FlowMatchVisualizationCallback(
                val_loader=val_loader,
                output_path=output_path,
                every_n_epochs=cfg.get("vis_every_n_epochs", 1),
                max_agents=cfg.get("vis_max_agents", 8),
                num_steps=cfg.get("vis_flow_steps", cfg.get("flow_steps", 50)),
            ),
        ],
    )

    print("\nStarting training...")
    print("=" * 60)

    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=cfg.get("init_from"),
    )

    print("\nTraining completed!")


def build_parser():
    parser = argparse.ArgumentParser(description="Flow Matching Training with VBD Pipeline")
    parser.add_argument("-cfg", "--cfg", type=str, default="config/FlowMatching.yaml")

    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size for testing (e.g., 100)")

    parser.add_argument("-name", "--model_name", type=str, default=None)
    parser.add_argument("-log", "--log_dir", type=str, default=None)

    parser.add_argument("-bs", "--batch_size", type=int, default=None)
    parser.add_argument("-lr", "--lr", type=float, default=None)
    parser.add_argument("-e", "--epochs", type=int, default=None)

    parser.add_argument("--flow_steps", type=int, default=None)
    parser.add_argument("--flow_t_min", type=float, default=None)
    parser.add_argument("--flow_t_max", type=float, default=None)

    parser.add_argument("-eV", "--encoder_version", type=str, default=None)
    parser.add_argument("-encoder", "--encoder_ckpt", type=str, default=None)
    parser.add_argument("--train_encoder", type=bool, default=None)

    parser.add_argument("-nN", "--num_nodes", type=int, default=1)
    parser.add_argument("-nG", "--num_gpus", type=int, default=-1)

    parser.add_argument("-init", "--init_from", type=str, default=None)
    parser.add_argument("-ckpt", "--ckpt_path", type=str, default=None)

    return parser


def load_cfg(args):
    cfg = load_config(args.cfg)
    for key, value in vars(args).items():
        if key == "cfg":
            continue
        if value is not None:
            cfg[key] = value
    return cfg


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_cfg(args)

    train(cfg)
