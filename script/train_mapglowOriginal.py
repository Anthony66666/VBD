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
import numpy as np

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
from vbd.model.MapGlowWrapperOriginal import MapGlowWrapper
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from matplotlib import pyplot as plt


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


def _plot_mapglow_sample_simple(batch, pred_traj, save_path, max_agents=8):
    """
    Simple global-frame visualization:
    map(gray) + history(blue) + gt(green) + pred(red)
    """
    b = 0
    agents_history = batch["agents_history"][b].detach().cpu().numpy()  # [A, T_hist, 8]
    agents_future = batch["agents_future"][b].detach().cpu().numpy()    # [A, T_fut, 5]
    agents_interested = batch["agents_interested"][b].detach().cpu().numpy()
    polylines = batch["polylines"][b].detach().cpu().numpy()
    polylines_valid = batch["polylines_valid"][b].detach().cpu().numpy()
    pred = pred_traj[b].detach().cpu().numpy()  # [A, T, 3]

    A_pred = pred.shape[0]
    agents_history = agents_history[:A_pred]
    agents_future = agents_future[:A_pred]
    agents_interested = agents_interested[:A_pred]

    if "agents_future_valid" in batch:
        future_valid = batch["agents_future_valid"][b].detach().cpu().numpy().astype(bool)[:A_pred]
    else:
        future_valid = (np.abs(agents_future[..., :2]).sum(axis=-1) > 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(polylines.shape[0]):
        if polylines_valid[i] == 0:
            continue
        line = polylines[i]
        line = line[np.abs(line[:, :2]).sum(axis=-1) > 0]
        if line.shape[0] > 1:
            ax.plot(line[:, 0], line[:, 1], ".", markersize=1, alpha=0.45, color="#999999")

    draw_idx = np.where(agents_interested > 0)[0]
    if draw_idx.size == 0:
        draw_idx = np.arange(min(max_agents, pred.shape[0]))
    else:
        draw_idx = draw_idx[:max_agents]

    for i in draw_idx:
        hist = agents_history[i]
        hist = hist[np.abs(hist[:, :2]).sum(axis=-1) > 0]
        if hist.shape[0] > 1:
            ax.plot(hist[:, 0], hist[:, 1], color="#1f77b4", alpha=0.9, linewidth=1.2)

        gt = agents_future[i]
        gt_mask = future_valid[i]
        gt = gt[1:1 + pred.shape[1]]
        gt_mask = gt_mask[1:1 + pred.shape[1]]
        gt_plot = gt.copy()
        gt_plot[~gt_mask] = np.nan
        if np.isfinite(gt_plot[:, :2]).any():
            ax.plot(gt_plot[:, 0], gt_plot[:, 1], color="#2ca02c", alpha=0.95, linewidth=1.4)

        pr = pred[i]
        if pr.shape[0] > 1:
            ax.plot(pr[:, 0], pr[:, 1], color="#d62728", alpha=0.95, linewidth=1.6)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("MapGlowOriginal sample: history(blue) gt(green) pred(red)")
    ax.axis("off")
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
                print(f"[Sampling] Wandb image logging failed: {e}", flush=True)
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
        rotate_scene=True,
    ):
        super().__init__()
        self.val_loader = val_loader
        self.output_path = output_path
        self.every_n_steps = int(every_n_steps)
        self.max_agents = int(max_agents)
        self.n_samples = int(n_samples)
        self.temperature = temperature
        self.rotate_scene = bool(rotate_scene)
        self._last_sample_step = 0
        self._sample_index = -1

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
                return default_collate([sample])
            except Exception as e:
                print(f"[Sampling] Failed to load sample index={sample_index}, fallback to dataloader: {e}", flush=True)

        try:
            return next(iter(self.val_loader))
        except Exception:
            if fallback_batch is not None:
                return fallback_batch
            raise RuntimeError("Sampling callback cannot get batch from val/train loader.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps <= 0:
            return
        step = int(trainer.global_step)
        if step <= 0:
            return
        if (step - self._last_sample_step) < self.every_n_steps:
            return
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        sample_index = self._next_sample_index()
        sample_batch = self._get_batch(sample_index=sample_index, fallback_batch=batch)
        sample_batch = _batch_to_device(sample_batch, pl_module.device)

        pl_module.eval()
        try:
            with torch.no_grad():
                pred = pl_module.sample(
                    sample_batch,
                    n_samples=max(self.n_samples, 1),
                    temperature=self.temperature,
                    return_global=True,
                )  # [B, n, A, T, 3]
                pred_global = pred[:, 0]

            save_path = os.path.join(self.output_path, f"sample_step_{step:08d}.png")
            _plot_mapglow_sample_simple(
                sample_batch,
                pred_global,
                save_path=save_path,
                max_agents=self.max_agents,
            )
            if sample_index is not None:
                print(f"[Sampling] Visualization sample index={sample_index}", flush=True)
            print(f"[Sampling] Saved sample visualization: {save_path}", flush=True)
            _log_wandb_image(trainer, save_path, step)
            self._last_sample_step = step
        except Exception as e:
            print(f"[Sampling] Failed at step {step}: {e}", flush=True)
        finally:
            pl_module.train()


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
    }
    try:
        sample = dataset[0]
        keys = set(sample.keys()) if isinstance(sample, dict) else set()
        out["missing_required"] = [k for k in required if k not in keys]
        out["missing_preferred"] = [k for k in preferred_masks if k not in keys]
        if out["missing_required"]:
            out["ok"] = False
        print(f"[Schema:{name}] keys={sorted(list(keys))}", flush=True)
        if out["missing_required"]:
            print(f"[Schema:{name}] Missing REQUIRED keys: {out['missing_required']}", flush=True)
        if out["missing_preferred"]:
            print(f"[Schema:{name}] Missing preferred mask/ego keys: {out['missing_preferred']}", flush=True)
            if "sdc_id" in out["missing_preferred"]:
                print(
                    f"[Schema:{name}] WARNING: missing sdc_id -> wrapper falls back to agent index 0 as ego frame. "
                    "This can significantly hurt training quality.",
                    flush=True,
                )

        # Lightweight value/shape sanity for mask-like fields
        mask_keys = [
            "agents_history_valid",
            "agents_future_valid",
            "polylines_valid",
            "polylines_point_valid",
            "traffic_light_valid",
            "agents_interested",
        ]
        for mk in mask_keys:
            if mk not in keys:
                continue
            v = sample[mk]
            if not isinstance(v, torch.Tensor):
                continue
            uniq = torch.unique(v)
            uniq_show = uniq[:8].cpu().tolist()
            print(
                f"[Schema:{name}] {mk}: shape={tuple(v.shape)}, dtype={v.dtype}, uniq(head)={uniq_show}",
                flush=True,
            )
    except Exception as e:
        out["ok"] = False
        print(f"[Schema:{name}] Failed to inspect sample: {e}", flush=True)
    return out


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
    
    callbacks = [
        ModelCheckpoint(
            dirpath=output_path,
            save_top_k=cfg.get("save_top_k", 20) if use_val else -1,
            save_last=not use_val,
            save_weights_only=False,
            monitor="val/loss" if use_val else None,
            mode="min" if use_val else "max",
            filename="epoch={epoch:02d}",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_on_train_epoch_end=(not use_val),
        ),
        LearningRateMonitor(logging_interval="step")
    ]

    sample_every_n_steps = int(cfg.get("sample_every_n_steps", 0) or 0)
    if sample_every_n_steps > 0:
        callbacks.append(
            MapGlowSamplingCallback(
                val_loader=val_loader if val_loader is not None else train_loader,
                output_path=output_path,
                every_n_steps=sample_every_n_steps,
                max_agents=cfg.get("sample_vis_max_agents", 8),
                n_samples=cfg.get("sample_n_samples", 1),
                temperature=cfg.get("sample_temperature", None),
                rotate_scene=cfg.get("sample_rotate_scene", True),
            )
        )
        print(f"Sampling callback enabled: every {sample_every_n_steps} steps", flush=True)

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
        callbacks=callbacks
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
    parser.add_argument("-cfg", "--cfg", type=str, default="config/MapGlowOriginal.yaml",
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
    parser.add_argument("-bs", "--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("-lr", "--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=None,
                        help="Number of epochs")
    parser.add_argument("--use_val", type=int, default=None,
                        help="Whether to run validation each epoch (1=true, 0=false)")
    parser.add_argument("--sample_every_n_steps", type=int, default=None,
                        help="Sampling/visualization interval in training steps (0 to disable)")
    parser.add_argument("--sample_vis_max_agents", type=int, default=None,
                        help="Maximum number of agents to draw in sample visualization")
    parser.add_argument("--sample_n_samples", type=int, default=None,
                        help="Number of stochastic samples to draw each visualization step")
    parser.add_argument("--sample_temperature", type=float, default=None,
                        help="Sampling temperature for visualization")
    parser.add_argument("--sample_rotate_scene", type=int, default=None,
                        help="Rotate visualization scenario across dataset (1=true, 0=false)")
    
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
    parser.add_argument("--valid_eps", type=float, default=None,
                        help="Epsilon for valid/non-valid geometric mask threshold")
    parser.add_argument("--use_agent_interaction", type=int, default=None,
                        help="Enable agent interaction module in context encoder (1=true, 0=false)")
    parser.add_argument("--use_lane_aware", type=int, default=None,
                        help="Enable lane-aware module in context encoder (1=true, 0=false)")
    
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
