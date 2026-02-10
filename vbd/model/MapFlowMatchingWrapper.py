"""
MapFlowMatching model wrapper for PyTorch Lightning training.
Reuses MapGlowWrapper data preprocessing pipeline but replaces NF with Flow Matching.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import lightning.pytorch as pl

from vbd.model.MapGlowWrapper import MapGlowWrapper

# Make project root importable (for MapGLow.py).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from MapGLow import ContextEncoder, ensure_not_all_pad  # noqa: E402


class TemporalFlowMatcher(nn.Module):
    """
    Velocity field network v_theta(x_t, t, context) for trajectory flow matching.
    """

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)

        self.in_proj = nn.Linear(self.in_channels, self.hidden_dim)
        self.t_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.self_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=dropout,
            )
            for _ in range(self.num_layers)
        ])
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                batch_first=True,
                dropout=dropout,
            )
            for _ in range(self.num_layers)
        ])
        self.ln1 = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.ln3 = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            )
            for _ in range(self.num_layers)
        ])

        self.out_proj = nn.Linear(self.hidden_dim, self.in_channels)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: Dict[str, torch.Tensor],
        timestep_mask: Optional[torch.Tensor] = None,
        vehicle_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: [B, C, T, V]
            t: [B] float in [0,1]
            context: dict from ContextEncoder with:
                - kv: [B*V, S, F]
                - kv_mask: [B*V, S] bool (True=padding)
            timestep_mask: [B, T, V] bool (True=valid)
            vehicle_mask: [B, V] bool (True=valid)
        Returns:
            v_pred: [B, C, T, V]
        """
        B, C, T, V = x_t.shape
        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {C}")

        # [B,C,T,V] -> [B*V,T,C]
        x = x_t.permute(0, 3, 2, 1).reshape(B * V, T, C)
        h = self.in_proj(x)

        # Time embedding: one scalar per scene, broadcast to agents/timesteps.
        t = t.view(B, 1).to(dtype=h.dtype)
        t_emb = self.t_mlp(t)  # [B, F]
        t_emb = t_emb.unsqueeze(1).expand(B, V, self.hidden_dim).reshape(B * V, self.hidden_dim)
        h = h + t_emb.unsqueeze(1)

        kv = context["kv"]
        kv_mask = context.get("kv_mask", None)
        if kv is None:
            raise RuntimeError("Context missing 'kv' for flow matching.")

        # Build temporal padding mask.
        time_pad = None
        if timestep_mask is not None:
            time_valid = timestep_mask.permute(0, 2, 1).reshape(B * V, T).bool()
            time_pad = ~time_valid
        if vehicle_mask is not None:
            invalid_agent = (~vehicle_mask.bool()).reshape(B * V, 1)
            if time_pad is None:
                time_pad = invalid_agent.expand(B * V, T)
            else:
                time_pad = time_pad | invalid_agent.expand(B * V, T)

        time_pad_clean, all_pad_time = ensure_not_all_pad(time_pad) if time_pad is not None else (None, None)
        kv_mask_clean, all_pad_kv = ensure_not_all_pad(kv_mask) if kv_mask is not None else (None, None)

        if all_pad_time is not None and all_pad_time.any():
            h = h.clone()
            h[all_pad_time] = 0.0
        if all_pad_kv is not None and all_pad_kv.any():
            kv = kv.clone()
            kv[all_pad_kv] = 0.0

        for i in range(self.num_layers):
            h_sa, _ = self.self_attn[i](h, h, h, key_padding_mask=time_pad_clean)
            if all_pad_time is not None and all_pad_time.any():
                h_sa = h_sa.clone()
                h_sa[all_pad_time] = 0.0
            h = self.ln1[i](h + h_sa)

            h_ca, _ = self.cross_attn[i](h, kv, kv, key_padding_mask=kv_mask_clean)
            if all_pad_time is not None and all_pad_time.any():
                h_ca = h_ca.clone()
                h_ca[all_pad_time] = 0.0
            h = self.ln2[i](h + h_ca)

            h_ffn = self.ffn[i](h)
            if all_pad_time is not None and all_pad_time.any():
                h_ffn = h_ffn.clone()
                h_ffn[all_pad_time] = 0.0
            h = self.ln3[i](h + h_ffn)

        v = self.out_proj(h)  # [B*V,T,C]
        v = v.reshape(B, V, T, C).permute(0, 3, 2, 1).contiguous()  # [B,C,T,V]

        # Force invalid slots to zero velocity.
        if vehicle_mask is not None:
            agent_valid = vehicle_mask.float().unsqueeze(1).unsqueeze(2)  # [B,1,1,V]
            v = v * agent_valid
        if timestep_mask is not None:
            step_valid = timestep_mask.float().unsqueeze(1)  # [B,1,T,V]
            v = v * step_valid

        return v


class MapFlowMatchingWrapper(MapGlowWrapper):
    """
    Flow Matching model using the same data pipeline as MapGlowWrapper.
    """

    def __init__(self, cfg: dict):
        # Reuse all preprocessing/transform utilities from MapGlowWrapper.
        super().__init__(cfg)
        self.cfg = cfg

        # Remove NF model from trainable graph to avoid double-parameter training.
        if hasattr(self, "glow"):
            del self.glow

        self._flow_steps = int(cfg.get("flow_steps", 50))
        self._sample_steps = int(cfg.get("sample_steps", self._flow_steps))
        self._fm_hidden_dim = int(cfg.get("fm_hidden_dim", 256))
        self._fm_heads = int(cfg.get("fm_heads", 8))
        self._fm_layers = int(cfg.get("fm_layers", 4))
        self._fm_dropout = float(cfg.get("fm_dropout", 0.1))
        self._fm_eval_samples = int(cfg.get("fm_eval_samples", 1))

        map_input_dim = 5 if self._lane_type_encoding == "embedding" else (4 + self._num_lane_types)
        traffic_light_input_dim = 3

        self.context_encoder = ContextEncoder(
            filter_size=self._fm_hidden_dim,
            num_heads=self._fm_heads,
            history_input_dim=self._in_channel,
            topk_lanes=self._topk_lanes,
            lane_selection_mode=self._lane_selection_mode,
            hybrid_global_lanes=self._hybrid_global_lanes,
            max_points=30,
            map_input_dim=map_input_dim,
            traffic_light_input_dim=traffic_light_input_dim,
            num_lane_types=self._num_lane_types,
            num_traffic_light_states=self._num_traffic_light_states,
            lane_type_encoding=self._lane_type_encoding,
            lane_type_embed_dim=self._lane_type_embed_dim,
            lane_tl_embed_dim=self._lane_tl_embed_dim,
            tl_state_embed_dim=self._tl_state_embed_dim,
        )

        self.flow_net = TemporalFlowMatcher(
            in_channels=self._in_channel,
            hidden_dim=self._fm_hidden_dim,
            num_heads=self._fm_heads,
            num_layers=self._fm_layers,
            dropout=self._fm_dropout,
        )

    def _build_context(self, inputs: dict) -> dict:
        return self.context_encoder(
            condition=None,
            agent_types=inputs["agent_types"],
            agent_shape=inputs["agent_shape"],
            map_data=inputs["map_data"],
            map_mask=inputs["map_mask"],
            traffic_light_data=inputs["traffic_light_data"],
            traffic_light_mask=inputs["traffic_light_mask"],
            history_data=inputs["history_data"],
            history_timestep_mask=inputs["history_timestep_mask"],
            target_vehicle_mask=inputs["target_vehicle_mask"],
            history_vehicle_mask=inputs["history_vehicle_mask"],
            B_hint=inputs["input"].shape[0],
            V_hint=inputs["input"].shape[-1],
        )

    def forward(self, batch):
        """
        Flow Matching training forward:
            x_t = (1-t) * x_1 + t * x_0
            target velocity = x_0 - x_1
        where x_0 is normalized target trajectory and x_1 ~ N(0, I).
        """
        inputs = self.prepare_glow_inputs(batch)
        x0 = inputs["input"]  # [B,C,T,V], normalized data
        B = x0.shape[0]

        x1 = torch.randn_like(x0)
        t = torch.rand(B, device=x0.device)
        t4 = t.view(B, 1, 1, 1)
        x_t = (1.0 - t4) * x1 + t4 * x0
        v_target = x0 - x1

        context = self._build_context(inputs)
        v_pred = self.flow_net(
            x_t,
            t=t,
            context=context,
            timestep_mask=inputs["timestep_mask"],
            vehicle_mask=inputs["target_vehicle_mask"],
        )

        return {
            "inputs": inputs,
            "x_t": x_t,
            "t": t,
            "v_pred": v_pred,
            "v_target": v_target,
            "context": context,
        }

    def compute_fm_loss(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        vehicle_mask: torch.Tensor,
        timestep_mask: torch.Tensor,
    ) -> torch.Tensor:
        # [B,1,T,V] valid mask
        valid = vehicle_mask.float().unsqueeze(1).unsqueeze(2) * timestep_mask.float().unsqueeze(1)
        denom = valid.sum().clamp(min=1.0)
        loss = ((v_pred - v_target) ** 2 * valid).sum() / denom
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(1e6, device=v_pred.device, requires_grad=True)
        return loss

    def forward_and_get_loss(self, batch, prefix="", debug=False):
        agents_future = batch["agents_future"][:, :self._agents_len]
        if "agents_future_valid" in batch:
            agents_future_valid = batch["agents_future_valid"][:, :self._agents_len].bool()
        else:
            agents_future_valid = agents_future[..., :2].abs().sum(-1) > self._valid_eps
        agents_interested = batch["agents_interested"][:, :self._agents_len]

        output_dict = self.forward(batch)
        inputs = output_dict["inputs"]

        fm_loss = self.compute_fm_loss(
            output_dict["v_pred"],
            output_dict["v_target"],
            inputs["target_vehicle_mask"],
            inputs["timestep_mask"],
        )

        log_dict = {
            f"{prefix}fm_loss": fm_loss.item(),
            f"{prefix}loss": fm_loss.item(),
            f"{prefix}valid_vehicle_mean": inputs["target_vehicle_mask"].float().sum(dim=1).mean().item(),
            f"{prefix}valid_timestep_ratio": inputs["timestep_mask"].float().mean().item(),
            f"{prefix}input_abs_max": inputs["input"].abs().max().item(),
        }

        debug_outputs = {}
        if (not self.training) or debug:
            with torch.no_grad():
                sampled_trajs = self.sample_trajectories(
                    output_dict,
                    temperature=self._temp,
                    num_steps=self._sample_steps,
                )  # [B,A,T,5] local
                gt_local = inputs["agents_future_local"][:, :, 1:, :3]
                pred_local = sampled_trajs[..., :3]
                future_valid = agents_future_valid[:, :, 1:]

                state_loss, yaw_loss = self.trajectory_loss(
                    pred_local, gt_local, future_valid, agents_interested
                )
                ade, fde = self.calculate_metrics(
                    pred_local, gt_local, future_valid, agents_interested
                )
                log_dict.update({
                    f"{prefix}state_loss": state_loss.item(),
                    f"{prefix}yaw_loss": yaw_loss.item(),
                    f"{prefix}ADE": ade,
                    f"{prefix}FDE": fde,
                })
                if debug:
                    debug_outputs["sampled_trajs"] = sampled_trajs
                    debug_outputs["gt_local"] = gt_local

        if debug:
            return fm_loss, log_dict, debug_outputs
        return fm_loss, log_dict

    @torch.no_grad()
    def sample_trajectories(
        self,
        output_dict: dict,
        temperature: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        if temperature is None:
            temperature = self._temp
        if num_steps is None:
            num_steps = self._sample_steps
        num_steps = max(1, int(num_steps))

        inputs = output_dict["inputs"]
        context = output_dict.get("context", None)
        if context is None:
            context = self._build_context(inputs)

        B, C, T, V = inputs["input"].shape
        x = torch.randn(B, C, T, V, device=self.device) * float(temperature)
        dt = 1.0 / float(num_steps)

        for i in range(num_steps):
            t = torch.full((B,), float(i) / float(num_steps), device=self.device)
            v = self.flow_net(
                x,
                t=t,
                context=context,
                timestep_mask=inputs["timestep_mask"],
                vehicle_mask=inputs["target_vehicle_mask"],
            )
            x = x + dt * v
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

        sample = self.unnormalize_trajectory(x)  # [B,C,T,V]
        sample = sample.permute(0, 3, 2, 1).contiguous()  # [B,A,T,5]

        agent_valid = inputs["target_vehicle_mask"].bool().unsqueeze(-1)  # [B,A,1]
        time_valid = inputs["timestep_mask"].permute(0, 2, 1).bool()      # [B,A,T]
        pred_valid = agent_valid & time_valid
        sample = sample.masked_fill(~pred_valid.unsqueeze(-1), 0.0)
        return sample

    @torch.no_grad()
    def sample(self, batch, n_samples=1, temperature=None, return_global=True):
        if temperature is None:
            temperature = self._temp
        n_samples = int(max(1, n_samples))

        inputs = self.prepare_glow_inputs(batch)
        context = self._build_context(inputs)
        output_dict = {"inputs": inputs, "context": context}

        all_samples = []
        for _ in range(n_samples):
            sample_local = self.sample_trajectories(
                output_dict,
                temperature=temperature,
                num_steps=self._sample_steps,
            )[..., :3]  # [B,A,T,3]

            pred_valid = (
                inputs["target_vehicle_mask"].bool().unsqueeze(-1)
                & inputs["timestep_mask"].permute(0, 2, 1).bool()
            )

            if return_global:
                current_states = inputs["current_states"]  # [B,A,3]
                sample_global = self._transform_local_to_global(sample_local, current_states)
                sample_global = sample_global.masked_fill(~pred_valid.unsqueeze(-1), 0.0)
                all_samples.append(sample_global)
            else:
                sample_local = sample_local.masked_fill(~pred_valid.unsqueeze(-1), 0.0)
                all_samples.append(sample_local)

        return torch.stack(all_samples, dim=1)  # [B,n_samples,A,T,3]

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.forward_and_get_loss(batch, prefix="train/")
        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.forward_and_get_loss(batch, prefix="val/")
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
