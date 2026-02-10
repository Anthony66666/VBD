"""
MapGlow model wrapper for PyTorch Lightning training.
Uses VBD's data pipeline and input format.
Fully compatible with VBD's training infrastructure.
"""
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.nn.functional import smooth_l1_loss
from math import log, pi
import numpy as np

# Import the original MapGlow model
import sys
sys.path.append('/home/anthony/VBD')
from MapGLow import Glow, create_timestep_mask

# Import VBD utilities for coordinate transforms
from vbd.model.model_utils import (
    batch_transform_trajs_to_global_frame,
    wrap_angle,
)


class MapGlowWrapper(pl.LightningModule):
    """
    PyTorch Lightning wrapper for MapGlow (Normalizing Flow) model.
    Adapts MapGlow to use VBD's data format and training pipeline.
    
    Key differences from VBD:
    - Uses Normalizing Flow instead of Diffusion for trajectory generation
    - Outputs full trajectory directly instead of actions
    - Uses NLL loss instead of diffusion loss
    """

    def __init__(self, cfg: dict):
        """
        Initialize the MapGlow wrapper.

        Args:
            cfg (dict): Configuration parameters for the model.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.cfg = cfg
        self._future_len = cfg['future_len']  # 80
        self._agents_len = cfg['agents_len']  # 32
        self._history_len = cfg.get('history_len', 11)  # 11 timesteps history
        # MapGlow specific parameters
        self._in_channel = cfg.get('in_channel', 5)  # x, y, vx, vy, yaw
        self._condition_dim = cfg.get('condition_dim', 256)  # Match encoder output dim
        self._n_flow = cfg.get('n_flow', 16)
        self._n_block = cfg.get('n_block', 3)
        self._affine = cfg.get('affine', True)
        self._conv_lu = cfg.get('conv_lu', True)
        self._temp = cfg.get('temperature', 0.7)
        self._valid_eps = cfg.get('valid_eps', 1e-6)
        self._num_lane_types = int(cfg.get('num_lane_types', 21))
        self._num_traffic_light_states = int(cfg.get('num_traffic_light_states', 8))
        self._lane_type_encoding = str(cfg.get('lane_type_encoding', 'embedding')).lower()
        if self._lane_type_encoding not in ('embedding', 'onehot'):
            self._lane_type_encoding = 'embedding'
        self._lane_type_embed_dim = int(cfg.get('lane_type_embed_dim', 16))
        self._lane_tl_embed_dim = int(cfg.get('lane_tl_embed_dim', 8))
        self._tl_state_embed_dim = int(cfg.get('tl_state_embed_dim', 8))
        self._use_traffic_light = bool(cfg.get('use_traffic_light', True))
        self._use_lane_tl_in_map = bool(cfg.get('use_lane_tl_in_map', self._use_traffic_light))
        self._lane_selection_mode = str(cfg.get('lane_selection_mode', 'hybrid')).lower()
        self._topk_lanes = int(cfg.get('topk_lanes', 16))
        self._hybrid_global_lanes = int(cfg.get('hybrid_global_lanes', 16))
        self._trajectory_mode = str(cfg.get('trajectory_mode', 'absolute')).lower()
        if self._trajectory_mode not in ('absolute', 'delta'):
            self._trajectory_mode = 'absolute'
        self._use_delta_trajectory = self._trajectory_mode == 'delta'
        self._delta_pos_scale = float(cfg.get('delta_pos_scale', 1.0))
        if self._delta_pos_scale <= 0:
            self._delta_pos_scale = 1.0

        map_input_dim = 5 if self._lane_type_encoding == 'embedding' else (4 + self._num_lane_types)
        traffic_light_input_dim = 3  # [x, y, tl_state_idx]
        
        # Build the Glow model
        self.glow = Glow(
            in_channel=self._in_channel,
            condition_dim=self._condition_dim,
            n_flow=self._n_flow,
            n_block=self._n_block,
            affine=self._affine,
            conv_lu=self._conv_lu,
            max_points=30,  # VBD uses 30 points per polyline
            map_input_dim=map_input_dim,
            traffic_light_input_dim=traffic_light_input_dim,
            num_lane_types=self._num_lane_types,
            num_traffic_light_states=self._num_traffic_light_states,
            lane_type_encoding=self._lane_type_encoding,
            lane_type_embed_dim=self._lane_type_embed_dim,
            lane_tl_embed_dim=self._lane_tl_embed_dim,
            tl_state_embed_dim=self._tl_state_embed_dim,
            lane_selection_mode=self._lane_selection_mode,
            topk_lanes=self._topk_lanes,
            hybrid_global_lanes=self._hybrid_global_lanes,
        )
        
        # Normalization parameters for trajectory data
        # Position normalized by 100 meters
        self.register_buffer('pos_scale', torch.tensor(100.0))  # position scale (meters)
        self.register_buffer('delta_pos_scale', torch.tensor(self._delta_pos_scale))  # delta x/y scale
        self.register_buffer('vel_scale', torch.tensor(10.0))   # velocity scale (m/s)
        self.register_buffer('yaw_scale', torch.tensor(1.0))    # yaw already in [-pi, pi]
        shape_scale_cfg = cfg.get('agent_shape_scale', [10.0, 5.0, 3.0])
        self.register_buffer('shape_scale', torch.tensor(shape_scale_cfg, dtype=torch.float32))
        self._warned_missing_sdc = False

    def _xyyaw_from_state5(self, traj5):
        """
        Extract [x, y, yaw] from state channels [x, y, vx, vy, yaw].
        """
        return torch.stack([traj5[..., 0], traj5[..., 1], traj5[..., 4]], dim=-1)

    def _extract_last_history_anchor(self, history_data, history_timestep_mask=None):
        """
        Extract per-agent anchor from last valid history timestep.
        history_data: [B, C, T_hist, V], channel order [x, y, vx, vy, yaw]
        history_timestep_mask: [B, T_hist, V] bool (optional)
        Returns:
            anchor_xyyaw: [B, 3, V] (x, y, yaw)
            anchor_valid: [B, V] bool
        """
        B, C, T_hist, V = history_data.shape
        device = history_data.device
        dtype = history_data.dtype

        if history_timestep_mask is None:
            history_timestep_mask = create_timestep_mask(history_data, padding_value=0.0)
        hist_mask = history_timestep_mask.bool()  # [B, T_hist, V]

        anchor_valid = hist_mask.any(dim=1)  # [B, V]
        time_idx = torch.arange(T_hist, device=device).view(1, T_hist, 1)
        weighted = torch.where(hist_mask, time_idx, torch.full_like(time_idx, -1))
        last_idx = weighted.max(dim=1).values.clamp(min=0).long()  # [B, V]

        # Gather last valid [x, y, yaw]
        traj_xyyaw = torch.stack(
            [history_data[:, 0], history_data[:, 1], history_data[:, 4]], dim=1
        )  # [B, 3, T_hist, V]
        traj_flat = traj_xyyaw.permute(0, 3, 2, 1).reshape(B * V, T_hist, 3)  # [B*V, T_hist, 3]
        gather_idx = last_idx.reshape(B * V, 1, 1).expand(-1, 1, 3)
        anchor = torch.gather(traj_flat, 1, gather_idx).squeeze(1).reshape(B, V, 3)  # [B, V, 3]
        anchor = torch.where(anchor_valid.unsqueeze(-1), anchor, torch.zeros_like(anchor))
        anchor_xyyaw = anchor.permute(0, 2, 1).to(dtype=dtype)  # [B, 3, V]
        return anchor_xyyaw, anchor_valid

    def _absolute_to_delta_target(self, traj, timestep_mask=None, anchor_xyyaw=None, anchor_valid=None):
        """
        Convert absolute target trajectory to delta target.
        Input/Output channel order: [x, y, vx, vy, yaw], only x/y/yaw become deltas.
        traj: [B, C, T, V]
        timestep_mask: [B, T, V], True = valid (optional)
        """
        if traj.size(2) == 0:
            return traj

        out = traj.clone()
        x = traj[:, 0]
        y = traj[:, 1]
        vx = traj[:, 2]
        vy = traj[:, 3]
        yaw = traj[:, 4]

        dx = torch.zeros_like(x)
        dy = torch.zeros_like(y)
        dyaw = torch.zeros_like(yaw)

        if anchor_xyyaw is None:
            dx[:, 0] = x[:, 0]
            dy[:, 0] = y[:, 0]
            dyaw[:, 0] = yaw[:, 0]
        else:
            anchor_x = anchor_xyyaw[:, 0]  # [B, V]
            anchor_y = anchor_xyyaw[:, 1]
            anchor_yaw = anchor_xyyaw[:, 2]
            dx[:, 0] = x[:, 0] - anchor_x
            dy[:, 0] = y[:, 0] - anchor_y
            dyaw[:, 0] = wrap_angle(yaw[:, 0] - anchor_yaw)

        if traj.size(2) > 1:
            dx[:, 1:] = x[:, 1:] - x[:, :-1]
            dy[:, 1:] = y[:, 1:] - y[:, :-1]
            dyaw[:, 1:] = wrap_angle(yaw[:, 1:] - yaw[:, :-1])

        if timestep_mask is not None:
            valid0_bool = timestep_mask[:, 0, :]
            if anchor_valid is not None:
                valid0_bool = valid0_bool & anchor_valid.bool()
            valid0 = valid0_bool.to(dtype=dx.dtype)
            dx[:, 0] = dx[:, 0] * valid0
            dy[:, 0] = dy[:, 0] * valid0
            dyaw[:, 0] = dyaw[:, 0] * valid0
            if traj.size(2) > 1:
                pair_valid = (timestep_mask[:, 1:, :] & timestep_mask[:, :-1, :]).to(dtype=dx.dtype)
                dx[:, 1:] = dx[:, 1:] * pair_valid
                dy[:, 1:] = dy[:, 1:] * pair_valid
                dyaw[:, 1:] = dyaw[:, 1:] * pair_valid

        out[:, 0] = dx
        out[:, 1] = dy
        out[:, 2] = vx
        out[:, 3] = vy
        out[:, 4] = dyaw
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _delta_to_absolute_target(self, traj, timestep_mask=None, anchor_xyyaw=None, anchor_valid=None):
        """
        Convert delta target back to absolute target.
        Input/Output channel order: [x, y, vx, vy, yaw], where x/y/yaw are integrated from deltas.
        traj: [B, C, T, V]
        timestep_mask: [B, T, V], True = valid (optional)
        """
        if traj.size(2) == 0:
            return traj

        out = traj.clone()
        dx = traj[:, 0]
        dy = traj[:, 1]
        vx = traj[:, 2]
        vy = traj[:, 3]
        dyaw = traj[:, 4]

        x = torch.cumsum(dx, dim=1)
        y = torch.cumsum(dy, dim=1)
        yaw = wrap_angle(torch.cumsum(dyaw, dim=1))
        if anchor_xyyaw is not None:
            x = x + anchor_xyyaw[:, 0].unsqueeze(1)
            y = y + anchor_xyyaw[:, 1].unsqueeze(1)
            yaw = wrap_angle(yaw + anchor_xyyaw[:, 2].unsqueeze(1))

        out[:, 0] = x
        out[:, 1] = y
        out[:, 2] = vx
        out[:, 3] = vy
        out[:, 4] = yaw

        if timestep_mask is not None:
            valid = timestep_mask.bool()
            if anchor_valid is not None:
                valid = valid & anchor_valid.bool().unsqueeze(1)
            out = out * valid.unsqueeze(1).to(dtype=out.dtype)

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out
        
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        Same as VBD's configuration for consistency.
        """
        # Collect trainable parameters
        params_to_update = [p for p in self.parameters() if p.requires_grad]
        assert len(params_to_update) > 0, 'No parameters to update'
        
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg['lr'],
            weight_decay=self.cfg.get('weight_decay', 0.01)
        )
        
        lr_warmup_step = self.cfg.get('lr_warmup_step', 1000)
        lr_step_freq = self.cfg.get('lr_step_freq', 1000)
        lr_step_gamma = self.cfg.get('lr_step_gamma', 0.98)
        
        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma ** n
            
            lr_scale = max(1e-2, min(1.0, lr_scale))
            return lr_scale
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(
                step, lr_warmup_step, lr_step_freq, lr_step_gamma
            )
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    def prepare_glow_inputs(self, batch):
        """
        Convert VBD batch format to MapGlow input format.
        
        VBD batch contains:
            - agents_history: [B, A, T_hist, 8] (x, y, yaw, vx, vy, length, width, height)
            - agents_future: [B, A, T_fut, 5] (x, y, yaw, vx, vy)
            - agents_type: [B, A]
            - agents_interested: [B, A]
            - polylines: [B, L, P, D] map polylines
            - polylines_valid: [B, L]
            - traffic_light_points: [B, TL, D]
            - relations: [B, N, N, 3] relation encodings
            - anchors: [B, A, Q, 2]
        
        MapGlow expects:
            - input: [B, C, T, V] trajectory data (target) in EGO frame
            - history_data: [B, C, T_hist, V] history trajectory in EGO frame
            - agent_shape: [B, V, 3] normalized shape (length, width, height)
            - map_data: [B, L, P, D_map] map data in EGO frame
            - map_mask: [B, L, P] map validity mask
            - traffic_light_data: [B, TL, D_tl] traffic light tokens in EGO frame
            - traffic_light_mask: [B, TL] traffic light validity mask
            - agent_types: [B, V] agent types
            - target_vehicle_mask: [B, V] valid vehicles
            - history_vehicle_mask: [B, V] valid vehicles in history
            - timestep_mask: [B, T, V] valid timesteps
            - condition: [B, V, D] condition from encoder (optional)
        
        Returns:
            dict: Prepared inputs for MapGlow
        """
        B = batch['agents_history'].shape[0]
        A = min(batch['agents_history'].shape[1], self._agents_len)
        
        # Extract and truncate to max agents
        agents_history = batch['agents_history'][:, :A]  # [B, A, T_hist, 8]
        agents_future = batch['agents_future'][:, :A]    # [B, A, T_fut, 5]
        agents_type = batch['agents_type'][:, :A]        # [B, A]
        agents_interested = batch['agents_interested'][:, :A]  # [B, A]
        
        # Pad to agents_len if needed
        if A < self._agents_len:
            pad_size = self._agents_len - A
            agents_history = torch.nn.functional.pad(agents_history, (0, 0, 0, 0, 0, pad_size))
            agents_future = torch.nn.functional.pad(agents_future, (0, 0, 0, 0, 0, pad_size))
            agents_type = torch.nn.functional.pad(agents_type, (0, pad_size))
            agents_interested = torch.nn.functional.pad(agents_interested, (0, pad_size))
        
        T_hist = agents_history.shape[2]  # 11
        T_fut = agents_future.shape[2]    # 81 (including current)
        if agents_history.shape[-1] >= 8:
            agent_shape_seq = agents_history[..., 5:8].clone()  # [B, A, T_hist, 3]
        else:
            agent_shape_seq = torch.zeros(
                agents_history.shape[0],
                agents_history.shape[1],
                agents_history.shape[2],
                3,
                device=agents_history.device,
                dtype=agents_history.dtype,
            )

        # Optional explicit valid bits from dataset (preferred).
        agents_history_valid = batch.get('agents_history_valid', None)
        agents_future_valid = batch.get('agents_future_valid', None)
        if agents_history_valid is not None:
            agents_history_valid = agents_history_valid[:, :A]
            if A < self._agents_len:
                agents_history_valid = torch.nn.functional.pad(agents_history_valid, (0, 0, 0, pad_size))
            agents_history_valid = agents_history_valid.bool()
        if agents_future_valid is not None:
            agents_future_valid = agents_future_valid[:, :A]
            if A < self._agents_len:
                agents_future_valid = torch.nn.functional.pad(agents_future_valid, (0, 0, 0, pad_size))
            agents_future_valid = agents_future_valid.bool()
        
        # Determine ego (SDC) index and pose for EGO frame transformation
        ego_states = self._get_ego_states(batch, agents_history)

        # Build explicit timestep validity masks (fallback to xy-based if not provided).
        if agents_history_valid is None:
            agents_history_valid = (agents_history[..., :2].abs().sum(-1) > self._valid_eps)
        if agents_future_valid is None:
            agents_future_valid = (agents_future[..., :2].abs().sum(-1) > self._valid_eps)
        
        # Transform to EGO frame (ego at origin, heading = 0)
        agents_history_local = self._transform_trajs_to_ego_frame(
            agents_history[..., :5], ego_states, valid_mask=agents_history_valid
        )
        agents_future_local = self._transform_trajs_to_ego_frame(
            agents_future, ego_states, valid_mask=agents_future_valid
        )
        
        # Extract trajectory features: [x, y, vx, vy, yaw]
        # History format after transform: [local_x, local_y, local_theta, local_vx, local_vy]
        history_traj = torch.stack([
            agents_history_local[..., 0],  # local_x
            agents_history_local[..., 1],  # local_y
            agents_history_local[..., 3],  # local_vx
            agents_history_local[..., 4],  # local_vy
            agents_history_local[..., 2],  # local_yaw
        ], dim=-1)  # [B, A, T_hist, 5]
        
        # Future format after transform: [local_x, local_y, local_yaw, local_vx, local_vy]
        future_traj = torch.stack([
            agents_future_local[..., 0],  # local_x
            agents_future_local[..., 1],  # local_y
            agents_future_local[..., 3],  # local_vx
            agents_future_local[..., 4],  # local_vy
            agents_future_local[..., 2],  # local_yaw
        ], dim=-1)  # [B, A, T_fut, 5]
        
        # Use only future part (skip current timestep which overlaps with history[-1])
        target_traj = future_traj[:, :, 1:, :]  # [B, A, 80, 5]
        
        # Reshape to MapGlow format: [B, C, T, V]
        history_data = history_traj.permute(0, 3, 2, 1)  # [B, 5, T_hist, A]
        target_data = target_traj.permute(0, 3, 2, 1)    # [B, 5, 80, A]
        
        # Clean data before normalization - replace any NaN/Inf with zeros
        if torch.isnan(history_data).any() or torch.isinf(history_data).any():
            print(f"Warning: Cleaning NaN/Inf in history_data before normalization")
            history_data = torch.nan_to_num(history_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(target_data).any() or torch.isinf(target_data).any():
            print(f"Warning: Cleaning NaN/Inf in target_data before normalization")
            target_data = torch.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create masks FIRST (before normalization so we can use them)
        # Vehicle mask: True if agent is valid (interested > 0)
        vehicle_mask = agents_interested > 0  # [B, A]
        
        # Timestep mask from explicit valid bits: [B, T, V]
        future_valid = agents_future_valid[:, :, 1:]  # [B, A, T]
        timestep_mask = future_valid.permute(0, 2, 1).bool()  # [B, T, A]
        
        # Additional check: if an agent has NO valid future timesteps at all, mark it as invalid
        # This prevents all-zero agents from causing numerical issues
        has_any_valid_timestep = timestep_mask.any(dim=1)  # [B, A] - True if agent has at least one valid timestep
        vehicle_mask = vehicle_mask & has_any_valid_timestep  # Combine both conditions

        # History mask from explicit valid bits
        history_valid = agents_history_valid.bool()  # [B, A, T_hist]
        history_vehicle_mask = history_valid.any(dim=2)  # [B, A]
        history_timestep_mask = history_valid.permute(0, 2, 1).bool()  # [B, T_hist, A]

        # Agent shape: use last valid history timestep per agent (length/width/height).
        B_s, A_s, T_s, _ = agent_shape_seq.shape
        t_index = torch.arange(T_s, device=agent_shape_seq.device).view(1, 1, T_s)
        weighted_t = torch.where(history_valid, t_index, torch.full_like(t_index, -1))
        last_valid_idx = weighted_t.max(dim=2).values.clamp(min=0).long()  # [B, A]
        shape_flat = agent_shape_seq.reshape(B_s * A_s, T_s, 3)
        gather_idx = last_valid_idx.reshape(B_s * A_s, 1, 1).expand(-1, 1, 3)
        agent_shape = torch.gather(shape_flat, 1, gather_idx).squeeze(1).reshape(B_s, A_s, 3)
        agent_shape = torch.where(
            history_vehicle_mask.unsqueeze(-1),
            agent_shape,
            torch.zeros_like(agent_shape),
        )
        agent_shape = torch.nan_to_num(agent_shape, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        agent_shape = agent_shape / self.shape_scale.view(1, 1, 3).to(agent_shape.device, agent_shape.dtype)
        
        # For safety: ensure at least one valid agent per batch
        # If no valid agents, mark the first agent as "valid" with dummy data
        no_valid_agents = ~vehicle_mask.any(dim=1)  # [B]
        if no_valid_agents.any():
            print(f"Warning: {no_valid_agents.sum()} batches have no valid agents, using dummy agent")
            vehicle_mask[no_valid_agents, 0] = True
            timestep_mask[no_valid_agents, :, 0] = True
            history_vehicle_mask[no_valid_agents, 0] = True
            history_timestep_mask[no_valid_agents, -1, 0] = True

        target_anchor_xyyaw, target_anchor_valid = self._extract_last_history_anchor(
            history_data, history_timestep_mask=history_timestep_mask
        )

        # Optional target representation switch:
        # - absolute: [x, y, vx, vy, yaw] (default)
        # - delta:    [dx, dy, vx, vy, dyaw]
        # NOTE: history_data must stay absolute because ContextEncoder relies on real positions.
        if self._use_delta_trajectory:
            target_data = self._absolute_to_delta_target(
                target_data,
                timestep_mask=timestep_mask,
                anchor_xyyaw=target_anchor_xyyaw,
                anchor_valid=target_anchor_valid,
            )
        
        # Normalize the trajectory data
        history_data = self.normalize_trajectory(history_data, use_delta_xy=False)
        target_data = self.normalize_trajectory(target_data, use_delta_xy=self._use_delta_trajectory)
        
        # Apply masks to zero out invalid data (important for numerical stability)
        # For completely invalid agents, replace with small non-zero values to avoid division by zero
        # Mask invalid agents
        valid_agent_mask = vehicle_mask.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, A]
        
        # For invalid agents, set to small constant values instead of zero
        # This prevents issues in normalizing flow when processing all-zero inputs
        target_data = target_data * valid_agent_mask + (1 - valid_agent_mask) * 1e-6
        history_data = history_data * valid_agent_mask + (1 - valid_agent_mask) * 1e-6
        
        # Mask invalid timesteps (but keep non-zero for invalid agents from above)
        timestep_valid_mask = timestep_mask.float().unsqueeze(1)  # [B, 1, T, A]
        # Only apply timestep mask to valid agents
        target_data = torch.where(
            valid_agent_mask.expand_as(target_data) > 0,
            target_data * timestep_valid_mask + (1 - timestep_valid_mask) * 1e-6,
            target_data  # Keep the 1e-6 values for invalid agents
        )
        
        # Prepare map data - transform to local frame
        polylines = batch['polylines']  # [B, L, P, 5] - x, y, heading, traffic_light, lane_type
        polylines_valid = batch['polylines_valid']  # [B, L]
        polyline_ids = batch.get('polyline_ids', None)  # [B, L], optional roadgraph ids
        lane_topology = batch.get('lane_topology', None)  # [B, L, L, 4], optional
        polyline_point_valid = batch.get('polylines_point_valid', None)
        if polyline_point_valid is None:
            polyline_point_valid = polylines[..., :2].abs().sum(-1) > self._valid_eps
        else:
            polyline_point_valid = polyline_point_valid.bool()

        polylines_local = self._transform_polylines_to_ego_frame(
            polylines, ego_states, point_valid_mask=polyline_point_valid
        )

        # Create map_mask: [B, L, P] - per-point validity
        # polylines_valid indicates lane-level validity; point validity uses non-zero xy
        L, P = polylines.shape[1], polylines.shape[2]
        point_valid = polyline_point_valid.bool()  # [B, L, P]
        map_mask = polylines_valid.unsqueeze(-1).bool() & point_valid  # [B, L, P]
        
        # Convert VBD format to MapGlow map format
        # VBD polyline point: [x, y, heading, traffic_light_state, lane_type]
        # MapGlow map point:
        #   embedding mode: [x, y, heading, lane_type_idx, lane_tl_state_idx]
        #   onehot mode:    [x, y, heading, lane_type_onehot..., lane_tl_state_idx]
        B = polylines_local.shape[0]
        
        # Extract features
        xy = polylines_local[..., :2]  # [B, L, P, 2]
        heading = polylines_local[..., 2:3]  # [B, L, P, 1]
        traffic_light = polylines_local[..., 3:4]  # [B, L, P, 1]
        lane_type = polylines_local[..., 4:5]  # [B, L, P, 1]

        lane_type_idx = lane_type.squeeze(-1).round().long().clamp(0, self._num_lane_types - 1)
        if self._use_lane_tl_in_map:
            traffic_light_idx = traffic_light.squeeze(-1).round().long().clamp(0, self._num_traffic_light_states - 1)
        else:
            traffic_light_idx = torch.zeros_like(lane_type_idx)
        traffic_light_idx_f = traffic_light_idx.unsqueeze(-1).to(dtype=xy.dtype)

        if self._lane_type_encoding == 'onehot':
            lane_type_onehot = torch.nn.functional.one_hot(
                lane_type_idx, num_classes=self._num_lane_types
            ).to(dtype=xy.dtype)
            map_data = torch.cat([xy, heading, lane_type_onehot, traffic_light_idx_f], dim=-1)
        else:
            lane_type_idx_f = lane_type_idx.unsqueeze(-1).to(dtype=xy.dtype)
            # Assemble map_data: [B, L, P, 5]
            map_data = torch.cat([xy, heading, lane_type_idx_f, traffic_light_idx_f], dim=-1)
        
        # Normalize map positions
        map_data[..., :2] = map_data[..., :2] / self.pos_scale
        map_data[..., 2] = map_data[..., 2] / self.yaw_scale  # heading
        map_data = map_data * map_mask.unsqueeze(-1).float()
        
        # Check for NaN/Inf in map data
        if torch.isnan(map_data).any() or torch.isinf(map_data).any():
            print("Warning: NaN/Inf detected in map data")
            map_data = torch.nan_to_num(map_data, nan=0.0, posinf=10.0, neginf=-10.0)

        # Prepare traffic light tokens (independent context branch, like VBD)
        traffic_light_points = batch.get('traffic_light_points', None)
        if self._use_traffic_light and (traffic_light_points is not None):
            traffic_light_local = self._transform_traffic_lights_to_ego_frame(traffic_light_points, ego_states)  # [B, TL, 3]
            tl_xy = traffic_light_local[..., :2]
            tl_state = traffic_light_local[..., 2].round().long().clamp(0, self._num_traffic_light_states - 1)
            traffic_light_valid = batch.get('traffic_light_valid', None)
            if traffic_light_valid is None:
                tl_mask = traffic_light_points[..., :2].abs().sum(dim=-1) > self._valid_eps  # [B, TL]
            else:
                tl_mask = traffic_light_valid.bool()
            tl_state_f = tl_state.unsqueeze(-1).to(dtype=tl_xy.dtype)
            traffic_light_data = torch.cat([tl_xy, tl_state_f], dim=-1)  # [B, TL, 3]
            traffic_light_data[..., :2] = traffic_light_data[..., :2] / self.pos_scale
            traffic_light_data = traffic_light_data * tl_mask.unsqueeze(-1).float()
            traffic_light_mask = tl_mask
        else:
            traffic_light_data = torch.zeros(
                B, 1, 3, device=map_data.device, dtype=map_data.dtype
            )
            traffic_light_mask = torch.zeros(B, 1, device=map_data.device, dtype=torch.bool)

        if torch.isnan(traffic_light_data).any() or torch.isinf(traffic_light_data).any():
            print("Warning: NaN/Inf detected in traffic light data")
            traffic_light_data = torch.nan_to_num(traffic_light_data, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Note: MapGlow's ContextEncoder expects 'condition' to be integer labels for embedding lookup
        # We don't use encoder outputs here; data inputs already contain scene context
        condition = None
        
        return {
            'input': target_data,  # [B, C, T, V] - target trajectory (what Glow learns)
            'history_data': history_data,  # [B, C, T_hist, V]
            'agent_shape': agent_shape,  # [B, V, 3] normalized [length,width,height]
            'map_data': map_data,  # [B, L, P, D]
            'map_mask': map_mask,  # [B, L, P]
            'polyline_ids': polyline_ids,  # [B, L] optional
            'lane_topology': lane_topology,  # [B, L, L, 4] optional
            'traffic_light_data': traffic_light_data,  # [B, TL, 3] = [x, y, tl_state_idx]
            'traffic_light_mask': traffic_light_mask,  # [B, TL]
            'agent_types': agents_type,  # [B, V]
            'target_vehicle_mask': vehicle_mask,  # [B, V]
            'history_vehicle_mask': history_vehicle_mask,  # [B, V]
            'history_timestep_mask': history_timestep_mask,  # [B, T_hist, V]
            'timestep_mask': timestep_mask,  # [B, T, V]
            'target_anchor_xyyaw': target_anchor_xyyaw,  # [B, 3, V], local current anchor
            'target_anchor_valid': target_anchor_valid,  # [B, V]
            'condition': condition,  # [B, V, D] or None
            # Keep original data for loss computation and metrics
            'agents_future': agents_future,  # [B, A, T_fut, 5]
            'agents_future_local': agents_future_local,  # [B, A, T_fut, 5] in local frame
            'agents_interested': agents_interested,  # [B, V]
            'current_states': ego_states[:, None, :].expand(-1, self._agents_len, -1),  # [B, A, 3]
        }

    def _get_ego_states(self, batch, agents_history):
        """
        Get ego (SDC) current state for each batch.
        Falls back to the first agent if sdc_id/agents_id are missing.
        """
        B = agents_history.shape[0]
        A = agents_history.shape[1]
        device = agents_history.device

        agents_id = batch.get('agents_id', None)
        sdc_id = batch.get('sdc_id', None)

        if agents_id is None or sdc_id is None:
            ego_idx = torch.zeros(B, dtype=torch.long, device=device)
            if not self._warned_missing_sdc:
                print(
                    "Warning: agents_id/sdc_id missing in batch; fallback ego_idx=0. "
                    "This may degrade ego-frame consistency.",
                    flush=True,
                )
                self._warned_missing_sdc = True
        else:
            if agents_id.dim() == 1:
                agents_id = agents_id.view(B, -1)
            # Keep id table aligned with truncated agent tensor.
            agents_id = agents_id[:, :A]
            sdc_id = sdc_id.view(B, 1)
            matches = agents_id.eq(sdc_id)  # [B, A]
            has_match = matches.any(dim=1)
            ego_idx = torch.argmax(matches.int(), dim=1)
            ego_idx = torch.where(has_match, ego_idx, torch.zeros_like(ego_idx))
            if (not has_match.all()) and (not self._warned_missing_sdc):
                n_miss = int((~has_match).sum().item())
                print(
                    f"Warning: sdc_id not found in truncated agents for {n_miss} samples; fallback ego_idx=0.",
                    flush=True,
                )
                self._warned_missing_sdc = True

        batch_idx = torch.arange(B, device=device)
        ego_states = agents_history[batch_idx, ego_idx, -1, :3]  # [B, 3]
        return ego_states

    def _transform_trajs_to_ego_frame(self, trajs, ego_states, valid_mask=None):
        """
        Transform trajectories to ego frame.

        Args:
            trajs: [B, A, T, 5] (x, y, yaw, vx, vy)
            ego_states: [B, 3] (ego_x, ego_y, ego_yaw)
        """
        x = trajs[..., 0]
        y = trajs[..., 1]
        yaw = trajs[..., 2]
        vx = trajs[..., 3]
        vy = trajs[..., 4]

        ref_x = ego_states[:, 0].view(-1, 1, 1)
        ref_y = ego_states[:, 1].view(-1, 1, 1)
        ref_yaw = ego_states[:, 2].view(-1, 1, 1)

        cos_ref = torch.cos(ref_yaw)
        sin_ref = torch.sin(ref_yaw)

        dx = x - ref_x
        dy = y - ref_y
        local_x = dx * cos_ref + dy * sin_ref
        local_y = -dx * sin_ref + dy * cos_ref

        local_yaw = wrap_angle(yaw - ref_yaw)

        local_vx = vx * cos_ref + vy * sin_ref
        local_vy = -vx * sin_ref + vy * cos_ref

        local_trajs = torch.stack([local_x, local_y, local_yaw, local_vx, local_vy], dim=-1)

        if valid_mask is not None:
            local_trajs = local_trajs.masked_fill(~valid_mask.unsqueeze(-1).bool(), 0.0)
        else:
            invalid_mask = (trajs[..., :2].abs().sum(-1) <= self._valid_eps).unsqueeze(-1)
            local_trajs = local_trajs.masked_fill(invalid_mask, 0.0)

        return local_trajs

    def _transform_polylines_to_ego_frame(self, polylines, ego_states, point_valid_mask=None):
        """
        Transform polylines to ego frame.

        Args:
            polylines: [B, L, P, 5] (x, y, heading, traffic_light, lane_type)
            ego_states: [B, 3]
        """
        x = polylines[..., 0]
        y = polylines[..., 1]
        heading = polylines[..., 2]

        ref_x = ego_states[:, 0].view(-1, 1, 1)
        ref_y = ego_states[:, 1].view(-1, 1, 1)
        ref_yaw = ego_states[:, 2].view(-1, 1, 1)

        cos_ref = torch.cos(ref_yaw)
        sin_ref = torch.sin(ref_yaw)

        dx = x - ref_x
        dy = y - ref_y
        local_x = dx * cos_ref + dy * sin_ref
        local_y = -dx * sin_ref + dy * cos_ref

        local_heading = wrap_angle(heading - ref_yaw)

        local_polylines = torch.stack([local_x, local_y, local_heading], dim=-1)
        if point_valid_mask is not None:
            local_polylines = local_polylines.masked_fill(~point_valid_mask.unsqueeze(-1).bool(), 0.0)
        else:
            local_polylines[polylines[..., :3] == 0] = 0

        polylines = torch.cat([local_polylines, polylines[..., 3:]], dim=-1)
        return polylines

    def _transform_traffic_lights_to_ego_frame(self, traffic_light_points, ego_states):
        """
        Transform traffic lights to ego frame.

        Args:
            traffic_light_points: [B, TL, 3] (x, y, state)
            ego_states: [B, 3] (ego_x, ego_y, ego_yaw)
        """
        x = traffic_light_points[..., 0]
        y = traffic_light_points[..., 1]
        state = traffic_light_points[..., 2]

        ref_x = ego_states[:, 0].view(-1, 1)
        ref_y = ego_states[:, 1].view(-1, 1)
        ref_yaw = ego_states[:, 2].view(-1, 1)

        dx = x - ref_x
        dy = y - ref_y

        cos_yaw = torch.cos(-ref_yaw)
        sin_yaw = torch.sin(-ref_yaw)

        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        local_points = torch.stack([local_x, local_y, state], dim=-1)
        return local_points
    
    def _transform_future_to_local(self, agents_future, current_states):
        """
        Transform future trajectory to local frame.
        
        Args:
            agents_future: [B, A, T, 5] (x, y, yaw, vx, vy)
            current_states: [B, A, 3] (x, y, yaw)
        
        Returns:
            local_future: [B, A, T, 5] (local_x, local_y, local_yaw, local_vx, local_vy)
        """
        x = agents_future[..., 0]
        y = agents_future[..., 1]
        yaw = agents_future[..., 2]
        vx = agents_future[..., 3]
        vy = agents_future[..., 4]
        
        ref_x = current_states[..., 0:1]  # [B, A, 1]
        ref_y = current_states[..., 1:2]
        ref_yaw = current_states[..., 2:3]
        
        cos_ref = torch.cos(ref_yaw)
        sin_ref = torch.sin(ref_yaw)
        
        # Transform position to local frame
        dx = x - ref_x
        dy = y - ref_y
        local_x = dx * cos_ref + dy * sin_ref
        local_y = -dx * sin_ref + dy * cos_ref
        
        # Transform yaw
        local_yaw = wrap_angle(yaw - ref_yaw)
        
        # Transform velocity
        local_vx = vx * cos_ref + vy * sin_ref
        local_vy = -vx * sin_ref + vy * cos_ref
        
        local_future = torch.stack([local_x, local_y, local_yaw, local_vx, local_vy], dim=-1)
        
        # Zero out invalid positions (where original was zero)
        invalid_mask = (agents_future[..., :2].abs().sum(-1) <= self._valid_eps).unsqueeze(-1)
        local_future = local_future.masked_fill(invalid_mask, 0.0)
        
        return local_future
    
    def normalize_trajectory(self, traj, use_delta_xy=False):
        """
        Normalize trajectory data.
        traj: [B, C, T, V] where C = [x, y, vx, vy, yaw]
        """
        traj = traj.clone()
        # use_delta_xy=True means channels 0:2 are [dx, dy], otherwise [x, y].
        pos_scale = self.delta_pos_scale if use_delta_xy else self.pos_scale
        traj[:, 0:2] = traj[:, 0:2] / pos_scale
        traj[:, 2:4] = traj[:, 2:4] / self.vel_scale  # vx, vy
        traj[:, 4:5] = traj[:, 4:5] / self.yaw_scale  # yaw
        
        # Check for NaN/Inf
        if torch.isnan(traj).any() or torch.isinf(traj).any():
            print("Warning: NaN/Inf detected in normalized trajectory")
            traj = torch.nan_to_num(traj, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return traj
    
    def unnormalize_trajectory(self, traj, use_delta_xy=False):
        """
        Unnormalize trajectory data.
        traj: [B, C, T, V] where C = [x, y, vx, vy, yaw]
        """
        traj = traj.clone()
        pos_scale = self.delta_pos_scale if use_delta_xy else self.pos_scale
        traj[:, 0:2] = traj[:, 0:2] * pos_scale
        traj[:, 2:4] = traj[:, 2:4] * self.vel_scale
        traj[:, 4:5] = traj[:, 4:5] * self.yaw_scale
        return traj
    
    def forward(self, batch):
        """
        Forward pass of the MapGlow model.
        Similar to VBD's forward, but uses Normalizing Flow instead of Diffusion.
        
        Args:
            batch: Input batch from VBD dataloader
            
        Returns:
            output_dict: Dictionary containing model outputs
        """
        output_dict = {}
        
        # Step 1: Prepare inputs for Glow (use VBD data pipeline only)
        inputs = self.prepare_glow_inputs(batch)
        
        # Additional check before forward pass
        input_data = inputs['input']
        if torch.isnan(input_data).any() or torch.isinf(input_data).any():
            print(f"ERROR: NaN/Inf in input data before Glow forward!")
            print(f"  NaN count: {torch.isnan(input_data).sum()}, Inf count: {torch.isinf(input_data).sum()}")
            input_data = torch.nan_to_num(input_data, nan=0.0, posinf=0.0, neginf=0.0)
            inputs['input'] = input_data
        
        # Step 3: Forward through Glow (condition is None - we don't use it)
        log_p, logdet, z_outs = self.glow(
            input=inputs['input'],
            condition=None,  # MapGlow expects integer labels, we don't use it
            map_data=inputs['map_data'],
            map_mask=inputs['map_mask'],
            traffic_light_data=inputs['traffic_light_data'],
            traffic_light_mask=inputs['traffic_light_mask'],
            agent_types=inputs['agent_types'],
            agent_shape=inputs['agent_shape'],
            history_data=inputs['history_data'],
            history_timestep_mask=inputs['history_timestep_mask'],
            target_vehicle_mask=inputs['target_vehicle_mask'],
            history_vehicle_mask=inputs['history_vehicle_mask'],
            timestep_mask=inputs['timestep_mask']
        )
        
        output_dict['log_p'] = log_p
        output_dict['logdet'] = logdet
        output_dict['z_outs'] = z_outs
        output_dict['inputs'] = inputs
        
        return output_dict
    
    def compute_loss(self, log_p, logdet, vehicle_mask, timestep_mask=None):
        """
        Compute the negative log-likelihood loss for normalizing flow.
        
        Args:
            log_p: Log probability from Gaussian prior [B]
            logdet: Log determinant of Jacobian [B]
            vehicle_mask: [B, V] mask for valid vehicles
            timestep_mask: [B, T, V] mask for valid timesteps (optional)
            
        Returns:
            loss: Scalar loss value
        """
        # Check for NaN/Inf in inputs
        if torch.isnan(log_p).any() or torch.isinf(log_p).any():
            print(f"Warning: NaN/Inf in log_p: nan={torch.isnan(log_p).sum()}, inf={torch.isinf(log_p).sum()}")
            # Print batch statistics for debugging
            valid_log_p = log_p[~torch.isnan(log_p) & ~torch.isinf(log_p)]
            if valid_log_p.numel() > 0:
                print(f"  log_p range: [{valid_log_p.min():.2f}, {valid_log_p.max():.2f}]")
            else:
                print(f"  log_p: all values are NaN/Inf!")
            print(f"  Valid agents per batch: {vehicle_mask.sum(dim=1).tolist()}")
            log_p = torch.nan_to_num(log_p, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(logdet).any() or torch.isinf(logdet).any():
            print(f"Warning: NaN/Inf in logdet: nan={torch.isnan(logdet).sum()}, inf={torch.isinf(logdet).sum()}")
            valid_logdet = logdet[~torch.isnan(logdet) & ~torch.isinf(logdet)]
            if valid_logdet.numel() > 0:
                print(f"  logdet range: [{valid_logdet.min():.2f}, {valid_logdet.max():.2f}]")
            else:
                print(f"  logdet: all values are NaN/Inf!")
            logdet = torch.nan_to_num(logdet, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # NLL = -(log_p + logdet)
        # log_p and logdet are already summed per sample
        nll = -(log_p + logdet)  # [B]
        
        # Normalize by the number of valid elements
        # Count valid elements: sum over valid agents and valid timesteps
        if timestep_mask is not None:
            # [B, T, V] * [B, 1, V] -> [B, T, V] then sum
            valid_elements = (timestep_mask.float() * vehicle_mask.unsqueeze(1).float()).sum(dim=[1, 2])  # [B]
            valid_elements = valid_elements.clamp(min=1.0)  # Avoid division by zero
            nll = nll / valid_elements  # Normalize per batch
        
        # Clamp to prevent extreme losses
        nll = torch.clamp(nll, -1e6, 1e6)
        
        # Average over batch
        loss = nll.mean()
        
        # Final check
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: NaN/Inf in final loss, returning large finite value")
            loss = torch.tensor(1e6, device=loss.device, requires_grad=True)
        
        return loss
    
    def trajectory_loss(self, pred_trajs, gt_trajs, future_valid, agents_interested):
        """
        Compute trajectory loss similar to VBD's denoise_loss.
        
        Args:
            pred_trajs: [B, A, T, 3] predicted trajectories (x, y, yaw) in local frame
            gt_trajs: [B, A, T, 3] ground truth trajectories (x, y, yaw) in local frame
            future_valid: [B, A, T] validity mask
            agents_interested: [B, A] interest mask
            
        Returns:
            state_loss: Position loss
            yaw_loss: Yaw angle loss
        """
        future_mask = future_valid * (agents_interested[..., None] > 0)
        
        # Position loss
        state_loss = smooth_l1_loss(pred_trajs[..., :2], gt_trajs[..., :2], reduction='none').sum(-1)
        
        # Yaw loss with proper angle wrapping
        yaw_error = pred_trajs[..., 2] - gt_trajs[..., 2]
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        yaw_loss = torch.abs(yaw_error)
        
        # Apply mask
        state_loss = state_loss * future_mask
        yaw_loss = yaw_loss * future_mask
        
        # Mean loss
        valid_count = future_mask.sum().clamp(min=1)
        state_loss_mean = state_loss.sum() / valid_count
        yaw_loss_mean = yaw_loss.sum() / valid_count
        
        return state_loss_mean, yaw_loss_mean
    
    def forward_and_get_loss(self, batch, prefix='', debug=False):
        """
        Forward pass and compute loss.
        Similar to VBD's forward_and_get_loss method.
        
        Args:
            batch: Input batch from VBD dataloader
            prefix: Prefix for logging
            debug: Whether to return debug outputs
            
        Returns:
            loss: Total loss
            log_dict: Dictionary of logged values
            debug_outputs: (optional) Debug information
        """
        # Data inputs
        agents_future = batch['agents_future'][:, :self._agents_len]
        # Prefer dataset-provided validity mask; fallback to xy heuristic.
        if 'agents_future_valid' in batch:
            agents_future_valid = batch['agents_future_valid'][:, :self._agents_len].bool()
        else:
            agents_future_valid = agents_future[..., :2].abs().sum(-1) > self._valid_eps
        agents_interested = batch['agents_interested'][:, :self._agents_len]
        
        # Forward pass
        output_dict = self.forward(batch)
        
        log_p = output_dict['log_p']
        logdet = output_dict['logdet']
        inputs = output_dict['inputs']
        
        log_dict = {}
        debug_outputs = {}
        total_loss = 0
        
        # NLL Loss (main loss for normalizing flow)
        nll_loss = self.compute_loss(log_p, logdet, inputs['target_vehicle_mask'], inputs['timestep_mask'])
        total_loss = total_loss + nll_loss
        
        # Compute bits per dimension
        B, C, T, V = inputs['input'].shape
        n_pixel = C * T * V
        bpd = nll_loss / (log(2) * n_pixel)
        
        log_dict.update({
            f'{prefix}nll_loss': nll_loss.item(),
            f'{prefix}log_p': log_p.mean().item(),
            f'{prefix}logdet': logdet.mean().item(),
            f'{prefix}bpd': bpd.item(),
        })

        valid_vehicle_mean = inputs['target_vehicle_mask'].float().sum(dim=1).mean()
        valid_timestep_ratio = inputs['timestep_mask'].float().mean()
        input_abs_max = inputs['input'].abs().max()
        history_abs_max = inputs['history_data'].abs().max()
        log_dict.update({
            f'{prefix}valid_vehicle_mean': valid_vehicle_mean.item(),
            f'{prefix}valid_timestep_ratio': valid_timestep_ratio.item(),
            f'{prefix}input_abs_max': input_abs_max.item(),
            f'{prefix}history_abs_max': history_abs_max.item(),
        })
        
        # Optional: Sample and compute trajectory metrics (like VBD)
        if not self.training or debug:
            with torch.no_grad():
                # Sample from the model
                sampled_trajs = self.sample_trajectories(output_dict)  # [B, A, T, 5]
                
                # Get ground truth in local frame
                gt_local = inputs['agents_future_local'][:, :, 1:, :3]  # [B, A, T, 3]
                pred_local = self._xyyaw_from_state5(sampled_trajs)  # [B, A, T, 3]
                future_valid = agents_future_valid[:, :, 1:]
                
                # Compute trajectory metrics
                state_loss, yaw_loss = self.trajectory_loss(
                    pred_local, gt_local, future_valid, agents_interested
                )
                
                # ADE and FDE
                ade, fde = self.calculate_metrics(
                    pred_local, gt_local, future_valid, agents_interested
                )
                
                log_dict.update({
                    f'{prefix}state_loss': state_loss.item(),
                    f'{prefix}yaw_loss': yaw_loss.item(),
                    f'{prefix}ADE': ade,
                    f'{prefix}FDE': fde,
                })
                
                if debug:
                    debug_outputs['sampled_trajs'] = sampled_trajs
                    debug_outputs['gt_local'] = gt_local
        
        log_dict[f'{prefix}loss'] = total_loss.item()
        
        if debug:
            return total_loss, log_dict, debug_outputs
        else:
            return total_loss, log_dict
    
    def sample_trajectories(self, output_dict, temperature=None):
        """
        Sample trajectories from the model.
        
        Args:
            output_dict: Output from forward pass
            temperature: Sampling temperature (default: self._temp)
            
        Returns:
            sampled_trajs: [B, A, T, 5] sampled trajectories in local frame
        """
        if temperature is None:
            temperature = self._temp
        
        inputs = output_dict['inputs']
        B, C, T, V = inputs['input'].shape
        
        # Generate z samples from prior
        z_shapes = self._get_z_shapes(C, T, V)
        z_list = [torch.randn(B, *shape, device=self.device) * temperature 
                  for shape in z_shapes]
        
        # Reverse pass
        sample = self.glow.reverse(
            z_list,
            condition=None,  # MapGlow expects integer labels, we don't use it
            map_data=inputs['map_data'],
            map_mask=inputs['map_mask'],
            traffic_light_data=inputs['traffic_light_data'],
            traffic_light_mask=inputs['traffic_light_mask'],
            agent_types=inputs['agent_types'],
            agent_shape=inputs['agent_shape'],
            history_data=inputs['history_data'],
            history_timestep_mask=inputs['history_timestep_mask'],
            target_vehicle_mask=inputs['target_vehicle_mask'],
            history_vehicle_mask=inputs['history_vehicle_mask'],
            timestep_mask=inputs['timestep_mask']
        )
        
        # Unnormalize: [B, C, T, V] -> [B, V, T, C] -> [B, A, T, 5]
        sample = self.unnormalize_trajectory(sample, use_delta_xy=self._use_delta_trajectory)
        if self._use_delta_trajectory:
            sample = self._delta_to_absolute_target(
                sample,
                timestep_mask=inputs['timestep_mask'],
                anchor_xyyaw=inputs.get('target_anchor_xyyaw', None),
                anchor_valid=inputs.get('target_anchor_valid', None),
            )
        sample = sample.permute(0, 3, 2, 1)  # [B, A, T, 5]

        # Mask invalid agents/timesteps to avoid leaking unconstrained predictions.
        # target_vehicle_mask: [B, A], timestep_mask: [B, T, A] -> [B, A, T]
        agent_valid = inputs['target_vehicle_mask'].bool().unsqueeze(-1)  # [B, A, 1]
        time_valid = inputs['timestep_mask'].permute(0, 2, 1).bool()      # [B, A, T]
        pred_valid = agent_valid & time_valid
        sample = sample.masked_fill(~pred_valid.unsqueeze(-1), 0.0)
        
        return sample
    
    @torch.no_grad()
    def calculate_metrics(self, pred_trajs, gt_trajs, future_valid, agents_interested, top_k=None):
        """
        Calculate ADE and FDE metrics.
        Same as VBD's calculate_metrics_denoise.
        
        Args:
            pred_trajs: [B, A, T, 2 or 3] predicted trajectories
            gt_trajs: [B, A, T, 2 or 3] ground truth trajectories
            future_valid: [B, A, T] validity mask
            agents_interested: [B, A] interest mask
            top_k: Number of top agents to consider
            
        Returns:
            ADE: Average Displacement Error
            FDE: Final Displacement Error
        """
        if top_k is None:
            top_k = self._agents_len
        
        pred = pred_trajs[:, :top_k, :, :2]
        gt = gt_trajs[:, :top_k, :, :2]
        mask = (future_valid[:, :top_k] & (agents_interested[:, :top_k, None] > 0)).bool()
        
        # Displacement error
        disp_error = torch.norm(pred - gt, dim=-1)  # [B, A, T]
        
        # ADE
        ade = disp_error[mask].mean()
        
        # FDE (final timestep)
        fde = disp_error[..., -1][mask[..., -1]].mean()
        
        return ade.item(), fde.item()
    
    def training_step(self, batch, batch_idx):
        """
        Training step - same interface as VBD.
        """
        loss, log_dict = self.forward_and_get_loss(batch, prefix='train/')
        self.log_dict(
            log_dict,
            on_step=True, on_epoch=False, sync_dist=True,
            prog_bar=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - same interface as VBD.
        """
        loss, log_dict = self.forward_and_get_loss(batch, prefix='val/')
        self.log_dict(
            log_dict,
            on_step=False, on_epoch=True, sync_dist=True,
            prog_bar=True
        )
        return loss
    
    @torch.no_grad()
    def sample(self, batch, n_samples=1, temperature=None, return_global=True):
        """
        Generate trajectory samples - main inference interface.
        
        Args:
            batch: Input batch
            n_samples: Number of samples to generate
            temperature: Sampling temperature (default: self._temp)
            return_global: Whether to return trajectories in global frame
            
        Returns:
            samples: [B, n_samples, A, T, 3] generated trajectories (x, y, yaw)
        """
        if temperature is None:
            temperature = self._temp
        
        # Forward to get prepared inputs
        output_dict = self.forward(batch)
        inputs = output_dict['inputs']
        B, C, T, V = inputs['input'].shape
        
        all_samples = []
        for _ in range(n_samples):
            # Generate z samples from prior
            z_shapes = self._get_z_shapes(C, T, V)
            z_list = [torch.randn(B, *shape, device=self.device) * temperature 
                      for shape in z_shapes]
            
            # Reverse pass
            sample = self.glow.reverse(
                z_list,
                condition=None,  # MapGlow expects integer labels, we don't use it
                map_data=inputs['map_data'],
                map_mask=inputs['map_mask'],
                traffic_light_data=inputs['traffic_light_data'],
                traffic_light_mask=inputs['traffic_light_mask'],
                agent_types=inputs['agent_types'],
                agent_shape=inputs['agent_shape'],
                history_data=inputs['history_data'],
                history_timestep_mask=inputs['history_timestep_mask'],
                target_vehicle_mask=inputs['target_vehicle_mask'],
                history_vehicle_mask=inputs['history_vehicle_mask'],
                timestep_mask=inputs['timestep_mask']
            )
            
            # Unnormalize: [B, C, T, V]
            sample = self.unnormalize_trajectory(sample, use_delta_xy=self._use_delta_trajectory)
            if self._use_delta_trajectory:
                sample = self._delta_to_absolute_target(
                    sample,
                    timestep_mask=inputs['timestep_mask'],
                    anchor_xyyaw=inputs.get('target_anchor_xyyaw', None),
                    anchor_valid=inputs.get('target_anchor_valid', None),
                )
            
            # Convert to [B, A, T, 5] format
            sample = sample.permute(0, 3, 2, 1)  # [B, A, T, 5]

            # Mask invalid agents/timesteps to avoid unconstrained noisy tails.
            agent_valid = inputs['target_vehicle_mask'].bool().unsqueeze(-1)  # [B, A, 1]
            time_valid = inputs['timestep_mask'].permute(0, 2, 1).bool()      # [B, A, T]
            pred_valid = agent_valid & time_valid
            sample = sample.masked_fill(~pred_valid.unsqueeze(-1), 0.0)
            
            if return_global:
                # Transform back to global frame
                current_states = inputs['current_states']  # [B, A, 3]
                sample_global = self._transform_local_to_global(self._xyyaw_from_state5(sample), current_states)
                sample_global = sample_global.masked_fill(~pred_valid.unsqueeze(-1), 0.0)
                all_samples.append(sample_global)
            else:
                all_samples.append(self._xyyaw_from_state5(sample))
        
        # Stack samples: [B, n_samples, A, T, 3]
        samples = torch.stack(all_samples, dim=1)
        return samples
    
    def _transform_local_to_global(self, local_trajs, current_states):
        """
        Transform trajectories from local frame back to global frame.
        
        Args:
            local_trajs: [B, A, T, 3] (local_x, local_y, local_yaw)
            current_states: [B, A, 3] (global_x, global_y, global_yaw)
            
        Returns:
            global_trajs: [B, A, T, 3] (global_x, global_y, global_yaw)
        """
        return batch_transform_trajs_to_global_frame(local_trajs, current_states)
    
    def _get_z_shapes(self, n_channel, T, V):
        """
        Calculate z shapes for each block.
        """
        # Match MapGLow.Block.forward:
        # - each block squeezes time by 2
        # - split=True block emits z_new with C=n_channel
        # - split=False block emits z_new with C=2*n_channel
        z_shapes = []
        cur_t = T
        for block in self.glow.blocks:
            cur_t = cur_t // 2
            if block.split:
                z_shapes.append((n_channel, cur_t, V))
            else:
                z_shapes.append((n_channel * 2, cur_t, V))
        return z_shapes
    
    ################### Helper Functions (same as VBD) ##############
    def batch_to_device(self, input_dict: dict, device: torch.device = 'cuda'):
        """
        Move the tensors in the input dictionary to the specified device.
        Same as VBD's implementation.
        """
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.to(device)
        return input_dict
    
    def reset_agent_length(self, agents_len):
        """
        Reset the maximum number of agents.
        Useful for inference with different agent counts.
        """
        self._agents_len = agents_len
    
    @torch.no_grad()
    def reconstruct(self, batch):
        """
        Reconstruct trajectories by encoding and decoding.
        Useful for checking model quality.
        
        Returns:
            reconstructed: [B, A, T, 3] reconstructed trajectories
            original: [B, A, T, 3] original trajectories
        """
        output_dict = self.forward(batch)
        inputs = output_dict['inputs']
        z_outs = output_dict['z_outs']
        
        # Use actual z values for reconstruction (should be near-perfect)
        reconstructed = self.glow.reverse(
            z_outs,
            condition=None,  # MapGlow expects integer labels, we don't use it
            map_data=inputs['map_data'],
            map_mask=inputs['map_mask'],
            traffic_light_data=inputs['traffic_light_data'],
            traffic_light_mask=inputs['traffic_light_mask'],
            agent_types=inputs['agent_types'],
            agent_shape=inputs['agent_shape'],
            history_data=inputs['history_data'],
            history_timestep_mask=inputs['history_timestep_mask'],
            reconstruct=True,
            target_vehicle_mask=inputs['target_vehicle_mask'],
            history_vehicle_mask=inputs['history_vehicle_mask'],
            timestep_mask=inputs['timestep_mask']
        )
        
        reconstructed = self.unnormalize_trajectory(reconstructed, use_delta_xy=self._use_delta_trajectory)
        if self._use_delta_trajectory:
            reconstructed = self._delta_to_absolute_target(
                reconstructed,
                timestep_mask=inputs['timestep_mask'],
                anchor_xyyaw=inputs.get('target_anchor_xyyaw', None),
                anchor_valid=inputs.get('target_anchor_valid', None),
            )
        reconstructed = reconstructed.permute(0, 3, 2, 1)  # [B, A, T, 5]
        
        original = inputs['agents_future_local'][:, :, 1:, :3]
        
        return self._xyyaw_from_state5(reconstructed), original
    
    @torch.no_grad()
    def compute_nll(self, batch):
        """
        Compute negative log-likelihood for a batch.
        Useful for evaluation.
        
        Returns:
            nll: [B] negative log-likelihood per sample
            bpd: [B] bits per dimension per sample
        """
        output_dict = self.forward(batch)
        log_p = output_dict['log_p']
        logdet = output_dict['logdet']
        inputs = output_dict['inputs']
        
        nll = -(log_p + logdet)
        
        B, C, T, V = inputs['input'].shape
        n_pixel = C * T * V
        bpd = nll / (log(2) * n_pixel)
        
        return nll, bpd
    
