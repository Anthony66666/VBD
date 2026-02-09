"""
Flow Matching Model for Trajectory Prediction

Flow Matching learns to transport samples from a simple prior distribution (Gaussian noise)
to the data distribution via a learned velocity field.

Key formulas (using optimal transport / rectified flow):
- Forward process: x_t = (1 - t) * x_0 + t * x_1, where x_0 = data, x_1 = noise
- Velocity field: v(x_t, t) ≈ x_1 - x_0 = noise - data
- Sampling: dx/dt = v(x, t), integrate from t=0 (data) to t=1 (noise), then reverse

In this implementation:
- t=0 corresponds to clean data (x_0)
- t=1 corresponds to pure noise (x_1)
- Training: predict v_target = noise - data at interpolated point x_t
- Sampling: start from noise (t=1), integrate backward to data (t=0)
"""

import torch
import lightning.pytorch as pl
from torch.nn.functional import smooth_l1_loss

from .modules import Encoder, Denoiser, GoalPredictor
from .model_utils import inverse_kinematics, roll_out, batch_transform_trajs_to_global_frame
from torch.nn.functional import cross_entropy


class FlowMatching(pl.LightningModule):
    """
    Flow Matching model for multi-agent trajectory prediction.
    Uses VBD's encoder and action-based representation.
    
    The model learns a velocity field v(x_t, t) that transports noise to data.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self._future_len = cfg["future_len"]
        self._agents_len = cfg["agents_len"]
        self._action_len = cfg["action_len"]
        self._encoder_layers = cfg["encoder_layers"]
        self._encoder_version = cfg.get("encoder_version", "v2")
        self._action_mean = cfg["action_mean"]
        self._action_std = cfg["action_std"]
        self._embeding_dim = cfg.get("embeding_dim", 5)

        # Flow matching parameters
        self._flow_steps = cfg.get("flow_steps", cfg.get("diffusion_steps", 100))
        self._flow_t_min = cfg.get("flow_t_min", 1e-5)  # Avoid singularity at t=0
        self._flow_t_max = cfg.get("flow_t_max", 1.0)
        
        # Sigma for conditional flow matching (adds stochasticity, can be 0 for OT path)
        self._sigma = cfg.get("flow_sigma", 0.0)
        
        # Whether to use rollout in the flow network
        # rollout=True: converts actions to trajectory before encoding (like VBD diffusion)
        # rollout=False: directly encodes actions (more suitable for Flow Matching)
        self._use_rollout = cfg.get("flow_use_rollout", False)
        
        # Input dimension for flow network: 5 if rollout (trajectory), 2 if no rollout (action)
        flow_input_dim = cfg.get("embeding_dim", 5) if self._use_rollout else 2

        self._train_encoder = cfg.get("train_encoder", True)
        self._train_flow = cfg.get("train_flow", cfg.get("train_denoiser", True))
        self._train_predictor = cfg.get("train_predictor", True)
        self._with_predictor = cfg.get("with_predictor", True)

        # Scene encoder (shared with VBD)
        self.encoder = Encoder(self._encoder_layers, version=self._encoder_version)
        
        # Velocity field network (reuses Denoiser architecture)
        # Input: noised actions, time step
        # Output: predicted velocity v(x_t, t)
        self.flow_net = Denoiser(
            future_len=self._future_len,
            action_len=self._action_len,
            agents_len=self._agents_len,
            steps=self._flow_steps,
            input_dim=flow_input_dim,
        )

        if self._with_predictor:
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        self.register_buffer("action_mean", torch.tensor(self._action_mean))
        self.register_buffer("action_std", torch.tensor(self._action_std))

    ################### Training Setup ###################
    def configure_optimizers(self):
        if not self._train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self._train_flow:
            for param in self.flow_net.parameters():
                param.requires_grad = False
        if self._with_predictor and (not self._train_predictor):
            for param in self.predictor.parameters():
                param.requires_grad = False

        params_to_update = [p for p in self.parameters() if p.requires_grad]
        assert len(params_to_update) > 0, "No parameters to update"

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg["lr"],
            weight_decay=self.cfg.get("weight_decay", 0.01),
        )

        lr_warmup_step = self.cfg.get("lr_warmup_step", 1000)
        lr_step_freq = self.cfg.get("lr_step_freq", 1000)
        lr_step_gamma = self.cfg.get("lr_step_gamma", 0.98)

        def lr_update(step, warmup_step, step_size, gamma):
            if step < warmup_step:
                lr_scale = 1 - (warmup_step - step) / warmup_step * 0.95
            else:
                n = (step - warmup_step) // step_size
                lr_scale = gamma**n

            lr_scale = max(1e-2, min(1.0, lr_scale))
            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_update(step, lr_warmup_step, lr_step_freq, lr_step_gamma),
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, inputs, xt_actions, t_index, t_scalar=None):
        """
        Forward pass of Flow Matching model.

        Args:
            inputs: VBD batch inputs (scene context).
            xt_actions: Interpolated actions at time t [B, A, T, 2] (unnormalized).
            t_index: Discrete time index for embedding lookup [B, A].
            t_scalar: Continuous time in [0, 1] for optional x0 reconstruction [B, A, 1, 1].

        Returns:
            dict with 'flow_pred' (velocity prediction) and optionally 'x0_pred', 'denoised_trajs'
        """
        output_dict = {}
        encoder_outputs = self.encoder(inputs)

        if self._train_flow:
            flow_outputs = self.forward_flow(encoder_outputs, xt_actions, t_index, t_scalar)
            output_dict.update(flow_outputs)

        if self._train_predictor:
            predictor_outputs = self.forward_predictor(encoder_outputs)
            output_dict.update(predictor_outputs)

        return output_dict

    def forward_flow(self, encoder_outputs, xt_actions, t_index, t_scalar=None):
        """
        Forward pass through the velocity field network.
        
        Args:
            encoder_outputs: Scene encodings from the encoder.
            xt_actions: Interpolated actions at time t [B, A, T, 2] (unnormalized for Denoiser).
            t_index: Time step index for the embedding [B, A].
            t_scalar: Continuous time value for x0 reconstruction [B, A, 1, 1].
            
        Returns:
            dict with velocity prediction and optional reconstructed trajectory.
        """
        # Predict velocity field v(x_t, t)
        # Pass rollout parameter to control whether to convert actions to trajectory
        flow_pred = self.flow_net(encoder_outputs, xt_actions, t_index, rollout=self._use_rollout)
        outputs = {"flow_pred": flow_pred}

        # Optionally reconstruct x0 from x_t using the predicted velocity
        # In normalized space: x_t = (1-t)*x_0 + t*x_1, v = x_1 - x_0
        # => x_0 = x_t - t*v
        if t_scalar is not None:
            # Convert everything to normalized space for reconstruction
            xt_normalized = self.normalize_actions(xt_actions)
            # flow_pred is in unnormalized space, convert to normalized
            flow_pred_normalized = flow_pred / self.action_std
            
            # Reconstruct x0 in normalized space
            x0_pred_normalized = xt_normalized - t_scalar * flow_pred_normalized
            x0_pred = self.unnormalize_actions(x0_pred_normalized)
            
            current_states = encoder_outputs["agents"][:, : self._agents_len, -1]
            denoised_trajs = roll_out(
                current_states,
                x0_pred,
                action_len=self._action_len,
                global_frame=True,
            )
            outputs.update({
                "x0_pred": x0_pred,
                "x0_pred_normalized": x0_pred_normalized,
                "denoised_trajs": denoised_trajs,
            })

        return outputs

    def forward_predictor(self, encoder_outputs):
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)

        current_states = encoder_outputs["agents"][:, : self._agents_len, -1]
        goal_actions = self.unnormalize_actions(goal_actions_normalized)
        goal_trajs = roll_out(
            current_states[:, :, None, :],
            goal_actions,
            action_len=self.predictor._action_len,
            global_frame=True,
        )

        return {
            "goal_actions_normalized": goal_actions_normalized,
            "goal_actions": goal_actions,
            "goal_scores": goal_scores,
            "goal_trajs": goal_trajs,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        batch,
        num_steps: int = 50,
        t_start: float = 1.0,
        t_end: float = 0.0,
        method: str = "euler",
    ):
        """
        Sample actions via ODE integration of the learned velocity field.
        
        Flow matching defines: x_t = (1-t)*x_0 + t*x_1 (data to noise)
        To sample, we integrate: dx/dt = v(x, t) from t=1 (noise) to t=0 (data)
        
        Args:
            batch: VBD batch inputs containing scene context.
            num_steps: Number of integration steps (more = better quality but slower).
            t_start: Starting time (1.0 = pure noise).
            t_end: Ending time (0.0 = clean data).
            method: Integration method - "euler" or "midpoint".

        Returns:
            dict: {"actions": sampled actions, "trajs": rolled-out trajectories, "t_schedule": time steps}
        """
        encoder_outputs = self.encoder(batch)

        B = batch["agents_history"].shape[0]
        A = min(batch["agents_history"].shape[1], self._agents_len)
        T = self._future_len // self._action_len
        device = encoder_outputs["encodings"].device

        # Start from pure noise (t=1)
        x = torch.randn(B, A, T, 2, device=device)
        
        # Time schedule from t_start to t_end
        t_schedule = torch.linspace(t_start, t_end, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t_curr = t_schedule[i]
            t_next = t_schedule[i + 1]
            dt = t_next - t_curr  # Will be negative (going from 1 to 0)
            
            # Current point in action space (unnormalized)
            x_unnorm = self.unnormalize_actions(x)
            
            # Get time index for embedding
            # t_curr is a 0-dim tensor, convert to int then create proper tensor
            t_idx_val = int((t_curr.item() * (self._flow_steps - 1)))
            t_idx_val = max(0, min(self._flow_steps - 1, t_idx_val))
            t_index = torch.full((B, A), t_idx_val, dtype=torch.long, device=device)
            
            if method == "euler":
                # Euler step: x_{t+dt} = x_t + v(x_t, t) * dt
                flow_pred = self.flow_net(encoder_outputs, x_unnorm, t_index, rollout=self._use_rollout)
                # flow_pred is in action space, convert to normalized for update
                flow_pred_normalized = flow_pred / self.action_std
                x = x + flow_pred_normalized * dt
                
            elif method == "midpoint":
                # Midpoint method for better accuracy
                flow_pred = self.flow_net(encoder_outputs, x_unnorm, t_index, rollout=self._use_rollout)
                flow_pred_normalized = flow_pred / self.action_std
                
                # Half step
                x_mid = x + flow_pred_normalized * (dt / 2)
                t_mid = (t_curr + t_next) / 2
                t_idx_mid_val = int((t_mid.item() * (self._flow_steps - 1)))
                t_idx_mid_val = max(0, min(self._flow_steps - 1, t_idx_mid_val))
                t_index_mid = torch.full((B, A), t_idx_mid_val, dtype=torch.long, device=device)
                
                x_mid_unnorm = self.unnormalize_actions(x_mid)
                flow_pred_mid = self.flow_net(encoder_outputs, x_mid_unnorm, t_index_mid, rollout=self._use_rollout)
                flow_pred_mid_normalized = flow_pred_mid / self.action_std
                
                # Full step using midpoint velocity
                x = x + flow_pred_mid_normalized * dt
            else:
                raise ValueError(f"Unknown integration method: {method}")

        # Convert final normalized actions to trajectories
        x_final_unnorm = self.unnormalize_actions(x)
        current_states = encoder_outputs["agents"][:, : self._agents_len, -1]
        trajs = roll_out(current_states, x_final_unnorm, action_len=self._action_len, global_frame=True)

        return {
            "actions": x_final_unnorm,
            "actions_normalized": x,
            "trajs": trajs,
            "t_schedule": t_schedule,
        }

    @torch.no_grad()
    def sample_k_actions(
        self,
        batch,
        k: int = 6,
        num_steps: int = 50,
        t_start: float = 1.0,
        t_end: float = 0.0,
        method: str = "euler",
    ):
        """
        Sample K diverse trajectory proposals per agent.
        
        Args:
            batch: VBD batch inputs.
            k: Number of samples per agent.
            num_steps: Number of integration steps.
            t_start: Starting time (1.0 = noise).
            t_end: Ending time (0.0 = data).
            method: Integration method.
            
        Returns:
            dict: {"actions": [B, A, K, T, 2], "trajs": [B, A, K, T_future, 3]}
        """
        encoder_outputs = self.encoder(batch)
        
        B = batch["agents_history"].shape[0]
        A = min(batch["agents_history"].shape[1], self._agents_len)
        T = self._future_len // self._action_len
        device = encoder_outputs["encodings"].device
        
        # Start from K different noise samples per agent
        x = torch.randn(B, A, k, T, 2, device=device)  # [B, A, K, T, 2]
        
        # Reshape to process all K samples together: [B*K, A, T, 2]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * k, A, T, 2)
        
        # Expand encoder outputs for K samples
        # Note: agents_type has shape [B*A] after encoder processing, needs special handling
        encoder_outputs_expanded = {}
        for key, val in encoder_outputs.items():
            if isinstance(val, torch.Tensor):
                if val.dim() == 1:
                    # 1D tensor like agents_type [B*A] -> repeat k times -> [B*K*A]
                    # Reshape to [B, A], expand to [B, K, A], then flatten to [B*K*A]
                    val_reshaped = val.reshape(B, -1)  # [B, A] (or similar)
                    expanded = val_reshaped.unsqueeze(1).expand(-1, k, -1)  # [B, K, A]
                    encoder_outputs_expanded[key] = expanded.reshape(-1)  # [B*K*A]
                else:
                    # Regular batched tensor [B, ...] -> [B*K, ...]
                    expanded = val.unsqueeze(1).expand(-1, k, *val.shape[1:])
                    encoder_outputs_expanded[key] = expanded.reshape(B * k, *val.shape[1:])
            else:
                encoder_outputs_expanded[key] = val
        
        # Time schedule
        t_schedule = torch.linspace(t_start, t_end, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t_curr = t_schedule[i]
            t_next = t_schedule[i + 1]
            dt = t_next - t_curr
            
            x_unnorm = self.unnormalize_actions(x)
            # t_curr is a 0-dim tensor, convert to int then create proper tensor
            t_idx_val = int((t_curr.item() * (self._flow_steps - 1)))
            t_idx_val = max(0, min(self._flow_steps - 1, t_idx_val))
            t_index = torch.full((B * k, A), t_idx_val, dtype=torch.long, device=device)
            
            flow_pred = self.flow_net(encoder_outputs_expanded, x_unnorm, t_index, rollout=self._use_rollout)
            flow_pred_normalized = flow_pred / self.action_std
            x = x + flow_pred_normalized * dt
        
        # Reshape back: [B*K, A, T, 2] -> [B, K, A, T, 2] -> [B, A, K, T, 2]
        x = x.reshape(B, k, A, T, 2).permute(0, 2, 1, 3, 4)
        x_unnorm = self.unnormalize_actions(x)
        
        # Roll out trajectories
        current_states = encoder_outputs["agents"][:, :A, -1]  # [B, A, state_dim]
        current_states_expanded = current_states.unsqueeze(2).expand(-1, -1, k, -1)  # [B, A, K, state_dim]
        
        # Flatten for roll_out: [B*A*K, 1, T, 2]
        x_flat = x_unnorm.reshape(B * A * k, T, 2).unsqueeze(1)
        states_flat = current_states_expanded.reshape(B * A * k, 1, -1)
        
        trajs_flat = roll_out(states_flat, x_flat, action_len=self._action_len, global_frame=True)
        trajs = trajs_flat.reshape(B, A, k, -1, trajs_flat.shape[-1])  # [B, A, K, T_future, 3]
        
        return {
            "actions": x_unnorm,  # [B, A, K, T, 2]
            "actions_normalized": x,
            "trajs": trajs,  # [B, A, K, T_future, 3]
        }

    def forward_and_get_loss(self, batch, prefix="", debug=False):
        """
        Compute Flow Matching loss and auxiliary losses.
        
        Flow Matching objective (Rectified Flow / OT path):
        - Sample x_0 (data), x_1 (noise) in NORMALIZED space
        - Interpolate: x_t = (1 - t) * x_0 + t * x_1
        - Target velocity: v_target = x_1 - x_0 (constant along the path)
        - Loss: ||v_pred(x_t, t) - v_target||^2
        
        Key insight: We work in normalized space for stability, but Denoiser
        expects unnormalized actions, so we convert at the interface.
        """
        agents_future = batch["agents_future"][:, : self._agents_len]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_interested = batch["agents_interested"][:, : self._agents_len]
        anchors = batch["anchors"][:, : self._agents_len]

        # Get ground truth actions via inverse kinematics
        gt_actions, gt_actions_valid = inverse_kinematics(
            agents_future,
            agents_future_valid,
            dt=0.1,
            action_len=self._action_len,
        )

        # x_0: Ground truth actions in NORMALIZED space
        x0 = self.normalize_actions(gt_actions)  # [B, A, T, 2]
        B, A, T, D = x0.shape

        # x_1: Standard Gaussian noise (same space as normalized x0)
        x1 = torch.randn_like(x0)

        # Sample time t uniformly in [t_min, t_max]
        t = torch.rand(B, A, 1, 1, device=x0.device)
        t = t * (self._flow_t_max - self._flow_t_min) + self._flow_t_min
        
        # Time index for embedding lookup
        t_index = (t.squeeze(-1).squeeze(-1) * (self._flow_steps - 1)).long()
        t_index = t_index.clamp(0, self._flow_steps - 1)

        # Interpolate in normalized space: x_t = (1 - t) * x_0 + t * x_1
        xt_normalized = (1 - t) * x0 + t * x1
        
        # Add small noise for conditional flow matching (optional)
        if self._sigma > 0:
            xt_normalized = xt_normalized + self._sigma * torch.randn_like(xt_normalized) * torch.sqrt(t * (1 - t))

        # Convert to unnormalized space for Denoiser input
        xt_actions = self.unnormalize_actions(xt_normalized)

        # Target velocity in NORMALIZED space: v = x_1 - x_0
        # Since Denoiser outputs in unnormalized space, we need v_target in unnormalized space too
        v_target_normalized = x1 - x0
        v_target = v_target_normalized * self.action_std  # Convert to unnormalized/action space

        log_dict = {}
        debug_outputs = {}
        total_loss = 0.0

        encoder_outputs = self.encoder(batch)

        if self._train_flow:
            flow_outputs = self.forward_flow(encoder_outputs, xt_actions, t_index, t_scalar=t)
            flow_pred = flow_outputs["flow_pred"]

            # Flow matching loss: ||v_pred - v_target||^2
            flow_loss = self.flow_loss(flow_pred, v_target, gt_actions_valid, agents_interested)
            total_loss = total_loss + flow_loss

            # Auxiliary loss on reconstructed trajectory
            denoised_trajs = flow_outputs["denoised_trajs"]
            state_loss_mean, yaw_loss_mean = self.denoise_loss(
                denoised_trajs,
                agents_future,
                agents_future_valid,
                agents_interested,
            )

            denoise_ade, denoise_fde = self.calculate_metrics_denoise(
                denoised_trajs,
                agents_future,
                agents_future_valid,
                agents_interested,
                8,
            )

            log_dict.update({
                prefix + "flow_loss": flow_loss.item(),
                prefix + "state_loss": state_loss_mean.item(),
                prefix + "yaw_loss": yaw_loss_mean.item(),
                prefix + "denoise_ADE": denoise_ade,
                prefix + "denoise_FDE": denoise_fde,
            })

            debug_outputs.update(flow_outputs)

        if self._train_predictor:
            goal_outputs = self.forward_predictor(encoder_outputs)
            debug_outputs.update(goal_outputs)

            goal_scores = goal_outputs["goal_scores"]
            goal_trajs = goal_outputs["goal_trajs"]

            goal_loss_mean, score_loss_mean = self.goal_loss(
                goal_trajs,
                goal_scores,
                agents_future,
                agents_future_valid,
                anchors,
                agents_interested,
            )

            pred_loss = goal_loss_mean + 0.05 * score_loss_mean
            total_loss = total_loss + pred_loss

            pred_ade, pred_fde = self.calculate_metrics_predict(
                goal_trajs,
                agents_future,
                agents_future_valid,
                agents_interested,
                8,
            )

            log_dict.update(
                {
                    prefix + "goal_loss": goal_loss_mean.item(),
                    prefix + "score_loss": score_loss_mean.item(),
                    prefix + "pred_ADE": pred_ade,
                    prefix + "pred_FDE": pred_fde,
                }
            )

        log_dict[prefix + "loss"] = float(total_loss)

        if debug:
            return total_loss, log_dict, debug_outputs
        return total_loss, log_dict

    def training_step(self, batch, batch_idx):
        loss, log_dict = self.forward_and_get_loss(batch, prefix="train/")
        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.forward_and_get_loss(batch, prefix="val/")
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    ################### Loss functions ###################
    def flow_loss(self, flow_pred, flow_target, actions_valid, agents_interested):
        """
        Compute flow matching loss: ||v_pred - v_target||^2
        
        Args:
            flow_pred: Predicted velocity [B, A, T, 2]
            flow_target: Target velocity [B, A, T, 2]
            actions_valid: Mask for valid actions [B, A, T]
            agents_interested: Mask for agents of interest [B, A]
        """
        # Create mask: valid actions for interested agents
        action_mask = actions_valid * (agents_interested[..., None] > 0)
        
        # Smooth L1 loss (Huber loss) - more robust than MSE
        loss = smooth_l1_loss(flow_pred, flow_target, reduction="none", beta=1.0)
        loss = loss.sum(-1)  # Sum over action dimensions
        loss = loss * action_mask
        
        # Average over valid entries
        denom = action_mask.sum().clamp(min=1.0)
        return loss.sum() / denom
    
    def flow_loss_mse(self, flow_pred, flow_target, actions_valid, agents_interested):
        """
        MSE version of flow loss (alternative to smooth L1).
        """
        action_mask = actions_valid * (agents_interested[..., None] > 0)
        loss = ((flow_pred - flow_target) ** 2).sum(-1)
        loss = loss * action_mask
        denom = action_mask.sum().clamp(min=1.0)
        return loss.sum() / denom

    def denoise_loss(self, denoised_trajs, agents_future, agents_future_valid, agents_interested):
        agents_future = agents_future[..., 1:, :3]
        future_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)

        state_loss = smooth_l1_loss(denoised_trajs[..., :2], agents_future[..., :2], reduction="none").sum(-1)
        yaw_error = denoised_trajs[..., 2] - agents_future[..., 2]
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        yaw_loss = torch.abs(yaw_error)

        state_loss = state_loss * future_mask
        yaw_loss = yaw_loss * future_mask

        denom = future_mask.sum().clamp(min=1.0)
        state_loss_mean = state_loss.sum() / denom
        yaw_loss_mean = yaw_loss.sum() / denom

        return state_loss_mean, yaw_loss_mean

    def goal_loss(self, trajs, scores, agents_future, agents_future_valid, anchors, agents_interested):
        current_states = agents_future[:, :, 0, :3]
        anchors_global = batch_transform_trajs_to_global_frame(anchors, current_states)
        num_batch, num_agents, num_query, _ = anchors_global.shape

        traj_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)

        goal_gt = agents_future[:, :, -1:, :2].flatten(0, 1)
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1)
        trajs = trajs.flatten(0, 1)[..., :3]
        anchors_global = anchors_global.flatten(0, 1)

        idx_anchor = torch.argmin(torch.norm(anchors_global - goal_gt, dim=-1), dim=-1)

        dist = torch.norm(trajs[:, :, :, :2] - trajs_gt[:, None, :, :2], dim=-1)
        dist = dist * traj_mask.flatten(0, 1)[:, None, :]
        idx = torch.argmin(dist.mean(-1), dim=-1)

        idx = torch.where(agents_future_valid[..., -1].flatten(0, 1), idx_anchor, idx)
        trajs_select = trajs[torch.arange(num_batch * num_agents), idx]

        traj_loss = smooth_l1_loss(trajs_select, trajs_gt, reduction="none").sum(-1)
        traj_loss = traj_loss * traj_mask.flatten(0, 1)

        scores = scores.flatten(0, 1)
        score_loss = cross_entropy(scores, idx, reduction="none")
        score_loss = score_loss * (agents_interested.flatten(0, 1) > 0)

        traj_loss_mean = traj_loss.sum() / traj_mask.sum().clamp(min=1.0)
        score_loss_mean = score_loss.sum() / (agents_interested > 0).sum().clamp(min=1.0)

        return traj_loss_mean, score_loss_mean

    @torch.no_grad()
    def calculate_metrics_denoise(self, denoised_trajs, agents_future, agents_future_valid, agents_interested, top_k=None):
        if not top_k:
            top_k = self._agents_len

        pred_traj = denoised_trajs[:, :top_k, :, :2]
        gt = agents_future[:, :top_k, 1:, :2]
        gt_mask = (
            agents_future_valid[:, :top_k, 1:] & (agents_interested[:, :top_k, None] > 0)
        ).bool()

        denoise_mse = torch.norm(pred_traj - gt, dim=-1)
        denoise_ade = denoise_mse[gt_mask].mean()
        denoise_fde = denoise_mse[..., -1][gt_mask[..., -1]].mean()

        return denoise_ade.item(), denoise_fde.item()

    @torch.no_grad()
    def calculate_metrics_predict(self, goal_trajs, agents_future, agents_future_valid, agents_interested, top_k=None):
        if not top_k:
            top_k = self._agents_len
        goal_trajs = goal_trajs[:, :top_k, :, :, :2]
        gt = agents_future[:, :top_k, 1:, :2]
        gt_mask = (
            agents_future_valid[:, :top_k, 1:] & (agents_interested[:, :top_k, None] > 0)
        ).bool()

        goal_mse = torch.norm(goal_trajs - gt[:, :, None, :, :], dim=-1)
        goal_mse = goal_mse * gt_mask[..., None, :]
        best_idx = torch.argmin(goal_mse.sum(-1), dim=-1)

        best_goal_mse = goal_mse[
            torch.arange(goal_mse.shape[0])[:, None],
            torch.arange(goal_mse.shape[1])[None, :],
            best_idx,
        ]

        goal_ade = best_goal_mse.sum() / gt_mask.sum().clamp(min=1.0)
        goal_fde = best_goal_mse[..., -1].sum() / gt_mask[..., -1].sum().clamp(min=1.0)

        return goal_ade.item(), goal_fde.item()

    ################### Helper Functions ###################
    def normalize_actions(self, actions: torch.Tensor):
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions: torch.Tensor):
        return actions * self.action_std + self.action_mean
