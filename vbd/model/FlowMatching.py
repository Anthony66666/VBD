"""
Flow Matching Model for Trajectory Prediction (Trajectory Space)
================================================================

This module implements Flow Matching (Rectified Flow / OT-Flow) for multi-agent
trajectory prediction, operating in TRAJECTORY SPACE following VBD's design.

Key Insight from VBD Paper
--------------------------
The VBD denoiser is designed to process TRAJECTORIES (x, y, θ, vx, vy) because:
1. Trajectories provide rich spatial/geometric information
2. The causal attention can reason about agent interactions in space
3. The encoder outputs are scene-centric, best matched with spatial inputs

Therefore, we do Flow Matching in TRAJECTORY SPACE (5D), not action space (2D):
- x_0: Ground truth trajectory [B, A, 80, 5] in local frame
- x_1: Noise trajectory [B, A, 80, 5]
- x_t: Interpolated trajectory
- v: Velocity field in trajectory space

Mathematical Foundation
-----------------------
Flow Matching learns a velocity field v(x, t) that transports samples from
noise distribution to data distribution along straight-line paths.

Forward Process (interpolation):
    x_t = (1 - t) * x_0 + t * x_1
    
Velocity (the derivative of x_t w.r.t. t):
    v = dx_t/dt = x_1 - x_0
    
Training Objective (in trajectory space):
    L = E_{t, x_0, x_1} || v_θ(x_t, t) - (x_1 - x_0) ||²
    
Sampling (reverse ODE from t=1 to t=0):
    dx/dt = v_θ(x, t)
    Integrate from x_1 ~ N(0, σ) at t=1 to get x_0 at t=0
    Then convert trajectory to actions via inverse kinematics

Time Convention
---------------
- t=0 corresponds to data (x_0, clean trajectory)
- t=1 corresponds to noise (x_1)
- Sampling integrates from t=1 to t=0
"""

import torch
import lightning.pytorch as pl
from torch.nn.functional import smooth_l1_loss, cross_entropy

from .modules import Encoder, Denoiser, GoalPredictor
from .model_utils import (
    inverse_kinematics, roll_out, 
    batch_transform_trajs_to_local_frame,
    batch_transform_trajs_to_global_frame
)


class FlowMatching(pl.LightningModule):
    """
    Flow Matching model for multi-agent trajectory prediction.
    
    Architecture Overview (Trajectory Space):
    ┌─────────────┐     ┌──────────────┐     ┌───────────────┐
    │   Encoder   │────►│  Flow Net    │────►│  Velocity v   │
    │  (Scene)    │     │ (Denoiser)   │     │  [B,A,80,5]   │
    └─────────────┘     └──────────────┘     └───────────────┘
                              ▲
                              │
                    ┌─────────┴─────────┐
                    │ x_t (noised traj) │
                    │  t (time step)    │
                    └───────────────────┘
    
    Key Design: Flow Matching in TRAJECTORY Space (5D)
    --------------------------------------------------
    Unlike action-space flow matching, we operate directly on trajectories:
    - Input: [x, y, θ, vx, vy] in local frame (agent-centric)
    - This matches VBD's denoiser design and provides spatial understanding
    - After sampling, convert back to actions via inverse kinematics
    
    Data Flow:
    1. Encoder processes scene context (history, map, traffic lights)
    2. During training: 
       - Get GT trajectory x_0 (local frame)
       - Sample t ~ U[0,1], noise x_1 ~ N(0, σ)
       - Compute x_t = (1-t)*x_0 + t*x_1
       - Predict v_θ(x_t, t), supervise with (x_1 - x_0)
    3. During sampling:
       - Start from x_1 ~ N(0, σ) in trajectory space
       - Integrate ODE: dx/dt = v_θ(x, t) from t=1 to t=0
       - Get trajectory x_0, convert to actions via inverse kinematics
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        
        # Core dimensions
        self._future_len = cfg["future_len"]       # 80 time steps
        self._agents_len = cfg["agents_len"]       # 32 agents max
        self._action_len = cfg["action_len"]       # 5 steps per action group
        self._num_actions = self._future_len // self._action_len  # 16 action groups
        
        # Encoder config
        self._encoder_layers = cfg["encoder_layers"]
        self._encoder_version = cfg.get("encoder_version", "v2")
        
        # Trajectory normalization parameters (for [x, y, θ, vx, vy])
        # These scale the trajectory dimensions to roughly similar magnitudes
        # Default values based on typical driving scenarios
        self._traj_std = cfg.get("traj_std", [10.0, 10.0, 1.0, 5.0, 5.0])
        
        # Action normalization (still needed for predictor and compatibility)
        self._action_mean = cfg["action_mean"]  # [0.0, 0.0]
        self._action_std = cfg["action_std"]    # [3.5, 1.6]
        
        # Flow matching hyperparameters
        self._flow_steps = cfg.get("flow_steps", 100)  # Number of time embeddings
        
        # Training flags
        self._train_encoder = cfg.get("train_encoder", True)
        self._train_flow = cfg.get("train_flow", True)
        self._train_predictor = cfg.get("train_predictor", True)
        self._with_predictor = cfg.get("with_predictor", True)

        # ============ Network Components ============
        
        # Scene encoder (shared with VBD)
        self.encoder = Encoder(self._encoder_layers, version=self._encoder_version)
        
        # Velocity field network (Denoiser)
        # CRITICAL: Use input_dim=5 for trajectory space (x, y, θ, vx, vy)
        # The Denoiser will process trajectories directly (not actions)
        # output_dim=5 because velocity field is also in trajectory space
        self.flow_net = Denoiser(
            future_len=self._future_len,
            action_len=self._action_len,
            agents_len=self._agents_len,
            steps=self._flow_steps,
            input_dim=5,    # TRAJECTORY space: [x, y, θ, vx, vy]
            output_dim=5,   # Velocity field in trajectory space
            causal=True,    # Multi-agent causal interaction
        )

        # Optional goal predictor (for guided generation)
        if self._with_predictor:
            self.predictor = GoalPredictor(
                future_len=self._future_len,
                agents_len=self._agents_len,
                action_len=self._action_len,
            )
        else:
            self.predictor = None
            self._train_predictor = False

        # Register normalization buffers
        self.register_buffer("action_mean", torch.tensor(self._action_mean))
        self.register_buffer("action_std", torch.tensor(self._action_std))
        self.register_buffer("traj_std", torch.tensor(self._traj_std))

    # =====================================================================
    # Optimizer Configuration
    # =====================================================================
    
    def configure_optimizers(self):
        # Freeze components if not training
        if not self._train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        if not self._train_flow:
            for param in self.flow_net.parameters():
                param.requires_grad = False
        if self._with_predictor and not self._train_predictor:
            for param in self.predictor.parameters():
                param.requires_grad = False

        params_to_update = [p for p in self.parameters() if p.requires_grad]
        assert len(params_to_update) > 0, "No parameters to update"

        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.cfg["lr"],
            weight_decay=self.cfg.get("weight_decay", 0.01),
        )

        # Learning rate schedule with warmup
        lr_warmup_step = self.cfg.get("lr_warmup_step", 1000)
        lr_step_freq = self.cfg.get("lr_step_freq", 1000)
        lr_step_gamma = self.cfg.get("lr_step_gamma", 0.98)

        def lr_lambda(step):
            if step < lr_warmup_step:
                # Linear warmup
                return 0.05 + 0.95 * step / lr_warmup_step
            else:
                # Exponential decay
                n = (step - lr_warmup_step) // lr_step_freq
                return max(0.01, lr_step_gamma ** n)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # =====================================================================
    # Normalization Utilities
    # =====================================================================
    
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to have similar scale across dimensions."""
        return (actions - self.action_mean) / self.action_std

    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions back to physical units."""
        return actions * self.action_std + self.action_mean

    def normalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """
        Normalize trajectory for flow matching.
        traj: [..., 5] with (x, y, θ, vx, vy) in LOCAL frame
        """
        return traj / self.traj_std

    def unnormalize_trajectory(self, traj: torch.Tensor) -> torch.Tensor:
        """Unnormalize trajectory."""
        return traj * self.traj_std

    # =====================================================================
    # Core Flow Matching: Velocity Prediction (Trajectory Space)
    # =====================================================================
    
    def predict_velocity(
        self, 
        encoder_outputs: dict, 
        x_t: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field v_θ(x_t, t) in TRAJECTORY space.
        
        Args:
            encoder_outputs: Scene encodings from Encoder
            x_t: Noised trajectory [B, A, 80, 5] in LOCAL frame (NOT normalized)
            t: Time values [B, A] in range [0, 1]
            
        Returns:
            v: Predicted velocity [B, A, 80, 5] in NORMALIZED trajectory space
        """
        # Convert continuous time to discrete step index for embedding
        t_index = (t * (self._flow_steps - 1)).long().clamp(0, self._flow_steps - 1)
        
        # Forward through denoiser with trajectory input
        # The denoiser expects trajectories in local frame
        # Note: We pass the trajectory directly (input_dim=5)
        # The denoiser's forward will detect input_dim=5 and NOT use rollout
        v_compressed = self.flow_net(
            encoder_outputs,
            x_t,  # Trajectory [B, A, 80, 5] in local frame
            t_index,
            rollout=False,  # x_t is already a trajectory, no need to rollout
        )
        # v_compressed: [B, A, 16, 5] due to max pooling in TransformerDecoder
        
        # Expand velocity from 16 steps back to 80 steps
        # Each of the 16 velocity values applies to 5 consecutive time steps
        v = v_compressed.repeat_interleave(self._action_len, dim=2)  # [B, A, 80, 5]
        
        return v

    # =====================================================================
    # Training: Loss Computation
    # =====================================================================
    
    def forward_and_get_loss(self, batch, prefix="", debug=False):
        """
        Compute Flow Matching loss in TRAJECTORY space.
        
        Flow Matching Training Algorithm (Trajectory Space):
        1. Get ground truth trajectory x_0 (in local frame)
        2. Normalize: x_0_norm = x_0 / traj_std
        3. Sample noise x_1 ~ N(0, I) (same shape as normalized trajectory)
        4. Sample time t ~ U[0, 1]
        5. Compute x_t_norm = (1-t) * x_0_norm + t * x_1
        6. Unnormalize for network: x_t = x_t_norm * traj_std
        7. Predict v = v_θ(x_t, t)
        8. Loss = ||v - (x_1 - x_0_norm)||² (in normalized space)
        """
        # ============ Data Preparation ============
        agents_future = batch["agents_future"][:, :self._agents_len]  # [B, A, 81, 8]
        agents_future_valid = torch.ne(agents_future.sum(-1), 0)
        agents_interested = batch["agents_interested"][:, :self._agents_len]
        anchors = batch["anchors"][:, :self._agents_len]

        B, A = agents_future.shape[:2]
        device = agents_future.device

        # ============ Encode Scene ============
        encoder_outputs = self.encoder(batch)
        current_states = encoder_outputs["agents"][:, :self._agents_len, -1, :5]  # [B, A, 5]

        # ============ Get Ground Truth Trajectory in Local Frame ============
        # agents_future: [B, A, 81, 8] includes current frame (t=0)
        # We need future trajectory: [B, A, 80, 5] (frames 1-80)
        gt_traj_global = agents_future[:, :, 1:, :5]  # [B, A, 80, 5] - (x, y, θ, vx, vy)
        
        # Transform to local frame (agent-centric at t=0)
        # First, add dummy dim and use batch_transform_trajs_to_local_frame
        gt_traj_with_current = agents_future[:, :, :, :5]  # [B, A, 81, 5] including current
        gt_traj_local = batch_transform_trajs_to_local_frame(gt_traj_with_current, ref_idx=0)
        gt_traj_local = gt_traj_local[:, :, 1:, :]  # [B, A, 80, 5] remove current frame
        
        # Trajectory validity mask
        traj_valid = agents_future_valid[:, :, 1:]  # [B, A, 80]
        traj_mask = traj_valid & (agents_interested[..., None] > 0)  # [B, A, 80]

        log_dict = {}
        debug_outputs = {}
        total_loss = 0.0

        # ============ Flow Matching Loss (Trajectory Space) ============
        if self._train_flow:
            # Step 1: Normalize ground truth trajectory (x_0)
            x_0 = self.normalize_trajectory(gt_traj_local)  # [B, A, 80, 5]
            
            # Step 2: Sample noise (x_1) in normalized trajectory space
            x_1 = torch.randn_like(x_0)  # [B, A, 80, 5]
            
            # Step 3: Sample time t ~ U[0, 1]
            t = torch.rand(B, A, 1, 1, device=device)  # [B, A, 1, 1]
            
            # Step 4: Interpolate in normalized space
            # x_t_norm = (1-t) * x_0 + t * x_1
            x_t_norm = (1 - t) * x_0 + t * x_1  # [B, A, 80, 5]
            
            # Step 5: Compute target velocity in normalized space
            # v_target = dx_t/dt = x_1 - x_0
            v_target = x_1 - x_0  # [B, A, 80, 5]
            
            # Step 6: Unnormalize trajectory for network input
            # The network expects trajectories in physical units (local frame)
            x_t = self.unnormalize_trajectory(x_t_norm)  # [B, A, 80, 5]
            
            # Step 7: Predict velocity
            t_flat = t.squeeze(-1).squeeze(-1)  # [B, A]
            v_pred = self.predict_velocity(encoder_outputs, x_t, t_flat)  # [B, A, 80, 5]
            
            # Step 8: Compute loss in normalized trajectory space (MSE)
            flow_loss = self._compute_flow_loss(v_pred, v_target, traj_mask)
            
            total_loss = total_loss + flow_loss
            log_dict[prefix + "flow_loss"] = flow_loss.item()
            
            # ============ Auxiliary Metrics ============
            # Compute denoised trajectory for metrics
            # Reconstruct x_0_pred from x_t_norm and predicted velocity:
            # x_0_pred = x_t_norm - t * v_pred (since x_t = x_0 + t*v)
            x_0_pred_norm = x_t_norm - t * v_pred  # [B, A, 80, 5]
            x_0_pred_local = self.unnormalize_trajectory(x_0_pred_norm)  # [B, A, 80, 5]
            
            # Transform back to global frame for metrics
            denoised_trajs = self._local_to_global_trajectory(x_0_pred_local, current_states)
            
            denoise_ade, denoise_fde = self._compute_trajectory_metrics(
                denoised_trajs,
                agents_future,
                agents_future_valid,
                agents_interested,
            )
            
            log_dict[prefix + "denoise_ADE"] = denoise_ade
            log_dict[prefix + "denoise_FDE"] = denoise_fde
            
            debug_outputs["x_0"] = x_0
            debug_outputs["x_1"] = x_1
            debug_outputs["x_t"] = x_t
            debug_outputs["v_pred"] = v_pred
            debug_outputs["v_target"] = v_target
            debug_outputs["denoised_trajs"] = denoised_trajs

        # ============ Predictor Loss (Optional) ============
        if self._train_predictor and self.predictor is not None:
            pred_outputs = self._forward_predictor(encoder_outputs)
            
            goal_trajs = pred_outputs["goal_trajs"]
            goal_scores = pred_outputs["goal_scores"]
            
            goal_loss, score_loss = self._compute_goal_loss(
                goal_trajs, goal_scores,
                agents_future, agents_future_valid,
                anchors, agents_interested,
            )
            
            pred_loss = goal_loss + 0.05 * score_loss
            total_loss = total_loss + pred_loss
            
            pred_ade, pred_fde = self._compute_predictor_metrics(
                goal_trajs, agents_future,
                agents_future_valid, agents_interested,
            )
            
            log_dict[prefix + "goal_loss"] = goal_loss.item()
            log_dict[prefix + "score_loss"] = score_loss.item()
            log_dict[prefix + "pred_ADE"] = pred_ade
            log_dict[prefix + "pred_FDE"] = pred_fde
            
            debug_outputs.update(pred_outputs)

        log_dict[prefix + "loss"] = float(total_loss)

        if debug:
            return total_loss, log_dict, debug_outputs
        return total_loss, log_dict

    def _local_to_global_trajectory(
        self, 
        traj_local: torch.Tensor, 
        current_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform trajectory from local frame back to global frame.
        
        Args:
            traj_local: [B, A, T, 5] trajectory in local frame
            current_states: [B, A, 5] current agent states (x, y, θ, vx, vy)
            
        Returns:
            traj_global: [B, A, T, 5] trajectory in global frame
        """
        # Get current position and heading
        x0 = current_states[..., 0:1]  # [B, A, 1]
        y0 = current_states[..., 1:2]  # [B, A, 1]
        theta0 = current_states[..., 2:3]  # [B, A, 1]
        
        # Extract local trajectory components
        x_local = traj_local[..., 0]  # [B, A, T]
        y_local = traj_local[..., 1]  # [B, A, T]
        theta_local = traj_local[..., 2]  # [B, A, T]
        vx_local = traj_local[..., 3]  # [B, A, T]
        vy_local = traj_local[..., 4]  # [B, A, T]
        
        # Rotate back to global frame
        cos_theta = torch.cos(theta0)
        sin_theta = torch.sin(theta0)
        
        x_global = x_local * cos_theta - y_local * sin_theta + x0
        y_global = x_local * sin_theta + y_local * cos_theta + y0
        theta_global = theta_local + theta0
        
        # Rotate velocities back
        vx_global = vx_local * cos_theta - vy_local * sin_theta
        vy_global = vx_local * sin_theta + vy_local * cos_theta
        
        return torch.stack([x_global, y_global, theta_global, vx_global, vy_global], dim=-1)

    def _compute_flow_loss(
        self, 
        v_pred: torch.Tensor, 
        v_target: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Flow Matching loss: MSE between predicted and target velocity.
        
        Both v_pred and v_target are in NORMALIZED trajectory space, so all
        dimensions (x, y, θ, vx, vy) contribute roughly equally to the loss.
        
        Args:
            v_pred: [B, A, 80, 5] predicted velocity in normalized trajectory space
            v_target: [B, A, 80, 5] target velocity in normalized trajectory space
            mask: [B, A, 80] validity mask
        """
        # MSE loss per element
        loss = (v_pred - v_target) ** 2  # [B, A, 80, 5]
        loss = loss.sum(dim=-1)  # [B, A, 80]
        
        # Apply mask (only compute loss for valid timesteps of interested agents)
        loss = loss * mask.float()
        
        # Mean over valid elements
        return loss.sum() / mask.sum().clamp(min=1.0)

    # =====================================================================
    # Training/Validation Steps
    # =====================================================================
    
    def training_step(self, batch, batch_idx):
        loss, log_dict = self.forward_and_get_loss(batch, prefix="train/")
        self.log_dict(log_dict, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, log_dict = self.forward_and_get_loss(batch, prefix="val/")
        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    # =====================================================================
    # Sampling: ODE Integration (Trajectory Space)
    # =====================================================================
    
    @torch.no_grad()
    def sample_actions(
        self,
        batch,
        num_steps: int = 50,
        method: str = "euler",
    ) -> dict:
        """
        Sample trajectories by integrating the learned velocity field.
        
        ODE: dx/dt = v_θ(x, t) in trajectory space
        Integrate from t=1 (noise) to t=0 (data)
        
        Args:
            batch: Input batch with scene context
            num_steps: Number of integration steps
            method: Integration method ('euler' or 'midpoint')
            
        Returns:
            Dictionary containing:
            - actions: Sampled actions [B, A, 16, 2] (via inverse kinematics)
            - trajs: Sampled trajectories [B, A, 80, 5] in global frame
        """
        # Encode scene
        encoder_outputs = self.encoder(batch)
        
        B = batch["agents_history"].shape[0]
        A = min(batch["agents_history"].shape[1], self._agents_len)
        device = encoder_outputs["encodings"].device
        
        # Get current states [B, A, 5] - only take first 5 dims (x, y, theta, vx, vy)
        current_states = encoder_outputs["agents"][:, :A, -1, :5]  # [B, A, 5]
        
        # Start from noise in NORMALIZED trajectory space at t=1
        # Shape: [B, A, 80, 5]
        # Note: This represents frames 1-80 in local frame (frame 0 is the current state)
        x = torch.randn(B, A, self._future_len, 5, device=device)
        
        # Time schedule: 1.0 -> 0.0 in num_steps
        dt = -1.0 / num_steps
        
        for step in range(num_steps):
            t_current = 1.0 - step / num_steps
            t = torch.full((B, A), t_current, device=device)
            
            # x is in normalized space, unnormalize for network
            x_unnorm = self.unnormalize_trajectory(x)  # [B, A, 80, 5] in local frame
            
            if method == "euler":
                v = self.predict_velocity(encoder_outputs, x_unnorm, t)
                x = x + v * dt
                
            elif method == "midpoint":
                # First half step
                v1 = self.predict_velocity(encoder_outputs, x_unnorm, t)
                x_mid = x + v1 * (dt / 2)
                
                # Midpoint
                x_mid_unnorm = self.unnormalize_trajectory(x_mid)
                t_mid = torch.full((B, A), t_current + dt / 2, device=device)
                
                v2 = self.predict_velocity(encoder_outputs, x_mid_unnorm, t_mid)
                x = x + v2 * dt
            else:
                raise ValueError(f"Unknown integration method: {method}")
        
        # x is now approximately x_0 (data) in normalized trajectory space
        x_0_local = self.unnormalize_trajectory(x)  # [B, A, 80, 5] in local frame
        
        # Transform to global frame
        trajs = self._local_to_global_trajectory(x_0_local, current_states)  # [B, A, 80, 5]
        
        # Convert trajectory to actions via inverse kinematics
        # Need to prepend current states for inverse kinematics
        trajs_with_current = torch.cat([
            current_states.unsqueeze(2),  # [B, A, 1, 5]
            trajs
        ], dim=2)  # [B, A, 81, 5]
        
        # Pad to 8 dims for inverse_kinematics (it expects [B, A, T, 8])
        trajs_padded = torch.cat([
            trajs_with_current,
            torch.zeros_like(trajs_with_current[..., :3])  # dummy length, width, height
        ], dim=-1)  # [B, A, 81, 8]
        
        trajs_valid = torch.ones(B, A, 81, dtype=torch.bool, device=device)
        
        actions, actions_valid = inverse_kinematics(
            trajs_padded,
            trajs_valid,
            dt=0.1,
            action_len=self._action_len,
        )  # [B, A, 16, 2]
        
        return {
            "actions": actions,
            "actions_normalized": self.normalize_actions(actions),
            "trajs": trajs,
        }

    @torch.no_grad()
    def sample_denoiser(
        self,
        batch,
        num_steps: int = 50,
        x_t: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Compatibility method for VBD-style interface.
        """
        result = self.sample_actions(batch, num_steps=num_steps)
        
        return {
            "denoised_trajs": result["trajs"],
            "denoised_actions": result["actions"],
            "denoised_actions_normalized": result.get("actions_normalized"),
        }

    @torch.no_grad()
    def sample_k_actions(
        self,
        batch,
        k: int = 6,
        num_steps: int = 50,
        method: str = "euler",
    ) -> dict:
        """
        Sample K diverse trajectory proposals per agent.
        """
        encoder_outputs = self.encoder(batch)
        
        B = batch["agents_history"].shape[0]
        A = min(batch["agents_history"].shape[1], self._agents_len)
        device = encoder_outputs["encodings"].device
        
        current_states = encoder_outputs["agents"][:, :A, -1, :5]  # [B, A, 5]
        
        # Start from K different noise samples in normalized trajectory space
        # Shape: [B, A, K, 80, 5]
        x = torch.randn(B, A, k, self._future_len, 5, device=device)
        
        # Reshape for batch processing: [B*K, A, 80, 5]
        x = x.permute(0, 2, 1, 3, 4).reshape(B * k, A, self._future_len, 5)
        
        # Expand encoder outputs for K samples
        encoder_outputs_expanded = self._expand_encoder_outputs(encoder_outputs, k)
        
        dt = -1.0 / num_steps
        
        for step in range(num_steps):
            t_current = 1.0 - step / num_steps
            t = torch.full((B * k, A), t_current, device=device)
            
            x_unnorm = self.unnormalize_trajectory(x)
            
            v = self.predict_velocity(encoder_outputs_expanded, x_unnorm, t)
            x = x + v * dt
        
        # Reshape back: [B*K, A, 80, 5] -> [B, K, A, 80, 5] -> [B, A, K, 80, 5]
        x = x.reshape(B, k, A, self._future_len, 5).permute(0, 2, 1, 3, 4)
        
        x_0_local = self.unnormalize_trajectory(x)  # [B, A, K, 80, 5]
        
        # Transform each sample to global frame
        # Expand current_states for K samples
        current_states_expanded = current_states.unsqueeze(2).expand(-1, -1, k, -1)  # [B, A, K, 5]
        
        # Process each K sample
        trajs_list = []
        actions_list = []
        for ki in range(k):
            traj_local_k = x_0_local[:, :, ki]  # [B, A, 80, 5]
            traj_global_k = self._local_to_global_trajectory(traj_local_k, current_states)
            trajs_list.append(traj_global_k)
            
            # Convert to actions
            trajs_with_current = torch.cat([
                current_states.unsqueeze(2),
                traj_global_k
            ], dim=2)
            
            trajs_padded = torch.cat([
                trajs_with_current,
                torch.zeros_like(trajs_with_current[..., :3])
            ], dim=-1)
            
            trajs_valid = torch.ones(B, A, 81, dtype=torch.bool, device=device)
            actions_k, _ = inverse_kinematics(trajs_padded, trajs_valid, dt=0.1, action_len=self._action_len)
            actions_list.append(actions_k)
        
        trajs = torch.stack(trajs_list, dim=2)  # [B, A, K, 80, 5]
        actions = torch.stack(actions_list, dim=2)  # [B, A, K, 16, 2]
        
        return {
            "actions": actions,
            "actions_normalized": self.normalize_actions(actions),
            "trajs": trajs,
        }

    def _expand_encoder_outputs(self, encoder_outputs: dict, k: int) -> dict:
        """Expand encoder outputs for K-sample generation."""
        B = encoder_outputs["encodings"].shape[0]
        expanded = {}
        
        for key, val in encoder_outputs.items():
            if isinstance(val, torch.Tensor):
                # Expand along batch dimension
                expanded_val = val.unsqueeze(1).expand(-1, k, *val.shape[1:])
                expanded[key] = expanded_val.reshape(B * k, *val.shape[1:])
            else:
                expanded[key] = val
                
        return expanded

    # =====================================================================
    # Predictor (Goal Prediction)
    # =====================================================================
    
    def _forward_predictor(self, encoder_outputs: dict) -> dict:
        """Forward pass through goal predictor."""
        goal_actions_normalized, goal_scores = self.predictor(encoder_outputs)
        
        current_states = encoder_outputs["agents"][:, :self._agents_len, -1, :5]
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

    # =====================================================================
    # Loss Functions
    # =====================================================================
    
    def _compute_goal_loss(
        self,
        goal_trajs, goal_scores,
        agents_future, agents_future_valid,
        anchors, agents_interested,
    ):
        """Compute goal prediction loss (same as VBD)."""
        current_states = agents_future[:, :, 0, :3]
        anchors_global = batch_transform_trajs_to_global_frame(anchors, current_states)
        num_batch, num_agents, num_query, _ = anchors_global.shape

        traj_mask = agents_future_valid[..., 1:] * (agents_interested[..., None] > 0)

        goal_gt = agents_future[:, :, -1:, :2].flatten(0, 1)
        trajs_gt = agents_future[:, :, 1:, :3].flatten(0, 1)
        trajs = goal_trajs.flatten(0, 1)[..., :3]
        anchors_global = anchors_global.flatten(0, 1)

        idx_anchor = torch.argmin(torch.norm(anchors_global - goal_gt, dim=-1), dim=-1)

        dist = torch.norm(trajs[:, :, :, :2] - trajs_gt[:, None, :, :2], dim=-1)
        dist = dist * traj_mask.flatten(0, 1)[:, None, :]
        idx = torch.argmin(dist.mean(-1), dim=-1)

        idx = torch.where(agents_future_valid[..., -1].flatten(0, 1), idx_anchor, idx)
        trajs_select = trajs[torch.arange(num_batch * num_agents), idx]

        traj_loss = smooth_l1_loss(trajs_select, trajs_gt, reduction="none").sum(-1)
        traj_loss = traj_loss * traj_mask.flatten(0, 1)

        scores = goal_scores.flatten(0, 1)
        score_loss = cross_entropy(scores, idx, reduction="none")
        score_loss = score_loss * (agents_interested.flatten(0, 1) > 0)

        traj_loss_mean = traj_loss.sum() / traj_mask.sum().clamp(min=1.0)
        score_loss_mean = score_loss.sum() / (agents_interested > 0).sum().clamp(min=1.0)

        return traj_loss_mean, score_loss_mean

    # =====================================================================
    # Metrics
    # =====================================================================
    
    def _compute_trajectory_metrics(
        self,
        pred_trajs,
        agents_future,
        agents_future_valid,
        agents_interested,
        top_k: int = 8,
    ):
        """Compute ADE and FDE metrics."""
        pred = pred_trajs[:, :top_k, :, :2]
        gt = agents_future[:, :top_k, 1:, :2]
        mask = (agents_future_valid[:, :top_k, 1:] & (agents_interested[:, :top_k, None] > 0)).bool()
        
        error = torch.norm(pred - gt, dim=-1)
        
        ade = error[mask].mean().item() if mask.any() else 0.0
        fde = error[..., -1][mask[..., -1]].mean().item() if mask[..., -1].any() else 0.0
        
        return ade, fde

    def _compute_predictor_metrics(
        self,
        goal_trajs,
        agents_future,
        agents_future_valid,
        agents_interested,
        top_k: int = 8,
    ):
        """Compute min-over-K ADE and FDE for predictor."""
        goal_trajs = goal_trajs[:, :top_k, :, :, :2]  # [B, A, K, T, 2]
        gt = agents_future[:, :top_k, 1:, :2]  # [B, A, T, 2]
        mask = (agents_future_valid[:, :top_k, 1:] & (agents_interested[:, :top_k, None] > 0)).bool()

        # Error for each mode: [B, A, K, T]
        error = torch.norm(goal_trajs - gt[:, :, None, :, :], dim=-1)
        error = error * mask[..., None, :].float()
        
        # Best mode selection (min over K)
        best_idx = torch.argmin(error.mean(-1), dim=-1)  # [B, A]
        
        best_error = error[
            torch.arange(error.shape[0])[:, None],
            torch.arange(error.shape[1])[None, :],
            best_idx,
        ]  # [B, A, T]

        ade = best_error.sum() / mask.sum().clamp(min=1.0)
        fde = best_error[..., -1].sum() / mask[..., -1].sum().clamp(min=1.0)

        return ade.item(), fde.item()
