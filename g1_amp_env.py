# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .g1_amp_env_cfg import G1AmpEnvCfg
from .motions import MotionLoader


class G1AmpEnv(DirectRLEnv):
    cfg: G1AmpEnvCfg

    def __init__(self, cfg: G1AmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        #外推力配置
        self._enable_push = True
        self._push_applied = False

        self._fixed_push = True
        # self._fixed_push = False
        self._push_step = 100
        self._push_force_vec = torch.tensor([0.0, 600.0, 0.0], device=self.device)  # 推力大小和方向
        self._step_count = 0
        self._push_applied = False
        self._pending_push = False

        #push-train
        # self._random_push = True
        self._random_push = False
        self._random_push_min = 50.0
        self._random_push_max = 300.0
        self._steps_to_next_push = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # self.push_interval = 100


        # load motion
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # DOF and key body indexes  
        key_body_names = [ "left_shoulder_pitch_link",
            "right_shoulder_pitch_link",
            "left_elbow_link",
            "right_elbow_link",
            "right_hip_yaw_link",
            "left_hip_yaw_link",
            "right_rubber_hand",
            "left_rubber_hand",
            "right_ankle_roll_link",
            "left_ankle_roll_link"]

        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        # Used to for reset strategy
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        self._push_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._ref_start_time_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._push_recovered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._push_start_step = torch.full((self.num_envs,),-1, dtype=torch.int32, device=self.device)

        self._vel_err = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._vel_recovery_time_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._vel_below_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        self._ref_root_lin_vel_cache = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._cur_root_lin_vel_cache = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self._vel_thresh = 0.5
        self._vel_hold_steps=10

    def _get_ref_root_lin_vel(self) -> torch.Tensor:
        t_s = self._ref_start_time_s + self.episode_length_buf.to(torch.float32) * self.step_dt
        self._ref_time_s_cache = t_s
        t_s = torch.remainder(t_s, float(self._motion_loader.duration))
        t_np = t_s.cpu().numpy()
        _, _, _, _, body_lin_vel, _ = self._motion_loader.sample(num_samples=self.num_envs, times=t_np)

        ref_v = body_lin_vel[:, self.motion_ref_body_index]
        return torch.as_tensor(ref_v, device=self.device, dtype=torch.float32)
    
    def _update_disturb_vel_metrics(self):
        ref_v = self._get_ref_root_lin_vel()
        cur_v = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        self._vel_err = torch.linalg.norm(ref_v - cur_v, dim=-1)

        self._ref_root_lin_vel_cache = ref_v
        self._cur_root_lin_vel_cache = cur_v

        active = self._push_active & (~self._push_recovered)
        if not bool(active.any()):
            return
        
        below = self._vel_err < self._vel_thresh
        self._vel_below_count[active & below] += 1
        self._vel_below_count[active & ~below] = 0

        recovered_now = active & (self._vel_below_count >= self._vel_hold_steps)
        if bool(recovered_now.any()):
            steps_since_push = (
                self.episode_length_buf[recovered_now] - self._push_start_step[recovered_now]
            )
            self._vel_recovery_time_s[recovered_now] = steps_since_push.to(torch.float32) * self.step_dt
            self._push_recovered[recovered_now] = True


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    """
    print(inspect.getsource(Articulation.set_external_force_and_torque))
    Args:
    forces: External forces in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
    torques: External torques in bodies' local frame. Shape is (len(env_ids), len(body_ids), 3).
    positions: Positions to apply external wrench. Shape is (len(env_ids), len(body_ids), 3). Defaults to None.
    body_ids: Body indices to apply external wrench to. Defaults to None (all bodies).
    env_ids: Environment indices to apply external wrench to. Defaults to None (all instances).
    is_global: Whether to apply the external wrench in the global frame. Defaults to False. If set to False,
        the external wrench is applied in the link frame of the articulations' bodies.
    """

    def _apply_push(self, reason: str):
        num_envs = self.num_envs

        num_bodies = self.robot.data.body_pos_w.shape[1]

        forces = torch.zeros((num_envs, num_bodies, 3), device=self.device)
        torques = torch.zeros_like(forces, device=self.device)

        forces[:, self.ref_body_index] = self._push_force_vec

        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques
        )

        self._push_active[:] = True
        self._push_recovered[:] = False
        self._push_start_step[:] = self.episode_length_buf.to(torch.int32)
        self._vel_below_count.zero_()
        self._vel_recovery_time_s.zero_()

        print(
            f"[G1AmpEnv] apply push at step {self._step_count} due to {reason}, "
            f"force = {self._push_force_vec.cpu().numpy()} on body index {self.ref_body_index}"
        )

        self._push_applied = True

    def _pre_physics_step(self, actions: torch.Tensor):
        # print("[G1AmpEnv] _pre_physics_step called")
        self.actions = actions.clone()
        # self.pre_actions = actions.clone()

        self._step_count += 1

        #clear external forces and torques at the beginning of each step
        num_envs = self.num_envs
        num_bodies = self.robot.data.body_pos_w.shape[1]
        zero_forces = torch.zeros((num_envs, num_bodies, 3), device=self.device)
        zero_torques = torch.zeros_like(zero_forces, device=self.device)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=zero_forces,
            torques=zero_torques
        )

        #method 1: fixed step to trigger push
        if(self._enable_push
           and self._fixed_push
           and (not self._push_applied)
           and self._step_count == self._push_step
        ):
            self._apply_push(reason="fixed step")
            print(f"[G1AmpEnv] fixed push triggered at step {self._step_count}")

        #method 2: external trigger to push (e.g. from keyboard)
        if self._pending_push:
            self._apply_push(reason="trigger")
            self._pending_push = False

    def _apply_action(self):
        # self.pre_actions = self.actions.clone()
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def trigger_push(self):
        "push next step"
        self._enable_push = True
        self._push_applied = False
        
        self._pending_push = True
        print(f"[G1AmpEnv] trigger push at step {self._step_count+1}")

    def _get_observations(self) -> dict:
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        self._update_disturb_vel_metrics()

        t_since_push_s = (self.episode_length_buf - self._push_start_step).to(torch.float32) * self.step_dt
        t_since_push_s = torch.where(self._push_active, t_since_push_s, torch.zeros_like(t_since_push_s))

        self.extras["disturb_vel"] = {
            "push_active": self._push_active,
            "push_recovered": self._push_recovered,

            "vel_err": self._vel_err,

            "ref_root_lin_vel": self._ref_root_lin_vel_cache,
            "cur_root_lin_vel": self._cur_root_lin_vel_cache,

            "vel_recovery_time_s": self._vel_recovery_time_s,
            "vel_thresh": self._vel_thresh,
            "hold_steps": self._vel_hold_steps,
            "t_since_push_s": t_since_push_s,

            "ref_time_s":self._ref_time_s_cache,
            "ref_start_time_s": self._ref_start_time_s,
        }

        return {"policy": obs}

    # def _get_rewards(self) -> torch.Tensor:
    #     return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
    def _get_rewards(self) -> torch.Tensor:
        total_reward, reward_log = compute_rewards(
            self.cfg.rew_termination,
            self.cfg.rew_action_l2,
            self.cfg.rew_joint_pos_limits,
            self.cfg.rew_joint_acc_l2,
            self.cfg.rew_joint_vel_l2,
            self.reset_terminated,
            self.actions,
            self.robot.data.joint_pos,
            self.robot.data.soft_joint_pos_limits,
            self.robot.data.joint_acc,
            self.robot.data.joint_vel,    
        )
        self.extras["log"] = reward_log
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        #push state reset
        self._push_active[env_ids] = False
        self._push_recovered[env_ids] = False
        self._push_start_step[env_ids] = -1
        self._vel_err[env_ids] = 0.0
        self._vel_below_count[env_ids] = 0
        self._vel_recovery_time_s[env_ids] = 0.0

        # # 推力计数归零（每个 episode 重来）
        self._step_count = 0
        self._push_applied = False
        self._pending_push = False
    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
        
        self._ref_start_time_s[env_ids] = torch.as_tensor(times, device=self.device, dtype=torch.float32)

        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["pelvis"])[0]
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.05  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # update AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs
@torch.jit.script
def compute_rewards(
    rew_scale_termination: float,
    rew_scale_action_l2: float,
    rew_scale_joint_pos_limits: float,
    rew_scale_joint_acc_l2: float,
    rew_scale_joint_vel_l2: float,
    reset_terminated: torch.Tensor,
    actions: torch.Tensor,
    joint_pos: torch.Tensor,
    soft_joint_pos_limits: torch.Tensor,
    joint_acc: torch.Tensor,
    joint_vel: torch.Tensor,
):
    rew_termination = rew_scale_termination * reset_terminated.float()
    rew_action_l2 = rew_scale_action_l2 * torch.sum(torch.square(actions), dim=1)
    
    out_of_limits = -(joint_pos - soft_joint_pos_limits[:,:,0]).clip(max=0.0)
    out_of_limits += (joint_pos - soft_joint_pos_limits[:,:,1]).clip(min=0.0)
    rew_joint_pos_limits = rew_scale_joint_pos_limits * torch.sum(out_of_limits, dim=1)
    
    rew_joint_acc_l2 = rew_scale_joint_acc_l2 * torch.sum(torch.square(joint_acc), dim=1)
    rew_joint_vel_l2 = rew_scale_joint_vel_l2 * torch.sum(torch.square(joint_vel), dim=1)
    total_reward = rew_termination + rew_action_l2 + rew_joint_pos_limits + rew_joint_acc_l2 + rew_joint_vel_l2
    
    log = {
        "rew_termination": (rew_termination).mean(),
        "rew_action_l2": (rew_action_l2).mean(),
        "rew_joint_pos_limits": (rew_joint_pos_limits).mean(),
        "rew_joint_acc_l2": (rew_joint_acc_l2).mean(),
        "rew_joint_vel_l2": (rew_joint_vel_l2).mean(),
        }
    return total_reward, log