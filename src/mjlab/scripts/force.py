"""Script to play RL agent with RSL-RL."""

import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _get_sensor_body_names(sensor) -> list[str]:
  """Extract unique body names from contact sensor slots, preserving order."""
  body_names = []
  seen = set()
  for slot in sensor._slots:
    if slot.primary_name not in seen:
      body_names.append(slot.primary_name)
      seen.add(slot.primary_name)
  return body_names


def _build_body_group_map(body_names: list[str]) -> Dict[str, str]:
  """Map body name -> group label."""
  group_map: Dict[str, str] = {}
  for name in body_names:
    upper = name.upper()
    if any(k in upper for k in ("HIP", "KNEE", "ANKLE")):
      if upper.endswith("_L"):
        group_map[name] = "left_leg"
      elif upper.endswith("_R"):
        group_map[name] = "right_leg"
    elif any(k in upper for k in ("SHOULDER", "ELBOW", "HAND", "WRIST")):
      if upper.endswith("_L"):
        group_map[name] = "left_arm"
      elif upper.endswith("_R"):
        group_map[name] = "right_arm"
    elif any(k in upper for k in ("TORSO", "WAIST", "BASE")):
      group_map[name] = "torso_base"
    elif name == "LINK_BASE":
      group_map[name] = "torso_base"
  return group_map


def _plot_contact_histograms(
  group_forces: dict[str, np.ndarray],
  bin_size: float,
  out_path: Path,
) -> None:
  """Plot 5 subplots (left/right leg, left/right arm, torso_base) with 10 kN bins."""
  ordered_groups = [
    ("left_leg", "Left Leg"),
    ("right_leg", "Right Leg"),
    ("left_arm", "Left Arm"),
    ("right_arm", "Right Arm"),
    ("torso_base", "Torso Base"),
  ]

  fig, axes = plt.subplots(len(ordered_groups), 1, figsize=(8, 12), sharex=True)

  for ax, (group_key, title) in zip(axes, ordered_groups):
    data = group_forces.get(group_key, np.array([]))
    if data.size > 0:
      max_force = float(data.max())
      min_force = float(data.min())
      # Set reasonable range: at least 0-50kN to see distribution
      max_force = max(max_force, 1500.0)
      bins = np.arange(0.0, max_force + bin_size, bin_size)
      counts, edges = np.histogram(data, bins=bins)
      # Print debug info
      print(f"[DEBUG] {title}: min={min_force:.2f}N, max={max_force:.2f}N, "
            f"non-zero samples={np.count_nonzero(data)}, total samples={data.size}")
    else:
      max_force = 1500.0
      bins = np.arange(0.0, max_force + bin_size, bin_size)
      counts = np.zeros(len(bins) - 1)
      edges = bins
      print(f"[DEBUG] {title}: no data collected")
    
    ax.bar(
      edges[:-1],
      counts,
      width=bin_size,
      align="edge",
      color="#4C72B0",
    )
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    # Set x-axis limit to show meaningful range
    ax.set_xlim(0, max_force)

  axes[-1].set_xlabel(f"Contact force bin (N, {bin_size} N width)")
  fig.suptitle("Ground Contact Force Histogram (5 s)")
  fig.tight_layout()
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_path, dpi=150)
  plt.close(fig)
  print(f"[INFO] Contact force histogram saved to: {out_path}")


def record_contact_forces(
  env: Any,
  policy,
  duration_s: float = 5.0,
  bin_size: float = 100.0,
  save_dir: Path | None = None,
) -> None:
  """Run policy for duration_s seconds and plot contact force histograms."""
  obs, _ = env.reset()

  # Access contact sensor.
  sensor = env.unwrapped.scene["body_contact_force"]
  body_names = _get_sensor_body_names(sensor)
  body_to_group = _build_body_group_map(body_names)
  
  # Print debug info about body grouping
  print(f"[INFO] Found {len(body_names)} bodies in contact sensor:")
  for body_name in body_names:
    group = body_to_group.get(body_name, "unmapped")
    print(f"  {body_name} -> {group}")
  
  # Count bodies per group
  group_counts = defaultdict(int)
  for body_name, group in body_to_group.items():
    group_counts[group] += 1
  print(f"[INFO] Body groups: {dict(group_counts)}")

  group_forces: dict[str, list[np.ndarray]] = defaultdict(list)

  step_dt = env.unwrapped.step_dt
  max_steps = max(1, int(duration_s / step_dt))

  with torch.no_grad():
    for _ in range(max_steps):
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)

      if sensor.data.force is None or sensor.data.found is None:
        continue

      forces = sensor.data.force  # [B, N, 3]
      found = sensor.data.found   # [B, N]
      # Only record forces when there is actual contact (found > 0)
      has_contact = found > 0  # [B, N]
      magnitudes = torch.norm(forces, dim=-1)  # [B, N]

      for idx, body_name in enumerate(body_names):
        group = body_to_group.get(body_name)
        if group is None:
          continue
        # Only include forces where contact was detected
        contact_mask = has_contact[:, idx]  # [B]
        contact_forces = magnitudes[:, idx][contact_mask]  # [num_contacts]
        if contact_forces.numel() > 0:
          group_forces[group].append(contact_forces.detach().cpu().numpy().ravel())

  # Concatenate per group.
  group_arrays: dict[str, np.ndarray] = {}
  for group, chunks in group_forces.items():
    if len(chunks) == 0:
      group_arrays[group] = np.array([])
    else:
      group_arrays[group] = np.concatenate(chunks)

  out_dir = save_dir or Path.cwd()
  out_path = out_dir / "contact_forces.png"
  _plot_contact_histograms(group_arrays, bin_size, out_path)


def run_play(task: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    if DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require `registry_name` when using dummy agents."
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  
  # Wrap env to monitor reset reasons
  class ResetMonitorWrapper:
    """Wrapper to monitor and print reset reasons."""
    def __init__(self, env):
      self.env = env
      # Get unwrapped env to access termination_manager
      self._unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
      if hasattr(self._unwrapped, 'termination_manager'):
        self._term_names = self._unwrapped.termination_manager.active_terms
      else:
        self._term_names = []
    
    def __getattr__(self, name):
      # Delegate all other attributes to wrapped env
      return getattr(self.env, name)
    
    def step(self, actions):
      obs, rew, dones, extras = self.env.step(actions)
      
      # Check if any envs were reset (dones indicates reset)
      if isinstance(dones, torch.Tensor) and dones.any():
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        
        if len(reset_env_ids) > 0:
          # Get termination reasons for reset envs
          reset_reasons = []
          
          # Try to get reasons from termination_manager
          if hasattr(self._unwrapped, 'termination_manager'):
            term_mgr = self._unwrapped.termination_manager
            
            for term_name in self._term_names:
              term_active = term_mgr.get_term(term_name)
              active_envs = term_active[reset_env_ids]
              if active_envs.any():
                # Count how many envs were reset due to this termination
                count = active_envs.sum().item()
                reset_reasons.append(f"{term_name}({count})")
          
          # Check for timeouts
          if hasattr(self._unwrapped, 'reset_time_outs'):
            timeouts = self._unwrapped.reset_time_outs[reset_env_ids]
            if timeouts.any():
              count = timeouts.sum().item()
              reset_reasons.append(f"time_out({count})")
          
          # Fallback: check extras for termination info
          if not reset_reasons and "log" in extras:
            log = extras["log"]
            termination_keys = [k for k in log.keys() if k.startswith("Episode_Termination/")]
            if termination_keys:
              for key in termination_keys:
                count = log[key]
                if count > 0:
                  term_name = key.replace("Episode_Termination/", "")
                  reset_reasons.append(f"{term_name}({count})")
          
          if reset_reasons:
            env_ids_str = ", ".join(map(str, reset_env_ids.cpu().tolist()))
            reasons_str = ", ".join(reset_reasons)
            print(f"[RESET] Env IDs: [{env_ids_str}] | Reasons: {reasons_str}")
          else:
            # If no specific reasons found, just report the reset
            env_ids_str = ", ".join(map(str, reset_env_ids.cpu().tolist()))
            print(f"[RESET] Env IDs: [{env_ids_str}] | (reason unknown)")
      
      return obs, rew, dones, extras
    
    def reset(self, *args, **kwargs):
      return self.env.reset(*args, **kwargs)
  
  env = ResetMonitorWrapper(env)
  
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    if is_tracking_task:
      runner = MotionTrackingOnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    else:
      runner = OnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  # Run fixed-duration play (5 seconds) and plot contact forces.
  save_dir = log_dir if log_dir is not None else Path.cwd()
  record_contact_forces(
    env=env,
    policy=policy,
    duration_s=5.0,
    bin_size=100.0,
    save_dir=save_dir,
  )

  env.close()
  return


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
