"""Script to resample npz motion files to a different fps."""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mjlab.utils.lab_api.math import quat_slerp


def resample_npz(
    input_file: str,
    output_file: str,
    target_fps: float = 50.0,
):
    """Resample npz motion file to target fps.
    
    Args:
        input_file: Path to input npz file
        output_file: Path to output npz file
        target_fps: Target frame rate (default 50.0 Hz to match env.step_dt=0.02)
    """
    print(f"Loading {input_file}...")
    data = np.load(input_file)
    
    # Get input fps
    if 'fps' in data:
        input_fps = float(data['fps'])
    else:
        raise ValueError("Input npz file must contain 'fps' key")
    
    print(f"Input fps: {input_fps} Hz")
    print(f"Target fps: {target_fps} Hz")
    
    # Get original data
    joint_pos = data['joint_pos']  # (num_frames, num_joints)
    joint_vel = data['joint_vel']  # (num_frames, num_joints)
    body_pos_w = data['body_pos_w']  # (num_frames, num_bodies, 3)
    body_quat_w = data['body_quat_w']  # (num_frames, num_bodies, 4) - wxyz format
    body_lin_vel_w = data['body_lin_vel_w']  # (num_frames, num_bodies, 3)
    body_ang_vel_w = data['body_ang_vel_w']  # (num_frames, num_bodies, 3)
    
    num_frames = joint_pos.shape[0]
    duration = (num_frames - 1) / input_fps
    
    print(f"Original frames: {num_frames}")
    print(f"Duration: {duration:.3f} seconds")
    
    # Create output time axis
    output_times = np.arange(0, duration, 1/target_fps)
    output_num_frames = len(output_times)
    
    print(f"Output frames: {output_num_frames}")
    
    # Create input time axis
    input_times = np.linspace(0, duration, num_frames)
    
    # Resample joint_pos and joint_vel (linear interpolation)
    from scipy.interpolate import interp1d
    
    print("Resampling joint_pos and joint_vel...")
    f_joint_pos = interp1d(
        input_times, joint_pos, axis=0, kind='linear',
        bounds_error=False, fill_value='extrapolate'
    )
    f_joint_vel = interp1d(
        input_times, joint_vel, axis=0, kind='linear',
        bounds_error=False, fill_value='extrapolate'
    )
    
    joint_pos_resampled = f_joint_pos(output_times)
    joint_vel_resampled = f_joint_vel(output_times)
    
    # Resample body_pos_w (linear interpolation)
    print("Resampling body_pos_w...")
    body_pos_resampled = np.zeros((output_num_frames, body_pos_w.shape[1], 3))
    for body_idx in range(body_pos_w.shape[1]):
        f_pos = interp1d(
            input_times, body_pos_w[:, body_idx, :], axis=0, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )
        body_pos_resampled[:, body_idx, :] = f_pos(output_times)
    
    # Resample body_quat_w (spherical linear interpolation)
    print("Resampling body_quat_w (using SLERP)...")
    body_quat_resampled = np.zeros((output_num_frames, body_quat_w.shape[1], 4))
    
    # Convert to torch for quat_slerp
    body_quat_torch = torch.from_numpy(body_quat_w).float()
    output_times_torch = torch.from_numpy(output_times).float()
    input_times_torch = torch.from_numpy(input_times).float()
    
    for body_idx in range(body_quat_w.shape[1]):
        quat_seq = body_quat_torch[:, body_idx, :]  # (num_frames, 4) wxyz
        
        resampled_quats = []
        for t in output_times:
            # Find the two frames to interpolate between
            idx = np.searchsorted(input_times, t, side='right') - 1
            idx = max(0, min(idx, num_frames - 2))
            
            t0, t1 = input_times[idx], input_times[idx + 1]
            q0 = quat_seq[idx]  # wxyz
            q1 = quat_seq[idx + 1]  # wxyz
            
            if abs(t1 - t0) < 1e-6:
                alpha = 0.0
            else:
                alpha = (t - t0) / (t1 - t0)
            
            # SLERP
            q_interp = quat_slerp(q0, q1, float(alpha))
            resampled_quats.append(q_interp.numpy())
        
        body_quat_resampled[:, body_idx, :] = np.stack(resampled_quats, axis=0)
    
    # Resample body_lin_vel_w and body_ang_vel_w (linear interpolation)
    print("Resampling body_lin_vel_w and body_ang_vel_w...")
    body_lin_vel_resampled = np.zeros((output_num_frames, body_lin_vel_w.shape[1], 3))
    body_ang_vel_resampled = np.zeros((output_num_frames, body_ang_vel_w.shape[1], 3))
    
    for body_idx in range(body_lin_vel_w.shape[1]):
        f_lin_vel = interp1d(
            input_times, body_lin_vel_w[:, body_idx, :], axis=0, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )
        f_ang_vel = interp1d(
            input_times, body_ang_vel_w[:, body_idx, :], axis=0, kind='linear',
            bounds_error=False, fill_value='extrapolate'
        )
        body_lin_vel_resampled[:, body_idx, :] = f_lin_vel(output_times)
        body_ang_vel_resampled[:, body_idx, :] = f_ang_vel(output_times)
    
    # Save resampled data
    print(f"\nSaving to {output_file}...")
    np.savez(
        output_file,
        fps=target_fps,
        joint_pos=joint_pos_resampled,
        joint_vel=joint_vel_resampled,
        body_pos_w=body_pos_resampled,
        body_quat_w=body_quat_resampled,
        body_lin_vel_w=body_lin_vel_resampled,
        body_ang_vel_w=body_ang_vel_resampled,
    )
    
    print(f"✓ Successfully resampled to {target_fps} Hz")
    print(f"  Input: {num_frames} frames @ {input_fps} Hz")
    print(f"  Output: {output_num_frames} frames @ {target_fps} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Resample npz motion file to target fps"
    )
    parser.add_argument("input_file", type=str, help="Input npz file path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output npz file path (default: input_file with _50fps suffix)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=50.0,
        help="Target frame rate (default: 50.0 Hz to match env.step_dt=0.02)",
    )
    
    args = parser.parse_args()
    
    # Determine output file path
    if args.output is None:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"{input_path.stem}_50fps{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    resample_npz(
        input_file=args.input_file,
        output_file=str(output_path),
        target_fps=args.target_fps,
    )


if __name__ == "__main__":
    main()

