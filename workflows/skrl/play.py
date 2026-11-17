# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=600, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default='Isaac-robot-US-guided-surgery-v0', help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default='/home/yunkao/git/IsaacLabExtensionTemplate/logs/experiments/us-guided-surgery/single/model-based-sim/PPO/2025-04-25_18-39-42_ppo_torch_PPO_default_US_net/checkpoints/best_agent.pt', help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO", "A2C"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--video_preroll",
    type=int,
    default=0,
    help="Number of additional still frames after reset and before policy execution.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import cProfile

import skrl
from packaging import version
import types
import imageio.v2 as imageio
import numpy as np

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import spinal_surgery
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
import wandb

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    try:
        experiment_cfg = load_cfg_from_registry(args_cli.task, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # -------------------------------------------------------------------------
    # Record every simulator render, including renders executed inside reset().
    # -------------------------------------------------------------------------
    video_writer = None
    video_frame_count = 0
    video_path = None

    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "play")
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, "full_sequence.mp4")

        # The environment configured here renders at:
        # physics_dt = 1 / 120 and render_interval = 2 -> 60 rendered frames/s.
        video_fps = 60

        video_writer = imageio.get_writer(
            video_path,
            fps=video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

        raw_env = env.unwrapped
        original_sim_render = raw_env.sim.render

        # Prevent recursion in case raw_env.render() internally reaches sim.render().
        capture_in_progress = False

        def render_and_record(*render_args, **render_kwargs):
            nonlocal video_frame_count, capture_in_progress

            # Perform the simulator render requested by the environment.
            result = original_sim_render(*render_args, **render_kwargs)

            if capture_in_progress:
                return result

            capture_in_progress = True

            try:
                # Retrieve the frame that has just been rendered.
                frame = raw_env.render(recompute=False)

                if frame is not None:
                    # Isaac Lab returns an RGB NumPy array.
                    frame = frame[..., :3]

                    if frame.dtype != np.uint8:
                        if np.issubdtype(frame.dtype, np.floating):
                            if frame.max() <= 1.0:
                                frame = frame * 255.0

                        frame = np.clip(frame, 0, 255).astype(np.uint8)

                    video_writer.append_data(frame)
                    video_frame_count += 1

            finally:
                capture_in_progress = False

            return result

        # Replace only this SimulationContext instance's render method.
        raw_env.sim.render = render_and_record

        print(f"[INFO] Full-sequence recording enabled: {video_path}")

    # get environment (physics) dt for real-time evaluation
    try:
        dt = env.physics_dt
    except AttributeError:
        dt = env.unwrapped.physics_dt



    # init wandb
    wandb.init(project=args_cli.task, config=env_cfg)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()

    # Optional still pre-roll after reset and before the surgery policy.
    if args_cli.video and args_cli.video_preroll > 0:
        print(
            f"[INFO] Adding {args_cli.video_preroll} still pre-roll frames "
            "before policy execution."
        )

        raw_env = env.unwrapped

        for _ in range(args_cli.video_preroll):
            frame = raw_env.render(recompute=False)

            if frame is not None:
                frame = frame[..., :3]

                if frame.dtype != np.uint8:
                    if np.issubdtype(frame.dtype, np.floating):
                        if frame.max() <= 1.0:
                            frame = frame * 255.0

                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                video_writer.append_data(frame)
                video_frame_count += 1

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if video_writer is not None:
        video_writer.close()
        print(
            f"[INFO] Saved {video_frame_count} frames to:\n"
            f"       {video_path}"
        )
    # close the simulator
    env.close()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("main_stats.prof")
    # close sim app
    simulation_app.close()
