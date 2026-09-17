"""Register the Wuji Hand DemoTrack task."""

from mjlab.tasks.registry import register_mjlab_task
from wuji_mjlab.rl.runner import WujiOnPolicyRunner

from ...agents import wuji_hand_demo_track_ppo_runner_cfg
from .env_cfg import wuji_hand_demo_track_env_cfg

register_mjlab_task(
  task_id="WujiHand_HumanDemoTracking",
  env_cfg=wuji_hand_demo_track_env_cfg(num_envs=4096),
  play_env_cfg=wuji_hand_demo_track_env_cfg(play=True),
  rl_cfg=wuji_hand_demo_track_ppo_runner_cfg(),
  runner_cls=WujiOnPolicyRunner,
)

# Full training profile: preserve the validated 4096-env / 7500-update recipe.
# TensorBoard keeps a complete local record without requiring W&B credentials.
register_mjlab_task(
  task_id="WujiHand_HumanDemoTracking_Light",
  env_cfg=wuji_hand_demo_track_env_cfg(num_envs=4096),
  play_env_cfg=wuji_hand_demo_track_env_cfg(play=True),
  rl_cfg=wuji_hand_demo_track_ppo_runner_cfg(
    run_name="DemoTrack_Light", max_iterations=7500, logger="tensorboard"
  ),
  runner_cls=WujiOnPolicyRunner,
)

# Validation profile: preserve the production Stage-1 environment and PPO
# settings while capping the run at 200 updates. This exercises checkpoints,
# statistics and learning dynamics without changing the production recipe.
register_mjlab_task(
  task_id="WujiHand_HumanDemoTracking_Validation",
  env_cfg=wuji_hand_demo_track_env_cfg(num_envs=4096),
  play_env_cfg=wuji_hand_demo_track_env_cfg(play=True),
  rl_cfg=wuji_hand_demo_track_ppo_runner_cfg(
    run_name="DemoTrack_Validation", max_iterations=200, logger="tensorboard"
  ),
  runner_cls=WujiOnPolicyRunner,
)
