## Script for training the RL algoritm
```
conda activate ur5_env
cd  --checkpoint logs/rl_games/pick_place_ur5/2025-04-09_16-52-12/nn/pick_place_ur5.pth

```

## Script for playing the trained RL algorithm

```
conda activate ur5_env
python scripts/rl_games/play.py --task=Isaac-UR5-PickAndPlace-PPO --num_envs 2 --checkpoint logs/rl_games/pick_place_ur5/2025-04-09_17-00-04/nn/pick_place_ur5.pth

```

## Script for Tensorboard
```
tensorboard --logdir logs/rl_games/pick_place_ur5/2025-05-12_08-52-26/
```


## Script for collecting video data
python scripts/rl_games/play.py     --task=Isaac-UR5-ObjCameraPoseTracking-PPO     --num_envs 1 --headless     --checkpoint logs/rl_games/camera_pose_tracking_ur5/final_obstacle_random/nn/camera_pose_tracking_ur5.pth     --enable_cameras     --record_camera     --record_duration 2.0     --record_episodes_hf 1500 --record --record_episodes 1500


Livestream client IP address
10.125.61.13
vnode003


## Scripts for Verification
export MKL_THREADING_LAYER=GNU && export OMP_NUM_THREADS=1 && python runtime_verifier_abcrown.py

##
