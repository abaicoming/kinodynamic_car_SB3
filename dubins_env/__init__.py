from gymnasium.envs.registration import register
from .dubins_env import KinematicCarEnv

# 二输入小车
register(
    id="KinematicCar-v0",
    entry_point="kinodynamic_car_SB3.dubins_env:KinematicCarEnv",
    kwargs=dict(
        L=1.0,
        dt=0.1,
        v_max=3.0,
        v_min=0.1,
        steer_max=1.2,       # 0.6 rad ≈ 34°
        allow_reverse=True, # 需要倒车就改 True
        goal_tol= 0.15,         # 位置容差（米）
        yaw_tol= 1.2,          # 朝向容差（弧度）
        max_episode_steps=800,
        w_dist=100,
        w_yaw=4,
        w_u_v=5e-4,
        w_u_steer=1e-3,
        w_goal=200.0,
        xy_range=(-6.0, 6.0),
        goal_xy_range=(-6.0, 6.0),
        min_start_goal_dist=0.1,
        obs_noise_std=0.0,
        render_mode=None,
        # ---- 新增避障相关参数 ----
        obstacle_center=(0.0, 0.0),
        obstacle_radius=1.0,
        safe_margin=0.3,
        w_obs_shaping=2,
        w_collision=100.0,
    ),
)
