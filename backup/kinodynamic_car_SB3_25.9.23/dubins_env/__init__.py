# from gymnasium.envs.registration import register
# from .dubins_env import DubinsEnv

# # 你可以注册多个变体（不同速度、限幅、奖励系数等）
# register(
#     id="Dubins-v0",
#     entry_point="kinodynamic_car_SB3.dubins_env:DubinsEnv",
#     kwargs=dict(
#         v=1.0,
#         omega_max=1.2,
#         dt=0.1,
#         goal_tol=0.15,
#         max_episode_steps=400,
#         w_dist=1.0,
#         w_yaw=0.1,
#         w_u=0.001,
#         w_goal=10.0,
#         xy_range=(-6.0, 6.0),
#         goal_xy_range=(-6.0, 6.0),
#         min_start_goal_dist=1.5,
#         obs_noise_std=0.0,
#         render_mode=None,
#     ),
# )

from gymnasium.envs.registration import register
from .dubins_env import KinematicCarEnv
# 旧的 Dubins-v0 注册（若已有可保留）
# register(...)

# 新的二输入小车
register(
    id="KinematicCar-v0",
    entry_point="kinodynamic_car_SB3.dubins_env:KinematicCarEnv",
    kwargs=dict(
        L=1.0,
        dt=0.1,
        v_max=1.5,
        v_min=0.0,
        steer_max=0.6,       # ≈34°
        allow_reverse=False, # 需要倒车就改 True
        goal_tol=0.15,
        max_episode_steps=400,
        w_dist=1.0,
        w_yaw=0.1,
        w_u_v=5e-4,
        w_u_steer=1e-3,
        w_goal=10.0,
        xy_range=(-6.0, 6.0),
        goal_xy_range=(-6.0, 6.0),
        min_start_goal_dist=1.5,
        obs_noise_std=0.0,
        render_mode=None,
    ),
)
