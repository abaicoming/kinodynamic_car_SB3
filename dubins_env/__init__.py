from gymnasium.envs.registration import register
from .dubins_env import KinematicCarEnv

# 二输入小车
#  sac
# register(
#     id="KinematicCar-v0",
#     entry_point="kinodynamic_car_SB3.dubins_env:KinematicCarEnv",
#     kwargs=dict(
#         L=1.0,
#         dt=0.05,
#         v_max=3.0,
#         v_min=0.1,
#         steer_max=1.2,          # 前轮最大转角（弧度）
#         allow_reverse=True,     # 允许倒车
#         goal_tol=0.15,
#         yaw_tol=1.2,
#         max_episode_steps=1000,
#         # 奖励项
#         # w_dist=40.0,
#         # w_yaw=2.0,
#         # w_u_v=5e-4,
#         # w_u_steer=1e-3,
#         # w_goal=200.0,
#         w_dist=2.5,
#         w_yaw=0.2,
#         w_u_v=5e-5,
#         w_u_steer=1e-4,
#         w_goal=20.0,
#         # 采样范围
#         xy_range=(-6.0, 6.0),
#         goal_xy_range=(-6.0, 6.0),
#         min_start_goal_dist=0.1,
#         obs_noise_std=0.0,
#         render_mode=None,
#         # 障碍物参数
#         obstacle_center=(0.0, 0.0),
#         obstacle_radius=1.0,
#         safe_margin=0.5,
#         # w_obs_shaping=4,
#         # w_collision=100.0,
#         w_obs_shaping=0.05,
#         w_collision=10.0,
#         # 避障 shaping 类型
#         obs_shaping_type="exp",   # ["exp","quad","barrier","reciprocal","none"]
#         obs_sigma=0.8,
#         obs_margin=10.0,
#         obs_power=2.0,
#         # state
#         input_mode = "old_input", # ["old_input", "new_input"]
#         fixed_goal_flag = False,
#         fixed_goal = (1.5,0,0)
#     ),
# )

# # 二输入小车 ddpg
register(
    id="KinematicCar-v0",
    entry_point="kinodynamic_car_SB3.dubins_env:KinematicCarEnv",
    kwargs=dict(
        L=1.0,
        dt=0.05,
        v_max=3.0,
        v_min=0.1,
        steer_max=1.2,          # 前轮最大转角（弧度）
        allow_reverse=True,     # 允许倒车
        goal_tol=0.15,
        yaw_tol=1.2,
        max_episode_steps=500,
        # 奖励项
        w_dist=0.25,
        w_yaw=0.01,
        w_u_v=5e-4,
        w_u_steer=1e-3,
        w_goal=200.0,
        # 采样范围
        xy_range=(-6.0, 6.0),
        goal_xy_range=(-6.0, 6.0),
        min_start_goal_dist=0.1,
        obs_noise_std=0.0,
        render_mode=None,
        # 障碍物参数
        obstacle_center=(0.0, 0.0),
        obstacle_radius=1.0,
        safe_margin=0.5,
        w_obs_shaping=0.5,
        w_collision=100.0,
        # 避障 shaping 类型
        obs_shaping_type="exp",   # ["exp","quad","barrier","reciprocal","none"]
        obs_sigma=0.8,
        obs_margin=10.0,
        obs_power=2.0,
    ),
)