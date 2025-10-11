import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 确保本地包在 PYTHONPATH
import kinodynamic_car_SB3.dubins_env  # noqa: F401  # 触发注册

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

import time

# ENV_ID = "Dubins-v0"
ENV_ID = "KinematicCar-v0"
SEED = 42

def make_env():
    def _thunk():
        env = gym.make(ENV_ID)  # 训练阶段不渲染更快
        env.reset(seed=SEED)
        return env
    return _thunk

# 单环境（Dubins 简单，单环境足够；你也可换成 SubprocVecEnv 并行）
venv = DummyVecEnv([make_env()])
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

# 动作噪声（高斯），按动作维度配置
tmp = gym.make(ENV_ID)
n_actions = tmp.action_space.shape[-1]
tmp.close()
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

ALG = "DDPG"  # or "TD3"
Algo = DDPG if ALG == "DDPG" else TD3

# ---------------- Train ----------------
log_dir = "/workspace/kinodynamic_car_SB3/logs"
os.makedirs(log_dir, exist_ok=True)

model = Algo(
    "MlpPolicy",
    venv,
    learning_rate=3e-4,
    buffer_size=300_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    action_noise=action_noise,
    tensorboard_log=log_dir,   # tensorboard log 也在这里
    seed=SEED,
    verbose=1,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
)

print("Starting training...")
strat = time.time()
model.learn(total_timesteps=200_000, log_interval=10)
end = time.time()
print(f"Training completed in {end - strat:.2f} seconds.")
model_path = os.path.join(log_dir, f"{ALG}_kinematic_model")
vecnorm_path = os.path.join(log_dir, f"{ALG}_vecnorm.pkl")

model.save(model_path)
venv.save(vecnorm_path)



# -------- 评估（无噪声、确定性） --------
def make_eval_env():
    def _thunk():
        env = gym.make(ENV_ID)          # 评估不渲染更快；要渲染改到推理块
        return Monitor(env)             # 消除 evaluation.py 的 Monitor 警告
    return _thunk

eval_venv = DummyVecEnv([make_eval_env()])
eval_venv = VecNormalize.load(f"{ALG}_vecnorm.pkl", eval_venv)  # 加载训练统计
eval_venv.training = False
eval_venv.norm_reward = False

mean_r, std_r = evaluate_policy(model, eval_venv, n_eval_episodes=10, deterministic=True)
print(f"[{ALG}] Eval return: {mean_r:.2f} ± {std_r:.2f}")


# -------- 批量轨迹推理与绘图（多条轨迹一张图 + 保存PNG） --------
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env_for_traj():
    def _thunk():
        # 不用 render_mode，手动收集状态后统一绘图
        env = gym.make(ENV_ID)
        return Monitor(env)  # 记录回合信息，也可消除评估时的 warning
    return _thunk

# 1) VecEnv + 载入训练时的归一化统计
traj_venv = DummyVecEnv([make_env_for_traj()])
traj_venv = VecNormalize.load(f"{ALG}_vecnorm.pkl", traj_venv)
traj_venv.training = False
traj_venv.norm_reward = False

# 2) 加载已训练好的策略
eval_venv = VecNormalize.load(vecnorm_path, eval_venv)
loaded = Algo.load(model_path, device="auto")
traj_venv = VecNormalize.load(vecnorm_path, traj_venv)


# 3) 获取底层环境引用（DummyVecEnv -> Monitor -> 你的 KinematicCarEnv）
base_env = traj_venv.venv.envs[0].unwrapped

def rollout_one_episode(max_steps=1000):
    """跑一条轨迹，返回 (xs, ys, start_state, goal_state, steps, success_flag)"""
    obs = traj_venv.reset()
    start = base_env.state
    goal  = base_env.goal

    xs, ys = [start[0]], [start[1]]
    success = False
    steps = 0

    for t in range(max_steps):
        action, _ = loaded.predict(obs, deterministic=True)
        obs, rewards, dones, infos = traj_venv.step(action)
        x, y, th = base_env.state
        xs.append(x); ys.append(y)
        steps += 1
        if dones[0]:
            # 终止条件（到达目标）或截断（超步数）已在 env 内部判定
            # 你也可以通过 infos[0] 里的字段（如果写了）来区分成功/截断
            success = True  # 简单认为 done 即成功；如需精细区分，可在 env 的 info 中加标志
            break
    return xs, ys, start, goal, steps, success

# 4) 批量收集多条轨迹
N = 8                      # 要画的轨迹条数
max_steps = 1200           # 每条轨迹最多步数
trajectories = []
succ_cnt = 0
for i in range(N):
    xs, ys, st, gl, steps, ok = rollout_one_episode(max_steps=max_steps)
    trajectories.append((xs, ys, st, gl, steps, ok))
    succ_cnt += int(ok)

# 5) 统一绘图
plt.figure(figsize=(7, 7))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, (xs, ys, st, gl, steps, ok) in enumerate(trajectories):
    c = colors[i % len(colors)]
    lbl = f"ep{i+1} ({'✓' if ok else '×'}, {steps} steps)"
    plt.plot(xs, ys, '-', lw=2, color=c, alpha=0.9, label=lbl)
    # 起点/终点
    plt.plot(xs[0], ys[0], 'o', color=c, ms=6, alpha=0.9)
    plt.plot(gl[0], gl[1], '*', color=c, ms=12, alpha=0.9)

# 可选：坐标系与标题
plt.axis('equal'); plt.grid(True, ls='--', alpha=0.5)
plt.xlabel('x'); plt.ylabel('y')
plt.title(f"{ENV_ID} trajectories ({ALG}) | {succ_cnt}/{N} done")
plt.legend(loc='best', fontsize=8, framealpha=0.8)

# 6) 保存PNG（带时间戳，避免覆盖）
ts = time.strftime("%Y%m%d-%H%M%S")
out_dir = "/workspace/kinodynamic_car_SB3/plots"   # 你可以改路径
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"{ALG}_{ENV_ID}_multi_traj_{N}_{ts}.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved trajectory figure to: {out_path}")

plt.show()

traj_venv.close()


# # -------- 推理（可视化） --------
# def make_render_env():
#     def _thunk():
#         env = gym.make(ENV_ID, render_mode="human")
#         return Monitor(env)
#     return _thunk

# vis_venv = DummyVecEnv([make_render_env()])
# vis_venv = VecNormalize.load(f"{ALG}_vecnorm.pkl", vis_venv)  # 同样加载统计
# vis_venv.training = False
# vis_venv.norm_reward = False

# loaded = Algo.load(f"{ALG}_dubins_model", device="auto")

# obs = vis_venv.reset()  # 注意：VecEnv 的 obs 形状是 (n_envs, obs_dim)
# for _ in range(500):
#     action, _ = loaded.predict(obs, deterministic=True)
#     obs, rewards, dones, infos = vis_venv.step(action)
#     if dones[0]:
#         obs = vis_venv.reset()
# vis_venv.close()

# # 评估（确定性、无噪声）
# venv.training = False
# venv.norm_reward = False
# mean_r, std_r = evaluate_policy(model, venv, n_eval_episodes=10, deterministic=True)
# print(f"[{ALG}] Eval return: {mean_r:.2f} ± {std_r:.2f}")

# # 推理 127

# env = gym.make(ENV_ID, render_mode="human")
# env = VecNormalize.load(f"{ALG}_vecnorm.pkl", env)
# env.training = False
# env.norm_reward = False

# loaded = Algo.load(f"{ALG}_dubins_model", device="auto")
# obs, _ = env.reset(seed=SEED+123)
# for _ in range(500):
#     action, _ = loaded.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, _ = env.reset()
# env.close()
