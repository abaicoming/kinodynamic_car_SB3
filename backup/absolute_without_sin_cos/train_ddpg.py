import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# 触发环境注册（确保 __init__.py 注册了 KinematicCar-v0）
import kinodynamic_car_SB3.dubins_env  # noqa: F401

ENV_ID = "KinematicCar-v0"
SEED = 42
ALG = "DDPG"  # or "TD3"
Algo = DDPG if ALG == "DDPG" else TD3

# 统一日志/模型目录
LOG_DIR = "/workspace/kinodynamic_car_SB3/logs"
PLOT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------- Train ----------------
def make_env_train():
    def _thunk():
        env = gym.make(ENV_ID)
        env.reset(seed=SEED)
        return env
    return _thunk

# 单环境（可切 SubprocVecEnv 并行）
venv = DummyVecEnv([make_env_train()])
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

# 动作噪声（2维环境建议初期大些帮助探索）
tmp = gym.make(ENV_ID)
n_actions = tmp.action_space.shape[-1]
tmp.close()
sigma = np.array([0.5, 0.4], dtype=np.float32) if n_actions == 2 else 0.3 * np.ones(n_actions, dtype=np.float32)
action_noise = NormalActionNoise(mean=np.zeros(n_actions, dtype=np.float32), sigma=sigma)

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
    tensorboard_log=LOG_DIR,  # TensorBoard 日志也放在 LOG_DIR
    seed=SEED,
    verbose=1,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
)

total_steps = 200_000
model.learn(total_timesteps=total_steps, log_interval=10)

# 统一保存到 LOG_DIR
model_path = os.path.join(LOG_DIR, f"{ALG}_kinematic_model")
vecnorm_path = os.path.join(LOG_DIR, f"{ALG}_vecnorm.pkl")
model.save(model_path)
venv.save(vecnorm_path)
print(f"Saved model to: {model_path}")
print(f"Saved VecNormalize stats to: {vecnorm_path}")

# ---------------- Evaluate ----------------
def make_env_eval():
    def _thunk():
        return Monitor(gym.make(ENV_ID))
    return _thunk

eval_venv = DummyVecEnv([make_env_eval()])
eval_venv = VecNormalize.load(vecnorm_path, eval_venv)
eval_venv.training = False
eval_venv.norm_reward = False

mean_r, std_r = evaluate_policy(model, eval_venv, n_eval_episodes=10, deterministic=True)
print(f"[{ALG}] Eval return: {mean_r:.2f} ± {std_r:.2f}")

# ---------------- Batch Trajectory Visualization ----------------
def make_env_for_traj():
    def _thunk():
        return Monitor(gym.make(ENV_ID))  # 不用 render_mode，收集坐标后统一绘图
    return _thunk

traj_venv = DummyVecEnv([make_env_for_traj()])
traj_venv = VecNormalize.load(vecnorm_path, traj_venv)
traj_venv.training = False
traj_venv.norm_reward = False

loaded = Algo.load(model_path, device="auto")
# DummyVecEnv -> Monitor -> 你的 KinematicCarEnv
base_env = traj_venv.venv.envs[0].unwrapped

def rollout_one_episode(max_steps=1200):
    """返回 (xs, ys, start_state, goal_state, steps, is_success)"""
    obs = traj_venv.reset()
    start = base_env.state
    goal = base_env.goal

    xs, ys = [start[0]], [start[1]]
    steps = 0
    is_success = False

    for _ in range(max_steps):
        action, _ = loaded.predict(obs, deterministic=True)
        obs, rewards, dones, infos = traj_venv.step(action)

        x, y, th = base_env.state
        xs.append(x); ys.append(y)
        steps += 1

        if dones[0]:
            # 如果你在 env.step() 里加了 info["is_success"]=terminated，这里会准确统计
            is_success = bool(infos[0].get("is_success", True))  # 没有该键就默认 True
            break

    return xs, ys, start, goal, steps, is_success

N = 8  # 画 N 条随机轨迹
trajectories = []
succ_cnt = 0
for i in range(N):
    xs, ys, st, gl, steps, ok = rollout_one_episode()
    trajectories.append((xs, ys, st, gl, steps, ok))
    succ_cnt += int(ok)

plt.figure(figsize=(7, 7))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, (xs, ys, st, gl, steps, ok) in enumerate(trajectories):
    c = colors[i % len(colors)]
    lbl = f"ep{i+1} ({'✓' if ok else '×'}, {steps} steps)"
    plt.plot(xs, ys, '-', lw=2, color=c, alpha=0.9, label=lbl)
    plt.plot(xs[0], ys[0], 'o', color=c, ms=6, alpha=0.9)   # start
    plt.plot(gl[0], gl[1], '*', color=c, ms=12, alpha=0.9)  # goal
plt.legend()
plt.axis('equal'); plt.grid(True, ls='--', alpha=0.5)
plt.xlabel('x'); plt.ylabel('y')
plt.title(f"{ENV_ID} trajectories ({ALG}) | {succ_cnt}/{N} reached")

ts = time.strftime("%Y%m%d-%H%M%S")
out_path = os.path.join(PLOT_DIR, f"{ALG}_{ENV_ID}_multi_traj_{N}_{ts}.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved trajectory figure to: {out_path}")
# plt.show()

traj_venv.close()
