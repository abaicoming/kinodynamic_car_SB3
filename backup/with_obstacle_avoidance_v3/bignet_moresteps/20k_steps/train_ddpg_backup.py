import os
import time
import shutil # 备份源码
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from matplotlib.patches import Circle

# 触发环境注册（确保 __init__.py 注册了 KinematicCar-v0）
import kinodynamic_car_SB3.dubins_env  # noqa: F401

from stable_baselines3.common.monitor import Monitor # for reward logging

# 定期评估曲线
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

# ========= 全局配置 =========
ENV_ID = "KinematicCar-v0"
SEED = 42
ALG = "DDPG"  # or "TD3"
Algo = DDPG if ALG == "DDPG" else TD3

# 统一日志/模型目录
# LOG_DIR = "/workspace/kinodynamic_car_SB3/backup/absolute_with_sin_cos_repair_plot_arrow_backup_python"
LOG_DIR = "/workspace/kinodynamic_car_SB3/backup/with_obstacle_avoidance_v3/bignet_moresteps/20k_steps"
PLOT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ----------------  备份源码（环境包 + 本脚本）----------------
def backup_sources():
    src_env_dir = os.path.join(os.path.dirname(__file__), "dubins_env")
    dst_env_dir = os.path.join(LOG_DIR, "dubins_env")
    try:
        shutil.copytree(src_env_dir, dst_env_dir)
        print(f"[backup] env -> {dst_env_dir}")
    except Exception as e:
        print(f"[WARN] copy env failed: {e}")

    src_script = os.path.abspath(__file__)
    dst_script = os.path.join(LOG_DIR, "train_ddpg_backup.py")
    try:
        shutil.copy2(src_script, dst_script)
        print(f"[backup] script -> {dst_script}")
    except Exception as e:
        print(f"[WARN] copy script failed: {e}")
backup_sources()
# ---------------- Train ----------------
def make_env_train():
    def _thunk():
        env = gym.make(ENV_ID)
        env = Monitor(env, LOG_DIR)   # 加上这一行：把训练环境包成 Monitor
        env.reset(seed=SEED)
        return env
    return _thunk

# 单环境（可切 SubprocVecEnv 并行）
# venv = DummyVecEnv([make_env_train()])
venv = DummyVecEnv([make_env_train() for _ in range(2)])
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
    buffer_size=400_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,  # 软更新系数
    gamma=0.99, 
    train_freq=1,
    gradient_steps=1,
    action_noise=action_noise,
    tensorboard_log=LOG_DIR,  # TensorBoard 日志也放在 LOG_DIR
    seed=SEED,
    verbose=1,
    policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256, 256], qf=[256, 256, 256, 256])),
)

# ---------------- 评估环境：也要用 Monitor，并且要用与训练相同的归一化统计 ----------------
def make_env_eval_for_callback():
    def _thunk():
        return Monitor(gym.make(ENV_ID), LOG_DIR)  # 评估也包 Monitor（会有 eval 的 npz + 也能离线画）
    return _thunk

eval_env = DummyVecEnv([make_env_eval_for_callback() for _ in range(10)])

eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
# 同步训练时的归一化统计到评估环境
sync_envs_normalization(venv, eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "best"),  # 会自动保存表现最好的模型
    log_path=LOG_DIR,          # 将 evaluations.npz 放在这里
    eval_freq=20_000,          # 每隔这么多 env steps 评估一次
    n_eval_episodes=100,
    deterministic=True,
)
print("[train] start")
t0 = time.time()
total_steps = 200_000
# 训练时加上 callback
model.learn(total_timesteps=total_steps, log_interval=10, callback=eval_callback)
t1 = time.time()
print(f"[train] done in {t1 - t0:.1f}s")
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
    """返回 (xs, ys, ths, start_state, goal_state, steps, is_success)"""
    obs = traj_venv.reset()
    start = base_env.state
    goal = base_env.goal

    xs, ys, ths = [start[0]], [start[1]], [start[2]]
    steps = 0
    is_success = False

    for _ in range(max_steps):
        action, _ = loaded.predict(obs, deterministic=True)
        obs, rewards, dones, infos = traj_venv.step(action)
        info0 = infos[0]

        if dones[0]:
            term = info0.get("terminal_state", None)
            if term is not None:
                xs.append(term[0]); ys.append(term[1]); ths.append(term[2])
            else:
                x, y, th = base_env.state
                xs.append(x); ys.append(y); ths.append(th)
            is_success = bool(info0.get("is_success", False))  # 兼容两种键名，且默认为False
            break
        else:
            st = info0.get("state", None)
            if st is not None:
                xs.append(st[0]); ys.append(st[1]); ths.append(st[2])
            else:
                x, y, th = base_env.state
                xs.append(x); ys.append(y); ths.append(th)
        steps += 1

    return xs, ys, ths, start, goal, steps, is_success


N = 8  # 画 N 条随机轨迹
trajectories = []
succ_cnt = 0
for i in range(N):
    xs, ys, ths, st, gl, steps, ok = rollout_one_episode()
    trajectories.append((xs, ys, ths, st, gl, steps, ok))
    succ_cnt += int(ok)

plt.figure(figsize=(7, 7))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# --- 取环境里的障碍参数并画出来 ---
oc = getattr(base_env, "obstacle_center", (0.0, 0.0))
orad = getattr(base_env, "obstacle_radius", 1.0)
ax = plt.gca()
ax.add_patch(Circle(oc, radius=orad, facecolor="k", edgecolor="k", alpha=0.15, lw=1.0))
# （可选）把安全边距也画出来，便于调参
safe_margin = getattr(base_env, "safe_margin", 0.0)
if safe_margin > 0:
    ax.add_patch(Circle(oc, radius=orad + safe_margin, fill=False, ls="--", lw=1.0, ec="k", alpha=0.25))

# --- 画轨迹 ---
arrow_stride = 10   # 每隔多少步画一个朝向箭头
arrow_len = 0.3     # 箭头长度（可按轨迹尺度调整）

for i, (xs, ys, ths, st, gl, steps, ok) in enumerate(trajectories):
    c = colors[i % len(colors)]
    lbl = f"ep{i+1} ({'✓' if ok else '×'}, {steps} steps)"

    # 轨迹线
    plt.plot(xs, ys, '-', lw=2, color=c, alpha=0.9, label=lbl)

    # 起点/终点位置
    plt.plot(xs[0], ys[0], 'o', color=c, ms=6, alpha=0.9)
    plt.plot(gl[0], gl[1], '*', color=c, ms=12, alpha=0.9)

    # 起点朝向箭头
    st_th = st[2]
    plt.arrow(st[0], st[1],
              2*arrow_len * np.cos(st_th), 2*arrow_len * np.sin(st_th),
              head_width=0.2, length_includes_head=True, color=c, alpha=0.9)

    # 终点（目标）朝向箭头
    gl_th = gl[2]
    plt.arrow(gl[0], gl[1],
              2*arrow_len * np.cos(gl_th), 2*arrow_len * np.sin(gl_th),
              head_width=0.2, length_includes_head=True, color=c, alpha=0.5)

    # 轨迹上按步距画朝向箭头（用 quiver 一次性更快）
    idx = np.arange(0, len(xs), arrow_stride)
    ux = arrow_len * np.cos(np.array(ths)[idx])
    uy = arrow_len * np.sin(np.array(ths)[idx])
    plt.quiver(np.array(xs)[idx], np.array(ys)[idx], ux, uy,
               angles='xy', scale_units='xy', scale=1.0,
               width=0.004, color=c, alpha=0.7)

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
