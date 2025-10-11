import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import shutil

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# 评估回调
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization # 同步归一化统计

# Todo: 画出训练过程中的 reward 曲线
# read dubins_env.py
# Todo: 画出 contour line
# Todo: check why all trajectory plots are only 399 setps(初步观察，发现，所有测试过程中最大为399步，怀疑是max_episode_steps=400的锅) --- IGNORE ---
# 弄懂DDPG训练流程以及各参数含义，如什么时rollout，buffer_size，eposide以及length等

# 触发环境注册（确保 __init__.py 注册了 KinematicCar-v0）
import kinodynamic_car_SB3.dubins_env  # noqa: F401

ENV_ID = "KinematicCar-v0"
SEED = 42
ALG = "DDPG"  # or "TD3"
Algo = DDPG if ALG == "DDPG" else TD3

# 统一日志/模型目录
ts = time.strftime("%Y%m%d-%H%M%S")
# ts = 20250927-174951
LOG_DIR = "/workspace/kinodynamic_car_SB3/logs"
LOG_DIR = os.path.join(LOG_DIR, f"{ts}")
# LOG_DIR = os.path.join(LOG_DIR, "20250927-174951")
PLOT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Logging to: {PLOT_DIR}")

# 保存 dubins_env 源码
SRC_ENV_DIR = os.path.join(os.path.dirname(__file__), "dubins_env")
BACKUP_ENV_DIR = os.path.join(LOG_DIR, "dubins_env")
try:
    shutil.copytree(SRC_ENV_DIR, BACKUP_ENV_DIR)
    print(f"Copied env source from {SRC_ENV_DIR} -> {BACKUP_ENV_DIR}")
except Exception as e:
    print(f"[WARN] Could not copy env source: {e}")

# 保存当前 train_ddpg.py 自身
SRC_SCRIPT = os.path.abspath(__file__)
BACKUP_SCRIPT = os.path.join(LOG_DIR, "train_ddpg_backup.py")
try:
    shutil.copy2(SRC_SCRIPT, BACKUP_SCRIPT)
    print(f"Copied script from {SRC_SCRIPT} -> {BACKUP_SCRIPT}")
except Exception as e:
    print(f"[WARN] Could not copy train_ddpg.py: {e}")

# ---------------- Train ----------------
def make_env_train():
    def _thunk():
        env = gym.make(ENV_ID)
        env = Monitor(env, LOG_DIR) # 把训练环境包成 Monitor 以记录 episode reward/length
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
    tau=0.005,  #软更新系数
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,  # 每步环境交互更新一次网络（off-policy 算法常见设置）
    action_noise=action_noise,
    tensorboard_log=LOG_DIR,  # TensorBoard 日志也放在 LOG_DIR
    seed=SEED,
    verbose=1,    # 日志显示
    policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
)


# 评估环境：也要用 Monitor，并且要用与训练相同的归一化统计
def make_env_eval_for_callback():
    def _thunk():
        return Monitor(gym.make(ENV_ID), LOG_DIR)  # 评估也包 Monitor（会有 eval 的 npz + 也能离线画）
    return _thunk

eval_env = DummyVecEnv([make_env_eval_for_callback()])
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
# 同步训练时的归一化统计到评估环境
sync_envs_normalization(venv, eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "best"),  # 会自动保存表现最好的模型
    log_path=LOG_DIR,          # 将 evaluations.npz 放在这里
    eval_freq=10_000,          # 每隔这么多 env steps 评估一次
    n_eval_episodes=5,
    deterministic=True,
)


total_steps = 300_000
print("Starting training...")
strat = time.time()
model.learn(total_timesteps=total_steps, log_interval=10, callback=eval_callback)
end = time.time()
print(f"Training completed in {end - strat:.2f} seconds.")
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

mean_r, std_r = evaluate_policy(model, eval_venv, n_eval_episodes=10, deterministic=True) # 'deterministic=True',评估时不加探索噪声;  每隔n_eval_episodes个episode评估一次智能体
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
# 把 Monitor/Gym 包裹剥掉，拿到自定义的 KinematicCarEnv 实例，这样可以读取 state、goal 这些环境内部变量（标准 Gym API 里看不到）。
base_env = traj_venv.venv.envs[0].unwrapped  

def _set_start_goal_and_get_obs(traj_venv, base_env, start_pose, goal_pose):
    """
    在 VecNormalize 外层的情况下，手动设置底层环境起终点，并返回“归一化后的 batched obs” (1, obs_dim)。
    """
    # 1) 覆盖底层环境的起点与终点
    sx, sy, sth = start_pose
    gx, gy, gth = goal_pose
    base_env.state = (float(sx), float(sy), float(sth))
    base_env.goal  = (float(gx), float(gy), float(gth))
    base_env._elapsed_steps = 0
    base_env.u0 = 0.0
    base_env.u1 = 0.0

    # 2) 拿“原始obs”，再用 VecNormalize 归一化成 batched 观测
    raw_obs = base_env._get_obs().astype(np.float32)           # (obs_dim,)
    batched_raw = raw_obs[None, :]                              # (1, obs_dim)
    norm_obs = traj_venv.normalize_obs(batched_raw)             # (1, obs_dim)
    return norm_obs

def rollout_one_episode(start_pose, max_steps=800):
    """返回 (xs, ys, start_state, goal_state, steps, is_success)"""
    # --- 你想要的起点/终点 ---
    # start_pose = (0.0, 0.0, 0.0)   # 起点 (x, y, theta)
    goal_pose  = (3.0, 3.0, 0.0)   # 终点 (xg, yg, thetag)

    # 先做一个“普通 reset”以初始化 VecNormalize 的内部状态（不要带 options）
    _ = traj_venv.reset()

    # 然后手动设置底层 env 的 state/goal，并取“归一化后的 obs”
    obs = _set_start_goal_and_get_obs(traj_venv, base_env, start_pose, goal_pose)

    start = base_env.state
    goal  = base_env.goal

    xs, ys = [start[0]], [start[1]]
    steps = 0
    is_success = False

    for _ in range(max_steps):
        action, _ = loaded.predict(obs, deterministic=True)
        obs, rewards, dones, infos = traj_venv.step(action)
        info0 = infos[0]

        if dones[0]:
            # 终止时“正确的终点”
            term = info0.get("terminal_state", None)
            if term is not None:
                xs.append(term[0]); ys.append(term[1])
            is_success = bool(info0.get("is_success", True))
            break
        else:
            # 正常步：追加这一步后的状态
            st = info0.get("state", None)
            if st is not None:
                xs.append(st[0]); ys.append(st[1])
            else:
                # 兜底（不推荐，但以防你忘了加info["state"]）
                x, y, _ = base_env.state
                xs.append(x); ys.append(y)
        steps += 1

    return xs, ys, start, goal, steps, is_success


N = 3  # 画 N 条随机轨迹
trajectories = []
succ_cnt = 0
for i in range(N):
    for j in range(N):
        xs, ys, st, gl, steps, ok = rollout_one_episode(start_pose = (2*i, 2*j, 0.0))
        trajectories.append((xs, ys, st, gl, steps, ok))
        succ_cnt += int(ok)
    # print(i)
# print("trajectories[0]:, ", trajectories[0])
plt.figure(figsize=(7, 7))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, (xs, ys, st, gl, steps, ok) in enumerate(trajectories):
    c = colors[i % len(colors)]
    lbl = f"ep{i+1} ({'✓' if ok else 'x'}, {steps} steps)"
    plt.plot(xs, ys, '-', lw=2, color=c, alpha=0.9, label=lbl)
    plt.plot(xs[0], ys[0], 'o', color=c, ms=6, alpha=0.9)   # start
    plt.plot(gl[0], gl[1], '*', color=c, ms=12, alpha=0.9)  # goal

plt.legend()
plt.axis('equal'); plt.grid(True, ls='--', alpha=0.5)
plt.xlabel('x'); plt.ylabel('y')
plt.title(f"{ENV_ID} trajectories ({ALG}) | {succ_cnt}/{N*N} reached")

ts = time.strftime("%Y%m%d-%H%M%S")
out_path = os.path.join(PLOT_DIR, f"{ALG}_{ENV_ID}_multi_traj_{N*N}_{ts}.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved trajectory figure to: {out_path}")
# plt.show()

traj_venv.close()
