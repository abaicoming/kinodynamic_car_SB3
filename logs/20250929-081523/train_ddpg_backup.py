import os
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

# 触发注册 (KinematicCar-v0)
import kinodynamic_car_SB3.dubins_env  # noqa: F401


# Todo: 画出训练过程中的 reward 曲线
# read dubins_env.py
# Todo: 画出 contour line
# Todo: check why all trajectory plots are only 399 setps(初步观察，发现，所有测试过程中最大为399步，怀疑是max_episode_steps=400的锅) --- IGNORE ---
# 弄懂DDPG训练流程以及各参数含义，如什么时rollout，buffer_size，eposide以及length等
# 发现增加训练步数，提升率降低，总reward降低，怀疑是欠拟合

# ========= 全局配置 =========
ENV_ID = "KinematicCar-v0"
SEED = 42
ALG = "DDPG"  # or "TD3"
Algo = DDPG if ALG == "DDPG" else TD3

TS = time.strftime("%Y%m%d-%H%M%S")
ROOT_LOG_DIR = "/workspace/kinodynamic_car_SB3/logs"
LOG_DIR = os.path.join(ROOT_LOG_DIR, TS)
PLOT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# 备份源码（环境包 + 本脚本）
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


# ========= 工厂函数 =========
def make_env_train():
    def _thunk():
        env = gym.make(ENV_ID)
        env = Monitor(env, LOG_DIR)      # 记录 rollout/ep_rew_mean 等
        env.reset(seed=SEED)
        return env
    return _thunk

def make_env_eval_for_callback():
    def _thunk():
        return Monitor(gym.make(ENV_ID), LOG_DIR)
    return _thunk

def make_env_plain():
    def _thunk():
        return Monitor(gym.make(ENV_ID))
    return _thunk


# ========= 1) 训练 =========
def train(total_steps: int = 300_000):
    # 向量环境 + 归一化
    venv = DummyVecEnv([make_env_train()])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # 动作噪声
    tmp = gym.make(ENV_ID)
    n_actions = tmp.action_space.shape[-1]
    tmp.close()
    sigma = np.array([0.5, 0.4], dtype=np.float32) if n_actions == 2 else 0.3 * np.ones(n_actions, dtype=np.float32)
    action_noise = NormalActionNoise(mean=np.zeros(n_actions, dtype=np.float32), sigma=sigma)

    # 算法
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
        tensorboard_log=LOG_DIR,
        seed=SEED,
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
    )

    # 评估回调（定期写 eval/mean_reward）
    eval_env = DummyVecEnv([make_env_eval_for_callback()])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    sync_envs_normalization(venv, eval_env)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best"),
        log_path=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print("[train] start")
    t0 = time.time()
    model.learn(total_timesteps=total_steps, log_interval=10, callback=eval_cb)
    t1 = time.time()
    print(f"[train] done in {t1 - t0:.1f}s")

    # 保存
    model_path = os.path.join(LOG_DIR, f"{ALG}_kinematic_model")
    vecnorm_path = os.path.join(LOG_DIR, f"{ALG}_vecnorm.pkl")
    model.save(model_path)
    venv.save(vecnorm_path)
    print(f"[train] saved model -> {model_path}")
    print(f"[train] saved vecnorm -> {vecnorm_path}")

    return model_path, vecnorm_path


# ========= 2) 评估 =========
def evaluate(model_path: str, vecnorm_path: str, n_eval_episodes: int = 10):
    model = Algo.load(model_path, device="auto")

    eval_venv = DummyVecEnv([make_env_plain()])
    eval_venv = VecNormalize.load(vecnorm_path, eval_venv)
    eval_venv.training = False
    eval_venv.norm_reward = False

    mean_r, std_r = evaluate_policy(model, eval_venv, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"[eval] return: {mean_r:.2f} ± {std_r:.2f}")
    return mean_r, std_r


# ========= 3) 批量轨迹可视化 =========
def _set_start_goal_and_get_obs(traj_venv, base_env, start_pose, goal_pose):
    sx, sy, sth = start_pose
    gx, gy, gth = goal_pose
    base_env.state = (float(sx), float(sy), float(sth))
    base_env.goal = (float(gx), float(gy), float(gth))
    base_env._elapsed_steps = 0
    base_env.u0 = 0.0
    base_env.u1 = 0.0

    raw_obs = base_env._get_obs().astype(np.float32)
    batched_raw = raw_obs[None, :]
    norm_obs = traj_venv.normalize_obs(batched_raw)
    return norm_obs

def rollout_one_episode(traj_venv, base_env, model, start_pose, goal_pose, max_steps=800):
    _ = traj_venv.reset()
    obs = _set_start_goal_and_get_obs(traj_venv, base_env, start_pose, goal_pose)

    start = base_env.state
    goal = base_env.goal

    xs, ys = [start[0]], [start[1]]
    steps, is_success = 0, False

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = traj_venv.step(action)
        info0 = infos[0]

        if dones[0]:
            term = info0.get("terminal_state", None)
            if term is not None:
                xs.append(term[0]); ys.append(term[1])
            is_success = bool(info0.get("is_success", True))
            break
        else:
            st = info0.get("state", None)
            if st is not None:
                xs.append(st[0]); ys.append(st[1])
            else:
                x, y, _ = base_env.state
                xs.append(x); ys.append(y)
        steps += 1

    return xs, ys, start, goal, steps, is_success

def visualize_batch(model_path: str, vecnorm_path: str,
                    starts_grid=((0.0, 0.0, 0.0),),
                    goal_pose=(3.0, 3.0, 0.0),
                    fig_name_prefix="multi_traj"):
    model = Algo.load(model_path, device="auto")

    traj_venv = DummyVecEnv([make_env_plain()])
    traj_venv = VecNormalize.load(vecnorm_path, traj_venv)
    traj_venv.training = False
    traj_venv.norm_reward = False

    base_env = traj_venv.venv.envs[0].unwrapped

    trajectories, succ_cnt = [], 0
    for sp in starts_grid:
        xs, ys, st, gl, steps, ok = rollout_one_episode(
            traj_venv, base_env, model, start_pose=sp, goal_pose=goal_pose, max_steps=800
        )
        trajectories.append((xs, ys, st, gl, steps, ok))
        succ_cnt += int(ok)

    plt.figure(figsize=(7, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (xs, ys, st, gl, steps, ok) in enumerate(trajectories):
        c = colors[i % len(colors)]
        lbl = f"ep{i+1} ({'✓' if ok else 'x'}, {steps} steps)"
        plt.plot(xs, ys, '-', lw=2, color=c, alpha=0.9, label=lbl)
        plt.plot(xs[0], ys[0], 'o', color=c, ms=6, alpha=0.9)
        plt.plot(gl[0], gl[1], '*', color=c, ms=12, alpha=0.9)

    plt.legend()
    plt.axis('equal'); plt.grid(True, ls='--', alpha=0.5)
    plt.xlabel('x'); plt.ylabel('y')
    plt.title(f"{ENV_ID} ({ALG}) | success {succ_cnt}/{len(starts_grid)}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(PLOT_DIR, f"{fig_name_prefix}_{len(starts_grid)}_{ts}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[viz] saved -> {out_path}")
    # plt.show()
    traj_venv.close()


# ========= main =========
if __name__ == "__main__":
    print(f"[logdir] {LOG_DIR}")
    backup_sources()

    # 1) 训练
    model_path, vecnorm_path = train(total_steps=200_000)

    # 2) 评估
    evaluate(model_path, vecnorm_path, n_eval_episodes=10)

    # 3) 轨迹可视化（示例：3x3 网格起点）
    starts = []
    for i in range(3):
        for j in range(3):
            starts.append((2.0 * i, 2.0 * j, 0.0))
    visualize_batch(
        model_path, vecnorm_path,
        starts_grid=starts,
        goal_pose=(3.0, 3.0, 0.0),
        fig_name_prefix="grid9"
    )
