import os
import time
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from matplotlib.patches import Circle

import matplotlib.cm as cm
from matplotlib.colors import to_rgba

# ===== SB3 =====
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

# 触发环境注册（确保 __init__.py 注册了 KinematicCar-v0）
import kinodynamic_car_SB3.dubins_env  # noqa: F401

# ========= 全局配置（默认值，可被命令行覆盖） =========
ENV_ID = "KinematicCar-v0"
SEED = 42
ALG = "DDPG"                 # or "TD3"
Algo = DDPG if ALG == "DDPG" else TD3
N_ENVS = 24                  # 并行环境个数
EVAL_ENVS = 8                # 评估并行环境个数
TOTAL_STEPS = 300_000

# ----------------  备份源码（环境包 + 本脚本）----------------
def backup_sources(log_dir: str):
    src_env_dir = os.path.join(os.path.dirname(__file__), "dubins_env")
    dst_env_dir = os.path.join(log_dir, "dubins_env")
    try:
        if not os.path.exists(dst_env_dir):
            shutil.copytree(src_env_dir, dst_env_dir)
            print(f"[backup] env -> {dst_env_dir}")
        else:
            print(f"[backup] env already exists -> {dst_env_dir}")
    except Exception as e:
        print(f"[WARN] copy env failed: {e}")

    src_script = os.path.abspath(__file__)
    dst_script = os.path.join(log_dir, "train_ddpg_backup.py")
    try:
        shutil.copy2(src_script, dst_script)
        print(f"[backup] script -> {dst_script}")
    except Exception as e:
        print(f"[WARN] copy script failed: {e}")

# ============ VecEnv 工厂函数 ============

def make_env_train(rank: int, log_dir: str):
    def _thunk():
        env = gym.make(ENV_ID)
        env = Monitor(env, log_dir)
        env.reset(seed=SEED + rank)
        return env
    return _thunk

def make_env_eval_for_callback(rank: int, log_dir: str):
    def _thunk():
        return Monitor(gym.make(ENV_ID), log_dir)
    return _thunk

def make_env_eval():
    def _thunk():
        return Monitor(gym.make(ENV_ID))
    return _thunk

def make_env_for_traj():
    def _thunk():
        return Monitor(gym.make(ENV_ID))
    return _thunk

# ============ 轨迹可视化（可单独调用） ============

def visualize_trajectories(
    log_dir: str,
    alg_name: str = ALG,
    episodes: int = 16,
    max_steps: int = 400,
    deterministic: bool = True,
    title_suffix: str = ""
):
    """
    从 log_dir 读取 <ALG>_kinematic_model 与 <ALG>_vecnorm.pkl，画多条轨迹并保存到 plots/ 下。
    """
    model_path = os.path.join(log_dir, f"{alg_name}_kinematic_model")
    vecnorm_path = os.path.join(log_dir, f"{alg_name}_vecnorm.pkl")
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 单环境评估/采样
    traj_venv = DummyVecEnv([make_env_for_traj()])
    traj_venv = VecNormalize.load(vecnorm_path, traj_venv)
    traj_venv.training = False
    traj_venv.norm_reward = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = Algo.load(model_path, device=device)
    base_env = traj_venv.venv.envs[0].unwrapped

    def rollout_one_episode(max_steps_=max_steps):
        obs = traj_venv.reset()
        start = base_env.state
        goal = base_env.goal

        xs, ys, ths = [start[0]], [start[1]], [start[2]]
        steps = 0
        is_success = False

        for _ in range(max_steps_):
            action, _ = loaded.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = traj_venv.step(action)
            info0 = infos[0]
            if dones[0]:
                term = info0.get("terminal_state", None)
                if term is not None:
                    xs.append(term[0]); ys.append(term[1]); ths.append(term[2])
                else:
                    x, y, th = base_env.state
                    xs.append(x); ys.append(y); ths.append(th)
                is_success = bool(info0.get("is_success", False))
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

    # 收集多条轨迹
    trajectories = []
    succ_cnt = 0
    for _ in range(episodes):
        xs, ys, ths, st, gl, steps, ok = rollout_one_episode()
        trajectories.append((xs, ys, ths, st, gl, steps, ok))
        succ_cnt += int(ok)

    # 绘图
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    # -------------------- 🔸颜色生成策略--------------------
    # 1. 成功轨迹用绿色渐变 ('Greens')
    # 2. 失败轨迹用红色渐变 ('Reds')
    succ_trajs = [t for t in trajectories if t[-1] is True]
    fail_trajs = [t for t in trajectories if t[-1] is False]

    succ_cmap = cm.get_cmap('Greens', len(succ_trajs) + 2)
    fail_cmap = cm.get_cmap('Reds', len(fail_trajs) + 2)

    succ_colors = [succ_cmap(i + 1) for i in range(len(succ_trajs))]
    fail_colors = [fail_cmap(i + 1) for i in range(len(fail_trajs))]

    # 为统一顺序重组轨迹和颜色列表
    colors = []
    for t in trajectories:
        if t[-1]:
            colors.append(succ_colors.pop(0))
        else:
            colors.append(fail_colors.pop(0))
    # -----------------------------------------------------

    # 画障碍物 + 安全圈
    oc = getattr(base_env, "obstacle_center", (0.0, 0.0))
    orad = getattr(base_env, "obstacle_radius", 1.0)
    ax.add_patch(Circle(oc, radius=orad, facecolor="k", edgecolor="k", alpha=0.15, lw=1.0))
    safe_margin = getattr(base_env, "safe_margin", 0.0)
    if safe_margin > 0:
        ax.add_patch(Circle(oc, radius=orad + safe_margin, fill=False, ls="--", lw=1.0, ec="k", alpha=0.25))

    arrow_stride = 10
    arrow_len = 0.3

    for i, (xs, ys, ths, st, gl, steps, ok) in enumerate(trajectories):
        c = colors[i % len(colors)]
        lbl = f"ep{i+1} ({'✓' if ok else '×'}, {steps} steps)"
        plt.plot(xs, ys, '-', lw=2, color=c, alpha=0.9, label=lbl)
        plt.plot(xs[0], ys[0], 'o', color=c, ms=6, alpha=0.9)
        plt.plot(gl[0], gl[1], '*', color=c, ms=12, alpha=0.9)

        st_th = st[2]
        plt.arrow(st[0], st[1],
                  2*arrow_len*np.cos(st_th), 2*arrow_len*np.sin(st_th),
                  head_width=0.2, length_includes_head=True, color=c, alpha=0.9)
        gl_th = gl[2]
        plt.arrow(gl[0], gl[1],
                  2*arrow_len*np.cos(gl_th), 2*arrow_len*np.sin(gl_th),
                  head_width=0.2, length_includes_head=True, color=c, alpha=0.5)

        idx = np.arange(0, len(xs), arrow_stride)
        ux = arrow_len * np.cos(np.array(ths)[idx])
        uy = arrow_len * np.sin(np.array(ths)[idx])
        plt.quiver(np.array(xs)[idx], np.array(ys)[idx], ux, uy,
                   angles='xy', scale_units='xy', scale=1.0,
                   width=0.004, color=c, alpha=0.7)

    # 放在下方
    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=4,            # 分 4 列横排
        frameon=False,
        fontsize=8
    )
    plt.tight_layout()

    plt.axis('equal'); plt.grid(True, ls='--', alpha=0.5)
    plt.xlabel('x'); plt.ylabel('y')
    ttl = f"{ENV_ID} trajectories ({ALG}) | {succ_cnt}/{episodes} reached"
    if title_suffix:
        ttl += f" | {title_suffix}"
    plt.title(ttl)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(plot_dir, f"{ALG}_{ENV_ID}_multi_traj_{episodes}_{ts}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[plot] Saved trajectory figure to: {out_path}")

    traj_venv.close()

# ============ 训练主流程 ============

def train_and_optionally_eval(args):
    # 线程限制
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    LOG_DIR = args.log_dir
    PLOT_DIR = os.path.join(LOG_DIR, "plots")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    backup_sources(LOG_DIR)

    tmp = gym.make(ENV_ID)
    n_actions = tmp.action_space.shape[-1]
    tmp.close()

    venv = SubprocVecEnv([make_env_train(i, LOG_DIR) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    sigma = np.array([0.5, 0.4], dtype=np.float32) if n_actions == 2 else 0.3 * np.ones(n_actions, dtype=np.float32)
    action_noise = NormalActionNoise(mean=np.zeros(n_actions, dtype=np.float32), sigma=sigma)

    model = Algo(
        "MlpPolicy",
        venv,
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=3e-4,
        buffer_size=400_000,
        learning_starts=10_000,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        tensorboard_log=LOG_DIR,
        seed=SEED,
        verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256, 256, 256, 256, 256],
                                         qf=[256, 256, 256, 256, 256, 256])),
    )

    eval_env = SubprocVecEnv([make_env_eval_for_callback(i, LOG_DIR) for i in range(args.eval_envs)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    sync_envs_normalization(venv, eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "best"),
        log_path=LOG_DIR,
        eval_freq=50_000,
        n_eval_episodes=100,
        deterministic=True,
    )

    print("[train] start")
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, log_interval=10, callback=eval_callback)
    t1 = time.time()
    mins, secs = divmod(t1 - t0, 60)
    print(f"[train] done in {int(mins)}m{secs:.1f}s")

    model_path = os.path.join(LOG_DIR, f"{ALG}_kinematic_model")
    vecnorm_path = os.path.join(LOG_DIR, f"{ALG}_vecnorm.pkl")
    model.save(model_path)
    venv.save(vecnorm_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved VecNormalize stats to: {vecnorm_path}")

    # 简单 eval（可选）
    eval_venv = DummyVecEnv([make_env_eval()])
    eval_venv = VecNormalize.load(vecnorm_path, eval_venv)
    eval_venv.training = False
    eval_venv.norm_reward = False
    mean_r, std_r = evaluate_policy(model, eval_venv, n_eval_episodes=10, deterministic=True)
    print(f"[{ALG}] Eval return: {mean_r:.2f} ± {std_r:.2f}")

# ============ CLI 入口 ============

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "plot", "train_and_plot"], default="train",
                   help="train: 只训练; plot: 只画图; train_and_plot: 训练完立刻画一次")
    p.add_argument("--log-dir", type=str,
                   default="/workspace/kinodynamic_car_SB3/backup/with_obstacle_avoidance_v7/0",
                   help="保存/读取模型的目录")
    p.add_argument("--n-envs", type=int, default=N_ENVS, help="并行环境个数")
    p.add_argument("--eval-envs", type=int, default=EVAL_ENVS, help="评估并行环境个数")
    p.add_argument("--total-steps", type=int, default=TOTAL_STEPS, help="训练总步数")

    # 可视化专用参数
    p.add_argument("--episodes", type=int, default=16, help="可视化的轨迹条数")
    p.add_argument("--max-steps", type=int, default=400, help="每条轨迹的最大步数")
    p.add_argument("--deterministic", action="store_true", help="可视化时用确定性策略")
    return p.parse_args()

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    if args.mode in ["train", "train_and_plot"]:
        train_and_optionally_eval(args)

    if args.mode in ["plot", "train_and_plot"]:
        visualize_trajectories(
            log_dir=args.log_dir,
            alg_name=ALG,
            episodes=args.episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            title_suffix=args.mode
        )
