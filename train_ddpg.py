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

def visualize_critic_contour(
    log_dir: str,
    goal_pose: tuple,
    x_range: tuple = (-6.0, 6.0),
    y_range: tuple = (-6.0, 6.0),
    grid_N: int = 81,                  # 网格密度（奇数居中，建议 51/81/101）
    start_theta: float = 0.0,          # 起点朝向固定值
    deterministic: bool = True,
    use_min_q_for_td3: bool = True,    # TD3 用 min(Q1, Q2)
    cmap: str = "viridis",
    title: str = "Critic Q(s, π(s)) field",
):
    """
    固定 goal，在 (x_range, y_range) 上均匀采样起点 (x, y, start_theta)，
    对每个起点用当前策略 π(s) 得到动作 a，再用 critic 评估 Q(s, a)，绘制等值面。
    """
    # ---- 路径 / 设备 / 环境 & 归一化 ----
    alg_name = ALG
    model_path = os.path.join(log_dir, f"{alg_name}_kinematic_model")
    vecnorm_path = os.path.join(log_dir, f"{alg_name}_vecnorm.pkl")
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 单环境 + 载入 vecnorm 统计
    traj_venv = DummyVecEnv([make_env_for_traj()])
    traj_venv = VecNormalize.load(vecnorm_path, traj_venv)
    traj_venv.training = False
    traj_venv.norm_reward = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Algo.load(model_path, device=device)
    policy = model.policy  # 方便下面直接用 actor/critic
    base_env = traj_venv.venv.envs[0].unwrapped

    # ---- 生成网格 ----
    xs = np.linspace(x_range[0], x_range[1], grid_N)
    ys = np.linspace(y_range[0], y_range[1], grid_N)
    Qmap = np.full((grid_N, grid_N), np.nan, dtype=np.float32)

    # ---- 遍历网格并评估 Q ----
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            start = (float(x), float(y), float(start_theta))
            goal  = (float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2]))

            # 把“下次 reset”的起终点传入环境
            traj_venv.env_method("set_start_goal_for_next_reset", start=start, goal=goal)
            obs = traj_venv.reset()  # 注意：这里返回的 obs 已经过 VecNormalize 处理

            # 用策略得到动作（deterministic=True 时用均值）
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device).float().unsqueeze(0)  # [1, obs_dim]
                # 用 actor 先出一个动作
                act_t = model.policy.actor(obs_t)

                critic = getattr(model, "critic", None)
                if critic is None:
                    critic = getattr(model, "critic_target", None)
                assert critic is not None, "No critic found on the model."

                # 统一拿到 q1 / q2（如果有）
                q1_t, q2_t = None, None
                if hasattr(critic, "q1_forward"):                 # TD3 / SAC 的 ContinuousCritic
                    q1_t = critic.q1_forward(obs_t, act_t)        # [1, 1]
                    if hasattr(critic, "q2_forward"):
                        q2_t = critic.q2_forward(obs_t, act_t)    # [1, 1]
                else:
                    q1_t = critic(obs_t, act_t)                   # DDPG 的 QNetwork: [1, 1] 或 [1]
                    # 有些版本可能返回 (tensor,) 这种奇怪包装，统一取第 0 项
                    if isinstance(q1_t, (tuple, list)):
                        q1_t = q1_t[0]

                # 取最终用于画等高线的 Q 值（TD3 取 min(Q1, Q2)，DDPG 就是 Q1）
                if q2_t is not None:
                    q_val = torch.min(q1_t, q2_t)
                else:
                    q_val = q1_t

                q_val_float = float(q_val.squeeze().cpu().item())


            # 某些非法起点可能被 reset 时重采样（比如落在障碍里且强约束），这会导致
            # obs 对应的实际起点与设定不同，但对“全局可达性近似场”仍有参考意义。
            # 如需强制严格使用该点，可在 env.reset 里放宽/关闭障碍点过滤，或跳过这些点。
            Qmap[iy, ix] = float(q_val)  # 注意行列：y 为行，x 为列

    # ---- 绘制等值面 ----
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    X, Y = np.meshgrid(xs, ys)

    # 为了可读性，可做简单归一化（可选）
    finite_vals = Qmap[np.isfinite(Qmap)]
    if finite_vals.size > 0:
        vmin, vmax = np.percentile(finite_vals, [5, 95])  # 稍微抑制极端值
    else:
        vmin, vmax = None, None

    cs = ax.contourf(X, Y, Qmap, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(cs, ax=ax, shrink=0.8)
    cb.set_label("Q(s, π(s)) (higher = better)", rotation=90)

    # 画障碍 & 安全圈
    oc = getattr(base_env, "obstacle_center", (0.0, 0.0))
    orad = getattr(base_env, "obstacle_radius", 1.0)
    ax.add_patch(Circle(oc, radius=orad, facecolor="k", edgecolor="k", alpha=0.18, lw=1.0))
    safe_margin = getattr(base_env, "safe_margin", 0.0)
    if safe_margin > 0:
        ax.add_patch(Circle(oc, radius=orad + safe_margin, fill=False, ls="--", lw=1.0, ec="k", alpha=0.25))

    # 标出目标点与朝向
    xg, yg, thg = goal_pose
    ax.plot([xg], [yg], marker="*", markersize=12, color="black", alpha=0.9)
    Lvis = 0.6
    ax.arrow(xg, yg, Lvis*np.cos(thg), Lvis*np.sin(thg),
             head_width=0.25, length_includes_head=True, color="black", alpha=0.7)

    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlabel("x (start)")
    ax.set_ylabel("y (start)")
    ax.set_title(f"{title}\nGoal=({xg:.2f},{yg:.2f},{thg:.2f}), start_theta={start_theta:.2f}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(plot_dir, f"{ALG}_{ENV_ID}_critic_contour_{ts}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[contour] Saved Q-field to: {out_path}")

    traj_venv.close()

# ============ 轨迹可视化（可单独调用） ============

def visualize_trajectories(
    log_dir: str,
    alg_name: str = ALG,
    episodes: int = 16,
    max_steps: int = 400,
    deterministic: bool = True,
    title_suffix: str = "",
    # 👇 新增：可选固定起点/终点（对全部 episodes 生效）
    start_pose: tuple | None = None,  # (x, y, theta)
    goal_pose: tuple | None = None,   # (xg, yg, thetag)
    # 👇 可选：也支持为每条轨迹准备一对起终点（长度需与 episodes 一致）
    start_goals: list[tuple[tuple, tuple]] | None = None
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

    def rollout_one_episode(max_steps_=max_steps, ep_idx=0):
        # 如果用户提供了固定起终点/或每集一对的列表，则通过 env_method 注入
        if start_goals is not None:
            assert ep_idx < len(start_goals), "start_goals 长度必须 >= episodes"
            s, g = start_goals[ep_idx]
        else:
            s, g = start_pose, goal_pose

        if (s is not None) or (g is not None):
            # 通过 VecNormalize/DummyVecEnv 的 env_method 传到底层环境
            traj_venv.env_method("set_start_goal_for_next_reset", start=s, goal=g)

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
    for ep in range(episodes):
        xs, ys, ths, st, gl, steps, ok = rollout_one_episode(ep_idx=ep)
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
    p.add_argument("--mode", choices=["train", "plot", "train_and_plot", "contour"], default="train",
                   help="train: 只训练; plot: 只画图; train_and_plot: 训练完立刻画一次; contour: 画出等势线")
    # contour 专用
    p.add_argument("--goal", type=str, default=None, help='目标 "xg,yg,thg"（必填）')
    p.add_argument("--x-range", type=str, default="-6,6", help='起点 x 采样范围 "xmin,xmax"')
    p.add_argument("--y-range", type=str, default="-6,6", help='起点 y 采样范围 "ymin,ymax"')
    p.add_argument("--grid-N", type=int, default=81, help="网格密度（建议奇数）")
    p.add_argument("--start-theta", type=float, default=0.0, help="起点朝向")
    
    # 基本参数
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

    def _parse_triplet(s: str):
        # "a,b,c" -> (a, b, c)
        parts = [float(x) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError("需要 3 个数，例如: --goal \"5.0,4.0,1.57\"")
        return tuple(parts)

    def _parse_pair(s: str):
        # "a,b" -> (a, b)
        parts = [float(x) for x in s.split(",")]
        if len(parts) != 2:
            raise ValueError("需要 2 个数，例如: --x-range \"-6,6\"")
        return tuple(parts)

    args = parse_args()

    if args.mode in ["train", "train_and_plot"]:
        train_and_optionally_eval(args)

    if args.mode in ["plot", "train_and_plot"]:
        starts_goals = [
            ((-4.0, -4.0, 0.0), (4.0, 1, 0.0)),
            ((-4.0, -4.0, 0.0), (4.0, 1.5, 0.0)),
            ((-4.0, -4.0, 0.0), (4.0, 2, 0.0)),
            ((-4.0, -4.0, 0.0), (4.0, 2.5, 0.0)),
            ((-4.0, -4.0, 0.0), (4.0, 3, 0.0)),
            ((-4.0, -4.0, 0.0), (4.0, 3.5, 0.0)),
        ]
        visualize_trajectories(
            log_dir=args.log_dir,
            alg_name=ALG,
            # episodes=args.episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            title_suffix=args.mode,

            start_goals=starts_goals,
            episodes=len(starts_goals),

            # start_pose=(-4.0, -4.0, 0.0),    # 起点 (x, y, θ)
            # goal_pose=(2.0, 1.5, 0.0),      # 终点 (x, y, θ)
        )
        
    if args.mode == "contour":
        if args.goal is None:
            raise SystemExit("必须提供 --goal \"xg,yg,thg\"")
        goal_pose = _parse_triplet(args.goal)
        x_rng = _parse_pair(args["x_range"] if isinstance(args, dict) else args.x_range)
        y_rng = _parse_pair(args["y_range"] if isinstance(args, dict) else args.y_range)

        visualize_critic_contour(
            log_dir=args.log_dir,
            goal_pose=goal_pose,
            x_range=x_rng,
            y_range=y_rng,
            grid_N=args.grid_N,
            start_theta=args.start_theta,
            deterministic=True,
        )

