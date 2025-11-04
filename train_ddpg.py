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

# ===== SB3 / Contrib =====
from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization
try:
    from sb3_contrib import TRPO
    HAS_TRPO = True
except Exception:
    TRPO = None
    HAS_TRPO = False

# 触发环境注册（确保 __init__.py 注册了 KinematicCar-v0）
import kinodynamic_car_SB3.dubins_env  # noqa: F401

# ========= 全局配置（默认值，可被命令行覆盖） =========
ENV_ID = "KinematicCar-v0"
SEED = 42

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

# ============ 通用：选择算法 & policy_kwargs ============
ALG_CHOICES = ["ddpg", "td3", "sac"] + (["trpo"] if HAS_TRPO else [])

def build_algo(algo_name: str):
    name = algo_name.lower()
    if name == "ddpg": return DDPG
    if name == "td3":  return TD3
    if name == "sac":  return SAC
    if name == "trpo":
        if not HAS_TRPO:
            raise ImportError("需要安装 sb3-contrib 才能使用 TRPO：pip install sb3-contrib==<同SB3版本>")
        return TRPO
    raise ValueError(f"Unsupported alg: {algo_name}")

def default_policy_kwargs(algo_name: str, hidden=([256]*6)):
    name = algo_name.lower()
    if name in ["ddpg", "td3"]:
        # actor / critic 均为 MLP，分别用 pi / qf
        return dict(net_arch=dict(pi=list(hidden), qf=list(hidden)))
    if name == "sac":
        # SAC: pi / qf
        return dict(net_arch=dict(pi=list(hidden), qf=list(hidden)))
    if name == "trpo":
        # TRPO: on-policy，使用 pi / vf 两支
        return dict(net_arch=dict(pi=list(hidden), vf=list(hidden)))
    raise ValueError

# ============ Critic/Value 等值面（可单独调用） ============
def visualize_critic_contour(
    log_dir: str,
    alg_name: str,
    goal_pose: tuple,
    x_range: tuple = (-6.0, 6.0),
    y_range: tuple = (-6.0, 6.0),
    grid_N: int = 81,
    start_theta: float = 0.0,
    cmap: str = "viridis",
    title: str = "Critic Q(s, π(s)) field",
):
    """
    固定 goal，在 (x_range, y_range) 上均匀采样起点 (x, y, start_theta)，
    DDPG/TD3/SAC: 画 Q(s, π(s))
    TRPO:         画 V(s)
    """
    model_path = os.path.join(log_dir, f"{alg_name}_kinematic_model")
    vecnorm_path = os.path.join(log_dir, f"{alg_name}_vecnorm.pkl")
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    traj_venv = DummyVecEnv([make_env_for_traj()])
    traj_venv = VecNormalize.load(vecnorm_path, traj_venv)
    traj_venv.training = False
    traj_venv.norm_reward = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Algo = build_algo(alg_name)
    model = Algo.load(model_path, device=device)
    base_env = traj_venv.venv.envs[0].unwrapped

    xs = np.linspace(x_range[0], x_range[1], grid_N)
    ys = np.linspace(y_range[0], y_range[1], grid_N)
    Qmap = np.full((grid_N, grid_N), np.nan, dtype=np.float32)

    is_trpo = alg_name.lower() == "trpo"

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            start = (float(x), float(y), float(start_theta))
            goal  = (float(goal_pose[0]), float(goal_pose[1]), float(goal_pose[2]))

            traj_venv.env_method("set_start_goal_for_next_reset", start=start, goal=goal)
            obs = traj_venv.reset()  # 已被 VecNormalize 处理（obs_norm）

            # —— 用策略得到动作（用于 Q(s,a)），或直接算 V(s)（TRPO）——
            if is_trpo:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=device).float()
                    v_t = model.policy.predict_values(obs_t)  # [1,1]
                    val = float(v_t.squeeze().cpu().item())
                Qmap[iy, ix] = val
                continue

            # OFF-policy：得到 π(s)
            action, _ = model.predict(obs, deterministic=True)
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, device=device).float()
                act_t = torch.as_tensor(action, device=device).float()

                critic = getattr(model, "critic", None)
                if critic is None:
                    critic = getattr(model, "critic_target", None)
                assert critic is not None, "No critic found on the model."

                q1_t, q2_t = None, None
                if hasattr(critic, "q1_forward"):            # TD3/SAC ContinuousCritic
                    q1_t = critic.q1_forward(obs_t, act_t)
                    q2_t = critic.q2_forward(obs_t, act_t) if hasattr(critic, "q2_forward") else None
                else:                                         # DDPG QNetwork
                    q1_t = critic(obs_t, act_t)
                    if isinstance(q1_t, (tuple, list)):
                        q1_t = q1_t[0]

                if q2_t is not None:
                    q_val = torch.min(q1_t, q2_t)
                else:
                    q_val = q1_t

                Qmap[iy, ix] = float(q_val.squeeze().cpu().item())

    # 绘图
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    X, Y = np.meshgrid(xs, ys)

    finite_vals = Qmap[np.isfinite(Qmap)]
    if finite_vals.size > 0:
        vmin, vmax = np.percentile(finite_vals, [5, 95])
    else:
        vmin, vmax = None, None

    label = "V(s)" if is_trpo else "Q(s, π(s)) (higher = better)"
    title2 = "Value V(s) field" if is_trpo else title

    cs = ax.contourf(X, Y, Qmap, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(cs, ax=ax, shrink=0.8); cb.set_label(label, rotation=90)

    # 画障碍/安全圈
    oc = getattr(base_env, "obstacle_center", (0.0, 0.0))
    orad = getattr(base_env, "obstacle_radius", 1.0)
    ax.add_patch(Circle(oc, radius=orad, facecolor="k", edgecolor="k", alpha=0.18, lw=1.0))
    safe_margin = getattr(base_env, "safe_margin", 0.0)
    if safe_margin > 0:
        ax.add_patch(Circle(oc, radius=orad + safe_margin, fill=False, ls="--", lw=1.0, ec="k", alpha=0.25))

    # 标出目标
    xg, yg, thg = goal_pose
    ax.plot([xg], [yg], marker="*", markersize=12, color="black", alpha=0.9)
    Lvis = 0.6
    ax.arrow(xg, yg, Lvis*np.cos(thg), Lvis*np.sin(thg),
             head_width=0.25, length_includes_head=True, color="black", alpha=0.7)

    ax.set_aspect("equal"); ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlabel("x (start)"); ax.set_ylabel("y (start)")
    ax.set_title(f"{title2}\nGoal=({xg:.2f},{yg:.2f},{thg:.2f}), start_theta={start_theta:.2f}")

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(plot_dir, f"{alg_name}_{ENV_ID}_contour_{ts}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[contour] Saved field to: {out_path}")

    traj_venv.close()

# 采样起点
def make_start_goals_in_box(
    x_range, y_range, th_range,
    goal_pose,
    mode="grid",      # "grid" 或 "random"
    nx=6, ny=6, nth=9,
    n_samples=120,    # random 模式使用
    seed=42,
):
    """
    在 [x_range]×[y_range]×[th_range] 里采样起点；返回 start_goals 列表:
    [ ((x,y,th), goal_pose), ... ]

    mode="grid": 均匀网格，数量 nx*ny*nth
    mode="random": 均匀随机，共 n_samples
    """
    rng = np.random.default_rng(seed)

    starts = []
    if mode == "grid":
        xs  = np.linspace(x_range[0],  x_range[1],  nx)
        ys  = np.linspace(y_range[0],  y_range[1],  ny)
        ths = np.linspace(th_range[0], th_range[1], nth)
        for x in xs:
            for y in ys:
                for th in ths:
                    starts.append((float(x), float(y), float(th)))
    elif mode == "random":
        for _ in range(n_samples):
            x  = rng.uniform(x_range[0],  x_range[1])
            y  = rng.uniform(y_range[0],  y_range[1])
            th = rng.uniform(th_range[0], th_range[1])
            starts.append((float(x), float(y), float(th)))
    else:
        raise ValueError("mode 必须是 'grid' 或 'random'")

    return [ (s, goal_pose) for s in starts ]

# ============ 轨迹可视化（可单独调用） ============
def visualize_trajectories(
    log_dir: str,
    alg_name: str,
    episodes: int = 16,
    max_steps: int = 1000,
    deterministic: bool = True,
    title_suffix: str = "",
    start_pose: tuple | None = None,   # (x, y, theta)
    goal_pose:  tuple | None = None,   # (xg, yg, thetag)
    start_goals: list[tuple[tuple, tuple]] | None = None
):
    model_path = os.path.join(log_dir, f"{alg_name}_kinematic_model")
    vecnorm_path = os.path.join(log_dir, f"{alg_name}_vecnorm.pkl")
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    traj_venv = DummyVecEnv([make_env_for_traj()])
    traj_venv = VecNormalize.load(vecnorm_path, traj_venv)
    traj_venv.training = False
    traj_venv.norm_reward = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Algo = build_algo(alg_name)
    loaded = Algo.load(model_path, device=device)
    base_env = traj_venv.venv.envs[0].unwrapped

    def rollout_one_episode(ep_idx=0):
        s, g = (start_pose, goal_pose)
        if start_goals is not None:
            assert ep_idx < len(start_goals), "start_goals 长度不足"
            s, g = start_goals[ep_idx]

        if (s is not None) or (g is not None):
            traj_venv.env_method("set_start_goal_for_next_reset", start=s, goal=g)

        obs = traj_venv.reset()
        start = base_env.state
        goal = base_env.goal

        xs, ys, ths = [start[0]], [start[1]], [start[2]]
        steps = 0; is_success = False

        for _ in range(max_steps):
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

    # 收集
    trajectories = []
    succ_cnt = 0
    for ep in range(episodes):
        xs, ys, ths, st, gl, steps, ok = rollout_one_episode(ep_idx=ep)
        trajectories.append((xs, ys, ths, st, gl, steps, ok))
        succ_cnt += int(ok)

    # 画图（成功=绿色渐变；失败=红色渐变）
    plt.figure(figsize=(7, 7)); ax = plt.gca()
    succ_trajs = [t for t in trajectories if t[-1] is True]
    fail_trajs = [t for t in trajectories if t[-1] is False]
    succ_cmap = cm.get_cmap('Greens', len(succ_trajs)+2)
    fail_cmap = cm.get_cmap('Reds',   len(fail_trajs)+2)
    succ_colors = [succ_cmap(i+1) for i in range(len(succ_trajs))]
    fail_colors = [fail_cmap(i+1) for i in range(len(fail_trajs))]
    colors = []
    for t in trajectories:
        colors.append(succ_colors.pop(0) if t[-1] else fail_colors.pop(0))

    # 障碍/安全圈
    oc = getattr(base_env, "obstacle_center", (0.0, 0.0))
    orad = getattr(base_env, "obstacle_radius", 1.0)
    ax.add_patch(Circle(oc, radius=orad, facecolor="k", edgecolor="k", alpha=0.15, lw=1.0))
    safe_margin = getattr(base_env, "safe_margin", 0.0)
    if safe_margin > 0:
        ax.add_patch(Circle(oc, radius=orad+safe_margin, fill=False, ls="--", lw=1.0, ec="k", alpha=0.25))

    arrow_stride = 10; arrow_len = 0.3
    for i, (xs, ys, ths, st, gl, steps, ok) in enumerate(trajectories):
        c = colors[i % len(colors)]
        lbl = f"ep{i+1} ({'✓' if ok else '×'}, {steps} steps)"
        plt.plot(xs, ys, '-', lw=2, color=c, alpha=0.9, label=lbl)
        plt.plot(xs[0], ys[0], 'o', color=c, ms=6, alpha=0.9)
        plt.plot(gl[0], gl[1], '*', color=c, ms=12, alpha=0.9)

        st_th = st[2]
        plt.arrow(st[0], st[1], 2*arrow_len*np.cos(st_th), 2*arrow_len*np.sin(st_th),
                  head_width=0.2, length_includes_head=True, color=c, alpha=0.9)
        gl_th = gl[2]
        plt.arrow(gl[0], gl[1], 2*arrow_len*np.cos(gl_th), 2*arrow_len*np.sin(gl_th),
                  head_width=0.2, length_includes_head=True, color=c, alpha=0.5)

        idx = np.arange(0, len(xs), arrow_stride)
        ux = arrow_len * np.cos(np.array(ths)[idx])
        uy = arrow_len * np.sin(np.array(ths)[idx])
        plt.quiver(np.array(xs)[idx], np.array(ys)[idx], ux, uy,
                   angles='xy', scale_units='xy', scale=1.0, width=0.004, color=c, alpha=0.7)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=8)
    plt.tight_layout()
    plt.axis('equal'); plt.grid(True, ls='--', alpha=0.5)
    plt.xlabel('x'); plt.ylabel('y')
    ttl = f"{ENV_ID} trajectories ({alg_name.upper()}) | {succ_cnt}/{episodes} reached"
    if title_suffix: ttl += f" | {title_suffix}"
    plt.title(ttl)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(plot_dir, f"{alg_name}_{ENV_ID}_multi_traj_{episodes}_{ts}.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[plot] Saved trajectory figure to: {out_path}")

    traj_venv.close()

# ============ 训练主流程 ============
def train_and_optionally_eval(args):
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    ALG = args.alg.lower()
    LOG_DIR = args.log_dir
    PLOT_DIR = os.path.join(LOG_DIR, "plots")
    os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(PLOT_DIR, exist_ok=True)

    backup_sources(LOG_DIR)

    tmp = gym.make(ENV_ID); n_actions = tmp.action_space.shape[-1]; tmp.close()

    venv = SubprocVecEnv([make_env_train(i, LOG_DIR) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # 动作噪声：仅 DDPG/TD3 用
    action_noise = None
    if ALG in ["ddpg", "td3"]:
        sigma = np.array([0.5, 0.4], dtype=np.float32) if n_actions == 2 else 0.3*np.ones(n_actions, np.float32)
        action_noise = NormalActionNoise(mean=np.zeros(n_actions, np.float32), sigma=sigma)

    Algo = build_algo(ALG)
    pk = default_policy_kwargs(ALG, hidden=[256]*6) # ddpg net_arch

    # 不同算法的关键超参
    algo_kwargs = dict(
        policy="MlpPolicy",
        env=venv,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=LOG_DIR,
        seed=SEED,
        verbose=1,
        policy_kwargs=pk,
    )

    if ALG in ["ddpg", "td3"]:
        algo_kwargs.update(dict(
            learning_rate=3e-4,
            buffer_size=400_000,
            learning_starts=10_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=action_noise,
        ))
    elif ALG == "sac":
        algo_kwargs.update(dict(
            learning_rate=1e-4,
            buffer_size=1_000_000,
            learning_starts=50_000,
            batch_size=1024,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            # 固定熵系数，不用自动调整
            ent_coef=0.05,        # ✅ 固定一个小值
            # 不要再传 target_entropy=None！
            policy_kwargs=dict(
                net_arch=dict(pi=[256,256,256,256,256], qf=[256,256,256,256,256]),
                use_sde=True,
                log_std_init=-2.0,
            ),
        ))
    elif ALG == "trpo":
        # TRPO（on-policy）
        algo_kwargs.update(dict(
            learning_rate=3e-4,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_kl=0.01,
            cg_damping=0.01,
        ))

    model = Algo(**algo_kwargs)

    # 评估环境
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

    print(f"[train:{ALG.upper()}] start")
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, log_interval=10, callback=eval_callback)
    t1 = time.time()
    mins, secs = divmod(t1 - t0, 60)
    print(f"[train] done in {int(mins)}m{secs:.1f}s")

    model_path = os.path.join(LOG_DIR, f"{ALG}_kinematic_model")
    vecnorm_path = os.path.join(LOG_DIR, f"{ALG}_vecnorm.pkl")
    model.save(model_path); venv.save(vecnorm_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved VecNormalize stats to: {vecnorm_path}")

    # 简单 eval
    eval_venv = DummyVecEnv([make_env_eval()])
    eval_venv = VecNormalize.load(vecnorm_path, eval_venv)
    eval_venv.training = False; eval_venv.norm_reward = False
    mean_r, std_r = evaluate_policy(model, eval_venv, n_eval_episodes=10, deterministic=True)
    print(f"[{ALG.upper()}] Eval return: {mean_r:.2f} ± {std_r:.2f}")

# ============ CLI ============
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alg", choices=ALG_CHOICES, default="ddpg", help="选择算法：ddpg/td3/sac/trpo")

    p.add_argument("--mode", choices=["train", "plot", "train_and_plot", "contour"], default="train",
                   help="train: 只训练; plot: 只画图; train_and_plot: 训练+画图; contour: 画等值面")

    # contour 专用
    p.add_argument("--goal", type=str, default="1.5,0.0,0.0", help='目标 "xg,yg,thg"（contour 必填）')
    p.add_argument("--x-range", nargs=2, type=float, metavar=("XMIN", "XMAX"), default=(-6.0, 6.0))
    p.add_argument("--y-range", nargs=2, type=float, metavar=("YMIN", "YMAX"), default=(-6.0, 6.0))

    p.add_argument("--grid-N", type=int, default=101, help="网格密度（建议奇数）")
    p.add_argument("--start-theta", type=float, default=0.0, help="起点朝向")

    # 基本参数
    p.add_argument("--log-dir", type=str,
                   default="/workspace/kinodynamic_car_SB3/backup/with_obstacle_avoidance_v7/0",
                   help="保存/读取模型的目录")
    p.add_argument("--n-envs", type=int, default=32, help="并行环境个数")
    p.add_argument("--eval-envs", type=int, default=16, help="评估并行环境个数")
    p.add_argument("--total-steps", type=int, default=400_000, help="训练总步数")

    # 轨迹图参数
    p.add_argument("--episodes", type=int, default=16, help="可视化的轨迹条数")
    p.add_argument("--max-steps", type=int, default=800, help="每条轨迹的最大步数")
    p.add_argument("--deterministic", action="store_true", help="可视化时用确定性策略")

    # 采样盒子与方式
    p.add_argument("--box", type=str,
                default="-4,-2,-2,2,-3.141592653589793,3.141592653589793",  # [-4,-2,-pi] 到 [-2,2,pi]
                help='起点盒子 "xlo,xhi,ylo,yhi,thlo,thhi"')
    p.add_argument("--sample-mode", choices=["grid","random", "specified"], default="specified",
                help="评测起点采样方式：grid 或 random")
    p.add_argument("--nx", type=int, default=6, help="grid: x 方向点数")
    p.add_argument("--ny", type=int, default=4, help="grid: y 方向点数")
    p.add_argument("--nth", type=int, default=9, help="grid: theta 方向点数")
    p.add_argument("--n-samples", type=int, default=120, help="random: 总样本数")
    p.add_argument("--sample-seed", type=int, default=42, help="random: 采样随机种子")
    p.add_argument("--save-samples", action="store_true", help="是否把采样的起点集合保存成 npz")

    return p.parse_args()

def _parse_box6(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 6:
        raise ValueError('需要 6 个数，例如: --box "-4,-2,-2,2,-3.141592653589793,3.141592653589793" 表示 '
                         '[xlo,xhi,ylo,yhi,thlo,thhi]')
    return (parts[0], parts[1]), (parts[2], parts[3]), (parts[4], parts[5])


def _parse_triplet(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError('需要 3 个数，例如: --goal "5.0,4.0,1.57"')
    return tuple(parts)

def _parse_pair(s: str):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError('需要 2 个数，例如: --x-range "-6,6"')
    return tuple(parts)

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    if args.mode in ["train", "train_and_plot"]:
        train_and_optionally_eval(args)

    # if args.mode in ["plot", "train_and_plot"]:
    #     # 示例：批量指定若干起终点
    #     starts_goals = [
    #         ((-4.0,  0.1, 0.0), ( 1.5, 0.0, 0.0)),
    #         ((-4.0, -0.1, 0.0), ( 1.5, 0.0, 0.0)),
    #         ((-4.0,  0.5, 0.0), ( 1.5, 0.0, 0.0)),
    #         ((-4.0, -0.5, 0.0), ( 1.5, 0.0, 0.0)),
    #         ((-4.0,  1.0, 0.0), ( 1.5, 0.0, 0.0)),
    #         ((-4.0, -1.0, 0.0), ( 1.5, 0.0, 0.0)),
    #     ]
    #     visualize_trajectories(
    #         log_dir=args.log_dir,
    #         alg_name=args.alg,
    #         max_steps=args.max_steps,
    #         deterministic=args.deterministic,
    #         title_suffix=args.mode,
    #         start_goals=starts_goals,
    #         episodes=len(starts_goals),
    #         # 或直接用固定起终点：
    #         # start_pose=(-4.0, -4.0, 0.0),
    #         # goal_pose=(2.0, 1.5, 0.0),
    #     )

    if args.mode in ["plot", "train_and_plot"]:
        if args.sample_mode == "specified":
            # 示例：批量指定若干起终点
            starts_goals = [
                ((-4.0,  0.1, 0.0), ( 1.5, 0.0, 0.0)),
                ((-4.0, -0.1, 0.0), ( 1.5, 0.0, 0.0)),
                ((-4.0,  0.5, 0.0), ( 1.5, 0.0, 0.0)),
                ((-4.0, -0.5, 0.0), ( 1.5, 0.0, 0.0)),
                ((-4.0,  1.0, 0.0), ( 1.5, 0.0, 0.0)),
                ((-4.0, -1.0, 0.0), ( 1.5, 0.0, 0.0)),
            ]
            visualize_trajectories(
                log_dir=args.log_dir,
                alg_name=args.alg,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
                title_suffix=args.mode,
                start_goals=starts_goals,
                episodes=len(starts_goals),
                # 或直接用固定起终点：
                # start_pose=(-4.0, -4.0, 0.0),
                # goal_pose=(2.0, 1.5, 0.0),
            )
        else:
            # 1) 解析盒子
            (xlo,xhi), (ylo,yhi), (thlo,thhi) = _parse_box6(args.box)

            # 2) 目标点：用 contour 的 --goal 或默认
            goal_pose = _parse_triplet(args.goal)  # 你现在的 --goal 仍是 "xg,yg,thg" 字符串

            # 3) 采样
            starts_goals = make_start_goals_in_box(
                x_range=(xlo,xhi),
                y_range=(ylo,yhi),
                th_range=(thlo,thhi),
                goal_pose=goal_pose,
                mode=args.sample_mode,
                nx=args.nx, ny=args.ny, nth=args.nth,
                n_samples=args.n_samples,
                seed=args.sample_seed,
            )

            # 4) 可选保存，便于和别的方法复现实验 & 共用同一批起点
            if args.save_samples:
                os.makedirs(os.path.join(args.log_dir, "eval_sets"), exist_ok=True)
                out_npz = os.path.join(args.log_dir, "eval_sets",
                                    f"starts_{args.sample_mode}_seed{args.sample_seed}.npz")
                np.savez_compressed(out_npz, starts_goals=starts_goals,
                                    x_range=(xlo,xhi), y_range=(ylo,yhi), th_range=(thlo,thhi),
                                    goal=goal_pose, mode=args.sample_mode,
                                    nx=args.nx, ny=args.ny, nth=args.nth,
                                    n_samples=args.n_samples)
                print(f"[evalset] saved to {out_npz}")

            # 5) 作图
            visualize_trajectories(
                log_dir=args.log_dir,
                alg_name=args.alg,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
                title_suffix=args.mode,
                start_goals=starts_goals,
                episodes=len(starts_goals),
            )


    if args.mode in ["contour", "train_and_plot"]:
        if args.goal is None:
            raise SystemExit('contour 模式必须提供 --goal "xg,yg,thg"')
        goal_pose = _parse_triplet(args.goal)
        x_rng = tuple(args.x_range)
        y_rng = tuple(args.y_range)

        visualize_critic_contour(
            log_dir=args.log_dir,
            alg_name=args.alg,
            goal_pose=goal_pose,
            x_range=x_rng,
            y_range=y_rng,
            grid_N=args.grid_N,
            start_theta=args.start_theta,
        )
