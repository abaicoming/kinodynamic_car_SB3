import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def wrap_angle(angle: float) -> float:
    a = (angle + math.pi) % (2 * math.pi) - math.pi
    if a == -math.pi:
        a = math.pi
    return a


class KinematicCarEnv(gym.Env):
    """
    二输入小车（u0=线速度, u1=前轮转角）：
        x_dot = u0 * cos(theta)
        y_dot = u0 * sin(theta)
        theta_dot = (u0 / L) * tan(u1)

    动作空间 (2D, 连续)：
        a = [a_v, a_delta] in [-1, 1]^2
        -> u0 (线速度), u1 (转角)

    观测 (10D，绝对坐标 + 角度cos/sin)：
        obs = [x, y, cos(theta), sin(theta), xg, yg, cos(thetag), sin(thetag), u0, u1]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        # 几何/积分
        L: float = 1.0,
        dt: float = 0.1,
        # 速度与转角限制
        v_max: float = 1.5,
        v_min: float = 0.0,            # 仅在 allow_reverse=False 时有效
        steer_max: float = 0.6,        # ≈34.4°，前轮最大转角（弧度）
        allow_reverse: bool = False,   # 是否允许倒车（u0<0）
        # 任务设置
        goal_tol: float = 0.15,
        goal_yaw_tol=0.2,  # 目标航向容差（弧度）
        max_episode_steps: int = 500,
        # 奖励权重
        w_dist: float = 1.0,
        w_yaw: float = 0.1,
        w_u_v: float = 0.0005,
        w_u_steer: float = 0.001,
        w_goal: float = 10.0,
        w_progress: float = 0.5,    # Δdist 的权重（正向奖励）
        w_time: float = 0.01,       # 每步小时间惩罚
        yaw_gate: float = 1.0,      # 到目标1m内才考核终
        # 采样范围（复位）
        xy_range: Tuple[float, float] = (-6.0, 6.0),
        theta_range: Tuple[float, float] = (-math.pi, math.pi),
        goal_xy_range: Tuple[float, float] = (-6.0, 6.0),
        goal_theta_range: Tuple[float, float] = (-math.pi, math.pi),
        min_start_goal_dist: float = 1.5,
        # 观测噪声
        obs_noise_std: float = 0.0,
        # 渲染
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # 参数
        self.L = float(L)
        self.dt = float(dt)
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.steer_max = float(steer_max)
        self.allow_reverse = bool(allow_reverse)

        self.goal_tol = float(goal_tol)
        self.goal_yaw_tol = float(goal_yaw_tol)
        self.max_episode_steps = int(max_episode_steps)

        self.w_dist = float(w_dist)
        self.w_yaw = float(w_yaw)
        self.w_u_v = float(w_u_v)
        self.w_u_steer = float(w_u_steer)
        self.w_goal = float(w_goal)

        self.w_progress   = float(w_progress)    # Δdist 的权重（正向奖励）
        self.w_time       = float(w_time)   # 每步小时间惩罚
        self.yaw_gate     = float(yaw_gate)    # 到目标1m内才考核终态朝向

        self.xy_range = xy_range
        self.theta_range = theta_range
        self.goal_xy_range = goal_xy_range
        self.goal_theta_range = goal_theta_range
        self.min_start_goal_dist = float(min_start_goal_dist)

        self.obs_noise_std = float(obs_noise_std)

        self.render_mode = render_mode
        self._renderer_initialized = False
        self._fig = None
        self._ax = None

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([+1.0, +1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 观测空间：obs = [x, y, cos(th), sin(th), xg, yg, cos(thg), sin(thg), u0, u1]
        low = np.array(
            [-np.inf, -np.inf, -1.0, -1.0, -np.inf, -np.inf, -1.0, -1.0, -self.v_max, -self.steer_max],
            dtype=np.float32,
        )
        high = np.array(
            [ np.inf,  np.inf,  1.0,  1.0,  np.inf,  np.inf,  1.0,  1.0,  self.v_max,  self.steer_max],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 状态
        self.state = None          # (x, y, theta)
        self.goal = None           # (xg, yg, thetag)
        self.u0 = 0.0              # 上一步线速度
        self.u1 = 0.0              # 上一步转角
        self._elapsed_steps = 0

        self.np_random, _ = gym.utils.seeding.np_random(None)

    # ---------- utils ----------
    def _sample_pose(self, xy_range, theta_range):
        x = self.np_random.uniform(xy_range[0], xy_range[1])
        y = self.np_random.uniform(xy_range[0], xy_range[1])
        th = self.np_random.uniform(theta_range[0], theta_range[1])
        return float(x), float(y), float(th)

    def _goal_reached(self) -> bool:
        dist, dth = self._distance_and_heading()
        return (dist < self.goal_tol) and (dth < self.goal_yaw_tol)


    def _distance_and_heading(self):
        x, y, th = self.state
        xg, yg, thg = self.goal
        dist = math.hypot(xg - x, yg - y)
        dth = abs(wrap_angle(thg - th))
        return dist, dth

    def _get_obs(self):
        x, y, th = self.state
        xg, yg, thg = self.goal
        obs = np.array(
            [x, y, math.cos(th), math.sin(th),
             xg, yg, math.cos(thg), math.sin(thg),
             self.u0, self.u1],
            dtype=np.float32,
        )
        if self.obs_noise_std > 0:
            obs += self.np_random.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)
        return obs
    
    # === 面向目标的方位角误差 ===
    def _bearing_error(self):
        x, y, th = self.state
        xg, yg, _ = self.goal
        bearing = math.atan2(yg - y, xg - x)     # 指向目标的方位角
        return wrap_angle(bearing - th)

    # ---------- Gym API ----------
    def seed(self, seed: Optional[int] = None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # 支持通过 options 指定起点/终点
        start = None
        goal = None
        if options is not None:
            start = options.get("start", None)
            goal = options.get("goal", None)

        if start is None or goal is None:
            # 随机采样起点/终点，保证最小间距
            while True:
                st = self._sample_pose(self.xy_range, self.theta_range)
                gl = self._sample_pose(self.goal_xy_range, self.goal_theta_range)
                x, y, _ = st
                xg, yg, _ = gl
                if math.hypot(xg - x, yg - y) >= self.min_start_goal_dist:
                    self.state = st
                    self.goal = gl
                    break
        else:
            sx, sy, sth = start
            gx, gy, gth = goal
            self.state = (float(sx), float(sy), float(sth))
            self.goal = (float(gx), float(gy), float(gth))

        self._elapsed_steps = 0
        self.u0 = 0.0
        self.u1 = 0.0

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        if self.render_mode == "human":
            self._render_frame()

        dist, _ = self._distance_and_heading()
        self._last_dist = dist
        
        return obs, info

    def step(self, action: np.ndarray):
        self._elapsed_steps += 1

        # ---- 动作缩放 ----
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        av = float(np.clip(action[0], -1.0, 1.0))
        adelta = float(np.clip(action[1], -1.0, 1.0))

        if self.allow_reverse:
            u0 = av * self.v_max
        else:
            # [v_min, v_max]
            u0 = 0.5 * (av + 1.0) * (self.v_max - self.v_min) + self.v_min

        u1 = adelta * self.steer_max
        # 保存以便观测/奖励
        self.u0, self.u1 = float(u0), float(u1)

        # ---- 欧拉积分 ----
        x, y, th = self.state
        x += u0 * math.cos(th) * self.dt
        y += u0 * math.sin(th) * self.dt
        th = wrap_angle(th + (u0 / self.L) * math.tan(u1) * self.dt)
        self.state = (x, y, th)

        # ---- 奖励与结束 ----
        # dist, dth = self._distance_and_heading()
        # reward = (
        #     - self.w_dist * dist
        #     - self.w_yaw * dth
        #     - self.w_u_v * (u0 ** 2)
        #     - self.w_u_steer * (u1 ** 2)
        # )

        # terminated = False
        # if self._goal_reached():
        #     reward += self.w_goal
        #     terminated = True

        dist, dth = self._distance_and_heading()
        progress = self._last_dist - dist            # >0 表示靠近
        bearing_err = abs(self._bearing_error())     # 面向目标的误差（[0,pi]）

        # 基本奖励：靠近就加分，远离就小扣分（clip避免异常值）
        r_progress = self.w_progress * progress
        r_progress = float(np.clip(r_progress, -1.0, 1.0))   # 可按需调/移除clip

        # 控制代价
        r_ctrl = - self.w_u_v * (u0 ** 2) - self.w_u_steer * (u1 ** 2)

        # 航向代价：用面向目标的误差；目标姿态只在近处才扣
        r_yaw_track = - self.w_yaw * bearing_err
        r_yaw_goal  = 0.0
        if dist < self.yaw_gate:
            dth_goal = abs(wrap_angle(self.goal[2] - self.state[2]))
            r_yaw_goal = - self.w_yaw * 0.5 * dth_goal   # 只在近处半权重考核终态朝向

        # 时间代价
        r_time = - self.w_time

        reward = r_progress + r_yaw_track + r_yaw_goal + r_ctrl + r_time

        terminated = False
        if self._goal_reached():    # 你的 _goal_reached 里已经加了姿态门限
            reward += self.w_goal   # 建议把 w_goal 调到 50~100 看看
            terminated = True

        self._last_dist = dist      # 记住用于下一步 Δdist

        truncated = self._elapsed_steps >= self.max_episode_steps

        obs = self._get_obs()

        state_tuple = (float(self.state[0]), float(self.state[1]), float(self.state[2]))
        
        info = {
            "dist": dist,
            "dtheta": dth,
            "u0": u0,
            "u1": u1,
            "is_success": terminated,  # 便于外部统计“到达率”
            "state": state_tuple,  # 每一步的状态
        }

        if terminated:
            info["terminal_state"] = state_tuple  # 仅在终止步提供
            
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    # ---------- Rendering ----------
    def _render_frame(self):
        import matplotlib.pyplot as plt

        if not self._renderer_initialized:
            self._fig, self._ax = plt.subplots()
            self._renderer_initialized = True

        ax = self._ax
        ax.clear()

        x, y, th = self.state
        xg, yg, thg = self.goal

        # 当前车与目标
        ax.plot([x], [y], marker="o")
        ax.plot([xg], [yg], marker="*", markersize=12)

        # 车朝向
        Lvis = 0.6
        ax.arrow(x, y, Lvis * math.cos(th), Lvis * math.sin(th),
                 head_width=0.2, length_includes_head=True)
        # 目标朝向
        ax.arrow(xg, yg, Lvis * math.cos(thg), Lvis * math.sin(thg),
                 head_width=0.2, length_includes_head=True, alpha=0.5)

        xr = self.xy_range
        ax.set_xlim(xr[0] - 1, xr[1] + 1)
        ax.set_ylim(xr[0] - 1, xr[1] + 1)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_title(f"KinematicCar | step={self._elapsed_steps}")

        self._fig.canvas.draw()
        if self.render_mode == "human":
            self._fig.canvas.flush_events()
            import time
            time.sleep(1.0 / self.metadata["render_fps"])

    def render(self):
        if self.render_mode == "rgb_array":
            self._render_frame()
            self._fig.canvas.draw()
            w, h = self._fig.canvas.get_width_height()
            img = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return img.reshape((h, w, 3))
        elif self.render_mode == "human":
            return None
        else:
            return None

    def close(self):
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self._fig)
            except Exception:
                pass
        self._fig = None
        self._ax = None
        self._renderer_initialized = False
