import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
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

    动作空间 (2D, 连续)：a=[a_v, a_delta]∈[-1,1]^2
    观测（13D，本实现用“车体中心 + 朝向前点”而非cos/sin，数值平滑些）：
        [x, y, x+0.5Lcosθ, y+0.5Lsinθ, xg, yg, xg+0.5Lcosθg, yg+0.5Lsinθg, u0, u1, x-ox, y-oy, ‖p-o‖]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        L: float = 1.0,
        dt: float = 0.1,
        v_max: float = 1.5,
        v_min: float = 0.0,
        steer_max: float = 0.6,
        allow_reverse: bool = False,
        goal_tol: float = 0.15,
        yaw_tol: float = 0.25,
        max_episode_steps: int = 500,
        w_dist: float = 1.0,
        w_yaw: float = 0.1,
        w_u_v: float = 5e-4,
        w_u_steer: float = 1e-3,
        w_goal: float = 10.0,
        xy_range: Tuple[float, float] = (-6.0, 6.0),
        theta_range: Tuple[float, float] = (-math.pi, math.pi),
        goal_xy_range: Tuple[float, float] = (-6.0, 6.0),
        goal_theta_range: Tuple[float, float] = (-math.pi, math.pi),
        min_start_goal_dist: float = 1.5,
        obs_noise_std: float = 0.0,
        render_mode: Optional[str] = None,
        obstacle_center: Tuple[float, float] = (0.0, 0.0),
        obstacle_radius: float = 1.0,
        safe_margin: float = 0.5,
        w_obs_shaping: float = 2.0,
        w_collision: float = 50.0,
        # shaping 选择
        obs_shaping_type: str = "exp",  # ["exp","quad","barrier","reciprocal","none"]
        obs_sigma: float = 0.8,
        obs_margin: float = 0.5,
        obs_power: float = 2.0,

        input_mode:str = "old_input", # ["old_input", "new_input"]
        fixed_goal_flag: bool = False,
        fixed_goal:Tuple[float,float,float] = (1.5,0,0)
    ):
        super().__init__()

        self.L = float(L); self.dt = float(dt)
        self.v_max = float(v_max); self.v_min = float(v_min)
        self.steer_max = float(steer_max); self.allow_reverse = bool(allow_reverse)
        self.goal_tol = float(goal_tol); self.yaw_tol = float(yaw_tol)
        self.max_episode_steps = int(max_episode_steps)

        self.w_dist = float(w_dist); self.w_yaw = float(w_yaw)
        self.w_u_v = float(w_u_v);   self.w_u_steer = float(w_u_steer)
        self.w_goal = float(w_goal)

        self.xy_range = xy_range; self.theta_range = theta_range
        self.goal_xy_range = goal_xy_range; self.goal_theta_range = goal_theta_range
        self.min_start_goal_dist = float(min_start_goal_dist)

        self.obs_noise_std = float(obs_noise_std)
        self.render_mode = render_mode
        self._renderer_initialized = False
        self._fig = None; self._ax = None

        self.obstacle_center = tuple(obstacle_center)
        self.obstacle_radius = float(obstacle_radius)
        self.safe_margin = float(safe_margin)
        self.w_obs_shaping = float(w_obs_shaping)
        self.w_collision = float(w_collision)

        self.obs_shaping_type = str(obs_shaping_type)
        self.obs_sigma = float(obs_sigma)
        self.obs_margin = float(obs_margin)
        self.obs_power = float(obs_power)

        self.input_mode = str(input_mode)
        self.fixed_goal_flag = fixed_goal_flag  # 控制目标是否固定
        self.fixed_goal = fixed_goal

        # 动作/观测空间
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], np.float32),
                                       high=np.array([+1.0, +1.0], np.float32),
                                       dtype=np.float32)
        if self.input_mode == "old_input":
            # old
            low = np.array(
                [-np.inf, -np.inf, -1.0, -1.0, -np.inf, -np.inf, -1.0, -1.0, -self.v_max, -self.steer_max, -np.inf, -np.inf, -np.inf],
                dtype=np.float32,
            )
            high = np.array(
                [ np.inf,  np.inf,  1.0,  1.0,  np.inf,  np.inf,  1.0,  1.0,  self.v_max,  self.steer_max, np.inf, np.inf, np.inf],
                dtype=np.float32,
            )
        elif self.input_mode == "new_input":
            #new
            low = np.array(
                [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -self.v_max, -self.steer_max, -np.inf, -np.inf, -np.inf],
                dtype=np.float32,
            )
            high = np.array(
                [ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.v_max,  self.steer_max, np.inf, np.inf, np.inf],
                dtype=np.float32,
            )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 状态
        self.state = None      # (x,y,theta)
        self.goal = None       # (xg,yg,thetag)
        self.u0 = 0.0; self.u1 = 0.0
        self._elapsed_steps = 0

        self._forced_start = None
        self._forced_goal = None

        self.np_random, _ = gym.utils.seeding.np_random(None)

    # 外部在“下一次 reset”设置起终点
    def set_start_goal_for_next_reset(self, start=None, goal=None):
        # print("i am in")
        # sys.exists()
        self._forced_start = start
        self._forced_goal = goal

    # 工具
    def _dist_to_obstacle(self) -> float:
        ox, oy = self.obstacle_center
        x, y, _ = self.state
        return math.hypot(x - ox, y - oy)

    def _collided(self) -> bool:
        return self._dist_to_obstacle() <= (self.obstacle_radius)

    def _sample_pose(self, xy_range, theta_range):
        x = self.np_random.uniform(xy_range[0], xy_range[1])
        y = self.np_random.uniform(xy_range[0], xy_range[1])
        th = self.np_random.uniform(theta_range[0], theta_range[1])
        return float(x), float(y), float(th)

    def _goal_reached(self) -> bool:
        x, y, th = self.state
        xg, yg, thg = self.goal
        pos_ok = math.hypot(xg - x, yg - y) < self.goal_tol
        yaw_ok = abs(wrap_angle(thg - th)) < self.yaw_tol
        return bool(pos_ok and yaw_ok)

    def _goal_pos_ok(self) -> bool:
        x, y, _ = self.state
        xg, yg, _ = self.goal
        return math.hypot(xg - x, yg - y) < self.goal_tol

    def _goal_yaw_ok(self) -> bool:
        _, _, th = self.state
        _, _, thg = self.goal
        return abs(wrap_angle(thg - th)) < self.yaw_tol

    def _distance_and_heading(self):
        x, y, th = self.state
        xg, yg, thg = self.goal
        dist = math.hypot(xg - x, yg - y)
        dth = abs(wrap_angle(thg - th))
        return dist, dth

    def _get_obs(self):
        x, y, th = self.state
        xg, yg, thg = self.goal
        ox, oy = self.obstacle_center
        dobs = math.hypot(x - ox, y - oy)
        if self.input_mode == "old_input":
            # old
            obs = np.array(
                [x, y, math.cos(th), math.sin(th),
                xg, yg, math.cos(thg), math.sin(thg),
                self.u0, self.u1,
                x - ox, y - oy, dobs
                ],
                dtype=np.float32,
            )
        elif self.input_mode == "new_input":
            # new
            obs = np.array(
                [x, y,
                x + 0.5*self.L*math.cos(th), y + 0.5*self.L*math.sin(th),
                xg, yg,
                xg + 0.5*self.L*math.cos(thg), yg + 0.5*self.L*math.sin(thg),
                self.u0, self.u1,
                x - ox, y - oy, dobs], dtype=np.float32,
            )

        if self.obs_noise_std > 0:
            obs += self.np_random.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)
        return obs

    # Gym API
    def seed(self, seed: Optional[int] = None):
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, *, seed: Optional[int]=None, options: Optional[Dict[str, Any]]=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        start = options.get("start") if options else None
        goal  = options.get("goal") if options else None

        if self._forced_start is not None or self._forced_goal is not None:
            start = self._forced_start if self._forced_start is not None else start
            goal  = self._forced_goal  if self._forced_goal  is not None else goal
            self._forced_start = None
            self._forced_goal = None

        if start is None or goal is None:
            if self.fixed_goal_flag:
                while True:
                    st = self._sample_pose(self.xy_range, self.theta_range)
                    gl = self.fixed_goal
                    if math.hypot(gl[0]-st[0], gl[1]-st[1]) >= self.min_start_goal_dist:
                        self.state, self.goal = st, gl
                        break
            else:
                while True:
                    st = self._sample_pose(self.xy_range, self.theta_range)
                    gl = self._sample_pose(self.goal_xy_range, self.goal_theta_range)
                    if math.hypot(gl[0]-st[0], gl[1]-st[1]) >= self.min_start_goal_dist:
                        self.state, self.goal = st, gl
                        break
        else:
            sx, sy, sth = start
            gx, gy, gth = goal
            self.state = (float(sx), float(sy), float(sth))
            self.goal  = (float(gx), float(gy), float(gth))

        # 确保起终点不在障碍物内（不强制 safe_margin）
        ox, oy = self.obstacle_center
        def ok(p): return math.hypot(p[0]-ox, p[1]-oy) > (self.obstacle_radius)
        if not ok(self.state) or not ok(self.goal):
            while True:
                st = self._sample_pose(self.xy_range, self.theta_range)
                gl = self._sample_pose(self.goal_xy_range, self.goal_theta_range)
                if ok(st) and ok(gl) and math.hypot(gl[0]-st[0], gl[1]-st[1]) >= self.min_start_goal_dist:
                    self.state, self.goal = st, gl
                    break

        self._elapsed_steps = 0
        self.u0 = 0.0; self.u1 = 0.0
        # print(self.state)
        # print(self.goal)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        self._elapsed_steps += 1

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        av = float(np.clip(action[0], -1.0, 1.0))
        adelta = float(np.clip(action[1], -1.0, 1.0))

        if self.allow_reverse:
            u0 = av * self.v_max
        else:
            u0 = 0.5*(av+1.0)*(self.v_max - self.v_min) + self.v_min
        u1 = adelta * self.steer_max
        self.u0, self.u1 = float(u0), float(u1)

        x, y, th = self.state
        x += u0*math.cos(th)*self.dt
        y += u0*math.sin(th)*self.dt
        th = wrap_angle(th + (u0/self.L)*math.tan(u1)*self.dt)
        self.state = (x, y, th)

        dist, dth = self._distance_and_heading()
        reward = (
            - self.w_dist * dist
            - self.w_yaw * dth
            - self.w_u_v * (u0**2)
            - self.w_u_steer * (u1**2)
        )

        # 避障 shaping
        d_obs = self._dist_to_obstacle()
        clearance = d_obs - self.obstacle_radius
        m = self.obs_margin; lam = self.w_obs_shaping; sig = self.obs_sigma

        if self.obs_shaping_type == "exp":
            shaping = - lam * math.exp(- max(clearance, 0.0) / max(sig, 1e-6)) if clearance <= m else 0.0
        elif self.obs_shaping_type == "quad":
            pen = max(0.0, m - clearance)
            shaping = - lam * (pen ** 2)
        elif self.obs_shaping_type == "barrier":
            if clearance > 0.0:
                shaping = lam * math.log(clearance / max(m, 1e-6) + 1e-6)
            else:
                shaping = - lam * 10.0
        elif self.obs_shaping_type == "reciprocal":
            shaping = - lam / ((max(clearance, 1e-3)) ** self.obs_power)
        else:
            shaping = 0.0
        reward += shaping

        pos_ok = self._goal_pos_ok()
        yaw_ok = self._goal_yaw_ok()
        is_goal = self._goal_reached()
        is_collision = self._collided()

        if is_goal:      reward += self.w_goal
        if is_collision: reward -= self.w_collision

        terminated = bool(is_goal or is_collision)
        truncated  = self._elapsed_steps >= self.max_episode_steps

        obs = self._get_obs()
        state_tuple = (float(self.state[0]), float(self.state[1]), float(self.state[2]))
        info = {
            "dist": dist, "dtheta": dth,
            "u0": u0, "u1": u1,
            "is_success": bool(is_goal), "success": bool(is_goal),
            "pos_ok": pos_ok, "yaw_ok": yaw_ok,
            "is_collision": is_collision,
            "state": state_tuple,
        }
        if terminated or truncated:
            info["terminal_state"] = state_tuple

        if self.render_mode == "human":
            self._render_frame()

        # # 打印 info（用于调试）
        # print(info)  # 这行可以帮助您在终端查看每一步的info
        return obs, reward, terminated, truncated, info

    # 渲染
    def _render_frame(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        if not self._renderer_initialized:
            self._fig, self._ax = plt.subplots()
            self._renderer_initialized = True
        ax = self._ax; ax.clear()
        x, y, th = self.state; xg, yg, thg = self.goal

        circ = Circle(self.obstacle_center, radius=self.obstacle_radius,
                      facecolor="k", edgecolor="k", alpha=0.2, lw=1.0)
        ax.add_patch(circ)

        ax.plot([x], [y], marker="o")
        ax.plot([xg], [yg], marker="*", markersize=12)

        Lvis = 0.6
        ax.arrow(x, y, Lvis*math.cos(th),  Lvis*math.sin(th),  head_width=0.2, length_includes_head=True)
        ax.arrow(xg, yg, Lvis*math.cos(thg), Lvis*math.sin(thg), head_width=0.2, length_includes_head=True, alpha=0.5)

        xr = self.xy_range
        ax.set_xlim(xr[0]-1, xr[1]+1)
        ax.set_ylim(xr[0]-1, xr[1]+1)
        ax.set_aspect("equal"); ax.grid(True)
        ax.set_title(f"KinematicCar | step={self._elapsed_steps}")

        self._fig.canvas.draw()
        if self.render_mode == "human":
            self._fig.canvas.flush_events()
            import time; time.sleep(1.0 / self.metadata["render_fps"])

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
        self._fig = None; self._ax = None; self._renderer_initialized = False
