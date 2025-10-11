## Debug

1. plot中并未画出障碍物

2. 碰撞也会触发终止，但是碰撞不能算是成功




## 先跑一个最小实验的建议顺序

1. 只加硬碰撞惩罚 + 终止（不加 shaping）。看看 success_rate 是否明显下降、是否能学会绕开（通常会很难）。

2. 再加上shaping（上面的 1/clearance 或 exp 版本），观察 rollout/ep_rew_mean 是否更顺滑上升。

3. 在可视化图上确认：轨迹是否明显“贴边绕圈”而不是直接穿过原点。

4. 逐步调：w_obs_shaping（近距离惩罚强度）、w_collision（一次碰撞扣多少）、obstacle_radius/safe_margin（几何难度）。