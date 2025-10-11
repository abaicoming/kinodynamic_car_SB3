1. 启动container
sudo docker run -it \
  --gpus all \
  --network host --ipc=host \
  --env="DISPLAY=:1" \
  --env="QT_X11_NO_MITSHM=1" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/ubuntu/xlh/Kinodynamic:/workspace \
  -v /usr/lib/x86_64-linux-gnu/:/glu \
  -v /home/n/.local:/.local \
  --name sb3_test0 \
  -w /workspace \
  --user $(id -u):$(id -g) \
  stablebaselines/stable-baselines3:latest

2. 容器内安装sb3
python -m pip install -U pip setuptools wheel packaging
python -m pip install -U stable-baselines3
python -m pip install -U opencv-python-headless

3. 快速验证
python - <<'PY'
import sys, torch, stable_baselines3 as sb3
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available())
print("SB3:", sb3.__version__)
PY

4. 跑一个小例子（2 秒就过）
python - <<'PY'
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("CartPole-v1", n_envs=1)
model = PPO("MlpPolicy", env, n_steps=128, batch_size=128, n_epochs=1, seed=0, verbose=1)
model.learn(2000)  # 小步数验证安装
print("训练完成！")
PY


5. train
export PYTHONPATH=/workspace:$PYTHONPATH
python /workspace/kinodynamic_car_SB3/train_ddpg.py



