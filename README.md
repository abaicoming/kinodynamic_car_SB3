# how to get a container
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

# ps: stable-baselines3-ready:latest has done the above things, just(example for a100):
sudo docker run -it \
  --gpus all \
  --network host --ipc=host \
  --env="DISPLAY=:1" \
  --env="QT_X11_NO_MITSHM=1" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/ac/data/xlh/KINODYNAMIC:/workspace \
  -v /usr/lib/x86_64-linux-gnu/:/glu \
  -v /home/n/.local:/.local \
  --name sb3_test0 \
  -w /workspace \
  --user $(id -u):$(id -g) \
  stable-baselines3-ready:latest


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
` export PYTHONPATH=/workspace:$PYTHONPATH `
` python /workspace/kinodynamic_car_SB3/train_ddpg.py `

6. open tensorboard to check results
`tensorboard --logdir /workspace/kinodynamic_car_SB3/logs`


## how to transfer files between two ubuntus
1. files
` scp -r kinodynamic_car_SB3 ac@10.86.96.6:/home/ac/data/xlh/KINODYNAMIC ` 

2. docker
  a. save docker image as '.tar' file
    `docker save -o sb3.tar stablebaselines/stable-baselines3:latest`
  b. transfer '.tar' file
    `scp sb3.tar ac@10.86.96.6:/home/ac/data/` or `rsync -avzP sb3.tar ac@10.86.96.6:/home/ac/data/`
  c. load docker image
    `docker load -i /home/ac/data/sb3.tar`


# pull
0. export https_proxy=http://127.0.0.1:7890;export http_proxy=http://127.0.0.1:7890;export all_proxy=socks5://127.0.0.1:7890
1. git add __pycache__ dubins_env __init__.py README.md train_ddpg.py
2. git commit -m "what happend"
3. git push -u origin main
4. 