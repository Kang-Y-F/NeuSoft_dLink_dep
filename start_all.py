import os
import subprocess
import time

PROJECT_ROOT = r"D:\NeuSoft\dLink_dep"
CONDA_ENV_PATH = r"D:\Program Files\anaconda3\envs\torch310_new"

os.chdir(PROJECT_ROOT)

# 1. 启动 Redis
print("启动 Redis...")
subprocess.Popen(
    ["redis-server"],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)
time.sleep(2)

# 2. 启动 Celery Worker（直接用环境里的 celery.exe，不用 conda run）
print("启动 Celery Worker...")
subprocess.Popen(
    [os.path.join(CONDA_ENV_PATH, "Scripts", "celery.exe"),
     "-A", "app.core.celery", "worker", "--pool=solo", "--loglevel=info"],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)
time.sleep(3)

# 3. 启动 FastAPI（直接用环境里的 python.exe）
print("启动 FastAPI 服务...")
subprocess.Popen(
    [os.path.join(CONDA_ENV_PATH, "python.exe"), "CTDetectionServer.py"],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)

print("✅ 所有服务已启动！")
print("Redis、Celery、FastAPI 三个窗口已打开，不要关闭它们。")
input("按 Enter 键退出主脚本（不会影响其他服务）...")