import os
import subprocess
import time

PROJECT_ROOT = "/root/autodl-tmp/dLink_dep"
CONDA_ENV_PATH = "/root/miniconda3/envs/torch310"

os.chdir(PROJECT_ROOT)

python_bin = os.path.join(CONDA_ENV_PATH, "bin", "python")
celery_bin = os.path.join(CONDA_ENV_PATH, "bin", "celery")

# 1. 启动 Redis（假设服务器已经装了redis-server，没装的话见下方说明）
print("启动 Redis...")
subprocess.Popen(["redis-server", "--daemonize", "yes"])
time.sleep(2)

# 2. 启动 Celery Worker
print("启动 Celery Worker...")
subprocess.Popen(
    [celery_bin, "-A", "app.core.celery", "worker", "--pool=solo", "--loglevel=info"],
    cwd=PROJECT_ROOT,
    stdout=open(os.path.join(PROJECT_ROOT, "celery.log"), "w"),
    stderr=subprocess.STDOUT,
)

# 3. 启动 FastAPI
print("启动 FastAPI 服务...")
subprocess.Popen(
    [python_bin, "CTDetectionServer.py"],
    cwd=PROJECT_ROOT,
    stdout=open(os.path.join(PROJECT_ROOT, "fastapi.log"), "w"),
    stderr=subprocess.STDOUT,
)

print("✅ 所有服务已启动！日志分别写入 celery.log 和 fastapi.log")