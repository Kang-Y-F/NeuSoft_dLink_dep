@echo off
:: 1. 启动 Redis
start cmd /k "redis-server"
timeout /t 2 /nobreak >nul

:: 2. 启动 Celery Worker（路径加了双引号）
start cmd /k "cd /d D:\NeuSoft\dLink_dep && ""D:\Program Files\anaconda3\Scripts\conda.exe"" activate torch310_new && celery -A app.core.celery worker --pool=solo --loglevel=info"
timeout /t 3 /nobreak >nul

:: 3. 启动 FastAPI 服务（路径加了双引号）
start cmd /k "cd /d D:\NeuSoft\dLink_dep && ""D:\Program Files\anaconda3\Scripts\conda.exe"" activate torch310_new && python CTDetectionServer.py"