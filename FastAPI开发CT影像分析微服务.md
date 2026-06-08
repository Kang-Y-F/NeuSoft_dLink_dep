# 使用FastAPI开发CT影像分析微服务

---

# 一、项目说明

代码是 **CT 金属伪影检测 AI 服务**，包含：

- FastAPI 主服务

- 一次性加载 AI 模型（全局只加载一次）

- NIfTI（\.nii/\.nii\.gz）文件上传

- AI 推理 \+ 掩码保存

- 结果文件下载接口

- 跨域支持（对接 Vue 前端）

---

# 二、PyCharm 环境搭建

## 1\. 新建项目

1. 打开 PyCharm → **New Project**

2. 选择：

    - **Location**：你的项目文件夹

    - **New environment using**：`VENV`

    - **Python version**：3\.8 \~ 3\.11

3. 点击 **Create**

## 2\. 项目结构（必须和你代码一致）

```Plain Text
你的项目根目录/
├── Detection/             # 你的模型推理目录
│   └── CTArtifactInfer.py # 推理类
├── model/              # 模型权重目录
│   └── UNet2D.py
│   └── AttentionUNet2D.py
│   └── weights/
│       └── best.pth       # 模型文件
├── uploads/               # 自动创建
├── results/               # 自动创建
├── CTDetectionServer.py                # 你的后端代码
└── requirements.txt       # 依赖文件
```

---

# 三、安装依赖（必须安装）

## 1\. 创建 requirements\.txt

```txt
altgraph==0.17.5
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
click==8.4.1
colorama==0.4.6
fastapi==0.136.3
filelock==3.29.0
fsspec==2026.4.0
h11==0.16.0
idna==3.18
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.4
packaging==26.2
pefile==2024.8.26
pillow==12.2.0
pydantic==2.13.4
pydantic_core==2.46.4
pyinstaller==6.20.0
pyinstaller-hooks-contrib==2026.5
python-multipart==0.0.32
pywin32-ctypes==0.2.3
setuptools==70.2.0
simpleitk==2.5.5
starlette==1.2.1
sympy==1.14.0
torch==2.11.0+cu128
torchaudio==2.11.0+cu128
torchvision==0.26.0+cu128
tqdm==4.68.1
typing-inspection==0.4.2
typing_extensions==4.15.0
uvicorn==0.49.0
```

## 2\. PyCharm 安装

- 打开 `requirements.txt`

- 点击顶部 **Install requirements**

- 等待完成

---

# 四、完整可运行后端代码（main\.py）

文件名：`CTDetectionServer.py`

```python
import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import SimpleITK as sitk

# 导入推理类
from Detection.CTArtifactInfer import CTArtifactInfer

app = FastAPI(title="CT金属伪影检测AI服务", version="2.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局加载模型（只加载一次，性能最优）
try:
    infer = CTArtifactInfer(model_weight_path="./Model/weights/nor_best.pth")
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    raise

# 目录创建
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------
# 工具函数：校验是否为NIfTI文件
# --------------------------
def is_nifti_file(filename: str):
    return filename.lower().endswith((".nii", ".nii.gz"))

# --------------------------
# 核心接口：上传CT + 伪影检测
# --------------------------
@app.post("/predict-ct-artifact")
async def predict_ct(file: UploadFile = File(...)):
    try:
        # 1. 校验文件类型
        if not is_nifti_file(file.filename):
            raise HTTPException(status_code=400, detail="只支持 .nii 或 .nii.gz 格式的CT文件")

        original_filename = file.filename
        unique_id = str(uuid.uuid4())  # 唯一ID，防止文件覆盖
        name_without_ext = original_filename.replace(".nii.gz", "").replace(".nii", "")

        # 2. 保存上传文件
        upload_path = os.path.join(UPLOAD_DIR, f"{unique_id}_{original_filename}")
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 3. 读取CT
        sitk_ct = sitk.ReadImage(upload_path)

        # 4. 掩码保存路径
        mask_filename = f"{unique_id}_{name_without_ext}_mask.nii.gz"
        mask_save_path = os.path.join(RESULT_DIR, mask_filename)

        # 5. AI推理
        mask_sitk = infer.predict_from_sitk(sitk_ct, save_mask_path=mask_save_path)

        # 6. 返回结果
        return {
            "status": "success",
            "message": "CT金属伪影检测完成",
            "original_file": original_filename,
            "mask_file": mask_filename,
            "shape": mask_sitk.GetSize(),
            "spacing": list(sitk_ct.GetSpacing()),
            "origin": list(sitk_ct.GetOrigin()),
            "direction": list(sitk_ct.GetDirection()),
            "download_url": f"/results/{mask_filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务处理失败: {str(e)}")

# --------------------------
# 下载掩码文件
# --------------------------
@app.get("/results/{mask_filename}")
async def download_mask(mask_filename: str):
    mask_path = os.path.join(RESULT_DIR, mask_filename)
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="掩码文件不存在")
    return FileResponse(
        path=mask_path,
        media_type="application/octet-stream",
        filename=mask_filename
    )

# --------------------------
# 健康检查
# --------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "CT金属伪影检测服务运行中"}

# --------------------------
# 启动服务
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

```

---

# 五、在 PyCharm 中运行项目

## 方法：直接运行 main\.py

当前代码已内置服务启动入口，无需额外配置启动脚本，操作简单高效：

右键项目内的 `CTDetectionServer.py` → 选择 **Run 'main'**

服务启动成功后，控制台会输出如下内容：

```Plain Text
INFO:     Started server process [xxx]
INFO:     Waiting for application startup.
✅ 模型加载成功
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

# 六、服务接口地址（前端对接专用）

- 服务根地址：`http://localhost:8000`

- CT伪影检测接口：`http://localhost:8000/predict-ct-artifact`

- 掩码文件下载接口：`http://localhost:8000/results/xxx_mask.nii.gz`

- 在线API调试文档：`http://localhost:8000/docs`

---

# 七、PyCharm 调试使用技巧

1. 在代码行号左侧点击，添加**断点**，用于精准调试代码逻辑

2. 右键 `CTDetectionServer.py` → 选择 **Debug 'main'** 启动调试模式

3. 前端发送接口请求后，程序会自动在断点处暂停

4. 可实时查看文件参数、影像信息、模型推理结果等变量内容，快速排查问题

---

# 八、Vue3 前端对接方案

前端通过 axios 发送表单请求，对接后端检测接口，代码示例如下：

```js
axios.post("http://localhost:8000/predict-ct-artifact", formData)
```

接口成功返回的 `download_url` 字段，可直接拼接域名用于文件下载：

```Plain Text
/results/xxxx_mask.nii.gz
```

---

# 九、常见报错与完整解决方案

## 1\. ModuleNotFoundError: No module named 'Detection'

**问题原因**：PyCharm 未识别项目源码目录，无法导入本地模块

**解决方法**：

- 右键项目根目录 → **Mark Directory as** → **Sources Root**

- 重启服务即可正常识别本地推理模块

## 2\. 模型权重文件不存在报错

**报错提示**：No such file or directory: '\./run\_norm/weights/best\.pth'

**解决方法**：严格匹配项目目录结构，确保权重文件路径无误：

```Plain Text
项目/
└── model/
    └── weights/
        └── best.pth
```

## 3\. 文件上传接口报错

**解决方法**：安装文件上传依赖库

```bash
pip install python-multipart
```

## 4\. SimpleITK 影像处理报错

**解决方法**：重装影像处理依赖

```bash
pip install SimpleITK
```

## 5\. 服务地址网页解析失败（http://0\.0\.0\.0:8000）

**问题原因**：`0.0.0.0` 为服务监听地址，仅用于后台监听所有网卡，**无法直接在浏览器打开访问**，属于正常现象，并非服务故障。

**正确访问方式**：前端、浏览器调试、接口测试均统一使用：`http://localhost:8000`

## 6\. 掩码下载URL报错、提示地址错误

**问题原因**：直接使用后端返回的相对路径拼接错误、或文件未生成成功、URL书写不完整。

**标准正确下载地址格式**：

```Plain Text
http://localhost:8000/results/xxx_mask.nii.gz
```

**排查步骤**：

- 先确认后端 `results` 目录下已生成对应掩码文件

- 完整拼接域名\+接口路径，不可仅使用相对路径

- 检查文件名是否完全匹配，区分大小写

---

# 十、项目打包与生产部署（正式上线方案）

本章提供**Windows通用打包部署方案**，可将 FastAPI 后端打包为可执行文件，脱离PyCharm、脱离Python环境直接运行，适配本地部署、服务器上线等生产场景，全程无需修改原有业务代码。

## 1\. 部署前置准备

打包前必须确保本地开发环境可正常运行，满足以下条件：

- PyCharm 中运行 main\.py，服务正常启动，无模型加载报错

- 上传、AI推理、文件下载接口均可正常调用

- 项目目录结构完整，`modle/weights/best.pth` 权重文件存在

- 所有依赖库已完整安装，无缺失报错

## 2\. 安装打包工具 PyInstaller

PyInstaller 为Python通用打包工具，可将项目打包为独立exe可执行文件，打开终端执行安装命令：

```bash
pip install pyinstaller
```

## 3\. 核心打包命令（适配本项目）

在项目**根目录**打开终端，执行专属打包命令，该命令会自动打包代码、静态资源、模型配置，规避路径报错：

```bash
pyinstaller -F -w CTDetectionServer.py"
pyinstaller -F CTDetectionServer.py"
```

参数说明：

- **\-F**：打包为单个独立exe文件，部署更便捷

- **\-w**：关闭运行黑窗口，后台静默运行（如需查看日志可删除该参数）

- **\-\-add\-data**：拷贝本地推理模块、模型权重目录，解决打包后资源缺失、路径报错问题

## 4\. 打包成果与目录说明

命令执行完成且无报错后，项目根目录会生成三个文件夹：`build`、`dist`、`__pycache__`，其中核心部署文件位置：

**可执行文件路径：项目根目录/dist/main\.exe**

将 `dist` 文件夹单独复制到服务器或目标部署设备即可，其余缓存文件夹可直接删除，不影响运行。

## 5\. 生产环境启动服务

### 5\.1 常规启动（临时运行，关闭窗口即停止）

进入 dist 目录，双击运行 `main.exe`，服务自动启动，默认监听 `0.0.0.0:8000`，启动成功后可通过 `http://localhost:8000` 访问服务。

### 5\.2 后台常驻启动（生产推荐）

Windows/Linux 服务器部署，推荐使用后台常驻方式，关闭终端窗口服务也不会停止：

Windows 终端后台启动命令：

```bash
start /b CTDetectionServer.exe
```

## 6\. 部署后核心注意事项

- 部署后的服务，**禁止使用 http://0\.0\.0\.0:8000** 访问，统一使用 `http://localhost:8000` 或服务器IP\+端口访问

- 文件下载接口必须完整拼接路径：`http://localhost:8000/results/xxx_mask.nii.gz`，仅使用后端返回的相对路径会触发URL报错

- 首次运行exe，会自动生成 `uploads`、`results` 目录，无需手动创建

- 部署设备需保证端口8000未被占用，若端口冲突，可修改main\.py中port参数后重新打包

## 7\. 打包部署常见报错解决方案

### 7\.1 打包后模型加载失败

**问题原因**：未拷贝权重目录或路径配置错误

**解决方案**：严格使用本文档专属打包命令，确保 `Detection`、`run_norm` 目录成功打包，打包后检查dist目录下是否存在对应文件夹及权重文件

### 7\.2 运行exe提示端口被占用

**解决方案**：关闭占用8000端口的程序，或修改启动端口：修改main\.py中 `port=8000` 为其他端口（如8080），重新打包部署

### 7\.3 部署后文件下载URL报错

**解决方案**：确认服务正常启动、推理完成后results目录存在对应文件，前端完整拼接域名\+下载路径，杜绝单独使用相对路径

## 8\. 服务关闭与重启方式

- 临时运行服务：直接关闭运行窗口即可停止服务

- 后台常驻服务：打开任务管理器，结束 `CTDetectionServer.exe` 进程即可关闭服务

- 重启服务：关闭进程后，重新双击运行exe或执行后台启动命令

---

# 十一、文档总结

本手册**完全基于个人真实后端代码定制**，所有配置、代码、运行方式、接口路径均一一对应，适配 CT金属伪影检测AI服务项目：

- 适配 PyCharm 专属开发、运行、调试环境

- 内置全局模型单次加载，保障服务运行性能

- 完整实现NIfTI文件上传、AI推理、掩码生成、文件下载全流程

- 完美兼容Vue3前端跨域请求，可直接对接项目前端

- 覆盖开发、调试、对接、报错排查、打包部署全场景问题
