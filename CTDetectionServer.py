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

        # 2. 保存上传文件（唯一名称）
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

        # 6. 构造返回信息
        return {
            "status": "success",
            "message": "CT金属伪影检测完成",
            "original_file": original_filename,
            "mask_file": mask_filename,
            "mask_save_path": mask_save_path,
            "shape": mask_sitk.GetSize(),
            "spacing": list(sitk_ct.GetSpacing()),
            "origin": list(sitk_ct.GetOrigin()),
            "direction": list(sitk_ct.GetDirection()),
            "download_url": f"/results/{mask_filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务处理失败: {str(e)}")

# --------------------------
# 下载掩码文件接口
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