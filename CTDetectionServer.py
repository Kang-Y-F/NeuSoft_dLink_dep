import os
import uuid
import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# 导入自定义模块
from app.core.celery import celery
from app.api.endpoints import router as api_router
from Detection.CTArtifactInfer import ModelType
from Detection.CTArtifactInfer import CTArtifactInfer

# ===================== 全局配置 =====================
# 临时文件目录
TEMP_DIR = "./temp"
# 患者特征存储目录
PATIENT_FEATURE_DIR = "./patient_features"
# 模型权重路径
WEIGHT_PATH = "./Model/weights/nor_best.pth"

# 创建目录（不存在则自动新建）
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(PATIENT_FEATURE_DIR, exist_ok=True)

# ===================== FastAPI 实例初始化 =====================
app = FastAPI(
    title="CT伪影分割服务",
    description="前端网页配套分割、特征提取、聚类分析接口",
    version="1.0.0"
)

# 跨域中间件（浏览器前端必需）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载 Celery 异步任务路由
app.include_router(api_router, prefix="/api", tags=["异步任务接口"])

# ===================== 基础接口 =====================
@app.get("/health", summary="服务健康检测（前端状态灯）")
async def health_check():
    """前端页面轮询检测服务是否在线"""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "success", "device": device}

# ===================== 核心推理接口（对接前端 /infer/nii） =====================
@app.post("/infer/nii", summary="NII 文件推理主接口")
async def infer_nii(
    file: UploadFile = File(...),
    save_mask: bool = Query(True, description="是否返回掩码文件"),
    mask_filename: str = Query("artifact_mask.nii.gz", description="掩码文件名"),
    model_type: str = Query(ModelType.UNET2D, description="模型类型"),
    save_feature: bool = Query(True, description="是否保存特征"),
    feature_filename: str = Query("ct_feature.npy", description="特征文件名")
):
    try:
        # 1. 保存上传的 NII 文件到临时目录
        temp_nii_name = f"{uuid.uuid4()}_{file.filename}"
        temp_nii_path = os.path.join(TEMP_DIR, temp_nii_name)

        with open(temp_nii_path, "wb") as f:
            f.write(await file.read())

        # 2. 初始化推理器
        infer_engine = CTArtifactInfer(model_weight_path=WEIGHT_PATH, model_type=model_type)

        # 3. 定义掩码、特征保存路径
        mask_save_path = os.path.join(TEMP_DIR, mask_filename) if save_mask else None
        feat_save_path = os.path.join(TEMP_DIR, feature_filename) if save_feature else None

        # 4. 执行推理
        sitk_mask, feature_vector = infer_engine.predict_from_nii(
            nii_path=temp_nii_path,
            save_mask_path=mask_save_path,
            save_feature_path=feat_save_path
        )

        # 5. 清理原始上传文件
        if os.path.exists(temp_nii_path):
            os.remove(temp_nii_path)

        # 6. 返回掩码文件流，同时在 Header 携带特征维度
        if save_mask and os.path.exists(mask_save_path):
            headers = {
                "Content-Disposition": f'attachment; filename="{mask_filename}"',
                "Feature-Shape": str(feature_vector.shape) if feature_vector is not None else "None"
            }
            return StreamingResponse(
                open(mask_save_path, "rb"),
                media_type="application/octet-stream",
                headers=headers
            )
        else:
            # 不保存掩码，仅返回基础信息
            pixel_count = np.sum(sitk.GetArrayFromImage(sitk_mask) > 0)
            return {
                "code": 200,
                "msg": "推理完成",
                "artifact_pixel_count": int(pixel_count),
                "feature_shape": str(feature_vector.shape) if feature_vector is not None else None
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理异常：{str(e)}")

# ===================== 单独特征提取接口 =====================
@app.post("/extract/feature", summary="仅提取CT全局特征")
async def extract_volume_feature(
    file: UploadFile = File(...),
    model_type: str = Query(ModelType.UNET2D),
    feature_filename: str = Query("ct_feature.npy"),
    return_average: bool = Query(True)
):
    try:
        # 临时保存文件
        temp_nii_name = f"{uuid.uuid4()}_{file.filename}"
        temp_nii_path = os.path.join(TEMP_DIR, temp_nii_name)
        with open(temp_nii_path, "wb") as f:
            f.write(await file.read())

        # 提取特征
        infer_engine = CTArtifactInfer(weight_path=WEIGHT_PATH, model_type=model_type)
        feat = infer_engine.extract_volume_features(
            nii_path=temp_nii_path,
            save_feature_path=None,
            return_average=return_average
        )

        # 清理临时文件
        if os.path.exists(temp_nii_path):
            os.remove(temp_nii_path)

        # 返回特征二进制流
        feat_bytes = feat.tobytes()
        return StreamingResponse(
            iter([feat_bytes]),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{feature_filename}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"特征提取异常：{str(e)}")

# ===================== 患者特征管理接口 =====================
@app.get("/patients/list", summary="获取所有已录入患者列表")
async def get_patient_list():
    pid_list = []
    for file_name in os.listdir(PATIENT_FEATURE_DIR):
        if file_name.endswith("_feature.npy"):
            patient_id = file_name.replace("_feature.npy", "")
            pid_list.append(patient_id)
    return {
        "count": len(pid_list),
        "patients": pid_list
    }

@app.delete("/patients/{patient_id}", summary="删除指定患者特征")
async def delete_patient(patient_id: str):
    feat_file = os.path.join(PATIENT_FEATURE_DIR, f"{patient_id}_feature.npy")
    if not os.path.exists(feat_file):
        raise HTTPException(status_code=404, detail="该患者特征不存在")
    os.remove(feat_file)
    return {"msg": f"患者 {patient_id} 特征删除成功"}

@app.post("/save/patient_feature", summary="保存患者特征文件到特征库")
async def save_patient_feature(
    patient_id: str = Query(..., description="患者编号"),
    feature_file: UploadFile = File(..., description="特征.npy文件")
):
    save_path = os.path.join(PATIENT_FEATURE_DIR, f"{patient_id}_feature.npy")
    with open(save_path, "wb") as f:
        f.write(await feature_file.read())
    return {"msg": f"患者 {patient_id} 特征入库成功"}

# ===================== 聚类分析接口（t-SNE + KMeans） =====================
@app.get("/cluster/analysis", summary="特征聚类+降维可视化")
async def cluster_analysis(cluster_num: int = Query(2, ge=2, le=10, description="聚类数量K")):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    # 读取所有患者特征
    feat_files = [f for f in os.listdir(PATIENT_FEATURE_DIR) if f.endswith(".npy")]
    if len(feat_files) < 3:
        raise HTTPException(status_code=400, detail="聚类至少需要3条及以上患者数据")

    all_features = []
    patient_ids = []
    for f_name in feat_files:
        pid = f_name.replace("_feature.npy", "")
        feat_path = os.path.join(PATIENT_FEATURE_DIR, f_name)
        feat_data = np.load(feat_path)
        all_features.append(feat_data.squeeze())
        patient_ids.append(pid)

    # 转为数组
    feat_array = np.array(all_features)

    # t-SNE 降维到2维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feat_files)-1))
    points_2d = tsne.fit_transform(feat_array)

    # KMeans 聚类
    kmeans = KMeans(n_clusters=cluster_num, random_state=42)
    cluster_labels = kmeans.fit_predict(feat_array)

    # 组装前端渲染数据
    result = []
    for idx, pid in enumerate(patient_ids):
        result.append({
            "patientId": pid,
            "x": float(points_2d[idx][0]),
            "y": float(points_2d[idx][1]),
            "cluster": int(cluster_labels[idx])
        })

    return {"cluster_result": result}

# ===================== 服务入口 =====================
if __name__ == "__main__":
    import uvicorn
    # 启动服务，端口 8000，和前端接口地址对应
    uvicorn.run(
        app="CTDetectionServer:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )