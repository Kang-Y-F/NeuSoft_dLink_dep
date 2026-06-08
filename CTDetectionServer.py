# CTDetectionServer.py
# ============================================================
# 融合版 FastAPI 服务：
#   - 老师版结构（上传 → 推理 → 下载掩码，UUID 防重名）
#   - 自写版功能（双模型切换、特征提取、患者管理、聚类分析）
# ============================================================

import os
import re
import shutil
import uuid

import numpy as np
import SimpleITK as sitk
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from Detection.CTArtifactInfer import CTArtifactInfer, ModelType

# ── 配置项（集中管理，改一处全局生效）────────────────────────────────
MODEL_WEIGHTS = {
    ModelType.UNET2D:           "./Model/weights/nor_best.pth",
    ModelType.ATTENTION_UNET2D: "./Model/weights/atten_best.pth",
}
UPLOAD_DIR  = "uploads"
RESULT_DIR  = "results"
FEATURE_DIR = "saved_features"

for d in (UPLOAD_DIR, RESULT_DIR, FEATURE_DIR):
    os.makedirs(d, exist_ok=True)

# ── 预加载所有模型（启动一次，请求零延迟）───────────────────────────
infer_engines: dict[str, CTArtifactInfer] = {}
try:
    for model_type, weight_path in MODEL_WEIGHTS.items():
        infer_engines[model_type] = CTArtifactInfer(
            model_weight_path=weight_path,
            model_type=model_type,
        )
    print("✅ 所有模型加载成功")
except Exception as e:
    raise RuntimeError(f"❌ 模型初始化失败：{e}")

# ── 患者特征内存缓存（启动时从磁盘恢复）───────────────────────────────
# key = patient_id (str)，value = np.ndarray shape (960,)
patient_feature_cache: dict[str, np.ndarray] = {}


def _load_saved_features():
    """扫描 FEATURE_DIR 中所有 {patient_id}_feature.npy，加载到内存"""
    count = 0
    for fname in os.listdir(FEATURE_DIR):
        if not fname.endswith("_feature.npy"):
            continue
        try:
            pid  = fname.replace("_feature.npy", "")
            arr  = np.load(os.path.join(FEATURE_DIR, fname)).flatten()
            patient_feature_cache[pid] = arr
            count += 1
            print(f"  📂 加载患者特征: {pid}")
        except Exception as ex:
            print(f"  ⚠️  加载失败 {fname}: {ex}")
    print(f"✅ 共加载 {count} 个患者特征")


_load_saved_features()

# ── FastAPI 初始化 ─────────────────────────────────────────────────
app = FastAPI(
    title="CT 金属伪影检测 AI 服务",
    description="基于 UNet2D / AttentionUNet2D 的 CT 伪影分割与特征分析接口",
    version="3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 生产环境替换为具体前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 工具函数 ───────────────────────────────────────────────────────

def _check_nifti(filename: str):
    if not filename.lower().endswith((".nii", ".nii.gz")):
        raise HTTPException(400, "仅支持 .nii 或 .nii.gz 格式的 CT 文件")


def _get_engine(model_type: str) -> CTArtifactInfer:
    engine = infer_engines.get(model_type)
    if engine is None:
        raise HTTPException(400, f"不支持的模型类型：{model_type}")
    return engine


# ══════════════════════════════════════════════════════════════════
# 基础接口
# ══════════════════════════════════════════════════════════════════

@app.get("/", summary="健康检查")
async def root():
    return {
        "status": "ok",
        "message": "CT 金属伪影检测服务运行中",
        "supported_models": list(infer_engines.keys()),
    }


# ══════════════════════════════════════════════════════════════════
# 核心推理接口
# ══════════════════════════════════════════════════════════════════

@app.post("/predict-ct-artifact", summary="上传 CT → 推理 → 返回掩码 + 特征向量")
async def predict_ct(
    file: UploadFile = File(..., description="CT 文件，.nii / .nii.gz"),
    model_type: str  = Query(ModelType.UNET2D, description="unet2d 或 attention_unet2d"),
    save_feature: bool = Query(True, description="是否保存特征向量到磁盘"),
    patient_id: str  = Query("", description="患者 ID，填写后自动注册到聚类库"),
):
    """
    主推理接口：
    1. 上传 CT → 分割推理 → 保存掩码
    2. 同时提取特征向量，可选保存 .npy
    3. 若提供 patient_id，自动注册到聚类内存缓存（持久化到磁盘）
    4. 返回掩码文件（FileResponse），特征信息附在响应头
    """
    _check_nifti(file.filename)

    uid              = str(uuid.uuid4())
    stem             = file.filename.replace(".nii.gz", "").replace(".nii", "")
    upload_path      = os.path.join(UPLOAD_DIR, f"{uid}_{file.filename}")
    mask_filename    = f"{uid}_{stem}_mask.nii.gz"
    mask_save_path   = os.path.join(RESULT_DIR, mask_filename)
    feature_filename = f"{uid}_{stem}_feature.npy"
    feature_save_path = os.path.join(FEATURE_DIR, feature_filename) if save_feature else None

    # 1. 保存上传文件
    try:
        with open(upload_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 2. 读取 CT
        sitk_ct = sitk.ReadImage(upload_path)

        # 3. 推理（返回掩码 + 特征向量）
        engine = _get_engine(model_type)
        sitk_mask, feature_vector = engine.predict_from_sitk(
            sitk_ct,
            save_mask_path=mask_save_path,
            save_feature_path=feature_save_path,
        )

        # 4. 若提供 patient_id，注册到聚类缓存
        _register_patient(patient_id, feature_vector)

        # 5. 构建响应头（特征信息）
        extra_headers = {
            "X-Feature-Shape":    str(feature_vector.shape) if feature_vector is not None else "None",
            "X-Feature-File":     feature_filename if save_feature else "",
            "X-CT-Shape":         str(sitk_mask.GetSize()),
            "X-Model-Type":       model_type,
        }

        return FileResponse(
            path=mask_save_path,
            filename=mask_filename,
            media_type="application/octet-stream",
            headers=extra_headers,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"推理失败：{e}")


# ══════════════════════════════════════════════════════════════════
# 仅提取特征接口（不返回掩码，速度更快）
# ══════════════════════════════════════════════════════════════════

@app.post("/extract/feature", summary="仅提取特征向量（不返回掩码）")
async def extract_feature(
    file: UploadFile = File(...),
    model_type: str = Query(ModelType.UNET2D),
    patient_id: str = Query("", description="患者 ID，填写后自动注册到聚类库"),
    return_average: bool = Query(True, description="True=全卷平均特征，False=每切片特征"),
):
    _check_nifti(file.filename)

    uid  = str(uuid.uuid4())
    temp = os.path.join(UPLOAD_DIR, f"tmp_{uid}.nii.gz")
    feat_filename = f"{uid}_feature.npy"
    feat_path     = os.path.join(FEATURE_DIR, feat_filename)

    try:
        with open(temp, "wb") as f:
            shutil.copyfileobj(file.file, f)

        engine = _get_engine(model_type)
        feature_vector = engine.extract_volume_features(
            nii_path=temp,
            save_feature_path=feat_path,
            return_average=return_average,
        )

        _register_patient(patient_id, feature_vector)

        return FileResponse(
            path=feat_path,
            filename=feat_filename,
            media_type="application/octet-stream",
            headers={"X-Feature-Shape": str(feature_vector.shape)},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"特征提取失败：{e}")
    finally:
        if os.path.exists(temp):
            os.remove(temp)


# ══════════════════════════════════════════════════════════════════
# 掩码下载
# ══════════════════════════════════════════════════════════════════

@app.get("/results/{mask_filename}", summary="下载分割掩码文件")
async def download_mask(mask_filename: str):
    path = os.path.join(RESULT_DIR, mask_filename)
    if not os.path.exists(path):
        raise HTTPException(404, "掩码文件不存在")
    return FileResponse(
        path=path,
        media_type="application/octet-stream",
        filename=mask_filename,
    )


# ══════════════════════════════════════════════════════════════════
# 患者特征管理
# ══════════════════════════════════════════════════════════════════

@app.post("/save/patient_feature", summary="手动上传患者特征 .npy 并注册")
async def save_patient_feature(
    patient_id: str,
    feature_file: UploadFile = File(...),
):
    try:
        arr = np.load(feature_file.file).flatten()
        _register_patient(patient_id, arr, force=True)
        return {"msg": "保存成功", "patient_id": patient_id}
    except Exception as e:
        raise HTTPException(500, f"保存失败：{e}")


@app.get("/patients/list", summary="获取已注册患者列表")
async def get_patient_list():
    return {
        "patients": list(patient_feature_cache.keys()),
        "count":    len(patient_feature_cache),
    }


@app.delete("/patients/{patient_id}", summary="删除患者特征")
async def delete_patient_feature(patient_id: str):
    if patient_id not in patient_feature_cache:
        raise HTTPException(404, f"患者 {patient_id} 不存在")
    del patient_feature_cache[patient_id]
    disk_path = os.path.join(FEATURE_DIR, f"{patient_id}_feature.npy")
    if os.path.exists(disk_path):
        os.remove(disk_path)
    return {"msg": "删除成功", "patient_id": patient_id}


# ══════════════════════════════════════════════════════════════════
# 聚类分析
# ══════════════════════════════════════════════════════════════════

@app.get("/cluster/analysis", summary="对已注册患者做 KMeans + t-SNE 可视化")
async def cluster_analysis(
    cluster_num: int = Query(2, ge=2, description="KMeans 簇数，最小 2"),
):
    """
    对 patient_feature_cache 中的特征做：
    1. StandardScaler 标准化
    2. t-SNE 降维到 2D（样本数 ≤ 3 时自动退化为 PCA）
    3. KMeans 聚类
    返回前端直接可用的坐标 + 标签 JSON。
    """
    n = len(patient_feature_cache)
    if n < 2:
        raise HTTPException(400, f"患者数量不足（当前 {n}），聚类至少需要 2 名患者")
    if cluster_num > n:
        raise HTTPException(400, f"簇数 {cluster_num} 不能大于患者数量 {n}")

    pids     = list(patient_feature_cache.keys())
    feat_arr = np.array([patient_feature_cache[k] for k in pids])   # (N, 960)

    scaler      = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_arr)

    # ── 降维（自适应 perplexity）────────────────────────────────
    if n <= 3:
        # 极少样本退化为 PCA
        n_comp = min(2, n)
        pca    = PCA(n_components=n_comp, random_state=42)
        xy_raw = pca.fit_transform(feat_scaled)
        xy_data = (
            np.column_stack([xy_raw, np.zeros(n)])
            if n_comp == 1 else xy_raw
        )
        method_used = "PCA"
    else:
        perplexity = min(30, max(2, n - 1))
        tsne       = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,
            learning_rate="auto",
        )
        xy_data     = tsne.fit_transform(feat_scaled)
        method_used = f"t-SNE (perplexity={perplexity})"

    # ── KMeans ─────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=cluster_num, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feat_scaled)

    result = [
        {
            "patientId": pids[i],
            "x":         float(xy_data[i][0]),
            "y":         float(xy_data[i][1]),
            "cluster":   int(labels[i]),
        }
        for i in range(n)
    ]

    return {
        "cluster_result": result,
        "method":         method_used,
        "patient_count":  n,
    }


# ══════════════════════════════════════════════════════════════════
# 内部工具函数
# ══════════════════════════════════════════════════════════════════

def _register_patient(
    patient_id: str,
    feature_vector: np.ndarray | None,
    force: bool = False,
):
    """
    将特征向量注册到内存缓存，并持久化到磁盘。
    patient_id 为空字符串时跳过。
    """
    if not patient_id or feature_vector is None:
        return

    arr = feature_vector.flatten()
    patient_feature_cache[patient_id] = arr

    disk_path = os.path.join(FEATURE_DIR, f"{patient_id}_feature.npy")
    np.save(disk_path, arr)
    print(f"✅ 患者 {patient_id} 特征已注册（内存 + 磁盘）")


# ══════════════════════════════════════════════════════════════════
# 启动
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "CTDetectionServer:app",
        host="0.0.0.0",
        port=8000,
        reload=True,   # 开发模式；生产环境改为 False 并增加 workers
        workers=1,
    )
