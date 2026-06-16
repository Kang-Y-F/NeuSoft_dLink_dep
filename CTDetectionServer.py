# CTDetectionServer.py
# ============================================================
# FastAPI 主服务
# 职责分工：
#   同步推理    → /infer/nii（推理完整写 MySQL）
#   异步推理    → /api/async/*（Celery，见 endpoints.py）
#   特征提取    → /extract/feature
#   患者管理    → /patients/*（读写 MySQL，不再用本地文件夹）
#   聚类分析    → /cluster/analysis（特征来自 MySQL）
#   健康检测    → /health
# ============================================================

import os
import uuid
import json
import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.celery import celery                      # noqa: F401（确保 Celery 初始化）
from app.api.endpoints import router as async_router
from app.db.mysql import (
    init_db,
    task_create, task_update,
    feature_upsert, feature_load_all,
    feature_list_patients, feature_delete,
    report_upsert, get_conn,
)
from Detection.CTArtifactInfer import CTArtifactInfer
from Detection.model_enum import ModelType

# ── 全局配置 ──────────────────────────────────────────────────

WEIGHT_PATH = "./Model/weights/nor_best.pth"
TMP_DIR     = "./tmp"          # 统一临时目录，tmp/ 和 temp/ 合并为 tmp/
os.makedirs(TMP_DIR, exist_ok=True)

# ── FastAPI 实例 ───────────────────────────────────────────────

app = FastAPI(
    title       = "CT 伪影分割微服务",
    description = "提供 CT 伪影分割推理、特征提取、患者管理、聚类分析接口",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# 挂载异步推理路由
app.include_router(async_router, prefix="/api", tags=["异步推理"])


# ── 启动事件：初始化数据库 ────────────────────────────────────

@app.on_event("startup")
def on_startup():
    init_db()
    print("✅ 服务启动完成，MySQL 初始化完毕")


# ══════════════════════════════════════════════════════════════
#  基础接口
# ══════════════════════════════════════════════════════════════

@app.get("/health", summary="服务健康检测", tags=["基础"])
async def health_check():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "success", "device": device}


# ══════════════════════════════════════════════════════════════
#  同步推理接口（前端主接口）
# ══════════════════════════════════════════════════════════════

@app.post("/infer/nii", summary="NII 文件同步推理", tags=["推理"])
async def infer_nii(
    file: UploadFile = File(...),
    patient_id:       str  = Query("", description="患者编号，非空时自动入特征库"),
    save_mask:        bool = Query(True,  description="是否返回掩码文件"),
    mask_filename:    str  = Query("artifact_mask.nii.gz", description="掩码文件名"),
    model_type:       str  = Query(ModelType.UNET2D, description="模型类型"),
    save_feature:     bool = Query(True,  description="是否保存特征到数据库"),
    order_id:         int  = Query(None,  description="HIS 检查订单 ID（有则写 check_report）"),
    his_patient_id:   int  = Query(None,  description="HIS 患者数字 ID（写 check_report 时使用）"),
):
    """
    同步推理主接口。流程：
    1. 保存上传文件到 tmp/
    2. 推理（分割 + 特征提取）
    3. patient_id 非空 → 特征写 ct_patient_features
    4. order_id 非空  → 结果写 check_report（HIS 联动）
    5. 返回掩码文件流（Header 携带 Feature-Shape）
    """
    # 1. 落盘临时文件
    task_id       = str(uuid.uuid4())
    temp_nii_name = f"{task_id}_{file.filename}"
    temp_nii_path = os.path.join(TMP_DIR, temp_nii_name)

    with open(temp_nii_path, "wb") as f:
        f.write(await file.read())

    if save_mask:
        safe_pid = patient_id.replace("/", "_") if patient_id else "unknown"
        actual_mask_filename = f"{safe_pid}_mask.nii.gz"
        mask_save_path = os.path.join(TMP_DIR, actual_mask_filename)
    else:
        actual_mask_filename = mask_filename
        mask_save_path = None

    try:
        # 2. 推理
        infer_engine = CTArtifactInfer(
            model_weight_path=WEIGHT_PATH,
            model_type=model_type,
        )
        sitk_mask, feature_vector = infer_engine.predict_from_nii(
            nii_path=temp_nii_path,
            save_mask_path=mask_save_path,
        )

        # 统计指标
        mask_arr      = sitk.GetArrayFromImage(sitk_mask)
        pixel_count   = int(np.sum(mask_arr > 0))
        feat_shape    = str(feature_vector.shape) if feature_vector is not None else "None"

        # 3. 特征写库
        if save_feature and feature_vector is not None and patient_id:
            feature_upsert(patient_id, feature_vector, model_type)

        # 4. 写 HIS check_report（可选）
        if order_id and his_patient_id:
            report_upsert(
                patient_id           = his_patient_id,
                order_id             = order_id,
                mask_path            = mask_save_path or "",
                artifact_pixel_count = pixel_count,
                feature_shape        = feat_shape,
            )

        # 5. 返回
        if save_mask and mask_save_path and os.path.exists(mask_save_path):
            headers = {
                "Content-Disposition": f'attachment; filename="{actual_mask_filename}"',
                "Feature-Shape": feat_shape,
                "Artifact-Pixels": str(pixel_count),
            }
            return StreamingResponse(
                open(mask_save_path, "rb"),
                media_type="application/octet-stream",
                headers=headers,
            )
        else:
            return {
                "code":                 200,
                "msg":                  "推理完成",
                "artifact_pixel_count": pixel_count,
                "feature_shape":        feat_shape,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理异常：{e}")

    finally:
        if os.path.exists(temp_nii_path):
            os.remove(temp_nii_path)


# ══════════════════════════════════════════════════════════════
#  特征提取接口（不做分割，仅提取特征）
# ══════════════════════════════════════════════════════════════

@app.post("/extract/feature", summary="仅提取 CT 全局特征", tags=["推理"])
async def extract_volume_feature(
    file: UploadFile = File(...),
    patient_id:    str  = Query("", description="患者编号，非空时入库"),
    model_type:    str  = Query(ModelType.UNET2D),
    return_average: bool = Query(True),
):
    temp_nii_name = f"{uuid.uuid4()}_{file.filename}"
    temp_nii_path = os.path.join(TMP_DIR, temp_nii_name)

    with open(temp_nii_path, "wb") as f:
        f.write(await file.read())

    try:
        infer_engine = CTArtifactInfer(
            model_weight_path=WEIGHT_PATH,
            model_type=model_type,
        )
        feat = infer_engine.extract_volume_features(
            nii_path       = temp_nii_path,
            return_average = return_average,
        )

        if patient_id:
            feature_upsert(patient_id, feat, model_type)

        feat_bytes = feat.tobytes()
        return StreamingResponse(
            iter([feat_bytes]),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": 'attachment; filename="feature.npy"',
                "Feature-Shape":       str(feat.shape),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"特征提取异常：{e}")
    finally:
        if os.path.exists(temp_nii_path):
            os.remove(temp_nii_path)


# ══════════════════════════════════════════════════════════════
#  患者管理接口（全部读写 MySQL，不再依赖本地文件夹）
# ══════════════════════════════════════════════════════════════

@app.get("/patients/list", summary="获取已录入患者列表", tags=["患者管理"])
async def get_patient_list():
    rows = feature_list_patients()
    return {
        "count":    len(rows),
        "patients": [r["patient_id"] for r in rows],
        "detail":   rows,
    }


@app.delete("/patients/{patient_id}", summary="删除指定患者特征", tags=["患者管理"])
async def delete_patient(patient_id: str):
    ok = feature_delete(patient_id)
    if not ok:
        raise HTTPException(status_code=404, detail="该患者特征不存在")
    return {"msg": f"患者 {patient_id} 特征已删除"}


@app.post("/save/patient_feature", summary="上传患者特征文件入库", tags=["患者管理"])
async def save_patient_feature(
    patient_id:   str        = Query(..., description="患者编号"),
    model_type:   str        = Query(ModelType.UNET2D),
    feature_file: UploadFile = File(..., description=".npy 特征文件"),
):
    """接收前端上传的 .npy 特征文件，反序列化后存入 MySQL。"""
    raw = await feature_file.read()
    try:
        import io
        feat = np.load(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"特征文件解析失败：{e}")

    ok = feature_upsert(patient_id, feat, model_type)
    if not ok:
        raise HTTPException(status_code=500, detail="特征入库失败")
    return {"msg": f"患者 {patient_id} 特征入库成功", "feature_shape": str(feat.shape)}


# ══════════════════════════════════════════════════════════════
#  聚类分析接口
# ══════════════════════════════════════════════════════════════

@app.get("/cluster/analysis", summary="特征聚类 + t-SNE 降维可视化", tags=["分析"])
async def cluster_analysis(
    cluster_num: int = Query(2, ge=2, le=10, description="聚类数 K"),
):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    # 从 MySQL 加载全量特征
    data = feature_load_all()
    if len(data) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"聚类至少需要 3 名患者，当前仅有 {len(data)} 名"
        )

    patient_ids = list(data.keys())
    feat_array  = np.array(
        [v.squeeze() for v in data.values()], dtype=np.float32
    )

    # t-SNE 降维
    perplexity = min(30, len(patient_ids) - 1)
    tsne       = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    points_2d  = tsne.fit_transform(feat_array)

    # KMeans 聚类
    kmeans         = KMeans(n_clusters=cluster_num, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(feat_array)

    result = [
        {
            "patientId": pid,
            "x":         float(points_2d[i][0]),
            "y":         float(points_2d[i][1]),
            "cluster":   int(cluster_labels[i]),
        }
        for i, pid in enumerate(patient_ids)
    ]

    return {"cluster_result": result}



@app.get("/patients/check/{patient_id}", summary="检查患者是否存在", tags=["患者管理"])
async def check_patient(patient_id: int):
    conn = get_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="数据库连接失败")
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, name, gender FROM pmi_patient WHERE id=%s", (patient_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"患者 ID {patient_id} 不存在于系统中")
    return {"exists": True, "patient": row}

# ══════════════════════════════════════════════════════════════
#  服务入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "CTDetectionServer:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = False,
        log_level = "info",
    )
