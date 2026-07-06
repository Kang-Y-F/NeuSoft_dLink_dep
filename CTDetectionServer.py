from langchain_community.cache import RedisCache
from langchain_core.globals import set_llm_cache
import redis

try:
    redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0)
    redis_client.ping()
    set_llm_cache(RedisCache(redis_client))
    print("[Cache] Redis 缓存已启用")
except Exception:
    from langchain_core.caches import InMemoryCache
    set_llm_cache(InMemoryCache())
    print("[Cache] Redis 不可用，使用内存缓存")
# ==========================================================================
import os
import uuid
import shutil
import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import lab_endpoints
from app.api import hl7_endpoints
from app.api import preview_endpoints

from app.core.celery import celery
from app.api.endpoints import router as async_router

# ── MySQL：继续存患者基本信息、检查单、报告等所有原有业务数据 ──────────
from app.db.mysql import (
    init_db,
    task_create, task_update,
    report_upsert, get_conn,
    get_pending_ct_orders, complete_ct_order,
)

# ── pgvector：只负责 CT 特征向量的存取（相似度检索 / 聚类分析）─────────
from app.db.pgvector_store import (
    init_pgvector_db,
    feature_upsert_vec,
    find_similar_vec,
    feature_delete_vec,
    feature_list_patients_vec,
    feature_load_all_vec,
)

from Detection.CTArtifactInfer import CTArtifactInfer
from Detection.model_enum import ModelType

WEIGHT_PATH      = "./Model/weights/nor_best.pth"
TMP_DIR          = "./tmp"
SERVICE_BASE_URL = "http://localhost:8000"
os.makedirs(TMP_DIR, exist_ok=True)

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

app.include_router(async_router, prefix="/api", tags=["异步推理"])
app.include_router(lab_endpoints.router)
app.include_router(hl7_endpoints.router)
app.include_router(preview_endpoints.router)


@app.on_event("startup")
def on_startup():
    init_db()
    print("✅ MySQL 初始化完毕")
    init_pgvector_db()
    print("✅ 服务启动完成")


# ══════════════════════════════════════════════════════════════
#  基础接口
# ══════════════════════════════════════════════════════════════

@app.get("/health", summary="服务健康检测", tags=["基础"])
async def health_check():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"status": "success", "device": device}


# ══════════════════════════════════════════════════════════════
#  文件下载接口（掩码 + 原始CT，供前端渲染）
# ══════════════════════════════════════════════════════════════

@app.get("/mask/{filename}", summary="获取掩码或CT文件", tags=["文件"])
async def get_mask_file(filename: str):
    safe_name = os.path.basename(filename)
    path = os.path.join(TMP_DIR, safe_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"文件不存在：{safe_name}")
    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'}
    )


# ══════════════════════════════════════════════════════════════
#  查询患者待执行CT检查单
# ══════════════════════════════════════════════════════════════

@app.get("/orders/pending/{patient_id}", summary="查询患者待执行CT检查单", tags=["检查单"])
async def get_pending_orders(patient_id: int):
    orders = get_pending_ct_orders(patient_id)
    return {"patient_id": patient_id, "count": len(orders), "orders": orders}


# ══════════════════════════════════════════════════════════════
#  同步推理接口
# ══════════════════════════════════════════════════════════════

@app.post("/infer/nii", summary="NII 文件同步推理", tags=["推理"])
async def infer_nii(
    file: UploadFile = File(...),
    patient_id:     str  = Query(""),
    save_mask:      bool = Query(True),
    mask_filename:  str  = Query("artifact_mask.nii.gz"),
    model_type:     str  = Query(ModelType.UNET2D),
    save_feature:   bool = Query(True),
    order_id:       int  = Query(None),
    his_patient_id: int  = Query(None),
):
    task_id       = str(uuid.uuid4())
    temp_nii_name = f"{task_id}_{file.filename}"
    temp_nii_path = os.path.join(TMP_DIR, temp_nii_name)

    with open(temp_nii_path, "wb") as f:
        f.write(await file.read())

    safe_pid = patient_id.replace("/", "_") if patient_id else "unknown"

    # 掩码路径
    actual_mask_filename = f"{safe_pid}_mask.nii.gz"
    mask_save_path = os.path.join(TMP_DIR, actual_mask_filename) if save_mask else None

    # 原始CT保存路径（推理前复制保留，用于医生端渲染）
    ct_save_filename = f"{safe_pid}_ct.nii.gz"
    ct_save_path     = os.path.join(TMP_DIR, ct_save_filename)

    try:
        # ── 推理前先保存原始CT文件 ────────────────────────────────
        shutil.copy2(temp_nii_path, ct_save_path)
        ct_url = f"{SERVICE_BASE_URL}/mask/{ct_save_filename}"

        # ── 推理 ──────────────────────────────────────────────────
        infer_engine = CTArtifactInfer(
            model_weight_path=WEIGHT_PATH,
            model_type=model_type,
        )
        sitk_mask, feature_vector = infer_engine.predict_from_nii(
            nii_path=temp_nii_path,
            save_mask_path=mask_save_path,
        )

        mask_arr    = sitk.GetArrayFromImage(sitk_mask)
        pixel_count = int(np.sum(mask_arr > 0))
        feat_shape  = str(feature_vector.shape) if feature_vector is not None else "None"

        # ── 特征写入 pgvector（原来是写 MySQL，现在改成写向量库）─────
        if save_feature and feature_vector is not None and patient_id:
            feature_upsert_vec(int(patient_id), feature_vector, model_type)

        # ── 写 check_report（掩码URL + CT URL 都存入，仍走 MySQL）────
        write_patient_id = his_patient_id or (
            int(patient_id) if patient_id and patient_id.isdigit() else None
        )
        if write_patient_id:
            mask_url = (
                f"{SERVICE_BASE_URL}/mask/{actual_mask_filename}"
                if save_mask else ""
            )
            report_upsert(
                patient_id           = write_patient_id,
                order_id             = order_id,
                mask_path            = mask_url,
                ct_path              = ct_url,       # ← 原始CT URL
                artifact_pixel_count = pixel_count,
                feature_shape        = feat_shape,
            )

        # ── 完成检查单 ────────────────────────────────────────────
        if order_id:
            complete_ct_order(order_id)

        # ── 返回掩码文件流 ────────────────────────────────────────
        if save_mask and mask_save_path and os.path.exists(mask_save_path):
            headers = {
                "Content-Disposition": f'attachment; filename="{actual_mask_filename}"',
                "Feature-Shape":       feat_shape,
                "Artifact-Pixels":     str(pixel_count),
                "Mask-Url":            f"{SERVICE_BASE_URL}/mask/{actual_mask_filename}",
                "Ct-Url":              ct_url,
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
                "ct_url":               ct_url,
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理异常：{e}")
    finally:
        if os.path.exists(temp_nii_path):
            os.remove(temp_nii_path)


# ══════════════════════════════════════════════════════════════
#  特征提取接口
# ══════════════════════════════════════════════════════════════

@app.post("/extract/feature", summary="仅提取 CT 全局特征", tags=["推理"])
async def extract_volume_feature(
    file: UploadFile = File(...),
    patient_id:     str  = Query(""),
    model_type:     str  = Query(ModelType.UNET2D),
    return_average: bool = Query(True),
):
    temp_nii_name = f"{uuid.uuid4()}_{file.filename}"
    temp_nii_path = os.path.join(TMP_DIR, temp_nii_name)
    with open(temp_nii_path, "wb") as f:
        f.write(await file.read())
    try:
        infer_engine = CTArtifactInfer(model_weight_path=WEIGHT_PATH, model_type=model_type)
        feat = infer_engine.extract_volume_features(
            nii_path=temp_nii_path, return_average=return_average
        )
        if patient_id:
            feature_upsert_vec(int(patient_id), feat, model_type)
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
#  患者管理接口
# ══════════════════════════════════════════════════════════════

@app.get("/patients/list", summary="获取已录入患者列表", tags=["患者管理"])
async def get_patient_list():
    rows = feature_list_patients_vec()
    return {"count": len(rows), "patients": [r["patient_id"] for r in rows], "detail": rows}


@app.delete("/patients/{patient_id}", summary="删除指定患者特征", tags=["患者管理"])
async def delete_patient(patient_id: str):
    ok = feature_delete_vec(int(patient_id))
    if not ok:
        raise HTTPException(status_code=404, detail="该患者特征不存在")
    return {"msg": f"患者 {patient_id} 特征已删除"}


@app.post("/save/patient_feature", summary="上传患者特征文件入库", tags=["患者管理"])
async def save_patient_feature(
    patient_id:   str        = Query(...),
    model_type:   str        = Query(ModelType.UNET2D),
    feature_file: UploadFile = File(...),
):
    raw = await feature_file.read()
    try:
        import io
        feat = np.load(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"特征文件解析失败：{e}")
    ok = feature_upsert_vec(int(patient_id), feat, model_type)
    if not ok:
        raise HTTPException(status_code=500, detail="特征入库失败")
    return {"msg": f"患者 {patient_id} 特征入库成功", "feature_shape": str(feat.shape)}


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
        raise HTTPException(status_code=404, detail=f"患者 ID {patient_id} 不存在")
    return {"exists": True, "patient": row}


# ══════════════════════════════════════════════════════════════
#  聚类分析接口
# ══════════════════════════════════════════════════════════════

@app.get("/cluster/analysis", summary="特征聚类 + t-SNE 降维可视化", tags=["分析"])
async def cluster_analysis(cluster_num: int = Query(2, ge=2, le=10)):
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans

    data = feature_load_all_vec()
    if len(data) < 3:
        raise HTTPException(status_code=400, detail=f"聚类至少需要 3 名患者，当前仅有 {len(data)} 名")

    patient_ids = [str(k) for k in data.keys()]
    feat_array  = np.array([v.squeeze() for v in data.values()], dtype=np.float32)
    perplexity  = min(30, len(patient_ids) - 1)
    tsne        = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    points_2d   = tsne.fit_transform(feat_array)
    kmeans      = KMeans(n_clusters=cluster_num, random_state=42, n_init=10)
    labels      = kmeans.fit_predict(feat_array)

    return {"cluster_result": [
        {"patientId": pid, "x": float(points_2d[i][0]), "y": float(points_2d[i][1]), "cluster": int(labels[i])}
        for i, pid in enumerate(patient_ids)
    ]}


# ══════════════════════════════════════════════════════════════
#  相似病例检索（pgvector：HNSW 近似最近邻 + 余弦距离）
# ══════════════════════════════════════════════════════════════

@app.get("/similar/{patient_id}", summary="检索相似病例", tags=["分析"])
async def find_similar_patients(
    patient_id: int,
    top_k: int = Query(5, ge=1, le=20, description="返回相似患者数量"),
):
    """
    基于CT特征向量的余弦相似度，检索与目标患者最相似的K个患者。
    底层由 pgvector 的 HNSW 索引完成近似最近邻检索，
    而不是把全部特征读到内存里逐个计算（那是旧版 MySQL 方案的做法）。
    """
    results = find_similar_vec(patient_id, top_k)
    if results is None:
        raise HTTPException(status_code=404, detail=f"患者 {patient_id} 无特征数据")
    return {"target": patient_id, "count": len(results), "results": results}


# ══════════════════════════════════════════════════════════════
#  AI多模态病历生成（LangChain）
# ══════════════════════════════════════════════════════════════

@app.post("/ai/generate-report/{patient_id}", summary="AI多模态综合诊疗报告", tags=["AI报告"])
async def generate_ai_report(patient_id: int):
    from app.langchain_report.report_chain import generate_report
    try:
        result = generate_report(patient_id)
        if result.get("error") and not result.get("final_report"):
            raise HTTPException(status_code=500, detail=f"AI生成失败：{result['error']}")
        return {
            "code": 200,
            "patient_id": patient_id,
            "patient_name": result["patient_name"],
            "chain_steps": result["chain_steps"],
            "report": result["final_report"],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"报告生成异常：{str(e)}")

@app.get("/ai/report-status/{patient_id}", summary="查询报告生成状态", tags=["AI报告"])
async def report_status(patient_id: int):
    """检查患者数据是否足够生成报告"""
    from app.langchain_report.data_collector import collect_patient_data

    try:
        data = collect_patient_data(patient_id)
        return {
            "patient_id": patient_id,
            "patient_name": data["patient_info"]["name"],
            "ready": True,
            "data_count": {
                "medical_records": len(data["medical_records"]),
                "check_reports": len(data["check_reports"]),
                "lab_reports": len(data["lab_reports"]),
                "prescriptions": len(data["prescriptions"]),
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
# ══════════════════════════════════════════════════════════════
#  静态文件 & CT前端
# ══════════════════════════════════════════════════════════════

app.mount("/static", StaticFiles(directory="HTML"), name="static")

@app.get("/ct-viewer", include_in_schema=False)
async def ct_viewer():
    return FileResponse("HTML/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("CTDetectionServer:app", host="0.0.0.0", port=8000, reload=False, log_level="info", timeout_keep_alive=180)