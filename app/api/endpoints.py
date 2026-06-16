# app/api/endpoints.py
# ============================================================
# 异步推理接口路由（挂载于 /api 前缀）
# 同步推理接口统一在 CTDetectionServer.py 中定义
# ============================================================

import os
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse

from app.tasks.infer_tasks import async_infer_task
from app.db.mysql import task_get
from Detection.model_enum import ModelType

router = APIRouter()
TMP_DIR = "./tmp"


# ── 提交异步推理任务 ──────────────────────────────────────────

@router.post(
    "/async/infer/submit",
    summary="提交异步推理任务",
    description="上传 .nii/.nii.gz 文件，携带患者编号，返回 task_id 供后续轮询。"
)
async def submit_async_task(
    file: UploadFile = File(...),
    patient_id: str  = Query(..., description="患者编号，如 P001"),
    model_type: str  = Query(ModelType.UNET2D, description="模型类型"),
):
    suffix = os.path.splitext(file.filename)[-1].lower()
    # .nii.gz 的 splitext 只取到 .gz，需特殊处理
    if file.filename.endswith(".nii.gz"):
        suffix = ".gz"
    elif suffix not in (".nii",):
        raise HTTPException(status_code=400, detail="仅支持 .nii / .nii.gz 文件")

    file_bytes = await file.read()

    task = async_infer_task.delay(
        file_bytes  = file_bytes,
        file_suffix = suffix,
        patient_id  = patient_id,
        model_type  = model_type,
    )
    return {
        "code":       200,
        "msg":        "任务已提交",
        "task_id":    task.id,
        "patient_id": patient_id,
    }


# ── 查询任务状态 ──────────────────────────────────────────────

@router.get(
    "/async/infer/status",
    summary="查询异步任务状态",
    description="优先读 MySQL 持久状态，Redis 作为降级兜底。"
)
async def get_task_status(
    task_id: str = Query(..., description="submit 接口返回的 task_id")
):
    # 优先读 MySQL（持久，重启后仍可查）
    row = task_get(task_id)
    if row:
        return {
            "source":       "mysql",
            "state":        row["status"].upper(),
            "task_id":      task_id,
            "patient_id":   row.get("patient_id"),
            "mask_path":    row.get("mask_path"),
            "feature_path": row.get("feature_path"),
            "created_at":   str(row.get("created_at", "")),
            "finished_at":  str(row.get("finished_at", "")),
            "error":        row.get("error"),
        }

    # 降级：读 Redis result backend
    task_result = async_infer_task.AsyncResult(task_id)
    state_map = {
        "PENDING":  "PENDING",
        "STARTED":  "RUNNING",
        "PROGRESS": "RUNNING",
        "SUCCESS":  "SUCCESS",
        "FAILURE":  "FAILURE",
    }
    state = state_map.get(task_result.state, task_result.state)
    resp  = {"source": "redis", "state": state, "task_id": task_id}

    if task_result.state == "SUCCESS":
        resp["data"] = task_result.result
    elif task_result.state == "FAILURE":
        resp["msg"] = str(task_result.info)

    return resp


# ── 下载异步推理生成的掩码文件 ────────────────────────────────

@router.get(
    "/async/infer/download",
    summary="下载异步推理掩码文件",
)
async def download_mask(
    task_id: str = Query(..., description="任务 ID")
):
    # 优先从 MySQL 拿路径
    row = task_get(task_id)
    mask_path = row.get("mask_path") if row else None

    # 兜底：按约定命名查找
    if not mask_path or not os.path.exists(mask_path):
        for ext in (".nii.gz", ".nii"):
            candidate = os.path.join(TMP_DIR, f"{task_id}_mask{ext}")
            if os.path.exists(candidate):
                mask_path = candidate
                break

    if not mask_path or not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="掩码文件不存在或任务未完成")

    return FileResponse(
        path=mask_path,
        filename=os.path.basename(mask_path),
        media_type="application/octet-stream",
    )
