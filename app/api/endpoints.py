import os
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
import uuid

from Detection.CTArtifactInfer import CTArtifactInfer, ModelType
from app.tasks.infer_tasks import async_infer_task
from Conf.Config import DEVICE

router = APIRouter()
WEIGHT_PATH = "./Model/weights/nor_best.pth"
TMP_DIR = "./tmp"
os.makedirs(TMP_DIR, exist_ok=True)

# ===================== 原有【同步接口】保留不动 =====================
@router.post("/infer/nii")
async def infer_nii(
    file: UploadFile = File(...),
    save_mask: bool = Query(True),
    mask_filename: str = Query("artifact_mask.nii.gz"),
    model_type: str = Query(ModelType.UNET2D),
    save_feature: bool = Query(True),
    feature_filename: str = Query("feature.npy")
):
    import uuid
    import numpy as np
    import SimpleITK as sitk

    temp_name = f"{uuid.uuid4()}_{file.filename}"
    temp_nii_path = os.path.join(TMP_DIR, temp_name)

    # 保存上传文件
    with open(temp_nii_path, "wb") as f:
        f.write(await file.read())

    # 初始化推理器
    infer_engine = CTArtifactInfer(
        model_weight_path=WEIGHT_PATH,
        model_type=model_type
    )

    mask_save_path = os.path.join(TMP_DIR, mask_filename) if save_mask else None
    feat_save_path = os.path.join(TMP_DIR, feature_filename) if save_feature else None

    sitk_mask, feature_vector = infer_engine.predict_from_nii(
        nii_path=temp_nii_path,
        save_mask_path=mask_save_path,
        save_feature_path=feat_save_path
    )

    # 删除临时上传文件
    if os.path.exists(temp_nii_path):
        os.remove(temp_nii_path)

    if save_mask and mask_save_path and os.path.exists(mask_save_path):
        return FileResponse(
            path=mask_save_path,
            filename=mask_filename,
            media_type="application/octet-stream"
        )
    else:
        pixel_count = int(np.sum(sitk.GetArrayFromImage(sitk_mask) > 0))
        return {
            "code": 200,
            "msg": "推理完成",
            "artifact_pixel_count": pixel_count,
            "feature_shape": list(feature_vector.shape) if feature_vector is not None else None
        }

# ===================== 新增【异步接口】=====================
@router.post("/async/infer/submit")
async def submit_async_task(file: UploadFile = File(...)):
    """提交异步推理任务，返回 task_id"""
    # 获取文件二进制 + 后缀
    file_bytes = await file.read()
    suffix = os.path.splitext(file.filename)[1]
    if suffix not in (".nii", ".gz"):
        raise HTTPException(status_code=400, detail="仅支持 .nii / .nii.gz 文件")

    # 派发Celery任务
    task = async_infer_task.delay(file_bytes, suffix)
    return {
        "code": 200,
        "msg": "任务已提交",
        "task_id": task.id
    }


@router.get("/async/infer/status")
async def get_task_status(task_id: str = Query(..., description="任务ID")):
    """查询异步任务状态"""
    task_result = async_infer_task.AsyncResult(task_id)
    if task_result.state == "PENDING":
        return {"state": "PENDING", "msg": "任务等待执行"}
    elif task_result.state == "PROGRESS":
        return {"state": "PROGRESS", "msg": "任务执行中"}
    elif task_result.state == "SUCCESS":
        return {"state": "SUCCESS", "data": task_result.result}
    elif task_result.state == "FAILURE":
        return {"state": "FAILURE", "msg": f"任务失败: {task_result.info}"}
    else:
        return {"state": task_result.state, "msg": "未知状态"}


@router.get("/async/infer/download")
async def download_mask(task_id: str = Query(...)):
    """下载异步推理生成的掩码文件"""
    mask_path = os.path.join(TMP_DIR, f"{task_id}_mask.nii.gz")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(TMP_DIR, f"{task_id}_mask.nii")
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="文件不存在或任务未完成")

    return FileResponse(
        path=mask_path,
        filename=os.path.basename(mask_path),
        media_type="application/octet-stream"
    )