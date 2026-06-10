from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
import shutil
from typing import Optional

from app.db.mysql import save_task, get_task, delete_patient
from app.tasks.infer_tasks import async_infer_task
from app.retrieval.service import retrieval
from Detection.CTArtifactInfer import ModelType

router = APIRouter()

# ---------------------- 异步推理接口（新）----------------------
@router.post("/async/predict")
async def async_predict(
    file: UploadFile = File(...),
    model_type: str = Query(ModelType.UNET2D),
    patient_id: str = Query(""),
    save_mask: bool = Query(True),
    save_feature: bool = Query(True)
):
    # 保存临时文件
    uid = str(uuid.uuid4())
    tmp_path = f"./uploads/{uid}_{file.filename}"
    os.makedirs("./uploads", exist_ok=True)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 创建任务
    task_id = str(uuid.uuid4())
    save_task(task_id, patient_id, model_type)

    # 提交 Celery 任务
    async_infer_task.delay(
        task_id=task_id,
        patient_id=patient_id,
        model_type=model_type,
        nii_path=tmp_path,
        save_mask=save_mask,
        save_feature=save_feature
    )

    return {"task_id": task_id, "status": "pending"}

# ---------------------- 查询任务状态 ----------------------
@router.get("/task/{task_id}")
def get_task_status(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return task

# ---------------------- 特征检索 ----------------------
@router.post("/retrieval/search")
def search_similar(
    feature_file: UploadFile = File(...),
    top_k: int = Query(5)
):
    import numpy as np
    try:
        feat = np.load(feature_file.file)
        feat = feat.flatten()
    except:
        raise HTTPException(400, "特征文件必须是 .npy 格式")

    result = retrieval.search(feat, top_k=top_k)
    return {"similar_patients": result}

# ---------------------- 患者管理 ----------------------
@router.delete("/patients/{patient_id}")
def remove_patient(patient_id: str):
    delete_patient(patient_id)
    retrieval.build_index()  # 重建索引
    return {"msg": "删除成功"}