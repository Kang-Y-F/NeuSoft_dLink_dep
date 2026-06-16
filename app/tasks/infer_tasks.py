# app/tasks/infer_tasks.py
# ============================================================
# Celery 异步推理任务
# 职责：接收文件字节流 + patient_id → 推理 → 写 MySQL（三张表）
# ============================================================

import os
import numpy as np
import SimpleITK as sitk

from app.core.celery import celery
from app.db.mysql import task_create, task_update, feature_upsert
from Detection.CTArtifactInfer import CTArtifactInfer
from Detection.model_enum import ModelType
from Conf.Config import DEVICE

WEIGHT_PATH = "./Model/weights/nor_best.pth"
TMP_DIR     = "./tmp"
os.makedirs(TMP_DIR, exist_ok=True)


@celery.task(bind=True, name="ct.infer")
def async_infer_task(
    self,
    file_bytes: bytes,
    file_suffix: str,
    patient_id: str,
    model_type: str = ModelType.UNET2D,
):
    """
    异步 CT 推理任务。

    参数
    ----
    file_bytes  : 上传的 .nii/.nii.gz 二进制内容
    file_suffix : 文件后缀，".nii" 或 ".gz"
    patient_id  : 患者编号（对应 ct_patient_features.patient_id）
    model_type  : 模型类型字符串

    返回（存入 Redis result backend）
    ----------------------------------
    {
        "status":       "success" | "failure",
        "task_id":      str,
        "patient_id":   str,
        "mask_path":    str,
        "feature_shape": str,
        "artifact_pixel_count": int,
        "msg":          str   # 仅 failure 时存在
    }
    """
    task_id = self.request.id

    # 1. 新建任务记录
    task_create(task_id, patient_id, model_type)

    tmp_nii_path  = os.path.join(TMP_DIR, f"{task_id}{file_suffix}")
    mask_save_path = os.path.join(TMP_DIR, f"{task_id}_mask.nii.gz")

    try:
        # 2. 落盘临时文件
        with open(tmp_nii_path, "wb") as f:
            f.write(file_bytes)

        # 3. 推理
        infer_engine = CTArtifactInfer(
            model_weight_path=WEIGHT_PATH,
            model_type=model_type,
            device=DEVICE,
        )
        sitk_mask, feat_vec = infer_engine.predict_from_nii(
            nii_path=tmp_nii_path,
            save_mask_path=mask_save_path,
        )

        # 4. 统计伪影像素
        mask_arr = sitk.GetArrayFromImage(sitk_mask)
        pixel_count = int(np.sum(mask_arr > 0))
        feat_shape  = str(feat_vec.shape) if feat_vec is not None else "None"

        # 5. 写患者特征到 MySQL
        if feat_vec is not None:
            feature_upsert(patient_id, feat_vec, model_type)

        # 6. 更新任务状态为 success
        task_update(
            task_id,
            status="success",
            mask_path=mask_save_path,
            feature_path=None,   # 特征已入库，无需单独存文件路径
        )

        result = {
            "status":               "success",
            "task_id":              task_id,
            "patient_id":           patient_id,
            "mask_path":            mask_save_path,
            "feature_shape":        feat_shape,
            "artifact_pixel_count": pixel_count,
        }
        return result

    except Exception as e:
        err_msg = str(e)
        task_update(task_id, status="failure", error=err_msg)
        return {
            "status":     "failure",
            "task_id":    task_id,
            "patient_id": patient_id,
            "msg":        err_msg,
        }

    finally:
        # 7. 清理临时上传文件（掩码文件保留，供下载）
        if os.path.exists(tmp_nii_path):
            os.remove(tmp_nii_path)
