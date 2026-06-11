import os
import time  # 新增导入
import uuid
import numpy as np
import SimpleITK as sitk
from app.core.celery import celery
from Detection.CTArtifactInfer import CTArtifactInfer, ModelType
from Conf.Config import DEVICE

# 全局配置
WEIGHT_PATH = "./Model/weights/nor_best.pth"
# 临时文件/结果存放目录
TMP_DIR = "./tmp"
os.makedirs(TMP_DIR, exist_ok=True)


@celery.task(bind=True)
def async_infer_task(self, file_bytes: bytes, file_suffix: str):
    """
    异步推理主任务
    :param file_bytes: 上传的nii文件二进制流
    :param file_suffix: 文件后缀 .nii / .nii.gz
    :return: 结果文件路径、特征向量
    """
    try:
        # ========== 模拟长耗时，测试用，正式上线删掉这行 ==========
        print("🔹 模拟耗时推理，等待10秒...")
        time.sleep(10)
        # ========================================================

        # 1. 生成临时文件名，保存上传文件
        task_id = self.request.id
        tmp_nii_path = os.path.join(TMP_DIR, f"{task_id}{file_suffix}")
        with open(tmp_nii_path, "wb") as f:
            f.write(file_bytes)

        # 2. 初始化推理器（复用你原有推理类）
        infer_engine = CTArtifactInfer(
            model_weight_path=WEIGHT_PATH,
            model_type=ModelType.UNET2D,
            device=DEVICE
        )

        # 3. 推理 + 保存掩码、特征
        mask_save_path = os.path.join(TMP_DIR, f"{task_id}_mask{file_suffix}")
        feat_save_path = os.path.join(TMP_DIR, f"{task_id}_feat.npy")
        sitk_mask, feat_vec = infer_engine.predict_from_nii(
            nii_path=tmp_nii_path,
            save_mask_path=mask_save_path,
            save_feature_path=feat_save_path
        )

        # 4. 组装返回数据
        feat_list = feat_vec.tolist() if feat_vec is not None else None

        return {
            "status": "success",
            "task_id": task_id,
            "mask_file": mask_save_path,
            "feature": feat_list
        }

    except Exception as e:
        return {
            "status": "fail",
            "msg": str(e)
        }