from app.core.celery import celery
from app.db.mysql import update_task, save_patient_feature
import os
import uuid
import numpy as np
import SimpleITK as sitk

# 导入你现有的推理类（路径不变）
from Detection.CTArtifactInfer import CTArtifactInfer, ModelType

# 全局预加载模型（和你原代码一致）
MODEL_WEIGHTS = {
    ModelType.UNET2D:           "./Model/weights/nor_best.pth",
    ModelType.ATTENTION_UNET2D: "./Model/weights/atten_best.pth",
}
infer_engines = {
    ModelType.UNET2D: CTArtifactInfer(MODEL_WEIGHTS[ModelType.UNET2D], ModelType.UNET2D),
    ModelType.ATTENTION_UNET2D: CTArtifactInfer(MODEL_WEIGHTS[ModelType.ATTENTION_UNET2D], ModelType.ATTENTION_UNET2D),
}

@celery.task(bind=True)
def async_infer_task(self, task_id, patient_id, model_type, nii_path, save_mask=True, save_feature=True):
    """
    Celery 异步推理
    """
    try:
        self.update_state(state="PROGRESS", meta={"status": "开始推理"})

        # 推理
        engine = infer_engines[model_type]
        uid = str(uuid.uuid4())
        stem = os.path.basename(nii_path).replace(".nii.gz", "").replace(".nii", "")
        mask_path = f"./results/{uid}_{stem}_mask.nii.gz"
        feat_path = f"./saved_features/{uid}_{stem}_feature.npy"

        sitk_mask, feature_vector = engine.predict_from_nii(
            nii_path=nii_path,
            save_mask_path=mask_path if save_mask else None,
            save_feature_path=feat_path if save_feature else None
        )

        # 保存患者特征到 MySQL
        if patient_id and feature_vector is not None:
            save_patient_feature(patient_id, feature_vector)

        # 更新任务成功
        update_task(
            task_id=task_id,
            status="success",
            mask_path=mask_path if save_mask else None,
            feature_path=feat_path if save_feature else None
        )
        return {"status": "success", "mask_path": mask_path, "feature_path": feat_path}

    except Exception as e:
        update_task(task_id=task_id, status="failed", error=str(e))
        return {"status": "failed", "error": str(e)}