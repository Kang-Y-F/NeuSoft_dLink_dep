# app/api/tumor_endpoints.py
# ============================================================
# 肿瘤（肺结节）检测 API 路由
# 独立文件，和现有 lab_endpoints / hl7_endpoints / preview_endpoints
# 一样，在 CTDetectionServer.py 里用 app.include_router() 挂进去，
# 不改动任何伪影相关的现有代码。
# ============================================================

import os
import uuid
import shutil

import numpy as np
import SimpleITK as sitk
from fastapi import APIRouter, File, UploadFile, Query, HTTPException

from Detection.TumorDetectionInfer import TumorDetectionInfer, TumorDetectionError

router = APIRouter(prefix="/infer", tags=["肿瘤检测"])

# ── 和 CTDetectionServer.py 里保持一致的约定 ──────────────────
TMP_DIR = "./tmp"
SERVICE_BASE_URL = "https://u762954-924e-d2896d39.westc.seetacloud.com:8443"
# 换成你机器上lung_nodule_ct_detection bundle的实际路径
TUMOR_BUNDLE_ROOT = os.environ.get("TUMOR_BUNDLE_ROOT", "/root/autodl-tmp/lung_nodule_ct_detection")
# ⚠️ 关键：必须指向 monai_tumor 这个conda环境的python.exe，
# 不能用当前FastAPI主服务所在环境的解释器（主服务环境大概率没装monai）。
# 换成你机器上实际的路径，用 `conda env list` 或 `where python`（在monai_tumor环境里执行）查看。
TUMOR_PYTHON_EXE = "/root/miniconda3/envs/torch310/bin/python"
os.makedirs(TMP_DIR, exist_ok=True)

# ── 全局单例：bundle CLI每次调用本身较重，这里只是持有配置，不常驻显存 ──
_tumor_infer = None


def get_tumor_infer():
    global _tumor_infer
    if _tumor_infer is None:
        if not os.path.exists(TUMOR_PYTHON_EXE):
            raise HTTPException(
                status_code=500,
                detail=f"TUMOR_PYTHON_EXE 路径不存在：{TUMOR_PYTHON_EXE}，"
                       f"请在环境变量或本文件里改成你monai_tumor环境的python.exe实际路径",
            )
        _tumor_infer = TumorDetectionInfer(
            bundle_root=TUMOR_BUNDLE_ROOT,
            python_exe=TUMOR_PYTHON_EXE,
        )
    return _tumor_infer


@router.post("/tumor", summary="肺结节检测推理", tags=["肿瘤检测"])
async def infer_tumor(
    file: UploadFile = File(...),
    patient_id:      str   = Query(""),
    score_threshold: float = Query(0.5, ge=0.0, le=1.0, description="置信度过滤阈值"),
):
    """
    上传CT，返回：
      - ct_url:   原始CT下载地址（前端CtViewer直接加载）
      - mask_url: 检出结节的二值mask下载地址（前端CtViewer按红色区域叠加显示）
      - boxes / scores / labels: 原始检出框数据，前端如果要自己画3D框可以用这个
    """
    task_id = str(uuid.uuid4())
    temp_nii_path = os.path.join(TMP_DIR, f"{task_id}_{file.filename}")
    with open(temp_nii_path, "wb") as f:
        f.write(await file.read())

    safe_pid = patient_id.replace("/", "_") if patient_id else "unknown"
    ct_save_filename   = f"{safe_pid}_tumor_ct.nii.gz"
    mask_save_filename = f"{safe_pid}_tumor_mask.nii.gz"
    ct_save_path   = os.path.join(TMP_DIR, ct_save_filename)
    mask_save_path = os.path.join(TMP_DIR, mask_save_filename)

    try:
        shutil.copy2(temp_nii_path, ct_save_path)
        ct_url = f"{SERVICE_BASE_URL}/mask/{ct_save_filename}"

        infer_engine = get_tumor_infer()
        infer_engine.score_threshold = score_threshold
        sitk_mask, boxes, scores, labels = infer_engine.infer_from_nii(temp_nii_path)

        sitk.WriteImage(sitk_mask, mask_save_path)
        mask_url = f"{SERVICE_BASE_URL}/mask/{mask_save_filename}"

        nodule_pixel_count = int(np.sum(sitk.GetArrayFromImage(sitk_mask) > 0))

        return {
            "code":               200,
            "msg":                f"检测完成，共检出 {len(boxes)} 个疑似结节",
            "ct_url":             ct_url,
            "mask_url":           mask_url,
            "nodule_pixel_count": nodule_pixel_count,
            "nodule_count":       len(boxes),
            "boxes":              boxes,
            "scores":             scores,
            "labels":             labels,
        }

    except TumorDetectionError as e:
        # 这类是"预期内会发生"的推理链路错误（数据/配置问题），用它的原始信息直接返回，方便排查
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"肿瘤检测推理异常：{e}")
    finally:
        if os.path.exists(temp_nii_path):
            os.remove(temp_nii_path)