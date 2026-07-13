import io
import os
import shutil
import tempfile

import nibabel as nib
import numpy as np
import requests
from PIL import Image
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["预览图生成"])

TMP_DIR = "./tmp"
SERVICE_BASE_URL = "https://u762954-924e-d2896d39.westc.seetacloud.com:8443"


class PreviewRequest(BaseModel):
    ctUrl: str
    maskUrl: str
    checkReportId: int


def _url_to_local_path(url: str) -> str | None:
    if url.startswith(SERVICE_BASE_URL + "/mask/"):
        filename = url[len(SERVICE_BASE_URL) + len("/mask/"):]
        local_path = os.path.join(TMP_DIR, os.path.basename(filename))
        if os.path.exists(local_path):
            return local_path
    return None


def _download_to_tmp(url: str, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        shutil.copyfileobj(r.raw, tmp)
    tmp.close()
    return tmp.name


@router.post("/internal/ct/generate-preview", summary="生成CT预览缩略图")
async def generate_preview(req: PreviewRequest):
    import traceback

    ct_path = None
    mask_path = None
    need_cleanup_ct = False
    need_cleanup_mask = False

    try:
        print(f"[Preview] ctUrl={req.ctUrl}, maskUrl={req.maskUrl}, id={req.checkReportId}")

        ct_local = _url_to_local_path(req.ctUrl)
        mask_local = _url_to_local_path(req.maskUrl)

        if ct_local:
            ct_path = ct_local
            print(f"[Preview] CT使用本地文件: {ct_path}")
        else:
            ct_path = _download_to_tmp(req.ctUrl, ".nii.gz")
            need_cleanup_ct = True
            print(f"[Preview] CT从URL下载, 大小={os.path.getsize(ct_path)} bytes")

        if mask_local:
            mask_path = mask_local
            print(f"[Preview] Mask使用本地文件: {mask_path}")
        else:
            mask_path = _download_to_tmp(req.maskUrl, ".nii.gz")
            need_cleanup_mask = True
            print(f"[Preview] Mask从URL下载, 大小={os.path.getsize(mask_path)} bytes")

        ct_img = nib.as_closest_canonical(nib.load(ct_path))
        mask_img = nib.as_closest_canonical(nib.load(mask_path))

        ct_data = ct_img.get_fdata()
        mask_data = mask_img.get_fdata()
        print(f"[Preview] CT shape={ct_data.shape}, Mask shape={mask_data.shape}")

        if ct_data.shape != mask_data.shape:
            raise HTTPException(
                status_code=400,
                detail=f"CT与mask shape不一致: {ct_data.shape} vs {mask_data.shape}",
            )

        z = ct_data.shape[2] // 2
        ct_slice = ct_data[:, :, z]
        mask_slice = mask_data[:, :, z]

        ww, wl = 80, 40
        lo, hi = wl - ww / 2, wl + ww / 2
        ct_norm = np.clip((ct_slice - lo) / (hi - lo), 0, 1)
        ct_gray = (ct_norm * 255).astype(np.uint8)

        rgb = np.stack([ct_gray] * 3, axis=-1)
        alpha = 0.4
        mask_bool = mask_slice > 0
        overlay_color = np.array([255, 0, 0])
        rgb[mask_bool] = (rgb[mask_bool] * (1 - alpha) + overlay_color * alpha).astype(np.uint8)

        out = np.transpose(rgb, (1, 0, 2))[::-1]

        img = Image.fromarray(out)

        save_filename = f"preview_{req.checkReportId}.png"
        save_path = os.path.join(TMP_DIR, save_filename)
        img.save(save_path, format="PNG")
        print(f"[Preview] PNG已保存: {save_path}")

        return {
            "code": 200,
            "data": {
                "previewImageUrl": f"{SERVICE_BASE_URL}/mask/{save_filename}",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预览图生成失败：{type(e).__name__}: {e}")
    finally:
        if need_cleanup_ct and ct_path and os.path.exists(ct_path):
            os.unlink(ct_path)
        if need_cleanup_mask and mask_path and os.path.exists(mask_path):
            os.unlink(mask_path)
