# Detection/CTArtifactInfer.py
# ============================================================
# 融合版推理类：
#   - 双模型支持（UNet2D / AttentionUNet2D，由 ModelType 枚举控制）
#   - 逐切片推理 + 全卷平均特征聚合
#   - ModelType 统一从 Detection.model_enum 导入
# ============================================================

import os
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm

from Conf.Config import DEVICE
from Model.UNet2D import UNet2D
from Model.AttentionUNet2D import UNet2D as AttentionUNet2D
from Detection.model_enum import ModelType   # ← 统一从枚举模块导入


class CTArtifactInfer:

    def __init__(
        self,
        model_weight_path: str,
        model_type: str = ModelType.UNET2D,
        device: str = None,
    ):
        """
        :param model_weight_path: 权重文件路径
        :param model_type:        ModelType.UNET2D 或 ModelType.ATTENTION_UNET2D
        :param device:            推理设备，默认读取 Config.DEVICE
        """
        self.device            = device or DEVICE
        self.model_weight_path = model_weight_path
        self.model_type        = model_type.lower() if isinstance(model_type, str) else model_type.value
        self.model             = self._load_model()
        self.slice_features: list[np.ndarray] = []

    # ── 模型加载 ──────────────────────────────────────────────────

    def _load_model(self):
        if self.model_type == ModelType.UNET2D:
            model = UNet2D().to(self.device)
            weight_path = "./Model/weights/nor_best.pth"
            print(f"📌 加载 UNet2D 权重: {weight_path}")
        elif self.model_type == ModelType.ATTENTION_UNET2D:
            model = AttentionUNet2D().to(self.device)
            weight_path = "./Model/weights/atten_best.pth"
            print(f"📌 加载 AttentionUNet2D 权重: {weight_path}")
        else:
            raise ValueError(f"不支持的模型类型：{self.model_type}")

        model.load_state_dict(
            torch.load(weight_path, map_location=self.device)
        )
        model.eval()
        return model

    # ── 单切片推理 ────────────────────────────────────────────────

    def predict_slice(
        self,
        img_slice: np.ndarray,
        extract_feature: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        推理单张 2D 切片。

        :param img_slice:       形状 (H, W) 的 CT 切片
        :param extract_feature: 是否同时提取融合特征向量
        :return: (pred_mask, feature_vector)
                 pred_mask      shape (H, W)，int16，0/1 二值
                 feature_vector shape (1, 960)，float32；extract_feature=False 时为 None
        """
        img_slice = img_slice.astype(np.float32)
        mean = img_slice.mean()
        std  = img_slice.std()
        img_slice = (img_slice - mean) / (std + 1e-7)

        tensor = (
            torch.from_numpy(img_slice)
            .unsqueeze(0).unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            output = self.model(tensor)
            pred_mask = (
                torch.sigmoid(output).squeeze().cpu().numpy() > 0.5
            ).astype(np.int16)

            feature_vector = None
            if extract_feature:
                feature_vector = (
                    self.model.extract_features().detach().cpu().numpy()
                )

        return pred_mask, feature_vector

    # ── 从 NIfTI 文件推理（主接口）───────────────────────────────

    def predict_from_nii(
        self,
        nii_path: str,
        save_mask_path: str = None,
        save_feature_path: str = None,
    ) -> tuple[sitk.Image, np.ndarray | None]:
        """
        输入 .nii / .nii.gz 路径，返回分割掩码 + 全卷平均特征向量。

        :param nii_path:          输入 CT 文件路径
        :param save_mask_path:    掩码保存路径（可选）
        :param save_feature_path: 特征向量保存路径 .npy（可选）
        :return: (sitk_mask, feature_vector)
        """
        sitk_ct = sitk.ReadImage(nii_path)
        ct_vol  = sitk.GetArrayFromImage(sitk_ct)
        D, H, W = ct_vol.shape

        mask_vol = np.zeros((D, H, W), dtype=np.int16)
        self.slice_features = []

        for z in tqdm(range(D), desc="推理切片"):
            mask_slice, feat = self.predict_slice(ct_vol[z], extract_feature=True)
            mask_vol[z] = mask_slice
            if feat is not None:
                self.slice_features.append(feat)

        sitk_mask = sitk.GetImageFromArray(mask_vol)
        sitk_mask.CopyInformation(sitk_ct)
        if save_mask_path:
            os.makedirs(os.path.dirname(save_mask_path) or ".", exist_ok=True)
            sitk.WriteImage(sitk_mask, save_mask_path)

        feature_vector = self._aggregate_and_save_features(save_feature_path)
        return sitk_mask, feature_vector

    # ── 从 SimpleITK 对象推理 ────────────────────────────────────

    def predict_from_sitk(
        self,
        sitk_ct: sitk.Image,
        save_mask_path: str = None,
        save_feature_path: str = None,
    ) -> tuple[sitk.Image, np.ndarray | None]:
        ct_vol  = sitk.GetArrayFromImage(sitk_ct)
        D, H, W = ct_vol.shape

        mask_vol = np.zeros((D, H, W), dtype=np.int16)
        self.slice_features = []

        for z in tqdm(range(D), desc="推理切片"):
            mask_slice, feat = self.predict_slice(ct_vol[z], extract_feature=True)
            mask_vol[z] = mask_slice
            if feat is not None:
                self.slice_features.append(feat)

        sitk_mask = sitk.GetImageFromArray(mask_vol)
        sitk_mask.CopyInformation(sitk_ct)
        if save_mask_path:
            os.makedirs(os.path.dirname(save_mask_path) or ".", exist_ok=True)
            sitk.WriteImage(sitk_mask, save_mask_path)

        feature_vector = self._aggregate_and_save_features(save_feature_path)
        return sitk_mask, feature_vector

    # ── 仅提取整卷特征 ───────────────────────────────────────────

    def extract_volume_features(
        self,
        nii_path: str,
        save_feature_path: str = None,
        return_average: bool = True,
    ) -> np.ndarray:
        sitk_ct = sitk.ReadImage(nii_path)
        ct_vol  = sitk.GetArrayFromImage(sitk_ct)
        D, *_   = ct_vol.shape

        self.slice_features = []
        for z in tqdm(range(D), desc="提取切片特征"):
            _, feat = self.predict_slice(ct_vol[z], extract_feature=True)
            if feat is not None:
                self.slice_features.append(feat)

        feat_array = np.concatenate(self.slice_features, axis=0)
        result = np.mean(feat_array, axis=0, keepdims=True) if return_average else feat_array

        if save_feature_path:
            os.makedirs(os.path.dirname(save_feature_path) or ".", exist_ok=True)
            np.save(save_feature_path, result)
            print(f"✅ 特征已保存：{save_feature_path}  shape={result.shape}")

        return result

    # ── 内部工具 ──────────────────────────────────────────────────

    def _aggregate_and_save_features(
        self, save_path: str = None
    ) -> np.ndarray | None:
        if not self.slice_features:
            return None

        feat_array     = np.concatenate(self.slice_features, axis=0)
        feature_vector = np.mean(feat_array, axis=0, keepdims=True)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            np.save(save_path, feature_vector)
            print(f"✅ 全卷平均特征已保存：{save_path}  shape={feature_vector.shape}")

        return feature_vector
