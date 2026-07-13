# ============================================================
# Detection/ArtifactRemovalInfer.py
# 金属伪影去除推理模块（InDuDoNet+ 桥接封装）
# 与现有 CTArtifactInfer（分割模型）配合使用：
#   1. CTArtifactInfer 输出金属mask
#   2. 本模块用mask + 原始CT，完成正弦图域计算，调用InDuDoNet+去伪影
# ============================================================

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
import scipy.ndimage
from sklearn.cluster import k_means
from PIL import Image
import PIL
import cv2

from network.indudonet_plus import InDuDoNet_plus
from deeplesion.build_gemotry import initialization, build_gemotry
from odl.contrib import torch as odl_torch


# ---------- 全局：投影几何 + 算子（模块加载时初始化一次，避免重复构建） ----------
_param = initialization()
_ray_trafo = build_gemotry(_param)
_op_modfp = odl_torch.OperatorModule(_ray_trafo)          # 正向投影：图像域→正弦图域
_op_modpT = odl_torch.OperatorModule(_ray_trafo.adjoint)  # 伴随算子：正弦图域→图像域（近似重建，非精确FBP）

_MIU_AIR = 0
_MIU_WATER = 0.192
_STARPOINT = np.zeros([3, 1])
_STARPOINT[0] = _MIU_AIR
_STARPOINT[1] = _MIU_WATER
_STARPOINT[2] = 2 * _MIU_WATER

# nmar先验用的高斯滤波核，路径需要和你项目里 deeplesion/gaussianfilter.mat 保持一致
_SM_FILTER = sio.loadmat('deeplesion/gaussianfilter.mat')['smFilter']


class ArtifactSeverity:
    """简单的伪影严重程度分级，用于报告展示"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax, add_batch=True):
    """与官方test_deeplesion.py保持一致的归一化+维度整理"""
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    if add_batch:
        data = np.expand_dims(data, 0)
    return data


def _nmarprior(im, threshWater, threshBone, miuAir, miuWater, smFilter):
    imSm = scipy.ndimage.convolve(im, smFilter, mode='nearest')
    priorimgHU = imSm
    priorimgHU[imSm <= threshWater] = miuAir
    h, w = imSm.shape[0], imSm.shape[1]
    priorimgHUvector = np.reshape(priorimgHU, h * w)
    region1_1d = np.where(priorimgHUvector > threshWater)
    region2_1d = np.where(priorimgHUvector < threshBone)
    region_1d = np.intersect1d(region1_1d, region2_1d)
    priorimgHUvector[region_1d] = miuWater
    return np.reshape(priorimgHUvector, (h, w))


def nmar_prior(XLI, M):
    """直接复用官方Dataset.py里的nmar先验计算逻辑"""
    XLI = XLI.copy()
    XLI[M == 1] = _MIU_WATER
    h, w = XLI.shape[0], XLI.shape[1]
    im1d = XLI.reshape(h * w, 1)
    _, labels, _ = k_means(im1d, n_clusters=3, init=_STARPOINT, max_iter=300)
    threshBone2 = np.min(im1d[labels == 2])
    threshBone2 = np.max([threshBone2, 1.2 * _MIU_WATER])
    threshWater2 = np.min(im1d[labels == 1])
    return _nmarprior(XLI, threshWater2, threshBone2, _MIU_AIR, _MIU_WATER, _SM_FILTER)


def _linear_interp_sinogram(sinogram: np.ndarray, trace_mask: np.ndarray) -> np.ndarray:
    """
    标准线性插值MAR（Kalender方法）:
    对正弦图(角度 x 探测器)每一行，沿探测器方向，
    把被trace_mask覆盖（金属投影轨迹）的像素用两端邻近值线性插值填补。

    ⚠️ 这是通用公开算法实现，不保证与InDuDoNet+官方训练数据预处理时
    用的具体插值算法完全一致，实际效果需要你自行验证。

    sinogram:   (n_angles, n_detectors)
    trace_mask: 同形状，1表示需要插值填补的金属轨迹区域
    """
    sino_li = sinogram.copy()
    n_angles, n_det = sinogram.shape
    x_full = np.arange(n_det)

    for a in range(n_angles):
        row_mask = trace_mask[a] > 0.5
        if not row_mask.any():
            continue
        valid = ~row_mask
        if valid.sum() < 2:
            # 整行几乎全被遮挡，用邻近行代替（极端兜底）
            continue
        sino_li[a, row_mask] = np.interp(
            x_full[row_mask], x_full[valid], sinogram[a, valid]
        )
    return sino_li


class ArtifactRemovalInfer:
    """
    金属伪影去除推理类，接口风格与现有 CTArtifactInfer 保持一致。

    使用方式：
        remover = ArtifactRemovalInfer(model_dir="Model/weights/InDuDoNet+_latest.pt")
        clean_slice = remover.remove_artifact(ct_slice, mask_slice)
    """

    def __init__(self, model_dir: str, device: str = None, opt=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if opt is None:
            # 官方默认超参数，和 test_deeplesion.py 里argparse默认值保持一致
            class Opt:
                num_channel = 32
                T = 4
                S = 10
                eta1 = 1
                eta2 = 5
                alpha = 0.5
            opt = Opt()
        self.opt = opt

        self.net = InDuDoNet_plus(opt).to(self.device)
        self.net.load_state_dict(torch.load(model_dir, map_location=self.device))
        self.net.eval()
        print(f"✅ InDuDoNet+ 权重加载完成: {model_dir}  device={self.device}")


    # ---------------- 核心：单切片去伪影 ----------------

    def remove_artifact(
            self,
            ct_slice: np.ndarray,
            mask_slice: np.ndarray,
            blend_radius: int = 25,
    ) -> np.ndarray:
        # ── 修正：用标准CT物理公式，把HU值转换成线衰减系数μ（网络真正训练时用的物理量）──
        def hu_to_mu(hu_data, mu_water=_MIU_WATER):
            """
            HU -> 线衰减系数μ（标准CT物理转换公式）
            μ_water=0.192 是这套网络训练时约定的水衰减系数
            """
            mu = mu_water * (hu_data / 1000.0 + 1.0)
            mu = np.clip(mu, 0.0, None)  # 空气以下的padding值(比如-3024)转换后会是负数，裁剪到0
            return mu.astype(np.float32)

        def mu_to_hu(mu_data, mu_water=_MIU_WATER):
            """线衰减系数μ -> HU（上面公式的逆运算）"""
            hu = (mu_data / mu_water - 1.0) * 1000.0
            return hu.astype(np.float32)
        """
        对单张CT切片去除金属伪影。

        :param ct_slice:   原始CT图像切片 (H, W)，float
        :param mask_slice: 金属mask，(H, W)，0/1二值图
        :return: 去伪影后的CT图像，尺寸与输入一致 (H, W)
        """
        assert ct_slice.shape == mask_slice.shape, "CT图像和mask尺寸必须一致"
        orig_H, orig_W = ct_slice.shape

        # ── 关键新增：先把HU值转换到网络期望的[0,1]范围 ──
        ct_slice_mu = hu_to_mu(ct_slice)

        NET_SIZE = 416
        need_resize = (orig_H != NET_SIZE or orig_W != NET_SIZE)

        if need_resize:
            Xma_raw = cv2.resize(ct_slice_mu, (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_LINEAR)
            M = cv2.resize(mask_slice.astype(np.float32), (NET_SIZE, NET_SIZE), interpolation=cv2.INTER_NEAREST)
            M = (M > 0.5).astype(np.float32)
        else:
            Xma_raw = ct_slice_mu
            M = (mask_slice > 0.5).astype(np.float32)


        # 1) 正向投影，得到带伪影正弦图 Sma_raw
        Xma_t = torch.from_numpy(Xma_raw).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            Sma_raw = _op_modfp(Xma_t).squeeze().cpu().numpy()

        # 2) mask同样正向投影，得到金属在投影域的轨迹，二值化 -> Tr
        M_t = torch.from_numpy(M).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            trace_raw = _op_modfp(M_t).squeeze().cpu().numpy()
        trace_binary = (trace_raw > (trace_raw.max() * 0.05)).astype(np.float32)

        # 3) 正弦图域线性插值，得到 SLI_raw
        SLI_raw = _linear_interp_sinogram(Sma_raw, trace_binary)

        # 4) SLI反投影回图像域，得到 XLI_raw
        SLI_t = torch.from_numpy(SLI_raw).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            XLI_raw = _op_modpT(SLI_t).squeeze().cpu().numpy()
        if XLI_raw.max() > XLI_raw.min():
            XLI_raw = (XLI_raw - XLI_raw.min()) / (XLI_raw.max() - XLI_raw.min())
            XLI_raw = XLI_raw * (Xma_raw.max() - Xma_raw.min()) + Xma_raw.min()

        # 5) nmar先验图
        Xprior_raw = nmar_prior(XLI_raw.copy(), M)

        # 6) 组装网络输入
        Xma_in = normalize(Xma_raw, image_get_minmax())
        XLI_in = normalize(XLI_raw, image_get_minmax())
        Sma_in = normalize(Sma_raw, proj_get_minmax())
        SLI_in = normalize(SLI_raw, proj_get_minmax())
        Xprior_in = normalize(Xprior_raw, image_get_minmax())

        Tr_in = 1 - trace_binary.astype(np.float32)
        Tr_in = np.expand_dims(np.transpose(np.expand_dims(Tr_in, 2), (2, 0, 1)), 0)

        Xma_t2 = torch.from_numpy(Xma_in).to(self.device)
        XLI_t2 = torch.from_numpy(XLI_in).to(self.device)
        Sma_t2 = torch.from_numpy(Sma_in).to(self.device)
        SLI_t2 = torch.from_numpy(SLI_in).to(self.device)
        Tr_t2 = torch.from_numpy(Tr_in).to(self.device)
        Xprior_t2 = torch.from_numpy(Xprior_in).to(self.device)

        # 7) 网络推理
        with torch.no_grad():
            ListX, ListS, ListYS = self.net(Xma_t2, XLI_t2, Sma_t2, SLI_t2, Tr_t2, Xprior_t2)

        Xout = ListX[-1].squeeze().cpu().numpy()

        mu_out = Xout / 255.0
        Xout = mu_to_hu(mu_out)

        if need_resize:
            Xout = cv2.resize(Xout, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)

        # ── 关键新增：只在金属附近区域采用处理结果，其余区域保留原图 ──
        # 用形态学膨胀，把mask区域向外扩一圈，作为"需要处理"的范围
        mask_orig_size = mask_slice.astype(np.uint8)
        kernel = np.ones((blend_radius, blend_radius), np.uint8)
        blend_region = cv2.dilate(mask_orig_size, kernel, iterations=1).astype(np.float32)

        # 边缘做一点羽化，避免"处理区/原图区"交界处出现明显生硬的边界线
        blend_region = cv2.GaussianBlur(blend_region, (15, 15), 0)
        blend_region = np.clip(blend_region, 0, 1)

        # 加权融合：金属附近用Xout，远处用原图ct_slice
        final_out = Xout * blend_region + ct_slice.astype(np.float32) * (1 - blend_region)

        debug_dir = '/root/autodl-tmp/temp_debug'
        os.makedirs(debug_dir, exist_ok=True)
        plt.imsave(f'{debug_dir}/debug_mask.png', mask_slice, cmap='gray')
        plt.imsave(f'{debug_dir}/debug_ct.png', np.clip(ct_slice, -200, 1000), cmap='gray')
        plt.imsave(f'{debug_dir}/debug_blend_region.png', blend_region, cmap='gray')
        plt.imsave(f'{debug_dir}/debug_xout.png', Xout, cmap='gray')
        print(f"mask覆盖比例: {mask_slice.sum() / mask_slice.size * 100:.1f}%")

        return final_out

    # ---------------- 整卷CT批量去伪影 ----------------

    def remove_artifact_volume(
        self,
        ct_volume: np.ndarray,
        mask_volume: np.ndarray,
    ) -> np.ndarray:
        """
        对整卷CT（多切片）逐层去伪影。

        :param ct_volume:   (D, H, W)
        :param mask_volume: (D, H, W)，来自你现有分割模型对整卷的推理结果
        :return: 去伪影后的整卷 (D, H, W)
        """
        assert ct_volume.shape == mask_volume.shape
        D = ct_volume.shape[0]
        out = np.zeros_like(ct_volume, dtype=np.float32)
        for z in range(D):
            m = mask_volume[z]
            if m.sum() == 0:
                # 该切片无金属伪影，直接保留原图，不需要跑重建流程
                out[z] = ct_volume[z]
                continue
            out[z] = self.remove_artifact(ct_volume[z], m)
        return out