# Detection/TumorDetectionInfer.py
# ============================================================
# 肿瘤（肺结节）检测推理封装
#   - 复用 MONAI Model Zoo 的 lung_nodule_ct_detection bundle
#   - 不重写 bundle 内部的 RetinaNet + anchor + NMS 逻辑，
#     而是每次推理时动态生成"只含当前上传文件"的 datalist，
#     通过子进程调用你已经手动跑通验证过的
#     `python -m monai.bundle run ...` 命令，稳，且和CLI手跑结果一致
#   - 输出的检出框(box)被转换成和CT同形状的二值mask，
#     这样前端可以直接复用 CtViewer.vue 现成的红色掩码叠加渲染，
#     不用新写3D框选可视化逻辑
#
# 使用前提（和你CLI手动验证的环境保持一致）：
#   1. bundle_root 下 configs/inference.json 里 checkpointloader
#      已经加了 "map_location": "@device"（你之前手动改过的那步）
#   2. conda环境里 torch 是能跑通bundle CLI的那个版本
# ============================================================

import os
import sys
import json
import glob
import shutil
import subprocess
import tempfile
import time

import numpy as np
import SimpleITK as sitk


class TumorDetectionError(Exception):
    """推理链路中出现的、需要暴露给API层的错误（区别于代码bug）"""
    pass


class TumorDetectionInfer:

    def __init__(
        self,
        bundle_root: str,
        python_exe: str = None,
        score_threshold: float = 0.5,
        timeout_sec: int = 1800,
    ):
        """
        :param bundle_root:     lung_nodule_ct_detection bundle 根目录
                                 例如 "E:/lung_nodule_ct_detection"
        :param python_exe:      指定python解释器路径（比如你conda环境里的python.exe），
                                 不传则用当前进程的 sys.executable
        :param score_threshold: 检出框置信度过滤阈值，低于此分数的框不计入结果
        :param timeout_sec:     单次CLI推理超时时间（CPU上大体积CT可能较慢，默认30分钟）
        """
        self.bundle_root = bundle_root
        self.python_exe = python_exe or sys.executable
        self.score_threshold = score_threshold
        self.timeout_sec = timeout_sec

        config_path = os.path.join(bundle_root, "configs", "inference.json")
        if not os.path.exists(config_path):
            raise TumorDetectionError(f"未找到bundle配置文件：{config_path}")
        self.config_path = config_path

    # ── 主接口：输入CT路径，返回 (mask sitk.Image, boxes, scores, labels) ──

    def infer_from_nii(self, nii_path: str) -> tuple:
        """
        :param nii_path: 单个 .nii/.nii.gz 文件路径（原始CT）
        :return: (sitk_mask, boxes, scores, labels)
                 sitk_mask: 与输入CT同形状/同spacing的二值mask（0/1），可直接喂给前端CtViewer
                 boxes:     list[[xmin,ymin,zmin,xmax,ymax,zmax]]，体素坐标系
                 scores:    list[float]
                 labels:    list[int]
        """
        sitk_ct = sitk.ReadImage(nii_path)

        with tempfile.TemporaryDirectory(prefix="tumor_infer_") as work_dir:
            dataset_dir = os.path.join(work_dir, "dataset")
            os.makedirs(dataset_dir, exist_ok=True)

            case_filename = "case_" + os.path.basename(nii_path)
            if not case_filename.endswith(".nii.gz") and not case_filename.endswith(".nii"):
                case_filename += ".nii.gz"
            case_path = os.path.join(dataset_dir, case_filename)
            shutil.copy2(nii_path, case_path)

            # 单条记录的datalist——彻底避开"整份LUNA16划分里其他病例文件缺失"的问题
            datalist_path = os.path.join(work_dir, "datalist.json")
            with open(datalist_path, "w", encoding="utf-8") as f:
                json.dump({"validation": [{"image": case_filename, "box": [], "label": []}]}, f)

            output_dir = os.path.join(work_dir, "eval")
            os.makedirs(output_dir, exist_ok=True)

            self._run_bundle_cli(dataset_dir, datalist_path, output_dir)

            boxes, scores, labels = self._parse_output(output_dir, case_filename)

        mask_arr = self._boxes_to_mask(boxes, sitk_ct)
        sitk_mask = sitk.GetImageFromArray(mask_arr)
        sitk_mask.CopyInformation(sitk_ct)

        return sitk_mask, boxes, scores, labels

    # ── 子进程调用bundle CLI ──────────────────────────────────

    def _run_bundle_cli(self, dataset_dir: str, datalist_path: str, output_dir: str):
        cmd = [
            self.python_exe, "-m", "monai.bundle", "run",
            "--config_file", self.config_path,
            "--bundle_root", self.bundle_root,
            "--dataset_dir", dataset_dir,
            "--data_list_file_path", datalist_path,
            "--output_dir", output_dir,
        ]
        print(f"📌 [TumorDetectionInfer] 执行命令: {' '.join(cmd)}")

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
            cwd=self.bundle_root,  # ⚠️ 必须设置：bundle的scripts/自定义模块（如RetinaNetInferer）
                                    # 靠cwd被加入sys.path才能被MONAI动态import找到，
                                    # 不设置的话，FastAPI进程从哪个目录启动，就会报
                                    # ModuleNotFoundError: Cannot locate class or function path: 'scripts.xxx'
        )

        if proc.returncode != 0:
            # 只截取尾部报错信息，避免把整个torch堆栈糊在API错误里
            tail = "\n".join(proc.stderr.strip().splitlines()[-40:])
            raise TumorDetectionError(f"bundle CLI 推理失败（returncode={proc.returncode}）:\n{tail}")

        print(proc.stdout[-2000:])

    # ── 解析bundle输出 ───────────────────────────────────────
    # ⚠️ 注意：lung_nodule_ct_detection bundle 用自带的
    # scripts/detection_saver.py 把结果写到 output_dir 下，
    # 具体文件名/字段名没有在官方文档里完整列出，这里做了兼容多种
    # 常见命名的容错解析。如果你实际跑出来的json结构和这里假设的不一致，
    # 把 output_dir 下生成的json内容贴给我，我按实际结构改这段解析逻辑。

    def _parse_output(self, output_dir: str, case_filename: str) -> tuple:
        json_files = glob.glob(os.path.join(output_dir, "**", "*.json"), recursive=True)
        if not json_files:
            raise TumorDetectionError(
                f"推理似乎已完成，但在输出目录 {output_dir} 下没有找到任何json结果文件，"
                f"请检查bundle的DetectionSaver配置（output_dir/output_ext等参数）"
            )

        # 取最新生成的json（一次只跑一个病例，理论上只有一个候选）
        json_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        result_path = json_files[0]
        print(f"📌 [TumorDetectionInfer] 解析结果文件: {result_path}")

        with open(result_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # 兼容: 顶层直接是list，或者是 {"validation":[...]}/{"box":[...]} 这种嵌套结构
        record = raw
        if isinstance(raw, dict):
            for key in ("validation", "results", "predictions"):
                if key in raw and isinstance(raw[key], list) and raw[key]:
                    record = raw[key][0]
                    break
        if isinstance(record, list) and record:
            record = record[0]

        if not isinstance(record, dict):
            raise TumorDetectionError(f"无法识别的结果json结构，请把 {result_path} 的内容贴给我")

        boxes = record.get("box") or record.get("boxes") or []
        scores = record.get("label_scores") or record.get("scores") or [1.0] * len(boxes)
        labels = record.get("label") or record.get("labels") or [0] * len(boxes)

        # 按置信度阈值过滤
        filtered = [
            (b, s, l) for b, s, l in zip(boxes, scores, labels)
            if s >= self.score_threshold
        ]
        if filtered:
            boxes, scores, labels = map(list, zip(*filtered))
        else:
            boxes, scores, labels = [], [], []

        print(f"✅ 检出 {len(boxes)} 个结节（置信度≥{self.score_threshold}）")
        return boxes, scores, labels

    # ── 检出框 -> 二值mask体积（供CtViewer复用红色叠加逻辑） ──

    @staticmethod
    def _boxes_to_mask(boxes: list, sitk_ct: sitk.Image) -> np.ndarray:
        size = sitk_ct.GetSize()  # (W, H, D) —— sitk是(x,y,z)顺序
        nx, ny, nz = size
        mask = np.zeros((nz, ny, nx), dtype=np.int16)  # numpy数组是(D,H,W)顺序

        for box in boxes:
            if len(box) != 6:
                continue
            xmin, ymin, zmin, xmax, ymax, zmax = box
            x0, x1 = sorted((int(round(xmin)), int(round(xmax))))
            y0, y1 = sorted((int(round(ymin)), int(round(ymax))))
            z0, z1 = sorted((int(round(zmin)), int(round(zmax))))
            x0, x1 = max(0, x0), min(nx, x1)
            y0, y1 = max(0, y0), min(ny, y1)
            z0, z1 = max(0, z0), min(nz, z1)
            if x1 > x0 and y1 > y0 and z1 > z0:
                mask[z0:z1, y0:y1, x0:x1] = 1

        return mask