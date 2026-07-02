# app/lab_prediction/cgm_predictor.py
import numpy as np
from typing import List, Dict, Any
from datetime import datetime


def parse_cgm_series(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    CGM 数据预处理：按时间排序、计算采样间隔、识别异常波动段。
    history 每项形如 {"time": "2026-06-15T08:00:00", "value": 6.8, "abnormal": False}
    """
    sorted_hist = sorted(history, key=lambda h: h["time"])
    values = [h["value"] for h in sorted_hist]
    times = [datetime.fromisoformat(h["time"]) for h in sorted_hist]

    if len(times) >= 2:
        gaps_min = [(times[i + 1] - times[i]).total_seconds() / 60 for i in range(len(times) - 1)]
        avg_gap = float(np.mean(gaps_min))
    else:
        avg_gap = 0.0

    return {"values": values, "times": times, "avg_gap_minutes": avg_gap}


def detect_glucose_events(values: List[float], low_thresh: float = 3.9, high_thresh: float = 10.0) -> Dict[str, Any]:
    """
    识别低血糖 / 高血糖事件区间（TIR 相关统计的简化版）。
    """
    if not values:
        return {"low_count": 0, "high_count": 0, "time_in_range_pct": 0.0}

    arr = np.array(values)
    low_count = int(np.sum(arr < low_thresh))
    high_count = int(np.sum(arr > high_thresh))
    in_range = int(np.sum((arr >= low_thresh) & (arr <= high_thresh)))
    tir_pct = round(in_range / len(arr) * 100, 1)

    return {
        "low_count": low_count,
        "high_count": high_count,
        "time_in_range_pct": tir_pct
    }


def predict_cgm_trend(values: List[float], steps: int = 3) -> Dict[str, Any]:
    """
    CGM 短期外推：取最近一段窗口做局部线性拟合，
    比对整段做回归更贴近血糖短期波动特性。
    后续可在此函数内部替换为 ARIMA / Holt-Winters / 小型 LSTM，
    对外接口签名保持不变即可，不影响调用方。
    """
    n = len(values)
    if n < 4:
        # 数据点太少，退化为简单差值外推
        if n < 2:
            return {"predictions": [], "trend": "平稳"}
        last, prev = values[-1], values[-2]
        delta = last - prev
        predictions = [
            {"step": f"预测{i}", "value": round(last + delta * i, 2), "isPrediction": True}
            for i in range(1, steps + 1)
        ]
        trend = "上升" if delta > 0.3 else "下降" if delta < -0.3 else "平稳"
        return {"predictions": predictions, "trend": trend}

    window = min(8, n)  # CGM 通常 5~15 分钟一个点，取最近 8 个点（约1~2小时）算局部趋势
    recent = np.array(values[-window:], dtype=float)
    x = np.arange(window)
    slope, intercept = np.polyfit(x, recent, 1)

    predictions = []
    for i in range(1, steps + 1):
        pred_val = recent[-1] + slope * i
        predictions.append({
            "step": f"预测{i}",
            "value": round(float(pred_val), 2),
            "isPrediction": True
        })

    if slope > 0.3:
        trend = "上升"
    elif slope < -0.3:
        trend = "下降"
    else:
        trend = "平稳"

    return {"predictions": predictions, "trend": trend}


def analyze_cgm(history: List[Dict[str, Any]], steps: int = 3) -> Dict[str, Any]:
    """
    对外统一入口：解析 + 事件检测 + 预测，一次性返回给 lab_endpoints 用。
    """
    parsed = parse_cgm_series(history)
    events = detect_glucose_events(parsed["values"])
    pred = predict_cgm_trend(parsed["values"], steps=steps)

    return {
        "avg_sample_interval_minutes": round(parsed["avg_gap_minutes"], 1),
        "events": events,
        **pred
    }