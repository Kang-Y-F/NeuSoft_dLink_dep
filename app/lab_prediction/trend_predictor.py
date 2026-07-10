# app/lab_prediction/trend_predictor.py
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
def predict_linear_trend(
    values: List[float],
    steps: int = 3
) -> Dict:
    """
    最基础版本：和 Java 原逻辑等价的简单线性外推，
    先保证接口契约打通，后续替换成 ARIMA/指数平滑/LSTM 时
    只需要改这个函数内部实现，不影响调用方。
    """
    if len(values) < 2:
        return {"predictions": [], "trend": "平稳"}

    last = values[-1]
    prev = values[-2]
    delta = last - prev

    predictions = []
    for i in range(1, steps + 1):
        predictions.append({
            "step": f"预测{i}",
            "value": round(last + delta * i, 2),
            "isPrediction": True
        })

    if delta > 0.05:
        trend = "上升"
    elif delta < -0.05:
        trend = "下降"
    else:
        trend = "平稳"

    return {"predictions": predictions, "trend": trend}


def predict_linear_regression(
    values: List[float],
    steps: int = 3,
    last_time: Optional[str] = None,
    avg_gap_minutes: float = 0
) -> Dict:
    n = len(values)
    if n < 2:
        return {"predictions": [], "trend": "平稳"}

    x = np.arange(n)
    y = np.array(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    predictions = []
    for i in range(1, steps + 1):
        pred_x = n - 1 + i
        pred_val = slope * pred_x + intercept
        item: Dict = {
            "value": round(float(pred_val), 2),
            "isPrediction": True
        }
        if last_time and avg_gap_minutes > 0:
            dt = datetime.fromisoformat(last_time) + timedelta(minutes=avg_gap_minutes * i)
            item["date"] = dt.strftime("%Y-%m-%d")
            item["time"] = dt.isoformat(timespec="seconds")
        predictions.append(item)

    if slope > 0.05:
        trend = "上升"
    elif slope < -0.05:
        trend = "下降"
    else:
        trend = "平稳"

    return {"predictions": predictions, "trend": trend}


def predict_cgm_trend(
    values: List[float],
    steps: int = 3,
    last_time: Optional[str] = None,
    avg_gap_minutes: float = 0
) -> Dict:
    n = len(values)
    if n < 4:
        return predict_linear_regression(values, steps, last_time, avg_gap_minutes)

    window = min(6, n)
    recent = np.array(values[-window:], dtype=float)
    x = np.arange(window)
    slope, intercept = np.polyfit(x, recent, 1)

    predictions = []
    for i in range(1, steps + 1):
        pred_val = recent[-1] + slope * i
        item: Dict = {
            "value": round(float(pred_val), 2),
            "isPrediction": True
        }
        if last_time and avg_gap_minutes > 0:
            dt = datetime.fromisoformat(last_time) + timedelta(minutes=avg_gap_minutes * i)
            item["date"] = dt.strftime("%Y-%m-%d")
            item["time"] = dt.isoformat(timespec="seconds")
        predictions.append(item)

    if slope > 0.5:
        trend = "上升"
    elif slope < -0.5:
        trend = "下降"
    else:
        trend = "平稳"

    return {"predictions": predictions, "trend": trend}