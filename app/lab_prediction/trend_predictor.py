# app/lab_prediction/trend_predictor.py
import numpy as np
from typing import List, Dict, Optional

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
    steps: int = 3
) -> Dict:
    """
    更稳的版本：用全部历史点做最小二乘线性回归，
    而不是只看最后两个点（Java 原逻辑只看最后两个点，
    数据有波动时很不稳定）。这是建议优先换上的版本。
    """
    n = len(values)
    if n < 2:
        return {"predictions": [], "trend": "平稳"}

    x = np.arange(n)
    y = np.array(values, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)  # 一次线性拟合

    predictions = []
    for i in range(1, steps + 1):
        pred_x = n - 1 + i
        pred_val = slope * pred_x + intercept
        predictions.append({
            "step": f"预测{i}",
            "value": round(float(pred_val), 2),
            "isPrediction": True
        })

    if slope > 0.05:
        trend = "上升"
    elif slope < -0.05:
        trend = "下降"
    else:
        trend = "平稳"

    return {"predictions": predictions, "trend": trend}


def predict_cgm_trend(
    values: List[float],
    steps: int = 3
) -> Dict:
    """
    CGM（动态血糖）专用：数据点多、间隔短，
    用最近一段的移动平均斜率做短期外推，
    比单纯线性回归更贴近血糖短期波动的特性。
    后续可以在这里换成 ARIMA / Holt-Winters / 小型 LSTM。
    """
    n = len(values)
    if n < 4:
        return predict_linear_regression(values, steps)

    window = min(6, n)  # 取最近 6 个点算局部趋势，CGM 通常 5~15 分钟一个点
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

    if slope > 0.5:
        trend = "上升"
    elif slope < -0.5:
        trend = "下降"
    else:
        trend = "平稳"

    return {"predictions": predictions, "trend": trend}