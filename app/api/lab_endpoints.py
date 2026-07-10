# app/api/lab_endpoints.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.lab_prediction.trend_predictor import (
    predict_linear_regression,
    predict_cgm_trend,
)

router = APIRouter(prefix="/predict", tags=["lab-prediction"])


class HistoryPoint(BaseModel):
    date: str
    time: str
    value: float
    abnormal: bool = False
    referenceRange: Optional[str] = None


class TrendPredictRequest(BaseModel):
    history: List[HistoryPoint]
    referenceRange: Optional[str] = None
    indicator: Optional[str] = None
    steps: int = 3
    granularity: str = "auto"   # "auto" | "labtest" | "cgm"


def _is_cgm_like(history: List[HistoryPoint]) -> bool:
    """根据采样间隔粗判断是否是 CGM 连续血糖数据"""
    if len(history) < 4:
        return False
    try:
        from datetime import datetime
        t0 = datetime.fromisoformat(history[0].time)
        t1 = datetime.fromisoformat(history[1].time)
        gap_minutes = abs((t1 - t0).total_seconds()) / 60
        return gap_minutes <= 30  # 间隔30分钟以内认为是CGM
    except Exception:
        return False


def _calc_avg_gap(history: List[HistoryPoint]) -> float:
    if len(history) < 2:
        return 0.0
    try:
        times = [datetime.fromisoformat(h.time) for h in history]
        gaps = [(times[i + 1] - times[i]).total_seconds() / 60 for i in range(len(times) - 1)]
        return float(sum(gaps) / len(gaps))
    except Exception:
        return 0.0


@router.post("/trend")
def predict_trend(req: TrendPredictRequest) -> Dict[str, Any]:
    values = [p.value for p in req.history]

    use_cgm = (
        req.granularity == "cgm"
        or (req.granularity == "auto" and _is_cgm_like(req.history))
    )

    last_time = req.history[-1].time if req.history else None
    avg_gap = _calc_avg_gap(req.history)

    if use_cgm:
        result = predict_cgm_trend(values, steps=req.steps, last_time=last_time, avg_gap_minutes=avg_gap)
    else:
        result = predict_linear_regression(values, steps=req.steps, last_time=last_time, avg_gap_minutes=avg_gap)

    return {
        "code": 200,
        "message": "ok",
        "data": result
    }