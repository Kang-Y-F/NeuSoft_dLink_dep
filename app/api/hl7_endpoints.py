# app/api/hl7_endpoints.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from app.hl7_sim.builder import build_single_test_oru, build_cgm_series_oru
from app.hl7_sim.sender import (
    send_single_test_to_java,
    send_cgm_series_to_java,
    write_raw_hl7_to_log,
)
from datetime import datetime, timedelta
from app.lab_prediction.trend_predictor import predict_cgm_trend

router = APIRouter(prefix="/hl7-sim", tags=["hl7-simulation"])


class SingleTestRequest(BaseModel):
    patientId: str
    checkOrderId: str
    patientName: str
    orderNo: str
    itemName: str
    testValue: float
    referenceRange: str
    unit: str = ""


class CgmPoint(BaseModel):
    time: str       # ISO格式时间，如 2026-06-15T08:00:00
    value: float
    abnormal: bool = False


class CgmSeriesRequest(BaseModel):
    patientId: str
    checkOrderId: str
    patientName: str
    orderNo: str
    itemName: str = "动态血糖"
    referenceRange: str = "3.9-10.0"
    suiteGroup: Optional[str] = None
    series: List[CgmPoint]


@router.post("/single-test")
async def simulate_single_test(req: SingleTestRequest):
    """
    单次抽血 HL7 仿真：构造 ORU^R01 报文 -> 落盘留痕 -> 回写 Java 检验报告接口
    """
    abnormal_flag = "N"
    try:
        lo, hi = map(float, req.referenceRange.split("-"))
        if req.testValue < lo or req.testValue > hi:
            abnormal_flag = "A"
    except Exception:
        pass

    hl7_msg = build_single_test_oru(
        patient_id=req.patientId,
        patient_name=req.patientName,
        order_no=req.orderNo,
        item_name=req.itemName,
        test_value=req.testValue,
        reference_range=req.referenceRange,
        unit=req.unit,
        abnormal_flag=abnormal_flag
    )
    write_raw_hl7_to_log(hl7_msg)

    result = await send_single_test_to_java(
        patient_id=req.patientId,
        check_order_id=req.checkOrderId,
        item_name=req.itemName,
        test_value=req.testValue,
        reference_range=req.referenceRange
    )

    return {
        "code": 200 if result.success else 500,
        "message": result.detail,
        "data": {"hl7Message": hl7_msg, "javaResponse": result.response_data}
    }


@router.post("/cgm-series")
async def simulate_cgm_series(req: CgmSeriesRequest):
    """
    3天动态血糖 HL7 仿真：构造单条 ORU^R01（OBX 多行）-> 落盘留痕
    -> 逐点回写 Java 接口，方便后续趋势预测接口直接读到历史数据
    """
    series_dicts = [p.dict() for p in req.series]

    hl7_msg = build_cgm_series_oru(
        patient_id=req.patientId,
        patient_name=req.patientName,
        order_no=req.orderNo,
        item_name=req.itemName,
        series=series_dicts,
        reference_range=req.referenceRange
    )
    write_raw_hl7_to_log(hl7_msg)

    result = await send_cgm_series_to_java(
        patient_id=req.patientId,
        check_order_id=req.checkOrderId,
        item_name=req.itemName,
        series=series_dicts,
        reference_range=req.referenceRange,
        suite_group=req.suiteGroup
    )

    return {
        "code": 200 if result.success else 500,
        "message": result.detail,
        "data": {"hl7Message": hl7_msg, "pointCount": len(series_dicts)}
    }


@router.post("/cgm-preview")
async def preview_cgm_series(req: CgmSeriesRequest):
    series_dicts = [p.dict() for p in req.series]

    hl7_msg = build_cgm_series_oru(
        patient_id=req.patientId,
        patient_name=req.patientName,
        order_no=req.orderNo,
        item_name=req.itemName,
        series=series_dicts,
        reference_range=req.referenceRange
    )
    write_raw_hl7_to_log(hl7_msg)

    values = [p["value"] for p in series_dicts]
    predict_result = predict_cgm_trend(values, steps=6)   # 未来30分钟，5分钟一个点
    raw_predictions = predict_result.get("predictions", [])  # [{"step","value","isPrediction"}, ...]

    last_time = datetime.fromisoformat(series_dicts[-1]["time"])
    predictions = [
        {
            "time": (last_time + timedelta(minutes=5 * (i + 1))).isoformat(timespec="seconds"),
            "value": p["value"]
        }
        for i, p in enumerate(raw_predictions)
    ]

    return {
        "code": 200,
        "message": "预览生成成功",
        "data": {
            "series": series_dicts,
            "predictions": predictions,
            "trend": predict_result.get("trend")
        }
    }