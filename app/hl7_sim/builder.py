# app/hl7_sim/builder.py
from datetime import datetime
from typing import List, Dict, Any
import uuid


def _ts(dt: datetime = None) -> str:
    """HL7 时间戳格式：YYYYMMDDHHMMSS"""
    dt = dt or datetime.now()
    return dt.strftime("%Y%m%d%H%M%S")


def build_msh(message_control_id: str, sending_app: str = "LAB_SIM") -> str:
    return (
        f"MSH|^~\\&|{sending_app}|HOSPITAL|LIS|HOSPITAL|{_ts()}||"
        f"ORU^R01|{message_control_id}|P|2.5"
    )


def build_pid(patient_id: str, patient_name: str) -> str:
    # PID 段：患者标识，姓名按 HL7 习惯用 ^ 分隔姓/名，这里简化处理中文姓名整体放一个域
    return f"PID|1||{patient_id}||{patient_name}||"


def build_orc(order_no: str) -> str:
    return f"ORC|RE|{order_no}||"


def build_single_test_oru(
    patient_id: str,
    patient_name: str,
    order_no: str,
    item_name: str,
    test_value: float,
    reference_range: str,
    unit: str = "",
    abnormal_flag: str = "N"
) -> str:
    """
    单次抽血结果 -> 一条 ORU^R01 消息，OBX 段只有一行结果。
    """
    msg_id = str(uuid.uuid4())[:20]
    obr = f"OBR|1|{order_no}||{item_name}"
    obx = (
        f"OBX|1|NM|{item_name}||{test_value}|{unit}|"
        f"{reference_range}|{abnormal_flag}|||F"
    )

    segments = [
        build_msh(msg_id),
        build_pid(patient_id, patient_name),
        build_orc(order_no),
        obr,
        obx,
    ]
    return "\r".join(segments)


def build_cgm_series_oru(
    patient_id: str,
    patient_name: str,
    order_no: str,
    item_name: str,
    series: List[Dict[str, Any]],   # [{"time": "2026-06-15T08:00:00", "value": 6.8}, ...]
    reference_range: str = "3.9-10.0",
    unit: str = "mmol/L"
) -> str:
    """
    3天动态血糖 -> 一条 ORU^R01 消息，OBX 段按时间顺序逐条列出，
    每条 OBX 的观测时间(OBX-14)单独标注，模拟 CGM 设备的批量回传。
    """
    msg_id = str(uuid.uuid4())[:20]
    obr = f"OBR|1|{order_no}||{item_name}"

    obx_lines = []
    for idx, point in enumerate(series, start=1):
        point_time = datetime.fromisoformat(point["time"])
        value = point["value"]
        abnormal_flag = "A" if point.get("abnormal") else "N"
        obx_lines.append(
            f"OBX|{idx}|NM|{item_name}||{value}|{unit}|"
            f"{reference_range}|{abnormal_flag}|||F|||{_ts(point_time)}"
        )

    segments = [
        build_msh(msg_id),
        build_pid(patient_id, patient_name),
        build_orc(order_no),
        obr,
        *obx_lines,
    ]
    return "\r".join(segments)