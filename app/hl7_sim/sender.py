# app/hl7_sim/sender.py
import httpx
from typing import List, Dict, Any, Optional
from app.core.config import settings  # 复用你现有的 config.py，里面应有 JAVA_LAB_API_BASE_URL 之类配置
# sender.py 顶部导入
import os



class HL7SendResult:
    def __init__(self, success: bool, detail: str = "", response_data: Any = None):
        self.success = success
        self.detail = detail
        self.response_data = response_data


async def send_single_test_to_java(
    patient_id: str,
    check_order_id: str,
    item_name: str,
    test_value: float,
    reference_range: str,
    description: str = "（HL7仿真数据）"
) -> HL7SendResult:
    """
    单次抽血：解析后直接调用 Java 现有的 /lab-report/create 接口写入，
    复用 Java 那边已有的入库、异常判定逻辑，避免两边各写一套校验规则。
    """
    url = f"{settings.JAVA_LAB_API_BASE_URL}/lab-report/create"
    payload = {
        "checkOrderId": check_order_id,
        "itemName": item_name,
        "testValue": str(test_value),
        "referenceRange": reference_range,
        "description": description
    }
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            resp = await client.post(url, json=payload)
            print(f"[DEBUG] 请求地址={url}")
            print(f"[DEBUG] 状态码={resp.status_code}")
            print(f"[DEBUG] 原始响应内容={resp.text!r}")
            data = resp.json()
            if resp.status_code == 200 and data.get("code") == 200:
                return HL7SendResult(True, "写入成功", data.get("data"))
            return HL7SendResult(False, data.get("message", "写入失败"))
    except Exception as e:
        return HL7SendResult(False, f"调用Java接口异常: {e}")


# app/hl7_sim/sender.py 替换 send_cgm_series_to_java 函数

async def send_cgm_series_to_java(
    patient_id: str,
    check_order_id: str,
    item_name: str,
    series: List[Dict[str, Any]],
    reference_range: str = "3.9-10.0",
    suite_group: Optional[str] = None
) -> HL7SendResult:
    url = f"{settings.JAVA_LAB_API_BASE_URL}/lab-report/batch-create"
    items = [
        {
            "orderId": int(check_order_id),
            "patientId": int(patient_id),
            "itemName": item_name,
            "suiteGroup": suite_group,
            "testValue": str(point["value"]),
            "referenceRange": reference_range,
            "description": "（HL7仿真-CGM）",
            "testTime": point["time"],
            "operatorId": 1
        }
        for point in series
    ]

    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False) as client:  # 批量数据多，超时适当放宽
            resp = await client.post(url, json={"items": items})
            print(f"[DEBUG] 请求地址={url}")
            print(f"[DEBUG] 状态码={resp.status_code}")
            print(f"[DEBUG] 原始响应内容={resp.text!r}")
            data = resp.json()
            if resp.status_code == 200 and data.get("code") == 200:
                result_data = data.get("data", {})
                detail = f"成功 {result_data.get('successCount', 0)} 条，失败 {result_data.get('failCount', 0)} 条"
                return HL7SendResult(True, detail, result_data)
            return HL7SendResult(False, data.get("message", "批量写入失败"))
    except Exception as e:
        return HL7SendResult(False, f"调用Java批量接口异常: {e}")


def write_raw_hl7_to_log(hl7_msg):
    log_path = "/tmp/hl7_sim_log.txt"
    # 拆分文件夹路径
    log_folder = os.path.dirname(log_path)
    # 不存在则创建文件夹
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    # 写入日志
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(hl7_msg + "\n")