# app/langchain_report/data_collector.py
"""
患者多模态数据采集器
从MySQL聚合：病历 + CT影像报告 + 检验报告 + 用药处方
"""

from app.db.mysql import get_conn
from datetime import datetime


def collect_patient_data(patient_id: int) -> dict:
    """
    采集指定患者的全部诊疗数据，供LangChain链式推理使用。
    返回结构化字典。
    """
    conn = get_conn()
    if not conn:
        raise RuntimeError("数据库连接失败")

    try:
        cur = conn.cursor(dictionary=True)
        data = {
            "patient_id": patient_id,
            "patient_info": None,
            "medical_records": [],
            "check_reports": [],
            "lab_reports": [],
            "prescriptions": [],
        }

        # 1. 患者基本信息
        cur.execute(
            "SELECT id, name, gender, phone, birth_date FROM pmi_patient WHERE id=%s",
            (patient_id,)
        )
        patient = cur.fetchone()
        if not patient:
            raise ValueError(f"患者 ID {patient_id} 不存在")

        if patient.get("birth_date"):
            age = (datetime.now().date() - patient["birth_date"]).days // 365
            patient["age"] = age
        patient["gender_text"] = "男" if patient.get("gender") == 1 else "女"
        data["patient_info"] = patient

        # 2. 病历记录（最近5条）
        cur.execute("""
            SELECT mr.id, mr.chief_complaint, mr.present_history,
                   mr.check_result, mr.diagnosis,
                   mr.ai_diagnosis, mr.ai_check_advice, mr.ai_drug_advice,
                   mr.ai_confirm_status, mr.create_time,
                   d.name AS doctor_name
            FROM medical_record mr
            LEFT JOIN doctor d ON d.id = mr.doctor_id
            WHERE mr.user_id = %s
            ORDER BY mr.create_time DESC
            LIMIT 5
        """, (patient_id,))
        records = cur.fetchall()
        for r in records:
            if isinstance(r.get("create_time"), datetime):
                r["create_time"] = r["create_time"].strftime("%Y-%m-%d %H:%M")
        data["medical_records"] = records

        # 3. CT影像报告
        cur.execute("""
            SELECT id, img_type, report_text, artifact_result,
                   ai_analysis, doctor_confirmed_text, ai_confirm_status,
                   create_time
            FROM check_report
            WHERE patient_id = %s
            ORDER BY create_time DESC
            LIMIT 5
        """, (patient_id,))
        ct_reports = cur.fetchall()
        for r in ct_reports:
            if isinstance(r.get("create_time"), datetime):
                r["create_time"] = r["create_time"].strftime("%Y-%m-%d %H:%M")
            # 解析伪影数据
            if r.get("artifact_result"):
                try:
                    import json
                    art = json.loads(r["artifact_result"])
                    r["artifact_pixel_count"] = art.get("artifact_pixel_count", 0)
                except:
                    r["artifact_pixel_count"] = 0
        data["check_reports"] = ct_reports

        # 4. 检验报告
        cur.execute("""
            SELECT id, item_name, test_value, reference_range,
                   abnormal_flag, audit_status, report_content,
                   create_time
            FROM lab_report
            WHERE patient_id = %s
            ORDER BY create_time DESC
            LIMIT 10
        """, (patient_id,))
        labs = cur.fetchall()
        for r in labs:
            if isinstance(r.get("create_time"), datetime):
                r["create_time"] = r["create_time"].strftime("%Y-%m-%d %H:%M")
            # 解析AI解读
            if r.get("report_content"):
                try:
                    import json
                    obj = json.loads(r["report_content"])
                    r["ai_summary"] = obj.get("desc", "")
                except:
                    r["ai_summary"] = ""
        data["lab_reports"] = labs

        # 5. 用药处方
        cur.execute("""
            SELECT p.id, p.drug_name, p.dosage, p.drug_usage,
                   p.quantity, p.days, p.unit_price, p.total_amount,
                   p.presc_status, p.pay_status, p.dispense_status,
                   p.create_time,
                   d.name AS doctor_name
            FROM prescription p
            LEFT JOIN doctor d ON d.id = p.doctor_id
            WHERE p.patient_id = %s
            ORDER BY p.create_time DESC
            LIMIT 10
        """, (patient_id,))
        prescriptions = cur.fetchall()
        for r in prescriptions:
            if isinstance(r.get("create_time"), datetime):
                r["create_time"] = r["create_time"].strftime("%Y-%m-%d %H:%M")
        data["prescriptions"] = prescriptions

        return data

    finally:
        cur.close()
        conn.close()


def format_data_for_prompt(data: dict) -> dict:
    """
    将采集的原始数据格式化为LangChain各Chain的输入文本
    """
    patient = data["patient_info"]
    patient_text = f"姓名：{patient['name']}，性别：{patient['gender_text']}，年龄：{patient.get('age', '未知')}岁"

    # 病历文本
    records_text = ""
    if data["medical_records"]:
        for i, r in enumerate(data["medical_records"]):
            records_text += f"\n第{i+1}次就诊（{r['create_time']}，{r.get('doctor_name', '')}）：\n"
            records_text += f"  主诉：{r.get('chief_complaint', '无')}\n"
            records_text += f"  现病史：{r.get('present_history', '无')}\n"
            records_text += f"  诊断：{r.get('diagnosis', '无')}\n"
            if r.get("ai_diagnosis"):
                records_text += f"  AI诊断建议：{r['ai_diagnosis']}\n"
    else:
        records_text = "暂无病历记录"
    # 新增：截断病历文本，限制1200字符
    records_text = records_text[:1200]

    # CT影像文本
    ct_text = ""
    if data["check_reports"]:
        for r in data["check_reports"]:
            ct_text += f"\n{r.get('img_type', 'CT')}报告（{r['create_time']}）：\n"
            ct_text += f"  报告结论：{r.get('report_text', '无')}\n"
            ct_text += f"  伪影像素数：{r.get('artifact_pixel_count', 0)}\n"
            if r.get("doctor_confirmed_text"):
                ct_text += f"  医生确认结论：{r['doctor_confirmed_text']}\n"
            elif r.get("ai_analysis"):
                ct_text += f"  AI影像分析：{r['ai_analysis'][:200]}...\n"
    else:
        ct_text = "暂无影像检查报告"
    # 新增：截断CT文本，限制1000字符
    ct_text = ct_text[:1000]

    # 检验文本
    lab_text = ""
    if data["lab_reports"]:
        for r in data["lab_reports"]:
            flag = "⚠异常" if r.get("abnormal_flag") == 1 else "正常"
            lab_text += f"  {r['item_name']}：{r['test_value']}（参考{r['reference_range']}，{flag}）\n"
            if r.get("ai_summary"):
                lab_text += f"    AI解读：{r['ai_summary'][:100]}\n"
    else:
        lab_text = "暂无检验报告"
    # 新增：截断检验文本，限制800字符
    lab_text = lab_text[:800]

    # 处方文本（可选，按需截断）
    rx_text = ""
    if data["prescriptions"]:
        for r in data["prescriptions"]:
            rx_text += f"  {r['drug_name']}，剂量{r.get('dosage', '—')}，用法{r.get('drug_usage', '—')}\n"
    else:
        rx_text = "暂无用药记录"

    return {
        "patient_info": patient_text,
        "medical_records": records_text,
        "ct_reports": ct_text,
        "lab_reports": lab_text,
        "prescriptions": rx_text,
    }
