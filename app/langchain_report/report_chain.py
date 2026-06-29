# app/langchain_report/report_chain.py
"""
LangChain 多模态病历生成链
4步链式推理：影像评估 → 检验分析 → 跨模态关联 → 综合报告
"""

import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from app.core.config import settings


def get_llm():
    print("当前读取的API_KEY:", repr(settings.DASHSCOPE_API_KEY))
    return ChatOpenAI(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-r1",
        temperature=0.7,
        max_tokens=4096,
        timeout=170
    )


# ══════════════════════════════════════════════════════════════
#  Chain 1: 影像质量评估
# ══════════════════════════════════════════════════════════════

imaging_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名资深影像科医师，擅长CT影像质量评估和伪影分析。请基于提供的CT报告数据做出专业评估。"),
    ("human", """
请对以下CT影像数据进行质量评估：

━━━ 患者信息 ━━━
{patient_info}

━━━ CT影像报告 ━━━
{ct_reports}

请输出以下内容（JSON格式）：
{{
    "image_quality": "影像质量等级（优/良/中/差）",
    "quality_detail": "质量评估详细说明（2-3句话）",
    "artifact_risk": "伪影对诊断的影响程度（无影响/轻微影响/需注意/建议重拍）",
    "key_findings": "影像关键发现（列出2-3个要点）"
}}

只输出JSON，不要其他内容。
""")
])


# ══════════════════════════════════════════════════════════════
#  Chain 2: 检验指标综合分析
# ══════════════════════════════════════════════════════════════

lab_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一名检验医学专家，擅长从多项检验指标中识别异常模式和临床意义。"),
    ("human", """
请对以下检验数据进行综合分析：

━━━ 患者信息 ━━━
{patient_info}

━━━ 检验报告 ━━━
{lab_reports}

请输出以下内容（JSON格式）：
{{
    "abnormal_count": "异常指标数量",
    "abnormal_items": "异常指标列表及临床意义",
    "pattern_analysis": "多指标组合模式分析（如代谢综合征模式、感染模式等）",
    "trend_warning": "趋势预警（基于历史数据判断是否有恶化趋势）",
    "suggested_recheck": "建议复查项目"
}}

只输出JSON，不要其他内容。
""")
])


# ══════════════════════════════════════════════════════════════
#  Chain 3: 跨模态关联推理（核心创新点）
# ══════════════════════════════════════════════════════════════

correlation_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名全科主任医师，擅长将影像、检验、病史多维度数据进行关联分析，
发现单一数据源无法揭示的诊断线索。这是你最重要的能力——跨模态临床推理。"""),
    ("human", """
请将以下多模态数据进行关联分析：

━━━ 患者信息 ━━━
{patient_info}

━━━ 病历记录 ━━━
{medical_records}

━━━ 影像评估结果 ━━━
{imaging_result}

━━━ 检验分析结果 ━━━
{lab_result}

━━━ 当前用药 ━━━
{prescriptions}

请进行跨模态关联推理，输出以下内容（JSON格式）：
{{
    "cross_modal_findings": "跨模态关联发现（影像+检验+病史的交叉验证结论，3-5条）",
    "primary_diagnosis": "综合诊断（按可能性排序，附置信度百分比）",
    "risk_assessment": "风险评估（高危/中危/低危，说明原因）",
    "missed_clues": "可能被忽略的诊断线索（基于数据矛盾或缺失提出）"
}}

只输出JSON，不要其他内容。
""")
])


# ══════════════════════════════════════════════════════════════
#  Chain 4: 综合诊疗报告生成
# ══════════════════════════════════════════════════════════════

report_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是智慧云脑诊疗平台的AI总结引擎。请基于前序分析结果，生成一份专业、完整、结构化的综合诊疗报告。
报告将展示给主治医师审阅，也将纳入患者电子病历档案。"""),
    ("human", """
请基于以下分析结果，生成综合诊疗报告：

━━━ 患者信息 ━━━
{patient_info}

━━━ 影像评估 ━━━
{imaging_result}

━━━ 检验分析 ━━━
{lab_result}

━━━ 跨模态关联推理 ━━━
{correlation_result}

━━━ 当前用药 ━━━
{prescriptions}

请生成完整报告（JSON格式）：
{{
    "report_title": "报告标题",
    "generated_at": "生成时间",
    "patient_summary": "患者概要（一句话）",
    "sections": [
        {{
            "title": "一、影像学评估",
            "content": "..."
        }},
        {{
            "title": "二、实验室检验分析",
            "content": "..."
        }},
        {{
            "title": "三、多模态关联诊断",
            "content": "..."
        }},
        {{
            "title": "四、风险评估与预警",
            "content": "..."
        }},
        {{
            "title": "五、治疗方案建议",
            "content": "（基于当前用药情况，给出调整建议或新增建议）"
        }},
        {{
            "title": "六、随访计划",
            "content": "..."
        }}
    ],
    "risk_level": "高危/中危/低危",
    "confidence_score": "诊断置信度（0-100）",
    "disclaimer": "本报告由AI辅助生成，仅供临床参考，最终诊断以主治医师判断为准。"
}}

只输出JSON，不要其他内容。
""")
])


# ══════════════════════════════════════════════════════════════
#  组装完整Chain
# ══════════════════════════════════════════════════════════════

def safe_parse_json(text: str) -> dict:
    """安全解析JSON，处理大模型可能输出的markdown代码块"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except:
        return {"raw_text": text, "parse_error": True}


def generate_report(formatted_data: dict) -> dict:
    """
    执行4步链式推理，生成完整报告。

    流程：
    Step1: 影像评估（CT数据 → 质量评估 + 关键发现）
    Step2: 检验分析（检验数据 → 异常模式 + 趋势预警）
    Step3: 跨模态关联（影像结果 + 检验结果 + 病历 → 综合诊断）
    Step4: 报告生成（所有结果 → 结构化报告）
    """
    llm = get_llm()
    str_parser = StrOutputParser()

    results = {
        "chain_steps": [],
        "final_report": None,
        "error": None,
    }

    try:
        # ── Step 1: 影像评估 ──
        print("[LangChain] Step 1/4: 影像质量评估...")
        imaging_chain = imaging_prompt | llm | str_parser
        imaging_raw = imaging_chain.invoke({
            "patient_info": formatted_data["patient_info"],
            "ct_reports": formatted_data["ct_reports"],
        })
        imaging_result = safe_parse_json(imaging_raw)
        results["chain_steps"].append({
            "step": 1,
            "name": "影像质量评估",
            "result": imaging_result,
        })
        print(f"[LangChain] Step 1 完成: {imaging_result.get('image_quality', '—')}")

        # ── Step 2: 检验分析 ──
        print("[LangChain] Step 2/4: 检验指标分析...")
        lab_chain = lab_prompt | llm | str_parser
        lab_raw = lab_chain.invoke({
            "patient_info": formatted_data["patient_info"],
            "lab_reports": formatted_data["lab_reports"],
        })
        lab_result = safe_parse_json(lab_raw)
        results["chain_steps"].append({
            "step": 2,
            "name": "检验指标分析",
            "result": lab_result,
        })
        print(f"[LangChain] Step 2 完成: 异常指标 {lab_result.get('abnormal_count', '—')} 项")

        # ── Step 3: 跨模态关联推理 ──
        print("[LangChain] Step 3/4: 跨模态关联推理...")
        correlation_chain = correlation_prompt | llm | str_parser

        # 序列化后截断，控制输入长度
        imaging_json = json.dumps(imaging_result, ensure_ascii=False)[:1000]
        lab_json = json.dumps(lab_result, ensure_ascii=False)[:1000]

        correlation_raw = correlation_chain.invoke({
            "patient_info": formatted_data["patient_info"],
            "medical_records": formatted_data["medical_records"],
            "imaging_result": imaging_json,
            "lab_result": lab_json,
            "prescriptions": formatted_data["prescriptions"],
        })
        correlation_result = safe_parse_json(correlation_raw)

        # ── Step 4: 综合报告生成 ──
        print("[LangChain] Step 4/4: 综合报告生成...")
        report_chain = report_prompt | llm | str_parser
        report_raw = report_chain.invoke({
            "patient_info": formatted_data["patient_info"],
            "imaging_result": json.dumps(imaging_result, ensure_ascii=False),
            "lab_result": json.dumps(lab_result, ensure_ascii=False),
            "correlation_result": json.dumps(correlation_result, ensure_ascii=False),
            "prescriptions": formatted_data["prescriptions"],
        })
        final_report = safe_parse_json(report_raw)
        results["chain_steps"].append({
            "step": 4,
            "name": "综合报告生成",
            "result": {"report_title": final_report.get("report_title", "")},
        })
        results["final_report"] = final_report
        print(f"[LangChain] 全部完成! 报告标题: {final_report.get('report_title', '—')}")
        print(f"[LangChain] Step 4 原始返回长度: {len(report_raw)}")
        print(f"[LangChain] Step 4 前200字: {report_raw[:200]}")

    except Exception as e:
        print(f"[LangChain] 生成失败: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    return results
