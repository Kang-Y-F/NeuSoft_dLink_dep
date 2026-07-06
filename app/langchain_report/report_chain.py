# app/langchain_report/report_chain.py
"""
LangChain + LangGraph 多模态病历生成链

相比原来的线性 4 步 Chain，这一版用 LangGraph 的 StateGraph 组织成一张图：

  collect_data ──┬─→ imaging ──┐
                 └─→ lab ──────┴─→ correlation_agent ⇄ correlation_tool
                                          │（无需再调用工具时）
                                          ▼
                                  correlation_finalize
                                          │
                                          ▼
                                   generate_report ⇄ validate_report ──→ fallback
                                                            │(通过)
                                                            ▼
                                                           END

改动点（对应答辩要讲的三件事）：
1. 影像评估(imaging) 和 检验分析(lab) 互不依赖，改成并行分支，而不是原来的顺序执行
2. generate_report → validate_report 之间有条件边：JSON 解析失败就回退重试（最多 MAX_REPORT_RETRY 次），
   而不是原来那样解析失败就摆烂把 raw_text 传下去
3. correlation_agent 绑定了 search_similar_patients 这个工具（包装自 pgvector 的相似病例检索），
   由大模型自己判断当前病例是否有必要参考历史相似病例，而不是代码里写死调用顺序——这是唯一真正
   意义上的"agent"行为（模型自主决策是否使用工具）

⚠️ 重要前提：下面 correlation_agent 用的是 LangChain 标准的 bind_tools() 原生工具调用（function calling）。
   这依赖你实际使用的模型/网关是否支持 OpenAI 风格的 tool_calls。如果 DashScope 兼容模式下的
   deepseek-r1 不支持原生 function calling，response.tool_calls 会一直是空，
   相当于 correlation_agent 永远直接进入 finalize，不会报错，但也不会真的调用工具。
   如果发现这一步一直没触发工具调用，需要改成"手动模式"：在 prompt 里要求模型输出一个
   {"need_tool": true/false, ...} 字段，代码里解析这个字段来决定是否调用 search_similar_patients，
   而不是依赖模型原生的 function calling 能力。需要这个手动回退版本可以再单独要。
"""

import json
import operator
from typing import TypedDict, List, Optional, Any
from typing_extensions import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END

from app.core.config import settings
from app.db.pgvector_store import find_similar_vec
from app.langchain_report.data_collector import collect_patient_data, format_data_for_prompt

MAX_TOOL_CALLS   = 2   # correlation_agent 最多调用几轮工具，防止死循环
MAX_REPORT_RETRY = 2   # 最终报告JSON解析失败最多重试几次


def get_llm():
    return ChatOpenAI(
        api_key=settings.DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="deepseek-r1",
        temperature=0.7,
        max_tokens=4096,
        timeout=170,
    )


def safe_parse_json(text: str) -> dict:
    """安全解析JSON，处理大模型可能输出的markdown代码块"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except Exception:
        return {"raw_text": text, "parse_error": True}


# ══════════════════════════════════════════════════════════════
#  Agent Tool：相似病例检索（包装 pgvector）
# ══════════════════════════════════════════════════════════════

@tool
def search_similar_patients(patient_id: int, top_k: int = 3) -> str:
    """检索与指定患者CT特征向量余弦相似度最高的历史病例，返回患者ID和相似度。
    当当前患者的影像评估或检验分析存在不确定性、诊断依据不够充分、
    或者需要参考历史上类似病例的处理方式时，调用这个工具辅助判断。
    不确定的情况下可以调用；数据充分、结论明确时不需要调用。
    """
    results = find_similar_vec(patient_id, top_k)
    if results is None:
        return f"患者 {patient_id} 尚无CT特征向量数据，无法检索相似病例。"
    if not results:
        return "特征库中暂无其他可比对的患者，无法检索相似病例。"
    lines = [f"- 患者{r['patientId']}，相似度 {r['similarity']*100:.1f}%" for r in results]
    return "检索到以下相似病例（按相似度降序）：\n" + "\n".join(lines)


TOOLS = [search_similar_patients]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ══════════════════════════════════════════════════════════════
#  Chain 1 / 2 的静态 Prompt（跟原来一样，纯文本分析，不需要工具）
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
        {{"title": "一、影像学评估", "content": "..."}},
        {{"title": "二、实验室检验分析", "content": "..."}},
        {{"title": "三、多模态关联诊断", "content": "..."}},
        {{"title": "四、风险评估与预警", "content": "..."}},
        {{"title": "五、治疗方案建议", "content": "（基于当前用药情况，给出调整建议或新增建议）"}},
        {{"title": "六、随访计划", "content": "..."}}
    ],
    "risk_level": "高危/中危/低危",
    "confidence_score": "诊断置信度（0-100）",
    "disclaimer": "本报告由AI辅助生成，仅供临床参考，最终诊断以主治医师判断为准。"
}}

只输出JSON，不要其他内容。
""")
])

CORRELATION_SYSTEM_PROMPT = """你是一名全科主任医师，擅长将影像、检验、病史多维度数据进行关联分析，
发现单一数据源无法揭示的诊断线索。

你可以调用 search_similar_patients 工具，检索历史上与当前患者CT特征相似的病例作为参考。
是否调用由你自行判断：如果当前影像和检验数据已经足够支撑明确结论，不需要调用；
如果存在不确定、需要历史参考佐证，再调用。最多调用2次。

当你不再需要调用工具、可以给出最终结论时，只输出以下JSON格式，不要输出其他任何内容（包括不要输出```标记）：
{
    "cross_modal_findings": "跨模态关联发现（影像+检验+病史的交叉验证结论，3-5条）",
    "primary_diagnosis": "综合诊断（按可能性排序，附置信度百分比）",
    "risk_assessment": "风险评估（高危/中危/低危，说明原因）",
    "missed_clues": "可能被忽略的诊断线索（基于数据矛盾或缺失提出）",
    "referenced_similar_cases": "如果调用过相似病例检索，说明参考了哪些患者、如何影响了你的判断；未调用则填'未参考'"
}
"""


# ══════════════════════════════════════════════════════════════
#  LangGraph State 定义
# ══════════════════════════════════════════════════════════════

class ReportState(TypedDict, total=False):
    patient_id: int
    patient_name: str
    formatted_data: dict

    imaging_result: dict
    lab_result: dict

    correlation_messages: List[Any]
    correlation_tool_calls: int
    correlation_result: dict

    final_report_raw: str
    final_report: Optional[dict]
    report_retry: int

    # 用 operator.add 作为reducer：imaging/lab 并行节点同一步都会写这个key，
    # 默认LastValue只能接受一个值会报错，加了这个之后LangGraph会自动把多个并行写入
    # 的列表拼接起来，而不是互相覆盖/冲突
    chain_steps: Annotated[List[dict], operator.add]
    error: Optional[str]


# ══════════════════════════════════════════════════════════════
#  节点实现
# ══════════════════════════════════════════════════════════════

def node_collect_data(state: ReportState) -> dict:
    raw = collect_patient_data(state["patient_id"])
    formatted = format_data_for_prompt(raw)
    return {
        "formatted_data": formatted,
        "patient_name": raw["patient_info"]["name"],
        "chain_steps": [],
        "report_retry": 0,
        "correlation_tool_calls": 0,
    }


def node_imaging(state: ReportState) -> dict:
    print("[LangGraph] 节点: 影像质量评估")
    chain = imaging_prompt | get_llm() | StrOutputParser()
    raw = chain.invoke({
        "patient_info": state["formatted_data"]["patient_info"],
        "ct_reports": state["formatted_data"]["ct_reports"],
    })
    result = safe_parse_json(raw)
    return {
        "imaging_result": result,
        # 只返回这一步新增的条目，累加交给 chain_steps 的 operator.add reducer去做，
        # 不要再手动 state["chain_steps"] + [...]，否则并行分支合并后会重复
        "chain_steps": [{"step": 1, "name": "影像质量评估", "result": result}],
    }


def node_lab(state: ReportState) -> dict:
    print("[LangGraph] 节点: 检验指标分析")
    chain = lab_prompt | get_llm() | StrOutputParser()
    raw = chain.invoke({
        "patient_info": state["formatted_data"]["patient_info"],
        "lab_reports": state["formatted_data"]["lab_reports"],
    })
    result = safe_parse_json(raw)
    return {
        "lab_result": result,
        "chain_steps": [{"step": 2, "name": "检验指标分析", "result": result}],
    }


def node_correlation_agent(state: ReportState) -> dict:
    """
    correlation_agent 的第一次调用：组装消息，绑定工具，让模型自己决定要不要调用。
    后续如果模型请求了工具，会在 node_correlation_tool 里执行，再回到这个节点继续对话
    （所以这个节点其实会被反复进入，靠 correlation_messages 里已有的历史区分是第几轮）。
    """
    print(f"[LangGraph] 节点: 跨模态关联推理（第 {state.get('correlation_tool_calls', 0) + 1} 轮）")

    if not state.get("correlation_messages"):
        human_content = f"""
━━━ 患者信息 ━━━
{state["formatted_data"]["patient_info"]}

━━━ 病历记录 ━━━
{state["formatted_data"]["medical_records"]}

━━━ 影像评估结果 ━━━
{json.dumps(state["imaging_result"], ensure_ascii=False)[:1000]}

━━━ 检验分析结果 ━━━
{json.dumps(state["lab_result"], ensure_ascii=False)[:1000]}

━━━ 当前用药 ━━━
{state["formatted_data"]["prescriptions"]}

请开始你的跨模态关联推理。患者ID是 {state["patient_id"]}，如需检索相似病例请用这个ID调用工具。
"""
        messages = [SystemMessage(content=CORRELATION_SYSTEM_PROMPT), HumanMessage(content=human_content)]
    else:
        messages = state["correlation_messages"]

    llm_with_tools = get_llm().bind_tools(TOOLS)
    response = llm_with_tools.invoke(messages)
    return {"correlation_messages": messages + [response]}


def node_correlation_tool(state: ReportState) -> dict:
    """执行模型请求的工具调用，把结果作为 ToolMessage 追加进对话历史"""
    messages = state["correlation_messages"]
    last: AIMessage = messages[-1]
    tool_messages = []

    for call in last.tool_calls:
        tool_fn = TOOLS_BY_NAME.get(call["name"])
        if tool_fn is None:
            content = f"未知工具：{call['name']}"
        else:
            try:
                content = tool_fn.invoke(call["args"])
            except Exception as e:
                content = f"工具调用异常：{e}"
        print(f"[LangGraph] Agent 调用工具 {call['name']}({call['args']}) → {content[:80]}...")
        tool_messages.append(ToolMessage(content=str(content), tool_call_id=call["id"]))

    return {
        "correlation_messages": messages + tool_messages,
        "correlation_tool_calls": state.get("correlation_tool_calls", 0) + 1,
    }


def route_correlation(state: ReportState) -> str:
    last = state["correlation_messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls and state.get("correlation_tool_calls", 0) < MAX_TOOL_CALLS:
        return "call_tool"
    return "finalize"


def node_correlation_finalize(state: ReportState) -> dict:
    last = state["correlation_messages"][-1]
    result = safe_parse_json(last.content)
    return {
        "correlation_result": result,
        "chain_steps": [{
            "step": 3,
            "name": "跨模态关联推理",
            "result": result,
            "tool_calls_used": state.get("correlation_tool_calls", 0),
        }],
    }


def node_generate_report(state: ReportState) -> dict:
    print(f"[LangGraph] 节点: 生成综合报告（第 {state.get('report_retry', 0) + 1} 次尝试）")
    chain = report_prompt | get_llm() | StrOutputParser()
    raw = chain.invoke({
        "patient_info": state["formatted_data"]["patient_info"],
        "imaging_result": json.dumps(state["imaging_result"], ensure_ascii=False),
        "lab_result": json.dumps(state["lab_result"], ensure_ascii=False),
        "correlation_result": json.dumps(state["correlation_result"], ensure_ascii=False),
        "prescriptions": state["formatted_data"]["prescriptions"],
    })
    return {"final_report_raw": raw}


def node_validate_report(state: ReportState) -> dict:
    result = safe_parse_json(state["final_report_raw"])
    if result.get("parse_error"):
        print(f"[LangGraph] 报告JSON解析失败，当前重试次数 {state.get('report_retry', 0)}")
        return {"final_report": None, "report_retry": state.get("report_retry", 0) + 1}
    return {
        "final_report": result,
        "chain_steps": [{"step": 4, "name": "综合报告生成", "result": {"report_title": result.get("report_title", "")}}],
    }


def route_validate(state: ReportState) -> str:
    if state.get("final_report") is not None:
        return "success"
    if state.get("report_retry", 0) >= MAX_REPORT_RETRY:
        return "give_up"
    return "retry"


def node_fallback(state: ReportState) -> dict:
    print("[LangGraph] 报告生成重试超限，走降级输出")
    return {
        "final_report": {
            "report_title": "AI报告生成异常",
            "sections": [],
            "risk_level": "未知",
            "confidence_score": 0,
            "disclaimer": "AI输出格式多次解析失败，请医生基于原始数据人工判断，或点击重新生成。",
            "raw_text": state.get("final_report_raw", ""),
        },
        "error": "报告JSON解析多次失败，已降级输出原始文本",
    }


# ══════════════════════════════════════════════════════════════
#  组装 Graph
# ══════════════════════════════════════════════════════════════

def build_graph():
    workflow = StateGraph(ReportState)

    workflow.add_node("collect_data", node_collect_data)
    workflow.add_node("imaging", node_imaging)
    workflow.add_node("lab", node_lab)
    workflow.add_node("correlation_agent", node_correlation_agent)
    workflow.add_node("correlation_tool", node_correlation_tool)
    workflow.add_node("correlation_finalize", node_correlation_finalize)
    workflow.add_node("generate_report", node_generate_report)
    workflow.add_node("validate_report", node_validate_report)
    workflow.add_node("fallback", node_fallback)

    workflow.set_entry_point("collect_data")

    # 并行分支：影像评估和检验分析互不依赖
    workflow.add_edge("collect_data", "imaging")
    workflow.add_edge("collect_data", "lab")

    # 两个分支都完成后，汇合进入跨模态关联推理
    workflow.add_edge("imaging", "correlation_agent")
    workflow.add_edge("lab", "correlation_agent")

    # agent 工具调用循环
    workflow.add_conditional_edges(
        "correlation_agent", route_correlation,
        {"call_tool": "correlation_tool", "finalize": "correlation_finalize"},
    )
    workflow.add_edge("correlation_tool", "correlation_agent")

    workflow.add_edge("correlation_finalize", "generate_report")
    workflow.add_edge("generate_report", "validate_report")

    # 报告JSON校验的条件重试
    workflow.add_conditional_edges(
        "validate_report", route_validate,
        {"success": END, "retry": "generate_report", "give_up": "fallback"},
    )
    workflow.add_edge("fallback", END)

    return workflow.compile()


_GRAPH = build_graph()


# ══════════════════════════════════════════════════════════════
#  对外暴露的入口函数（供 CTDetectionServer.py 的 /ai/generate-report/{id} 调用）
# ══════════════════════════════════════════════════════════════

def generate_report(patient_id: int) -> dict:
    """
    执行完整的图流程，返回：
    {
        "chain_steps": [...],      # 每一步的中间结果，供前端展示推理过程
        "final_report": {...},     # 最终报告
        "patient_name": "...",
        "error": None 或 错误信息,
    }
    """
    try:
        final_state = _GRAPH.invoke({"patient_id": patient_id})
        return {
            "chain_steps": final_state.get("chain_steps", []),
            "final_report": final_state.get("final_report"),
            "patient_name": final_state.get("patient_name", ""),
            "error": final_state.get("error"),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"chain_steps": [], "final_report": None, "patient_name": "", "error": str(e)}