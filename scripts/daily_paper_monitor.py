#!/usr/bin/env python3
"""
每日论文监控 — arXiv API + 本地 Qwen 筛选
用法: conda run -n vllm python daily_paper_monitor.py
配合 cron: 0 9 * * * cd /home/kuko/Kuko1414 && conda run -n vllm python scripts/daily_paper_monitor.py
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import os
import re
from datetime import datetime, timedelta
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
ARXIV_CATEGORIES = ["cs.RO", "cs.CV", "cs.AI", "cs.CL"]
MAX_RESULTS_PER_CATEGORY = 30
OUTPUT_DIR = os.path.expanduser("~/Kuko1414/Report/daily_papers")

# Qwen 本地 API
QWEN_CLIENT = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="no-key",
)

# ============================================================
# 关键词加权（用于检索，不是硬过滤）
# ============================================================
KEYWORDS_HIGH = [
    # A 组：VLM 导航 + 语义记忆
    "VLM navigation", "vision language model robot", "semantic mapping robot",
    "spatial memory", "scene graph robot", "navigation VLA",
    "open vocabulary navigation", "language grounded navigation",
    "structured spatial memory",
    # B 组：混合架构
    "state machine LLM robot", "supervisor LLM", "code as policies",
    "neuro-symbolic robot", "hierarchical planning LLM",
    "hybrid edge cloud LLM robot", "confidence escalation LLM",
    "deterministic safety", "skill switching",
    # C 组：VLA / Agent
    "function calling robot", "MCP robot protocol",
    "agentic robot planning", "tool use VLM robot",
]

KEYWORDS_MEDIUM = [
    "embodied AI", "robot navigation", "VLA model",
    "LLM planning robot", "instruction following robot",
    "visual grounding robot", "semantic SLAM",
    "exploration agent", "long-horizon task",
    "mobile manipulation", "indoor navigation",
]


def fetch_arxiv(category: str, max_results: int = 30) -> list[dict]:
    """从 arXiv API 拉取某个学科分类的最新论文"""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  ❌ 获取 {category} 失败: {e}")
        return []

    root = ET.fromstring(data)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        published_el = entry.find("atom:published", ns)
        cat_els = entry.findall("atom:category", ns)

        title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
        summary = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
        arxiv_id = id_el.text.strip() if id_el is not None else ""
        published = published_el.text.strip()[:10] if published_el is not None else ""
        cats = [c.get("term", "") for c in cat_els]

        papers.append({
            "arxiv_id": arxiv_id.split("/")[-1] if "/" in arxiv_id else arxiv_id,
            "title": title,
            "summary": summary[:1500],
            "published": published,
            "categories": cats,
            "url": arxiv_id,
            "score": 0,  # 关键词匹配分
        })

    return papers


def score_paper(paper: dict) -> int:
    """基于关键词匹配给论文打分（0-10）"""
    text = (paper["title"] + " " + paper["summary"]).lower()
    score = 0

    for kw in KEYWORDS_HIGH:
        if kw.lower() in text:
            score += 3
    for kw in KEYWORDS_MEDIUM:
        if kw.lower() in text:
            score += 1

    # 加分：有实际实验
    if re.search(r"(real.robot|real.world|deploy|hardware|physical)", text):
        score += 2
    # 加分：开源
    if re.search(r"(github|open.source|code.available)", text):
        score += 1
    # 加分：室内导航相关
    if re.search(r"(indoor|room|house|apartment|corridor)", text):
        score += 1

    return min(score, 10)


def qwen_filter(papers: list[dict], top_n: int = 5) -> list[dict]:
    """用本地 Qwen 对高分论文做最终筛选，返回最相关的 top_n 篇"""
    if len(papers) <= top_n:
        return papers

    # 构建 prompt
    paper_list = "\n\n".join([
        f"[{i+1}] {p['title']}\n   {p['summary'][:300]}..."
        for i, p in enumerate(papers[:15])  # 只送前 15 篇给 Qwen
    ])

    prompt = f"""你是一个机器人学研究者。以下是从 arXiv 最新论文中筛选的候选论文。
请根据以下标准选出最相关的 5 篇：
1. 涉及 VLM/LLM 用于移动机器人导航、建图、任务规划
2. 涉及结构化空间记忆（JSON、场景图、知识图谱等）
3. 涉及代码驱动或状态机与 LLM 混合架构
4. 有真实机器人实验（不是纯仿真）

请只返回论文编号（如 "1,3,5,7,9"），不要解释。

论文列表：
{paper_list}"""

    try:
        resp = QWEN_CLIENT.chat.completions.create(
            model="/home/kuko/.cache/huggingface/hub/qwen/Qwen3-VL-8B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        # 提取数字
        indices = [int(s) for s in re.findall(r"\d+", content) if 1 <= int(s) <= len(papers)]
        return [papers[i - 1] for i in indices[:top_n] if i - 1 < len(papers)]
    except Exception as e:
        print(f"  ⚠️ Qwen 筛选失败: {e}，回退到关键词排序")
        return papers[:top_n]


def generate_markdown(selected: list[dict], date_str: str) -> str:
    """生成 Markdown 日报"""
    lines = [
        f"# 📄 每日论文速递 — {date_str}",
        "",
        f"> 🔍 扫描范围: {', '.join(ARXIV_CATEGORIES)}",
        f"> 📊 今日筛选: {len(selected)} 篇",
        "",
        "---",
        "",
    ]

    for i, p in enumerate(selected):
        # 判断适配的贡献维度
        text = (p["title"] + " " + p["summary"]).lower()
        tags = []
        if any(kw.lower() in text for kw in ["navigation", "indoor", "mobile robot", "waypoint"]):
            tags.append("🟢 VLM导航")
        if any(kw.lower() in text for kw in ["spatial memory", "scene graph", "semantic map", "knowledge graph", "topological"]):
            tags.append("🔵 空间记忆")
        if any(kw.lower() in text for kw in ["state machine", "supervisor", "code as policies", "neuro-symbolic", "deterministic", "safety layer"]):
            tags.append("🟠 混合架构")
        if any(kw.lower() in text for kw in ["vla", "function calling", "tool use", "mcp", "agentic"]):
            tags.append("🟣 VLA/Agent")
        if any(kw.lower() in text for kw in ["small model", "edge", "deploy", "real-time", "local", "jeston"]):
            tags.append("⚪ 本地部署")
        if not tags:
            tags.append("⚫ 泛类")

        tag_str = " ".join(tags)

        lines += [
            f"## {i + 1}. {p['title']}",
            "",
            f"**标签:** {tag_str}",
            f"**arXiv:** [{p['arxiv_id']}]({p['url']})",
            f"**领域:** {', '.join(p['categories'][:3])}",
            "",
            f"> {p['summary'][:500]}{'...' if len(p['summary']) > 500 else ''}",
            "",
            "---",
            "",
        ]

    lines += [
        "## 💡 今日阅读建议",
        "",
        "优先阅读标记为 🟠 混合架构 和 🔵 空间记忆 的论文——这些最接近你的 Supervisor+STG 方案。",
        "标记为 🟢 VLM导航 的论文重点看他们的实验设计和 baseline。",
        "",
        f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]

    return "\n".join(lines)


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"📡 开始扫描 arXiv ({today})...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: 从 arXiv 拉取
    all_papers = []
    for cat in ARXIV_CATEGORIES:
        print(f"  📂 {cat}...")
        papers = fetch_arxiv(cat, MAX_RESULTS_PER_CATEGORY)
        # 只保留最近 3 天
        cutoff = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        papers = [p for p in papers if p["published"] >= cutoff]
        print(f"     获取 {len(papers)} 篇（3日内）")
        all_papers.extend(papers)

    # Step 2: 去重
    seen = set()
    unique = []
    for p in all_papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)
    print(f"  📋 去重后: {len(unique)} 篇")

    # Step 3: 关键词打分
    for p in unique:
        p["score"] = score_paper(p)
    unique.sort(key=lambda x: x["score"], reverse=True)

    # 高分论文
    candidates = [p for p in unique if p["score"] >= 4]
    print(f"  ⭐ 高分 (≥4): {len(candidates)} 篇")

    # Step 4: Qwen 终筛
    selected = qwen_filter(candidates, top_n=5)
    print(f"  ✅ Qwen 精选: {len(selected)} 篇")

    # Step 5: 生成 Markdown
    md = generate_markdown(selected, today)
    output_path = os.path.join(OUTPUT_DIR, f"papers_{today}.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"📝 日报已保存: {output_path}")

    # 打印摘要
    for i, p in enumerate(selected):
        print(f"  {i+1}. {p['title'][:100]}")


if __name__ == "__main__":
    main()
