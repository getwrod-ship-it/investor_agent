"""冒烟测试: StartLux-Decision /v1/systemone (text2sql 路由场景)
用法: 先运行 scripts/start_server.ps1, 再 .venv\\Scripts\\python.exe scripts\\test_decision.py
"""
import json
import subprocess
import sys
import time
import urllib.request

URL = "http://127.0.0.1:9090/v1/systemone"

# 模拟 text2sql agent 的三类决策: 管线路由(choice) / 是否检索(noul) / 复杂度评分(score)
REQUEST = {
    "state": {
        "user_query": "上个月华东地区销售额前十的商品是哪些？对照知识库里的促销规则看哪些参与了满减",
        "dialect": "MySQL",
        "available_tables": ["orders", "order_items", "products", "regions"],
    },
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "该用户问题应路由到哪条处理管线？",
            "criteria": {
                "text2sql": "可用单条 SQL 直接回答的结构化数据查询",
                "rag": "主要依赖业务知识库文档（规则、流程）才能回答",
                "hybrid": "既要查数据库又要检索知识库",
                "reject": "与业务数据无关、超范围或有风险的请求",
            },
        },
        "need_schema_retrieval": {
            "type": "noul",
            "instructions": "回答前是否需要先检索数据库 schema 与业务知识库？",
        },
        "complexity": {
            "type": "score",
            "instructions": "该问题对应的 SQL 复杂度评分",
            "criteria": ["单表简单查询", "多表连接", "嵌套子查询", "多步骤分析", "需要窗口函数或复杂逻辑"],
        },
    },
}


def post(url, payload, timeout=600):
    t0 = time.perf_counter()
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, (time.perf_counter() - t0) * 1000


def main():
    print("=" * 60)
    print("测试 1: text2sql 路由问题 (choice + noul + score)")
    print("=" * 60)
    out, ms = post(URL, REQUEST)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f">> 延迟 {ms:.1f} ms, input_tokens={out.get('usage', {}).get('input_tokens')}")

    route = out["answers"]["route"]
    probs = sorted(route["probabilities"].values(), reverse=True)
    if probs[0] - probs[-1] < 1e-3:
        print("!! 概率分布接近均匀分布 —— 疑似命中 llama.cpp 决策头 bug，请更换构建版本")
        sys.exit(2)
    print(f">> 概率分布正常: 最高 {probs[0]:.4f} / 最低 {probs[-1]:.4f}（非塌缩）")

    print()
    print("=" * 60)
    print("测试 2: 延迟统计（预热 1 次后计时 10 次）")
    print("=" * 60)
    post(URL, REQUEST)
    lat = []
    for i in range(10):
        _, ms = post(URL, REQUEST)
        lat.append(ms)
        print(f"  第{i + 1:2d}次: {ms:8.1f} ms")
    lat.sort()
    print(f">> 平均 {sum(lat) / len(lat):.1f} | 中位 {lat[len(lat) // 2]:.1f} | 最小 {lat[0]:.1f} | 最大 {lat[-1]:.1f} ms")

    print()
    print("=" * 60)
    print("GPU 显存占用")
    print("=" * 60)
    subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"], shell=True)


if __name__ == "__main__":
    main()
