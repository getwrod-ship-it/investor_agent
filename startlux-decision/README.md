# StartLux-Decision-v1-4B 本地部署（RTX 3060 12G / Windows）

> `text-to-sql` 主项目的**可选组件**：本地决策模型部署。主项目默认不依赖它（全链路 fail-open），
> 在 `.env` 中打开 `DECISION_ENABLED=true` 即接入（见文末「接入 text-to-sql」）。
> 模型主页: startlux-models/StartLux-Decision-v1-4B-Q8_0-GGUF（HuggingFace）
> 底座: qwen3_5 · 权重许可 CC BY-NC 4.0（**非商用**）· 推理代码 Apache-2.0

## 一、为什么选 4B Q8_0

| 档位 | 大小 | 与原版决策一致率 | JevBench(231) | 备注 |
|---|---|---|---|---|
| **4B Q8_0（本部署）** | 4.48GB | **100%** | 204 (88.3%) | 官方"recommended" |
| 4B Q4_K_M | 2.71GB | 98.3% | 201 (87.0%) | 官方"smallest"，备用 |
| 4B BF16 | 8.42GB | 100% | 204 (88.3%) | 原版权重，显存吃不消 |
| 2B（对比） | ~2.2GB | — | 196 (84.9%) | 平均分 88.46 vs 4B 的 91.17，DI 44.19 vs 52.75 |

- 4B 全面优于 2B（JevBench +8 题，Intern avg 91.17 vs 88.46，DI 52.75 vs 44.19），H200 单次三问延迟 26ms vs 15.5ms，3060 上差异可忽略
- Q8_0 相比 Q4_K_M 决策一致率 100% vs 98.3%，且 3060 12G 装得下 → 取 Q8_0
- 显存账: 4B Q8_0 ~5.5GB（含 KV）+ BGE-M3 ~1.2GB + reranker ~1.1GB ≈ **8-10GB，12G 卡可三模型共存**（见下方实测）

## 二、部署架构（关键认知）

```
客户端 ──POST /v1/systemone──▶ gguf_server (Python 包装层, 127.0.0.1:9090)
                                   │ 渲染提示词 / 选项字母读出 / 按类型温度校准 / 归一化
                                   ▼ /completion (n_probs 取全量 logprobs)
                               llama-server (GGUF GPU 推理, 127.0.0.1:9081)
```

- **决策头不在权重里**: 直接用 llama.cpp 的 chat 接口拿不到选项概率；`startlux_decision.gguf_server` 包装层负责概率读出（llama.cpp ≥b10454 原生支持该架构推理，但概率输出必须配官方包装层）
- **GGUF 纯文本输入**，对 text2sql 路由场景（state 全是文本/JSON）无影响
- 一次请求的所有问题在**一次前向**里完成，零生成，延迟极低
- 三种问题类型（decision_config.json 温度）: `choice` 1.3872 / `noul`(二元) 1.5905 / `score` 1.4172

## 三、目录结构

```
startlux-decision\
├── models\StartLux-Decision-v1-4B-Q8_0-GGUF\   # HF 仓库完整下载（GGUF + startlux_decision 包 + tokenizer）
├── llama.cpp\                                   # b11567 官方预编译（win-cuda-12.4-x64）
├── scripts\
│   ├── start_server.ps1                         # 一键启动（llama-server + gguf_server）
│   ├── stop_server.ps1                          # 一键停止
│   ├── test_decision.py                         # 冒烟测试（text2sql 路由场景）
│   └── pdownload.sh                             # 多连接并行下载器（走本地代理）
├── logs\                                        # 服务日志
└── .venv\                                       # 独立 Python 3.13（CPU torch 2.14.1 + transformers 5.19.0）
```

## 四、使用方法（PowerShell）

在本目录（`startlux-decision/`）下执行：

```powershell
# 启动（先起 llama-server 9081，就绪后再起决策头 9090）
powershell -ExecutionPolicy Bypass -File .\scripts\start_server.ps1

# 冒烟测试（text2sql 路由场景，5 问应全部语义合理）
.\.venv\Scripts\python.exe .\scripts\test_decision.py

# 停止
powershell -ExecutionPolicy Bypass -File .\scripts\stop_server.ps1
```

## 五、接口示例（TypeSafe /v1/systemone）

```bash
curl -s http://127.0.0.1:9090/v1/systemone -H "Content-Type: application/json" -d '{
  "state": {"user_query": "上月华东销售额前十的商品", "dialect": "MySQL"},
  "questions": {
    "route": {"type": "choice", "instructions": "路由到哪条管线?",
              "criteria": {"text2sql": "单条SQL可答", "rag": "查知识库", "hybrid": "两者都要", "reject": "超范围"}},
    "need_retrieval": {"type": "noul", "instructions": "回答前需要检索 schema/知识库?"}
  }
}'
# → {"answers": {"route": {"type":"choice","choice":"...","confidence":...,"probabilities":{...}},
#                "need_retrieval": {"type":"noul","noul":0.97}}, "usage": {"input_tokens":N}, "model":"..."}
```

- `choice` → 每个选项的概率字典 + 最高项；`noul` → true 的概率；`score` → 加权分 + 各档概率
- text2sql 侧接入: 把 typesafe-sdk 的 baseUrl 指向 `http://127.0.0.1:9090` 即可（协议与 Jev 兼容）

## 六、实测记录（3060 12G, 驱动 561.17, 2026-10-12）

- llama.cpp 构建: b11567（win-cuda-12.4，驱动 561 只支持 CUDA 12.x，勿装 cu13.4 包）
- 端口: 9081 (llama-server) / 9090 (决策头)。避开 Hyper-V 保留段 7979-8778 和主项目 8888/8889
- 模型完整性: 下载后 SHA256 与 HF 官方一致（3de47f14...e34df）
- stop/start 脚本完整生命周期验证通过（llama-server 冷启动到就绪 ~4s，决策服务 ~3s）

| 指标 | 数值 |
|---|---|
| 首次请求（冷，含全链路） | 648.6 ms |
| 稳态延迟（3 问/次，438 tokens，10 次均值） | **287.9 ms**（中位 290.9 / 最小 265.0 / 最大 315.0），约 96ms/问 |
| 单问短 prompt（~150 tokens） | ~90-100 ms（重启后复验 text2sql 置信 0.904） |
| 决策服务显存（llama-server + 包装层） | **6425 MiB / 12288 MiB** |
| 加上 BGE-M3 + reranker（FP16 约 2.3GB）预估 | ~8.8 GB，12G 卡三模型共存可行 |
| 概率分布校验 | 通过（choice 最高 0.718 / 最低 0.016，未塌缩） |
| 语义路由合理性 | 5/5（text2sql 0.84 / reject 0.89 / 危险请求拒绝 0.835 / 业务规则 rag 0.535 / 明显SQL 0.904） |

延迟说明: 文档预期的"几十 ms"是 H200 上纯前向的时间；这里的 ~96ms/问 含 HTTP 往返、提示词渲染、
每问一次 llama-server /completion 调用（包装层逐问取 logprobs）。对 agent 路由场景完全够用；
若以后要压延迟，可让包装层复用 llama-server 的 prompt cache 或批处理问题。

## 七、接入 text-to-sql（已实现）

主项目已通过四个门控接入本服务（`text-to-sql/backend/app/agent/decision.py` 薄客户端，任一失败返回 None → 走 DeepSeek 原路径）：

1. 在 `text-to-sql/.env` 中设置 `DECISION_ENABLED=true`（`DECISION_BASE_URL` 默认 `http://127.0.0.1:9090`）
2. 四个门控（各自独立开关，默认随总开关全开）：
   - `DECISION_GUARD_ENABLED` — GUARD 意图级联：本地四分类置信 ≥ `DECISION_INTENT_THRESHOLD`(0.70) 直接采用，省一次 DeepSeek
   - `DECISION_LTM_GATE_ENABLED` — 长期记忆检索门控：P(需要) < `DECISION_LTM_GATE_THRESHOLD`(0.60) 跳过召回
   - `DECISION_WRITE_GATE_ENABLED` — 长期记忆写入门控：后台异步
   - `DECISION_REWRITE_GATE_ENABLED` — 检索改写门控：问题已自包含则直通，省一次 DeepSeek
3. 影子模式灰度：`DECISION_SHADOW_MODE=true` 只记录不采用
4. `text-to-sql/start_all.ps1` 会尝试顺带拉起本服务，失败不阻塞

实测（影子评测）：意图一致率 91.7%、unsafe 采用集零漏判、LTM 门控完美分离；三模型 GPU 共存 9995/12288 MiB。

## 八、下载备忘（国内网络）

- ModelScope 无此模型镜像（已确认）；hf-mirror.com 只代理 API，Xet 大文件会 302 到 cas-bridge.xethub.hf.co（直连仅 ~560KB/s）
- **可行路线: 本地代理 127.0.0.1:7897 → huggingface.co 直源 + pdownload.sh 8 连接并行**（实测聚合 ~3MB/s，4.48GB 约 25 分钟）
- llama.cpp 走 GitHub release 直链同样用 pdownload.sh 并行（GitHub API 有 IP 限流，用 releases.atom + expanded_assets 页面找 tag）
- ⚠️ 教训: 不要编辑正在运行的 bash 脚本（运行中进程按字节偏移继续读文件，会解析错乱）
