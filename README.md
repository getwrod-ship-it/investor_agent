# investor_agent — 基于 JEV 决策模型的中文 Text-to-SQL 智能体

> 用中文对业务数据库提问（如 *"待付款订单有多少个？"*、"上个月华东卖得最好的 10 个商品是什么"），
> Agent 先**看懂意图**（查数 / 闲聊 / 危险操作 / 歧义），**记得住上下文**（会话内逐字记忆 + 跨会话长期记忆），
> 再检索你的表结构与业务知识库，生成 MySQL SQL，安全校验后**只读执行**，最后用业务语言解释结果——
> 解释前还会自问"**结果真的回答了问题吗**"，答非所问就换策略重查，确实回答不了就诚实说明，而不是硬编。
> 关键路口由 **JEV 决策模型**（本地 StartLux-Decision 4B，llama.cpp 部署）做毫秒级门控——
> 意图识别、记忆检索、改写、写入四道判断本地秒出概率，省下一次次 LLM 调用；全链路 fail-open，模型不在场时自动回退纯 LLM 行为。

演示业务域为 [litemall](https://github.com/linlinjava/litemall) 电商数据库（600 用户 + 5 万订单 + 优惠券 + 售后单，附造数脚本可复现）。

## 仓库结构

```
investor_agent/
├── text-to-sql/                 # 主项目：FastAPI + LangGraph 的 Text-to-SQL Agent
│   ├── backend/app/             #   11 节点状态机、RAG、记忆系统、API 路由（≈6,700 行）
│   ├── frontend/app.py          #   Streamlit 网页（:8889）
│   ├── mcp_server/server.py     #   MCP Server（供 Claude 等 AI 客户端调用）
│   ├── knowledge_base/          #   业务知识库（9 篇中文业务文档，RAG 数据源）
│   ├── data/litemall/           #   建库 SQL + 5 万订单造数脚本（可复现）
│   ├── tests/                   #   208 个自动化测试（28 个文件，≈4,200 行）
│   └── start_all.ps1 等         #   一键启动 / 停止 / 记忆重置脚本
└── startlux-decision/           # 可选组件：本地决策模型部署（llama.cpp + 4B GGUF）
    ├── scripts/                 #   一键启停、冒烟测试、并行下载器
    └── README.md                #   选型记录、部署方法、实测数据
```

> 模型权重（BGE-M3 / BGE-reranker / StartLux GGUF）、虚拟环境、运行时数据库均不入库，
> 按[快速部署](#快速部署)步骤下载即可。

## 核心特性与新颖之处

1. **三模型级联，JEV 决策模型坐镇关键路口**：DeepSeek（秒级 SQL 生成）+ JEV 决策模型（本地 StartLux-Decision 4B，毫秒级门控）+ 本地 BGE-M3/reranker（检索）。JEV 提供 **4 个门控**——意图级联、长期记忆检索门控、写入门控、改写门控——全部默认关闭、逐门控开启，任一失败即回退 DeepSeek 行为（fail-open）。影子评测实测：意图一致率 91.7%、unsafe 零漏判；三模型在 RTX 3060 12G 上共存（9995 MiB）。
2. **EXPLAIN 可回答性判断**：SQL 合法、执行成功，但**答非所问**也算失败——EXPLAIN 节点单次调用产出三分 verdict（`answer` / `retry` / `give_up`），retry 携带失败原因回纠错环换法重查；`give_up` 让 LLM 自主止损、诚实收尾，不等重试配额烧完。空结果 + 条件一致 = 合法答案（"库里没有"），不会退化成"空行就重试"。
3. **语义失败三通道共享纠错环**：校验错 / 执行错 / 语义失败共用重试配额（MAX_RETRIES=3），CORRECT 节点注入此前全部尝试并禁止重复策略，修正后必须重新过静态校验，不存在跳过校验直接执行的路径。
4. **双层记忆**：短期 = LangGraph 官方 SqliteSaver checkpoint（thread_id = session_id），最近 5 轮逐字 + 更早轮滚动摘要（≤600 字，硬性保留表名/口径/数值/时间范围），崩溃后从审计行自愈重建；长期 = mem0 双实例——**事实记忆**（LLM 抽取 + 写时合并，改名走 UPDATE 取代不残留）与**情景记忆**（原文 append-only 零 LLM 成本），读在请求内（1.5s 超时降级）、写在响应后（后台无感），全部可从审计行无损重建。
5. **检索改写双保险**：LLM 先把追问（"那前个月呢"）改写为自包含检索问句；但改写**只服务检索**，SQL 生成用原问题 + 全量会话历史自行消解指代——改写错误不会传播到生成。
6. **三层安全防线**：GUARD 意图拦截（危险请求静态秒拒、零 LLM 成本）→ sqlglot 解析 + 禁词扫描 + 单语句约束 → 执行器只读连接 / 30s 超时 / 1000 行截断 / 二次禁词扫描。unsafe 轮永不写入任何记忆。
7. **三个等价入口**：Streamlit 网页、REST API、MCP Server（3 个工具，Claude 等 AI 客户端可直连）。
8. **即插即用**：换个 MySQL 库 + 换一套 `knowledge_base/*.md`，重新 `make embed` 即切换业务域；启动时强校验连接串必须是 MySQL。

## 架构

### 部署拓扑

```mermaid
flowchart LR
    subgraph entry["入口层（三个等价客户端，零业务逻辑）"]
        UI["Streamlit :8889"]
        MCP["MCP Server<br/>stdio 子进程"]
        CURL["curl / HTTP 客户端"]
    end
    subgraph backend["后端进程（uvicorn :8888 · GPU 常驻）"]
        API["FastAPI 5 组路由<br/>记忆读写调度在这里"]
        GRAPH["LangGraph 状态机（11 节点）<br/>thread_id = session_id"]
        MEM["mem0 长期记忆<br/>facts / episodic 双实例"]
        BGE["本地模型（进程级单例）<br/>bge-m3 嵌入 · bge-reranker 重排"]
    end
    subgraph docker["Docker"]
        QD[("Qdrant :6333<br/>5 集合 · 1024 维 COSINE")]
    end
    subgraph decision["可选：本地决策服务（默认不启动）"]
        DEC["gguf_server 概率包装层 :9090"]
        LS["llama-server :9081<br/>StartLux-Decision 4B Q8_0"]
        DEC -->|"/completion 取选项概率"| LS
    end
    subgraph ext["外部服务"]
        DS["DeepSeek API<br/>deepseek-flash"]
        MY[("业务 MySQL :3306<br/>litemall 库 · 只读")]
    end
    UI -->|"HTTP"| API
    MCP -->|"HTTP"| API
    CURL -->|"HTTP"| API
    API --> GRAPH
    GRAPH <-->|"短期记忆 checkpoint"| CKPT[("data/checkpoints.db")]
    API --> MEM
    GRAPH -->|"LLM 调用"| DS
    GRAPH -->|"只读 SQL"| MY
    GRAPH -->|"RETRIEVE"| BGE
    BGE --> QD
    MEM <-->|"向量读写"| QD
    API --> AUD[("data/system.db<br/>全量审计")]
```

### 11 节点状态机（三类失败共享纠错环）

```mermaid
flowchart TD
    START(("用户提问")) --> GUARD{"① GUARD 意图四分类<br/>结合会话历史 · 可本地级联"}
    GUARD -->|"闲聊"| CHAT["CHAT 数据助手应答（终态）"]
    GUARD -->|"危险"| REJECT["REJECT 静态拒绝（终态）<br/>零 LLM 成本"]
    GUARD -->|"歧义"| CLARIFY["CLARIFY 澄清（终态）"]
    GUARD -->|"查数"| REWRITE["② REWRITE 检索改写<br/>追问 → 自包含问句"]
    REWRITE --> RETRIEVE["③ RETRIEVE 检索<br/>表结构 + 知识库双路召回重排<br/>按来源配额各留席位"]
    RETRIEVE --> GENERATE["④ GENERATE 生成 SQL<br/>json_mode 结构化输出<br/>注入长期记忆与全量历史"]
    GENERATE --> VALIDATE{"⑤ VALIDATE 静态校验"}
    VALIDATE -->|"不通过"| CORRECT["⑦ CORRECT 纠错<br/>三通道失败原因喂回 LLM<br/>禁止重复策略"]
    VALIDATE -->|"通过"| EXECUTE{"⑥ EXECUTE 只读执行"}
    EXECUTE -->|"报错 · retry 小于 3"| CORRECT
    EXECUTE -->|"报错 · retry 达 3"| CLARIFY
    CORRECT -->|"修正后必须重新校验"| VALIDATE
    EXECUTE -->|"成功"| EXPLAIN{"⑧ EXPLAIN 可回答性三分<br/>answer / retry / give_up"}
    EXPLAIN -->|"answer / give_up"| DONE(("结果 + 中文解释"))
    EXPLAIN -->|"retry：答非所问，换法重试"| CORRECT
```

> 架构纪律：**图内节点是纯函数**（只读写 state，可独立单测），持久化与记忆调度全部归路由层。

## 快速部署

以下命令以 Windows PowerShell 为准（项目脚本 `start_all.ps1` 等均为 ps1）；Linux/macOS 把 `.venv\Scripts\python.exe` 换成 `.venv/bin/python` 即可。

### 前置要求

| 依赖 | 说明 |
|---|---|
| Python 3.12+ | 主项目 venv |
| MySQL 8.x | 业务数据库（本机或可达均可） |
| Docker | 仅用于跑 Qdrant 向量库 |
| [DeepSeek API Key](https://platform.deepseek.com/) | LLM 生成 / 解释 / 记忆抽取 |
| NVIDIA GPU（可选） | ≥6GB 显存跑本地 BGE 模型更快；无 GPU 把 `EMBEDDING_DEVICE` 改 `cpu` 也能跑 |

### 第 1 步：安装依赖

```powershell
git clone https://github.com/getwrod-ship-it/investor_agent.git
cd investor_agent\text-to-sql
python -m venv .venv
.venv\Scripts\Activate.ps1

# GPU 环境：先装 CUDA 版 torch（pyproject 故意不含 torch，避免装成 CPU 版）
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 纯 CPU 环境直接: pip install torch

pip install -e ".[dev]"
```

### 第 2 步：下载本地检索模型（约 4.4GB + 2.2GB）

```powershell
pip install modelscope
modelscope download --model BAAI/bge-m3 --local_dir models/bge-m3
modelscope download --model BAAI/bge-reranker-v2-m3 --local_dir models/bge-reranker-v2-m3
```

（海外网络可用 `huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3` 等价替换。`models/` 已被 .gitignore 忽略。）

### 第 3 步：建库导数据

```powershell
# 建库 + 建只读友好账号（litemall / litemall123456）
mysql -uroot -p < data\litemall\litemall_schema.sql
mysql -uroot -p litemall < data\litemall\litemall_table.sql     # 11 张核心表 DDL
mysql -uroot -p litemall < data\litemall\litemall_data.sql      # 基础种子数据

# 可选：生成 5 万订单（近 18 个月，含双11/618 高峰，固定种子可复现）
#   先编辑 data\litemall\generate_orders.py 顶部的 DB 连接信息
.venv\Scripts\python.exe data\litemall\generate_orders.py
```

### 第 4 步：配置环境变量

```powershell
copy .env.example .env
# 编辑 .env，至少填两项：
#   DEEPSEEK_API_KEY=sk-...
#   BUSINESS_DATABASE_URL=mysql+pymysql://litemall:litemall123456@localhost:3306/litemall
```

### 第 5 步：启动 Qdrant 并灌入向量

```powershell
docker compose up -d qdrant          # :6333
.venv\Scripts\python.exe -m app.embed_cli   # 爬表结构 + 切知识库 → 嵌入 Qdrant（幂等，可重跑）
```

### 第 6 步：启动服务

```powershell
# 方式 A：一键脚本（MySQL 检查 → Qdrant → 可选决策服务 → 后端 → 前端，幂等）
powershell -ExecutionPolicy Bypass -File .\start_all.ps1

# 方式 B：手动分起（两个终端）
.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8888
.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8889
```

打开 <http://localhost:8889> 即可开始提问。后端健康检查：<http://127.0.0.1:8888/health>。

> 端口说明：8888/8889 是为了避开部分 Windows 机器上 Hyper-V 保留的 7979–8778 端口段，可按需修改。

### 可选：接入本地决策模型（门控加速）

见 [`startlux-decision/README.md`](startlux-decision/README.md)：下载 4B GGUF 权重与 llama.cpp 后 `scripts\start_server.ps1` 一键启动（:9081/:9090），再把 `.env` 中 `DECISION_ENABLED=true` 打开。不开（或服务挂掉）时行为与纯 DeepSeek 版完全一致。

## 使用示例

**Streamlit**：<http://localhost:8889>，输入问题即可，同一会话内可连续追问。

**REST API**：

```bash
curl -s http://127.0.0.1:8888/query -H "Content-Type: application/json" \
  -d '{"question": "待付款订单有多少个？", "session_id": "demo-1"}'
# 返回含 intent（意图分类）、rewritten_question、sql、rows、answer、trace_id
```

**MCP**：`mcp_server/server.py` 提供 `query_tool` / `schema_tool` / `history_tool` 三个工具，供 Claude 等 MCP 客户端直连（示例配置见 `text-to-sql/.mcp.json`）。

## 核心配置项

完整清单见 [`text-to-sql/.env.example`](text-to-sql/.env.example)（每项都有注释）。

| 变量 | 说明 | 默认 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（必填） | — |
| `BUSINESS_DATABASE_URL` | 业务库连接串，必须是 `mysql+pymysql://`（必填，启动校验） | — |
| `DEEPSEEK_MODEL` | 模型 ID | `deepseek-flash` |
| `EMBEDDING_DEVICE` | 嵌入设备 | `cuda` |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant 地址 | `localhost` / `6333` |
| `MEMORY_ENABLED` | mem0 长期记忆总开关（false = 读空写跳过，fail-open） | `true` |
| `MEMORY_WINDOW_SIZE` | 会话内逐字保留的轮数 | `5` |
| `CLARIFY_AMBIGUITY_ENABLED` | GUARD 歧义澄清开关 | `true` |
| `EXPLAIN_ANSWERABILITY_ENABLED` | EXPLAIN 可回答性判断开关（关 = 旧纯文本路径） | `true` |
| `DECISION_ENABLED` | 本地决策门控总开关 | `false` |

## 测试

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v --tb=short    # 或 make test
```

208 个自动化测试（169 单测 + 39 集成），全部本地 mock 可跑；另有 4 个 mem0 / 9 个决策服务的真集成测试，用环境变量门控：`MEM0_IT=1`、`DECISION_IT=1`。

## 项目规模

| 项 | 数量 |
|---|---|
| 主项目源码（backend + frontend + mcp_server） | 68 个 Python 文件，≈6,700 行 |
| 自动化测试 | 28 个文件，≈4,200 行，208 个用例 |
| 业务知识库 | 9 篇中文文档（业务背景 / 口径映射 / 指标定义 / 典型问句解析等） |
| 造数脚本 | 600 用户 + 5 万订单 + 优惠券 + 售后，固定种子可复现 |

## 许可

- 本仓库代码：[MIT](LICENSE)
- `startlux-decision/` 引用的 StartLux-Decision 模型权重为第三方资产（CC BY-NC 4.0，**非商用**），推理代码 Apache-2.0，遵循[原仓库](https://huggingface.co/startlux-models)许可
- BGE-M3 / bge-reranker-v2-m3 遵循 BAAI 各自许可；litemall 数据库结构源自 [linlinjava/litemall](https://github.com/linlinjava/litemall)（MIT）
