# text2sql 一键启动脚本（Windows PowerShell 5.1+）
#   用法（任选其一）：
#     1) 终端里运行:  powershell -ExecutionPolicy Bypass -File .\start_all.ps1
#     2) 右键本文件 → "使用 PowerShell 运行"
#
#   启动顺序: MySQL(仅检查) → Qdrant(docker) → 决策服务(可选) → 后端(等就绪) → 前端
#   - 幂等：已在运行的服务自动跳过，可重复执行
#   - 后端/前端在各自独立窗口运行（标题 text2sql-backend / text2sql-frontend），关窗即停该服务
#   - 决策服务启动失败不阻塞（后端全链路 fail-open 回退 DeepSeek）
#   - 前端必须等后端就绪才启动（模型加载需要几十秒）

$ErrorActionPreference = "Stop"
$ROOT = $PSScriptRoot
Set-Location $ROOT
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$DECISION_START = Join-Path (Split-Path -Parent $ROOT) "startlux-decision\scripts\start_server.ps1"

function Test-Http($url) {
    try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true }
    catch { return $false }
}

function Wait-Http($url, $timeoutSec, $label) {
    foreach ($i in 1..$timeoutSec) {
        if (Test-Http $url) {
            Write-Host "      $label 就绪（等待 $i 秒）" -ForegroundColor Green
            return $true
        }
        if ($i % 15 -eq 0) { Write-Host "      ...已等 $i 秒（$label 还没好）" -ForegroundColor DarkGray }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Test-PortListen($port) {
    return $null -ne (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

function Start-ServiceWindow($title, $argLine) {
    # 在独立 PowerShell 窗口里启动长驻服务，日志直接显示在窗口里
    $cmd = "`$Host.UI.RawUI.WindowTitle = '$title'; Set-Location '$ROOT'; & '$PY' $argLine"
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $cmd) -WindowStyle Normal | Out-Null
}

Write-Host "==== text2sql 一键启动 ====" -ForegroundColor Cyan

# ---- [1/5] MySQL（Windows 服务，一般常驻；启动需要管理员，这里只检查） ----
if (Test-PortListen 3306) {
    Write-Host "[1/5] MySQL :3306 已在运行" -ForegroundColor Green
} else {
    Write-Host "[1/5] 警告：MySQL :3306 未监听！请以管理员身份运行: net start mysql" -ForegroundColor Yellow
}

# ---- [2/5] Qdrant（Docker 容器 :6333） ----
if (Test-Http "http://localhost:6333/collections") {
    Write-Host "[2/5] Qdrant :6333 已在运行" -ForegroundColor Green
} else {
    Write-Host "[2/5] 启动 Qdrant 容器..."
    docker compose up -d qdrant
    if (-not (Wait-Http "http://localhost:6333/collections" 30 "Qdrant")) {
        Write-Host "      Qdrant 未就绪（docker compose logs qdrant 查看原因）" -ForegroundColor Red
    }
}

# ---- [3/5] 决策服务（可选：llama-server :9081 + gguf_server :9090） ----
if (Test-Http "http://127.0.0.1:9090/") {
    Write-Host "[3/5] 决策服务 :9090 已在运行" -ForegroundColor Green
} elseif (Test-Path $DECISION_START) {
    Write-Host "[3/5] 启动决策服务（4B 模型上 GPU，约 10~20 秒）..."
    # 用子 powershell 跑：对方脚本失败时 exit 1，不能带崩本脚本
    & powershell -NoProfile -ExecutionPolicy Bypass -File $DECISION_START
    if (-not (Wait-Http "http://127.0.0.1:9090/" 20 "决策服务")) {
        Write-Host "      决策服务未就绪（可忽略，后端会回退 DeepSeek；日志在 startlux-decision\logs\）" -ForegroundColor Yellow
    }
} else {
    Write-Host "[3/5] 跳过决策服务（未找到 startlux-decision 目录，后端自动回退 DeepSeek）" -ForegroundColor DarkGray
}

# ---- [4/5] 后端（FastAPI :8888，加载 BGE-M3/reranker 需要几十秒） ----
if (Test-Http "http://127.0.0.1:8888/docs") {
    Write-Host "[4/5] 后端 :8888 已在运行" -ForegroundColor Green
} else {
    Write-Host "[4/5] 启动后端 :8888（新窗口 text2sql-backend，本地模型加载中...）"
    Start-ServiceWindow "text2sql-backend" "-m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8888"
    if (-not (Wait-Http "http://127.0.0.1:8888/docs" 300 "后端")) {
        Write-Host "      后端 300 秒内未就绪！查看 text2sql-backend 窗口里的报错" -ForegroundColor Red
        exit 1
    }
}

# ---- [5/5] 前端（Streamlit :8889，必须等后端就绪） ----
if (Test-Http "http://127.0.0.1:8889/") {
    Write-Host "[5/5] 前端 :8889 已在运行" -ForegroundColor Green
} else {
    Write-Host "[5/5] 启动前端 :8889（新窗口 text2sql-frontend）"
    Start-ServiceWindow "text2sql-frontend" "-m streamlit run frontend/app.py --server.port 8889"
    [void](Wait-Http "http://127.0.0.1:8889/" 60 "前端")
}

Write-Host ""
Write-Host "==== 全部就绪 ====" -ForegroundColor Cyan
Write-Host "  前端页面:   http://localhost:8889   （streamlit 也会自动开浏览器）"
Write-Host "  后端 API:   http://127.0.0.1:8888/docs"
Write-Host "  决策服务:   http://127.0.0.1:9090   （可选，后端不依赖）"
Write-Host "  停止全部:   powershell -ExecutionPolicy Bypass -File .\stop_all.ps1"
