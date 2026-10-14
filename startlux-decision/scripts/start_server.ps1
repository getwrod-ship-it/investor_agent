# 启动 StartLux-Decision 决策服务
#   llama-server (GPU 推理, 端口 9081) + gguf_server (决策头 /v1/systemone, 端口 9090)
# 端口说明: Hyper-V 保留了 7979-8778, 8888/8889 是 text2sql 主项目, 故选 9081/9090
$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
$MODEL = Join-Path $ROOT "models\StartLux-Decision-v1-4B-Q8_0-GGUF"
$GGUF = Join-Path $MODEL "StartLux-Decision-4B-Q8_0.gguf"
$LLAMA = Join-Path $ROOT "llama.cpp\llama-server.exe"
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LLAMA_PORT = 9081
$DECISION_PORT = 9090
$LOGS = Join-Path $ROOT "logs"
New-Item -ItemType Directory -Force -Path $LOGS | Out-Null

if (-not (Test-Path $GGUF)) { throw "找不到 GGUF 文件: $GGUF" }
if (-not (Test-Path $LLAMA)) { throw "找不到 llama-server: $LLAMA（先解压 llama.cpp 压缩包）" }

# 若服务已在跑, 先停掉
& (Join-Path $PSScriptRoot "stop_server.ps1") 2>$null

Write-Host "[1/3] 启动 llama-server (GPU, 端口 $LLAMA_PORT)..."
$llamaProc = Start-Process -FilePath $LLAMA `
    -ArgumentList @("-m", $GGUF, "-ngl", "99", "-c", "16384", "--parallel", "4", "--port", "$LLAMA_PORT") `
    -WorkingDirectory (Join-Path $ROOT "llama.cpp") `
    -RedirectStandardOutput (Join-Path $LOGS "llama-server.log") `
    -RedirectStandardError (Join-Path $LOGS "llama-server.err") `
    -WindowStyle Hidden -PassThru
Write-Host "      llama-server PID: $($llamaProc.Id)"

# 等待 llama-server 健康检查通过 (模型加载上 GPU 需要一些时间)
Write-Host "[2/3] 等待 llama-server 就绪..."
$ready = $false
foreach ($i in 1..120) {
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$LLAMA_PORT/health" -TimeoutSec 2
        if ($h.status -eq "ok") { $ready = $true; break }
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) {
    Write-Host "      llama-server 未就绪, 查看日志: $LOGS\llama-server.err" -ForegroundColor Red
    exit 1
}
Write-Host "      llama-server 就绪 (等待 $($i)s)"

# 启动决策头包装服务 (需以模型目录为工作目录, startlux_decision 包在其下)
Write-Host "[3/3] 启动决策头服务 /v1/systemone (端口 $DECISION_PORT)..."
$decProc = Start-Process -FilePath $PY `
    -ArgumentList @("-m", "startlux_decision.gguf_server", "--model-dir", ".", "--llama", "http://127.0.0.1:$LLAMA_PORT", "--port", "$DECISION_PORT") `
    -WorkingDirectory $MODEL `
    -RedirectStandardOutput (Join-Path $LOGS "decision.log") `
    -RedirectStandardError (Join-Path $LOGS "decision.err") `
    -WindowStyle Hidden -PassThru
Write-Host "      decision-server PID: $($decProc.Id)"

Start-Sleep -Seconds 1
$decReady = $false
foreach ($j in 1..60) {
    try {
        $st = Invoke-RestMethod -Uri "http://127.0.0.1:$DECISION_PORT/" -TimeoutSec 2
        $decReady = $true; break
    } catch { Start-Sleep -Seconds 1 }
}
if ($decReady) {
    Write-Host "OK 决策服务已就绪: http://127.0.0.1:$DECISION_PORT/v1/systemone  (model: $($st.model), 等待 $($j)s)" -ForegroundColor Green
} else {
    Write-Host "决策服务未响应, 查看日志: $LOGS\decision.err" -ForegroundColor Red
    exit 1
}
