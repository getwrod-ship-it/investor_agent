# text2sql 一键停止脚本：前端 → 后端 → 决策服务 → Qdrant（MySQL 是系统服务，不动）
#   用法: powershell -ExecutionPolicy Bypass -File .\stop_all.ps1

$ErrorActionPreference = "Continue"
$ROOT = $PSScriptRoot
$DECISION_STOP = Join-Path (Split-Path -Parent $ROOT) "startlux-decision\scripts\stop_server.ps1"

function Stop-ByPort($port, $name) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "  $name :$port 未在运行" -ForegroundColor DarkGray
        return
    }
    $conns | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        $procId = $_
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "  已停止 $name（PID $procId）" -ForegroundColor Green
        } catch {
            Write-Host "  停止 $name（PID $procId）失败：$_" -ForegroundColor Yellow
        }
    }
}

Write-Host "==== text2sql 一键停止 ====" -ForegroundColor Cyan

Write-Host "[1/4] 停止前端 :8889"
Stop-ByPort 8889 "前端"

Write-Host "[2/4] 停止后端 :8888"
Stop-ByPort 8888 "后端"

Write-Host "[3/4] 停止决策服务 :9081/:9090"
if (Test-Path $DECISION_STOP) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $DECISION_STOP
} else {
    Stop-ByPort 9090 "决策头"
    Stop-ByPort 9081 "llama-server"
}

Write-Host "[4/4] 停止 Qdrant 容器"
Push-Location $ROOT
docker compose stop qdrant
Pop-Location

Write-Host "==== 已全部停止（MySQL 系统服务保持原样） ====" -ForegroundColor Cyan
