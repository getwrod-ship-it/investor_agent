# text2sql 记忆归零脚本：把 agent 积累的会话/审计/长期记忆全部清空，像全新部署一样
#   用法: powershell -ExecutionPolicy Bypass -File .\reset_memory.ps1
#   可选: -NoBackup   跳过备份直接删（默认先打包 data_backup_yyyyMMdd_HHmmss.zip 到项目根）
#
#   清理范围（仅 agent 记忆/历史）:
#     - data\checkpoints.db(+ -wal/-shm)   LangGraph 会话 checkpoint（含滚动摘要）
#     - data\system.db                     query_run / chat_session 等审计与会话列表
#     - data\mem0_*.db(+ -wal/-shm)        mem0 长期记忆写入审计
#     - Qdrant: text2sql_memory_facts / text2sql_memory_episodic / mem0migrations
#   不动（业务数据）:
#     MySQL litemall 业务库、Qdrant text2sql_schema / text2sql_docs、data\litemall\
#
#   清理后用 start_all.ps1 重新启动即可，空库/空 collection 会在启动和首次使用时自动重建。

param([switch]$NoBackup)

$ErrorActionPreference = "Continue"
$ROOT = $PSScriptRoot
$DATA = Join-Path $ROOT "data"

function Test-Http($url) {
    try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true }
    catch { return $false }
}

function Stop-ByPort($port, $name) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $conns) {
        Write-Host "  $name :$port 未在运行" -ForegroundColor DarkGray
        return
    }
    $conns | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
        try {
            Stop-Process -Id $_ -Force -ErrorAction Stop
            Write-Host "  已停止 $name（PID $_）" -ForegroundColor Green
        } catch {
            Write-Host "  停止 $name（PID $_）失败：$_" -ForegroundColor Yellow
        }
    }
}

Write-Host "==== text2sql 记忆归零 ====" -ForegroundColor Cyan

# ---- [1/4] 停后端（避免 sqlite 文件被占用；MySQL/Qdrant 不动） ----
Write-Host "[1/4] 停止后端 :8888"
Stop-ByPort 8888 "后端"

# ---- [2/4] 备份并删除记忆库文件 ----
Write-Host "[2/4] 清理 data\ 下的记忆库文件"
$targets = @(Get-ChildItem $DATA -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -match '^(checkpoints\.db|system\.db|mem0_[a-z_]+\.db)(-(wal|shm))?$'
})
if ($targets.Count -eq 0) {
    Write-Host "  无记忆库文件（可能已清理过）" -ForegroundColor DarkGray
} else {
    if (-not $NoBackup) {
        $zip = Join-Path $ROOT ("data_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".zip")
        Compress-Archive -Path $targets.FullName -DestinationPath $zip -Force
        Write-Host ("  已备份 {0} 个文件 -> {1}" -f $targets.Count, $zip) -ForegroundColor Green
    }
    $targets | Remove-Item -Force
    Write-Host ("  已删除: " + (($targets | Select-Object -ExpandProperty Name) -join ", ")) -ForegroundColor Green
}

# ---- [3/4] 删除 Qdrant 记忆 collection（业务 collection 不在名单里） ----
Write-Host "[3/4] 清理 Qdrant 记忆 collection"
if (Test-Http "http://localhost:6333/collections") {
    foreach ($c in @("text2sql_memory_facts", "text2sql_memory_episodic", "mem0migrations")) {
        try {
            Invoke-RestMethod -Method Delete -Uri "http://localhost:6333/collections/$c" | Out-Null
            Write-Host "  已删除 $c" -ForegroundColor Green
        } catch {
            Write-Host "  跳过 $c（不存在或删除失败）" -ForegroundColor DarkGray
        }
    }
    $left = (Invoke-RestMethod -Uri "http://localhost:6333/collections").result.collections.name
    Write-Host ("  剩余 collections: " + ($left -join ", ")) -ForegroundColor DarkGray
} else {
    Write-Host "  Qdrant 未运行，已跳过（启动 Qdrant 后重跑本脚本可补删）" -ForegroundColor Yellow
}

# ---- [4/4] 完成 ----
Write-Host "==== 记忆已归零（MySQL 业务库 / 业务知识库 / data\litemall\ 均未动） ====" -ForegroundColor Cyan
Write-Host "  重新开始: powershell -ExecutionPolicy Bypass -File .\start_all.ps1"
