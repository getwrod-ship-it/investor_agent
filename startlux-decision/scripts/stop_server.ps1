# 停止 StartLux-Decision 决策服务 (llama-server + gguf_server)
$ports = @(9081, 9090)
foreach ($p in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        $procId = $c.OwningProcess
        $name = (Get-Process -Id $procId -ErrorAction SilentlyContinue).ProcessName
        Write-Host "停止端口 $p 上的进程: $name (PID $procId)"
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "✓ 服务已停止"
