# 最小调用示例: 向 StartLux-Decision 决策服务发一次 text2sql 路由决策
# 前提: 已运行 scripts\start_server.ps1
$ErrorActionPreference = "Stop"

$body = @{
    state = @{
        user_query = "查一下上个月每个商品类目的销售额和销量排名"
        dialect    = "MySQL"
    }
    questions = @{
        route = @{
            type         = "choice"                                    # choice=多选一 / noul=是否 / score=评分
            instructions = "该用户问题应路由到哪条处理管线？"
            criteria     = @{
                text2sql = "可用单条 SQL 直接回答的结构化数据查询"
                rag      = "主要依赖业务知识库文档（规则、流程）才能回答"
                hybrid   = "既要查数据库又要检索知识库"
                reject   = "与业务无关、超范围或有风险的请求"
            }
        }
        need_retrieval = @{
            type         = "noul"
            instructions = "回答前是否需要先检索数据库 schema 与业务知识库？"
        }
        complexity = @{
            type         = "score"
            instructions = "该问题对应的 SQL 复杂度评分（0 简单 ... 4 很复杂）"
            criteria     = @("单表简单查询", "多表连接", "嵌套子查询", "多步骤分析", "需要窗口函数或复杂逻辑")
        }
    }
} | ConvertTo-Json -Depth 6

# PS5.1 里直接传字符串body中文会乱码, 必须显式按 UTF-8 编码成字节
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$resp = Invoke-RestMethod -Uri "http://127.0.0.1:9090/v1/systemone" -Method Post -Body $bytes -ContentType "application/json"

Write-Host "`n===== 原始返回 =====" -ForegroundColor Cyan
$resp | ConvertTo-Json -Depth 6

Write-Host "`n===== 重点字段 =====" -ForegroundColor Cyan
Write-Host "路由: $($resp.answers.route.choice)  (置信度 $([math]::Round($resp.answers.route.confidence, 3)))"
Write-Host "各选项概率:"
$resp.answers.route.probabilities.PSObject.Properties | Sort-Object Value -Descending | ForEach-Object {
    Write-Host ("  {0,-10} {1}" -f $_.Name, [math]::Round($_.Value, 4))
}
Write-Host "需要检索? $($resp.answers.need_retrieval.noul)"
Write-Host "SQL复杂度评分: $([math]::Round($resp.answers.complexity.score, 2))"
