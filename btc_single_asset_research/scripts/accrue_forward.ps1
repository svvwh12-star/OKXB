# OKXB 多周期 forward 测试 — 无人值守累积。
# 顺序跑 BTC + ETH 的 evaluate, 各自把一行最新 verdict 追加到 forward/{asset}/forward_status.csv。
# 由 Windows 计划任务 "OKXB_Forward_Accrue" 每 4 小时调用一次。
# evaluate 已 fail-soft: 受限网络拒绝/空拉时本轮干净跳过(不崩); 数据只拉一次多周期复用(已提速)。
$ErrorActionPreference = "Continue"
# 自动定位项目根: 本脚本在 <root>\btc_single_asset_research\scripts\ -> 上溯两级 = 项目根。
# 这样换电脑/换 U 盘盘符 (F:/H:/...) 都不用改路径。
$root   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
# 自动找 python: 优先 PATH 上的 python/py, 都没有再退回常见安装路径。
$py     = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $py) { $py = "C:\Python313\python.exe" }
$script = Join-Path $root "btc_single_asset_research\scripts\run_forward_shadow.py"
$env:PYTHONPATH = Join-Path $root "src"
$env:LOKY_MAX_CPU_COUNT = "$([Environment]::ProcessorCount)"   # 静默 loky 无wmic的探测噪音(须在python启动前设)
$log    = Join-Path $root "btc_single_asset_research\forward\accrue.log"

"`n===== run $(Get-Date -Format o) =====" | Out-File -Append -FilePath $log -Encoding utf8
foreach ($a in @("btc", "eth")) {
    "--- $a evaluate ---" | Out-File -Append -FilePath $log -Encoding utf8
    & $py $script --asset $a --mode evaluate 2>&1 |
        Where-Object { $_ -match '^\[|evaluate |appended|跳过|为空|ConnectError|Error|Traceback' } |
        Out-File -Append -FilePath $log -Encoding utf8
}
# 股票日历候选 ST720 的 forward 评估(池化MU/SOXL/NVDA/TSLA, 与加密一并累积)
$stock = Join-Path $root "stock_calendar_research\scripts\run_stock_forward.py"
if (Test-Path $stock) {
    "--- stock ST720 evaluate ---" | Out-File -Append -FilePath $log -Encoding utf8
    & $py $stock --mode evaluate 2>&1 |
        Where-Object { $_ -match '^\[ST|->|Error|Traceback' } |
        Out-File -Append -FilePath $log -Encoding utf8
}
# 15/30min 日内均值回归 (IMR): 先采集最新bar, 再评估 (首次自动冻结; 公共行情免密钥)
$imr = Join-Path $root "intraday_mr_research\scripts\run_intraday_mr.py"
if (Test-Path $imr) {
    "--- IMR 15/30min collect+evaluate ---" | Out-File -Append -FilePath $log -Encoding utf8
    & $py $imr --mode collect 2>&1 |
        Where-Object { $_ -match '新增|freeze|链校验|collect|Error|Traceback' } |
        Out-File -Append -FilePath $log -Encoding utf8
    & $py $imr --mode evaluate 2>&1 |
        Where-Object { $_ -match 'IMR_|verdict|PENDING|KILL|PASS|Error|Traceback' } |
        Out-File -Append -FilePath $log -Encoding utf8
}
# AI 前向验证: auto(AI_FORWARD_AUTO 开才调AI耗token, 关则仅免费结算) + 评估
$aif = Join-Path $root "ai_forward_research\scripts\run_ai_forward.py"
if (Test-Path $aif) {
    "--- AI forward auto+evaluate ---" | Out-File -Append -FilePath $log -Encoding utf8
    & $py $aif --mode auto 2>&1 |
        Where-Object { $_ -match '记录|结算|冻结|关闭|链校验|Error|Traceback' } |
        Out-File -Append -FilePath $log -Encoding utf8
    & $py $aif --mode evaluate 2>&1 |
        Where-Object { $_ -match 'PENDING|KILL|PASS|net|IC|Error|Traceback' } |
        Out-File -Append -FilePath $log -Encoding utf8
}
"--- done $(Get-Date -Format o) ---" | Out-File -Append -FilePath $log -Encoding utf8
