# OKXB 多周期 forward 测试 — 无人值守累积。
# 顺序跑 BTC + ETH 的 evaluate, 各自把一行最新 verdict 追加到 forward/{asset}/forward_status.csv。
# 由 Windows 计划任务 "OKXB_Forward_Accrue" 每 4 小时调用一次。
# evaluate 已 fail-soft: 受限网络拒绝/空拉时本轮干净跳过(不崩); 数据只拉一次多周期复用(已提速)。
$ErrorActionPreference = "Continue"
$root   = "F:\临时\0609\OKXB"
$py     = "C:\Python313\python.exe"
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
"--- done $(Get-Date -Format o) ---" | Out-File -Append -FilePath $log -Encoding utf8
