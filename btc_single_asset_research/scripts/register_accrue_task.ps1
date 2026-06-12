# 一次性注册计划任务 OKXB_Forward_Accrue —— 每 4 小时无人值守跑 BTC+ETH 的 forward evaluate。
# 需【管理员】运行(注册计划任务要提权)。两种方式任选其一:
#   A) 右键本文件 → "使用 PowerShell 运行"(若弹 UAC 选是);
#   B) 开一个【管理员】PowerShell(Win+X → 终端(管理员)), 执行:
#        powershell -ExecutionPolicy Bypass -File "F:\临时\0609\OKXB\btc_single_asset_research\scripts\register_accrue_task.ps1"
$ErrorActionPreference = "Stop"
$elev = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $elev) {
    Write-Host "未以管理员运行 —— 请用管理员 PowerShell 再跑一次本脚本(见文件顶部方式 B)。" -ForegroundColor Yellow
    exit 1
}
$runner = "F:\临时\0609\OKXB\btc_single_asset_research\scripts\accrue_forward.ps1"
$arg = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $runner + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg
$anchor = (Get-Date).Date
$trigger = New-ScheduledTaskTrigger -Once -At $anchor -RepetitionInterval (New-TimeSpan -Hours 4) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "OKXB_Forward_Accrue" -Action $action -Trigger $trigger -Settings $settings -Description "OKXB 多周期 forward 累积 (BTC+ETH evaluate, 每4h)" -Force | Out-Null
Write-Host "OK: 已注册计划任务 OKXB_Forward_Accrue (每 4 小时跑一次, 当前用户登录时执行)。" -ForegroundColor Green
Get-ScheduledTask -TaskName "OKXB_Forward_Accrue" | Select-Object TaskName, State | Format-List
Write-Host '立即测试一次(可选): Start-ScheduledTask OKXB_Forward_Accrue'
Write-Host '看运行日志:           Get-Content "F:\临时\0609\OKXB\btc_single_asset_research\forward\accrue.log" -Tail 40'
Write-Host '以后想停掉:           Unregister-ScheduledTask OKXB_Forward_Accrue -Confirm:$false'
