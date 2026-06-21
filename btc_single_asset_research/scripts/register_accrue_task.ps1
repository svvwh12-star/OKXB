# 一次性注册计划任务 OKXB_Forward_Accrue —— 每 4 小时无人值守跑 BTC + ETH + 股票(ST720) 的 forward evaluate。
# 需【管理员】运行(注册计划任务要提权)。两种方式任选其一:
#   A) 右键本文件 → "使用 PowerShell 运行"(若弹 UAC 选是);
#   B) 开一个【管理员】PowerShell(Win+X → 终端(管理员)), 执行:
#        powershell -ExecutionPolicy Bypass -File "F:\临时\0609\OKXB\btc_single_asset_research\scripts\register_accrue_task.ps1"
# 设计为【一次成功】: 三种注册方法依次回退, 至少一种会成功; 末尾自检 + 立即试跑一次确认能真跑。
$ErrorActionPreference = "Stop"
$name = "OKXB_Forward_Accrue"
$runner = "F:\临时\0609\OKXB\btc_single_asset_research\scripts\accrue_forward.ps1"
$desc = "OKXB forward 累积 (BTC+ETH+股票ST720 evaluate, 每4h)"

# --- 前置自检 ---
$elev = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $elev) {
    Write-Host "未以管理员运行 —— 请用管理员 PowerShell 再跑一次本脚本(见文件顶部方式 B)。" -ForegroundColor Yellow
    Write-Host ('  powershell -ExecutionPolicy Bypass -File "' + $PSCommandPath + '"')
    exit 1
}
if (-not (Test-Path $runner)) {
    Write-Host "找不到运行器: $runner —— 路径不对, 先确认仓库位置。" -ForegroundColor Red
    exit 1
}

$arg = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $runner + '"'
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -MultipleInstances IgnoreNew
$anchor = (Get-Date).Date.AddMinutes(1)        # 今天 00:01, 作为锚点
$method = $null

# 方法1: -Once + 每4h重复(长 duration ≈ 永久)
if (-not $method) {
    try {
        $t = New-ScheduledTaskTrigger -Once -At $anchor -RepetitionInterval (New-TimeSpan -Hours 4) -RepetitionDuration (New-TimeSpan -Days 3650)
        Register-ScheduledTask -TaskName $name -Action $action -Trigger $t -Settings $settings -Description $desc -Force -ErrorAction Stop | Out-Null
        $method = "方法1 (-Once + 每4h, duration 3650天)"
    } catch { Write-Host "方法1 失败($($_.Exception.Message)); 试方法2..." -ForegroundColor DarkYellow }
}
# 方法2: Daily 锚 + 当日每4h重复(最通用, 每天自动续)
if (-not $method) {
    try {
        $t = New-ScheduledTaskTrigger -Daily -At $anchor
        $t.Repetition = (New-ScheduledTaskTrigger -Once -At $anchor -RepetitionInterval (New-TimeSpan -Hours 4) -RepetitionDuration (New-TimeSpan -Hours 24)).Repetition
        Register-ScheduledTask -TaskName $name -Action $action -Trigger $t -Settings $settings -Description $desc -Force -ErrorAction Stop | Out-Null
        $method = "方法2 (Daily + 当日每4h)"
    } catch { Write-Host "方法2 失败($($_.Exception.Message)); 试方法3(schtasks)..." -ForegroundColor DarkYellow }
}
# 方法3: schtasks /SC HOURLY /MO 4(原生"每4小时永久", 最后兜底)
if (-not $method) {
    $tr = 'powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File \"' + $runner + '\"'
    & schtasks.exe /Create /TN $name /TR $tr /SC HOURLY /MO 4 /ST 00:01 /F | Out-Null
    if ($LASTEXITCODE -eq 0) { $method = "方法3 (schtasks HOURLY /MO 4)" }
}

if (-not $method) {
    Write-Host "三种注册方法都失败了 —— 请把上面的报错发我。" -ForegroundColor Red
    exit 1
}

# --- 自检: 任务存在 + 下次运行时间 ---
Write-Host "OK: 已注册 $name (每 4 小时, 当前用户登录时执行) —— $method" -ForegroundColor Green
try {
    $info = Get-ScheduledTaskInfo -TaskName $name
    $st = (Get-ScheduledTask -TaskName $name).State
    Write-Host ("  状态={0}  下次运行={1}" -f $st, $info.NextRunTime)
} catch { Write-Host "  (任务已建, 但读取下次运行时间失败: $($_.Exception.Message))" -ForegroundColor DarkYellow }

# --- 立即试跑一次, 确认真能跑(不只是注册成功) ---
Write-Host "立即试跑一次以验证 (约 1 分钟, 后台静默)..." -ForegroundColor Cyan
try { Start-ScheduledTask -TaskName $name } catch { Write-Host "  Start 失败: $($_.Exception.Message)" -ForegroundColor DarkYellow }
Start-Sleep -Seconds 8
$log = "F:\临时\0609\OKXB\btc_single_asset_research\forward\accrue.log"
Write-Host "几分钟后看日志确认三段(btc/eth/stock)都跑了:" -ForegroundColor Cyan
Write-Host ('  Get-Content "' + $log + '" -Tail 25')
Write-Host "以后想停掉: Unregister-ScheduledTask $name -Confirm:`$false"
