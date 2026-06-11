# 一键打包 OKXB.exe (在项目根目录运行: powershell -ExecutionPolicy Bypass -File build_exe.ps1)
# 产物: dist\OKXB.exe (单文件, 双击运行)
$ErrorActionPreference = "Stop"
Write-Host "[1/3] 安装/确认依赖 ..." -ForegroundColor Cyan
python -m pip install --disable-pip-version-check `
    customtkinter pyinstaller httpx websockets aiosqlite sortedcontainers PyYAML python-dotenv | Out-Null
# DeepSeek / OpenAI 兼容 的 AI 事件分类只用 httpx (已含), 无需额外库。
# 仅当你想用 provider=claude 时才需要 anthropic, 取消下一行注释即可纳入 (体积略增)。
# python -m pip install anthropic | Out-Null

Write-Host "[2/3] 清理旧产物 ..." -ForegroundColor Cyan
if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path dist\OKXB.exe) { Remove-Item dist\OKXB.exe -Force }

Write-Host "[3/3] 打包 (约数分钟) ..." -ForegroundColor Cyan
python -m PyInstaller OKXB.spec --noconfirm

if (Test-Path dist\OKXB.exe) {
    $mb = [math]::Round((Get-Item dist\OKXB.exe).Length / 1MB, 1)
    Write-Host "完成! -> dist\OKXB.exe ($mb MB), 双击即可运行。" -ForegroundColor Green
} else {
    Write-Host "打包失败, 请检查上面的输出。" -ForegroundColor Red
}
