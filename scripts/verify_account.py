#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OKXB · 账户与权限校验脚本 (独立、零 SDK 依赖)
================================================
轮换密钥后在本地运行:  python scripts/verify_account.py

作用 (全部为只读 GET 请求, 不下单、不动资金):
  1. 验证密钥是否有效、权限是否为 读取/交易 (且未开提现)
  2. 显示 API key 绑定的 IP (未绑定会高亮警告)
  3. 显示账户模式 (acctLv) / 持仓模式 (posMode) / 保证金模式
  4. 显示账户权益与当前持仓
  5. 探测可交易合约 (加密永续 + 股票永续是否可见/账户可访问)
  6. 检查本地时钟与 OKX 服务器的偏移 (高频签名/下单对时钟敏感)

安全:
  - 密钥从 .env 读取, 脚本内不硬编码, 也绝不打印密钥明文。
  - 默认走 OKXB_MODE 指定的环境 (demo / live)。

依赖:  pip install requests python-dotenv
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("缺少依赖: pip install requests")

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit("缺少依赖: pip install python-dotenv")

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# 区域路由: 账户所属地区决定 base host (用错会鉴权/路由失败)
REST_HOSTS = {"global": "https://www.okx.com", "us": "https://us.okx.com", "eea": "https://eea.okx.com"}
BASE_URL = REST_HOSTS.get(os.getenv("OKX_REGION", "global").strip().lower(), REST_HOSTS["global"])

# ----- 颜色 (Windows 终端通常支持 ANSI; 不支持也只是多几个字符) -----
G, R, Y, B, X = "\033[92m", "\033[91m", "\033[93m", "\033[94m", "\033[0m"


def _ok(s):   print(f"{G}[OK]{X}   {s}")
def _warn(s): print(f"{Y}[警告]{X} {s}")
def _err(s):  print(f"{R}[错误]{X} {s}")
def _info(s): print(f"{B}[ ]{X}   {s}")
def _hr():    print("-" * 64)


def _timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sign(secret: str, ts: str, method: str, path: str, body: str = "") -> str:
    msg = f"{ts}{method}{path}{body}"
    mac = hmac.new(secret.encode(), msg.encode(), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()


def _load_keys():
    mode = os.getenv("OKXB_MODE", "demo").strip().lower()
    if mode == "live":
        ak = os.getenv("OKX_LIVE_API_KEY", "")
        sk = os.getenv("OKX_LIVE_SECRET_KEY", "")
        pp = os.getenv("OKX_LIVE_PASSPHRASE", "")
    else:
        mode = "demo"
        ak = os.getenv("OKX_DEMO_API_KEY", "")
        sk = os.getenv("OKX_DEMO_SECRET_KEY", "")
        pp = os.getenv("OKX_DEMO_PASSPHRASE", "")
    missing = [n for n, v in [("api_key", ak), ("secret_key", sk), ("passphrase", pp)] if not v]
    if missing:
        _err(f"{mode} 模式缺少: {', '.join(missing)} — 请先在 .env 填入【轮换后】的新密钥。")
        sys.exit(1)
    return mode, ak, sk, pp


def _request(method, path, mode, ak, sk, pp, params=None, auth=True):
    url = BASE_URL + path
    if params:
        from urllib.parse import urlencode
        qs = "?" + urlencode(params)
        path_for_sign = path + qs
        url = url + qs
    else:
        path_for_sign = path
    headers = {"Content-Type": "application/json"}
    if mode == "demo":
        headers["x-simulated-trading"] = "1"
    if auth:
        ts = _timestamp()
        headers.update({
            "OK-ACCESS-KEY": ak,
            "OK-ACCESS-SIGN": _sign(sk, ts, method, path_for_sign, ""),
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": pp,
        })
    resp = requests.request(method, url, headers=headers, timeout=15)
    try:
        data = resp.json()
    except Exception:
        _err(f"非 JSON 响应 ({resp.status_code}): {resp.text[:200]}")
        sys.exit(1)
    return data


def check_clock(mode):
    _info("检查时钟偏移 ...")
    try:
        local_ms = int(time.time() * 1000)
        data = _request("GET", "/api/v5/public/time", mode, "", "", "", auth=False)
        server_ms = int(data["data"][0]["ts"])
        skew = abs(server_ms - local_ms)
        if skew < 1000:
            _ok(f"时钟偏移 {skew} ms (正常)")
        elif skew < 5000:
            _warn(f"时钟偏移 {skew} ms — 建议开启系统自动对时 (NTP), 高频签名敏感。")
        else:
            _err(f"时钟偏移 {skew} ms 过大 — 可能导致签名失败/下单被拒, 请同步系统时间!")
    except Exception as e:
        _warn(f"时钟检查失败: {e}")


def check_account(mode, ak, sk, pp):
    _info("读取账户配置 (/api/v5/account/config) ...")
    data = _request("GET", "/api/v5/account/config", mode, ak, sk, pp)
    if data.get("code") != "0":
        _err(f"鉴权失败 code={data.get('code')} msg={data.get('msg')}")
        _err("常见原因: 密钥/passphrase 错误、IP 未在白名单、demo/live 模式与密钥不匹配。")
        sys.exit(1)

    cfg = data["data"][0]
    _ok("密钥有效, 鉴权通过。")

    perm = cfg.get("perm", "")
    _hr()
    _info(f"权限 (perm): {perm}")
    if "trade" in perm:
        _ok("含交易权限 (trade)")
    else:
        _warn("无交易权限 — 只能读取, 无法下单。")
    if "withdraw" in perm:
        _err("!! 含提现权限 (withdraw) — 强烈建议去 OKX 关闭提现权限 !!")
    else:
        _ok("无提现权限 (安全)")

    ip = cfg.get("ip", "")
    if ip:
        _ok(f"已绑定 IP 白名单: {ip}")
    else:
        _warn("API key 未绑定 IP — 高危! 请在 OKX 后台绑定本机/VPS 出口 IP。")

    acct_lv = {"1": "现货", "2": "现货+合约", "3": "跨币种保证金", "4": "组合保证金", "5": "组合保证金(Portfolio)"}.get(cfg.get("acctLv"), cfg.get("acctLv"))
    pos_mode = cfg.get("posMode")
    _hr()
    _info(f"账户层级 acctLv = {cfg.get('acctLv')} ({acct_lv})")
    _info(f"持仓模式 posMode = {pos_mode}  (期望: net_mode 或 long_short_mode)")
    _info(f"KYC 等级 kycLv = {cfg.get('kycLv')}")
    if cfg.get("acctLv") in ("1",):
        _warn("账户层级为'现货', 可能无法开合约 — 需在 OKX 升级账户模式。")
    return cfg


def check_balance(mode, ak, sk, pp):
    _hr()
    _info("读取账户权益 (/api/v5/account/balance) ...")
    data = _request("GET", "/api/v5/account/balance", mode, ak, sk, pp)
    if data.get("code") != "0":
        _warn(f"读取余额失败: {data.get('msg')}")
        return
    d = data["data"][0]
    total = d.get("totalEq", "?")
    _ok(f"账户总权益 totalEq = {total} USD")
    for det in d.get("details", []):
        if float(det.get("eq", 0) or 0) != 0:
            _info(f"  {det.get('ccy')}: eq={det.get('eq')} avail={det.get('availBal')}")


def check_positions(mode, ak, sk, pp):
    _hr()
    _info("读取当前持仓 (/api/v5/account/positions) ...")
    data = _request("GET", "/api/v5/account/positions", mode, ak, sk, pp)
    if data.get("code") != "0":
        _warn(f"读取持仓失败: {data.get('msg')}")
        return
    poss = [p for p in data.get("data", []) if float(p.get("pos", 0) or 0) != 0]
    if not poss:
        _ok("当前无持仓。")
    else:
        _warn(f"当前有 {len(poss)} 个持仓:")
        for p in poss:
            _info(f"  {p.get('instId')} pos={p.get('pos')} avgPx={p.get('avgPx')} upl={p.get('upl')}")


def check_instruments(mode, ak, sk, pp):
    """探测可交易合约: 加密永续 + 股票永续是否可见。"""
    _hr()
    _info("探测可交易合约 (/api/v5/public/instruments?instType=SWAP) ...")
    data = _request("GET", "/api/v5/public/instruments", mode, ak, sk, pp,
                    params={"instType": "SWAP"})
    if data.get("code") != "0":
        _warn(f"读取合约失败: {data.get('msg')}")
        return
    insts = data.get("data", [])
    live = [i for i in insts if i.get("state") == "live"]
    _ok(f"SWAP 合约共 {len(insts)} 个 (live: {len(live)})")

    # 加密永续抽查
    ids = {i["instId"] for i in live}
    for sym in ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"):
        if sym in ids:
            spec = next(i for i in live if i["instId"] == sym)
            _ok(f"  {sym}: tickSz={spec.get('tickSz')} ctVal={spec.get('ctVal')} lever<= {spec.get('lever')}")
        else:
            _warn(f"  {sym}: 未在 SWAP 列表中找到")

    # 股票永续探测 (symbol 格式 / instType 待核验)
    _hr()
    _info("探测股票永续 (格式/instType 以 OKX 现状为准) ...")
    stock_tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL"]
    found = []
    for tk in stock_tickers:
        matches = [i["instId"] for i in live if tk in i["instId"].upper() and "USDT" in i["instId"].upper()
                   and not i["instId"].upper().startswith(("BTC", "ETH", "SOL"))]
        if matches:
            found.append((tk, matches[:3]))
    if found:
        _ok("在 SWAP 列表中疑似找到股票永续:")
        for tk, ms in found:
            _info(f"  {tk}: {ms}")
    else:
        _warn("SWAP 列表未匹配到股票永续 — 可能: (a) 使用了独立 instType; "
              "(b) 你的账户所在地区不支持股票永续; (c) 需单独开通。")
        _warn("=> 后台调研会确认 OKX 股票永续的当前 instType/symbol 与地区准入。"
              "若不可交易, 将仅启用加密永续策略。")


def main():
    print()
    _hr()
    print("  OKXB 账户与权限校验")
    _hr()
    mode, ak, sk, pp = _load_keys()
    _info(f"运行模式: {mode.upper()}  ({'模拟盘' if mode == 'demo' else '实盘'})")
    _info(f"API key 末4位: ...{ak[-4:]}  (仅显示尾号以确认是哪把钥匙)")
    _hr()
    check_clock(mode)
    _hr()
    check_account(mode, ak, sk, pp)
    check_balance(mode, ak, sk, pp)
    check_positions(mode, ak, sk, pp)
    check_instruments(mode, ak, sk, pp)
    _hr()
    _ok("校验完成。若有红色[错误]/黄色[警告], 请先处理再进入下一步。")
    print()


if __name__ == "__main__":
    main()
