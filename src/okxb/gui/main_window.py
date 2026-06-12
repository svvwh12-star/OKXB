"""OKXB 主窗口 (CustomTkinter)。

布局: 顶部(模式切换/演练开关/启动停止/状态) + 三标签页(控制台/账户与密钥/日志)。
安全: 默认"只演练不下单"; 实盘真金下单需勾选 + 二次确认弹窗。
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from .. import paths
from . import multiperiod
from .controller import TOOLTIPS
from .controller import (PARAM_DEFS, EngineController, account_brief_sync, ai_analyze_sync,
                         ai_pick_sync, apply_calibration_sync, apply_params, calib_files_info,
                         current_params, delete_preset, get_preset, list_presets,
                         manual_bracket_sync, manual_cancel_all_sync, manual_close_all_sync,
                         manual_close_sync, manual_place_sync, manual_set_leverage_sync,
                         read_env, run_calibration_sync, save_preset, stock_symbol_set,
                         verify_ai_sync, verify_edgar_sync, verify_finnhub_sync,
                         verify_okx_sync, verify_telegram_sync, write_env)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT = ("Microsoft YaHei UI", 13)
FONT_B = ("Microsoft YaHei UI", 15, "bold")
FONT_BIG = ("Microsoft YaHei UI", 22, "bold")
MONO = ("Consolas", 12)

GREEN, RED, GREY, AMBER = "#27c08a", "#f0616d", "#8a8d93", "#e0a800"
TXT = "#d6d6d6"   # 普通文字色 (CustomTkinter 不接受 text_color=None)


class _ToolTip:
    """鼠标悬停显示讲解 (小白向参数说明)。用原生 tk.Toplevel, 稳定可靠。"""
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text or ""
        self.tip = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _e=None):
        if self.tip or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 24
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, justify="left", bg="#10243a", fg="#eaf2ff",
                 font=("Microsoft YaHei UI", 10), wraplength=460, relief="solid",
                 borderwidth=1, padx=8, pady=6).pack()

    def _hide(self, _e=None):
        if self.tip:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


class OKXBApp(ctk.CTk):
    # 提供商显示名 -> AI_PROVIDER 值
    PROVIDER_MAP = [("规则(免费)", "rule"), ("DeepSeek", "deepseek"),
                    ("OpenAI兼容", "openai_compatible"), ("Claude", "claude")]
    # 选型策略显示名 -> AI_TIER_POLICY 值
    TIER_MAP = [("自动(按难度)", "auto"), ("只用Flash(省钱)", "flash"), ("只用Pro(最强)", "pro")]
    # 手动下单类型显示名 -> ordType
    ORD_TYPE_MAP = [("只挂单(maker)", "post_only"), ("限价", "limit"), ("市价IOC", "optimal_limit_ioc")]

    def __init__(self):
        super().__init__()
        self.title("OKXB · 自动量化交易")
        self.geometry("1040x720")
        self.minsize(960, 640)

        self.ctrl = EngineController()
        self._table_rows: dict[str, dict] = {}
        self._built_universe: tuple = ()
        self._calib_profit_cfg = None
        self._calib_stable_cfg = None
        self._ai_last = None            # 最近一次 AI 结构化结果 (供一键导入手动交易)
        self._tick_count = 0
        self._orders_busy = False
        self._acct_brief = None       # 控制台卡片真实账户快照 (独立于引擎)
        self._acct_busy = False
        self._sort_col = "long"          # 控制台表格排序列(行字典键)
        self._sort_desc = True           # True=由大到小
        self._table_order = None         # 上次packing签名, 避免无谓重排
        self._hdr_buttons: dict = {}     # 列键 -> (按钮, 基础名)
        try:
            self._stock_set = stock_symbol_set()
        except Exception:
            self._stock_set = set()

        self._build_header()
        self._tabs = ctk.CTkTabview(self, width=1000, height=560)
        self._tabs.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._tab_console = self._tabs.add("控制台")
        self._tab_manual = self._tabs.add("手动交易")
        self._tab_calib = self._tabs.add("策略校准")
        self._tab_cred = self._tabs.add("账户与密钥")
        self._tab_log = self._tabs.add("日志")
        self._tab_multi = self._tabs.add("多周期研究")
        self._build_console_tab()
        self._build_manual_tab()
        self._build_calib_tab()
        self._build_credentials_tab()
        self._build_logs_tab()
        self._build_multi_tab()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(800, self._refresh)

    # ----------------- 顶部 -----------------

    def _build_header(self) -> None:
        bar = ctk.CTkFrame(self, height=64)
        bar.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(bar, text="OKXB", font=FONT_BIG).pack(side="left", padx=(16, 8))
        ctk.CTkLabel(bar, text="OKX 股票永续 + 加密永续 自动量化",
                     font=FONT, text_color=GREY).pack(side="left", padx=(0, 16))

        self.start_btn = ctk.CTkButton(bar, text="▶ 启动", font=FONT_B, width=110,
                                       fg_color=GREEN, hover_color="#1fa476",
                                       command=self._on_start)
        self.start_btn.pack(side="right", padx=(8, 16))
        self.stop_btn = ctk.CTkButton(bar, text="■ 停止", font=FONT_B, width=90,
                                      fg_color=RED, hover_color="#c94d57",
                                      command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="right", padx=8)

        self.status_dot = ctk.CTkLabel(bar, text="● 已停止", font=FONT_B, text_color=GREY)
        self.status_dot.pack(side="right", padx=16)

        # 演练/实盘开关
        self.live_switch = ctk.CTkSwitch(bar, text="实际下单", font=FONT,
                                         command=self._on_live_toggle)
        self.live_switch.pack(side="right", padx=8)

        # 模式切换 虚拟盘/实盘
        self.mode_seg = ctk.CTkSegmentedButton(
            bar, values=["虚拟盘", "实盘"], font=FONT_B, command=self._on_mode_change)
        env = read_env()
        self.mode_seg.set("实盘" if env.get("OKXB_MODE") == "live" else "虚拟盘")
        self.mode_seg.pack(side="right", padx=8)
        ctk.CTkLabel(bar, text="模式:", font=FONT).pack(side="right", padx=(8, 0))

    # ----------------- 控制台 -----------------

    def _build_console_tab(self) -> None:
        t = self._tab_console
        cards = ctk.CTkFrame(t, fg_color="transparent")
        cards.pack(fill="x", padx=8, pady=8)
        self._cards: dict[str, ctk.CTkLabel] = {}
        for key, title in [("equity", "账户权益(USDT)"), ("day_pnl", "浮动盈亏(USDT)·点看统计"),
                           ("state", "系统状态"), ("positions", "持仓数(实时)·点查看"),
                           ("data_age_ms", "数据延迟(ms)")]:
            c = ctk.CTkFrame(cards, corner_radius=10)
            c.pack(side="left", expand=True, fill="x", padx=6)
            tl = ctk.CTkLabel(c, text=title, font=FONT, text_color=GREY)
            tl.pack(pady=(10, 0))
            v = ctk.CTkLabel(c, text="--", font=FONT_B)
            v.pack(pady=(0, 10))
            self._cards[key] = v
            if key == "positions":      # 点『持仓数』卡片 -> 跳手动交易看全部持仓
                for w in (c, tl, v):
                    w.bind("<Button-1>", lambda e: self._goto_positions())
            if key == "day_pnl":        # 点『浮动盈亏』卡片 -> 日/周/月/季 盈亏统计
                for w in (c, tl, v):
                    w.bind("<Button-1>", lambda e: self._show_pnl_stats())

        pbar = ctk.CTkFrame(t, fg_color="transparent")
        pbar.pack(fill="x", padx=14, pady=(6, 0))
        ctk.CTkLabel(pbar, text="范围(过滤列表&选品):", font=FONT, anchor="w").pack(side="left")
        self.pick_pool = ctk.CTkSegmentedButton(pbar, values=["加密", "美股", "全部"], font=FONT,
                                                width=160, command=self._pool_changed)
        self.pick_pool.set("全部"); self.pick_pool.pack(side="left", padx=8)
        ctk.CTkButton(pbar, text="🤖 AI选品推荐", font=FONT_B, width=120, fg_color="#7a5cff",
                      command=self._ai_pick_run).pack(side="left", padx=4)
        ctk.CTkLabel(pbar, text="(点列名可排序↑↓)", font=("Microsoft YaHei UI", 11),
                     text_color=GREY).pack(side="left", padx=8)
        ctk.CTkButton(pbar, text="⚠ 一键全平", font=FONT_B, width=100, fg_color="#a83232",
                      hover_color="#922", command=self._console_close_all).pack(side="right", padx=6)

        ctk.CTkLabel(t, text="实时行情 / 信号 (达入场门槛+确认+连续N拍才触发; 点某行可带入『手动交易』)",
                     font=FONT_B, anchor="w").pack(fill="x", padx=14, pady=(6, 2))
        ctk.CTkLabel(
            t, text="列说明: 中间价=最新中价 · 价差bps=买卖价差(万分之) · OBI_z/OFI_z=盘口/订单流"
                    "失衡标准分(正=偏多,负=偏空) · 多/空=方向信号分, 多+空=100、50=中性、≥66高亮 · "
                    "质量=行情质量/可交易环境(0–100, 流动性×波动; 高≠该交易, 只代表好执行) · "
                    "入场=该标的入场状态(✓候选/方向不足/质量低/未确认/死盘/预热/价差宽) · 持仓=●持有。"
                    "真正开仓还需在『✓候选』基础上 连续N拍+冷却+净edge。点列名排序(再点切升/降)。",
            font=("Microsoft YaHei UI", 11), text_color=GREY, anchor="w",
            wraplength=1020, justify="left").pack(fill="x", padx=14, pady=(0, 4))
        self._table = ctk.CTkScrollableFrame(t, height=320)
        self._table.pack(fill="both", expand=True, padx=8, pady=4)
        self._build_table_header()

    # 列定义: (显示名, 行字典键, 宽度)
    TABLE_COLS = [("标的", "inst", 140), ("中间价", "mid", 100), ("价差bps", "spread_bps", 72),
                  ("OBI_z", "obi_z", 64), ("OFI_z", "ofi_z", 64), ("多", "long", 52),
                  ("空", "short", 52), ("质量", "trad", 52), ("入场", "entry", 80), ("持仓", "has_pos", 54)]

    def _build_table_header(self) -> None:
        self._col_widths = [w for _, _, w in self.TABLE_COLS]
        hdr = ctk.CTkFrame(self._table, fg_color="#2b2b2b")
        hdr.pack(fill="x", pady=(0, 2))
        for name, key, w in self.TABLE_COLS:
            b = ctk.CTkButton(hdr, text=name, font=FONT_B, width=w, height=24, anchor="w",
                              fg_color="#2b2b2b", hover_color="#3a3a3a", text_color=TXT,
                              command=lambda k=key: self._set_sort(k))
            b.pack(side="left", padx=2)
            self._hdr_buttons[key] = (b, name)
        self._update_header_arrows()

    def _set_sort(self, key: str) -> None:
        if self._sort_col == key:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = key
            self._sort_desc = (key != "inst")     # 标的默认升序, 其余默认降序
        self._table_order = None                  # 强制重排
        self._update_header_arrows()
        self._apply_table_view(self.ctrl.rows())

    def _update_header_arrows(self) -> None:
        for key, (b, name) in self._hdr_buttons.items():
            if key == self._sort_col:
                b.configure(text=name + (" ↓" if self._sort_desc else " ↑"), text_color="#7fb0ff")
            else:
                b.configure(text=name, text_color=TXT)

    def _pool_changed(self, _v=None) -> None:
        self._table_order = None
        self._apply_table_view(self.ctrl.rows())

    def _inst_in_pool(self, inst: str) -> bool:
        pool = self.pick_pool.get()
        if pool == "全部":
            return True
        is_stock = inst.split("-")[0].upper() in self._stock_set
        return is_stock if pool == "美股" else (not is_stock)

    def _ensure_table_rows(self, insts: list[str]) -> None:
        if tuple(insts) == self._built_universe:
            return
        for r in self._table_rows.values():
            r["frame"].destroy()
        self._table_rows.clear()
        for inst in insts:
            fr = ctk.CTkFrame(self._table, fg_color="transparent")    # 先不pack, 由_apply排布
            labels = {}
            keys = ["inst", "mid", "spread", "obi", "ofi", "long", "short", "trad", "entry", "pos"]
            for k, w in zip(keys, self._col_widths):
                lb = ctk.CTkLabel(fr, text="-", font=MONO, width=w, anchor="w")
                lb.pack(side="left", padx=2)
                labels[k] = lb
            labels["inst"].configure(text=inst, font=FONT)
            handler = lambda e, i=inst: self._on_row_click(i)   # 点行 -> 带入手动交易
            fr.bind("<Button-1>", handler)
            for lb in labels.values():
                lb.bind("<Button-1>", handler)
            self._table_rows[inst] = {"frame": fr, "labels": labels}
        self._built_universe = tuple(insts)
        self._table_order = None

    def _apply_table_view(self, rows: dict) -> None:
        """按当前 范围过滤 + 排序列 重新排布行 (仅在顺序变化时重排, 减少闪烁)。"""
        if not rows:
            return
        col = self._sort_col
        visible = [i for i in rows if i in self._table_rows and self._inst_in_pool(i)]

        def key_of(inst):
            if col == "inst":
                return inst or ""
            v = rows[inst].get(col)
            if col == "has_pos":
                return 1 if v else 0
            return v if v is not None else float("-inf")
        try:
            visible.sort(key=key_of, reverse=self._sort_desc)
        except TypeError:
            visible.sort(key=lambda i: str(key_of(i)), reverse=self._sort_desc)
        sig = (self.pick_pool.get(), col, self._sort_desc, tuple(visible))
        if sig == self._table_order:
            return
        self._table_order = sig
        for r in self._table_rows.values():
            r["frame"].pack_forget()
        for inst in visible:
            self._table_rows[inst]["frame"].pack(fill="x", pady=1)

    # ----------------- 策略校准 -----------------

    def _build_calib_tab(self) -> None:
        t = self._tab_calib
        ctk.CTkLabel(t, text="策略校准 与 参数管理", font=FONT_B, anchor="w").pack(
            fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(
            t, text="左=查看/手动调整每个参数 + 存取命名预设; 右=用录制数据回测找最优并一键应用。"
                    "『校准(最近录制)』只用最新一个 calib 文件(本次会话); 『校准(全部)』合并所有文件(更多数据)。"
                    "二者算法相同, 只是数据范围不同。",
            font=("Microsoft YaHei UI", 11), text_color=GREY, anchor="w",
            wraplength=1980, justify="left").pack(fill="x", padx=12)

        split = ctk.CTkFrame(t, fg_color="transparent")
        split.pack(fill="both", expand=True, padx=8, pady=4)
        left = ctk.CTkFrame(split, width=560); left.pack(side="left", fill="y", padx=(4, 6))
        left.pack_propagate(False)
        right = ctk.CTkFrame(split, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)

        # ---- 左: 参数管理 ----
        ctk.CTkLabel(left, text="参数 (当前值 · 可改后应用或存预设)", font=FONT_B,
                     anchor="w").pack(fill="x", padx=8, pady=(8, 2))
        pr = ctk.CTkFrame(left, fg_color="transparent"); pr.pack(fill="x", padx=8, pady=2)
        self.preset_menu = ctk.CTkOptionMenu(pr, values=list_presets(), font=FONT, width=210)
        self.preset_menu.pack(side="left", padx=(0, 4))
        ctk.CTkButton(pr, text="载入", font=FONT, width=54,
                      command=self._preset_load).pack(side="left", padx=2)
        ctk.CTkButton(pr, text="删除", font=FONT, width=54, fg_color="#a83232",
                      command=self._preset_delete).pack(side="left", padx=2)
        pr2 = ctk.CTkFrame(left, fg_color="transparent"); pr2.pack(fill="x", padx=8, pady=2)
        self.preset_name = ctk.CTkEntry(pr2, font=FONT, width=210, placeholder_text="预设名(如 我的宽松)")
        self.preset_name.pack(side="left", padx=(0, 4))
        ctk.CTkButton(pr2, text="保存为预设", font=FONT, width=110,
                      command=self._preset_save).pack(side="left", padx=2)

        pf = ctk.CTkScrollableFrame(left, height=300); pf.pack(fill="both", expand=True, padx=8, pady=4)
        ctk.CTkLabel(pf, text="把鼠标停在参数名上看详细讲解 ⓘ", font=("Microsoft YaHei UI", 11),
                     text_color="#7fb0ff", anchor="w").pack(fill="x", pady=(0, 2))
        self._param_entries = {}
        cur = current_params()
        for key, label, typ in PARAM_DEFS:
            row = ctk.CTkFrame(pf, fg_color="transparent"); row.pack(fill="x", pady=1)
            lab = ctk.CTkLabel(row, text="ⓘ " + label, font=("Microsoft YaHei UI", 11),
                               width=260, anchor="w", text_color="#cfe0ff")
            lab.pack(side="left")
            tip = TOOLTIPS.get(key, "")
            _ToolTip(lab, tip)
            e = ctk.CTkEntry(row, font=MONO, width=90)
            v = cur.get(key)
            e.insert(0, "" if v is None else str(v))
            e.pack(side="left", padx=4)
            _ToolTip(e, tip)
            self._param_entries[key] = (e, typ)

        ar = ctk.CTkFrame(left, fg_color="transparent"); ar.pack(fill="x", padx=8, pady=4)
        ctk.CTkButton(ar, text="✓ 应用到引擎(写config)", font=FONT_B, width=190, fg_color=GREEN,
                      command=self._params_apply).pack(side="left", padx=2)
        ctk.CTkButton(ar, text="↻ 重读", font=FONT, width=70,
                      command=self._params_reload).pack(side="left", padx=2)
        ctk.CTkLabel(left, text="想多出手/更满仓→载入『宽松(多出手)』, 或调低 入场分/确认/持续/edge + 调高 "
                     "同时持仓数/每笔风险预算; 改完点『应用』, 重启引擎生效。"
                     "注: 信号是间歇的, 调大并发只是'有信号时能多开', 不保证时刻满仓。",
                     font=("Microsoft YaHei UI", 11), text_color=GREY, anchor="w",
                     wraplength=540, justify="left").pack(fill="x", padx=8)

        # ---- 右: 校准 ----
        srow = ctk.CTkFrame(right, fg_color="transparent"); srow.pack(fill="x", padx=4, pady=2)
        self.calib_status = ctk.CTkLabel(srow, text="录制文件: -", font=FONT, anchor="w")
        self.calib_status.pack(side="left")
        ctk.CTkButton(srow, text="↻", font=FONT, width=32,
                      command=self._calib_refresh).pack(side="left", padx=4)
        ctk.CTkButton(srow, text="打开目录", font=FONT, width=76,
                      command=self._rec_open).pack(side="left", padx=2)
        ctk.CTkButton(srow, text="备份zip", font=FONT, width=70,
                      command=self._rec_backup).pack(side="left", padx=2)
        ctk.CTkButton(srow, text="删旧", font=FONT, width=56, fg_color="#a83232",
                      command=self._rec_clear).pack(side="left", padx=2)
        brow = ctk.CTkFrame(right, fg_color="transparent"); brow.pack(fill="x", padx=4, pady=4)
        self.calib_run_last = ctk.CTkButton(brow, text="▶ 校准(最近录制)", font=FONT_B,
                                            width=150, command=lambda: self._calib_run(False))
        self.calib_run_last.pack(side="left", padx=3)
        self.calib_run_all = ctk.CTkButton(brow, text="▶ 校准(全部)", font=FONT_B, width=120,
                                           fg_color="#3a7ebf", command=lambda: self._calib_run(True))
        self.calib_run_all.pack(side="left", padx=3)
        self.calib_validate = ctk.CTkButton(brow, text="🔬 信号检验", font=FONT_B, width=110,
                                            fg_color="#7a5cff", command=self._calib_validate)
        self.calib_validate.pack(side="left", padx=3)
        self.calib_apply_stable = ctk.CTkButton(
            brow, text="◆ 应用最稳健", font=FONT_B, width=120, fg_color=GREEN,
            state="disabled", command=lambda: self._calib_apply("stable"))
        self.calib_apply_stable.pack(side="left", padx=(10, 3))
        self.calib_apply_profit = ctk.CTkButton(
            brow, text="★ 应用收益最高", font=FONT_B, width=130, fg_color="#a87d32",
            state="disabled", command=lambda: self._calib_apply("profit"))
        self.calib_apply_profit.pack(side="left", padx=3)

        self.calib_box = ctk.CTkTextbox(right, font=MONO)
        self.calib_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.calib_box.insert("end", "尚未校准。先让虚拟盘跑一段积累 recordings/calib_*.jsonl, 再点校准。\n"
                              "应用后会自动存为『校准_时间』预设(左侧可载入/删除)。回测对成交偏乐观, 非收益承诺。\n")
        self.calib_box.configure(state="disabled")
        self._calib_refresh()

    def _calib_validate(self) -> None:
        from .controller import validate_signal_sync
        self.calib_validate.configure(state="disabled")
        self._calib_set("信号有效性检验中 (用录制数据测分数是否真能预测未来涨跌) ...")

        def work():
            r = validate_signal_sync(True)
            self.after(0, lambda: (self._calib_set(r), self.calib_validate.configure(state="normal")))
        threading.Thread(target=work, daemon=True).start()

    # ---- 录制文件管理 ----
    def _rec_open(self) -> None:
        from .controller import open_recordings_dir
        self._calib_set(open_recordings_dir())

    def _rec_backup(self) -> None:
        from .controller import recordings_backup_sync
        self._calib_set("... 备份中 ...")

        def work():
            r = recordings_backup_sync()
            self.after(0, lambda: self._calib_set(r))
        threading.Thread(target=work, daemon=True).start()

    def _rec_clear(self) -> None:
        from .controller import recordings_clear_sync
        if not messagebox.askyesno("删除旧录制",
                                   "删除除【最新1个】外的所有 calib 录制文件? (释放硬盘; 建议先『备份zip』)"):
            return
        self._calib_set(recordings_clear_sync(keep_latest=True))
        self._calib_refresh()

    # ---- 参数编辑器 ----
    def _read_param_entries(self) -> dict:
        out = {}
        for key, (e, typ) in self._param_entries.items():
            s = e.get().strip()
            if s == "":
                continue
            try:
                out[key] = int(round(float(s))) if typ == "int" else float(s)
            except ValueError:
                pass
        return out

    def _params_reload(self) -> None:
        cur = current_params()
        for key, (e, typ) in self._param_entries.items():
            v = cur.get(key)
            e.delete(0, "end"); e.insert(0, "" if v is None else str(v))
        self._calib_set("已重读当前 config 参数。")

    def _params_apply(self) -> None:
        params = self._read_param_entries()
        if not params:
            self._calib_set("没有可应用的参数。")
            return
        pretty = "\n".join(f"  {k} = {v}" for k, v in params.items())
        if not messagebox.askyesno("应用参数到引擎", f"写入 config.yaml?\n\n{pretty}\n\n(重启引擎后生效)"):
            return

        def work():
            r = apply_params(params)
            self.after(0, lambda: self._calib_set(r))
        threading.Thread(target=work, daemon=True).start()

    def _refresh_preset_menu(self) -> None:
        names = list_presets()
        self.preset_menu.configure(values=names)
        if names and self.preset_menu.get() not in names:
            self.preset_menu.set(names[0])

    def _preset_load(self) -> None:
        name = self.preset_menu.get()
        p = get_preset(name)
        if not p:
            self._calib_set(f"预设『{name}』为空。")
            return
        for key, (e, typ) in self._param_entries.items():
            if key in p:
                e.delete(0, "end"); e.insert(0, str(p[key]))
        self._calib_set(f"已载入预设『{name}』到左侧编辑框(尚未写入; 点『应用到引擎』才生效)。")

    def _preset_save(self) -> None:
        r = save_preset(self.preset_name.get().strip(), self._read_param_entries())
        self._refresh_preset_menu()
        self._calib_set(r)

    def _preset_delete(self) -> None:
        name = self.preset_menu.get()
        if not messagebox.askyesno("删除预设", f"删除预设『{name}』?"):
            return
        r = delete_preset(name)
        self._refresh_preset_menu()
        self._calib_set(r)

    def _calib_refresh(self) -> None:
        try:
            n, mb = calib_files_info()
        except Exception:
            n, mb = 0, 0.0
        live = ""
        st = self.ctrl.status()
        if st.get("rec_rows"):
            live = f" · 本次会话已录 {st['rec_rows']} 行"
        self.calib_status.configure(text=f"录制文件: {n} 个 / {mb:.1f} MB{live}")

    def _calib_set(self, text: str, append: bool = False) -> None:
        self.calib_box.configure(state="normal")
        if not append:
            self.calib_box.delete("1.0", "end")
        self.calib_box.insert("end", text + "\n")
        self.calib_box.see("end")
        self.calib_box.configure(state="disabled")

    def _calib_run(self, use_all: bool) -> None:
        self._calib_refresh()
        self.calib_run_last.configure(state="disabled")
        self.calib_run_all.configure(state="disabled")
        self.calib_apply_stable.configure(state="disabled")
        self.calib_apply_profit.configure(state="disabled")
        self._calib_set("分析中 ... 网格回测可能需要 10秒~2分钟 (数据越多越久), 请稍候。")

        def work():
            res = run_calibration_sync(use_all)

            def done():
                self.calib_run_last.configure(state="normal")
                self.calib_run_all.configure(state="normal")
                self._calib_set(res.get("report", "(无输出)"))
                self._calib_profit_cfg = res.get("profit_cfg")
                self._calib_stable_cfg = res.get("stable_cfg")
                if self._calib_stable_cfg:
                    self.calib_apply_stable.configure(state="normal")
                if self._calib_profit_cfg:
                    self.calib_apply_profit.configure(state="normal")
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    def _calib_apply(self, which: str) -> None:
        cfg = self._calib_stable_cfg if which == "stable" else self._calib_profit_cfg
        if not cfg:
            return
        label = "最稳健" if which == "stable" else "收益最高"
        pretty = "\n".join(f"  {k} = {v}" for k, v in cfg.items())
        if not messagebox.askyesno("应用校准配置",
                                   f"把【{label}】配置写入 config.yaml?\n\n{pretty}\n\n"
                                   "（会修改入场门槛/盈亏比/持仓时长等; 重启引擎后生效）"):
            return
        self._calib_set(f"... 应用{label}配置中 ...", append=True)
        import datetime as _dt
        preset_name = "校准_" + _dt.datetime.now().strftime("%Y%m%d_%H%M") + ("_稳健" if which == "stable" else "_收益")

        def work():
            r = apply_calibration_sync(cfg)
            sp = save_preset(preset_name, cfg)        # 同时存为带时间的预设, 方便回看/调用/删除
            self._refresh_preset_menu()

            def done():
                self._calib_set(r, append=True)
                self._calib_set(sp, append=True)
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    # ----------------- AI 选品 -----------------

    def _ai_pick_run(self) -> None:
        pool = {"加密": "crypto", "美股": "stock", "全部": "all"}.get(self.pick_pool.get(), "all")
        rows = self.ctrl.rows()    # 引擎在跑则带实时信号分; 没跑则用交易所24h真实数据兜底
        win = ctk.CTkToplevel(self)
        win.title("AI 选品推荐")
        win.geometry("720x520")
        win.transient(self)
        ctk.CTkLabel(win, text=f"AI 选品 · {self.pick_pool.get()} (分析中, 约数秒…)",
                     font=FONT_B).pack(anchor="w", padx=12, pady=(10, 4))
        box = ctk.CTkTextbox(win, height=180, font=MONO)
        box.pack(fill="x", padx=12, pady=4)
        listfr = ctk.CTkScrollableFrame(win, height=240)
        listfr.pack(fill="both", expand=True, padx=12, pady=6)

        def work():
            res = ai_pick_sync(pool, rows)

            def done():
                box.configure(state="normal")
                box.delete("1.0", "end")
                box.insert("end", res.get("text", "(无输出)"))
                box.configure(state="disabled")
                for p in res.get("picks", []) or []:
                    self._pick_row(listfr, p, win)
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    def _pick_row(self, parent, p: dict, win) -> None:
        fr = ctk.CTkFrame(parent, fg_color="#2b2b2b"); fr.pack(fill="x", pady=2)
        dirn = "做多" if p.get("direction") == "long" else "做空"
        txt = (f"{p['inst']}  {dirn}  {p.get('strategy','')}  {p.get('leverage')}x  "
               f"置信{p.get('confidence',0):.0%}\n  {p.get('reason','')}")
        ctk.CTkLabel(fr, text=txt, font=MONO, anchor="w", justify="left").pack(side="left", padx=6, pady=2)
        ctk.CTkButton(fr, text="⬇ 导入", font=FONT, width=70, fg_color="#5a4bbf",
                      command=lambda: self._import_pick(p, win)).pack(side="right", padx=6, pady=4)

    def _import_pick(self, p: dict, win) -> None:
        struct = {"inst": p.get("inst"), "direction": p.get("direction"),
                  "order_type": "post_only", "leverage": p.get("leverage")}
        self._ai_last = struct
        self.m_import_btn.configure(state="normal")
        self._import_ai(struct)
        try:
            win.destroy()
        except Exception:
            pass
        self._m_set(f"已从 AI选品导入 {p.get('inst')} {('做多' if p.get('direction')=='long' else '做空')}。"
                    "建议再点『AI分析』获取精确止盈止损/仓位, 或直接设杠杆后建仓。")

    def _show_pnl_stats(self) -> None:
        from .controller import pnl_stats_sync
        win = ctk.CTkToplevel(self)
        win.title("盈亏统计"); win.geometry("440x320"); win.transient(self)
        ctk.CTkLabel(win, text="盈亏统计 (已实现按平仓时间; 近3月数据)", font=FONT_B).pack(
            anchor="w", padx=14, pady=(12, 6))
        box = ctk.CTkTextbox(win, font=MONO)
        box.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        box.insert("end", "读取中 ...")
        box.configure(state="disabled")

        def work():
            r = pnl_stats_sync()

            def done():
                box.configure(state="normal"); box.delete("1.0", "end")
                if not r.get("ok"):
                    box.insert("end", f"读取失败: {r.get('error','?')}\n(需当前模式已配密钥)")
                else:
                    rz, ct = r.get("realized", {}), r.get("count", {})
                    lines = [f"模式: {r.get('mode')}",
                             f"当前持仓: {r.get('open_n', 0)} 个",
                             f"当前浮动盈亏(uPnL): {r.get('upl', 0):+.3f} USDT", "",
                             "已实现盈亏 (USDT):"]
                    for k in ("今日", "近7天", "近30天", "近90天"):
                        lines.append(f"  {k:<5}: {rz.get(k, 0):+.3f}   ({ct.get(k, 0)} 笔)")
                    lines.append("")
                    lines.append(f"近3月累计已实现: {r.get('total_all', 0):+.3f} ({r.get('hist_n',0)} 笔)")
                    lines.append("注: 交易所平仓历史默认近3个月; 『年』需更早归档, 暂以累计近似。")
                    box.insert("end", "\n".join(lines))
                box.configure(state="disabled")
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    def _goto_positions(self) -> None:
        """控制台点『持仓数』-> 跳手动交易页并刷新, 那里列出全部持仓(点某行再带入操作)。"""
        self._tabs.set("手动交易")
        self._refresh_orders()

    def _console_close_all(self) -> None:
        if not messagebox.askyesno("⚠ 一键全平确认",
                                   "确认【市价平掉所有持仓】?\n这会平掉当前模式下账户的全部仓位, 不可撤销。\n"
                                   "(点『否』取消)"):
            return
        from .controller import manual_close_all_sync as _ca

        def work():
            r = _ca()

            def done():
                self._append_log(f"[一键全平] {r}")
                messagebox.showinfo("一键全平", r)
                if self._tabs.get() == "手动交易":
                    self._refresh_orders()
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    # ----------------- 账户与密钥 -----------------

    def _build_credentials_tab(self) -> None:
        t = self._tab_cred
        env = read_env()
        scroll = ctk.CTkScrollableFrame(t)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)

        ctk.CTkLabel(scroll, text="区域 (账户所属地; 用错会鉴权失败)",
                     font=FONT_B, anchor="w").pack(fill="x", pady=(4, 0))
        self.region_menu = ctk.CTkOptionMenu(scroll, values=["global", "us", "eea"], font=FONT)
        self.region_menu.set(env.get("OKX_REGION", "global"))
        self.region_menu.pack(anchor="w", pady=4)
        ctk.CTkLabel(scroll, text="global=全球站(非美非欧)  us=美(仅现货)  eea=欧盟",
                     font=("Microsoft YaHei UI", 11), text_color=GREY, anchor="w").pack(fill="x")

        self.cred_entries: dict[str, ctk.CTkEntry] = {}
        self._cred_section(scroll, "虚拟盘 (Demo / 模拟盘) — 需在 OKX「Demo Trading」区创建的密钥",
                           "OKX_DEMO_API_KEY", "OKX_DEMO_SECRET_KEY", "OKX_DEMO_PASSPHRASE", env)
        self._cred_section(scroll, "实盘 (Live / 真金) — 务必关提现 + 绑IP",
                           "OKX_LIVE_API_KEY", "OKX_LIVE_SECRET_KEY", "OKX_LIVE_PASSPHRASE", env)

        ctk.CTkLabel(scroll, text="可选: 外部数据 (留空则用免费源/规则)",
                     font=FONT_B, anchor="w").pack(fill="x", pady=(12, 2))
        self._opt_entry(scroll, "FINNHUB_API_KEY (财报日历)", "FINNHUB_API_KEY", env)
        self._opt_entry(scroll, "EDGAR_USER_AGENT (姓名 邮箱, SEC 合规要求)",
                        "EDGAR_USER_AGENT", env, mask=False)

        # ---- AI 事件分类 (提供商无关) ----
        ctk.CTkLabel(scroll, text="AI 事件分类 (新闻/公告智能判断; 选『规则』则免费, 不调用任何AI)",
                     font=FONT_B, anchor="w").pack(fill="x", pady=(14, 2))
        prow = ctk.CTkFrame(scroll, fg_color="transparent")
        prow.pack(fill="x", pady=2)
        ctk.CTkLabel(prow, text="提供商", font=FONT, width=110, anchor="w").pack(side="left")
        self.ai_provider_menu = ctk.CTkOptionMenu(
            prow, values=[d for d, _ in self.PROVIDER_MAP], font=FONT, width=200)
        cur = {v: d for d, v in self.PROVIDER_MAP}.get(env.get("AI_PROVIDER", "rule"), "规则(免费)")
        self.ai_provider_menu.set(cur)
        self.ai_provider_menu.pack(side="left", padx=6)
        ctk.CTkLabel(prow, text="按任务难度自动选模型 (简单→便宜, 复杂→更强)",
                     font=("Microsoft YaHei UI", 11), text_color=GREY).pack(side="left", padx=8)

        self._opt_entry(scroll, "AI API Key", "AI_API_KEY", env)
        self._opt_entry(scroll, "Base URL", "AI_BASE_URL", env, mask=False,
                        default="https://api.deepseek.com")
        self._opt_entry(scroll, "简单任务模型 (便宜快)", "AI_MODEL_SIMPLE", env, mask=False,
                        default="deepseek-v4-flash")
        self._opt_entry(scroll, "复杂任务模型 (更强)", "AI_MODEL_HARD", env, mask=False,
                        default="deepseek-v4-pro")
        trow = ctk.CTkFrame(scroll, fg_color="transparent")
        trow.pack(fill="x", pady=2)
        ctk.CTkLabel(trow, text="选型策略", font=FONT, width=110, anchor="w").pack(side="left")
        self.ai_tier_menu = ctk.CTkOptionMenu(
            trow, values=[d for d, _ in self.TIER_MAP], font=FONT, width=200)
        self.ai_tier_menu.set({v: d for d, v in self.TIER_MAP}.get(
            env.get("AI_TIER_POLICY", "auto"), "自动(按难度)"))
        self.ai_tier_menu.pack(side="left", padx=6)
        ctk.CTkLabel(trow, text="每次调用都会在『日志』里打印实际用了哪个模型",
                     font=("Microsoft YaHei UI", 11), text_color=GREY).pack(side="left", padx=8)
        ctk.CTkLabel(scroll, text="DeepSeek: 选『DeepSeek』即可(默认已填好)。"
                     "OpenAI兼容: 改 Base URL 加 /v1 并填对应模型。Claude 需另装 anthropic 库。",
                     font=("Microsoft YaHei UI", 11), text_color=GREY,
                     anchor="w", wraplength=820, justify="left").pack(fill="x", pady=(0, 4))

        # ---- Telegram 告警 (可选) ----
        ctk.CTkLabel(scroll, text="Telegram 告警 (可选; 推送下单/平仓/状态变化)",
                     font=FONT_B, anchor="w").pack(fill="x", pady=(14, 2))
        self._opt_entry(scroll, "Bot Token (找 @BotFather 创建)", "TELEGRAM_BOT_TOKEN", env)
        self._opt_entry(scroll, "Chat ID (找 @userinfobot 获取)", "TELEGRAM_CHAT_ID", env, mask=False)

        btns = ctk.CTkFrame(scroll, fg_color="transparent")
        btns.pack(fill="x", pady=12)
        ctk.CTkButton(btns, text="💾 保存", font=FONT_B, command=self._save_credentials,
                      width=90).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="🔍 验证虚拟盘", font=FONT_B, width=130, fg_color="#3a7ebf",
                      command=lambda: self._on_verify("demo")).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="🔍 验证实盘", font=FONT_B, width=120, fg_color="#3a7ebf",
                      command=lambda: self._on_verify("live")).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="🔍 验证AI", font=FONT_B, width=92, fg_color="#3a7ebf",
                      command=lambda: self._on_verify("ai")).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="🔍 验证Finnhub", font=FONT_B, width=120, fg_color="#3a7ebf",
                      command=lambda: self._on_verify("finnhub")).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="🔍 验证EDGAR", font=FONT_B, width=120, fg_color="#3a7ebf",
                      command=lambda: self._on_verify("edgar")).pack(side="left", padx=4)
        ctk.CTkButton(btns, text="🔍 验证TG", font=FONT_B, width=92, fg_color="#3a7ebf",
                      command=lambda: self._on_verify("tg")).pack(side="left", padx=4)

        ctk.CTkLabel(scroll, text="验证结果 (3个分别验证):", font=FONT, anchor="w").pack(fill="x", pady=(8, 0))
        self.verify_box = ctk.CTkTextbox(scroll, height=180, font=MONO)
        self.verify_box.pack(fill="x", pady=4)
        self.verify_box.insert("end", "填入密钥 → 保存 → 验证。密钥仅存本机 .env, 不外传。\n"
                               "提示: 密钥若曾泄露请先到 OKX 删除重建并绑定 IP。")
        self.verify_box.configure(state="disabled")

    def _cred_section(self, parent, title, k_api, k_sec, k_pass, env) -> None:
        ctk.CTkLabel(parent, text=title, font=FONT_B, anchor="w").pack(fill="x", pady=(14, 2))
        for label, key in [("API Key", k_api), ("Secret Key", k_sec), ("Passphrase", k_pass)]:
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=label, font=FONT, width=110, anchor="w").pack(side="left")
            e = ctk.CTkEntry(row, font=FONT, width=460, show="•")
            if env.get(key):
                e.insert(0, env[key])
            e.pack(side="left", padx=6)
            self.cred_entries[key] = e

    def _opt_entry(self, parent, label, key, env, mask=True, default="") -> None:
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label, font=FONT, width=260, anchor="w").pack(side="left")
        e = ctk.CTkEntry(row, font=FONT, width=320, show="•" if mask else "")
        e.insert(0, env.get(key) or default)
        e.pack(side="left", padx=6)
        self.cred_entries[key] = e

    # ----------------- 日志 -----------------

    def _build_logs_tab(self) -> None:
        self.log_box = ctk.CTkTextbox(self._tab_log, font=MONO)
        self.log_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_box.insert("end", "引擎日志将显示在此。\n")
        self.log_box.configure(state="disabled")

    # ----------------- 多周期研究 (只读监控 + demo 影子执行验证) -----------------

    def _build_multi_tab(self) -> None:
        t = self._tab_multi
        ctk.CTkLabel(t, text="多周期研究监控 (只读 + demo 影子执行验证)", font=FONT_B,
                     anchor="w").pack(fill="x", padx=12, pady=(6, 0))
        ctk.CTkLabel(t, text="只读研究面板: 不替换控制台方向源、不驱动实盘下单。亮灯由 forward 状态驱动"
                     "(非历史回测gate): 🟢PASS=进第二阶段(仍非实盘) 🟡PENDING=观察中 🔴KILL=已关闭。"
                     "现状 A/B/C 全 PENDING(研究观察期, 不可交易)。影子=仅模拟盘执行真实性验证, 非盈利, 预期亏/打平。",
                     font=("Microsoft YaHei UI", 11), text_color=GREY, anchor="w",
                     wraplength=960, justify="left").pack(fill="x", padx=12)
        self.multi_status = ctk.CTkTextbox(t, height=190, font=MONO)
        self.multi_status.pack(fill="x", padx=8, pady=(6, 2))
        self.multi_status.configure(state="disabled")
        br = ctk.CTkFrame(t, fg_color="transparent")
        br.pack(fill="x", padx=8, pady=2)
        ctk.CTkButton(br, text="↻ 刷新状态", font=FONT, width=90,
                      command=self._multi_refresh_now).pack(side="left", padx=4)
        ctk.CTkButton(br, text="🔬 影子干运行(不下单)", font=FONT_B, width=170, fg_color="#7a5cff",
                      command=lambda: self._multi_shadow(False)).pack(side="left", padx=4)
        self.multi_arm_btn = ctk.CTkButton(br, text="🎯 在模拟盘执行(demo)", font=FONT_B, width=170,
                                           fg_color=AMBER, command=lambda: self._multi_shadow(True))
        self.multi_arm_btn.pack(side="left", padx=4)
        ctk.CTkButton(br, text="🤖 AI 解读对照", font=FONT_B, width=130, fg_color="#7a5cff",
                      command=self._multi_ai_compare).pack(side="left", padx=4)
        ctk.CTkLabel(br, text="(demo 真下单: 顶部切『虚拟盘』+ 配 demo 密钥)",
                     font=("Microsoft YaHei UI", 11), text_color=GREY).pack(side="left", padx=6)
        self.multi_result = ctk.CTkTextbox(t, font=MONO)
        self.multi_result.pack(fill="both", expand=True, padx=8, pady=(2, 8))
        self.multi_result.insert("end", "点『影子干运行』看 A/B/C 当前会不会触发 tau(纯公共数据, 不下单)。\n")
        self.multi_result.configure(state="disabled")
        self._multi_busy = False
        self._multi_refresh_now()

    def _multi_set_status(self, data: dict) -> None:
        lines = []
        if not data.get("ok"):
            lines.append(data.get("note", "无 forward 状态 (先在 scripts 跑 --mode evaluate)"))
        else:
            asof = data["rows"][0].get("asof_utc", "") if data["rows"] else ""
            lines.append(f"A/B/C forward 状态 (asof {asof}):")
            emj = {"PASS": "🟢", "KILL": "🔴"}
            for r in data["rows"]:
                v = str(r.get("verdict", "PENDING"))
                lines.append(f"  {emj.get(v, '🟡')} {r.get('code', '?')} H={r.get('H', '?')}min "
                             f"n_ts={r.get('n_ts', '-')} net10={r.get('net10_bps', '-')} "
                             f"net15={r.get('net15_bps', '-')} fwd_ic={r.get('fwd_ic', '-')} -> {v}")
            if data.get("shadow"):
                lines.append("影子成交(最近):")
                for s in data["shadow"][-6:]:
                    lines.append(f"  {s.get('entry_utc', '')} {s.get('code', '')} {s.get('side', '')} "
                                 f"{s.get('event', '')} {s.get('exit_reason', '')} "
                                 f"名义{s.get('notional', '')}U")
        self.multi_status.configure(state="normal")
        self.multi_status.delete("1.0", "end")
        self.multi_status.insert("end", "\n".join(lines) + "\n")
        self.multi_status.configure(state="disabled")

    def _multi_refresh_now(self) -> None:
        def work():
            data = multiperiod.forward_status()
            self.after(0, lambda: self._multi_set_status(data))
        threading.Thread(target=work, daemon=True).start()

    def _maybe_refresh_multi(self) -> None:
        try:
            if self._tabs.get() != "多周期研究" or self._tick_count % 12 != 0:
                return
        except Exception:
            return
        self._multi_refresh_now()

    def _multi_result_set(self, text: str) -> None:
        self.multi_result.configure(state="normal")
        self.multi_result.delete("1.0", "end")
        self.multi_result.insert("end", text + "\n")
        self.multi_result.configure(state="disabled")

    def _multi_shadow(self, armed: bool) -> None:
        if armed and not messagebox.askyesno(
                "确认在模拟盘执行",
                "将在【模拟盘(demo)】按 A/B/C 信号自动下单(执行真实性验证, 非盈利, 预期亏/打平)。\n"
                "需顶部已切到『虚拟盘』且配好 demo 密钥。确认?"):
            return
        if self._multi_busy:
            return
        self._multi_busy = True
        self.multi_arm_btn.configure(state="disabled")
        self._multi_result_set("... 影子运行中 (拉数据 + 冻结模型打分, 约 30-90 秒) ...")

        def work():
            txt = multiperiod.shadow_run(armed)

            def done():
                self._multi_result_set(txt)
                self._multi_busy = False
                self.multi_arm_btn.configure(state="normal")
                self._multi_refresh_now()
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    def _multi_ai_compare(self) -> None:
        """把多周期量化结论喂给 AI 做只读对照解读 (AI 被硬约束: PENDING/KILL 不得说成机会)。"""
        if self._multi_busy:
            return
        self._multi_busy = True
        self.multi_arm_btn.configure(state="disabled")
        self._multi_result_set("... AI 解读对照中 (跑一次当前打分 + AI 调用, 约 1-2 分钟) ...")

        def work():
            txt = multiperiod.ai_compare()

            def done():
                self._multi_result_set(txt)
                self._multi_busy = False
                self.multi_arm_btn.configure(state="normal")
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    # ----------------- 手动交易 -----------------

    def _build_manual_tab(self) -> None:
        t = self._tab_manual
        ctk.CTkLabel(t, text="手动交易 (用你保存的密钥, 按顶部 虚拟盘/实盘; 控制台点某标的或持仓会带入)",
                     font=FONT_B, anchor="w").pack(fill="x", padx=12, pady=(6, 0))
        ctk.CTkLabel(t, text="⚠ 实盘=真金, 实盘操作二次确认。做多=买, 做空=卖。手动交易无需启动引擎; "
                     "引擎『实际下单』运行时死手开关断线会撤所有挂单(正常停止先解除)。",
                     font=("Microsoft YaHei UI", 11), text_color=AMBER, anchor="w",
                     wraplength=1980, justify="left").pack(fill="x", padx=12)

        split = ctk.CTkFrame(t, fg_color="transparent")
        split.pack(fill="both", expand=True, padx=8, pady=4)
        f = ctk.CTkFrame(split, width=560)            # 左列: 下单表单
        f.pack(side="left", fill="y", padx=(4, 6)); f.pack_propagate(False)
        right = ctk.CTkFrame(split, fg_color="transparent")   # 右列: 行情/结果/持仓/历史
        right.pack(side="left", fill="both", expand=True)

        WL = 540   # 左列说明换行宽度
        # 标的 + 方向
        r1 = ctk.CTkFrame(f, fg_color="transparent"); r1.pack(fill="x", pady=(8, 3), padx=8)
        ctk.CTkLabel(r1, text="标的", font=FONT, width=42, anchor="w").pack(side="left")
        self.m_inst = ctk.CTkEntry(r1, font=FONT, width=180)
        self.m_inst.insert(0, "BTC-USDT-SWAP"); self.m_inst.pack(side="left", padx=6)
        self.m_side = ctk.CTkSegmentedButton(r1, values=["买/做多", "卖/做空"], font=FONT_B)
        self.m_side.set("买/做多"); self.m_side.pack(side="left", padx=8)
        # AI 行
        ra = ctk.CTkFrame(f, fg_color="transparent"); ra.pack(fill="x", pady=3, padx=8)
        ctk.CTkButton(ra, text="🤖 AI分析此标的", font=FONT_B, width=150, fg_color="#7a5cff",
                      command=self._m_ai).pack(side="left", padx=(0, 6))
        self.m_import_btn = ctk.CTkButton(ra, text="⬇ 导入AI建议", font=FONT_B, width=130,
                                          fg_color="#5a4bbf", state="disabled",
                                          command=self._import_ai)
        self.m_import_btn.pack(side="left", padx=3)

        # 数量
        r2 = ctk.CTkFrame(f, fg_color="transparent"); r2.pack(fill="x", pady=3, padx=8)
        ctk.CTkLabel(r2, text="数量", font=FONT, width=42, anchor="w").pack(side="left")
        self.m_unit = ctk.CTkSegmentedButton(r2, values=["USDT", "张"], font=FONT, width=90,
                                             command=self._m_unit_changed)
        self.m_unit.set("USDT"); self.m_unit.pack(side="left", padx=4)
        ctk.CTkButton(r2, text="－", font=FONT_B, width=30,
                      command=lambda: self._step(self.m_amt, -self._amt_step())).pack(side="left", padx=(6, 1))
        self.m_amt = ctk.CTkEntry(r2, font=FONT, width=110); self.m_amt.insert(0, "300")
        self.m_amt.pack(side="left", padx=1)
        ctk.CTkButton(r2, text="＋", font=FONT_B, width=30,
                      command=lambda: self._step(self.m_amt, self._amt_step())).pack(side="left", padx=1)
        self.m_amt_hint = ctk.CTkLabel(f, text="数量=USDT金额(自动换算张数); 价格仅限价/只挂单用。",
                                       font=("Microsoft YaHei UI", 11), text_color=GREY, anchor="w")
        self.m_amt_hint.pack(fill="x", padx=10)
        # 类型 + 价格
        r2b = ctk.CTkFrame(f, fg_color="transparent"); r2b.pack(fill="x", pady=3, padx=8)
        ctk.CTkLabel(r2b, text="类型", font=FONT, width=42, anchor="w").pack(side="left")
        self.m_type = ctk.CTkOptionMenu(r2b, values=[d for d, _ in self.ORD_TYPE_MAP], font=FONT, width=150)
        self.m_type.pack(side="left", padx=6)
        ctk.CTkLabel(r2b, text="价格", font=FONT, width=42, anchor="e").pack(side="left", padx=(10, 2))
        self.m_px = ctk.CTkEntry(r2b, font=FONT, width=120); self.m_px.pack(side="left", padx=2)
        ctk.CTkLabel(f, text="只挂单(maker)=被动挂, 省费但可能不成交; 限价=按填价挂; 市价IOC=吃价即成(taker,费高)。",
                     font=("Microsoft YaHei UI", 11), text_color=GREY, anchor="w",
                     wraplength=WL, justify="left").pack(fill="x", padx=10)

        # 止盈/止损
        rb = ctk.CTkFrame(f, fg_color="transparent"); rb.pack(fill="x", pady=3, padx=8)
        ctk.CTkLabel(rb, text="止盈价", font=FONT, width=42, anchor="w").pack(side="left")
        self.m_tp = ctk.CTkEntry(rb, font=FONT, width=110); self.m_tp.pack(side="left", padx=4)
        ctk.CTkLabel(rb, text="止损价", font=FONT, width=42, anchor="e").pack(side="left", padx=(10, 2))
        self.m_sl = ctk.CTkEntry(rb, font=FONT, width=110); self.m_sl.pack(side="left", padx=4)

        # 杠杆
        r3 = ctk.CTkFrame(f, fg_color="transparent"); r3.pack(fill="x", pady=3, padx=8)
        ctk.CTkLabel(r3, text="杠杆x", font=FONT, width=42, anchor="w").pack(side="left")
        ctk.CTkButton(r3, text="－", font=FONT_B, width=30,
                      command=lambda: self._step(self.m_lev, -1, 1, 50)).pack(side="left", padx=(6, 1))
        self.m_lev = ctk.CTkEntry(r3, font=FONT, width=70); self.m_lev.insert(0, "3")
        self.m_lev.pack(side="left", padx=1)
        ctk.CTkButton(r3, text="＋", font=FONT_B, width=30,
                      command=lambda: self._step(self.m_lev, 1, 1, 50)).pack(side="left", padx=1)
        ctk.CTkButton(r3, text="设置杠杆", font=FONT, width=86,
                      command=self._m_lev_set).pack(side="left", padx=8)

        # 一键套单
        rk = ctk.CTkFrame(f, fg_color="transparent"); rk.pack(fill="x", pady=(8, 2), padx=8)
        ctk.CTkButton(rk, text="① 建仓+止盈止损 一键套单", font=FONT_B, width=240, fg_color="#2e8b6f",
                      command=self._m_bracket).pack(side="left", padx=2)
        ctk.CTkLabel(f, text="套单=入场+止盈+止损(OCO)一次下; 成交后自动挂, 先触发者撤另一个。",
                     font=("Microsoft YaHei UI", 11), text_color=GREY, anchor="w",
                     wraplength=WL, justify="left").pack(fill="x", padx=10)

        # 分开下单: 建仓 / 挂止盈 / 挂止损
        rsep = ctk.CTkFrame(f, fg_color="transparent"); rsep.pack(fill="x", pady=(6, 2), padx=8)
        ctk.CTkLabel(rsep, text="或分开:", font=FONT, width=56, anchor="w").pack(side="left")
        ctk.CTkButton(rsep, text="建仓/加仓", font=FONT_B, width=92, fg_color=GREEN,
                      command=lambda: self._m_place(False)).pack(side="left", padx=2)
        ctk.CTkButton(rsep, text="挂止盈", font=FONT_B, width=72, fg_color="#2e8b6f",
                      command=lambda: self._m_algo("tp")).pack(side="left", padx=2)
        ctk.CTkButton(rsep, text="挂止损", font=FONT_B, width=72, fg_color="#a87d32",
                      command=lambda: self._m_algo("sl")).pack(side="left", padx=2)
        ctk.CTkLabel(f, text="分开=先『建仓/加仓』, 再单独『挂止盈』『挂止损』(均按上面方向与数量, 自动只减仓)。",
                     font=("Microsoft YaHei UI", 11), text_color=GREY, anchor="w",
                     wraplength=WL, justify="left").pack(fill="x", padx=10)

        # 减 / 平仓
        r5 = ctk.CTkFrame(f, fg_color="transparent"); r5.pack(fill="x", pady=(6, 2), padx=8)
        ctk.CTkLabel(r5, text="减/平:", font=FONT, width=56, anchor="w", text_color=RED).pack(side="left")
        ctk.CTkButton(r5, text="减仓", font=FONT_B, width=64,
                      command=lambda: self._m_place(True)).pack(side="left", padx=2)
        ctk.CTkButton(r5, text="平该标的", font=FONT_B, width=84, fg_color=RED,
                      command=self._m_close).pack(side="left", padx=2)
        ctk.CTkButton(r5, text="撤挂单", font=FONT_B, width=72,
                      command=self._m_cancel).pack(side="left", padx=2)
        ctk.CTkButton(r5, text="全部平仓", font=FONT_B, width=84, fg_color="#a83232",
                      command=self._m_close_all).pack(side="left", padx=2)

        # ---- 右列 ----
        self.m_live = ctk.CTkLabel(right, text="实时行情: — (填标的并点『刷新』, 或启动引擎)",
                                   font=("Consolas", 13, "bold"), anchor="w", text_color="#9fd0ff")
        self.m_live.pack(fill="x", padx=4, pady=(4, 2))

        ctk.CTkLabel(right, text="AI 分析 / 操作结果", font=FONT_B, anchor="w").pack(fill="x", padx=4)
        self.m_result = ctk.CTkTextbox(right, height=210, font=MONO)
        self.m_result.pack(fill="x", padx=4, pady=(2, 4))
        self.m_result.insert("end", "点『AI分析此标的』在这里查看研判; 下单/平仓/撤单结果也显示在此。\n"
                             "金额默认 USDT, 自动换算张数。\n")
        self.m_result.configure(state="disabled")

        hrow = ctk.CTkFrame(right, fg_color="transparent"); hrow.pack(fill="x", padx=4, pady=(2, 0))
        ctk.CTkLabel(hrow, text="持仓 / 挂单 / 止盈止损", font=FONT_B, anchor="w").pack(side="left")
        ctk.CTkButton(hrow, text="↻ 刷新", font=FONT, width=60,
                      command=self._refresh_orders).pack(side="left", padx=8)
        self.orders_hint = ctk.CTkLabel(hrow, text="(每5秒自动刷新; 点持仓行可带入左侧)",
                                        font=("Microsoft YaHei UI", 11), text_color=GREY)
        self.orders_hint.pack(side="left", padx=4)
        self.orders_frame = ctk.CTkScrollableFrame(right, height=150)
        self.orders_frame.pack(fill="both", expand=True, padx=4, pady=(2, 2))
        self._orders_rendered = None

        ctk.CTkLabel(right, text="历史成交 (近期 · 当前标的)", font=FONT_B, anchor="w").pack(fill="x", padx=4)
        self.fills_frame = ctk.CTkScrollableFrame(right, height=110)
        self.fills_frame.pack(fill="x", padx=4, pady=(2, 4))
        self._fills_rendered = None

    # ---- 手动交易辅助 ----
    def _amt_step(self) -> float:
        return 100.0 if self.m_unit.get() == "USDT" else 1.0

    def _m_unit_changed(self, _=None) -> None:
        self.m_amt_hint.configure(
            text="数量单位=USDT金额(自动换算张数); 价格仅限价/只挂单时用。"
            if self.m_unit.get() == "USDT" else "数量单位=合约张数; 价格仅限价/只挂单时用。")

    def _step(self, entry, delta, lo=0, hi=10 ** 9) -> None:
        try:
            v = float(entry.get())
        except ValueError:
            v = 0.0
        v = max(lo, min(hi, v + delta))
        entry.delete(0, "end")
        entry.insert(0, str(int(v) if v == int(v) else round(v, 4)))

    def _m_mode(self) -> str:
        return read_env().get("OKXB_MODE", "demo")

    def _m_confirm(self, action: str) -> bool:
        if self._m_mode() == "live":
            return messagebox.askyesno("实盘真金确认", f"你正在【实盘】{action}, 会动用真金。确认?")
        return True

    def _m_set(self, text: str) -> None:
        self.m_result.configure(state="normal")
        self.m_result.insert("end", text + "\n")
        self.m_result.see("end")
        self.m_result.configure(state="disabled")

    def _m_run(self, fn, label: str) -> None:
        self._m_set(f"... {label}中 ...")

        def work():
            r = fn()
            self.after(0, lambda: self._m_set(r))
        threading.Thread(target=work, daemon=True).start()

    def _m_place(self, reduce_only: bool = False) -> None:
        action = "减仓" if reduce_only else "建仓/加仓"
        if not self._m_confirm(action):
            return
        inst = self.m_inst.get().strip().upper()
        side = "buy" if self.m_side.get().startswith("买") else "sell"
        otype = dict(self.ORD_TYPE_MAP).get(self.m_type.get(), "post_only")
        amt = self.m_amt.get().strip()
        unit = "usdt" if self.m_unit.get() == "USDT" else "ct"
        px = self.m_px.get().strip()
        if not inst or not amt:
            self._m_set("请填写标的和数量。")
            return
        self._m_run(lambda: manual_place_sync(inst, side, otype, amt, unit, px,
                                              reduce_only), action)

    def _m_cancel(self) -> None:
        self._m_run(lambda: manual_cancel_all_sync(self.m_inst.get().strip().upper()), "撤单")

    def _m_close(self) -> None:
        if not self._m_confirm("平仓"):
            return
        self._m_run(lambda: manual_close_sync(self.m_inst.get().strip().upper()), "平仓")

    def _m_close_all(self) -> None:
        if not self._m_confirm("全部平仓"):
            return
        self._m_run(manual_close_all_sync, "全部平仓")

    def _m_lev_set(self) -> None:
        self._m_run(lambda: manual_set_leverage_sync(self.m_inst.get().strip().upper(),
                                                     self.m_lev.get().strip()), "设杠杆")

    def _m_algo(self, kind: str) -> None:
        """单独挂止盈(tp)或止损(sl): reduce-only 条件单。平仓方向=与开仓方向相反。"""
        inst = self.m_inst.get().strip().upper()
        amt = self.m_amt.get().strip()
        unit = "usdt" if self.m_unit.get() == "USDT" else "ct"
        px = (self.m_tp.get() if kind == "tp" else self.m_sl.get()).strip()
        name = "挂止盈" if kind == "tp" else "挂止损"
        if not inst or not amt:
            self._m_set("请填写标的和数量。")
            return
        if not px:
            self._m_set(f"请先填{'止盈价' if kind == 'tp' else '止损价'}。")
            return
        if not self._m_confirm(name):
            return
        from .controller import manual_algo_sync
        close_side = "sell" if self.m_side.get().startswith("买") else "buy"
        self._m_run(lambda: manual_algo_sync(inst, close_side, kind, px, amt, unit), name)
        self.after(1000, self._refresh_orders)

    def _load_inst(self, inst: str) -> None:
        """点持仓行: 把该标的带入左侧表单并刷新行情/历史。"""
        self.m_inst.delete(0, "end"); self.m_inst.insert(0, inst)
        self._refresh_orders()
        self._m_set(f"已带入 {inst}: 可在左侧改杠杆/数量后『减仓/平该标的』, 或『挂止盈/止损』。")

    def _m_bracket(self) -> None:
        if not self._m_confirm("套单(建仓+止盈+止损)"):
            return
        inst = self.m_inst.get().strip().upper()
        side = "buy" if self.m_side.get().startswith("买") else "sell"
        otype = dict(self.ORD_TYPE_MAP).get(self.m_type.get(), "post_only")
        amt = self.m_amt.get().strip()
        unit = "usdt" if self.m_unit.get() == "USDT" else "ct"
        px = self.m_px.get().strip()
        tp = self.m_tp.get().strip()
        sl = self.m_sl.get().strip()
        if not inst or not amt:
            self._m_set("请填写标的和数量。")
            return
        if not tp and not sl:
            self._m_set("套单需至少填止盈价或止损价 (否则请用『建仓/加仓』)。")
            return
        self._m_run(lambda: manual_bracket_sync(inst, side, otype, amt, unit, px, tp, sl), "套单")

    def _m_ai(self) -> None:
        inst = self.m_inst.get().strip().upper()
        if not inst:
            self._m_set("请先填标的。")
            return
        row = dict(self.ctrl.rows().get(inst, {}))
        self._m_set(f"... AI 量化分析 {inst} 中 (用更强模型, 约数秒) ...")

        def work():
            res = ai_analyze_sync(inst, row)

            def done():
                self._m_set(res.get("text", "(无输出)"))
                self._ai_last = res.get("struct")
                if self._ai_last:
                    self._ai_last.setdefault("inst", inst)
                    self.m_import_btn.configure(state="normal")
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    def _import_ai(self, struct: dict = None) -> None:
        """把 AI 结构化建议填入下单区 (不下单)。struct=None 时用最近一次分析结果。"""
        s = struct or self._ai_last
        if not s:
            self._m_set("还没有可导入的 AI 建议, 请先点『AI分析』或用『AI选品』。")
            return
        inst = s.get("inst") or self.m_inst.get().strip().upper()
        self.m_inst.delete(0, "end"); self.m_inst.insert(0, inst)
        if s.get("direction") in ("long", "short"):
            self.m_side.set("买/做多" if s["direction"] == "long" else "卖/做空")
        ot = {"post_only": "只挂单(maker)", "limit": "限价",
              "optimal_limit_ioc": "市价IOC"}.get(s.get("order_type"))
        if ot:
            self.m_type.set(ot)
        if s.get("entry_px"):
            self.m_px.delete(0, "end"); self.m_px.insert(0, str(s["entry_px"]))
        if s.get("tp_px"):
            self.m_tp.delete(0, "end"); self.m_tp.insert(0, str(s["tp_px"]))
        if s.get("sl_px"):
            self.m_sl.delete(0, "end"); self.m_sl.insert(0, str(s["sl_px"]))
        if s.get("size_usdt"):
            self.m_unit.set("USDT"); self._m_unit_changed()
            self.m_amt.delete(0, "end"); self.m_amt.insert(0, str(int(s["size_usdt"])))
        if s.get("leverage"):
            self.m_lev.delete(0, "end"); self.m_lev.insert(0, str(int(s["leverage"])))
        self._tabs.set("手动交易")
        self._m_set(f"已导入 AI 建议: {inst} {s.get('direction')} {s.get('leverage')}x "
                    f"≈{s.get('size_usdt')}U  止盈{s.get('tp_px')}/止损{s.get('sl_px')}。\n"
                    "→ 先点『设置杠杆』, 再点『建仓+止盈止损(套单)』一次性下入场+止盈+止损; "
                    "或只点『建仓/加仓』。不会自动下单。")

    def _on_row_click(self, inst: str) -> None:
        d = self.ctrl.rows().get(inst, {})
        self.m_inst.delete(0, "end"); self.m_inst.insert(0, inst)
        self.m_side.set("买/做多" if d.get("long", 0) >= d.get("short", 0) else "卖/做空")
        if d.get("mid"):
            self.m_px.delete(0, "end"); self.m_px.insert(0, str(d["mid"]))
        self._tabs.set("手动交易")
        self._m_set(f"已带入 {inst} (方向按当前信号预选, 可改)。可点『AI分析』或直接『建仓/加仓』。")

    # ---- 实时面板: 行情 + 持仓 + 挂单 + 止盈止损 + 历史成交 ----
    def _refresh_orders(self) -> None:
        from .controller import manual_panel_sync
        inst = self.m_inst.get().strip().upper()

        def work():
            snap = manual_panel_sync(inst)
            self.after(0, lambda: self._render_panel(snap))
        threading.Thread(target=work, daemon=True).start()

    def _render_panel(self, snap: dict) -> None:
        if not snap.get("ok"):
            self.orders_hint.configure(text=f"(无法获取: {snap.get('error','?')[:60]})")
            return
        self._update_live(snap.get("ticker", {}))
        self._render_orders(snap)
        self._render_fills(snap.get("fills", []), snap.get("mode"))

    def _update_live(self, tk: dict) -> None:
        inst = self.m_inst.get().strip().upper()
        if not tk:
            return
        last = float(tk.get("last", 0) or 0)
        op = float(tk.get("open24h", 0) or 0)
        bid = float(tk.get("bidPx", 0) or 0)
        ask = float(tk.get("askPx", 0) or 0)
        chg = (last / op - 1) * 100 if op > 0 else 0.0
        mid = (bid + ask) / 2 if (bid and ask) else last
        sp = (ask - bid) / mid * 1e4 if mid else 0.0
        extra = ""
        d = self.ctrl.rows().get(inst)
        if d:
            extra = f"  |  信号 多{d.get('long', 0):.0f}/空{d.get('short', 0):.0f}"
        self.m_live.configure(
            text=f"实时 {inst}: 现价 {last:g}  24h {chg:+.2f}%  "
                 f"买{bid:g}/卖{ask:g} (价差{sp:.1f}bps){extra}",
            text_color=(GREEN if chg >= 0 else RED))

    def _render_orders(self, snap: dict) -> None:
        poss = snap.get("positions", [])
        orders = snap.get("orders", [])
        algos = snap.get("algos", [])
        self.orders_hint.configure(
            text=f"({snap.get('mode')}) 持仓{len(poss)} 挂单{len(orders)} 止盈止损{len(algos)} · 每5秒自动刷新")
        sig = (snap.get("mode"),
               tuple((p.get("instId"), p.get("pos"), p.get("avgPx"), round(float(p.get("upl", 0) or 0), 3)) for p in poss),
               tuple((o.get("ordId"), o.get("px"), o.get("sz"), o.get("state")) for o in orders),
               tuple((a.get("algoId"), a.get("tpTriggerPx"), a.get("slTriggerPx"), a.get("state")) for a in algos))
        if sig == self._orders_rendered:
            return
        self._orders_rendered = sig
        for w in self.orders_frame.winfo_children():
            w.destroy()
        if not poss and not orders and not algos:
            ctk.CTkLabel(self.orders_frame, text="当前无持仓、无挂单、无止盈止损单。", font=FONT,
                         text_color=GREY).pack(anchor="w", padx=6, pady=6)
            return
        for p in poss:
            self._pos_row(p)
        for o in orders:
            self._order_row(o)
        for a in algos:
            self._algo_row(a)

    def _render_fills(self, fills: list, mode) -> None:
        sig = tuple((f.get("tradeId"), f.get("fillPx"), f.get("fillSz")) for f in fills)
        if sig == self._fills_rendered:
            return
        self._fills_rendered = sig
        for w in self.fills_frame.winfo_children():
            w.destroy()
        if not fills:
            ctk.CTkLabel(self.fills_frame, text="近期无该标的成交记录。", font=FONT,
                         text_color=GREY).pack(anchor="w", padx=6, pady=4)
            return
        import datetime as _dt
        for f in fills:
            try:
                ts = _dt.datetime.fromtimestamp(int(f.get("ts", 0)) / 1000).strftime("%m-%d %H:%M:%S")
            except Exception:
                ts = "-"
            side = f.get("side", "")
            sidecn = "买" if side == "buy" else ("卖" if side == "sell" else side)
            pnl = f.get("fillPnl", "")
            txt = (f"{ts}  {sidecn}  价{f.get('fillPx','-')}  量{f.get('fillSz','-')}"
                   + (f"  盈亏{pnl}" if pnl not in ("", "0", None) else ""))
            col = GREEN if side == "buy" else RED
            ctk.CTkLabel(self.fills_frame, text=txt, font=MONO, anchor="w",
                         text_color=col).pack(anchor="w", padx=6, pady=0)

    def _algo_row(self, a: dict) -> None:
        fr = ctk.CTkFrame(self.orders_frame, fg_color="#2e2a1f"); fr.pack(fill="x", pady=1)
        inst, aid = a.get("instId", "?"), a.get("algoId", "")
        tp = a.get("tpTriggerPx") or "-"
        sl = a.get("slTriggerPx") or "-"
        txt = (f"止盈止损 {inst}  止盈触发{tp}  止损触发{sl}  "
               f"{a.get('ordType','')}  {a.get('state','')}")
        ctk.CTkLabel(fr, text=txt, font=MONO, anchor="w", text_color="#e0c060").pack(side="left", padx=6)
        ctk.CTkButton(fr, text="撤销", font=FONT, width=54,
                      command=lambda i=inst, x=aid: self._algo_cancel(i, x)).pack(side="right", padx=3, pady=2)

    def _algo_cancel(self, inst: str, algo_id: str) -> None:
        from .controller import cancel_algo_sync
        if not self._m_confirm(f"撤销 {inst} 止盈止损单"):
            return
        self._m_run(lambda: cancel_algo_sync(inst, algo_id), "撤策略委托")
        self.after(900, self._refresh_orders)

    def _pos_row(self, p: dict) -> None:
        fr = ctk.CTkFrame(self.orders_frame, fg_color="#23311f"); fr.pack(fill="x", pady=1)
        inst = p.get("instId", "?")
        pos = float(p.get("pos", 0) or 0)
        side = "多" if pos > 0 else "空"
        upl = float(p.get("upl", 0) or 0)
        txt = (f"持仓 {inst}  {side} {abs(pos):g}张  开仓{p.get('avgPx','?')}  "
               f"标记{p.get('markPx','?')}  uPnL {upl:+.3f}  杠杆{p.get('lever','?')}x")
        lbl = ctk.CTkLabel(fr, text=txt, font=MONO, anchor="w", text_color=GREEN if upl >= 0 else RED)
        lbl.pack(side="left", padx=6)
        ctk.CTkButton(fr, text="市价平", font=FONT, width=64, fg_color=RED,
                      command=lambda i=inst: self._pos_close(i)).pack(side="right", padx=4, pady=2)
        # 点持仓行(标签/底框)即带入左侧表单
        for w in (fr, lbl):
            w.bind("<Button-1>", lambda e, i=inst: self._load_inst(i))

    def _order_row(self, o: dict) -> None:
        fr = ctk.CTkFrame(self.orders_frame, fg_color="#2b2b2b"); fr.pack(fill="x", pady=1)
        inst, oid = o.get("instId", "?"), o.get("ordId", "")
        txt = (f"挂单 {inst}  {o.get('side')}  {o.get('ordType')}  价{o.get('px','-')}  "
               f"量{o.get('sz')}(已成{o.get('accFillSz','0')})  {o.get('state')}")
        ctk.CTkLabel(fr, text=txt, font=MONO, anchor="w").pack(side="left", padx=6)
        ctk.CTkButton(fr, text="撤销", font=FONT, width=54,
                      command=lambda i=inst, x=oid: self._order_cancel(i, x)).pack(side="right", padx=3, pady=2)
        ctk.CTkButton(fr, text="改价", font=FONT, width=54, fg_color="#3a7ebf",
                      command=lambda i=inst, x=oid, px=o.get("px"): self._order_amend(i, x, px)
                      ).pack(side="right", padx=3, pady=2)

    def _pos_close(self, inst: str) -> None:
        if not self._m_confirm(f"市价平仓 {inst}"):
            return
        self._m_run(lambda: manual_close_sync(inst), f"平仓{inst}")
        self.after(900, self._refresh_orders)

    def _order_cancel(self, inst: str, oid: str) -> None:
        from .controller import cancel_one_sync
        if not self._m_confirm(f"撤销 {inst} 挂单"):
            return
        self._m_run(lambda: cancel_one_sync(inst, oid), "撤单")
        self.after(900, self._refresh_orders)

    def _order_amend(self, inst: str, oid: str, cur_px) -> None:
        from tkinter import simpledialog
        from .controller import amend_one_sync
        new_px = simpledialog.askstring("改价", f"{inst} 新价格 (当前 {cur_px}):", parent=self)
        if not new_px:
            return
        if not self._m_confirm(f"改 {inst} 挂单价为 {new_px}"):
            return
        self._m_run(lambda: amend_one_sync(inst, oid, new_px=new_px.strip()), "改单")
        self.after(900, self._refresh_orders)

    # ----------------- 行为 -----------------

    def _on_mode_change(self, _v=None) -> None:
        mode = "live" if self.mode_seg.get() == "实盘" else "demo"
        write_env({"OKXB_MODE": mode})

    def _on_live_toggle(self) -> None:
        if self.live_switch.get() and self.mode_seg.get() == "实盘":
            messagebox.showwarning("实盘真金警告",
                                   "你打开了【实盘 + 实际下单】。这会用真金交易, 可能亏损本金。\n"
                                   "建议先用『虚拟盘』充分验证。启动时还会再次确认。")

    def _provider_value(self) -> str:
        return dict(self.PROVIDER_MAP).get(self.ai_provider_menu.get(), "rule")

    def _tier_value(self) -> str:
        return dict(self.TIER_MAP).get(self.ai_tier_menu.get(), "auto")

    def _collect_updates(self) -> dict:
        updates = {"OKX_REGION": self.region_menu.get(),
                   "OKXB_MODE": "live" if self.mode_seg.get() == "实盘" else "demo",
                   "AI_PROVIDER": self._provider_value(),
                   "AI_TIER_POLICY": self._tier_value()}
        for key, entry in self.cred_entries.items():
            val = entry.get().strip()
            if val:
                updates[key] = val
        return updates

    def _save_credentials(self) -> None:
        write_env(self._collect_updates())
        messagebox.showinfo("已保存", "凭据已写入本机 .env。\n"
                            f"位置: {paths.ENV_PATH}")

    def _on_verify(self, kind: str) -> None:
        self._save_credentials_silent()
        names = {"demo": "虚拟盘", "live": "实盘", "ai": "AI", "tg": "Telegram",
                 "finnhub": "Finnhub", "edgar": "EDGAR"}
        self._set_verify_text(f"正在验证 {names.get(kind, kind)} ...")

        def work():
            if kind == "ai":
                result = verify_ai_sync()
            elif kind == "tg":
                result = verify_telegram_sync()
            elif kind == "finnhub":
                result = verify_finnhub_sync()
            elif kind == "edgar":
                result = verify_edgar_sync()
            else:
                result = verify_okx_sync(kind)
            self.after(0, lambda: self._set_verify_text(result))
        threading.Thread(target=work, daemon=True).start()

    def _save_credentials_silent(self) -> None:
        write_env(self._collect_updates())

    def _set_verify_text(self, text: str) -> None:
        self.verify_box.configure(state="normal")
        self.verify_box.delete("1.0", "end")
        self.verify_box.insert("end", text)
        self.verify_box.configure(state="disabled")

    def _on_start(self) -> None:
        if self.ctrl.running:
            return
        mode = "live" if self.mode_seg.get() == "实盘" else "demo"
        env = read_env()
        kp = ("OKX_LIVE_" if mode == "live" else "OKX_DEMO_")
        if not env.get(kp + "API_KEY"):
            messagebox.showerror("缺少密钥", f"未配置{'实盘' if mode=='live' else '虚拟盘'}密钥, "
                                 "请到『账户与密钥』填写并保存。")
            return
        dry_run = not self.live_switch.get()
        if not dry_run and mode == "live":
            if not messagebox.askyesno(
                    "⚠ 实盘真金下单确认",
                    "你将以【实盘真金】自动下单, 可能造成实际资金损失。\n\n"
                    "已确认: 密钥已轮换、关提现、绑IP, 且接受全部风险?\n\n"
                    "（强烈建议先用虚拟盘验证数日）"):
                return
        self._append_log(f"启动: 模式={mode} {'实际下单' if not dry_run else '只演练'}")
        self.ctrl.start(dry_run)
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.mode_seg.configure(state="disabled")

    def _on_stop(self) -> None:
        self._append_log("正在停止自动策略 ... (停止=不再自动开新仓; 现有仓位保留不动, "
                         "工作挂单会被死手开关数秒内撤销; 如需平仓请用『手动交易』页)")
        self.ctrl.stop()
        self.stop_btn.configure(state="disabled")

    def _on_close(self) -> None:
        if self.ctrl.running:
            self.ctrl.stop()
        self.after(400, self.destroy)

    # ----------------- 刷新循环 -----------------

    def _refresh(self) -> None:
        try:
            self._refresh_once()
        except Exception as e:                 # 渲染异常绝不能杀死刷新循环
            try:
                self._append_log(f"[ui] 刷新异常(已忽略): {e!r}")
            except Exception:
                pass
        finally:
            self.after(800, self._refresh)     # 无论如何都重排下一次刷新

    def _refresh_once(self) -> None:
        self._tick_count += 1
        for line in self.ctrl.drain_logs():
            self._append_log(line)
        self._maybe_autorefresh_orders()
        self._maybe_refresh_multi()

        running = self.ctrl.running
        # 确定性按运行态设置按钮 (修复"停止点不动": 之前竞态会在启动后误禁用停止)
        if running:
            self.status_dot.configure(text="● 运行中", text_color=GREEN)
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.mode_seg.configure(state="disabled")
        else:
            self.status_dot.configure(text="● 已停止", text_color=GREY)
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.mode_seg.configure(state="normal")
        if self.ctrl.error:
            self._append_log(f"⚠ {self.ctrl.error}")
            self.ctrl._error = None     # 只报一次

        self._maybe_refresh_console_account()
        st = self.ctrl.status()
        ab = self._acct_brief if (self._acct_brief and self._acct_brief.get("ok")) else None
        if st or ab:
            # 账户权益/持仓数/浮动盈亏 优先用独立账户快照(如实), 否则回退引擎状态
            eq = ab["equity"] if ab else (st or {}).get("equity", 0)
            self._cards["equity"].configure(text=f"{eq:.1f}")
            dp = ab["upl"] if ab else (st or {}).get("upl", 0.0)
            self._cards["day_pnl"].configure(
                text=f"{dp:+.3f}", text_color=GREEN if dp >= 0 else RED)
            sstate = (st or {}).get("state", "--")
            self._cards["state"].configure(
                text=sstate, text_color=AMBER if sstate != "normal" else GREEN)
            pos_n = ab["positions"] if ab else (st or {}).get("positions", 0)
            self._cards["positions"].configure(text=str(pos_n))
            age = (st or {}).get("data_age_ms", 10 ** 9)
            self._cards["data_age_ms"].configure(
                text=("—" if age >= 10 ** 8 else str(age)),
                text_color=GREEN if age < 600 else RED)

        rows = self.ctrl.rows()
        if rows:
            self._ensure_table_rows(list(rows.keys()))
            for inst, d in rows.items():
                r = self._table_rows.get(inst)
                if not r:
                    continue
                lb = r["labels"]
                lb["mid"].configure(text=f"{d['mid']:.4f}" if d.get("mid") else "-")
                lb["spread"].configure(text=f"{d['spread_bps']:.1f}" if d.get("spread_bps") is not None else "-")
                lb["obi"].configure(text=f"{d['obi_z']:.2f}" if d.get("obi_z") is not None else "-")
                lb["ofi"].configure(text=f"{d['ofi_z']:.2f}" if d.get("ofi_z") is not None else "-")
                ls, ss = d.get("long") or 0, d.get("short") or 0
                lb["long"].configure(text=f"{ls:.0f}", text_color=GREEN if ls >= 66 else TXT)
                lb["short"].configure(text=f"{ss:.0f}", text_color=RED if ss >= 66 else TXT)
                tr = d.get("trad")
                lb["trad"].configure(text=f"{tr*100:.0f}" if tr is not None else "-",
                                     text_color=TXT if (tr is None or tr >= 0.5) else "#c08020")
                est = d.get("entry") or "-"
                lb["entry"].configure(text=est, text_color=GREEN if est == "✓候选" else GREY)
                lb["pos"].configure(text="●" if d.get("has_pos") else "-",
                                    text_color=AMBER if d.get("has_pos") else GREY)
            self._apply_table_view(rows)

    def _maybe_refresh_console_account(self) -> None:
        """每~5秒独立拉真实账户(权益/持仓数/浮动盈亏)给控制台卡片, 不依赖引擎是否运行。"""
        if self._tick_count % 6 != 0 or self._acct_busy:
            return
        mode = read_env().get("OKXB_MODE", "demo")
        if not read_env().get(("OKX_LIVE_" if mode == "live" else "OKX_DEMO_") + "API_KEY"):
            return
        self._acct_busy = True

        def work():
            b = account_brief_sync()

            def done():
                self._acct_busy = False
                if b.get("ok"):
                    self._acct_brief = b
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    def _maybe_autorefresh_orders(self) -> None:
        """手动交易页激活时, 每~5秒自动刷新持仓/挂单 (仅当本模式已配密钥, 避免无谓报错)。"""
        try:
            if self._tabs.get() != "手动交易":
                return
        except Exception:
            return
        if self._tick_count % 6 != 0 or self._orders_busy:
            return
        mode = read_env().get("OKXB_MODE", "demo")
        if not read_env().get(("OKX_LIVE_" if mode == "live" else "OKX_DEMO_") + "API_KEY"):
            self.orders_hint.configure(text="(本模式未配密钥; 到『账户与密钥』填写后可见持仓/挂单)")
            return
        self._orders_busy = True
        from .controller import manual_panel_sync
        inst = self.m_inst.get().strip().upper()

        def work():
            snap = manual_panel_sync(inst)

            def done():
                self._orders_busy = False
                if snap.get("ok"):
                    self._render_panel(snap)
                else:
                    self.orders_hint.configure(text=f"(无法获取: {snap.get('error','?')[:50]})")
            self.after(0, done)
        threading.Thread(target=work, daemon=True).start()

    def _append_log(self, line: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line + "\n")
        self.log_box.see("end")
        # 限制行数
        if int(self.log_box.index("end-1c").split(".")[0]) > 800:
            self.log_box.delete("1.0", "200.0")
        self.log_box.configure(state="disabled")


def run() -> None:
    app = OKXBApp()
    app.mainloop()
