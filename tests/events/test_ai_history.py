"""AI 历史持久化: 单标的分析/选品结果要能回看 (修复"过一会在结果窗口找不到")。"""
from okxb.gui import controller as ctl


def test_ai_history_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "_ai_history_path", lambda: tmp_path / "ai_history.jsonl")
    ctl.ai_history_append("单标的分析", "BTC-USDT-SWAP", "research text 1")
    ctl.ai_history_append("AI选品", "全部", "pick text 2")
    recs = ctl.ai_history_recent(10)
    assert len(recs) == 2
    assert recs[0]["kind"] == "AI选品"              # newest first
    assert recs[1]["text"] == "research text 1"
    assert recs[0]["ts"]                            # has a timestamp


def test_ai_history_empty_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "_ai_history_path", lambda: tmp_path / "none.jsonl")
    assert ctl.ai_history_recent() == []


def test_ai_history_caps_at_200(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "_ai_history_path", lambda: tmp_path / "ai_history.jsonl")
    for i in range(250):
        ctl.ai_history_append("单标的分析", "BTC-USDT-SWAP", f"t{i}")
    lines = (tmp_path / "ai_history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 200
    assert ctl.ai_history_recent(1)[0]["text"] == "t249"   # newest retained
