"""AI 前向验证候选 — 净额公式 + 判决闸门 (含 IC 闸门) + runner 无网络安全。"""
from okxb.research import ai_forward as af


def test_net_bps_long_short_and_guard():
    assert abs(af.net_bps(1, 100, 101, 15) - 85.0) < 1e-6      # 多 +1% =100bps, 扣15 ->85
    assert abs(af.net_bps(-1, 100, 99, 10) - 90.0) < 1e-6      # 空, -1% -> +100bps, 扣10 ->90
    assert af.net_bps(1, 0, 100, 15) == 0.0                    # 无效入场价 -> 0


def test_evaluate_pending_then_pass():
    v = af.evaluate([4.0, 6.0] * 25, [9.0, 11.0] * 25, [50.0] * 50)   # n=50 <100
    assert v.verdict == "PENDING" and "insufficient" in v.reason
    v = af.evaluate([4.0, 6.0] * 60, [9.0, 11.0] * 60, [20.0] * 120)  # strong, ic>0
    assert v.verdict == "PASS", v.reason


def test_evaluate_sticky_kill_no_edge():
    v = af.evaluate([-2.0] * 40, [-1.0] * 40, [-5.0] * 40)            # net10<=0, n>=30
    assert v.verdict == "KILL"


def test_evaluate_requires_positive_ic():
    # net15 positive但 AI 方向与未来负相关(ic<=0) -> 不 PASS
    v = af.evaluate([4.0, 6.0] * 60, [9.0, 11.0] * 60, [-1.0] * 120)
    assert v.verdict == "PENDING" and "ai_ic<=0" in v.reason


def test_runner_status_evaluate_without_meta(tmp_path, monkeypatch):
    from okxb import paths
    from okxb.research import ai_forward_runner as run
    monkeypatch.setattr(paths, "APP_DIR", tmp_path)
    monkeypatch.setenv("RESEARCH_DATA_DIR", "")        # 用默认 APP_DIR/data
    assert "尚未冻结" in run.status()
    assert "尚未冻结" in run.evaluate()


def test_research_base_honors_env(tmp_path, monkeypatch):
    from okxb.research.datadir import research_base
    target = tmp_path / "myresearch"
    monkeypatch.setenv("RESEARCH_DATA_DIR", str(target))
    assert research_base() == target and target.exists()


def test_truthy_and_auto_enabled(monkeypatch):
    from okxb.research import ai_forward_runner as run
    for v in ("1", "true", "YES", "on", "On"):
        assert run.truthy(v)
    for v in ("", "0", "no", "off", None):
        assert not run.truthy(v)
    monkeypatch.setenv("AI_FORWARD_AUTO", "1")
    assert run.auto_enabled() is True
    monkeypatch.setenv("AI_FORWARD_AUTO", "")
    assert run.auto_enabled() is False
