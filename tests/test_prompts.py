"""Prompt 模板系统的测试。

最重要的一条:**模板变动必须使 KV 前缀缓存失效**。不失效的话模型照跑,
只是在用旧政策前缀 —— 这类错误不会报任何异常。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sg2.config import MidtierConfig, PromptConfig, SG2Config
from sg2.models import build_midtier
from sg2.prompts import (PRESETS, PromptBuilder, PromptTemplates, TemplateError,
                         load_preset)

POLICY = "SW-C1: 禁止暴力内容。"


# ==================== 模板校验 ====================

def test_missing_required_placeholder_is_refused():
    with pytest.raises(TemplateError, match="policy_text"):
        PromptTemplates(policy_header="没有占位符")


def test_missing_n_frames_is_refused():
    with pytest.raises(TemplateError, match="n_frames"):
        PromptTemplates(task_judge="判断一下")


def test_unknown_field_is_refused():
    with pytest.raises(TemplateError, match="未知模板字段"):
        PromptTemplates.from_dict({"nonexistent": "x"})


def test_braces_in_format_spec_do_not_break():
    """输出格式说明里全是 {} —— 用 str.format 会炸,必须用 string.Template。"""
    p = PromptBuilder().build(policy_text=POLICY, n_frames=2)
    assert '{"action"' in p


def test_undefined_variable_raises():
    t = PromptTemplates(policy_header="${policy_text} ${nope}")
    with pytest.raises(TemplateError, match="缺少变量"):
        PromptBuilder(t).policy_header(POLICY)


# ==================== 装配 ====================

def test_all_four_segments_appear_in_order():
    b = PromptBuilder()
    p = b.build(policy_text=POLICY, n_frames=3, checklist="- 有无武器",
                evidence=["t=10s 疑似刀具"], timestamps=[1.0, 2.0, 3.0])
    for seg in (b.templates.sink, POLICY, "有无武器", "疑似刀具", "[帧 0"):
        assert seg in p
    assert p.index(b.templates.sink) < p.index(POLICY) < p.index("疑似刀具")


def test_empty_optional_segments_are_omitted():
    p = PromptBuilder().build(policy_text=POLICY, n_frames=1)
    assert "已积累的证据" not in p and "[帧 " not in p


def test_perception_mode_swaps_task_and_format():
    b = PromptBuilder()
    judge = b.build(policy_text=POLICY, n_frames=1)
    perc = b.build(policy_text=POLICY, n_frames=1, perception_only=True)
    assert "policy_citation" in judge and "policy_citation" not in perc
    assert "不做任何判断" in perc


def test_exemplars_are_numbered():
    p = PromptBuilder().build(policy_text=POLICY, n_frames=1,
                              exemplars=[("输入A", "输出A"), ("输入B", "输出B")])
    assert "示例 1" in p and "示例 2" in p


def test_sink_can_be_disabled():
    b = PromptBuilder()
    assert b.templates.sink not in b.build(
        policy_text=POLICY, n_frames=1, include_sink=False)


# ==================== 缓存键（最重要） ====================

def test_same_policy_and_template_gives_same_key():
    b = PromptBuilder()
    assert b.cache_key(POLICY) == b.cache_key(POLICY)


def test_different_policy_gives_different_key():
    b = PromptBuilder()
    assert b.cache_key(POLICY) != b.cache_key("SW-C2: 禁止色情内容。")


def test_template_change_invalidates_cache_key():
    """改模板却命中旧缓存 = 在用旧前缀,且不会报任何错。"""
    a = PromptBuilder().cache_key(POLICY)
    b = PromptBuilder(load_preset("minimal")).cache_key(POLICY)
    assert a != b


def test_checklist_change_invalidates_cache_key():
    b = PromptBuilder()
    assert b.cache_key(POLICY, "- 有无武器") != b.cache_key(POLICY, "- 有无血迹")


def test_fingerprint_changes_with_any_field():
    base = PromptTemplates().fingerprint()
    assert PromptTemplates(joiner="\n").fingerprint() != base
    assert PromptTemplates(sink="别的开头").fingerprint() != base


# ==================== 与后端集成 ====================

def test_backend_caches_then_invalidates_on_template_change():
    m = build_midtier(MidtierConfig(name="mock"))
    assert m.set_policy(POLICY) is False        # 首装
    assert m.set_policy(POLICY) is True         # 命中

    cfg = MidtierConfig(name="mock")
    cfg.prompt = PromptConfig(preset="minimal")
    m2 = build_midtier(cfg)
    m2.ctx = m.ctx                              # 共用上下文模拟同一缓存
    assert m2.set_policy(POLICY) is False, "换模板后必须重编码"


# ==================== 配置与预设 ====================

@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_are_valid(name):
    t = load_preset(name)
    assert PromptBuilder(t).build(policy_text=POLICY, n_frames=1)


def test_unknown_preset_lists_options():
    with pytest.raises(TemplateError, match="可用"):
        load_preset("nope")


def test_overrides_apply_on_top_of_preset():
    c = PromptConfig(preset="minimal", overrides={"sink": "自定义。"})
    t = c.build_templates()
    assert t.sink == "自定义。" and t.name == "minimal"


def test_config_roundtrip_keeps_prompt_section():
    c = SG2Config()
    c.midtier.prompt = PromptConfig(preset="minimal", checklist="- 检查项")
    r = SG2Config.from_dict(c.to_dict())
    assert r.midtier.prompt.preset == "minimal"
    assert r.midtier.prompt.checklist == "- 检查项"


def test_external_yaml_roundtrip(tmp_path):
    p = PromptTemplates(name="mine", sink="我的开头。").save(tmp_path / "t.yaml")
    assert PromptTemplates.load(p).sink == "我的开头。"


def test_perception_only_is_read_from_config_not_hardcoded():
    """回归:step() 曾硬编码 perception_only=False,导致 perception_first
    预设下模板问感知、格式说明却要判决,实测模型照样输出 flag。
    """
    cfg = MidtierConfig(name="mock")
    cfg.prompt = PromptConfig(preset="perception_first", perception_only=True)
    m = build_midtier(cfg)
    p = m.prompts.build(policy_text=POLICY, n_frames=1,
                        perception_only=cfg.prompt.perception_only)
    assert "policy_citation" not in p
    assert cfg.prompt.perception_only is True


def test_perception_flag_survives_config_roundtrip():
    c = SG2Config()
    c.midtier.prompt = PromptConfig(preset="perception_first",
                                    perception_only=True)
    assert SG2Config.from_dict(c.to_dict()).midtier.prompt.perception_only
