"""组件注册表:按名字换实现。

改用哪个模型 = 改配置里一个字符串,不改任何调用方代码。这是"换 backbone
免费变强"(docs/06 §3.5)能成为**可测主张**的前提 —— 接口层不统一的话,
每换一次模型都要改一遍调用方,那个实验就做不成。

    @register("encoder", "siglip2")
    class SigLIP2Encoder: ...

    enc = build("encoder", cfg.sentinel.encoder.name, cfg.sentinel.encoder)
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {}


class UnknownComponent(KeyError):
    """请求了一个没注册的实现。错误信息里列出所有可用名字。"""


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    """把一个类注册为某类组件的某个实现。"""
    def deco(cls: type[T]) -> type[T]:
        slot = _REGISTRY.setdefault(kind, {})
        if name in slot and slot[name] is not cls:
            raise ValueError(f"{kind}/{name} 已被 {slot[name].__name__} 占用")
        slot[name] = cls
        cls._sg2_kind = kind      # type: ignore[attr-defined]
        cls._sg2_name = name      # type: ignore[attr-defined]
        return cls
    return deco


def build(kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
    """按名字构造实现。名字不存在时报错并列出可选项。"""
    slot = _REGISTRY.get(kind, {})
    if name not in slot:
        raise UnknownComponent(
            f"未知的 {kind}: {name!r}。可用: {sorted(slot) or '(无)'}。"
            f"新实现需要用 @register({kind!r}, ...) 装饰并确保模块被导入。")
    return slot[name](*args, **kwargs)


def available(kind: str | None = None) -> dict[str, list[str]] | list[str]:
    if kind is None:
        return {k: sorted(v) for k, v in sorted(_REGISTRY.items())}
    return sorted(_REGISTRY.get(kind, {}))


def clear(kind: str | None = None) -> None:
    """仅供测试使用。"""
    if kind is None:
        _REGISTRY.clear()
    else:
        _REGISTRY.pop(kind, None)
