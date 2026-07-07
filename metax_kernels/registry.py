"""Kernel registry — swap eager vs fused MACA implementations at runtime."""

from __future__ import annotations

from typing import Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable)


class KernelRegistry:
    _kernels: Dict[str, Callable] = {}
    _default_impl: str = "eager"

    @classmethod
    def register(cls, name: str, impl: str = "fused") -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            cls._kernels[f"{name}:{impl}"] = fn
            return fn

        return decorator

    @classmethod
    def get(cls, name: str, impl: Optional[str] = None) -> Callable:
        key = f"{name}:{impl or cls._default_impl}"
        if key not in cls._kernels:
            eager_key = f"{name}:eager"
            if eager_key in cls._kernels:
                return cls._kernels[eager_key]
            raise KeyError(f"kernel not found: {key}")
        return cls._kernels[key]

    @classmethod
    def set_default_impl(cls, impl: str) -> None:
        cls._default_impl = impl

    @classmethod
    def list_kernels(cls) -> list[str]:
        return sorted(cls._kernels.keys())


def register_kernel(name: str, impl: str = "fused") -> Callable[[F], F]:
    return KernelRegistry.register(name, impl)
