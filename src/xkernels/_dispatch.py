"""Backend registry and selection.

Backends self-register with `@register(kernel_name, Backend.X)`. The public op
calls `dispatch(kernel_name, *args, backend="auto", **kwargs)`, which resolves:
explicit arg -> env override (XKERNELS_BACKEND) -> auto (per-vendor preference),
falling back to REFERENCE.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

from ._backends import Backend, detect_vendor

# kernel_name -> {Backend: callable}
_REGISTRY: dict[str, dict[Backend, Callable]] = {}


@dataclass(frozen=True)
class BackendFailure:
    kernel_name: str
    backend: Backend
    source: str
    exc_type: str
    message: str
    exception: BaseException


# kernel_name -> suppressed backend import/registration failures
_BACKEND_FAILURES: dict[str, list[BackendFailure]] = {}
_REFERENCE_FALLBACK_WARNED: set[tuple[str, tuple[tuple[str, str, str], ...]]] = set()

# Per-vendor preference order for "auto" selection (first available wins).
_AUTO_ORDER: dict[str, list[Backend]] = {
    "nvidia": [Backend.CUDA, Backend.TRITON, Backend.REFERENCE],
    "amd": [Backend.HIP, Backend.TRITON, Backend.REFERENCE],
    "none": [Backend.REFERENCE],
}


def _strict_backend_failures() -> bool:
    return os.environ.get("XKERNELS_STRICT_BACKENDS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def record_backend_failure(
    kernel_names: str | list[str] | tuple[str, ...],
    backend: Backend,
    exc: BaseException,
    *,
    source: str,
) -> None:
    """Record an optional backend import/registration failure for diagnostics."""
    names = (kernel_names,) if isinstance(kernel_names, str) else tuple(kernel_names)
    for kernel_name in names:
        failure = BackendFailure(
            kernel_name=kernel_name,
            backend=backend,
            source=source,
            exc_type=type(exc).__name__,
            message=str(exc),
            exception=exc,
        )
        _BACKEND_FAILURES.setdefault(kernel_name, []).append(failure)


@contextmanager
def backend_registration_guard(
    kernel_names: str | list[str] | tuple[str, ...],
    backend: Backend,
    *,
    source: str,
):
    """Record optional backend registration failures and honor strict mode."""
    try:
        yield
    except Exception as exc:
        record_backend_failure(kernel_names, backend, exc, source=source)
        if _strict_backend_failures():
            raise


def register(kernel_name: str, backend: Backend) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        _REGISTRY.setdefault(kernel_name, {})[backend] = fn
        return fn

    return deco


def registered_backends(kernel_name: str) -> list[Backend]:
    return list(_REGISTRY.get(kernel_name, {}).keys())


def registered_kernels() -> list[str]:
    """Return all kernel names that have at least one registered backend."""
    return list(_REGISTRY.keys())


def backend_failures(kernel_name: str | None = None) -> dict[str, list[BackendFailure]]:
    """Return suppressed backend failures, optionally for one kernel."""
    if kernel_name is not None:
        return {kernel_name: list(_BACKEND_FAILURES.get(kernel_name, ()))}
    return {name: list(failures) for name, failures in _BACKEND_FAILURES.items()}


def backend_diagnostics() -> dict[str, dict[str, object]]:
    """Return registered backends and suppressed failures for every known kernel."""
    names = set(_REGISTRY) | set(_BACKEND_FAILURES)
    diagnostics: dict[str, dict[str, object]] = {}
    for name in sorted(names):
        diagnostics[name] = {
            "registered": [backend.value for backend in registered_backends(name)],
            "failures": [
                {
                    "backend": failure.backend.value,
                    "source": failure.source,
                    "type": failure.exc_type,
                    "message": failure.message,
                }
                for failure in _BACKEND_FAILURES.get(name, ())
            ],
        }
    return diagnostics


def _coerce(backend: Backend | str) -> Backend:
    return backend if isinstance(backend, Backend) else Backend(backend)


def _failure_for(kernel_name: str, backend: Backend) -> BackendFailure | None:
    for failure in reversed(_BACKEND_FAILURES.get(kernel_name, ())):
        if failure.backend is backend:
            return failure
    return None


def _raise_unregistered_backend(
    kernel_name: str,
    backend: Backend,
    impls: dict[Backend, Callable],
) -> None:
    failure = _failure_for(kernel_name, backend)
    if failure is not None:
        raise RuntimeError(
            f"backend {backend.name} failed to register for '{kernel_name}' "
            f"from {failure.source}: {failure.exc_type}: {failure.message}"
        ) from failure.exception
    raise KeyError(
        f"backend {backend.name} not registered for '{kernel_name}'; "
        f"have {[b.name for b in impls]}"
    )


def _select_backend_with_source(
    kernel_name: str,
    backend: Backend | str = "auto",
) -> tuple[Backend, str]:
    if kernel_name not in _REGISTRY:
        raise KeyError(f"no backends registered for kernel '{kernel_name}'")
    impls = _REGISTRY[kernel_name]

    if backend != "auto":
        chosen = _coerce(backend)
        if chosen not in impls:
            _raise_unregistered_backend(kernel_name, chosen, impls)
        return chosen, "explicit"

    env = os.environ.get("XKERNELS_BACKEND")
    if env:
        chosen = Backend(env.lower())
        if chosen in impls:
            return chosen, "env"
        _raise_unregistered_backend(kernel_name, chosen, impls)

    for candidate in _AUTO_ORDER.get(detect_vendor(), [Backend.REFERENCE]):
        if candidate in impls:
            return candidate, "auto"
    # Last resort: anything registered.
    return next(iter(impls)), "fallback_any"


def select_backend(kernel_name: str, backend: Backend | str = "auto") -> Backend:
    return _select_backend_with_source(kernel_name, backend)[0]


def _has_cuda_tensor(obj) -> bool:
    if getattr(obj, "is_cuda", False):
        return True
    if isinstance(obj, dict):
        return any(_has_cuda_tensor(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_cuda_tensor(value) for value in obj)
    return False


def _warn_reference_fallback_if_needed(
    kernel_name: str,
    backend: Backend | str,
    chosen: Backend,
    source: str,
    args: tuple,
    kwargs: dict,
) -> None:
    if backend != "auto" or chosen is not Backend.REFERENCE or source != "auto":
        return
    if not (_has_cuda_tensor(args) or _has_cuda_tensor(kwargs)):
        return
    failures = [
        failure
        for failure in _BACKEND_FAILURES.get(kernel_name, ())
        if failure.backend is not Backend.REFERENCE
    ]
    if not failures:
        return
    key = (
        kernel_name,
        tuple((failure.backend.value, failure.source, failure.message) for failure in failures),
    )
    if key in _REFERENCE_FALLBACK_WARNED:
        return
    _REFERENCE_FALLBACK_WARNED.add(key)
    summary = "; ".join(
        f"{failure.backend.value} from {failure.source}: "
        f"{failure.exc_type}: {failure.message}"
        for failure in failures
    )
    warnings.warn(
        f"backend='auto' selected REFERENCE for '{kernel_name}' on a GPU tensor "
        f"after optimized backend registration failed ({summary})",
        RuntimeWarning,
        stacklevel=3,
    )


def dispatch(kernel_name: str, *args, backend: Backend | str = "auto", **kwargs):
    """Dispatch ``kernel_name`` to the selected backend and invoke it."""
    chosen, source = _select_backend_with_source(kernel_name, backend)
    _warn_reference_fallback_if_needed(kernel_name, backend, chosen, source, args, kwargs)
    return _REGISTRY[kernel_name][chosen](*args, **kwargs)
