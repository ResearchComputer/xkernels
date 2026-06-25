import warnings

import pytest

from xkernels._backends import Backend
from xkernels._dispatch import (
    backend_diagnostics,
    backend_registration_guard,
    dispatch,
    record_backend_failure,
    register,
    registered_backends,
)


class _CudaLike:
    is_cuda = True


def setup_function():
    # Register a couple of fake backends for an isolated kernel name.
    @register("_unit", Backend.REFERENCE)
    def _ref(x):
        return ("reference", x)

    @register("_unit", Backend.TRITON)
    def _triton(x):
        return ("triton", x)


def test_registered_backends_lists_what_was_registered():
    assert set(registered_backends("_unit")) >= {Backend.REFERENCE, Backend.TRITON}


def test_explicit_backend_is_honored():
    assert dispatch("_unit", 5, backend=Backend.TRITON)[0] == "triton"


def test_string_backend_is_accepted():
    assert dispatch("_unit", 5, backend="reference")[0] == "reference"


def test_unknown_backend_raises():
    with pytest.raises(KeyError):
        dispatch("_unit", 5, backend=Backend.CUDA)


def test_unknown_kernel_raises():
    with pytest.raises(KeyError):
        dispatch("_does_not_exist", 5)


def test_backend_diagnostics_report_suppressed_failure():
    exc = ImportError("unit import failed")
    record_backend_failure(
        "_diag_report",
        Backend.TRITON,
        exc,
        source="tests.fake_triton",
    )

    diag = backend_diagnostics()
    assert diag["_diag_report"]["failures"] == [
        {
            "backend": "triton",
            "source": "tests.fake_triton",
            "type": "ImportError",
            "message": "unit import failed",
        }
    ]


def test_backend_registration_guard_honors_strict_mode(monkeypatch):
    monkeypatch.setenv("XKERNELS_STRICT_BACKENDS", "1")

    with pytest.raises(ValueError, match="strict failure"):
        with backend_registration_guard(
            "_diag_strict",
            Backend.TRITON,
            source="tests.fake_triton",
        ):
            raise ValueError("strict failure")

    failures = backend_diagnostics()["_diag_strict"]["failures"]
    assert failures[0]["type"] == "ValueError"
    assert failures[0]["message"] == "strict failure"


def test_explicit_missing_backend_raises_recorded_failure():
    exc = ImportError("triton missing")
    record_backend_failure(
        "_diag_explicit",
        Backend.TRITON,
        exc,
        source="tests.fake_triton",
    )

    @register("_diag_explicit", Backend.REFERENCE)
    def _ref(x):
        return x

    with pytest.raises(RuntimeError, match="failed to register") as err:
        dispatch("_diag_explicit", 1, backend=Backend.TRITON)
    assert err.value.__cause__ is exc


def test_env_backend_override_raises_recorded_failure(monkeypatch):
    exc = ImportError("triton missing")
    record_backend_failure(
        "_diag_env",
        Backend.TRITON,
        exc,
        source="tests.fake_triton",
    )

    @register("_diag_env", Backend.REFERENCE)
    def _ref(x):
        return x

    monkeypatch.setenv("XKERNELS_BACKEND", "triton")
    with pytest.raises(RuntimeError, match="failed to register") as err:
        dispatch("_diag_env", 1)
    assert err.value.__cause__ is exc


def test_auto_reference_gpu_fallback_warns_once():
    exc = ImportError("triton missing")
    record_backend_failure(
        "_diag_auto_warn",
        Backend.TRITON,
        exc,
        source="tests.fake_triton",
    )

    @register("_diag_auto_warn", Backend.REFERENCE)
    def _ref(x):
        return x

    with pytest.warns(RuntimeWarning, match="selected REFERENCE"):
        assert dispatch("_diag_auto_warn", _CudaLike()) is not None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dispatch("_diag_auto_warn", _CudaLike())
    assert caught == []
