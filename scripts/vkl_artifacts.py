#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ruff: noqa: E402, I001
"""Emit or check registry artifacts generated from VKL sources."""
from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv/bin/python"
if sys.prefix == sys.base_prefix and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from xkernels.vkl.artifacts import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
