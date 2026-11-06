"""Shared fixtures for project tests.
Runs against project_hooks.
"""
import importlib
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _hook_modules():
    return [pytest.param(importlib.import_module("project_hooks"), id="project")]


@pytest.fixture(params=_hook_modules())
def hooks(request):
    return request.param


@pytest.fixture
def require(hooks):
    def _require(*names):
        missing = [n for n in names if not hasattr(hooks, n)]
        if missing:
            pytest.skip(f"hooks module lacks: {', '.join(missing)}")
    return _require
