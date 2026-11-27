"""Load this repository as the ``ml_manager`` package.

The distribution's ``pyproject.toml`` points setuptools at a ``src`` tree that
is not in this commit, and ``ml_manager/__init__.py`` imports the training
stack. The startup path under test does not. Registering the package this way
executes the real modules, including ``ml_manager.py``, without that import.
"""

import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _register_package() -> None:
    package = sys.modules.get("ml_manager")
    if package is None:
        package = types.ModuleType("ml_manager")
        package.__package__ = "ml_manager"
        package.__file__ = str(_ROOT / "__init__.py")
        sys.modules["ml_manager"] = package
    package.__path__ = [str(_ROOT)]


_register_package()


def pytest_configure() -> None:
    _register_package()
