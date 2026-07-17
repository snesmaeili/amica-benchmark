from .base import AmicaAdapter
from .amica_python_adapter import AmicaPythonAdapter
from .fortran_adapter import FortranAdapter

try:  # PyAMICA is not required for Python-versus-Fortran parity campaigns.
    from .pyamica_adapter import PyamicaAdapter
except ImportError:  # pragma: no cover - depends on the optional Torch stack
    PyamicaAdapter = None

__all__ = [
    "AmicaAdapter",
    "AmicaPythonAdapter",
    "PyamicaAdapter",
    "FortranAdapter",
]
