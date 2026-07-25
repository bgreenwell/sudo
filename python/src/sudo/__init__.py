"""SUDO: Surrogate-Assisted Double Machine Learning."""

from .dgp import (generate_binary, generate_binary_cloglog,
                  generate_binary_nonlinear, generate_ordinal)
from .estimator import SudoDML
from .pooling import pool_rubin
from .surrogate import complete_surrogate

__all__ = [
    "SudoDML",
    "complete_surrogate",
    "pool_rubin",
    "generate_binary",
    "generate_binary_nonlinear",
    "generate_ordinal",
    "generate_binary_cloglog",
]
