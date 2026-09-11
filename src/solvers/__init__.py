"""
Módulo de algoritmos determinísticos para resolução de Subset Sum.
"""

from src.solvers.base import BaseSolver
from src.solvers.brute_force import BruteForceSolver
from src.solvers.backtracking import BacktrackingSolver
from src.solvers.dynamic_programming import DynamicProgrammingSolver
from src.solvers.branch_and_bound import BranchAndBoundSolver
from src.solvers.solver_factory import get_solver, compare_solvers, SOLVER_REGISTRY

__all__ = [
    "BaseSolver",
    "BruteForceSolver",
    "BacktrackingSolver",
    "DynamicProgrammingSolver",
    "BranchAndBoundSolver",
    "get_solver",
    "compare_solvers",
    "SOLVER_REGISTRY"
]
