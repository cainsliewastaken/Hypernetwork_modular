"""hypercond: conditioning iterated solvers ("wheels") with hypernetwork-emitted per-sample weight updates.

The package is problem-agnostic. A problem plugs in by subclassing :class:`hypercond.task.Task`.
"""
from .config import Config, load_config
from .task import Task
from .update_space import UpdateSpace, ObsBasis
from .conditioned import ConditionedWheel
from .hypernet import HyperNetwork, HyperEnsemble, FlattenEncoder
from .pipeline import Pipeline

__all__ = ["Config", "load_config", "Task", "UpdateSpace", "ObsBasis", "ConditionedWheel",
           "HyperNetwork", "HyperEnsemble", "FlattenEncoder", "Pipeline"]
