"""Evaluation harness for the fsh-nifi-bridge toolchain (Paper 1).

This package holds the *offline* evaluation machinery — comparison specs and
metric computations — kept separate from the shipped toolchain in ``src/`` and
from the unit suite in ``tests/``. It is rerunnable for the frozen
generalization run and for paper revisions.

Modules
-------
canonicalize    P1 — canonical form of a JSON StructureMap (rule-level P/R/F1).
normalize_fhir  P2 — normal form of output FHIR (behavioral equivalence).
"""
