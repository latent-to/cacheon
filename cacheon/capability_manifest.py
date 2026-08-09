"""Declared public imports of digest-bound private callers.

Some validator deployments load private ``module:function`` capabilities whose
code is not in this repository. Static reachability scanning
(``scripts/check_islands.py``) cannot see those imports, so any public module
consumed only by such a private caller must be declared here. The declaration
is the consumer: an undeclared module with no other production importer is an
island and fails repository validation.

Keep this list exact. Declaring a module here asserts that a currently
deployed private capability imports it; remove entries when that stops being
true.
"""

from __future__ import annotations

CAPABILITY_MODULES: tuple[str, ...] = (
    "cacheon.eval.b300_qualification_graph_evidence_store",
    "cacheon.eval.tokenizer_identity",
)
