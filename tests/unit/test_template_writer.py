"""Unit tests for template_writer.set_value_in_temp_dict.

Covers the P1 regression: on nested/deeply-sliced element paths the cardinality
lookup must happen at the accumulated depth, not against ``type + "." + key``.
The old code resolved the wrong ElementDefinition on nested paths and crashed
with ``'dict' object has no attribute 'append'`` (nested extension slices on
``mii-pr-fall-kontakt-gesundheitseinrichtung``). Single-level slicing, which
already worked, must remain byte-identical.
"""

from types import SimpleNamespace

import pytest

from parser.resource_parser.template_writer import set_value_in_temp_dict

pytestmark = pytest.mark.unit


def _sd(*paths_and_max):
    """Minimal stand-in exposing ``.data.type`` and ``.data.snapshot.element``."""
    elements = [SimpleNamespace(path=p, max=m) for p, m in paths_and_max]
    return SimpleNamespace(
        data=SimpleNamespace(
            type="Encounter", snapshot=SimpleNamespace(element=elements)
        )
    )


def test_single_level_scalar_leaf_unchanged():
    sd = _sd(("Encounter.status", "1"))
    out = set_value_in_temp_dict({}, ["status"], "finished", sd)
    assert out == {"status": "finished"}


def test_single_level_list_backbone_descends():
    sd = _sd(("Encounter.identifier", "*"), ("Encounter.identifier.system", "1"))
    out = set_value_in_temp_dict({}, ["identifier", "system"], "http://x", sd)
    assert out == {"identifier": [{"system": "http://x"}]}


def test_nested_list_uses_depth_correct_cardinality():
    """The crash case: a nested element sharing a leaf name with an ancestor.

    ``Encounter.extension`` is a list, and its child ``...extension.extension``
    is also a list. The old code looked up ``Encounter.extension`` for BOTH
    depths (wrong), which is why writing a scalar leaf then a list under the same
    key blew up. With depth-correct lookup the nested list is built properly.
    """
    sd = _sd(
        ("Encounter.extension", "*"),
        ("Encounter.extension.extension", "*"),
        ("Encounter.extension.extension.url", "1"),
    )
    out = set_value_in_temp_dict({}, ["extension", "extension", "url"], "u", sd)
    assert out == {"extension": [{"extension": [{"url": "u"}]}]}


def test_nested_scalar_leaf_not_confused_with_ancestor_list():
    """``url`` under a nested extension is scalar even though a same-named leaf
    might not exist at ``Encounter.url`` — depth-correct path resolves it."""
    sd = _sd(
        ("Encounter.extension", "*"),
        ("Encounter.extension.url", "1"),
        ("Encounter.extension.valueString", "1"),
    )
    out = set_value_in_temp_dict({}, ["extension", "valueString"], "v", sd)
    assert out == {"extension": [{"valueString": "v"}]}


def test_list_leaf_appends_multiple_values():
    sd = _sd(("Encounter.category", "*"))
    d = set_value_in_temp_dict({}, ["category"], "a", sd)
    d = set_value_in_temp_dict(d, ["category"], "b", sd)
    assert d == {"category": ["a", "b"]}


def test_scalar_dict_then_list_does_not_crash():
    """Fail-soft guard: if a key is first written as a scalar/dict and a later
    write resolves it as a list (or the reverse), degrade instead of raising."""
    sd = _sd(("Encounter.value", "*"))
    d = {"value": {"already": "dict"}}
    # must not raise AttributeError
    out = set_value_in_temp_dict(d, ["value"], "x", sd)
    assert out["value"] == "x"
