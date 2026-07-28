"""Unit tests for the profile-constrained target tree.

Fixtures are the exact shapes the foreign-IG batch broke on (revision_todos Tier 3c/3d):
hmb's fixed leaves under a slice (N1), evo13's fixed `Reference.type` beneath an optional
parent (N2), ontario's polymorphic `allowed[x]` (N3), capable's prohibited `coding` (N4),
and a nested CodeableConcept whose children the profile does not spell out (F4).
"""

import pytest

from mapping.target_tree import TargetTree

pytestmark = pytest.mark.unit


def _el(eid, path, min=0, max="1", types=None, slice_name=None, **extra):
    element = {"id": eid, "path": path, "min": min, "max": max}
    if types:
        element["type"] = [{"code": t} for t in types]
    if slice_name:
        element["sliceName"] = slice_name
    element.update(extra)
    return element


def _sd(elements, type_="Observation"):
    return {
        "resourceType": "StructureDefinition",
        "type": type_,
        "snapshot": {"element": elements},
    }


def _chain(node):
    cur = node.parent
    while cur is not None:
        yield cur
        cur = cur.parent


# --------------------------------------------------------------- N1 + slices
def _hmb_sd():
    """hmb: category:obstetrics whose fixed content sits on coding.system/code."""
    return _sd(
        [
            _el("Observation", "Observation", max="*"),
            _el("Observation.category", "Observation.category", min=1, types=["CodeableConcept"]),
            _el("Observation.category:obstetrics", "Observation.category", min=1,
                types=["CodeableConcept"], slice_name="obstetrics"),
            _el("Observation.category:obstetrics.coding", "Observation.category.coding",
                min=1, types=["Coding"]),
            _el("Observation.category:obstetrics.coding.system", "Observation.category.coding.system",
                types=["uri"], fixedUri="http://terminology.hl7.org/CodeSystem/observation-category"),
            _el("Observation.category:obstetrics.coding.code", "Observation.category.coding.code",
                types=["code"], patternCode="social-history"),
        ]
    )


def test_slice_identity_survives_in_the_parent_chain():
    tree = TargetTree.from_snapshot(_hmb_sd())
    node = tree.node("Observation.category:obstetrics.coding.system")
    assert node is not None
    assert [a.eid for a in _chain(node)] == [
        "Observation.category:obstetrics.coding",
        "Observation.category:obstetrics",
        "Observation.category",
        "Observation",
    ]


def test_fixed_leaves_under_a_slice_are_discoverable():
    """N1: these are the values the generator failed to emit, so hmb scored 0/4."""
    tree = TargetTree.from_snapshot(_hmb_sd())
    fixed = tree.fixed_descendants("Observation.category:obstetrics")
    assert {n.path.split(".")[-1] for n in fixed} == {"system", "code"}
    assert [n.fixed_value for n in fixed if n.path.endswith(".code")] == ["social-history"]


def test_pattern_and_fixed_are_distinguished():
    tree = TargetTree.from_snapshot(_hmb_sd())
    system = tree.node("Observation.category:obstetrics.coding.system")
    code = tree.node("Observation.category:obstetrics.coding.code")
    assert system.is_pattern is False
    assert code.is_pattern is True


# ------------------------------------------------------------------------ N2
def test_depth_below_lets_the_emitter_refuse_to_hoist():
    """N2: Procedure.reasonReference.type must not be attached to the Procedure root."""
    tree = TargetTree.from_snapshot(
        _sd(
            [
                _el("Procedure", "Procedure", max="*"),
                _el("Procedure.reasonReference", "Procedure.reasonReference", types=["Reference"]),
                _el("Procedure.reasonReference.type", "Procedure.reasonReference.type",
                    types=["uri"], fixedUri="Task"),
            ],
            type_="Procedure",
        )
    )
    leaf = tree.node("Procedure.reasonReference.type")
    assert leaf.depth_below(tree.root) == 2, "two levels below the root, so not root-attachable"
    assert leaf.parent.required is False, "optional parent -> no provider -> do not emit"


# ------------------------------------------------------------------------ N3
def test_effective_types_expose_the_profile_narrowing():
    tree = TargetTree.from_snapshot(
        _sd(
            [
                _el("MedicationRequest", "MedicationRequest", max="*"),
                _el("MedicationRequest.substitution", "MedicationRequest.substitution"),
                _el("MedicationRequest.substitution.allowed[x]",
                    "MedicationRequest.substitution.allowed[x]", min=1,
                    types=["boolean", "CodeableConcept"]),
            ],
            type_="MedicationRequest",
        )
    )
    node = tree.node("MedicationRequest.substitution.allowed[x]")
    assert node.is_choice is True
    assert tree.effective_types("MedicationRequest.substitution.allowed[x]") == [
        "boolean",
        "CodeableConcept",
    ]


# ------------------------------------------------------------------------ N4
def _capable_sd():
    """capable: Goal.description forbids coding (max=0) but requires text."""
    return _sd(
        [
            _el("Goal", "Goal", max="*"),
            _el("Goal.description", "Goal.description", min=1, types=["CodeableConcept"]),
            _el("Goal.description.coding", "Goal.description.coding", max="0", types=["Coding"]),
            _el("Goal.description.coding.system", "Goal.description.coding.system", types=["uri"]),
            _el("Goal.description.text", "Goal.description.text", min=1, types=["string"]),
        ],
        type_="Goal",
    )


def test_prohibited_subtree_is_pruned():
    tree = TargetTree.from_snapshot(_capable_sd())
    assert tree.node("Goal.description.coding") is None
    assert tree.node("Goal.description.coding.system") is None, "descendants go too"
    assert tree.node("Goal.description.text") is not None


def test_prohibited_children_are_not_reachable_from_the_parent():
    """N4: container expansion walks children, so the prohibited one must be absent."""
    tree = TargetTree.from_snapshot(_capable_sd())
    description = tree.node("Goal.description")
    assert "coding" not in description.children
    assert "text" in description.children


def test_is_prohibited_reports_pruned_elements_and_their_descendants():
    tree = TargetTree.from_snapshot(_capable_sd())
    assert tree.is_prohibited("Goal.description.coding") is True
    assert tree.is_prohibited("Goal.description.coding.system") is True
    assert tree.is_prohibited("Goal.description.text") is False


# ------------------------------------------------------------------------ F4
def test_introspected_children_are_added_where_the_profile_is_silent():
    """F4: reaction.substance is a bare CodeableConcept; its children must still exist."""
    tree = TargetTree.from_snapshot(
        _sd(
            [
                _el("AllergyIntolerance", "AllergyIntolerance", max="*"),
                _el("AllergyIntolerance.reaction", "AllergyIntolerance.reaction", min=1),
                _el("AllergyIntolerance.reaction.substance", "AllergyIntolerance.reaction.substance",
                    min=1, types=["CodeableConcept"]),
            ],
            type_="AllergyIntolerance",
        ),
        introspect_children=lambda code, parent: [
            {"path": f"{parent}.coding", "type": [{"code": "Coding"}], "cardinality": {"min": 0, "max": "*"}},
            {"path": f"{parent}.text", "type": [{"code": "string"}], "cardinality": {"min": 0, "max": "1"}},
        ],
    )
    substance = tree.node("AllergyIntolerance.reaction.substance")
    assert set(substance.children) == {"coding", "text"}
    assert substance.children["coding"].introspected is True


def test_introspection_does_not_overwrite_profile_defined_children():
    tree = TargetTree.from_snapshot(
        _hmb_sd(),
        introspect_children=lambda code, parent: [
            {"path": f"{parent}.text", "type": [{"code": "string"}], "cardinality": {}}
        ],
    )
    slice_node = tree.node("Observation.category:obstetrics")
    assert slice_node.children["coding"].introspected is False


def test_introspection_never_revives_a_pruned_branch():
    tree = TargetTree.from_snapshot(
        _capable_sd(),
        introspect_children=lambda code, parent: [
            {"path": f"{parent}.coding", "type": [{"code": "Coding"}], "cardinality": {}}
        ],
    )
    assert tree.node("Goal.description.coding") is None


# -------------------------------------------------------------------- manifest
def test_manifest_marks_requirements_under_optional_parents_inactive():
    tree = TargetTree.from_snapshot(
        _sd(
            [
                _el("Observation", "Observation", max="*"),
                _el("Observation.status", "Observation.status", min=1, types=["code"]),
                _el("Observation.note", "Observation.note", types=["Annotation"]),
                _el("Observation.note.text", "Observation.note.text", min=1, types=["markdown"]),
            ]
        )
    )
    manifest = {e["id"]: e for e in tree.required_manifest()}
    assert manifest["Observation.status"]["active"] is True
    assert manifest["Observation.note.text"]["active"] is False


def test_manifest_excludes_prohibited_requirements():
    """The manifest and the emitter read the same tree, so neither can see a pruned node."""
    tree = TargetTree.from_snapshot(_capable_sd())
    ids = {e["id"] for e in tree.required_manifest()}
    assert "Goal.description.text" in ids
    assert not any(i.startswith("Goal.description.coding") for i in ids)
