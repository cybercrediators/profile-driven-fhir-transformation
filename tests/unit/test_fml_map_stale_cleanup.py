"""Stale-file cleanup after StructureMap generation (fml_map).

A removed/renamed profile (or a shifted index prefix on a fresh reprocess) leaves the
previously generated SM file behind; it would still be uploaded and transform against a
stale source model. `_cleanup_stale_structure_maps` deletes exactly the generated-named
files this run did not produce — and nothing else.
"""

from types import SimpleNamespace
import json

import pytest

from fhir.resources.R4B.quantity import Quantity

from mapping.fml_map import StructureMapGenerator
from helpers.utils import fhir_name_token

pytestmark = pytest.mark.unit


def _generator(tmp_path, map_name="structure_map_proj"):
    data_io = SimpleNamespace(
        ProjectFolders=SimpleNamespace(
            STRUCTURE_MAPS=SimpleNamespace(value="structure_maps")
        ),
        project_dir=tmp_path,
        delete_file=lambda rel: (tmp_path / rel).unlink(),
    )
    smg = object.__new__(StructureMapGenerator)
    smg.app_state = SimpleNamespace(dataIO=data_io)
    smg.map_name = map_name
    return smg


def test_stale_generated_maps_are_deleted_current_and_foreign_kept(tmp_path):
    sm_dir = tmp_path / "structure_maps"
    sm_dir.mkdir()
    (sm_dir / "001_structure_map_proj-patient.json").write_text("{}")
    # stale: same index as a current file but a profile no longer generated
    (sm_dir / "001_structure_map_proj-removed-profile.json").write_text("{}")
    # stale: index shifted on regen
    (sm_dir / "003_structure_map_proj-patient.json").write_text("{}")
    # hand-authored / foreign names must never be touched
    (sm_dir / "custom_handwritten_map.json").write_text("{}")
    (sm_dir / "002_structure_map_OTHERproj-x.json").write_text("{}")

    smg = _generator(tmp_path)
    current = [SimpleNamespace(name="001_structure_map_proj-patient")]
    smg._cleanup_stale_structure_maps(current)

    remaining = {f.name for f in sm_dir.iterdir()}
    assert remaining == {
        "001_structure_map_proj-patient.json",
        "custom_handwritten_map.json",
        "002_structure_map_OTHERproj-x.json",
    }


def test_cleanup_is_noop_without_structure_maps_dir(tmp_path):
    smg = _generator(tmp_path)
    smg._cleanup_stale_structure_maps([SimpleNamespace(name="001_structure_map_proj-a")])


def test_fhir_name_token_and_structure_map_invariant_normalization():
    target = SimpleNamespace(
        context="target", contextType=None, element="active"
    )
    rule = SimpleNamespace(name="map-active", target=[target], rule=[])
    sm = SimpleNamespace(
        name=fhir_name_token("001_structure-map-profile"),
        group=[SimpleNamespace(rule=[rule])],
    )

    StructureMapGenerator._normalize_and_validate_structure_map(sm)

    assert sm.name == "Map_001_structure_map_profile"
    assert target.contextType == "variable"


def test_structure_map_invariant_rejects_target_element_without_context():
    bad = SimpleNamespace(
        name="bad", target=[SimpleNamespace(context=None, element="active")], rule=[]
    )
    sm = SimpleNamespace(name="ValidMap", group=[SimpleNamespace(rule=[bad])])

    with pytest.raises(ValueError, match="without a context"):
        StructureMapGenerator._normalize_and_validate_structure_map(sm)


def test_todo_sanitizer_keeps_source_scaffolds_and_removes_unsafe_todos():
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "PatientProfile"
    generator.mapping_diagnostics = []
    source_scaffold = SimpleNamespace(
        name="map-required",
        source=[
            SimpleNamespace(
                context="source",
                element="TODO_MAP_REQUIRED",
                variable="src-required",
            )
        ],
        target=[
            SimpleNamespace(
                context="target",
                element="status",
                transform="copy",
                parameter=None,
            )
        ],
        dependent=None,
        rule=[],
    )
    unsafe_target_todo = SimpleNamespace(
        name="map-unsafe-target",
        source=[SimpleNamespace(context="source", element="status")],
        target=[
            SimpleNamespace(
                context="target",
                element="TODO_TARGET_STATUS",
                transform="copy",
                parameter=None,
            )
        ],
        dependent=None,
        rule=[],
    )
    unsafe_source_expression = SimpleNamespace(
        name="map-unsafe-check",
        source=[
            SimpleNamespace(
                context="source",
                element="status",
                check="TODO_DEFINE_SOURCE_CHECK",
            )
        ],
        target=[
            SimpleNamespace(
                context="target",
                element="status",
                transform="copy",
                parameter=None,
            )
        ],
        dependent=None,
        rule=[],
    )
    deferred_reference = SimpleNamespace(
        name="TODO-resolve-reference-Patient-generalPractitioner",
        documentation="Reference<Patient.generalPractitioner> -> Practitioner",
        source=[SimpleNamespace(element=None)],
        target=None,
        dependent=None,
        rule=[],
    )
    valid = SimpleNamespace(
        name="map-active",
        source=[SimpleNamespace(element="active")],
        target=[SimpleNamespace(element="active", parameter=None)],
        dependent=None,
        rule=[source_scaffold, unsafe_target_todo, unsafe_source_expression],
    )
    group = SimpleNamespace(name="Transform-Patient", rule=[valid, deferred_reference])
    structure_map = SimpleNamespace(group=[group])

    generator._sanitize_todo_rules(structure_map)

    assert group.rule == [valid, deferred_reference]
    assert valid.rule == [source_scaffold]
    assert source_scaffold.source[0].element == "TODO_MAP_REQUIRED"
    assert {item["code"] for item in generator.mapping_diagnostics} == {
        "unresolved-map-placeholder",
        "deferred-reference",
    }


def _extension_rule(name, source_element, value_child=None, sub_rules=None):
    """An extension rule shaped like the generator emits: create + a url child."""
    ext_var = f"ext-{name}"
    url_child = SimpleNamespace(
        name=f"set-extension-url-{name}",
        source=[SimpleNamespace(context=f"src-{name}")],
        target=[
            SimpleNamespace(
                context=ext_var,
                element="url",
                transform="copy",
                parameter=[SimpleNamespace(valueString="http://example.org/ext")],
            )
        ],
        dependent=None,
        rule=[],
    )
    children = [url_child] + list(sub_rules or [])
    if value_child is not None:
        children.append(value_child)
    return SimpleNamespace(
        name=f"map-extension-{name}",
        source=[
            SimpleNamespace(
                context="source", element=source_element, variable=f"src-{name}"
            )
        ],
        target=[
            SimpleNamespace(
                context="target",
                element="extension",
                variable=ext_var,
                transform="create",
                parameter=[SimpleNamespace(valueString="Extension")],
            )
        ],
        dependent=None,
        rule=children,
    )


def _unresolvable_translate(name):
    """A value rule the sanitizer drops: its target carries a TODO placeholder."""
    return SimpleNamespace(
        name=f"map-{name}-translate",
        source=[SimpleNamespace(context=f"src-{name}", element="bundesland")],
        target=[
            SimpleNamespace(
                context=f"ext-{name}",
                element="valueCode",
                transform="translate",
                parameter=[SimpleNamespace(valueString="TODO-resolveConceptMap")],
            )
        ],
        dependent=None,
        rule=[],
    )


def test_extension_left_url_only_by_sanitizing_is_pruned_too():
    # Dropping the value rule is correct on its own, but leaving its parent behind
    # ships an extension carrying nothing but a url, which violates ext-1.
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "PatientProfile"
    generator.mapping_diagnostics = []
    extension = _extension_rule(
        "federalState", "bundesland", value_child=_unresolvable_translate("federalState")
    )
    group = SimpleNamespace(name="Transform-Patient", rule=[extension])

    generator._sanitize_todo_rules(SimpleNamespace(group=[group]))

    assert group.rule == []
    assert "url-only-extension-pruned" in {
        item["code"] for item in generator.mapping_diagnostics
    }


def test_url_only_extension_behind_a_todo_source_is_kept():
    # This rule never matches at runtime, so it ships nothing invalid. It is the
    # supported sparse-map scaffold and must survive as a signal of an unmapped field.
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "PatientProfile"
    generator.mapping_diagnostics = []
    extension = _extension_rule(
        "strength", "TODO-MAP-STRENGTH_SOURCE", value_child=None
    )
    group = SimpleNamespace(name="Transform-Medication", rule=[extension])

    generator._sanitize_todo_rules(SimpleNamespace(group=[group]))

    assert group.rule == [extension]
    assert "url-only-extension-pruned" not in {
        item["code"] for item in generator.mapping_diagnostics
    }


def test_url_only_pruning_cascades_from_sub_extension_to_parent():
    # A complex extension is only as alive as its sub-extensions: once the last one
    # is pruned the parent satisfies neither half of ext-1 either.
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "ConditionProfile"
    generator.mapping_diagnostics = []
    sub = _extension_rule(
        "YesNoUnknownExtension",
        "raucher",
        value_child=_unresolvable_translate("YesNoUnknownExtension"),
    )
    parent = _extension_rule("existance", "raucher", sub_rules=[sub])
    group = SimpleNamespace(name="Transform-Condition", rule=[parent])

    generator._sanitize_todo_rules(SimpleNamespace(group=[group]))

    assert group.rule == []


def test_extension_keeping_a_live_value_rule_survives():
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "PatientProfile"
    generator.mapping_diagnostics = []
    value_child = SimpleNamespace(
        name="set-extension-value-federalState",
        source=[SimpleNamespace(context="src-federalState", element="bundesland")],
        target=[
            SimpleNamespace(
                context="ext-federalState",
                element="valueCode",
                transform="copy",
                parameter=[SimpleNamespace(valueId="srcValue")],
            )
        ],
        dependent=None,
        rule=[],
    )
    extension = _extension_rule("federalState", "bundesland", value_child=value_child)
    group = SimpleNamespace(name="Transform-Patient", rule=[extension])

    generator._sanitize_todo_rules(SimpleNamespace(group=[group]))

    assert group.rule == [extension]
    assert extension.rule == [extension.rule[0], value_child]


def test_profile_facets_become_json_safe_diagnostics():
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "ObservationProfile"
    generator.mapping_diagnostics = []
    obj = SimpleNamespace(
        mappable_fields=[
            {
                "path": "Observation.value[x]",
                "id": "Observation.value[x]",
                "default_value": {
                    "type": "Quantity",
                    "value": Quantity(value=2, unit="mg"),
                },
                "constraints": [
                    {
                        "key": "obs-1",
                        "severity": "error",
                        "expression": "dataAbsentReason.empty() or value.empty()",
                    }
                ],
                "is_modifier": True,
            }
        ]
    )

    generator._record_profile_facet_diagnostics(obj)

    json.dumps(generator.mapping_diagnostics)
    assert {item["code"] for item in generator.mapping_diagnostics} == {
        "target-default-value",
        "target-fhirpath-constraint",
        "target-modifier-element",
    }
    assert all(
        item["profile"] == "ObservationProfile"
        for item in generator.mapping_diagnostics
    )


def test_explicitly_mapped_modifier_does_not_request_mapping_intent_again():
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "ObservationProfile"
    generator.mapping_diagnostics = []
    obj = SimpleNamespace(
        mappable_fields=[
            {
                "path": "Observation.status",
                "id": "Observation.status",
                "is_modifier": True,
            }
        ]
    )

    generator._record_profile_facet_diagnostics(
        obj, mapped_paths={"Observation.status"}
    )

    assert not any(
        item["code"] == "target-modifier-element"
        for item in generator.mapping_diagnostics
    )


def test_coverage_report_is_written_when_only_diagnostics_exist():
    stored = {}
    generator = object.__new__(StructureMapGenerator)
    generator._coverage = {}
    generator.mapping_diagnostics = [
        {
            "code": "ambiguous-slice-selection",
            "message": "explicit selection required",
        }
    ]
    generator.map_name = "diagnostic-map"
    generator.app_state = SimpleNamespace(
        dataIO=SimpleNamespace(
            ProjectFolders=SimpleNamespace(SOURCE_DATA="source_data"),
            store_project_file=lambda folder, name, body, **kwargs: stored.update(
                {"folder": folder, "name": name, "body": body}
            ),
        )
    )

    typed_report = generator._save_coverage_report()

    assert stored["name"] == "diagnostic-map_coverage.json"
    report = json.loads(stored["body"])
    assert typed_report is generator.coverage_report
    assert json.loads(typed_report.model_dump_json(exclude_none=True)) == report
    assert report["report_version"] == 3
    assert report["summary"]["profiles"] == 0
    assert report["mapping_diagnostics"][0]["code"] == (
        "ambiguous-slice-selection"
    )
    assert report["mapping_diagnostics"][0]["diagnostic_id"].startswith("diag-")
    assert report["mapping_diagnostics"][0]["actionability"] == "advisory"


def test_existing_v2_coverage_report_is_loaded_as_typed_result(tmp_path):
    raw = {
        "report_version": 2,
        "map": "existing-map",
        "note": "Coverage describes providers, not validation.",
        "summary": {
            "profiles": 0,
            "required_total": 0,
            "required_mapped": 0,
            "required_unmapped": 0,
            "static_required_total": 0,
            "latent_required_total": 0,
            "required_coverage_pct": 100.0,
        },
        "profiles": {},
        "mapping_diagnostics": [
            {
                "code": "mapping-source-path-not-found",
                "source": "src.missing",
            }
        ],
    }
    source_data = SimpleNamespace(value="source_data")
    data_io = SimpleNamespace(
        project_dir=tmp_path,
        ProjectFolders=SimpleNamespace(SOURCE_DATA=source_data),
        load_project_file=lambda folder, filename: raw,
    )
    generator = object.__new__(StructureMapGenerator)
    generator.app_state = SimpleNamespace(dataIO=data_io)
    generator.map_name = "existing-map"
    generator.coverage_report = None
    generator.coverage_report_path = None

    report = generator._load_coverage_report()

    assert report.report_version == 2
    assert report.mapping_diagnostics[0].actionability.value == (
        "mapping-input-required"
    )
    assert generator.coverage_report_path == (
        tmp_path / "source_data" / "existing-map_coverage.json"
    )


# ── FHIR id-typed names (WP9) ────────────────────────────────────────────────────
def test_the_map_validator_rejects_a_group_name_that_is_not_a_fhir_id():
    """`group.name`, unlike `StructureMap.name`, is a FHIR `id`. Everything here
    is built with `model_construct`, so nothing else catches it before the
    server does."""
    sm = SimpleNamespace(
        name="ValidMap",
        group=[
            SimpleNamespace(
                name="Transform-Communication_Profile",
                extends=None,
                input=[],
                rule=[],
            )
        ],
    )

    with pytest.raises(ValueError, match="A-Za-z0-9"):
        StructureMapGenerator._normalize_and_validate_structure_map(sm)


def test_the_map_validator_rejects_an_underscored_variable():
    rule = SimpleNamespace(
        name="map-thing",
        source=[SimpleNamespace(context="source", variable="src_thing")],
        target=[],
        dependent=[],
        rule=[],
    )
    sm = SimpleNamespace(
        name="ValidMap",
        group=[SimpleNamespace(name="Transform-thing", extends=None, input=[], rule=[rule])],
    )

    with pytest.raises(ValueError, match="src_thing"):
        StructureMapGenerator._normalize_and_validate_structure_map(sm)


def test_the_map_validator_accepts_hyphens_and_dots():
    rule = SimpleNamespace(
        name="map-name.family",
        source=[SimpleNamespace(context="source", variable="src-name")],
        target=[
            SimpleNamespace(
                context="target", contextType="variable", element="name",
                variable="tgt-name",
            )
        ],
        dependent=[SimpleNamespace(name="BuildName")],
        rule=[],
    )
    sm = SimpleNamespace(
        name="ValidMap",
        group=[
            SimpleNamespace(
                name="Transform-Patient",
                extends=None,
                input=[SimpleNamespace(name="source"), SimpleNamespace(name="target")],
                rule=[rule],
            )
        ],
    )

    StructureMapGenerator._normalize_and_validate_structure_map(sm)


def test_fhir_map_token_sanitizes_to_the_id_pattern():
    from helpers.utils import FHIR_ID_PATTERN, fhir_map_token

    for value in ("Transform-Communication_Profile", "Transform-74_PR_ETS_Patient",
                  "AMP PropNopro response", "a_id"):
        assert FHIR_ID_PATTERN.match(fhir_map_token(value)), value
    # A legal token is left alone.
    assert fhir_map_token("Transform-mii-pr-diagnose") == "Transform-mii-pr-diagnose"


# ── shadowed variables and duplicate rule names (WP9) ────────────────────────────
def _rule(name, *, src_ctx, src_el=None, src_var=None, tgt_ctx, tgt_el,
          tgt_var=None, param=None, children=None):
    return SimpleNamespace(
        name=name,
        source=[SimpleNamespace(context=src_ctx, element=src_el, variable=src_var)],
        target=[
            SimpleNamespace(
                context=tgt_ctx, contextType="variable", element=tgt_el,
                variable=tgt_var,
                parameter=[SimpleNamespace(valueId=param)] if param else [],
            )
        ],
        dependent=[],
        rule=children or [],
    )


def _map(rules):
    return SimpleNamespace(
        name="ValidMap",
        group=[
            SimpleNamespace(
                name="Transform-Observation",
                extends=None,
                input=[SimpleNamespace(name="source"), SimpleNamespace(name="target")],
                rule=rules,
            )
        ],
    )


def _generator_for_normalization():
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "ObservationProfile"
    generator.mapping_diagnostics = []
    return generator


def test_a_variable_shadowing_one_in_scope_is_renamed():
    """`Observation.code.coding.code` derives `src-code` at two levels. FML
    scopes a variable to its nested rules, so the inner declaration hides the
    outer one from everything below it."""
    inner = _rule(
        "map-code", src_ctx="src-coding", src_el="observationCode",
        src_var="src-code", tgt_ctx="tgt-coding", tgt_el="code",
        tgt_var="tgt-code", param="src-code",
    )
    middle = _rule(
        "map-coding", src_ctx="src-code", src_var="src-coding",
        tgt_ctx="tgt-code", tgt_el="coding", tgt_var="tgt-coding",
        children=[inner],
    )
    outer = _rule(
        "map-code", src_ctx="source", src_var="src-code", tgt_ctx="target",
        tgt_el="code", tgt_var="tgt-code", children=[middle],
    )

    _generator_for_normalization()._disambiguate_shadowed_variables(_map([outer]))

    assert outer.source[0].variable == "src-code"
    assert inner.source[0].variable == "src-code-2"
    assert inner.target[0].variable == "tgt-code-2"
    # The reference is rewritten with the declaration, or the rule would read
    # the outer binding it no longer means.
    assert inner.target[0].parameter[0].valueId == "src-code-2"
    # The middle rule still refers to the outer variables by their own names.
    assert middle.source[0].context == "src-code"
    assert middle.target[0].context == "tgt-code"


def test_a_sibling_keeps_the_outer_binding():
    """Renaming inside one rule must not reach its siblings — they are still in
    the outer scope and still mean the outer variable."""
    shadowing = _rule(
        "map-inner", src_ctx="src-code", src_el="x", src_var="src-code",
        tgt_ctx="tgt-code", tgt_el="code", param="src-code",
    )
    sibling = _rule(
        "map-display", src_ctx="src-code", src_el="y", src_var="src-display",
        tgt_ctx="tgt-code", tgt_el="display", param="src-display",
    )
    outer = _rule(
        "map-code", src_ctx="source", src_var="src-code", tgt_ctx="target",
        tgt_el="code", tgt_var="tgt-code", children=[shadowing, sibling],
    )

    _generator_for_normalization()._disambiguate_shadowed_variables(_map([outer]))

    assert shadowing.source[0].variable == "src-code-2"
    assert sibling.source[0].context == "src-code"


def test_a_map_without_shadowing_is_untouched():
    rule = _rule(
        "map-name", src_ctx="source", src_el="fullName", src_var="src-name",
        tgt_ctx="target", tgt_el="name", tgt_var="tgt-name", param="src-name",
    )

    _generator_for_normalization()._disambiguate_shadowed_variables(_map([rule]))

    assert rule.source[0].variable == "src-name"
    assert rule.target[0].variable == "tgt-name"


def test_an_identical_duplicated_rule_is_dropped():
    """Two emitters accounting for the same profile-fixed leaf produce the same
    rule twice. The second writes the same value to the same element."""
    generator = _generator_for_normalization()
    parent = SimpleNamespace(
        name="map-identifier",
        source=[], target=[], dependent=[],
        rule=[
            _rule("set-system", src_ctx="source", tgt_ctx="tgt-identifier",
                  tgt_el="system"),
            _rule("set-system", src_ctx="source", tgt_ctx="tgt-identifier",
                  tgt_el="system"),
        ],
    )
    document = _map([parent])

    generator._resolve_duplicate_rule_names(document)

    assert [rule.name for rule in parent.rule] == ["set-system"]
    assert generator.mapping_diagnostics == []


def test_two_rules_with_one_name_and_different_values_are_both_kept_and_reported():
    """A profile-fixed code and a mapped source field writing the same element
    is a decision about the user's data. The generator makes the map valid and
    says so; it does not pick a winner."""
    generator = _generator_for_normalization()
    parent = SimpleNamespace(
        name="map-category",
        source=[], target=[], dependent=[],
        rule=[
            _rule("set-coding-sct-code", src_ctx="src-category",
                  tgt_ctx="tgt-coding-sct", tgt_el="code"),
            _rule("set-coding-sct-code", src_ctx="src-category",
                  src_el="bpCatCode", src_var="src-coding-sct-code",
                  tgt_ctx="tgt-coding-sct", tgt_el="code",
                  param="src-coding-sct-code"),
        ],
    )
    document = _map([parent])

    generator._resolve_duplicate_rule_names(document)

    assert [rule.name for rule in parent.rule] == [
        "set-coding-sct-code",
        "set-coding-sct-code-2",
    ]
    assert [d["code"] for d in generator.mapping_diagnostics] == [
        "duplicate-rule-name-conflict"
    ]


def test_a_source_list_mode_without_an_element_is_dropped():
    """A list mode selects among the repetitions of `source.element`. A source
    that reads its context whole has none, and the rule compiler rejects the
    combination — so carrying it through emits a map the tool's own semantic
    check calls invalid."""
    rule = SimpleNamespace(
        name="map-identifier",
        source=[SimpleNamespace(context="source", element=None, variable="src-id",
                                listMode="first")],
        target=[],
        dependent=[],
        rule=[],
    )
    sm = SimpleNamespace(
        name="ValidMap",
        group=[SimpleNamespace(name="Transform-Patient", extends=None, input=[],
                               rule=[rule])],
    )

    StructureMapGenerator._normalize_and_validate_structure_map(sm)

    assert rule.source[0].listMode is None


def test_a_source_list_mode_on_a_real_element_survives():
    rule = SimpleNamespace(
        name="map-identifier",
        source=[SimpleNamespace(context="source", element="identifiers",
                                variable="src-id", listMode="first")],
        target=[],
        dependent=[],
        rule=[],
    )
    sm = SimpleNamespace(
        name="ValidMap",
        group=[SimpleNamespace(name="Transform-Patient", extends=None, input=[],
                               rule=[rule])],
    )

    StructureMapGenerator._normalize_and_validate_structure_map(sm)

    assert rule.source[0].listMode == "first"


def test_an_overlong_name_is_reported_rather_than_truncated():
    """A group or rule name can be referenced by `dependent.name` or `extends`,
    so shortening one silently is how a map acquires a dangling reference. 46
    names in this repository already exceed the limit, all derived from profile
    identifiers the tool does not own."""
    generator = object.__new__(StructureMapGenerator)
    generator.current_profile_name = "LongProfile"
    generator.mapping_diagnostics = []
    long_name = "Transform-" + "x" * 60
    sm = SimpleNamespace(
        name="ValidMap",
        group=[SimpleNamespace(name=long_name, extends=None, input=[], rule=[])],
    )

    generator._report_overlong_map_tokens(sm)

    assert [d["code"] for d in generator.mapping_diagnostics] == ["map-token-too-long"]
    # Reported, not rewritten.
    assert sm.group[0].name == long_name


def test_a_sanitized_name_is_not_shortened():
    from helpers.utils import fhir_map_token

    long_value = "Transform-" + "a_b" * 30
    assert len(fhir_map_token(long_value)) == len(long_value)
    assert "_" not in fhir_map_token(long_value)
