"""Map targets reachable only through a ``Reference.targetProfile``.

A profile can require data that lives in a *separate* resource without the project
ever declaring that resource as a map target. The clearest case is an extension
whose value is a ``Reference``: the extension's own StructureDefinition names what
it points at with ``targetProfile``, but nothing in the project builds that
resource. The reference then has nothing to resolve to, so the extension is emitted
carrying only its url — which violates ``ext-1`` — and is dropped.

The ``targetProfile`` is part of the profile being processed, so building the
referenced resource is not invention: the target is declared, its shape comes from
its own snapshot, and the data comes from the source field the mapping table
already binds to the referencing element. This module finds those cases and turns
each into an additional map target, so the ordinary generation path emits a
StructureMap for it like any other profile.

Scope is deliberately narrow — only an extension whose ``value[x]`` is a Reference
whose target no declared root already satisfies. Every Reference in a profile has a
``targetProfile``; synthesising for all of them would manufacture a Patient for
every ``subject`` and a Practitioner for every ``asserter``, which the bundle
assembler already wires from resources the project does declare.

Nothing here knows about any particular source format. It reasons about
``Reference``, ``targetProfile`` and ``Extension.value[x]`` — all FHIR. Source
specific semantics (a REDCap field note that says a weight is in kilograms) reach
it only as enrichment already attached to the source field metadata, so the
generated StructureMap stays portable: Matchbox executes it with no plugin present.
"""

from dataclasses import dataclass
from typing import Optional
import logging

from data_handling.url_resolver.fhir_url_resolver import (
    get_resource_from_local_package,
    resolve_url,
)
from helpers import utils
from helpers.utils import resource_identity
from mapping.fml_creator.fml_helper import attr as _attr, canonical_primitive

logger = logging.getLogger(__name__)

# A source value can legitimately land in a target element of a different but
# compatible primitive type. Kept deliberately tight: a widening that always holds
# (an integer is a valid decimal) rather than anything that needs a coercion rule.
_WIDENS_TO = {
    "decimal": {"decimal", "integer", "unsignedInt", "positiveInt"},
    "integer": {"integer", "unsignedInt", "positiveInt"},
    "string": {"string"},
    "uri": {"uri", "url", "canonical"},
    "code": {"code"},
    "dateTime": {"dateTime", "date", "instant"},
    "boolean": {"boolean"},
}


@dataclass(frozen=True)
class DerivedResource:
    """One resource to build because a Reference in another profile requires it."""

    target_profile: str
    target_type: str
    target_identity: str
    owner_profile_url: str
    owner_res_type: str
    owner_path: str
    extension_url: str
    source_field: str
    value_path: Optional[str] = None
    value_type: Optional[str] = None

    @property
    def mapping_target(self) -> Optional[str]:
        """Where the referencing element's source lands in the derived profile."""
        if not self.value_path:
            return None
        return f"{self.target_identity}.{self.value_path}"


def _release_key(version) -> str:
    """``4.0.1`` and ``4.0`` are the same release; ``5.0.0`` is a different one."""
    parts = str(version or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else ""


def _resolve_for_release(app_state, canonical, project_release):
    """Resolve a derived target, insisting on the project's FHIR release.

    Ordinary resolution prefers the cache, which can hold a resource from a
    different release than the project targets — the R5 ``bodyweight`` profile
    carries ``Observation.triggeredBy``, an element R4 has no place for. That is
    survivable when a value set is being expanded and fatal when a whole resource
    is being generated from the snapshot, so the pinned core package is consulted
    as a second opinion before the target is refused.
    """
    try:
        sd = resolve_url(canonical, app_state)
    except Exception as exc:
        logger.debug("Could not resolve derived target %s: %s", canonical, exc)
        sd = None
    if sd is not None and (
        not project_release
        or not _attr(sd, "fhirVersion", None)
        or _release_key(_attr(sd, "fhirVersion", None)) == project_release
    ):
        return sd, None
    mismatched = _release_key(_attr(sd, "fhirVersion", None)) if sd is not None else None
    try:
        raw = get_resource_from_local_package(canonical, app_state.conf)
    except Exception as exc:
        logger.debug("Local package lookup failed for %s: %s", canonical, exc)
        raw = None
    if raw:
        candidate = utils.json_to_obj(raw, raw.get("resourceType"))
        version = _release_key(_attr(candidate, "fhirVersion", None))
        if not version or version == project_release:
            return candidate, None
        mismatched = version
    return None, mismatched


def _parsed_extension_elements(field):
    """Elements of the extension definition a field references, if they were parsed.

    ``element_definition_parser`` stores the referenced extension's parsed fields
    under ``type[].profile`` when the extension SD is in the registry, and the bare
    canonical when it is not. Reading them here keeps discovery offline — no
    resolution, no network — for every extension the project already ingested.
    """
    for type_ref in _attr(field, "type", None) or []:
        if not isinstance(type_ref, dict) or type_ref.get("code") != "Extension":
            continue
        for entry in type_ref.get("profile") or []:
            if isinstance(entry, list):
                yield from (el for el in entry if isinstance(el, dict))


def reference_target_profiles(field) -> list:
    """``targetProfile`` canonicals of an extension whose ``value[x]`` is a Reference."""
    for element in _parsed_extension_elements(field):
        path = str(element.get("id") or element.get("path") or "")
        if not path.endswith("value[x]"):
            continue
        for type_ref in element.get("type") or []:
            if not isinstance(type_ref, dict) or type_ref.get("code") != "Reference":
                continue
            return [
                profile.split("|", 1)[0]
                for profile in (type_ref.get("targetProfile") or [])
                if isinstance(profile, str) and profile
            ]
    return []


def _iter_fields(fields):
    for field in fields or []:
        if not isinstance(field, dict):
            continue
        yield field
        yield from _iter_fields(field.get("children"))
        yield from _iter_fields(field.get("slices"))


def _compatible(source_type, target_type) -> bool:
    source = canonical_primitive(source_type or "") or (source_type or "")
    target = canonical_primitive(target_type or "") or (target_type or "")
    if not source or not target:
        return False
    source, target = source.lower(), target.lower()
    return source == target or source in _WIDENS_TO.get(target, set())


def _leaf_type(field):
    type_list = field.get("type")
    if isinstance(type_list, str):
        return type_list
    if isinstance(type_list, list) and len(type_list) == 1:
        entry = type_list[0]
        return entry.get("code") if isinstance(entry, dict) else entry
    return None


def infer_value_element(sd, source_type, res_type):
    """The derived profile's required value leaf that the source can fill.

    Only elements under ``value[x]`` are considered, and only required ones: an
    optional element is not what the reference exists to carry, and filling one
    would be a guess about intent rather than a reading of the profile. A unique
    type-compatible candidate is used; ambiguity is refused so the caller can
    report it rather than pick arbitrarily.

    Read from the snapshot so the answer does not depend on the profile having
    been parsed yet — the unit pin below has to be applied before that happens.
    """
    candidates = []
    for element in _snapshot_elements(sd):
        element_id = str(_attr(element, "id", "") or "")
        if f"{res_type}.value[x]" not in element_id:
            continue
        if int(_attr(element, "min", 0) or 0) < 1:
            continue
        types = _attr(element, "type", []) or []
        if len(types) != 1:
            continue
        leaf_type = _attr(types[0], "code", None)
        if not leaf_type or not _compatible(source_type, leaf_type):
            continue
        relative = element_id[len(res_type) + 1:]
        candidates.append((relative, leaf_type))
    unique = {relative for relative, _ in candidates}
    if len(unique) != 1:
        return None, None, sorted(unique)
    return candidates[0][0], candidates[0][1], []


_ALLOWED_UNITS_URL = (
    "http://hl7.org/fhir/StructureDefinition/elementdefinition-allowedUnits"
)
_UCUM_SYSTEM = "http://unitsofmeasure.org"


def _declared_unit(field):
    """The UCUM unit a source element declares via ``elementdefinition-allowedUnits``.

    A standard FHIR extension, so anything that can state a unit on an element is
    understood — the REDCap plugin's reading of a Field Note is one producer of it,
    not a special case handled here.
    """
    for extension in _attr(field, "extension", None) or []:
        if _attr(extension, "url", None) != _ALLOWED_UNITS_URL:
            continue
        concept = _attr(extension, "valueCodeableConcept", None)
        for coding in _attr(concept, "coding", None) or []:
            if _attr(coding, "system", None) == _UCUM_SYSTEM:
                code = _attr(coding, "code", None)
                if code:
                    return str(code)
    return None


def _source_field_units(source_fields) -> dict:
    units = {}
    for field in _iter_fields(source_fields):
        path = field.get("path") or field.get("id")
        unit = _declared_unit(field)
        if path and unit:
            units[path] = unit
    return units


def pin_quantity_unit(sd, value_path, unit, res_type) -> list:
    """Pin a declared unit on the derived profile's Quantity, in memory.

    ``Quantity.unit`` and ``Quantity.code`` are required by the vital-signs
    profiles but left unpinned, because the unit depends on what is being
    measured. When the bound source declares one, treating it as a profile-pinned
    value is what lets the ordinary fixed-value machinery emit it — no rule shape
    peculiar to derived resources is needed.

    Applied to the snapshot rather than to already-parsed fields: slices are
    re-derived from the snapshot further down the pipeline, so a late edit of the
    parsed dictionaries is silently discarded.
    """
    if not (value_path and unit):
        return []
    container = value_path.rsplit(".", 1)[0]
    wanted = {
        f"{res_type}.{container}.unit": "fixedString",
        f"{res_type}.{container}.code": "fixedCode",
    }
    pinned = []
    for element in _snapshot_elements(sd):
        element_id = str(_attr(element, "id", "") or "")
        attribute = wanted.get(element_id)
        if not attribute:
            continue
        if any(
            _attr(element, name, None) is not None
            for name in ("fixedString", "fixedCode", "fixedUri", "patternString")
        ):
            continue  # the profile already answers this
        try:
            setattr(element, attribute, str(unit))
        except Exception as exc:
            logger.debug("Could not pin %s on %s: %s", attribute, element_id, exc)
            continue
        pinned.append(element_id)
    return pinned


def _snapshot_elements(sd):
    snapshot = _attr(sd, "snapshot", None)
    return (_attr(snapshot, "element", []) if snapshot else []) or []


def _source_field_types(source_fields) -> dict:
    """Declared type per source path, from the source definition's parsed fields."""
    types = {}
    for field in _iter_fields(source_fields):
        path = field.get("path") or field.get("id")
        leaf = _leaf_type(field)
        if path and leaf:
            types[path] = leaf
    return types


def discover(app_state, resources, mapping_table, source_fields, record_diagnostic):
    """Reference-valued extensions whose target no declared root satisfies.

    Refusals are reported rather than skipped silently: a project that expected a
    derived resource and did not get one should be able to see why from the run.
    """
    declared, owners = set(), []
    for entry in resources or []:
        for url, fields in entry.items():
            obj = app_state.registry.get_obj_by_name(url)
            if obj is None or obj.data is None:
                continue
            declared.add(url)
            owners.append((url, obj, fields))

    table = mapping_table if isinstance(mapping_table, dict) else {}
    source_types = _source_field_types(source_fields)
    source_units = _source_field_units(source_fields)
    found, seen = [], set()

    for owner_url, obj, fields in owners:
        res_type = getattr(obj.data, "type", None) or obj.res_type
        owner_identity = resource_identity(obj.data, owner_url)
        for field in _iter_fields(fields):
            profiles = reference_target_profiles(field)
            if not profiles:
                continue
            owner_path = str(field.get("id") or field.get("path") or "")
            source_field = _authored_source(table, owner_path, res_type, owner_identity)
            if not source_field:
                # No data is bound to the reference, so there is nothing to build the
                # target from. The url-only extension is reported by the ext-1 guard.
                continue
            if len(profiles) > 1:
                record_diagnostic(
                    "derived-resource-ambiguous-target",
                    f"{owner_path} references {len(profiles)} target profiles "
                    f"({', '.join(profiles)}); a derived resource is only built when "
                    "the extension names exactly one.",
                    path=owner_path,
                    severity="warning",
                )
                continue
            target_profile = profiles[0]
            if target_profile in declared:
                continue
            if target_profile in seen:
                continue
            derived = _build_spec(
                app_state,
                target_profile,
                owner_url,
                res_type,
                owner_path,
                field,
                source_field,
                source_types.get(source_field),
                source_units.get(source_field),
                record_diagnostic,
            )
            if derived is not None:
                seen.add(target_profile)
                found.append(derived)
    return found


def _authored_source(table, owner_path, res_type, owner_identity):
    """The source field the mapping table binds to this element, if any."""
    relative = owner_path.split(".", 1)[1] if "." in owner_path else owner_path
    wanted = {owner_path, f"{res_type}.{relative}", f"{owner_identity}.{relative}"}
    for source_key, value in table.items():
        target = value.get("target") if isinstance(value, dict) else value
        if isinstance(target, str) and target in wanted:
            return source_key
    return None


def _build_spec(
    app_state,
    target_profile,
    owner_url,
    owner_res_type,
    owner_path,
    field,
    source_field,
    source_type,
    source_unit,
    record_diagnostic,
):
    from parser.resource_parser import resource_obj_parser
    from fhir_spec.context import ensure_fhir_spec_context

    try:
        project_release = _release_key(
            ensure_fhir_spec_context(app_state).fhir_version
        )
    except Exception as exc:
        logger.debug("No FHIR spec context available: %s", exc)
        project_release = ""

    sd, mismatched_release = _resolve_for_release(
        app_state, target_profile, project_release
    )
    if sd is None:
        if mismatched_release:
            record_diagnostic(
                "derived-resource-release-mismatch",
                f"{owner_path} references target profile {target_profile}, but the "
                f"only definition available declares FHIR {mismatched_release} while "
                f"this project targets {project_release}. Generating from it would "
                "emit elements the project's release does not define, so no derived "
                "resource is built.",
                path=owner_path,
                severity="error",
            )
        else:
            record_diagnostic(
                "derived-resource-unresolvable",
                f"{owner_path} references target profile {target_profile}, which "
                "could not be resolved; no derived resource is built for it.",
                path=owner_path,
                severity="warning",
            )
        return None
    kind = _attr(sd, "kind", "") or ""
    target_type = _attr(sd, "type", None)
    if "resource" not in str(kind) or not target_type:
        record_diagnostic(
            "derived-resource-not-a-resource",
            f"{owner_path} references {target_profile}, whose kind is "
            f"'{kind or 'unknown'}'; only resource profiles can be built.",
            path=owner_path,
            severity="warning",
        )
        return None

    # Both of these read and write the snapshot, so they must run before the profile
    # is parsed: slices are re-derived from the snapshot downstream, which discards
    # any edit made to the parsed representation.
    # Pin onto whichever definition will actually be generated from. A profile
    # already in the registry belongs to the project, not to us, so it is used as
    # found — only a definition this module introduced may be constrained.
    registered = app_state.registry.get_obj_by_name(target_profile)
    ours = registered is None
    if ours:
        try:
            registered = app_state.registry.add_fhir_object(sd, "StructureDefinition")
        except ReferenceError:
            registered, ours = app_state.registry.get_obj_by_name(target_profile), False
    if registered is None:
        return None
    target_sd = registered.data if registered.data is not None else sd

    value_path, value_type, ambiguous = infer_value_element(
        target_sd, source_type, target_type
    )
    if value_path is None:
        record_diagnostic(
            "derived-resource-value-not-inferable",
            f"{owner_path} binds source '{source_field}'"
            + (f" (type {source_type})" if source_type else "")
            + f", but {target_profile} offers "
            + (
                f"{len(ambiguous)} type-compatible required value elements "
                f"({', '.join(ambiguous)})"
                if ambiguous
                else "no type-compatible required value element"
            )
            + "; the derived resource is built without a value provider.",
            path=owner_path,
            severity="warning",
        )
    pinned = []
    if value_path and source_unit and ours:
        pinned = pin_quantity_unit(target_sd, value_path, source_unit, target_type)

    # Slices are re-derived from the snapshot at parse time, so a pin only takes
    # effect if parsing happens after it.
    if pinned or not registered.mappable_fields:
        registered.mappable_fields = []
        resource_obj_parser.parse_resource(registered, app_state)

    return DerivedResource(
        target_profile=target_profile,
        target_type=target_type,
        target_identity=resource_identity(sd, target_profile),
        owner_profile_url=owner_url,
        owner_res_type=owner_res_type,
        owner_path=owner_path,
        extension_url=str(field.get("extension_url") or _extension_url(field) or ""),
        source_field=source_field,
        value_path=value_path,
        value_type=value_type,
    )


def _extension_url(field):
    for type_ref in _attr(field, "type", None) or []:
        if isinstance(type_ref, dict) and type_ref.get("code") == "Extension":
            for canonical in type_ref.get("profile_canonical") or []:
                if isinstance(canonical, str) and canonical:
                    return canonical
    return None


def registry_entries(app_state, derived):
    """``{url: mappable_fields}`` entries so the generator emits a map per target."""
    entries = []
    for spec in derived:
        obj = app_state.registry.get_obj_by_name(spec.target_profile)
        if obj is None or not obj.mappable_fields:
            continue
        entries.append({spec.target_profile: obj.mappable_fields})
    return entries


def mapping_overlay(base_table, spec):
    """The mapping table to use while generating one derived profile's map.

    The referencing source is re-pointed at the derived profile's value element.
    It cannot simply be added to the shared table: that table is one target per
    source, and this source is already bound to the extension that references the
    derived resource. Scoping the extra binding to this one map keeps the authored
    table the single record of authored intent.
    """
    table = dict(base_table or {})
    target = spec.mapping_target
    if target:
        table[spec.source_field] = target
    return table
