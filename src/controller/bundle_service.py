from copy import deepcopy
from typing import List, Dict, Any, Optional
import json
import uuid
import re
import logging
logger = logging.getLogger(__name__)

class BundleService:
    """Assembles FHIR Bundles with optional reference resolution."""

    @staticmethod
    def create_bundle(
        resources: List[Dict[str, Any]],
        bundle_type: str = "transaction",
        registry=None,
        structure_maps: Optional[List[Dict]] = None,
        resolve_references: bool = True,
        plugins: Optional[list] = None,
        source_record: Optional[Dict[str, Any]] = None,
        map_outputs: Optional[Dict[str, list]] = None,
        external_reference_defaults: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Create a FHIR Bundle from a list of resources.

        When resolve_references is True (default):
          - Assigns a temporary urn:uuid to every resource for intra-bundle reference wiring.
            Resources do NOT get an 'id' field — the FHIR server assigns real IDs on POST.
          - Wires cross-resource references found in TODO_resolve_reference rules
            inside the provided structure_maps using the urn:uuid values.
          - If registry is provided, warns about required reference fields that
            had no TODO rule and therefore no wired value.
          - Applies only explicitly configured ``external_reference_defaults``
            to reference fields that are still empty. These declarations are
            intended for cross-project/external targets and never overwrite a
            mapped or bundle-wired reference.
          - Sets entry.fullUrl to the urn:uuid per the FHIR transaction bundle spec.
        """
        for res in resources:
            BundleService.normalize_list_cardinality(res)

        # Map Python object id → temporary urn:uuid (never written to the resource itself)
        urn_map: Dict[int, str] = {}
        if resolve_references:
            for res in resources:
                if isinstance(res, dict) and "resourceType" in res:
                    urn_map[id(res)] = f"urn:uuid:{uuid.uuid4()}"

            array_paths = BundleService._build_array_paths(registry)
            specs = BundleService._specs_from_structure_maps(structure_maps or [])
            profile_type_map = BundleService._build_profile_type_map(registry)
            prohibited_map = BundleService._build_profile_prohibited_map(registry)
            if specs:
                BundleService._wire_references(
                    resources, specs, urn_map, array_paths, profile_type_map,
                    prohibited_map,
                )
            if registry:
                BundleService._resolve_unresolved_required(
                    resources, registry, specs, urn_map, array_paths, prohibited_map
                )

            if external_reference_defaults:
                BundleService._apply_external_reference_defaults(
                    resources, external_reference_defaults, array_paths, prohibited_map
                )

            if registry:
                drop = BundleService._collect_unresolvable_required(
                    resources, registry, profile_type_map
                )
                if drop:
                    resources = [r for r in resources if id(r) not in drop]

            for res in resources:
                BundleService.normalize_list_cardinality(res)

        bundle = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": bundle_type,
            "entry": [],
        }
        for res in resources:
            if not res or "resourceType" not in res:
                continue
            request: Dict[str, Any] = {"method": "POST", "url": res["resourceType"]}
            cond = BundleService._if_none_exist(res)
            if cond:
                request["ifNoneExist"] = cond
            entry: Dict[str, Any] = {"resource": res, "request": request}
            urn = urn_map.get(id(res))
            if urn:
                entry["fullUrl"] = urn
            bundle["entry"].append(entry)

        ref_by_obj = {}
        for entry in bundle["entry"]:
            res = entry["resource"]
            ref = entry.get("fullUrl")
            if not ref and res.get("id"):
                ref = f"{res['resourceType']}/{res['id']}"
            if ref:
                ref_by_obj[id(res)] = ref
        map_output_refs = {
            url: refs
            for url, outputs in (map_outputs or {}).items()
            if (refs := [ref_by_obj[id(r)] for r in outputs if id(r) in ref_by_obj])
        }

        for plugin in plugins or []:
            hook = getattr(plugin, "post_bundle", None)
            if not callable(hook):
                continue
            try:
                result = hook(
                    bundle,
                    {
                        "source_record": source_record,
                        "structure_maps": structure_maps or [],
                        "map_outputs": map_output_refs,
                    },
                )
                if isinstance(result, dict) and result.get("resourceType") == "Bundle":
                    bundle = result
            except Exception as e:
                logger.error(
                    "post_bundle hook of plugin '%s' failed (%s: %s) — bundle left unchanged.",
                    getattr(plugin, "plugin_id", type(plugin).__name__),
                    type(e).__name__,
                    e,
                )
        return bundle

    @staticmethod
    def _if_none_exist(res: dict):
        """Return an ``ifNoneExist`` match query for a resource's business key, or None.

        """
        from urllib.parse import quote
        ident = res.get("identifier")
        if isinstance(ident, list) and ident and isinstance(ident[0], dict):
            system, value = ident[0].get("system"), ident[0].get("value")
            if system and value:
                return f"identifier={system}|{quote(str(value), safe='')}"
        if res.get("resourceType") == "Medication":
            coding = (res.get("code", {}) or {}).get("coding") or []
            if coding and coding[0].get("system") and coding[0].get("code"):
                return f"code={coding[0]['system']}|{quote(str(coding[0]['code']), safe='')}"
        return None

    @staticmethod
    def _specs_from_structure_maps(structure_maps: list) -> list:
        """Return (source_type, field_path, target_type, source_profile) tuples from
        TODO_resolve_reference rules."""
        # Build profile-name → FHIR-base-type lookup from SM structure + group inputs.
        # e.g. MinimalCondition3 → Condition
        profile_to_type: dict = {}
        for sm in structure_maps:
            for group in sm.get("group", []):
                fhir_type = next(
                    (
                        inp["type"]
                        for inp in group.get("input", [])
                        if inp.get("mode") == "target"
                    ),
                    None,
                )
                if fhir_type:
                    for struct in sm.get("structure", []):
                        if struct.get("mode") == "target":
                            name = struct.get("url", "").split("/")[-1]
                            profile_to_type[name] = fhir_type

        specs = []
        for sm in structure_maps:
            # Canonical URL of this map's target profile — the profile that declares
            # the references collected below.
            source_profile = next(
                (
                    struct.get("url")
                    for struct in sm.get("structure", [])
                    if struct.get("mode") == "target" and struct.get("url")
                ),
                None,
            )
            for group in sm.get("group", []):
                for rule in group.get("rule", []):
                    BundleService._collect_todo_refs(
                        rule, specs, profile_to_type, source_profile
                    )
        # remove duplicates as well while preserving order, in case multiple rules point to the same reference field
        return list(dict.fromkeys(specs))

    @staticmethod
    def _collect_todo_refs(
        rule: dict, specs: list, profile_to_type: dict = None,
        source_profile: str = None,
    ) -> None:
        """Recursively collect (source_type, field_path, target_type) tuples from rules with
        TODO_resolve_reference_*, parsing the target_type from the rule documentation."""
        documentation = rule.get("documentation", "")
        marker = "FHIRBRIDGE_REFERENCE:"
        if documentation.startswith(marker):
            try:
                contract = json.loads(documentation[len(marker) :])
                source_type = contract["sourceType"]
                field_path = contract["path"]
                target_types = contract["targetTypes"]
                if not (
                    isinstance(source_type, str)
                    and source_type
                    and isinstance(field_path, str)
                    and field_path
                    and isinstance(target_types, list)
                    and target_types
                    and all(isinstance(item, str) and item for item in target_types)
                ):
                    raise ValueError("missing sourceType, path, or targetTypes")
                if field_path.endswith("[x]"):
                    field_path = field_path[: -len("[x]")] + "Reference"
                resolved_types = []
                for target in target_types:
                    raw = target.split("/")[-1] if "/" in target else target
                    resolved_types.append(
                        (profile_to_type or {}).get(raw, raw)
                    )
                specs.append(
                    (
                        source_type,
                        field_path,
                        "|".join(dict.fromkeys(resolved_types)),
                        source_profile,
                        "bundled",
                        contract.get("match", "byOrder"),
                        contract.get("sourceKey"),
                        contract.get("targetKey"),
                        contract.get("referenceMode", "urn"),
                        contract.get("targetProfile"),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Ignoring malformed bundled-reference contract: %s", exc)
        elif rule.get("name", "").lower().startswith("todo-resolve-reference-"):
            m = re.match(
                r"Reference<(\w+)\.([^>]+)>\s*(?:→|->)\s*(\S+)",
                documentation,
            )
            if m:
                raw = m.group(3).rstrip(" —-").rstrip("—").strip()
                # Strip full URL to last segment
                if "/" in raw:
                    raw = raw.split("/")[-1]
                # Resolve profile name to FHIR base type if available
                if profile_to_type:
                    raw = profile_to_type.get(raw, raw)
                field_path = m.group(2)
                if field_path.endswith("[x]"):
                    field_path = field_path[: -len("[x]")] + "Reference"
                specs.append((m.group(1), field_path, raw, source_profile))
        for nested in rule.get("rule", []):
            BundleService._collect_todo_refs(
                nested, specs, profile_to_type, source_profile
            )

    @staticmethod
    def _build_array_paths(registry) -> dict:
        """Per FHIR source type, the set of resource-relative paths whose cardinality
        max is >1 ('*' or n>1). Used to wire references into list backbones correctly.
        """
        result: dict = {}
        if not registry:
            return result
        for obj in registry.registry_objects.values():
            ftype = getattr(getattr(obj, "data", None), "type", None)
            fields = getattr(obj, "mappable_fields", None)
            if not ftype or not fields:
                continue
            BundleService._collect_array_paths(fields, result.setdefault(ftype, set()))
        return result

    @staticmethod
    def _collect_array_paths(fields: list, out: set) -> None:
        """Recursively collect the set of resource-relative paths whose cardinality max is >1."""
        for field in fields:
            if not isinstance(field, dict):
                continue
            path = field.get("path", "")
            parts = path.split(".")
            rel = ".".join(parts[1:]) if len(parts) > 1 else path
            mx = str((field.get("cardinality", {}) or {}).get("max", "1"))
            if rel and (mx == "*" or (mx.isdigit() and int(mx) > 1)):
                out.add(rel)
            for key in ("children", "slices"):
                if field.get(key):
                    BundleService._collect_array_paths(field[key], out)

    @staticmethod
    @staticmethod
    def _build_profile_type_map(registry) -> dict:
        """Map a profile's last URL segment (id) → its base FHIR ``type``.
        """
        profile_type_map: dict = {}
        if not registry:
            return profile_type_map
        for obj in registry.registry_objects.values():
            url = getattr(obj.data, "url", "") or ""
            fhir_type = getattr(obj.data, "type", None)
            if url and fhir_type:
                profile_type_map[url.split("/")[-1]] = fhir_type
        return profile_type_map

    @staticmethod
    def _build_profile_prohibited_map(registry) -> dict:
        """Map profile canonical URL"""
        prohibited: dict = {}
        if not registry:
            return prohibited
        for obj in registry.registry_objects.values():
            url = getattr(obj.data, "url", "") or ""
            paths = getattr(obj, "prohibited_paths", None)
            if url and paths:
                prohibited[url] = set(paths)
        return prohibited

    @staticmethod
    def _resource_profiles(resource: dict) -> list:
        """Version-stripped canonical URLs from a resource meta.profile"""
        actual = (resource.get("meta") or {}).get("profile") or []
        if isinstance(actual, str):
            actual = [actual]
        return [p.split("|", 1)[0] for p in actual if isinstance(p, str)]

    @staticmethod
    def _path_is_prohibited(field_path: str, prohibited: set) -> bool:
        """check if field_path is forbidden by prohibited
        """
        if not prohibited:
            return False
        if field_path in prohibited:
            return True
        for s in prohibited:
            if s.endswith("[x]"):
                base = s[:-3]
                if field_path.startswith(base) and field_path[len(base):][:1].isupper():
                    return True
            if field_path.endswith("[x]"):
                base = field_path[:-3]
                if s.startswith(base) and s[len(base):][:1].isupper():
                    return True
        return False

    @staticmethod
    def _wiring_forbidden(resource: dict, field_path: str, prohibited_map: dict) -> bool:
        """check if wiring field_path onto resource is forbidden by any of
        the resource own declared profiles"""
        if not prohibited_map:
            return False
        for prof in BundleService._resource_profiles(resource):
            if BundleService._path_is_prohibited(field_path, prohibited_map.get(prof, set())):
                logger.info(
                    "Refusing to wire %s.%s — prohibited (max=0) by profile %s",
                    resource.get("resourceType"),
                    field_path,
                    prof,
                )
                return True
        return False

    @staticmethod
    def _resource_matches_obj(resource: dict, obj_url: str, source_type: str) -> bool:
        """Whether a fallback/co-occurrence rule derived from the profile"""
        profiles = BundleService._resource_profiles(resource)
        if profiles:
            return obj_url in profiles
        return resource.get("resourceType") == source_type

    @staticmethod
    def _wire_references(
        resources: list, specs: list, urn_map: dict, array_paths: dict = None,
        profile_type_map: dict = None, prohibited_map: dict = None,
    ) -> None:
        """Wire references for the given (source_type, field_path, target_type) specs using the
        urn_map for reference values"""
        array_paths = array_paths or {}
        profile_type_map = profile_type_map or {}
        by_type: dict = {}
        for r in resources:
            rt = r.get("resourceType")
            if rt:
                by_type.setdefault(rt, []).append(r)

        for spec in specs:
            source_type, field_path, target_type = spec[0], spec[1], spec[2]
            source_profile = spec[3] if len(spec) > 3 else None
            sources = by_type.get(source_type, [])
            if source_profile:
                scoped = [
                    s
                    for s in sources
                    if not BundleService._resource_profiles(s)
                    or source_profile in BundleService._resource_profiles(s)
                ]
                sources = scoped
            strategy = spec[4] if len(spec) > 4 else "legacy"
            match = spec[5] if len(spec) > 5 else "byOrder"
            source_key = spec[6] if len(spec) > 6 else None
            target_key = spec[7] if len(spec) > 7 else None
            reference_mode = spec[8] if len(spec) > 8 else "urn"
            target_profile = spec[9] if len(spec) > 9 else None
            # Resolve each candidate (base type or profile id) to a present type.
            candidates = [
                profile_type_map.get(t, t) for t in target_type.split("|") if t
            ]
            chosen = next((c for c in candidates if by_type.get(c)), None)
            if match == "all":
                targets = [
                    target
                    for candidate in candidates
                    for target in by_type.get(candidate, [])
                ]
            else:
                targets = by_type.get(chosen, []) if chosen else []
            if not sources or not targets:
                logger.warning(
                    f"Reference {source_type}.{field_path} → {target_type}: "
                    f"no candidate target type in bundle — skipping"
                )
                continue
            parts = field_path.split(".")
            for i, source in enumerate(sources):
                if len(parts) > 1 and BundleService._get_nested(source, parts[:-1]) is None:
                    logger.info(
                        f"Skipping {source_type}.{field_path}: parent backbone absent "
                        f"(optional, unpopulated) — not materialising it for a reference."
                    )
                    continue
                avail = [t for t in targets if t is not source]
                if target_profile:
                    canonical = target_profile.split("|", 1)[0]
                    avail = [
                        target
                        for target in avail
                        if canonical in BundleService._resource_profiles(target)
                    ]
                if not avail:
                    continue
                if BundleService._wiring_forbidden(source, field_path, prohibited_map):
                    continue
                selected = []
                if strategy != "bundled" or match == "byOrder":
                    selected = [avail[i % len(avail)]]
                elif match == "singleton":
                    if len(avail) != 1:
                        logger.warning(
                            "Reference %s.%s requires one target, found %d — skipping",
                            source_type,
                            field_path,
                            len(avail),
                        )
                        continue
                    selected = avail
                elif match == "identifier":
                    source_value = BundleService._reference_match_value(
                        source, source_key, source_type
                    )
                    selected = [
                        target
                        for target in avail
                        if BundleService._reference_match_value(
                            target, target_key, target.get("resourceType")
                        )
                        == source_value
                    ]
                    if source_value is None or len(selected) != 1:
                        logger.warning(
                            "Reference %s.%s identifier correlation found %d matches — skipping",
                            source_type,
                            field_path,
                            len(selected) if source_value is not None else 0,
                        )
                        continue
                elif match == "all":
                    selected = avail
                else:
                    logger.warning(
                        "Reference %s.%s has unsupported match policy %r — skipping",
                        source_type,
                        field_path,
                        match,
                    )
                    continue

                refs = [
                    BundleService._reference_value(
                        target, urn_map, reference_mode
                    )
                    for target in selected
                ]
                refs = list(dict.fromkeys(refs))
                if match == "all":
                    BundleService._set_nested_all(
                        source,
                        parts,
                        [{"reference": ref} for ref in refs],
                        array_paths.get(source_type, set()),
                    )
                else:
                    BundleService._set_nested(
                        source,
                        parts,
                        {"reference": refs[0]},
                        array_paths.get(source_type, set()),
                    )
                logger.info(
                    "Wired %s.%s → %s",
                    source_type,
                    field_path,
                    ", ".join(refs),
                )

    @staticmethod
    def _reference_match_value(resource: dict, path: str, resource_type: str):
        if not isinstance(path, str) or not path:
            return None
        parts = path.split(".")
        if parts and parts[0] == resource_type:
            parts = parts[1:]
        value = BundleService._get_nested(resource, parts)
        if isinstance(value, (dict, list)):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return value

    @staticmethod
    def _reference_value(target: dict, urn_map: dict, mode: str) -> str:
        if mode == "relative" and target.get("id"):
            return f"{target.get('resourceType')}/{target['id']}"
        return urn_map.get(
            id(target),
            f"{target.get('resourceType')}/{target.get('id', 'unknown')}",
        )

    @staticmethod
    def _set_nested_all(
        obj: dict, path: list, values: list, array_paths: set = None
    ) -> None:
        """Assign all values to a repeating leaf without collapsing to the first."""
        if len(path) == 1:
            obj[path[0]] = values
            return
        parent = BundleService._get_nested(obj, path[:-1])
        if isinstance(parent, list):
            parent = parent[0] if parent else None
        if isinstance(parent, dict):
            parent[path[-1]] = values
            return
        BundleService._set_nested(obj, path, values, array_paths)

    @staticmethod
    def _apply_external_reference_defaults(
        resources: list, declarations: list, array_paths: dict = None,
        prohibited_map: dict = None,
    ) -> None:
        """Fill still-empty references from explicit external declarations.
        """
        array_paths = array_paths or {}
        if not isinstance(declarations, list):
            logger.warning(
                "external_reference_defaults must be a list of declarations; ignoring %r",
                type(declarations).__name__,
            )
            return

        for declaration in declarations:
            if not isinstance(declaration, dict):
                logger.warning("Ignoring non-object external reference declaration")
                continue

            raw_path = declaration.get("path")
            source_type = declaration.get("source_type")
            if not isinstance(raw_path, str) or not raw_path.strip():
                logger.warning("External reference declaration has no path; ignoring it")
                continue

            path = raw_path.strip().strip(".")
            if "." in path:
                prefix, relative = path.split(".", 1)
                if source_type and prefix == source_type:
                    path = relative
                elif source_type and prefix[:1].isupper():
                    logger.warning(
                        "External reference declaration path %s conflicts with source_type %s; "
                        "ignoring it",
                        raw_path,
                        source_type,
                    )
                    continue
                elif not source_type and prefix[:1].isupper():
                    source_type, path = prefix, relative
            if not isinstance(source_type, str) or not source_type.strip():
                logger.warning(
                    "External reference declaration for %s has no source_type; ignoring it",
                    raw_path,
                )
                continue
            source_type = source_type.strip()
            if path.endswith("[x]"):
                path = path[: -len("[x]")] + "Reference"

            value = declaration.get("value")
            if value is None:
                value = {
                    key: deepcopy(declaration[key])
                    for key in ("reference", "type", "identifier", "display")
                    if declaration.get(key) is not None
                }
            if not isinstance(value, dict):
                logger.warning(
                    "External reference declaration for %s must provide a Reference object",
                    raw_path,
                )
                continue
            value = deepcopy(value)
            literal = value.get("reference")
            identifier = value.get("identifier")
            if not (isinstance(literal, str) and literal.strip()) and not (
                isinstance(identifier, dict) and identifier
            ):
                logger.warning(
                    "External reference declaration for %s needs reference or identifier",
                    raw_path,
                )
                continue
            if isinstance(literal, str):
                value["reference"] = literal.strip()
                if not value.get("type") and "://" not in literal and "/" in literal:
                    candidate = literal.split("/", 1)[0]
                    if candidate and not candidate.startswith("urn:"):
                        value["type"] = candidate

            expected_profiles = declaration.get("source_profile")
            if isinstance(expected_profiles, str):
                expected_profiles = [expected_profiles]
            elif expected_profiles is None:
                expected_profiles = []
            elif not isinstance(expected_profiles, list):
                logger.warning(
                    "source_profile for %s must be a string or list; ignoring declaration",
                    raw_path,
                )
                continue

            parts = path.split(".")
            for resource in resources:
                if (
                    not isinstance(resource, dict)
                    or resource.get("resourceType") != source_type
                ):
                    continue
                if expected_profiles and not BundleService._matches_any_profile(
                    resource, expected_profiles
                ):
                    continue
                if BundleService._get_nested(resource, parts) is not None:
                    continue
                if len(parts) > 1 and BundleService._get_nested(
                    resource, parts[:-1]
                ) is None:
                    continue
                if BundleService._wiring_forbidden(resource, path, prohibited_map):
                    continue
                BundleService._set_nested(
                    resource,
                    parts,
                    deepcopy(value),
                    array_paths.get(source_type, set()),
                )
                logger.info(
                    "Applied declared external reference to %s.%s",
                    source_type,
                    path,
                )

    @staticmethod
    def _matches_any_profile(resource: dict, expected_profiles: list) -> bool:
        """Match a resource's ``meta.profile`` against canonical URLs or profile ids."""
        actual = (resource.get("meta") or {}).get("profile") or []
        if isinstance(actual, str):
            actual = [actual]
        actual = [
            profile.split("|", 1)[0]
            for profile in actual
            if isinstance(profile, str)
        ]
        for expected in expected_profiles:
            if not isinstance(expected, str) or not expected:
                continue
            expected = expected.split("|", 1)[0]
            if "/" in expected:
                if expected in actual:
                    return True
            else:
                if any(
                    isinstance(profile, str)
                    and profile.rstrip("/").rsplit("/", 1)[-1] == expected
                    for profile in actual
                ):
                    return True
        return False

    @staticmethod
    def _resolve_unresolved_required(
        resources: list,
        registry,
        wired_specs: list,
        urn_map: dict,
        array_paths: dict = None,
        prohibited_map: dict = None,
    ) -> None:
        """For **required** reference fields that are still empty after SM-declared wiring:
        - If there is exactly one candidate of the target type in the bundle, auto-wire it.
        - Otherwise warn once per (source_type, field_path).
        """
        array_paths = array_paths or {}
        by_type: dict = {}
        for r in resources:
            rt = r.get("resourceType")
            if rt:
                by_type.setdefault(rt, []).append(r)

        # Profile id (last URL segment) → base type, e.g. "WAVESPatient" → "Patient".
        profile_type_map = BundleService._build_profile_type_map(registry)

        for obj in registry.registry_objects.values():
            if not obj.is_root or not obj.mappable_fields:
                continue
            source_type = getattr(obj.data, "type", None)
            if not source_type or source_type not in by_type:
                continue
            obj_url = getattr(obj.data, "url", "") or ""
            for (
                field_path,
                raw_target,
                is_required,
                _ref_only,
            ) in BundleService._iter_reference_fields(obj.mappable_fields):
                if not is_required:
                    continue
                present = [
                    profile_type_map.get(c, c)
                    for c in raw_target.split("|")
                    if by_type.get(profile_type_map.get(c, c))
                ]
                target_type = (
                    present[0]
                    if len(present) == 1
                    else (
                        raw_target
                        if present
                        else profile_type_map.get(raw_target, raw_target)
                    )
                )
                targets = by_type.get(target_type, [])
                sources = by_type.get(source_type, [])
                path_parts = field_path.split(".")
                empty_sources = [
                    res
                    for res in sources
                    if BundleService._resource_matches_obj(res, obj_url, source_type)
                    and not BundleService._get_nested(res, path_parts)
                ]
                if not empty_sources:
                    continue
                if len(targets) == 1:
                    for res in empty_sources:
                        if targets[0] is res:
                            continue
                        if BundleService._wiring_forbidden(
                            res, field_path, prohibited_map
                        ):
                            continue
                        ref = urn_map.get(
                            id(targets[0]),
                            f"{target_type}/{targets[0].get('id', 'unknown')}",
                        )
                        BundleService._set_nested(
                            res,
                            field_path.split("."),
                            {"reference": ref},
                            array_paths.get(source_type, set()),
                        )
                        logger.info(f"Auto-wired {source_type}.{field_path} → {ref}")
                elif is_required:
                    logger.warning(
                        f"Required reference {source_type}.{field_path} → {target_type} "
                        f"has no value and no TODO rule — mapping may be incomplete"
                    )

    @staticmethod
    def _collect_unresolvable_required(
        resources: list, registry, profile_type_map: dict = None
    ) -> set:
        """get the id of resources to drop"""
        profile_type_map = profile_type_map or {}
        by_type: dict = {}
        for r in resources:
            rt = r.get("resourceType")
            if rt:
                by_type.setdefault(rt, []).append(r)

        CONTEXT_TYPES = {"Patient", "Encounter"}
        drop: set = set()
        for obj in registry.registry_objects.values():
            if not getattr(obj, "is_root", False) or not getattr(
                obj, "mappable_fields", None
            ):
                continue
            source_type = getattr(obj.data, "type", None)
            if not source_type or source_type not in by_type:
                continue
            obj_url = getattr(obj.data, "url", "") or ""
            for field_path, raw_target, is_required, ref_only in BundleService._iter_reference_fields(
                obj.mappable_fields
            ):
                if not is_required or not ref_only:
                    continue  # optional, or a choice satisfiable without the reference
                targets = [t for t in raw_target.split("|") if t]
                if targets and all(t in CONTEXT_TYPES for t in targets):
                    continue  # context-only ref: absence does not invalidate the resource
                present = [
                    profile_type_map.get(c, c)
                    for c in raw_target.split("|")
                    if by_type.get(profile_type_map.get(c, c))
                ]
                if present:
                    continue  # a candidate target exists — wiring resolves it
                parts = field_path.split(".")
                if len(parts) != 1:
                    continue
                for res in by_type.get(source_type, []):
                    if not BundleService._resource_matches_obj(
                        res, obj_url, source_type
                    ):
                        continue
                    if not BundleService._get_nested(res, parts):
                        drop.add(id(res))
        return drop

    @staticmethod
    def _iter_reference_fields(fields: list):
        """Recursively yield (relative_field_path, reference_target_last_segment,
        is_required, ref_only)
        """
        for field in fields:
            path = field.get("path", "")
            parts = path.split(".")
            relative = ".".join(parts[1:]) if len(parts) > 1 else path

            target = field.get("reference_target")
            if target and target not in ("", "TYPE-NOT-FOUND"):
                raw = target.split("/")[-1] if "/" in target else target
                cardinality = field.get("cardinality", {})
                is_required = (
                    bool(field.get("is_required")) or int(cardinality.get("min", 0)) > 0
                )
                ftype = field.get("type")
                if isinstance(ftype, list):
                    codes = [t.get("code") for t in ftype if isinstance(t, dict)]
                    ref_only = bool(codes) and all(c == "Reference" for c in codes)
                else:
                    ref_only = True
                # A choice element ``foo[x]`` holding a Reference value must serialise
                # as ``fooReference`` per FHIR (not the literal ``[x]`` placeholder);
                # since this branch is Reference-typed, expand the leaf accordingly.
                if relative.endswith("[x]"):
                    relative = relative[: -len("[x]")] + "Reference"
                yield relative, raw, is_required, ref_only

            if field.get("children"):
                yield from BundleService._iter_reference_fields(field["children"])
            if field.get("slices"):
                yield from BundleService._iter_reference_fields(field["slices"])

    @staticmethod
    def _set_nested(
        obj: dict, path: list, value, array_paths: set = None, _prefix: str = ""
    ) -> None:
        """Set `value` at a nested dotted path, building backbones along the way."""

        array_paths = array_paths or set()
        key = path[0]
        cur = f"{_prefix}.{key}" if _prefix else key
        is_array = cur in array_paths
        existing = obj.get(key)

        if len(path) == 1:
            if is_array or isinstance(existing, list):
                lst = (
                    existing
                    if isinstance(existing, list)
                    else ([] if existing is None else [existing])
                )
                obj[key] = lst
                if not lst:
                    lst.append({})
                if isinstance(value, dict) and isinstance(lst[0], dict):
                    lst[0].update(value)
                else:
                    lst[0] = value
                return
            obj[key] = value
            return

        if is_array or isinstance(existing, list):
            lst = (
                existing
                if isinstance(existing, list)
                else ([] if existing is None else [existing])
            )
            obj[key] = lst
            if not lst or not isinstance(lst[0], dict):
                if not lst:
                    lst.append({})
                elif not isinstance(lst[0], dict):
                    lst[0] = {}
            BundleService._set_nested(lst[0], path[1:], value, array_paths, cur)
            return

        if not isinstance(existing, dict):
            obj[key] = {}
        BundleService._set_nested(obj[key], path[1:], value, array_paths, cur)

    @staticmethod
    def _get_nested(obj: dict, path: list):
        """Return the value at a nested dotted path, or None if absent. Descends into the
        first element of any list backbone encountered (e.g. diagnosis[0].condition)."""
        for key in path:
            if isinstance(obj, list):
                obj = obj[0] if obj else None
            if not isinstance(obj, dict):
                return None
            obj = obj.get(key)
        return obj

    _FIELD_SPEC_CACHE: dict = {}

    @staticmethod
    def _field_specs_for_class(model_cls):
        """Per fhir.resources model class, map each FHIR field name: (is_list, element_class).
        element_class is the (possibly complex/backbone) type carried by the field annotation,
        used to recurse; None for primitives. Driven entirely by the fhir.resources models.
        """
        cached = BundleService._FIELD_SPEC_CACHE.get(model_cls)
        if cached is not None:
            return cached
        from parser.resource_parser.fhir_type_introspection import (
            extract_inner_type,
            get_fhir_type_name,
        )

        specs: dict = {}
        for name, fi in getattr(model_cls, "model_fields", {}).items():
            alias = getattr(fi, "alias", None) or name
            if alias.startswith("_") or alias in ("resourceType", "fhir_comments"):
                continue
            ann = getattr(fi, "annotation", None)
            if ann is None:
                continue
            inner, is_list, _, _ = extract_inner_type(ann)
            elem_cls = BundleService._resolve_model_class(get_fhir_type_name(inner))
            specs[alias] = (is_list, elem_cls)
        BundleService._FIELD_SPEC_CACHE[model_cls] = specs
        return specs

    @staticmethod
    def _resolve_model_class(type_name, _cache: dict = {}):
        """Import the fhir.resources R4B model class for a FHIR type name, or None."""
        if not isinstance(type_name, str):
            return None
        if type_name in _cache:
            return _cache[type_name]
        cls = None
        try:
            import importlib

            mod = importlib.import_module(f"fhir.resources.R4B.{type_name.lower()}")
            cls = getattr(mod, type_name, None)
        except Exception:
            cls = None
        if cls is not None and not hasattr(cls, "model_fields"):
            cls = None
        _cache[type_name] = cls
        return cls

    @staticmethod
    def normalize_list_cardinality(resource: dict) -> dict:
        """Wrap any element that fhir.resources defines as a list (base-type cardinality >1)
        but that is present as a bare object/scalar, so the output is spec-conformant JSON
        regardless of matchbox honoring a profile's narrowed cardinality."""
        if not isinstance(resource, dict):
            return resource
        rt = resource.get("resourceType")
        if not rt:
            return resource
        try:
            import importlib

            mod = importlib.import_module(f"fhir.resources.R4B.{rt.lower()}")
            cls = getattr(mod, rt)
        except Exception as e:
            logger.debug(
                "No fhir.resources model for resourceType %s (%s: %s) — "
                "skipping list-cardinality normalization.",
                rt, type(e).__name__, e,
            )
            return resource
        BundleService._normalize_obj(resource, cls)
        return resource

    @staticmethod
    def _normalize_obj(obj: dict, cls) -> None:
        """Recursively normalize list cardinality on obj according to the field specs for the class."""
        if not isinstance(obj, dict) or cls is None:
            return
        specs = BundleService._field_specs_for_class(cls)
        for name, val in list(obj.items()):
            spec = specs.get(name)
            if spec is None:
                continue
            is_list, elem_cls = spec
            if is_list and not isinstance(val, list):
                val = obj[name] = [val]
            for child in (val if isinstance(val, list) else [val]):
                BundleService._normalize_obj(child, elem_cls)
