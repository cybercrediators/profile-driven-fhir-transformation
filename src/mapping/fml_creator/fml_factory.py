from fhir.resources.R4B.structuremap import (
    StructureMap,
    StructureMapGroup,
    StructureMapGroupInput,
    StructureMapGroupRule,
    StructureMapGroupRuleTarget,
    StructureMapGroupRuleTargetParameter,
    StructureMapGroupRuleSource,
)

from data_handling.url_resolver.fhir_url_resolver import resolve_url

import logging

from mapping.fml_creator.fml_helper import (
    find_reference_fields,
    is_primitive_type,
    parse_slice_info,
    get_transform_for_type,
    clean_field_name,
    fhir_type_suffix,
    canonical_primitive,
    local_element_name,
    is_extension_type,
    is_reference_type,
    type_codes,
    fixed_scalar_literal,
    attr as _attr,
    BASE_META_SUFFIXES,
    has_fixed_value,
    is_meaningful_modifier_extension,
)
from parser.resource_parser.fhir_type_introspection import get_complex_type_fields
from functools import lru_cache

from mapping.fml_creator.fml_extension import _ExtensionRulesMixin
from mapping.fml_creator.fml_slice import _SliceRulesMixin
from mapping.fml_creator.fml_coded import _CodedRulesMixin
from mapping.target_tree import TargetTree

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _choice_type_direct_fields(type_code: str) -> frozenset:
    """Direct (leaf) field names of a complex datatype, derived from fhir.resources — used to
    infer which value[x] choice type a set of mapped sub-fields belongs to."""
    return frozenset(
        f["path"].split(".")[-1] for f in get_complex_type_fields(type_code)
    )


def _is_system_value_pseudo(field) -> bool:
    """True for the ``.value`` child whose type is a FHIRPath ``System.*`` primitive"""

    if (field.get("path", "") or "").split(".")[-1] != "value":
        return False
    t = field.get("type")
    codes = (
        [t]
        if isinstance(t, str)
        else [(x.get("code") if isinstance(x, dict) else x) for x in (t or [])]
    )
    return any(
        str(c or "").startswith("http://hl7.org/fhirpath/System.") for c in codes
    )


class FMLRuleFactory(_ExtensionRulesMixin, _SliceRulesMixin, _CodedRulesMixin):
    def __init__(
        self,
        app_state,
        overwrite=False,
        map_url: str = "http://example.org",
        plugins=None,
    ):
        self.app_state = app_state
        self.overwrite = overwrite
        self.map_url = map_url
        self.plugins = plugins or []
        self.source_field_types: dict = {}
        self.source_field_max: dict = {}
        self.collection_rules: dict = {}
        self._target_tree_cache = (None, None)
        self.diagnostics = []

    def record_diagnostic(self, code, message, **details):
        if not hasattr(self, "diagnostics"):
            self.diagnostics = []
        diagnostic = {
            "code": code,
            "message": message,
            "profile": getattr(self, "_current_profile_id", "unknown"),
            **details,
        }
        if diagnostic not in self.diagnostics:
            self.diagnostics.append(diagnostic)
        logger.warning("%s: %s", code, message)
        return diagnostic

    def profile_tree(self):
        """Profile-constrained target tree for the profile currently being emitted.

        Built without datatype introspection: prohibition and parentage come from the
        snapshot alone, and introspecting every complex type on every profile is far too
        expensive for the big modules (icu has 69 profiles).
        """
        sd = getattr(self, "_current_profile_sd", None)
        if sd is None:
            return None
        cached_sd, cached_tree = self._target_tree_cache
        if cached_sd is sd:
            return cached_tree
        try:
            tree = TargetTree.from_snapshot(sd)
        except Exception as exc:
            logger.debug("target tree unavailable (%s: %s)", type(exc).__name__, exc)
            tree = None
        self._target_tree_cache = (sd, tree)
        return tree

    def is_prohibited_target(self, path: str) -> bool:
        """True when the profile forbids this element (``max = 0``) — see N4."""
        if not path:
            return False
        tree = self.profile_tree()
        return bool(tree and tree.is_prohibited(path))

    def _resolve_choice_element(
        self, field, base_name: str, path: str, automapped_mappings
    ):
        """Concrete element name for a polymorphic element, or ``None`` if ambiguous.

        Resolution order: the profile's own narrowing to a single type, then the mapped
        source field's declared type. Several candidates left means the author has to say
        which one — guessing produced the wrong variant (backlog item 25).
        """
        tree = self.profile_tree()
        tree_key = field.get("id") or path
        candidates = [
            c for c in ((tree.effective_types(tree_key) if tree else []) or [])
            if c and c != "N/A"
        ]
        if not candidates:
            candidates = [
                c
                for c in (type_codes(field.get("type")) or [])
                if c and c != "N/A"
            ]
        if not candidates:
            candidates = [c for c in (field.get("choice_types") or []) if c]
        if len(candidates) > 1 and automapped_mappings:
            src_id = automapped_mappings.get(path) or ""
            src_name = src_id.split(".")[-1] if src_id else ""
            src_type = self.source_field_types.get(src_name, "") if src_name else ""
            preferred = (canonical_primitive(src_type) or src_type) if src_type else None
            if preferred:
                narrowed = [c for c in candidates if c.lower() == preferred.lower()]
                if narrowed:
                    candidates = narrowed
        if len(candidates) != 1:
            return None
        suffix = self._choice_suffix(candidates[0])
        return f"{base_name}{suffix}" if suffix else base_name

    def _choice_suffix(self, choice_type: str) -> str:
        if not isinstance(choice_type, str):
            return ""
        return fhir_type_suffix(choice_type)

    @staticmethod
    def _as_local_element(source_id: str) -> str:
        """Strip type-qualified prefix from a source field ID for use as a FHIRPath element.
        Thin wrapper over fml_helper.local_element_name (shared with the questionnaire creator).
        """
        return local_element_name(source_id)

    def _apply_repeating_list_modes(
        self, cardinality, source, target, rule_name, field_path=None
    ):
        collection = (getattr(self, "collection_rules", None) or {}).get(field_path)
        if collection is not None:
            if collection.source_list_mode:
                source.listMode = collection.source_list_mode
            if collection.target_list_modes:
                target.listMode = list(collection.target_list_modes)
            if collection.list_rule_id:
                target.listRuleId = collection.list_rule_id
            return
        if cardinality.get("max") in ["*", "n"]:
            target.listMode = ["share"]
            target.listRuleId = rule_name

    @staticmethod
    def _normalize_single_type_field(field):
        """Return a generation-only view of a single-type parser field.

        """
        codes = type_codes(field.get("type"))
        if len(codes) != 1:
            return field, None

        normalized = dict(field)
        normalized["type"] = codes[0]
        raw_types = field.get("type")
        if isinstance(raw_types, list) and raw_types:
            type_info = raw_types[0]
            if (
                isinstance(type_info, dict)
                and type_info.get("type_structure")
                and not normalized.get("type_structure")
            ):
                normalized["type_structure"] = type_info["type_structure"]
        return normalized, codes[0]

    @staticmethod
    def _provided_type_structure(
        fields,
        automapped_mappings,
        *,
        include_fixed=True,
        unsliced_fixed_only=False,
    ):
        """Keep only complex-type branches with an explicit provider.

        """
        mappings = automapped_mappings or {}

        def _provided(path):
            return bool(
                path
                and any(key == path or key.startswith(path + ".") for key in mappings)
            )

        kept = []
        for child in fields or []:
            if not isinstance(child, dict):
                continue
            child_copy = dict(child)
            nested = FMLRuleFactory._provided_type_structure(
                child.get("type_structure") or child.get("children") or [],
                mappings,
                include_fixed=include_fixed,
                unsliced_fixed_only=unsliced_fixed_only,
            )
            if child.get("type_structure") is not None:
                child_copy["type_structure"] = nested
            elif child.get("children") is not None:
                child_copy["children"] = nested
            if (
                _provided(child.get("path"))
                or (
                    include_fixed
                    and has_fixed_value(child.get("fixed_value"))
                    and (
                        not unsliced_fixed_only
                        or ":" not in str(child.get("id") or "")
                    )
                )
                or nested
            ):
                kept.append(child_copy)
        return kept

    @classmethod
    def _unsliced_generation_field(cls, field, automapped_mappings):
        """Build the base-element view used for an authored unsliced provider.

        Named-slice descendants can share the same FHIR ``path`` as the base
        element.  Once those descendants have been attached to the parser tree,
        recursively emitting the full tree would also recreate unrelated slice
        scaffolds (Ontario's translation extensions are the concrete example).
        The generic entry therefore contains only explicitly addressed branches;
        named slices are emitted separately by their slice-qualified rules.
        """
        base = dict(field)
        base["slices"] = []
        base["slicing"] = {}
        for key in ("children", "type_structure"):
            if field.get(key) is not None:
                base[key] = cls._provided_type_structure(
                    field.get(key) or [],
                    automapped_mappings,
                    include_fixed=True,
                    unsliced_fixed_only=True,
                )

        raw_types = field.get("type")
        if isinstance(raw_types, list):
            normalized_types = []
            for raw_type in raw_types:
                if not isinstance(raw_type, dict):
                    normalized_types.append(raw_type)
                    continue
                type_copy = dict(raw_type)
                if raw_type.get("type_structure") is not None:
                    type_copy["type_structure"] = cls._provided_type_structure(
                        raw_type.get("type_structure") or [],
                        automapped_mappings,
                        include_fixed=True,
                        unsliced_fixed_only=True,
                    )
                normalized_types.append(type_copy)
            base["type"] = normalized_types
        return base

    @staticmethod
    def _has_descendant_slices(field):
        """Return whether a complex field contains a sliced child container."""
        pending = list(field.get("children") or [])
        pending.extend(field.get("type_structure") or [])
        raw_types = field.get("type")
        if isinstance(raw_types, list):
            for raw_type in raw_types:
                if isinstance(raw_type, dict):
                    pending.extend(raw_type.get("type_structure") or [])

        while pending:
            child = pending.pop()
            if not isinstance(child, dict):
                continue
            if child.get("slices"):
                return True
            pending.extend(child.get("children") or [])
            pending.extend(child.get("type_structure") or [])
        return False

    @staticmethod
    def _unsliced_provider_keys(field, automapped_mappings):
        """Return providers addressed to the unsliced element or its descendants.

        A key below ``Condition.code.coding`` is intentionally distinct from
        ``Condition.code.coding:icd10``.  The former may populate an ordinary entry
        when slicing is open; the latter explicitly selects a named slice.
        """
        path = field.get("path")
        if not path:
            return []
        return [
            key
            for key in (automapped_mappings or {})
            if key == path or key.startswith(path + ".")
        ]

    @staticmethod
    def _slicing_rules(field):
        """Return the FHIR slicing rule, accepting parser variants on child slices."""
        slicing = field.get("slicing") or {}
        if not slicing:
            slicing = next(
                (
                    candidate.get("slicing")
                    for candidate in (field.get("slices") or [])
                    if isinstance(candidate, dict) and candidate.get("slicing")
                ),
                {},
            )
        return slicing.get("rules", "open")

    def _reference_base_type(self, url):
        """Resolve a canonical profile URL to its FHIR base type (e.g. a MinimalCondition3
        profile URL → 'Condition'). Returns None when it cannot be resolved."""
        if not url or not isinstance(url, str):
            return None
        try:
            sd = resolve_url(url, self.app_state)
        except Exception as e:
            logger.debug(
                "Could not resolve profile URL %s to a base type (%s: %s)",
                url,
                type(e).__name__,
                e,
            )
            return None
        if not sd:
            return None
        t = _attr(sd, "type")
        if isinstance(t, list):
            t = t[0] if t else None
        return t

    def create_field_rules(
        self,
        res_type,
        fields,
        parent_source_context,
        parent_target_context,
        create_references=True,
        automapped_mappings=None,
        parent_path=None,
    ):
        rules = []

        def _iter_slices(slice_fields):
            for candidate in slice_fields or []:
                if not isinstance(candidate, dict):
                    continue
                yield candidate
                yield from _iter_slices(candidate.get("slices"))

        def _has_direct_slice_provider(parent, candidate):
            identity = self._slice_identity(parent, candidate)
            return any(
                key == identity or key.startswith(identity + ".")
                for key in (automapped_mappings or {})
            )

        def _reslice_satisfies_parent(parent, candidate):
            return any(
                ((child.get("cardinality") or {}).get("min", 0) or 0) > 0
                or _has_direct_slice_provider(parent, child)
                for child in _iter_slices(candidate.get("slices"))
            )

        for field in fields:
            logger.info("Creating rule for field %s", field.get("id"))

            if _is_system_value_pseudo(field):
                logger.info(
                    "Skipping primitive .value pseudo-element %s", field.get("id")
                )
                continue

            is_choice_field = (
                bool(field.get("is_type_choice"))
                or field.get("type") == "choice"
                or "[x]" in str(field.get("path", ""))
            )
            has_nonext_slices = (not is_choice_field) and any(
                isinstance(s, dict) and not is_extension_type(s.get("type"))
                for s in (field.get("slices") or [])
            )

            unsliced_provider_keys = (
                self._unsliced_provider_keys(field, automapped_mappings)
                if has_nonext_slices
                else []
            )
            slicing_rules = (
                self._slicing_rules(field) if has_nonext_slices else None
            )

            def _append_base_rule():
                base_field = (
                    self._unsliced_generation_field(field, automapped_mappings)
                    if has_nonext_slices
                    else field
                )
                rule = self.create_mappable_field_rule(
                    base_field,
                    res_type,
                    parent_source_context,
                    parent_target_context,
                    create_references=create_references,
                    automapped_mappings=automapped_mappings,
                    parent_path=parent_path,
                )
                if rule:
                    rules.append(rule)

            # FHIR open slicing permits entries that do not match a named slice.
            # Preserve an explicitly authored unsliced provider as exactly one base
            # entry instead of dropping it or broadcasting it into every slice.
            # openAtEnd has the same semantics but constrains its output order.
            if not has_nonext_slices or (
                unsliced_provider_keys and slicing_rules == "open"
            ):
                _append_base_rule()
            elif unsliced_provider_keys and slicing_rules == "closed":
                identity = field.get("id") or field.get("path")
                self.record_diagnostic(
                    "unsliced-provider-for-closed-slicing",
                    f"Unsliced mapping target {identity} cannot create an entry in "
                    "closed slicing. Use a slice-qualified mapping target.",
                    path=identity,
                    provider_keys=unsliced_provider_keys,
                    slicing_rules=slicing_rules,
                    severity="error",
                )
            elif unsliced_provider_keys and slicing_rules not in (
                "open",
                "openAtEnd",
            ):
                identity = field.get("id") or field.get("path")
                self.record_diagnostic(
                    "unsupported-slicing-rules",
                    f"Unsliced mapping target {identity} uses unsupported slicing "
                    f"rules {slicing_rules!r}.",
                    path=identity,
                    provider_keys=unsliced_provider_keys,
                    slicing_rules=slicing_rules,
                    severity="error",
                )

            def _tc(v):
                t = v.get("type")
                if isinstance(t, list):
                    t = (
                        (t[0].get("code") if isinstance(t[0], dict) else t[0])
                        if t
                        else None
                    )
                return t

            field_type_code = _tc(field)
            slices = (
                []
                if is_choice_field
                else list(_iter_slices(field.get("slices", [])))
            )
            for slice_field in slices:
                if (
                    slice_field.get("path")
                    and field.get("path")
                    and slice_field.get("path") != field.get("path")
                ):
                    # Some parser trees retain descendant slices in an ancestor's
                    # flattened slice collection as well as below their real
                    # element. Emitting one here hoists it to the wrong target
                    # context (e.g. Coding.display.extension under
                    # CodeableConcept). Its owning child will handle it when that
                    # child branch is actually selected.
                    logger.info(
                        "Skipping descendant slice %s while emitting %s; "
                        "its element path is %s.",
                        slice_field.get("id") or slice_field.get("sliceName"),
                        field.get("path"),
                        slice_field.get("path"),
                    )
                    continue
                if (
                    slice_field.get("slices")
                    and not _has_direct_slice_provider(field, slice_field)
                    and _reslice_satisfies_parent(field, slice_field)
                ):
                    # A required/emitted reslice is already an instance of its parent
                    # slice. Do not create a second unconditional parent instance.
                    continue
                sl_tc = _tc(slice_field)
                if (
                    not is_extension_type(slice_field.get("type"))
                    and field_type_code
                    and sl_tc
                    and sl_tc != field_type_code
                ):
                    logger.info(
                        "Skipping misplaced descendant slice %s (type %s != field %s type %s)",
                        slice_field.get("sliceName"),
                        sl_tc,
                        field.get("path"),
                        field_type_code,
                    )
                    continue
                # Extension slices keep their dedicated handler (url-discriminated, works well).
                if is_extension_type(slice_field.get("type")):
                    slice_rule = self.create_mappable_field_rule(
                        slice_field,
                        res_type,
                        parent_source_context,
                        parent_target_context,
                        create_references=create_references,
                        automapped_mappings=automapped_mappings,
                        parent_path=parent_path,
                    )
                else:
                    slice_rule = self._create_slice_instance_rule(
                        field,
                        slice_field,
                        res_type,
                        parent_source_context,
                        parent_target_context,
                        automapped_mappings,
                    )
                if slice_rule:
                    rules.append(slice_rule)

            if (
                has_nonext_slices
                and unsliced_provider_keys
                and slicing_rules == "openAtEnd"
            ):
                _append_base_rule()

        return rules

    def create_mappable_field_rule(
        self,
        field,
        res_type,
        parent_source_context,
        parent_target_context,
        create_references=True,
        automapped_mappings=None,
        parent_path=None,
    ):
        """create a StructureMapGroupRule for a field that can be mapped (non-slice, non-choice)"""
        path = field["path"]
        collection = (getattr(self, "collection_rules", None) or {}).get(path)
        if collection is not None and collection.invalid:
            logger.warning(
                "Omitting %s because its collection correlation declaration is invalid.",
                path,
            )
            return None
        field_type = field.get("type", "string")
        children = field.get("children", [])
        cardinality = field.get("cardinality", {})

        if parent_path and path.startswith(f"{parent_path}."):
            clean_path = path[len(parent_path) + 1 :]
        elif path.startswith(f"{res_type}."):
            clean_path = path[len(res_type) + 1 :]
        else:
            clean_path = path

        if path.endswith(BASE_META_SUFFIXES) and not is_meaningful_modifier_extension(
            field
        ):
            return None

        if is_extension_type(field_type):
            return self.create_extension_rule(
                field,
                clean_path,
                parent_source_context,
                parent_target_context,
                automapped_mappings=automapped_mappings,
            )

        if type_codes(field_type) == ["N/A"]:
            has_na_provider = bool(
                automapped_mappings
                and any(
                    k == path or k.startswith(path + ".") for k in automapped_mappings
                )
            )
            if not has_na_provider:
                logger.info(
                    "Skipping unsourced field %s: unresolved datatype (N/A) cannot be created.",
                    path,
                )
                return None

        def _ref_target(f, ft):
            rt = f.get("reference_target")
            if rt:
                if isinstance(rt, str) and rt.startswith("http"):
                    return self._reference_base_type(rt) or rt.split("/")[-1]
                return rt
            if isinstance(ft, list):
                candidates = []
                for t in ft:
                    if isinstance(t, dict):
                        for p in t.get("targetProfile") or []:
                            base = self._reference_base_type(p) or p.split("/")[-1]
                            if base and base not in candidates:
                                candidates.append(base)
                if candidates:
                    return "|".join(candidates)
            return None

        is_reference = is_reference_type(field_type)
        reference_target = _ref_target(field, field_type) if is_reference else None

        # Regular field handling
        field_name = clean_path.split(".")[-1]

        base_field_name = (
            field_name.replace("[x]", "") if "[x]" in field_name else field_name
        )

        # Handle fixed values for all types (primitives, etc.)
        fixed_value = field.get("fixed_value")
        if has_fixed_value(fixed_value):
            # N2: a dot left in clean_path means this leaf sits below the element the
            # current target context represents, i.e. its parent chain was never
            # materialised. Attaching the leaf here would write it at the wrong level —
            # `Procedure.reasonReference.type` (fixedUri "Task") became `Procedure.type`,
            # which R4 does not define, and the engine aborted the whole transform.
            # A nested fixed leaf may only be emitted once its parent owns a context.
            if "." in clean_path:
                logger.info(
                    "Not emitting fixed value for %s: its parent chain has no target "
                    "context (optional parent without a provider).",
                    path,
                )
                return None
            is_pattern = bool(field.get("is_pattern"))
            min_c = int((cardinality or {}).get("min", 0) or 0)
            is_req = min_c > 0 or bool(field.get("is_required"))
            has_provider = bool(
                automapped_mappings
                and any(
                    k == path or k.startswith(path + ".") for k in automapped_mappings
                )
            )
            codes = type_codes(field_type)
            is_coded = len(codes) == 1 and codes[0] in (
                "code",
                "Coding",
                "CodeableConcept",
            )
            if is_pattern and not is_req and not has_provider:
                logger.info(
                    "Optional pattern-only element %s has no source — emitting nothing.",
                    path,
                )
                return None
            if is_pattern and has_provider and not is_coded:
                pass
            else:
                return self._create_fixed_value_rule(
                    field_type,
                    fixed_value,
                    field_name,
                    parent_target_context,
                    parent_source_context,
                )

        var_suffix = clean_field_name(base_field_name)

        normalized_field, single_type = self._normalize_single_type_field(field)
        coded_type = (
            single_type
            if single_type in ("code", "Coding", "CodeableConcept")
            else None
        )
        coded_builder_type = coded_type in ("Coding", "CodeableConcept") or (
            coded_type == "code" and isinstance(field_type, str)
        )
        is_raw_complex_coded = isinstance(field_type, list) and coded_type in (
            "Coding",
            "CodeableConcept",
        )
        has_coded_provider = bool(
            automapped_mappings
            and any(
                key == path or key.startswith(path + ".") for key in automapped_mappings
            )
        )
        has_coding_slices = is_raw_complex_coded and (
            bool(field.get("slices"))
            or self._has_descendant_slices(field)
            or bool(self._fixed_coding_slices(normalized_field))
        )
        may_route_raw_complex = not is_raw_complex_coded or (
            has_coded_provider and not has_coding_slices
        )
        is_slice, slice_info = parse_slice_info(field_name, field)
        has_binding_options = (
            bool(field.get("options")) and coded_builder_type and may_route_raw_complex
        )
        has_fixed_coding_system = (
            coded_type in ("Coding", "CodeableConcept")
            and may_route_raw_complex
            and self._fixed_system_from_field(normalized_field) is not None
        )
        if has_binding_options or has_fixed_coding_system:
            return self.create_conditional_coding_rules(
                normalized_field,
                base_field_name,
                parent_source_context,
                parent_target_context,
                is_slice,
                slice_info if is_slice else None,
                automapped_mappings=automapped_mappings,
            )

        if (
            coded_type in ("Coding", "CodeableConcept")
            and has_coded_provider
            and not has_coding_slices
        ):
            field = normalized_field
            field_type = coded_type
            children = field.get("children", [])
            cardinality = field.get("cardinality", {})
            if field.get("type_structure"):
                field = dict(field)
                field["type_structure"] = self._provided_type_structure(
                    field["type_structure"], automapped_mappings
                )

        has_complex_descendant_provider = bool(
            automapped_mappings
            and any(
                key.startswith(path + ".") or key.startswith(path + ":")
                for key in automapped_mappings
            )
        )
        if (
            isinstance(field_type, list)
            and single_type
            and single_type not in ("code", "Coding", "CodeableConcept", "Reference")
            and not is_primitive_type(single_type)
            and has_complex_descendant_provider
            and normalized_field.get("type_structure")
            and not field.get("slices")
        ):
            field = dict(normalized_field)
            field["type_structure"] = self._provided_type_structure(
                normalized_field["type_structure"], automapped_mappings
            )
            field_type = single_type
            children = field.get("children", [])
            cardinality = field.get("cardinality", {})

        if is_slice:
            # print(slice_info)
            rule_name = f"map-{slice_info['base_info']}-{slice_info['slice_name']}"
            display_name = slice_info["base_info"]
        else:
            rule_name = f"map-{clean_field_name(base_field_name)}"
            display_name = base_field_name

        rule = StructureMapGroupRule.model_construct()
        rule.name = rule_name

        source_element = f"TODO_MAP_{base_field_name.upper()}"
        if automapped_mappings:
            prof = getattr(self, "_current_profile_id", None)
            qkey = (
                f"{prof}.{path[len(res_type) + 1:]}"
                if prof and path.startswith(f"{res_type}.")
                else None
            )
            if qkey and qkey in automapped_mappings:
                source_element = self._as_local_element(automapped_mappings[qkey])
            elif path in automapped_mappings:
                source_element = self._as_local_element(automapped_mappings[path])

        if "." in clean_path and not is_slice and source_element.startswith("TODO"):
            return None

        source = StructureMapGroupRuleSource.model_construct()
        source.context = parent_source_context
        source.variable = f"src-{var_suffix}"
        rule.source = [source]

        # Create target
        target = StructureMapGroupRuleTarget.model_construct()
        target.context = parent_target_context
        target.element = display_name
        target.variable = f"tgt-{var_suffix}"

        self._apply_repeating_list_modes(
            cardinality, source, target, rule_name, field_path=path
        )
        if is_slice:
            target.listMode = ["share"]

        choice_variant_mapped = bool(
            "[x]" in path
            and automapped_mappings
            and any(
                key.startswith(path + ".") or key.startswith(path + ":")
                for key in automapped_mappings
            )
        )
        if (
            is_reference
            and create_references
            and reference_target
            and not choice_variant_mapped
        ):
            ref_src_key = None
            if automapped_mappings:
                prof = getattr(self, "_current_profile_id", None)
                qref_key = (
                    f"{prof}.{path[len(res_type) + 1:]}.reference"
                    if prof and path.startswith(f"{res_type}.")
                    else None
                )
                if qref_key and qref_key in automapped_mappings:
                    ref_src_key = qref_key
                elif f"{path}.reference" in automapped_mappings:
                    ref_src_key = f"{path}.reference"
            if ref_src_key and automapped_mappings:
                disp_key = ref_src_key[: -len(".reference")] + ".display"
                disp_src = (
                    self._as_local_element(automapped_mappings[disp_key])
                    if disp_key in automapped_mappings
                    else None
                )
                return self._reference_source_rule(
                    rule,
                    display_name,
                    var_suffix,
                    parent_source_context,
                    parent_target_context,
                    self._as_local_element(automapped_mappings[ref_src_key]),
                    disp_src,
                    cardinality=cardinality,
                    rule_name=rule_name,
                )
            return self._reference_placeholder_rule(
                rule,
                path,
                clean_path,
                res_type,
                base_field_name,
                display_name,
                reference_target,
            )

        type_structure = field.get("type_structure", [])
        is_choice = (
            "[x]" in path
            or bool(field.get("is_type_choice"))
            or field.get("type") == "choice"
        )
        suppress_source_element = self._apply_scalar_target_transform(
            field_type, target, display_name, source.variable, is_choice=is_choice
        )
        rule.target = [target]
        # Complex targets normally use a context-only source because their
        # legacy descendants are flat source fields. A collection parent is
        # different: binding its repeated source element is what establishes
        # one nested source/target context pair per repetition.
        if collection is not None or not suppress_source_element:
            source.element = source_element
        if collection is not None and collection.source_key:
            # The enclosing source iteration provides the correlation scope.
            # Requiring one key per selected item prevents silently emitting an
            # unidentifiable repetition when a key was declared.
            source.check = f"{collection.source_key}.count() = 1"

        # Add documentation (enhanced for slices)
        rule.documentation = self.generate_rule_documentation(
            field, is_slice, slice_info
        )
        if collection is not None:
            key_doc = (
                f"; key {collection.source_key} -> {collection.target_key}"
                if collection.source_key and collection.target_key
                else (
                    f"; key {collection.source_key}"
                    if collection.source_key
                    else ""
                )
            )
            rule.documentation = (
                f"{rule.documentation} | Collection correlation: "
                f"{collection.source} -> {collection.target}{key_doc}"
            )

        has_choice_provider = bool(
            automapped_mappings
            and any(
                key == path
                or key.startswith(path + ".")
                or key.startswith(path + ":")
                for key in automapped_mappings
            )
        )
        if (
            field.get("is_type_choice")
            or field_type == "choice"
            or ("[x]" in path and has_choice_provider)
        ):
            return self._choice_field_rule(
                field,
                rule,
                field_name,
                var_suffix,
                source_element,
                parent_source_context,
                parent_target_context,
                cardinality,
                is_slice,
                slice_info,
                automapped_mappings,
            )
        # Handle nested fields with inline nested rules
        if children:
            # print("CHILDREN IN MAPPABEL RULE FIELD GENERATOR FOUND!")
            nested_rules = self.create_field_rules(
                res_type,
                children,
                parent_source_context=f"src-{var_suffix}",
                parent_target_context=f"tgt-{var_suffix}",
                create_references=create_references,
                automapped_mappings=automapped_mappings,
                parent_path=path,
            )
            # For slices, add discriminator hint rule first
            if is_slice and slice_info.get("discriminator"):
                discriminator_rule = self.create_discriminator_hint_rule(
                    slice_info, f"src-{field_name}", f"tgt-{field_name}"
                )
                if discriminator_rule:
                    nested_rules.insert(0, discriminator_rule)
            if nested_rules:
                rule.rule = nested_rules

        # Handle expanded complex type structure
        elif type_structure:
            logger.info(
                f"Generating rules for expanded complex type: {field_type} ({len(type_structure)} fields)"
            )

            nested_rules = self.create_field_rules(
                field_type,
                type_structure,
                parent_source_context=f"src-{var_suffix}",
                parent_target_context=f"tgt-{var_suffix}",
                create_references=create_references,
                automapped_mappings=automapped_mappings,
                parent_path=path,
            )
            if nested_rules:
                rule.rule = nested_rules

        return rule

    def _reference_placeholder_rule(
        self,
        rule,
        path,
        clean_path,
        res_type,
        base_field_name,
        display_name,
        reference_target,
    ):
        """placeholder rule for a Reference field, to be resolved by the bundle assembler post-transform"""
        if path.startswith(f"{res_type}."):
            full_rel_path = path[len(res_type) + 1 :]
        else:
            full_rel_path = clean_path
        if "." in clean_path:
            return None
        type_part = f"-{clean_field_name(res_type)}" if res_type else ""
        rule.name = (
            f"TODO-resolve-reference{type_part}-{clean_field_name(base_field_name)}"
        )
        rule.documentation = f"Reference<{res_type}.{full_rel_path}> → {reference_target} — resolve via bundle assembler"
        # This is an instruction for the post-transform bundle assembler, not an
        # executable target assignment.  A target element without a context violates
        # StructureMap's target invariants.
        rule.target = None
        return rule

    def _reference_source_rule(
        self,
        rule,
        display_name,
        var_suffix,
        parent_source_context,
        parent_target_context,
        ref_source_element,
        display_source_element=None,
        cardinality=None,
        rule_name=None,
    ):
        """populate references directly"""
        source = StructureMapGroupRuleSource.model_construct()
        source.context = parent_source_context
        source.variable = f"src-{var_suffix}"
        rule.source = [source]

        target = StructureMapGroupRuleTarget.model_construct()
        target.context = parent_target_context
        target.element = display_name
        target.variable = f"tgt-{var_suffix}"
        target.transform = "create"
        target.parameter = [{"valueString": "Reference"}]
        self._apply_repeating_list_modes(
            cardinality or {}, source, target, rule_name or f"map-{var_suffix}"
        )
        rule.target = [target]

        children = [
            self._scalar_child_copy_rule(
                f"map-{var_suffix}-reference",
                source.variable,
                ref_source_element,
                target.variable,
                "reference",
            )
        ]
        if display_source_element:
            children.append(
                self._scalar_child_copy_rule(
                    f"map-{var_suffix}-display",
                    source.variable,
                    display_source_element,
                    target.variable,
                    "display",
                )
            )
        rule.rule = children
        rule.documentation = (
            f"Reference.reference from mapped source '{ref_source_element}' "
            "(literal reference; not deferred to bundle assembler)"
        )
        return rule

    @staticmethod
    def _scalar_child_copy_rule(
        name, src_context, src_element, tgt_context, tgt_element
    ):
        """A leaf rule copying a flat source field into a scalar target element."""
        child = StructureMapGroupRule.model_construct()
        child.name = name
        s = StructureMapGroupRuleSource.model_construct()
        s.context = src_context
        s.element = src_element
        s.variable = f"src-{tgt_element}"
        child.source = [s]
        t = StructureMapGroupRuleTarget.model_construct()
        t.context = tgt_context
        t.element = tgt_element
        t.transform = "copy"
        t.parameter = [{"valueId": f"src-{tgt_element}"}]
        child.target = [t]
        return child

    def _apply_scalar_target_transform(
        self, field_type, target, display_name, source_variable, is_choice=False
    ):
        """apply the appropriate transform to a target for a scalar field type (primitive or complex)"""
        type_code = field_type
        if isinstance(field_type, list):
            if len(field_type) > 0:
                first = field_type[0]
                if isinstance(first, dict):
                    type_code = first.get("code", "BackboneElement")
                elif isinstance(first, str):
                    type_code = first
                else:
                    type_code = str(first)
            else:
                type_code = "BackboneElement"
        elif not isinstance(field_type, str):
            type_code = str(field_type)

        if not is_primitive_type(type_code):
            target.transform = "create"
            target.parameter = [{"valueString": type_code}]
            if (
                is_choice
                and isinstance(type_code, str)
                and display_name.endswith(type_code)
                and display_name != type_code
            ):
                target.element = display_name[: -len(type_code)]
            return True

        transform_info = get_transform_for_type(type_code, source_variable)
        if transform_info:
            target.transform = transform_info["transform"]
            params = transform_info.get("parameters")
            if params:
                target.parameter = params
        return False

    def _choice_field_rule(
        self,
        field,
        rule,
        field_name,
        var_suffix,
        source_element,
        parent_source_context,
        parent_target_context,
        cardinality,
        is_slice,
        slice_info,
        automapped_mappings,
    ):
        """create a StructureMapGroupRule for a polymorphic value[x]/choice field, with nested rules per choice type"""
        base_name = field_name.replace("[x]", "")
        rule.name = f"map-{clean_field_name(base_name)}"  # CT-2: use clean base name
        rule.target = []  # CT-1: no outer target element for choice types
        field_path = field.get("path", "")
        tree = self.profile_tree()
        tree_key = field.get("id") or field_path
        choice_types = list(
            (tree.effective_types(tree_key) if tree else [])
            or field.get("choice_types")
            or []
        )
        derived_raw_choice_types = False
        if not choice_types and automapped_mappings:
            has_choice_provider = any(
                key == field_path
                or key.startswith(field_path + ".")
                or key.startswith(field_path + ":")
                for key in automapped_mappings
            )
            if has_choice_provider:
                choice_types = type_codes(field.get("type"))
                derived_raw_choice_types = bool(choice_types)
        guard_empty_choice = False

        if choice_types and automapped_mappings:
            src_id = automapped_mappings.get(field.get("path", ""), "")
            src_field_name = src_id.split(".")[-1] if src_id else ""
            src_type = (
                self.source_field_types.get(src_field_name, "")
                if src_field_name
                else ""
            )
            if src_type:
                preferred = canonical_primitive(src_type) or src_type
                if preferred:
                    narrowed = [
                        ct for ct in choice_types if ct.lower() == preferred.lower()
                    ]
                    if narrowed:
                        logger.debug(
                            "Narrowed value[x] for '%s' to [%s] from source type '%s'.",
                            field_name,
                            narrowed[0],
                            src_type,
                        )
                        choice_types = narrowed
                        if preferred != "string":
                            guard_empty_choice = True  # consumed in the per-type loop

        rule.documentation = self.generate_rule_documentation(
            field, is_slice, slice_info
        )
        if choice_types:
            rule.documentation += f" | Choice types: {', '.join(choice_types)}"

        chosen_type = None
        sub_field_maps = []  # (subfield_path, source_local_element)
        if automapped_mappings:
            for k, v in automapped_mappings.items():
                ctype = None
                sub = None
                concrete_root = False
                if k.startswith(field_path + ":"):
                    rest = k[len(field_path) + 1 :]  # e.g. "valueQuantity.value"
                    head, dot, sub = rest.partition(
                        "."
                    )  # head="valueQuantity", sub="value"
                    ctype = (
                        head[len(base_name) :]
                        if head.lower().startswith(base_name.lower())
                        else head
                    )
                    concrete_root = bool(not dot and ctype)
                elif k.startswith(field_path + "."):
                    sub = k[len(field_path) + 1 :]
                if sub or concrete_root:
                    if ctype and not chosen_type:
                        chosen_type = ctype
                    sub_field_maps.append((sub, self._as_local_element(v)))
        if not chosen_type and sub_field_maps:
            _INFERRABLE_CHOICE_TYPES = (
                "Quantity",
                "Range",
                "Ratio",
                "CodeableConcept",
                "Period",
            )
            subs = {s.split(".", 1)[0] for s, _ in sub_field_maps}
            best, best_overlap = None, 0
            for ct in choice_types:
                if ct not in _INFERRABLE_CHOICE_TYPES:
                    continue
                fields = _choice_type_direct_fields(ct)
                if not fields:
                    continue
                overlap = len(subs & fields)
                if overlap and subs <= fields and overlap > best_overlap:
                    best, best_overlap = ct, overlap
            chosen_type = best

        if not chosen_type and len(choice_types) == 1 and sub_field_maps:
            chosen_type = choice_types[0]

        # A direct provider for the raw choice root carries no target-type suffix.
        # It is safe only after the profile/source narrowing above leaves exactly one
        # candidate.  Preserve the source for the concrete primitive/complex copy rule.
        direct_source = (automapped_mappings or {}).get(field_path)
        if direct_source and not sub_field_maps and len(choice_types) == 1:
            chosen_type = choice_types[0]
            sub_field_maps.append(("", self._as_local_element(direct_source)))

        if chosen_type and chosen_type not in choice_types:
            match = next(
                (ct for ct in choice_types if ct.lower() == chosen_type.lower()), None
            )
            if match:
                chosen_type = match
            else:
                self.record_diagnostic(
                    "unsupported-choice-type",
                    f"Concrete mapping for {field_path} selects unsupported choice "
                    f"type {chosen_type}; allowed types are "
                    f"[{', '.join(choice_types)}].",
                    path=field_path,
                    selected_type=chosen_type,
                    allowed_types=choice_types,
                    severity="error",
                )
                return None

        if chosen_type and chosen_type in choice_types:
            narrowed_rule = self._create_choice_populate_rule(
                base_name,
                chosen_type,
                sub_field_maps,
                parent_source_context,
                parent_target_context,
                cardinality,
            )
            if narrowed_rule:
                rule.rule = [narrowed_rule]
                return rule

        if direct_source and len(choice_types) != 1:
            example = (
                f"{base_name}{self._choice_suffix(choice_types[0])}"
                if choice_types
                else f"{base_name}<Type>"
            )
            self.record_diagnostic(
                "ambiguous-choice-type",
                f"Direct mapping to choice {field_path} is ambiguous across "
                f"[{', '.join(choice_types) or 'unknown types'}]; map to a "
                f"concrete target element such as {example}.",
                path=field_path,
                allowed_types=choice_types,
                severity="error",
            )
            return None

        if derived_raw_choice_types and len(choice_types) > 1:
            self.record_diagnostic(
                "ambiguous-choice-type",
                f"Mapped choice {field_path} does not identify one concrete target "
                "type; it was left unresolved.",
                path=field_path,
                allowed_types=choice_types,
                severity="error",
            )
            return None

        nested_rules = []
        for choice_type in choice_types:
            choice_suffix = self._choice_suffix(choice_type)
            transform_info = get_transform_for_type(choice_type, f"src-{var_suffix}")

            is_typed_create = (
                bool(transform_info) and transform_info.get("transform") == "create"
            )
            if is_typed_create:
                target_element = base_name
            else:
                target_element = (
                    f"{base_name}{choice_suffix}" if choice_suffix else base_name
                )

            choice_rule = StructureMapGroupRule.model_construct()
            choice_rule.name = (
                f"map-{clean_field_name(base_name)}-{clean_field_name(choice_type)}"
            )

            choice_source = StructureMapGroupRuleSource.model_construct()
            choice_source.context = parent_source_context
            choice_source.element = source_element
            choice_source.variable = f"src-{var_suffix}"
            # Guard against REDCap's empty-string representation of missing values
            if guard_empty_choice:
                choice_source.condition = "$this != ''"
            choice_rule.source = [choice_source]

            choice_target = StructureMapGroupRuleTarget.model_construct()
            choice_target.context = parent_target_context
            choice_target.element = target_element
            choice_target.variable = f"tgt-{var_suffix}-{clean_field_name(choice_type)}"

            self._apply_repeating_list_modes(
                cardinality,
                choice_source,
                choice_target,
                choice_rule.name,
                field_path=field_path,
            )

            if transform_info:
                choice_target.transform = transform_info.get("transform")
                params = transform_info.get("parameters")
                if params:
                    choice_target.parameter = params

            choice_rule.target = [choice_target]
            nested_rules.append(choice_rule)

        if nested_rules:
            rule.rule = nested_rules
        return rule

    def create_meta_profile_rule(
        self, profile_url, source_context="source", target_context="target"
    ):
        """meta profile rule: stamp the transformed resource with a meta.profile declaration for its profile URL"""
        rule = StructureMapGroupRule.model_construct(name="set-meta-profile")
        rule.documentation = f"Declare conformance to {profile_url}"
        rule.source = [
            StructureMapGroupRuleSource.model_construct(context=source_context)
        ]
        meta_tgt = StructureMapGroupRuleTarget.model_construct(
            context=target_context,
            element="meta",
            variable="tgt-meta",
            transform="create",
        )
        meta_tgt.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(valueString="Meta")
        ]
        rule.target = [meta_tgt]

        profile_rule = StructureMapGroupRule.model_construct(
            name="set-meta-profile-url"
        )
        profile_rule.source = [
            StructureMapGroupRuleSource.model_construct(context=source_context)
        ]
        profile_tgt = StructureMapGroupRuleTarget.model_construct(
            context="tgt-meta", element="profile", transform="copy"
        )
        profile_tgt.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(
                valueString=profile_url
            )
        ]
        profile_rule.target = [profile_tgt]
        rule.rule = [profile_rule]
        return rule

    def create_ref_groups(self, fields):
        """create helper StructureMapGroup objects for Reference fields, to be used by the bundle assembler"""
        groups = []
        referenced_types = set()
        # print(fields)
        for resource_info in fields:
            refs = find_reference_fields([resource_info])
            # print(refs)
            referenced_types.update(refs)

        for ref_type in referenced_types:
            group = StructureMapGroup.model_construct(
                name=f"Reference{ref_type}", typeMode="none"
            )
            # print(ref_type)
            group.name = f"Reference{ref_type}"
            group.typeMode = "none"
            group.documentation = f"Helper group to create Reference to {ref_type}"
            group.input = [
                StructureMapGroupInput.model_construct(
                    name="source", mode="source", type="Any"
                ),
                StructureMapGroupInput.model_construct(
                    name="target", mode="target", type="Any"
                ),
            ]

            rule = StructureMapGroupRule.model_construct()
            rule.name = "set-reference"
            source = StructureMapGroupRuleSource.model_construct()
            source.context = "source"
            rule.source = [source]

            target = StructureMapGroupRuleTarget.model_construct()
            target.context = "target"
            target.element = "reference"

            # Use conditional reference pattern: Type?identifier=[source]
            # Use FHIRPath concatenation: 'Type?identifier=' + %source
            target.transform = "evaluate"
            target.parameter = [
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString=f"'{ref_type}?identifier=' + %source"
                )
            ]
            rule.target = [target]
            group.rule = [rule]
            groups.append(group)
        return groups

    def create_base_structure_map(
        self, map_url: str, map_name: str, map_title: str, status: str = "draft"
    ) -> StructureMap:
        """
        Create a base/empty StructureMap object

        :param self: Description
        :param map_url: Description
        :type map_url: str
        :param map_name: Description
        :type map_name: str
        :param map_title: Description
        :type map_title: str
        :param status: Description
        :type status: str
        :return: Description
        :rtype: StructureMap
        """
        return StructureMap.model_construct(
            url=map_url,
            name=map_name,
            title=map_title,
            status=status,
        )

    def create_discriminator_hint_rule(
        self, slice_info, source_context, target_context
    ):
        if not slice_info.get("discriminator"):
            return None

        disc = slice_info["discriminator"]
        disc_path = disc.get("path", "")
        disc_type = disc.get("type", "value")

        if not disc_path:
            return None

        if disc_type not in ("value", "pattern"):
            logger.info(
                "Slice discriminator '%s' (type %s) is not value-based — no discriminator "
                "value rule emitted (the created type/profile/presence distinguishes it).",
                disc_path,
                disc_type,
            )
            return None

        value = disc.get("value")
        if not has_fixed_value(value) or isinstance(value, (dict, list)):
            self.record_diagnostic(
                "missing-discriminator-provider",
                f"Discriminator {disc_type}:{disc_path} has no scalar profile value; "
                "no executable placeholder was emitted.",
                path=disc_path,
                severity="warning",
            )
            return None

        rule = StructureMapGroupRule.model_construct()
        rule.name = f"set-discriminator-{disc_path}"
        rule.documentation = f"Set {disc_path} to distinguish this slice (discriminator type: {disc_type})."

        rule.source = [
            StructureMapGroupRuleSource.model_construct(context=source_context)
        ]

        target = StructureMapGroupRuleTarget.model_construct()
        target.context = target_context
        target.element = disc_path
        target.transform = "copy"
        target.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(
                valueString=fixed_scalar_literal(value)
            )
        ]
        rule.target = [target]

        return rule

    def generate_rule_documentation(self, field, is_slice=False, slice_info=None):
        """generate a documentation string for a StructureMapGroupRule based on the field's metadata"""
        docs = [f"Maps to {field['path']}"]

        if field.get("type"):
            # Render the canonical type code(s), not the raw parser ``type`` shape
            # (which may still be a list of StructureDefinition type objects).
            codes = type_codes(field.get("type"))
            type_repr = ", ".join(codes) if codes else field["type"]
            docs.append(f"Type: {type_repr}")

        cardinality = field.get("cardinality", {})
        if cardinality:
            min_card = cardinality.get("min", 0)
            max_card = cardinality.get("max", "*")
            docs.append(f"Cardinality: {min_card}..{max_card}")

        if field.get("binding"):
            binding = field["binding"]
            strength = binding.get("strength", "unknown")
            valueset = binding.get("valueSet", "N/A")
            docs.append(f"Binding: {strength} to {valueset}")

        if is_slice and slice_info:
            docs.append(f"SLICE: {slice_info['slice_name']}")
            if slice_info.get("discriminator"):
                disc = slice_info["discriminator"]
                docs.append(
                    f"Discriminator: {disc['path']} = '{disc.get('value', 'N/A')}'"
                )

        if field.get("children"):
            docs.append(f"Complex type with {len(field['children'])} child elements")

        return " | ".join(docs)
