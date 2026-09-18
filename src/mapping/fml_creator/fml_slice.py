from fhir.resources.R4B.structuremap import (
    StructureMapGroupRule,
    StructureMapGroupRuleTarget,
    StructureMapGroupRuleTargetParameter,
    StructureMapGroupRuleSource,
)
from mapping.fml_creator.fml_helper import (
    fit_rule_name,
    is_primitive_type,
    get_transform_for_type,
    canonical_primitive,
    clean_field_name,
    fixed_scalar_parameter,
    has_fixed_value,
    type_codes,
    fhir_type_suffix,
    attr as _attr,
)
from parser.resource_parser.fhir_type_introspection import get_complex_type_fields
import logging
import re

logger = logging.getLogger(__name__)


def _rule_key(candidate):
    targets = candidate.target or []
    target = targets[0] if targets else None
    return (
        candidate.name,
        getattr(target, "context", None),
        getattr(target, "element", None),
        getattr(target, "transform", None),
    )


def _append_or_merge(rules, candidate):
    """Merge shared complex prefixes such as coding.code/system or telecom.system/value.

    """
    match = next(
        (existing for existing in rules if _rule_key(existing) == _rule_key(candidate)),
        None,
    )
    if match is None:
        rules.append(candidate)
        return
    if match.rule is None:
        match.rule = []
    for child in candidate.rule or []:
        _append_or_merge(match.rule, child)


class _SliceRulesMixin:
    @staticmethod
    def _element_path_without_slices(path):
        """Return the underlying ElementDefinition path without slice selectors."""
        return ".".join(
            segment.split(":", 1)[0] for segment in str(path or "").split(".")
        )

    @staticmethod
    def _slice_identity(parent_field, slice_field):
        identity = slice_field.get("slice_identity") or slice_field.get("id")
        if identity and ":" in identity:
            return identity
        base_path = parent_field.get("path") or slice_field.get("path", "")
        slice_name = slice_field.get("sliceName") or slice_field.get(
            "slice_selector", ""
        )
        return f"{base_path}:{slice_name}" if slice_name else base_path

    @staticmethod
    def _slice_discriminator_field(slice_field, discriminator_path):
        """Find a discriminator field relative to the slice root."""
        if discriminator_path in ("", "$this"):
            return slice_field

        parts = [
            part.replace("[x]", "")
            for part in discriminator_path.split(".")
            if part and part not in ("$this", "resolve()")
        ]
        current = slice_field
        for part in parts:
            current = next(
                (
                    child
                    for child in (current.get("children") or [])
                    if (child.get("path", "") or "")
                    .split(".")[-1]
                    .replace("[x]", "")
                    == part
                ),
                None,
            )
            if current is None:
                return None
        return current

    def _ancestor_slice_selector(self, parent_field, slice_field):
        """How the bundle assembler can find this slice's repeat at runtime.

        A deferred reference below a sliced repeating backbone flattens to a path
        that no longer names the slice — ontario's six
        ``Composition.section:<name>.entry`` all become ``section.entry``. The
        assembler therefore needs the discriminator the profile slices on plus the
        value pinned there, which together identify exactly one repeat regardless
        of the order the sections were generated in.

        Returns ``None`` unless the profile pins a value on a plain
        ``value``/``pattern`` discriminator: ``$this``, ``resolve()``, ``type`` and
        ``exists`` discriminators cannot be evaluated against generated JSON here,
        and guessing a repeat is worse than deferring nothing.
        """
        slicing = parent_field.get("slicing") or {}
        discriminators = slicing.get("discriminators") or []
        if not discriminators:
            legacy = parent_field.get("discriminator")
            discriminators = [legacy] if legacy else []
        element = (parent_field.get("path") or "").rsplit(".", 1)[-1].split(":", 1)[0]
        if not element:
            return None
        for discriminator in discriminators:
            if not isinstance(discriminator, dict):
                continue
            if discriminator.get("type") not in ("value", "pattern"):
                continue
            disc_path = str(discriminator.get("path") or "")
            if not disc_path or "resolve()" in disc_path or "$this" in disc_path:
                continue
            target_field = self._slice_discriminator_field(slice_field, disc_path)
            if target_field is None:
                continue
            value = target_field.get("fixed_value")
            if hasattr(value, "model_dump"):
                try:
                    value = value.model_dump(exclude_none=True)
                except Exception as exc:
                    logger.warning(
                        "Unreadable discriminator value on %s (%s: %s)",
                        self._slice_identity(parent_field, slice_field),
                        type(exc).__name__,
                        exc,
                    )
                    continue
            if not has_fixed_value(value):
                continue
            return {
                "path": element,
                "discriminator": disc_path.replace("[x]", ""),
                "value": value,
            }
        return None

    def _profile_defines_discriminator(
        self, parent_field, slice_field, discriminator
    ):
        """Whether target constraints distinguish the slice without source policy."""
        disc_type = discriminator.get("type")
        disc_path = discriminator.get("path", "")
        target_field = self._slice_discriminator_field(slice_field, disc_path)

        if disc_type in ("value", "pattern"):
            if not target_field:
                return False
            if has_fixed_value(target_field.get("fixed_value")):
                return True

            # A complex discriminator can be generatively constrained by fixed
            # or pattern values on its descendants. HMB, for example, slices
            # Observation.category on ``coding`` while fixing
            # ``coding.system``, ``coding.code`` and ``coding.display``. N1 can
            # emit those values from the target tree, so rejecting the slice
            # here would undo that emission.
            def _has_fixed_descendant(field):
                for child in field.get("children") or []:
                    if has_fixed_value(child.get("fixed_value")):
                        return True
                    if _has_fixed_descendant(child):
                        return True
                return False

            if _has_fixed_descendant(target_field):
                return True

            tree_fn = getattr(self, "profile_tree", None)
            tree = tree_fn() if callable(tree_fn) else None
            if tree is None:
                return False
            target_key = (
                target_field.get("id")
                or target_field.get("slice_identity")
                or target_field.get("path")
            )
            return bool(target_key and tree.fixed_descendants(target_key))

        if disc_type == "type":
            slice_codes = type_codes(slice_field.get("type"))
            parent_codes = type_codes(parent_field.get("type"))
            return len(slice_codes) == 1 and (
                "[x]" in str(parent_field.get("path", ""))
                or len(parent_codes) > 1
                or slice_codes != parent_codes
            )

        if disc_type == "profile":
            # A profile declaration constrains the selected value but does not
            # manufacture a conforming source value or reference target.
            return False

        if disc_type == "exists":
            if target_field is None:
                # R4/R4B permits discriminator FHIRPaths such as
                # content.extension(url='...'). The parser tree cannot represent
                # that function call as an ordinary child lookup, but a required
                # extension slice with the same canonical URL makes the existence
                # condition generatively true.
                match = re.search(
                    r"extension\(\s*(?:url\s*=\s*)?(['\"])(.*?)\1\s*\)",
                    str(disc_path),
                )
                if not match:
                    return False
                canonical = match.group(2)
                sd = getattr(self, "_current_profile_sd", None)
                snapshot = _attr(sd, "snapshot") if sd else None
                elements = (_attr(snapshot, "element", []) if snapshot else []) or []
                slice_identity = (
                    slice_field.get("id")
                    or slice_field.get("slice_identity")
                    or self._slice_identity(parent_field, slice_field)
                )
                for element in elements:
                    element_id = _attr(element, "id") or ""
                    if not element_id.startswith(f"{slice_identity}."):
                        continue
                    if (_attr(element, "min", 0) or 0) < 1:
                        continue
                    for type_ref in _attr(element, "type", []) or []:
                        if canonical in (_attr(type_ref, "profile", []) or []):
                            return True
                return False
            cardinality = target_field.get("cardinality") or {}
            if cardinality.get("max") == "0":
                return True
            return bool(
                (cardinality.get("min", 0) or 0) > 0
                and has_fixed_value(target_field.get("fixed_value"))
            )

        # position depends on source order. A resolve() path depends on reference
        # provider policy. Both therefore need an explicit slice-qualified rule.
        return False

    def _slice_selection_is_safe(
        self,
        parent_field,
        slice_field,
        explicit_provider,
        source_providers=None,
    ):
        slicing = slice_field.get("slicing") or parent_field.get("slicing") or {}
        discriminators = slicing.get("discriminators") or []
        if not discriminators:
            legacy = slice_field.get("discriminator") or parent_field.get(
                "discriminator"
            )
            discriminators = [legacy] if legacy else []

        identity = self._slice_identity(parent_field, slice_field)
        rules = slicing.get("rules", "open")
        ordered = bool(slicing.get("ordered"))
        allowed = {"value", "pattern", "type", "profile", "exists", "position"}
        unknown = [
            disc.get("type")
            for disc in discriminators
            if disc.get("type") not in allowed
        ]
        if unknown:
            self.record_diagnostic(
                "unsupported-slice-discriminator",
                f"Slice {identity} uses unsupported discriminator type(s): "
                f"{', '.join(str(value) for value in unknown)}.",
                path=identity,
                severity="error",
            )
            return False

        if any(disc.get("type") == "position" for disc in discriminators) and not ordered:
            self.record_diagnostic(
                "invalid-position-slicing",
                f"Slice {identity} uses a position discriminator but "
                "slicing.ordered is false.",
                path=identity,
                severity="error",
            )
            return False
        if any(disc.get("type") == "position" for disc in discriminators):
            source_max = getattr(self, "source_field_max", {}) or {}
            ambiguous_sources = [
                provider
                for provider in (source_providers or [])
                if source_max.get(str(provider).split(".")[-1]) not in ("1", 1)
            ]
            if not explicit_provider or ambiguous_sources:
                self.record_diagnostic(
                    "ambiguous-position-source",
                    f"Position slice {identity} requires an explicit, non-repeating "
                    "source provider for each selected position.",
                    path=identity,
                    source_providers=list(source_providers or []),
                    severity="error",
                )
                return False

        unresolved = [
            disc
            for disc in discriminators
            if "resolve()" in str(disc.get("path", ""))
            or not self._profile_defines_discriminator(
                parent_field, slice_field, disc
            )
        ]
        if unresolved and not explicit_provider:
            rendered = ", ".join(
                f"{disc.get('type')}:{disc.get('path')}" for disc in unresolved
            )
            self.record_diagnostic(
                "ambiguous-slice-selection",
                f"Slice {identity} cannot be selected from profile constraints alone "
                f"({rendered}). Add a slice-qualified mapping rule.",
                path=identity,
                slicing_rules=rules,
                ordered=ordered,
                discriminators=discriminators,
                severity="error",
            )
            return False

        if unresolved:
            self.record_diagnostic(
                "explicit-slice-selection",
                f"Slice {identity} relies on its slice-qualified mapping rule for "
                "selection.",
                path=identity,
                slicing_rules=rules,
                ordered=ordered,
                discriminators=discriminators,
                severity="information",
            )
        return True

    def _create_choice_populate_rule(
        self,
        base_name,
        choice_type,
        sub_field_maps,
        parent_source_context,
        parent_target_context,
        cardinality,
        field=None,
    ):
        """emit a single rule that creates and populates a choice-typed element (value[x] or similar)"""
        bn = clean_field_name(base_name)
        ct = clean_field_name(choice_type)
        rule = StructureMapGroupRule.model_construct(name=f"map-{bn}-{ct}")
        fixed_descendants = self._choice_fixed_descendants(field, choice_type)
        fixed_paths = {
            relative
            for relative, node in fixed_descendants
            if not node.is_pattern
        }

        whole_value_maps = [
            (sub, source) for sub, source in sub_field_maps if sub == ""
        ]
        descendant_maps = [
            (sub, source) for sub, source in sub_field_maps if sub != ""
        ]
        # Mapping-table expansion supplies intermediate ancestors so their target
        # contexts can be materialised.  Once a more specific descendant exists,
        # however, the ancestor is not an independent value provider.  Treating it
        # as one creates a second, empty complex value (for example a bare Coding
        # beside the Coding populated by ``coding.code``/``coding.system``).
        descendant_paths = {sub for sub, _ in descendant_maps}
        shadowed_ancestors = {
            sub
            for sub in descendant_paths
            if any(
                other != sub and other.startswith(sub + ".")
                for other in descendant_paths
            )
        }
        if shadowed_ancestors:
            descendant_maps = [
                (sub, source)
                for sub, source in descendant_maps
                if sub not in shadowed_ancestors
            ]
            sub_field_maps = [
                (sub, source)
                for sub, source in sub_field_maps
                if sub not in shadowed_ancestors
            ]
            self.record_diagnostic(
                "choice-ancestor-provider-shadowed",
                f"Intermediate mapping provider(s) for {base_name} were ignored "
                "because mapped descendants populate the same complex value.",
                target=base_name,
                shadowed_paths=sorted(shadowed_ancestors),
                severity="information",
            )
        if whole_value_maps and descendant_maps:
            # A generated required-field placeholder can coexist with authored
            # descendant mappings for the same complex choice. Copying the whole
            # placeholder as well would compete with the populated create branch;
            # treating "" as a child also serializes an invalid empty target element.
            # Prefer the more specific descendant assignments and expose the decision.
            self.record_diagnostic(
                "whole-complex-provider-shadowed",
                f"Whole-value provider for {base_name}{self._choice_suffix(choice_type)} "
                "was ignored because more specific descendant mappings populate the "
                "same complex choice.",
                target=f"{base_name}{self._choice_suffix(choice_type)}",
                source_providers=[source for _, source in whole_value_maps],
                severity="information",
            )
            sub_field_maps = descendant_maps

        if is_primitive_type(choice_type):
            suffix = self._choice_suffix(choice_type)
            src = StructureMapGroupRuleSource.model_construct(
                context=parent_source_context
            )
            tgt = StructureMapGroupRuleTarget.model_construct()
            tgt.context = parent_target_context
            tgt.element = f"{base_name}{suffix}" if suffix else base_name
            if sub_field_maps:
                tgt.element = base_name
                src.element = sub_field_maps[0][1]
                src.variable = "src-choiceval"
                if field is not None:
                    self._apply_source_bound_check(
                        field, src, target_type=choice_type
                    )
                info = get_transform_for_type(choice_type, "src-choiceval")
                canonical_choice = canonical_primitive(choice_type) or choice_type
                if (
                    info.get("transform") == "copy"
                    and str(canonical_choice).lower() != "string"
                ):
                    info = {
                        "transform": "cast",
                        "parameters": [
                            StructureMapGroupRuleTargetParameter.model_construct(
                                valueId="src-choiceval"
                            ),
                            StructureMapGroupRuleTargetParameter.model_construct(
                                valueString=canonical_choice
                            ),
                        ],
                    }
                tgt.transform = info.get("transform")
                if info.get("parameters"):
                    tgt.parameter = info["parameters"]
            rule.source = [src]
            rule.target = [tgt]
            return rule

        # A concrete complex choice can also be copied as a whole.  This is distinct
        # from descendant mappings, where the datatype must be created and populated.
        if whole_value_maps and not descendant_maps:
            src = StructureMapGroupRuleSource.model_construct(
                context=parent_source_context,
                element=whole_value_maps[0][1],
                variable="src-choiceval",
            )
            tgt = StructureMapGroupRuleTarget.model_construct(
                context=parent_target_context,
                element=f"{base_name}{self._choice_suffix(choice_type)}",
                variable=f"tgt-{bn}-{ct}",
                transform="copy",
                parameter=[
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueId="src-choiceval"
                    )
                ],
            )
            rule.source = [src]
            rule.target = [tgt]
            rule.rule = self._choice_fixed_descendant_rules(
                field,
                choice_type,
                tgt.variable,
                parent_source_context,
                bn,
                sub_field_maps,
                descendants=fixed_descendants,
            )
            return rule

        # Complex choice type: create then populate direct sub-fields.
        var = f"tgt-{bn}-{ct}"
        src = StructureMapGroupRuleSource.model_construct(context=parent_source_context)
        rule.source = [src]
        tgt = StructureMapGroupRuleTarget.model_construct()
        tgt.context = parent_target_context
        tgt.element = base_name
        tgt.variable = var
        tgt.transform = "create"
        tgt.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(
                valueString=choice_type
            )
        ]
        rule.target = [tgt]

        nested = []

        for sub, src_local in sub_field_maps:
            if sub in fixed_paths:
                self.record_diagnostic(
                    "fixed-choice-descendant-overrides-source",
                    f"Source mapping for fixed target descendant "
                    f"{field.get('path') if field else base_name}.{sub} was ignored; "
                    "the profile-fixed value is authoritative.",
                    path=(
                        f"{field.get('path')}.{sub}"
                        if field and field.get("path")
                        else f"{base_name}.{sub}"
                    ),
                    source=src_local,
                    severity="information",
                )
                continue
            if "." in sub:
                deep = self._emit_subpath_population(
                    var, sub, src_local, choice_type, bn, parent_source_context
                )
                if deep:
                    _append_or_merge(nested, deep)
                continue
            sub_id = clean_field_name(sub)
            sub_var = f"src-{bn}-{sub_id}"
            s = StructureMapGroupRuleSource.model_construct(
                context=parent_source_context, element=src_local, variable=sub_var
            )
            sr = StructureMapGroupRule.model_construct(name=f"set-{bn}-{sub_id}")
            sr.source = [s]
            st = StructureMapGroupRuleTarget.model_construct(context=var, element=sub)
            leaf_type = self._resolve_leaf_type(choice_type, sub)
            info = get_transform_for_type(leaf_type or "string", sub_var)
            st.transform = info.get("transform", "copy")
            st.parameter = info.get("parameters") or [
                StructureMapGroupRuleTargetParameter.model_construct(valueId=sub_var)
            ]
            sr.target = [st]
            nested.append(sr)
        for fixed_rule in self._choice_fixed_descendant_rules(
            field,
            choice_type,
            var,
            parent_source_context,
            bn,
            sub_field_maps,
            descendants=fixed_descendants,
        ):
            _append_or_merge(nested, fixed_rule)
        rule.rule = nested
        return rule

    def _snapshot_nested_extension_specs(
        self, owner_id, automapped_mappings=None
    ):
        """Recover immediate extension slices omitted from the parser field tree.

        Snapshot ``ElementDefinition.id`` retains both the owning slice and the
        nested extension slice, so it is the authoritative fallback for paths such
        as ``contact:forschungskontakt.extension:rolle``.
        """

        if not owner_id:
            return []
        sd = getattr(self, "_current_profile_sd", None)
        snapshot = _attr(sd, "snapshot") if sd else None
        elements = (_attr(snapshot, "element", []) if snapshot else []) or []
        prefix = f"{owner_id}."
        specs = []
        for element in elements:
            identity = _attr(element, "id") or ""
            if not identity.startswith(prefix):
                continue
            relative_to_owner = identity[len(prefix) :]
            match = re.fullmatch(
                r"(extension|modifierExtension):([^.]+)", relative_to_owner
            )
            if not match or str(_attr(element, "max", "1")) == "0":
                continue
            source = (automapped_mappings or {}).get(identity)
            minimum = int(_attr(element, "min", 0) or 0)
            if minimum < 1 and not source:
                continue
            specs.append(
                {
                    "owner_id": owner_id,
                    "sub": relative_to_owner,
                    "relative": relative_to_owner,
                    "source": source,
                    "identity": identity,
                }
            )
        return specs

    def _choice_fixed_descendants(self, field, choice_type=None):
        """Profile-fixed/pattern leaves below one choice element, relative to it."""

        if not field:
            return []
        tree_fn = getattr(self, "profile_tree", None)
        tree = tree_fn() if callable(tree_fn) else None
        key = field.get("id") or field.get("path")
        node = tree.node(key) if tree and key else None
        if node is None:
            return []

        descendants = []
        for descendant in tree.fixed_descendants(key):
            if descendant.slice_name and (
                descendant.parent is not node
                or (
                    choice_type
                    and choice_type not in descendant.types
                )
            ):
                continue
            slice_ancestors = []
            ancestor = descendant.parent
            while ancestor is not None and ancestor is not node:
                if ancestor.slice_name:
                    slice_ancestors.append(ancestor)
                ancestor = ancestor.parent
            if ancestor is not node:
                continue
            if slice_ancestors:
                # A direct type slice of the choice may describe the selected
                # concrete branch. Any deeper slice (for example
                # medicationCodeableConcept.coding:UNII) needs an explicit
                # slice provider and must not leak its pattern into the generic
                # CodeableConcept instance.
                direct_type_slice = (
                    len(slice_ancestors) == 1
                    and slice_ancestors[0].parent is node
                    and (
                        not choice_type
                        or choice_type in slice_ancestors[0].types
                    )
                )
                if not direct_type_slice:
                    continue
            if descendant.eid.startswith(node.eid + "."):
                relative = descendant.eid[len(node.eid) + 1 :]
            elif descendant.path.startswith(node.path + "."):
                relative = descendant.path[len(node.path) + 1 :]
            else:
                continue
            # Sliced descendants require their own instance-selection machinery.
            # Treating a slice marker as a normal target property would create an
            # invalid StructureMap element name.
            if not relative or any(":" in part for part in relative.split(".")):
                continue
            descendants.append((relative, descendant))
        return descendants

    def _choice_fixed_descendant_rules(
        self,
        field,
        choice_type,
        target_var,
        source_context,
        name_prefix,
        mapped_subfields,
        *,
        descendants=None,
    ):
        """Emit fixed/pattern leaves inside a concrete complex choice branch."""

        descendants = (
            self._choice_fixed_descendants(field, choice_type)
            if descendants is None
            else descendants
        )
        if not descendants:
            return []

        mapped_paths = {sub for sub, _ in mapped_subfields}
        fixed = {}
        compatible_children = self._fixed_structure_children(field, choice_type)
        for relative, node in descendants:
            if self._fixed_child_schema(
                compatible_children, relative.split(".", 1)[0]
            )[0] is None:
                continue
            # A pattern constrains an authored value but need not replace it.
            # Fixed values are authoritative and are emitted even when a broader
            # complex provider exists.
            if node.is_pattern and any(
                not mapped
                or relative == mapped
                or relative.startswith(mapped + ".")
                for mapped in mapped_paths
            ):
                continue
            current = fixed
            parts = relative.split(".")
            for part in parts[:-1]:
                existing = current.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    current[part] = existing
                current = existing
            current[parts[-1]] = node.fixed_value

        if not fixed:
            return []
        return self._fixed_pattern_rules(
            target_var,
            fixed,
            source_context,
            name_prefix,
            parent_type=choice_type,
            structure=field,
        )

    def _resolve_leaf_type(self, complex_type, sub, children=None):
        """Resolve a direct sub-field's type on a complex datatype: a profile-declared child
        element wins (its type may be constrained), then generic type introspection."""
        logical_sub = sub.split(":", 1)[0].replace("[x]", "")
        for c in children or []:
            child_name = (c.get("path", "") or "").split(".")[-1]
            if child_name.replace("[x]", "") == logical_sub:
                codes = type_codes(c.get("type"))
                if codes:
                    return codes[0]
        for f in get_complex_type_fields(complex_type) or []:
            if f.get("path", "").split(".")[-1] == sub:
                t = f.get("type")
                return (
                    (t[0].get("code") if isinstance(t[0], dict) else t[0])
                    if isinstance(t, list) and t
                    else (t if isinstance(t, str) else None)
                )
        return None

    def _create_slice_instance_rule(
        self,
        parent_field,
        slice_field,
        res_type,
        parent_source_context,
        parent_target_context,
        automapped_mappings,
    ):
        """emit a rule that creates and populates a slice instance (BackboneElement or similar)"""
        slice_path = self._slice_identity(parent_field, slice_field)
        collection = (getattr(self, "collection_rules", None) or {}).get(slice_path)
        if collection is not None and collection.invalid:
            logger.warning(
                "Omitting %s because its collection correlation declaration is invalid.",
                slice_path,
            )
            return None
        slice_name = slice_field.get("sliceName", "") or slice_path.rsplit(":", 1)[-1]
        last_seg = slice_path.split(".")[-1]  # name:name
        base_element = last_seg.split(":")[0]  # name
        st_codes = type_codes(slice_field.get("type"))
        slice_type = st_codes[0] if st_codes else "BackboneElement"

        pt_codes = type_codes(parent_field.get("type"))
        pt = pt_codes[0] if pt_codes else None
        if slice_type == "Coding" and pt == "CodeableConcept":
            logger.info(
                "Coding-slice '%s' of a CodeableConcept deferred (nested coding-slice).",
                slice_path,
            )
            self.record_diagnostic(
                "nested-coding-slice-deferred",
                f"Coding slice {slice_path} was not emitted as a sibling of its "
                "CodeableConcept; it is handled by the owning coding branch.",
                path=slice_path,
                severity="information",
            )
            return None

        fixed = slice_field.get("fixed_value")
        if hasattr(fixed, "model_dump"):
            try:
                fixed = fixed.model_dump(exclude_none=True)
            except Exception as e:
                logger.warning(
                    "Dropping unreadable fixed pattern on slice %s (%s: %s)",
                    slice_path,
                    type(e).__name__,
                    e,
                )
                fixed = None

        plain_prefix = slice_path + "."
        marker_prefix = (
            plain_prefix if ":" in slice_path else f"{slice_path}:{slice_name}."
        )

        def _sub_of(key):
            if key.startswith(marker_prefix):
                return key[len(marker_prefix) :]
            if key.startswith(plain_prefix):
                return key[len(plain_prefix) :]
            return None

        subs = [
            (rest, self._as_local_element(v))
            for k, v in (automapped_mappings or {}).items()
            if (rest := _sub_of(k))
        ]

        def _nested_extension_specs(owner):
            """Required/mapped extension slices immediately below one owner."""

            owner_id = (
                owner.get("slice_identity")
                or owner.get("id")
                or owner.get("path")
            )
            specs = []
            for extension_field in owner.get("children") or []:
                element = (extension_field.get("path", "") or "").split(".")[-1]
                if element not in {"extension", "modifierExtension"}:
                    continue
                for extension_slice in extension_field.get("slices") or []:
                    slice_name = extension_slice.get("sliceName")
                    if not slice_name:
                        continue
                    identity = (
                        extension_slice.get("slice_identity")
                        or extension_slice.get("id")
                    )
                    if not identity and owner_id:
                        identity = f"{owner_id}.{element}:{slice_name}"
                    mapping_source = (automapped_mappings or {}).get(identity)
                    minimum = (
                        extension_slice.get("cardinality") or {}
                    ).get("min", 0) or 0
                    if minimum < 1 and not mapping_source:
                        continue
                    relative = (
                        identity[len(slice_path) + 1 :]
                        if identity and identity.startswith(slice_path + ".")
                        else f"{element}:{slice_name}"
                    )
                    specs.append(
                        {
                            "owner_id": owner_id,
                            "sub": f"{element}:{slice_name}",
                            "relative": relative,
                            "source": mapping_source,
                            "identity": identity,
                        }
                    )
            indexed = {
                spec.get("identity")
                or (
                    f"{owner_id}.{spec['sub']}"
                    if owner_id
                    else spec["sub"]
                )
                for spec in specs
            }
            for recovered in self._snapshot_nested_extension_specs(
                owner_id, automapped_mappings
            ):
                if recovered["identity"] in indexed:
                    continue
                relative = (
                    recovered["identity"][len(slice_path) + 1 :]
                    if recovered["identity"].startswith(slice_path + ".")
                    else recovered["relative"]
                )
                specs.append({**recovered, "relative": relative})
            return specs

        direct_extension_specs = _nested_extension_specs(slice_field)
        child_extension_specs = {
            child.get("id") or child.get("path"): _nested_extension_specs(child)
            for child in (slice_field.get("children") or [])
        }
        handled_extension_paths = {
            spec["relative"]
            for spec in direct_extension_specs
        }
        for specs in child_extension_specs.values():
            handled_extension_paths.update(spec["relative"] for spec in specs)
        subs = [
            (rest, source)
            for rest, source in subs
            if rest not in handled_extension_paths
        ]
        root_provider = (automapped_mappings or {}).get(slice_path)
        _sub_paths = {s for s, _ in subs}
        subs = [
            (s, v)
            for (s, v) in subs
            if not any(o != s and o.startswith(s + ".") for o in _sub_paths)
        ]

        min_c = (slice_field.get("cardinality") or {}).get("min", 0) or 0
        extension_providers = [
            spec.get("source")
            for spec in direct_extension_specs
            if spec.get("source")
        ]
        for specs in child_extension_specs.values():
            extension_providers.extend(
                spec.get("source") for spec in specs if spec.get("source")
            )
        explicit_provider = bool(root_provider or subs or extension_providers)
        if not explicit_provider and min_c == 0:
            return None
        if not self._slice_selection_is_safe(
            parent_field,
            slice_field,
            explicit_provider,
            source_providers=[
                provider
                for provider in [
                    root_provider,
                    *(value for _, value in subs),
                    *extension_providers,
                ]
                if provider
            ],
        ):
            return None

        nm = f"{clean_field_name(base_element)}-{clean_field_name(slice_name)}"
        rule = StructureMapGroupRule.model_construct(name=f"map-{nm}")
        rule.documentation = (
            f"Slice {slice_name} of {parent_field.get('path','')} (type {slice_type})"
        )
        src = StructureMapGroupRuleSource.model_construct(context=parent_source_context)
        rule.source = [src]
        if collection is not None and root_provider:
            src.element = self._as_local_element(root_provider)
            src.variable = "src-slice"
            if collection.source_key:
                src.check = f"{collection.source_key}.count() = 1"
        nested_source_context = (
            src.variable if collection is not None and src.variable else parent_source_context
        )

        if is_primitive_type(slice_type):
            tgt = StructureMapGroupRuleTarget.model_construct(
                context=parent_target_context, element=base_element
            )
            if collection is not None:
                self._apply_repeating_list_modes(
                    slice_field.get("cardinality") or {},
                    src,
                    tgt,
                    rule.name,
                    field_path=slice_path,
                )
            else:
                tgt.listMode = ["share"]
            primitive_provider = root_provider or (subs[0][1] if subs else None)
            if primitive_provider:
                src.element = self._as_local_element(primitive_provider)
                src.variable = "src-slice"
                info = get_transform_for_type(slice_type, "src-slice")
                tgt.transform = info.get("transform")
                if info.get("parameters"):
                    tgt.parameter = info["parameters"]
            rule.target = [tgt]
            return rule

        if root_provider and not subs:
            src.element = self._as_local_element(root_provider)
            src.variable = "src-slice"
            tgt = StructureMapGroupRuleTarget.model_construct(
                context=parent_target_context,
                element=base_element,
                transform="copy",
                parameter=[
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueId="src-slice"
                    )
                ],
            )
            if collection is not None:
                self._apply_repeating_list_modes(
                    slice_field.get("cardinality") or {},
                    src,
                    tgt,
                    rule.name,
                    field_path=slice_path,
                )
            else:
                tgt.listMode = ["share"]
            rule.target = [tgt]
            return rule

        var = f"tgt-{nm}"
        tgt = StructureMapGroupRuleTarget.model_construct(
            context=parent_target_context, element=base_element, variable=var
        )
        tgt.transform = "create"
        tgt.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(valueString=slice_type)
        ]
        if collection is not None:
            self._apply_repeating_list_modes(
                slice_field.get("cardinality") or {},
                src,
                tgt,
                rule.name,
                field_path=slice_path,
            )
        else:
            tgt.listMode = ["share"]
        rule.target = [tgt]

        nested = []
        _mapped_roots = {
            sub.split(".", 1)[0].split(":", 1)[0].replace("[x]", "") for sub, _ in subs
        }
        if isinstance(fixed, dict):
            nested += self._fixed_pattern_rules(
                var,
                fixed,
                nested_source_context,
                nm,
                parent_type=slice_type,
                structure=slice_field,
                skip_keys=_mapped_roots,
            )
        for extension_spec in direct_extension_specs:
            extension_rule = self._slice_sub_extension_rule(
                extension_spec["owner_id"],
                extension_spec["sub"],
                var,
                nm,
                nested_source_context,
                automapped_mappings,
                src_local=extension_spec["source"],
            )
            if extension_rule:
                nested.append(extension_rule)
            else:
                self.record_diagnostic(
                    "nested-extension-canonical-unresolved",
                    f"Required or mapped nested extension "
                    f"{extension_spec['owner_id']}.{extension_spec['sub']} "
                    "could not be emitted because its canonical URL is absent.",
                    path=f"{extension_spec['owner_id']}.{extension_spec['sub']}",
                    severity="error",
                )
        base_depth = len((slice_field.get("path", "") or "").split("."))
        slice_children = slice_field.get("children") or []
        direct_children = [
            c
            for c in slice_children
            if len((c.get("path", "") or "").split(".")) == base_depth + 1
        ]
        mapped_roots = _mapped_roots
        slice_eid = slice_path if ":" in slice_path else f"{slice_path}:{slice_name}"
        nested += self._slice_child_fixed_rules(
            direct_children,
            var,
            nested_source_context,
            nm,
            mapped_roots,
            slice_eid=slice_eid,
        )

        root_fixed_keys = set(fixed) if isinstance(fixed, dict) else set()

        def _has_direct_fixed(child):
            value = child.get("fixed_value")
            return value is not None and not (
                isinstance(value, (list, dict, str)) and len(value) == 0
            )

        unresolved_subs = []
        for child in direct_children:
            min_cardinality = (child.get("cardinality") or {}).get("min", 0) or 0
            if not (child.get("is_required") or min_cardinality >= 1):
                continue
            element = (child.get("path", "") or "").split(".")[-1]
            element = element.replace("[x]", "")
            if (
                not element
                or element in mapped_roots
                or element in root_fixed_keys
                or _has_direct_fixed(child)
            ):
                continue
            placeholder = (
                f"TODO-MAP-{clean_field_name(nm).upper()}-"
                f"{clean_field_name(element).upper()}_SOURCE"
            )
            unresolved_subs.append((element, placeholder))
        subs.extend(unresolved_subs)

        for sub, src_local in subs:
            if "." in sub:
                deep = self._emit_subpath_population(
                    var,
                    sub,
                    src_local,
                    slice_type,
                    nm,
                    nested_source_context,
                    root_children=slice_children,
                )
                if deep:
                    _append_or_merge(nested, deep)
                continue
            if sub.split(":")[0] in ("extension", "modifierExtension") and ":" in sub:
                ext_rule = self._slice_sub_extension_rule(
                    slice_path,
                    sub,
                    var,
                    nm,
                    nested_source_context,
                    automapped_mappings,
                    src_local=src_local,
                )
                if ext_rule:
                    nested.append(ext_rule)
                else:
                    logger.warning(
                        "Dropping mapped extension slice '%s' of slice %s — "
                        "extension canonical not resolvable from the profile "
                        "snapshot (source '%s' will NOT be populated).",
                        sub,
                        slice_path,
                        src_local,
                    )
                continue
            _ref_child = next(
                (
                    c
                    for c in direct_children
                    if (c.get("path", "") or "").split(".")[-1] == sub
                ),
                None,
            )
            _ref_target = _ref_child.get("reference_target") if _ref_child else None
            if (
                _ref_target
                and _ref_child
                and "Reference" in (type_codes(_ref_child.get("type")) or [])
                and str(src_local).startswith("TODO")
            ):
                if slice_path.startswith(f"{res_type}."):
                    _rel_base = slice_path[len(res_type) + 1 :]
                else:
                    _rel_base = base_element
                _rel_base = ".".join(
                    segment.split(":", 1)[0]
                    for segment in _rel_base.split(".")
                )
                # A reference that is itself profile-sliced carries far more than
                # the unsliced child's `reference_target` (which is only
                # `Resource|4.0.1` for ontario's section entries). Prefer the
                # correlated contract, scoped to this slice's repeat so it does not
                # collapse onto the parent list's first element.
                _selector = self._ancestor_slice_selector(parent_field, slice_field)
                _sliced_ref = (
                    self._required_profile_reference_slices_rule(
                        _ref_child,
                        res_type,
                        nested_source_context,
                        automapped_mappings,
                        selectors=[_selector] if _selector else None,
                        name_hint=f"{_rel_base}-{slice_name}-{sub}",
                    )
                    if _selector
                    else None
                )
                if _sliced_ref is not None:
                    nested.append(_sliced_ref)
                    continue
                if _ref_child.get("slices"):
                    self.record_diagnostic(
                        "sliced-reference-not-deferrable",
                        f"{slice_path}.{sub} is sliced by target profile but the "
                        "contract could not be built (no discriminating value on "
                        "the owning slice, or a required slice without a target "
                        "profile). Falling back to an unscoped reference, which "
                        "cannot satisfy the profile slices.",
                        path=f"{slice_path}.{sub}",
                        severity="warning",
                    )
                ref_rule = StructureMapGroupRule.model_construct(
                    name=fit_rule_name(
                        f"TODO-resolve-reference-{clean_field_name(res_type)}"
                        f"-{clean_field_name(sub)}"
                    )
                )
                ref_rule.source = [
                    StructureMapGroupRuleSource.model_construct(
                        context=nested_source_context,
                        variable=f"src-{nm}-{clean_field_name(sub)}",
                    )
                ]
                ref_rule.target = None
                ref_rule.documentation = (
                    f"Reference<{res_type}.{_rel_base}.{sub}> → {_ref_target} "
                    f"— resolve via bundle assembler"
                )
                nested.append(ref_rule)
                continue
            sub_var = f"src-{nm}-{clean_field_name(sub)}"
            s = StructureMapGroupRuleSource.model_construct(
                context=nested_source_context, element=src_local, variable=sub_var
            )
            sr = StructureMapGroupRule.model_construct(
                name=f"set-{nm}-{clean_field_name(sub)}"
            )
            sr.source = [s]
            # Type-aware transform, like _create_choice_populate_rule: a numeric/date leaf
            # needs a cast, a bare copy crashes the engine on non-string sources.
            target_element = sub
            leaf_type = self._resolve_leaf_type(
                slice_type, sub, children=direct_children
            )
            logical_sub = sub.split(":", 1)[0].replace("[x]", "")
            choice_child = next(
                (
                    child
                    for child in direct_children
                    if (child.get("path", "") or "")
                    .split(".")[-1]
                    .replace("[x]", "")
                    == logical_sub
                    and (child.get("path", "") or "").split(".")[-1].endswith(
                        "[x]"
                    )
                ),
                None,
            )
            if choice_child:
                target_element = self._resolve_choice_element(
                    choice_child,
                    logical_sub,
                    choice_child.get("path", ""),
                    {choice_child.get("path", ""): src_local},
                )
                choice_types = type_codes(choice_child.get("type"))
                if target_element is None:
                    self.record_diagnostic(
                        "ambiguous-choice-type",
                        f"Slice child "
                        f"{choice_child.get('id') or choice_child.get('path')} "
                        "does not identify one concrete target choice type.",
                        path=choice_child.get("id") or choice_child.get("path"),
                        allowed_types=choice_types,
                        severity="error",
                    )
                    continue
                if len(choice_types) == 1:
                    leaf_type = choice_types[0]
            info = get_transform_for_type(leaf_type or "string", sub_var)
            stt = StructureMapGroupRuleTarget.model_construct(
                context=var,
                element=target_element,
                transform=info.get("transform", "copy"),
            )
            stt.parameter = info.get("parameters") or [
                StructureMapGroupRuleTargetParameter.model_construct(valueId=sub_var)
            ]
            sr.target = [stt]
            extension_specs = child_extension_specs.get(
                choice_child.get("id") or choice_child.get("path")
                if choice_child
                else None,
                [],
            )
            if extension_specs:
                primitive_var = f"{var}-{clean_field_name(logical_sub)}"
                stt.variable = primitive_var
                sr.rule = []
                for extension_spec in extension_specs:
                    extension_rule = self._slice_sub_extension_rule(
                        extension_spec["owner_id"],
                        extension_spec["sub"],
                        primitive_var,
                        f"{nm}-{clean_field_name(logical_sub)}",
                        nested_source_context,
                        automapped_mappings,
                        src_local=extension_spec["source"],
                    )
                    if extension_rule:
                        sr.rule.append(extension_rule)
                    else:
                        self.record_diagnostic(
                            "nested-extension-canonical-unresolved",
                            f"Required or mapped nested extension "
                            f"{extension_spec['owner_id']}."
                            f"{extension_spec['sub']} could not be emitted "
                            "because its canonical URL is absent.",
                            path=(
                                f"{extension_spec['owner_id']}."
                                f"{extension_spec['sub']}"
                            ),
                            severity="error",
                        )
            nested.append(sr)
        rule.rule = _drop_empty_duplicate_creates(nested)
        return rule

    def _slice_sub_extension_rule(
        self,
        slice_path,
        sub,
        slice_var,
        nm,
        parent_source_context,
        automapped_mappings,
        src_local=None,
    ):
        """emit correct extension:[] rules"""
        from mapping.fml_creator.fml_helper import attr as _attr

        _, ext_name = sub.split(":", 1)
        sd = getattr(self, "_current_profile_sd", None)
        snapshot = _attr(sd, "snapshot") if sd else None
        elements = (_attr(snapshot, "element", []) if snapshot else []) or []
        base_path = slice_path.split(":")[0]  # e.g. Organization.contact
        exact_id = f"{slice_path}.{sub}"
        canonical = None
        exact = [
            element
            for element in elements
            if (_attr(element, "id") or "") == exact_id
        ]
        fallbacks = [
            element
            for element in elements
            if not exact
            and (_attr(element, "id") or "").startswith(base_path)
            and (_attr(element, "id") or "").endswith(f".{sub}")
        ]
        for el in [*exact, *fallbacks]:
            for t in _attr(el, "type", []) or []:
                profiles = _attr(t, "profile", []) or []
                if profiles:
                    canonical = profiles[0]
                    break
            if canonical:
                break
        if not canonical:
            return None

        field = {
            "path": exact_id,
            "type": "Extension",
            "sliceName": ext_name,
            "extension_url": canonical,
            "children": [],
        }
        ext_mappings = automapped_mappings
        if src_local:
            ext_mappings = {**(automapped_mappings or {}), exact_id: src_local}
        rule = self.create_extension_rule(
            field,
            sub,
            parent_source_context,
            slice_var,
            automapped_mappings=ext_mappings,
        )
        if rule is None:
            return None

        def _suffix_names(r):
            if getattr(r, "name", None):
                r.name = f"{r.name}-{nm}"
            for child in r.rule or []:
                _suffix_names(child)

        _suffix_names(rule)
        return rule

    def _fixed_container_rule(self, child, elem, leaves, slice_var, src_ctx, nm):
        """Create a complex child and copy the profile-fixed leaves the tree found (N1)."""
        tc = type_codes(child.get("type"))
        ctype = tc[0] if tc else "CodeableConcept"
        cvar = f"{slice_var}-{clean_field_name(elem)}"
        rule = StructureMapGroupRule.model_construct(
            name=f"set-{nm}-{clean_field_name(elem)}"
        )
        rule.source = [StructureMapGroupRuleSource.model_construct(context=src_ctx)]
        target = StructureMapGroupRuleTarget.model_construct(
            context=slice_var, element=elem, variable=cvar, transform="create"
        )
        target.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(valueString=ctype)
        ]
        rule.target = [target]
        leaf_rules = []
        for leaf in leaves:
            value = leaf.fixed_value
            if isinstance(value, (dict, list)):
                continue  # complex fixed content is handled by _fixed_pattern_rules
            leaf_rule = StructureMapGroupRule.model_construct(
                name=f"set-{nm}-{clean_field_name(elem)}-{clean_field_name(leaf.name)}"
            )
            leaf_rule.source = [
                StructureMapGroupRuleSource.model_construct(context=src_ctx)
            ]
            leaf_target = StructureMapGroupRuleTarget.model_construct(
                context=cvar, element=leaf.name, transform="copy"
            )
            leaf_target.parameter = [fixed_scalar_parameter(value, leaf.types)]
            leaf_rule.target = [leaf_target]
            leaf_rules.append(leaf_rule)
        rule.rule = leaf_rules
        return rule

    def _tree_fixed_children(self, container_eid):
        """Direct children of `container_eid` that the profile fixes (N1).

        A slice may carry no fixed value of its own while its *grandchildren* are fixed —
        hmb pins `category:obstetrics.coding.system|code|display`, and looking only one
        level down emitted an empty Coding, so the element serialised away entirely.
        """
        if not container_eid:
            return []
        tree_fn = getattr(self, "profile_tree", None)
        tree = tree_fn() if callable(tree_fn) else None
        if tree is None:
            return []
        node = tree.node(container_eid)
        if node is None:
            return []
        return [
            child
            for child in node.children.values()
            if child.fixed_value is not None and not child.prohibited
        ]

    def _tree_fixed_choice_types(self, choice_eid):
        """Concrete type slices carrying a fixed/pattern value below ``choice_eid``."""

        tree_fn = getattr(self, "profile_tree", None)
        tree = tree_fn() if callable(tree_fn) else None
        node = tree.node(choice_eid) if tree and choice_eid else None
        if node is None:
            return []
        return [
            child
            for child in node.children.values()
            if child.slice_name
            and child.fixed_value is not None
            and len(child.types) == 1
            and not child.prohibited
        ]

    def _slice_child_fixed_rules(
        self, slice_children, slice_var, src_ctx, nm, mapped_roots=None, slice_eid=None
    ):
        """create fixed-pattern rules for a slice's child elements that carry a fixed/pattern value"""
        mapped_roots = mapped_roots or set()
        out = []
        for child in slice_children or []:
            fv = child.get("fixed_value")
            if hasattr(fv, "model_dump"):
                try:
                    fv = fv.model_dump(exclude_none=True)
                except Exception as e:
                    logger.warning(
                        "Dropping unreadable fixed pattern on slice child %s (%s: %s)",
                        child.get("path"),
                        type(e).__name__,
                        e,
                    )
                    fv = None
            elem = child.get("path", "").split(".")[-1]
            if (
                "[x]" in elem
                and elem.replace("[x]", "") in mapped_roots
                and slice_eid
            ):
                concrete_patterns = self._tree_fixed_choice_types(
                    f"{slice_eid}.{elem}"
                )
                if len(concrete_patterns) == 1:
                    pattern = concrete_patterns[0]
                    base = elem.replace("[x]", "")
                    ctype = pattern.types[0]
                    cvar = f"{slice_var}-{clean_field_name(base)}"
                    r = StructureMapGroupRule.model_construct(
                        name=f"set-{nm}-{clean_field_name(base)}"
                    )
                    r.source = [
                        StructureMapGroupRuleSource.model_construct(context=src_ctx)
                    ]
                    r.target = [
                        StructureMapGroupRuleTarget.model_construct(
                            context=slice_var,
                            element=base,
                            variable=cvar,
                            transform="create",
                            parameter=[
                                StructureMapGroupRuleTargetParameter.model_construct(
                                    valueString=ctype
                                )
                            ],
                        )
                    ]
                    r.rule = self._fixed_pattern_rules(
                        cvar,
                        pattern.fixed_value,
                        src_ctx,
                        f"{nm}-{clean_field_name(base)}",
                        parent_type=ctype,
                        structure=child,
                    )
                    out.append(r)
                    continue
            if fv is None or (isinstance(fv, (list, dict, str)) and len(fv) == 0):
                # N1: no fixed value on the child itself, but the profile may fix its
                # children (`category:obstetrics.coding.system|code|display`). Create the
                # container and copy those leaves in; otherwise the container serialises
                # empty and the whole element — and its required slice — disappears.
                leaves = (
                    self._tree_fixed_children(f"{slice_eid}.{elem}")
                    if slice_eid and elem
                    else []
                )
                if leaves:
                    out.append(
                        self._fixed_container_rule(
                            child, elem, leaves, slice_var, src_ctx, nm
                        )
                    )
                continue
            if not elem:
                continue
            tc = type_codes(child.get("type"))
            ctype = tc[0] if tc else "CodeableConcept"
            if "[x]" in elem and elem.replace("[x]", "") not in mapped_roots:
                logger.info(
                    "Skipping unsourced pattern-only choice %s in slice %s "
                    "(no value provider) — leaving the element absent.",
                    child.get("path"),
                    nm,
                )
                continue
            if "[x]" in elem:
                elem = elem.replace("[x]", fhir_type_suffix(ctype))
            if isinstance(fv, dict):
                cvar = f"{slice_var}-{clean_field_name(elem)}"
                r = StructureMapGroupRule.model_construct(
                    name=f"set-{nm}-{clean_field_name(elem)}"
                )
                r.source = [
                    StructureMapGroupRuleSource.model_construct(context=src_ctx)
                ]
                t = StructureMapGroupRuleTarget.model_construct(
                    context=slice_var, element=elem, variable=cvar, transform="create"
                )
                t.parameter = [
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueString=ctype
                    )
                ]
                r.target = [t]
                r.rule = self._fixed_pattern_rules(
                    cvar,
                    fv,
                    src_ctx,
                    f"{nm}-{clean_field_name(elem)}",
                    parent_type=ctype,
                    structure=child,
                )
                out.append(r)
            elif isinstance(fv, (str, bool, int, float)):
                r = StructureMapGroupRule.model_construct(
                    name=f"set-{nm}-{clean_field_name(elem)}"
                )
                r.source = [
                    StructureMapGroupRuleSource.model_construct(context=src_ctx)
                ]
                t = StructureMapGroupRuleTarget.model_construct(
                    context=slice_var, element=elem, transform="copy"
                )
                t.parameter = [fixed_scalar_parameter(fv, child.get("type"))]
                r.target = [t]
                out.append(r)
        return out

    def _emit_subpath_population(
        self,
        parent_var,
        sub_path,
        src_local,
        root_type,
        name_prefix,
        src_ctx,
        root_children=None,
    ):
        """emit a nested rule that populates a sub-path of a complex type (e.g. value[x]:valueQuantity.value/.unit)"""

        def _resolve(seg, cur_type, children):
            concrete_type = None
            lookup_seg = seg
            if ":" in seg:
                lookup_seg, selector = seg.split(":", 1)
                logical = lookup_seg.replace("[x]", "")
                if selector.lower().startswith(logical.lower()):
                    concrete_type = selector[len(logical) :]
            for c in children or []:
                last = (c.get("path", "") or "").split(".")[-1]
                if last == lookup_seg or last == f"{lookup_seg}[x]":
                    tc = type_codes(c.get("type"))
                    ty = concrete_type or (tc[0] if tc else None)
                    name = lookup_seg
                    if concrete_type and last.endswith("[x]"):
                        # Complex choices are created on the unsuffixed FML target
                        # element (``value = create('Quantity')``).
                        name = lookup_seg.replace("[x]", "")
                    elif last.endswith("[x]") and ty:
                        base = lookup_seg.replace("[x]", "")
                        name = (
                            f"{base}{ty[:1].upper()}{ty[1:]}"
                        )  # value + Integer -> valueInteger
                    return name, ty, (c.get("children") or [])
            for f in get_complex_type_fields(cur_type) if cur_type else []:
                if (f.get("path", "") or "").split(".")[-1] == seg:
                    t = f.get("type")
                    if isinstance(t, list):
                        t = (
                            (t[0].get("code") if isinstance(t[0], dict) else t[0])
                            if t
                            else None
                        )
                    return seg, t, []
            return seg, None, []

        def build(pvar, segs, cur_type, prefix, children=None):
            seg = segs[0]
            rest = segs[1:]
            elem, stype, child_nodes = _resolve(seg, cur_type, children)
            sid = clean_field_name(seg.split(":", 1)[0].replace("[x]", ""))
            if not rest:
                var = f"src-{prefix}-{sid}"
                r = StructureMapGroupRule.model_construct(name=f"set-{prefix}-{sid}")
                r.source = [
                    StructureMapGroupRuleSource.model_construct(
                        context=src_ctx, element=src_local, variable=var
                    )
                ]
                t = StructureMapGroupRuleTarget.model_construct(
                    context=pvar, element=elem
                )
                info = get_transform_for_type(stype or "string", var)
                t.transform = info.get("transform")
                if info.get("parameters"):
                    t.parameter = info["parameters"]
                r.target = [t]
                return r
            if not stype or is_primitive_type(stype):
                logger.info(
                    "Sub-path segment '%s' (type %s) under %s not a resolvable complex type — skipped.",
                    seg,
                    stype,
                    name_prefix,
                )
                return None
            newvar = f"{pvar}-{sid}"
            r = StructureMapGroupRule.model_construct(name=f"set-{prefix}-{sid}")
            r.source = [StructureMapGroupRuleSource.model_construct(context=src_ctx)]
            t = StructureMapGroupRuleTarget.model_construct(
                context=pvar, element=elem, variable=newvar, transform="create"
            )
            t.parameter = [
                StructureMapGroupRuleTargetParameter.model_construct(valueString=stype)
            ]
            r.target = [t]
            child = build(newvar, rest, stype, f"{prefix}-{sid}", child_nodes)
            if not child:
                return None
            r.rule = [child]
            return r

        return build(
            parent_var, sub_path.split("."), root_type, name_prefix, root_children
        )


def _drop_empty_duplicate_creates(rules):
    """Keep the populated rule when two rules create the same child element.

    The N1 repair emits a container together with its profile-fixed leaves, while the
    generic child walk may already have emitted a bare `create` for the same element.
    Both would run, producing two codings against a `max=1` element — keep whichever
    actually carries content.
    """
    def _key(rule):
        target = (rule.target or [None])[0]
        if target is None:
            return None
        return (getattr(target, "context", None), getattr(target, "element", None))

    best = {}
    order = []
    for rule in rules:
        key = _key(rule)
        if key is None or key[1] is None:
            order.append(rule)
            continue
        previous = best.get(key)
        if previous is None:
            best[key] = rule
            order.append(rule)
            continue
        previous_empty = not (previous.rule or [])
        current_empty = not (rule.rule or [])
        # Only an *empty* create is a duplicate worth dropping. Repeated populated rules
        # are legitimate: person's `category:todesDiagnose` requires two codings (SNOMED
        # and LOINC, `coding` min=2), and collapsing them cost a resource.
        if previous_empty and not current_empty:
            order[order.index(previous)] = rule
            best[key] = rule
        elif current_empty and not previous_empty:
            continue
        else:
            order.append(rule)
    return order
