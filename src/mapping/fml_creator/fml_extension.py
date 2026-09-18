import json
import re

import logging
from fhir.resources.R4B.structuremap import (
    StructureMapGroupRule,
    StructureMapGroupRuleTarget,
    StructureMapGroupRuleTargetParameter,
    StructureMapGroupRuleSource,
)
from mapping.fml_creator.fml_helper import (
    fit_rule_name,
    emits_only_url,
    fixed_scalar_parameter,
    has_fixed_value,
    infer_extension_value_type,
    is_primitive_type,
    parse_slice_info,
    clean_field_name,
    rule_can_fire,
    attr as _attr,
)
from parser.resource_parser.value_expander import expand_valueset
from data_handling.url_resolver.fhir_url_resolver import resolve_url

logger = logging.getLogger(__name__)

_DOC_TYPE_MAXLEN = 4096


class _ExtensionRulesMixin:
    @staticmethod
    def _sub_extension_slices(children):
        """Named extension slices declared one level below this extension."""
        found = []
        for child in children or []:
            if child.get("path", "").endswith(".extension") and child.get("slices"):
                for candidate in child["slices"]:
                    if isinstance(candidate, dict) and candidate.get("sliceName"):
                        found.append(candidate)
        return found

    @staticmethod
    def _extension_has_provider(mapping_key, path, lookup_path, automapped_mappings):
        """Whether anything authored feeds this extension or something below it.

        With a provider the outer rule resolves at runtime and fires; without one it
        is gated behind a TODO placeholder that never matches. That distinction
        decides whether a value-less extension is invalid output or dead scaffolding.
        """
        if mapping_key:
            return True
        prefixes = tuple(f"{p}." for p in (path, lookup_path) if p)
        return any(
            key.startswith(prefixes) for key in (automapped_mappings or {})
        )

    def _reference_target_from_profiles(self, target_profiles):
        """The resource type a Reference-valued extension points at.

        Resolves each declared ``targetProfile`` to the type it constrains, so the
        emitted rule names a FHIR resource type rather than a profile canonical.
        Falls back to the canonical's last segment, which is the type itself for the
        base definitions (``.../StructureDefinition/Patient``).
        """
        types = []
        for profile in target_profiles or []:
            resolved = None
            try:
                sd = resolve_url(profile, self.app_state)
            except Exception as exc:
                logger.debug("Could not resolve reference target %s: %s", profile, exc)
                sd = None
            if sd is not None:
                resolved = _attr(sd, "type", None)
            candidate = resolved or profile.rsplit("/", 1)[-1]
            if candidate and candidate not in types:
                types.append(candidate)
        # Several accepted targets cannot be expressed as one dependent group, and
        # guessing one would silently narrow the profile's intent.
        return types[0] if len(types) == 1 else None

    @staticmethod
    def _extension_reference_contract_rule(
        path,
        var_suffix,
        extension_url,
        reference_target,
        target_profiles,
        parent_source_context,
    ):
        """Hand a Reference-valued extension to the bundle assembler to wire.

        The assembler is what knows which resource in the bundle satisfies a target
        profile, so — exactly as for an ordinary ``subject`` — the map writes the
        extension's url and states the contract, and the reference itself is filled
        at assembly time. Emitting a ``create Reference`` here instead would write an
        empty Reference that has nothing to point at.

        The contract path is a runtime JSON path, so every repeat of ``extension``
        flattens to the same string; the selector names *which* repeat by the url the
        profile pins on this slice.
        """
        res_type = path.split(".", 1)[0]
        contract = {
            "sourceType": res_type,
            "path": "extension.valueReference",
            "targetTypes": [reference_target],
            "targetProfiles": list(target_profiles) or [reference_target],
            "match": "byOrder",
            "sourceKey": None,
            "targetKey": None,
            "referenceMode": "urn",
            "selectors": [
                {
                    "path": "extension",
                    "discriminator": "url",
                    "value": extension_url,
                }
            ],
        }
        return StructureMapGroupRule.model_construct(
            name=fit_rule_name(
                "TODO-resolve-reference-"
                f"{clean_field_name(res_type)}-extension-"
                f"{clean_field_name(var_suffix)}"
            ),
            source=[
                StructureMapGroupRuleSource.model_construct(
                    context=parent_source_context
                )
            ],
            documentation="FHIRBRIDGE_REFERENCE:"
            + json.dumps(contract, sort_keys=True, separators=(",", ":")),
        )

    def _report_url_only_extension(self, lookup_path, extension_url, because):
        self.record_diagnostic(
            "extension-value-emission-failed",
            f"Extension {lookup_path} has a source provider but no value rule could "
            f"be generated for it, because {because}. Emitting it would produce a "
            "url-only extension, which violates ext-1; it is omitted instead.",
            path=lookup_path,
            extension_url=extension_url,
            severity="error",
        )

    def _subextension_slices_from_definition(self, extension_url, owner_path):
        """Sub-extension slices taken from the extension's own definition.

        Paths are rebuilt resource-qualified against ``owner_path`` so that a
        mapping authored on the profile's own identity
        (``Condition.extension:existance.extension:YesNoUnknownExtension``) still
        resolves once recursion reaches the sub-extension. Keying them on the
        extension-definition-relative form instead is what loses the provider.
        """
        if not owner_path:
            return []
        specs = self._subextension_specs(extension_url) or {}
        slices = []
        for slice_name, spec in specs.items():
            cardinality = {
                "min": spec.get("min", 0) or 0,
                "max": spec.get("max", "1") or "1",
            }
            slices.append(
                {
                    "path": f"{owner_path}.extension",
                    "id": f"{owner_path}.extension:{slice_name}",
                    "slice_identity": f"{owner_path}.extension:{slice_name}",
                    "sliceName": slice_name,
                    "cardinality": cardinality,
                    "type": [{"code": "Extension"}],
                    "children": [],
                }
            )
        return slices

    @staticmethod
    def _subtree_pins_a_value(field):
        """Whether this element or anything below it carries a fixed/pattern value."""
        if has_fixed_value(field.get("fixed_value")):
            return True
        nested = list(field.get("children") or [])
        nested += list(field.get("type_structure") or [])
        for type_ref in field.get("type") or []:
            if isinstance(type_ref, dict):
                nested += list(type_ref.get("type_structure") or [])
        return any(
            _ExtensionRulesMixin._subtree_pins_a_value(child)
            for child in nested
            if isinstance(child, dict)
        )

    @staticmethod
    def _emittable_extension_children(children, mappings):
        """Children an extension can actually populate.

        Either a mapping supplies the value or the profile pins it. With neither,
        the generated rule would be a TODO placeholder writing an element that FHIR
        does not define (a bare ``value`` rather than ``valueString``), nested inside
        an extension that itself never fires. Relativising the child paths (see the
        ``parent_path`` argument below) removed the depth-based guard in
        ``create_mappable_field_rule`` that used to suppress these by accident, so
        the exclusion is made explicit here instead.
        """
        keep = []
        for child in children:
            path = child.get("path", "")
            provided = path and any(
                key == path or key.startswith(f"{path}.") for key in (mappings or {})
            )
            if provided or _ExtensionRulesMixin._subtree_pins_a_value(child):
                keep.append(child)
        return keep

    @staticmethod
    def _value_children_required_by_ext1(children, extension_min, has_sub_extensions):
        """Mark a profile-pinned ``value[x]`` of a required extension as required.

        ``ext-1`` says an extension carries either sub-extensions or a value, never
        neither. So a required extension slice with no sub-extensions must have a
        value even where the snapshot leaves ``value[x]`` at ``min = 0`` — which is
        the usual shape, since the cardinality is inherited from the Extension
        datatype rather than restated by the profile (nictiz cio pins
        ``patternCodeableConcept`` on an otherwise optional ``value[x]``).

        Only pinned values are promoted. Without a fixed/pattern value there is
        nothing to emit, and inventing a placeholder would be worse than the gap.
        """
        if extension_min < 1 or has_sub_extensions:
            return children
        promoted = []
        for child in children:
            path = str(child.get("path", ""))
            if path.rsplit(".", 1)[-1].startswith("value") and has_fixed_value(
                child.get("fixed_value")
            ):
                child = {**child, "is_required": True}
            promoted.append(child)
        return promoted

    @staticmethod
    def _extension_is_profile_determined(children, extension_min, has_sub_extensions):
        """Whether url + value are both pinned, so no source data is involved."""
        if extension_min < 1 or has_sub_extensions:
            return False
        return any(
            str(child.get("path", "")).rsplit(".", 1)[-1].startswith("value")
            and has_fixed_value(child.get("fixed_value"))
            for child in children
        )

    def _extension_is_modifier(self, extension_url):
        """Whether the referenced Extension definition declares ``isModifier``."""

        if not extension_url or extension_url == "TODO_EXTENSION_URL":
            return False
        cache = getattr(self, "_extension_modifier_cache", None)
        if cache is None:
            cache = {}
            self._extension_modifier_cache = cache
        if extension_url in cache:
            return cache[extension_url]
        try:
            sd = resolve_url(extension_url, self.app_state)
        except Exception as exc:
            logger.debug(
                "Could not resolve extension SD %s for modifier classification: %s",
                extension_url,
                exc,
            )
            cache[extension_url] = False
            return False
        snapshot = _attr(sd, "snapshot") if sd else None
        is_modifier = False
        for element in (_attr(snapshot, "element", []) if snapshot else []) or []:
            if _attr(element, "id") == "Extension":
                is_modifier = bool(_attr(element, "isModifier", False))
                break
        cache[extension_url] = is_modifier
        return is_modifier

    def _subextension_specs(self, extension_url):
        """resolve sub-extension slices of complex extension StructureDefinition"""
        specs = {}
        if not extension_url or extension_url == "TODO_EXTENSION_URL":
            return specs
        try:
            sd = resolve_url(extension_url, self.app_state)
        except Exception as e:
            logger.warning(
                "Could not resolve extension SD %s for sub-extension specs: %s",
                extension_url,
                e,
            )
            return specs
        if not sd:
            return specs

        snapshot = _attr(sd, "snapshot")
        elements = (_attr(snapshot, "element", []) if snapshot else []) or []
        for el in elements:
            eid = _attr(el, "id") or ""
            if not eid.startswith("Extension.extension:"):
                continue
            rest = eid[len("Extension.extension:") :]
            parts = rest.split(".", 1)
            sname = parts[0]
            spec = specs.setdefault(
                sname,
                {
                    "url": sname,
                    "value_type": None,
                    "value_set": None,
                    "type_profile": None,
                    "min": 0,
                    "max": "1",
                },
            )
            if len(parts) == 1:
                spec["min"] = _attr(el, "min", 0) or 0
                spec["max"] = _attr(el, "max", "1") or "1"
                for t in _attr(el, "type", []) or []:
                    profiles = _attr(t, "profile", []) or []
                    if profiles:
                        spec["type_profile"] = profiles[0]
                        break
            elif parts[1] == "url":
                fixed = (
                    _attr(el, "fixedUri")
                    or _attr(el, "fixedString")
                    or _attr(el, "fixedCode")
                )
                if fixed:
                    spec["url"] = fixed
            elif parts[1].startswith("value"):
                types = _attr(el, "type", []) or []
                if len(types) == 1:
                    code = _attr(types[0], "code")
                    if code:
                        spec["value_type"] = code
                binding = _attr(el, "binding")
                vs = _attr(binding, "valueSet") if binding else None
                if vs:
                    spec["value_set"] = vs
        return specs

    def _source_context_for(self, parent_source_context, source_element):
        """Bind a root-level source field to the source root, not a parent variable"""
        if parent_source_context != "source" and source_element in getattr(
            self, "source_field_types", {}
        ):
            return "source"
        return parent_source_context

    @staticmethod
    def _extension_value_def(sd):
        """Return (value_type, value_set, target_profiles) for an Extension SD's value"""
        snapshot = _attr(sd, "snapshot")
        for el in (_attr(snapshot, "element", []) if snapshot else []) or []:
            element_id = str(_attr(el, "id") or "")
            if not element_id.startswith("Extension.value"):
                continue
            if element_id != "Extension.value[x]" and "." in element_id[len("Extension."):]:
                continue  # a child of the value, not the value itself
            types = _attr(el, "type", []) or []
            vt = _attr(types[0], "code") if len(types) == 1 else None
            if vt is None and element_id != "Extension.value[x]":
                # `Extension.valueReference` names its own type even with no type list.
                vt = element_id[len("Extension.value"):] or None
            binding = _attr(el, "binding")
            vs = _attr(binding, "valueSet") if binding else None
            targets = []
            for type_ref in types:
                targets.extend(_attr(type_ref, "targetProfile", []) or [])
            return vt, vs, [t.split("|", 1)[0] for t in targets if t]
        return None, None, []

    def _resolve_coded_extension_value(self, spec, slice_name, parent_extension_url):
        """check if sub-extension value choice type is coded bound to value set"""
        value_type = spec.get("value_type")
        value_set = spec.get("value_set")

        def _safe_resolve(url):
            try:
                return resolve_url(url, self.app_state)
            except Exception as e:
                logger.warning(
                    "Could not resolve extension SD %s for coded value: %s", url, e
                )
                return None

        if not value_set and spec.get("type_profile"):
            sd = _safe_resolve(spec["type_profile"])
            if sd:
                vt, vs, _ = self._extension_value_def(sd)
                if vs:
                    value_type, value_set = vt or value_type, vs

        if not value_set and value_type is None and parent_extension_url and slice_name:
            base = parent_extension_url.rsplit("/", 1)[0]
            kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", slice_name).lower()
            candidate = f"{base}/{kebab}"
            sd = _safe_resolve(candidate)
            if sd:
                vt, vs, _ = self._extension_value_def(sd)
                if vs:
                    value_type, value_set = vt, vs

        if not value_set:
            return None
        coded_type = (
            value_type
            if value_type in ("code", "Coding", "CodeableConcept")
            else "code"
        )
        options = []
        try:
            options = expand_valueset(value_set, [], self.app_state) or []
        except Exception as e:
            logger.warning(
                "Could not expand value set %s for coded extension value: %s",
                value_set,
                e,
            )
        return {"value_type": coded_type, "value_set": value_set, "options": options}

    def create_extension_rule(
        self,
        field,
        clean_path,
        parent_source_context,
        parent_target_context,
        automapped_mappings=None,
    ):
        """
        Creates a rule for FHIR extensions
        """
        path = field["path"]
        field_type = field.get("type", "Extension")
        children = field.get("children", [])

        extension_url = (
            field.get("extension_url") or field.get("url") or "TODO_EXTENSION_URL"
        )

        path_parts = clean_path.split(".")
        field_name = path_parts[-1]

        if extension_url == "TODO_EXTENSION_URL" and ":" in field_name:
            extension_url = field_name.split(":")[-1]

        if extension_url == "TODO_EXTENSION_URL" and isinstance(field_type, list):
            for _t in field_type:
                _profiles = (_t.get("profile") if isinstance(_t, dict) else None) or []
                if _profiles:
                    extension_url = _profiles[0]
                    break

        is_slice, slice_info = parse_slice_info(field_name, field)

        if not is_slice and extension_url == "TODO_EXTENSION_URL":
            return None

        lookup_path = path
        if is_slice and slice_info.get("slice_name"):
            if ":" not in path.rsplit(".", 1)[-1]:
                lookup_path = f"{path}:{slice_info['slice_name']}"
        mapping_key = None
        if automapped_mappings:
            for _k in (lookup_path, path):
                if _k in automapped_mappings:
                    mapping_key = _k
                    break
        minimum = int((field.get("cardinality") or {}).get("min", 0) or 0)
        value_is_pinned = self._extension_is_profile_determined(
            children, minimum, bool(self._sub_extension_slices(children))
        )
        if is_slice and minimum > 0 and mapping_key is None and not value_is_pinned:
            recorder = getattr(self, "record_diagnostic", None)
            if callable(recorder):
                recorder(
                    "required-extension-provider-missing",
                    f"Required extension slice {lookup_path} has no authored "
                    "source provider. Its URL is known, but its semantic value "
                    "cannot be inferred from the target profile.",
                    path=lookup_path,
                    extension_url=extension_url,
                    severity="error",
                )

        if is_slice:
            slice_name = slice_info["slice_name"]
            # rule/variable names must match ^[A-Za-z0-9\-.]+$ (sanitize e.g. underscores)
            var_suffix = clean_field_name(slice_name)
            rule_name = f"map-extension-{var_suffix}"
        else:
            path_suffix = clean_field_name(clean_path.replace(".", "-"))
            rule_name = f"map-{path_suffix}"
            var_suffix = path_suffix

        rule = StructureMapGroupRule.model_construct()
        rule.name = rule_name
        type_repr = str(field_type)
        if len(type_repr) > _DOC_TYPE_MAXLEN:
            type_repr = (
                type_repr[:_DOC_TYPE_MAXLEN]
                + f"… [truncated; full structure in the registry entry for {extension_url}]"
            )
        rule.documentation = (
            f"Maps to extension: {extension_url} | "
            f"Type: {type_repr} | "
            f"Path: {path}"
        )

        source_element = f"TODO-MAP-{var_suffix.upper()}_SOURCE"
        if mapping_key:
            source_element = self._as_local_element(automapped_mappings[mapping_key])

        # Non-slice extensions with no mapped source generate unusable TODO rules — skip them.
        if not is_slice and source_element.startswith("TODO"):
            return None

        source = StructureMapGroupRuleSource.model_construct()
        source.context = self._source_context_for(parent_source_context, source_element)
        source.element = source_element
        source.variable = f"src-{var_suffix}"
        rule.source = [source]

        profile_determined = not mapping_key and self._extension_is_profile_determined(
            children, minimum, bool(self._sub_extension_slices(children))
        )
        if profile_determined:
            source.element = None

        declared_modifier = self._extension_is_modifier(extension_url)
        profile_element = path.rsplit(".", 1)[-1].split(":", 1)[0]
        target_element = (
            "modifierExtension"
            if profile_element == "modifierExtension"
            else "extension"
        )
        if declared_modifier and target_element != "modifierExtension":
            recorder = getattr(self, "record_diagnostic", None)
            if callable(recorder):
                recorder(
                    "modifier-extension-path-mismatch",
                    f"{path} references modifier extension {extension_url}; "
                    "emission follows the profile-declared extension slice path.",
                    path=path,
                    extension_url=extension_url,
                    severity="warning",
                )
        target = StructureMapGroupRuleTarget.model_construct()
        target.context = parent_target_context
        target.element = target_element
        target.variable = f"ext-{var_suffix}"
        target.transform = "create"
        target.parameter = [{"valueString": "Extension"}]
        rule.target = [target]

        nested_rules = []
        # URL Rule - Fixed Value
        url_rule = StructureMapGroupRule.model_construct()
        url_rule.name = fit_rule_name(f"set-extension-url-{var_suffix}")
        url_rule.documentation = f"Sets the extension URL to {extension_url}"

        url_target = StructureMapGroupRuleTarget.model_construct()
        url_target.context = f"ext-{var_suffix}"
        url_target.element = "url"
        url_target.transform = "copy"  # Will set value to parameter
        url_target.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(
                valueString=extension_url
            )
        ]
        url_rule.target = [url_target]
        url_source = StructureMapGroupRuleSource.model_construct()
        url_source.context = f"src-{var_suffix}"
        url_rule.source = [url_source]

        nested_rules.append(url_rule)
        sub_ext_slices = self._sub_extension_slices(children)
        if not sub_ext_slices:
            sub_ext_slices = self._subextension_slices_from_definition(
                extension_url, lookup_path
            )

        if sub_ext_slices:
            specs = self._subextension_specs(extension_url)
            for sl in sub_ext_slices:
                sname = sl.get("sliceName")
                spec = specs.get(sname, {})
                sub_min = spec.get("min", sl.get("cardinality", {}).get("min", 0))
                candidates = [c for c in (sl.get("id"), sl.get("slice_identity"),
                                          sl.get("path")) if c]
                is_mapped = any(
                    key == candidate or key.startswith(f"{candidate}.")
                    for key in (automapped_mappings or {})
                    for candidate in candidates
                )
                if sub_min < 1 and not is_mapped:
                    continue
                enriched = dict(sl)
                enriched["extension_url"] = spec.get("url") or sname
                if spec.get("value_type"):
                    enriched["value_type"] = spec["value_type"]
                coded = self._resolve_coded_extension_value(spec, sname, extension_url)
                if coded:
                    enriched["_value_binding"] = coded
                sub_clean_path = enriched["path"].split(".")[
                    -1
                ]  # e.g. extension slices
                sub_rule = self.create_extension_rule(
                    enriched,
                    sub_clean_path,
                    parent_source_context,
                    f"ext-{var_suffix}",
                    automapped_mappings=automapped_mappings,
                )
                if sub_rule:
                    nested_rules.append(sub_rule)

        _META_SUFFIXES = (
            ".id",
            ".meta",
            ".implicitRules",
            ".language",
            ".text",
            ".contained",
            ".modifierExtension",
            ".extension",
        )
        non_url_children = [
            c
            for c in (children or [])
            if not (c.get("path", "").endswith(".url") and c.get("fixed_value"))
            and not c.get("path", "").endswith(_META_SUFFIXES)
        ]
        if non_url_children:
            child_mappings = (
                dict(automapped_mappings) if automapped_mappings else {}
            )
            if is_slice and lookup_path != path and automapped_mappings:
                qual_prefix = f"{lookup_path}."
                for _k, _v in automapped_mappings.items():
                    if _k.startswith(qual_prefix):
                        child_mappings.setdefault(
                            f"{path}.{_k[len(qual_prefix):]}", _v
                        )
            if path in child_mappings:
                ext_source = child_mappings[path]
                for child in non_url_children:
                    child_path = child.get("path", "")
                    if child_path and child_path not in child_mappings:
                        child_mappings[child_path] = ext_source

            non_url_children = self._value_children_required_by_ext1(
                self._emittable_extension_children(non_url_children, child_mappings),
                minimum,
                bool(sub_ext_slices),
            )
            value_rules = self.create_field_rules(
                "",
                fields=non_url_children,
                parent_source_context=parent_source_context,
                parent_target_context=f"ext-{var_suffix}",
                automapped_mappings=child_mappings,
                parent_path=path,
            )
            if value_rules:
                nested_rules.extend(value_rules)
            elif self._extension_has_provider(
                mapping_key, path, lookup_path, automapped_mappings
            ):
                self._report_url_only_extension(
                    lookup_path, extension_url,
                    "none of its value children produced an executable rule",
                )
                return None
        elif not sub_ext_slices:
            coded = field.get("_value_binding")
            if not coded:
                coded = self._resolve_coded_extension_value(
                    {
                        "value_type": field.get("value_type"),
                        "value_set": field.get("valueSetUrl"),
                        "type_profile": (
                            extension_url
                            if extension_url != "TODO_EXTENSION_URL"
                            else None
                        ),
                    },
                    slice_info["slice_name"] if is_slice else None,
                    None,
                )
            if coded:
                value_field = dict(field)
                value_field["path"] = lookup_path
                value_field["type"] = coded["value_type"]
                value_field["options"] = coded.get("options", [])
                value_field["valueSetUrl"] = coded.get("value_set")
                cm_url = (
                    self._generate_concept_map(
                        value_field,
                        coded.get("options", []),
                        var_suffix,
                        automapped_mappings=automapped_mappings,
                    )
                    or "TODO-resolveConceptMap"
                )
                value_rule = self._create_translate_rule(
                    value_field,
                    "value",
                    parent_source_context,
                    f"ext-{var_suffix}",
                    cm_url,
                    automapped_mappings,
                )
                nested_rules.append(value_rule)
                rule.rule = nested_rules
                return rule

            sd_target_profiles = []
            if not field.get("value_type") and extension_url != "TODO_EXTENSION_URL":
                try:
                    ext_sd = resolve_url(extension_url, self.app_state)
                except Exception:
                    ext_sd = None
                if ext_sd:
                    sd_value_type, _, sd_target_profiles = self._extension_value_def(
                        ext_sd
                    )
                    if sd_value_type:
                        field = {**field, "value_type": sd_value_type}
            value_type = infer_extension_value_type(field_type, field)

            reference_target = field.get("reference_target")
            if not reference_target and value_type == "valueReference":
                # The extension's own definition names what it points at. The profile
                # element that carries the extension is typed `Extension`, so the
                # parser never sees the Reference and cannot fill `reference_target`
                # from the profile alone — read it back off the extension definition.
                reference_target = self._reference_target_from_profiles(
                    sd_target_profiles
                )
            if value_type == "valueReference" and not reference_target:
                # A Reference-valued extension whose target cannot be resolved has
                # nothing to point at, so only the url would be written. With a live
                # source the rule fires and ships an extension that satisfies neither
                # half of ext-1; without one it is dead scaffolding. Refuse in the
                # first case and say why — a reported gap beats invalid output.
                if self._extension_has_provider(
                    mapping_key, path, lookup_path, automapped_mappings
                ):
                    self._report_url_only_extension(
                        lookup_path, extension_url,
                        "its value is a Reference whose target profile could not be "
                        "resolved to a generatable resource",
                    )
                    return None
                rule.rule = nested_rules
                return rule
            if reference_target and value_type == "valueReference":
                nested_rules.append(
                    self._extension_reference_contract_rule(
                        path,
                        var_suffix,
                        extension_url,
                        reference_target,
                        sd_target_profiles,
                        parent_source_context,
                    )
                )
                rule.rule = nested_rules
                return rule

            mapped_source_element = (
                self._as_local_element(automapped_mappings[mapping_key])
                if mapping_key
                else None
            )

            target_code = field.get("value_type") or ""
            src_type = (
                (getattr(self, "source_field_types", None) or {}).get(
                    mapped_source_element, ""
                )
                if mapped_source_element
                else ""
            )
            type_mismatch = (
                mapped_source_element
                and target_code
                and is_primitive_type(target_code)
                and src_type
                and src_type.lower() != target_code.lower()
            )
            if type_mismatch and target_code == "boolean":
                for suffix, cond, literal in (
                    ("true", "$this = '1' or $this = 'true'", True),
                    ("false", "$this = '0' or $this = 'false'", False),
                ):
                    bool_rule = StructureMapGroupRule.model_construct()
                    bool_rule.name = (
                        f"set-extension-value-{clean_field_name(var_suffix)}-{suffix}"
                    )
                    bool_rule.documentation = (
                        f"Sets the extension value to {literal} "
                        f"(boolean coercion from {src_type})"
                    )
                    bool_source = StructureMapGroupRuleSource.model_construct()
                    bool_source.context = self._source_context_for(
                        parent_source_context, mapped_source_element
                    )
                    bool_source.element = mapped_source_element
                    bool_source.variable = "srcValue"
                    bool_source.condition = cond
                    bool_rule.source = [bool_source]
                    bool_target = StructureMapGroupRuleTarget.model_construct()
                    bool_target.context = f"ext-{var_suffix}"
                    bool_target.element = "valueBoolean"
                    bool_target.transform = "copy"
                    bool_target.parameter = [
                        fixed_scalar_parameter(literal, "boolean")
                    ]
                    bool_rule.target = [bool_target]
                    nested_rules.append(bool_rule)
                rule.rule = nested_rules
                return rule

            value_rule = StructureMapGroupRule.model_construct()
            value_rule.name = f"set-extension-value-{clean_field_name(var_suffix)}"
            value_rule.documentation = f"Sets the extension value (type: {value_type})"

            value_source = StructureMapGroupRuleSource.model_construct()
            value_source.context = self._source_context_for(
                parent_source_context, mapped_source_element
            )
            value_source.element = mapped_source_element or "TODO-VALUE-SOURCE"
            value_source.variable = "srcValue"
            _STRING_VALUE_TYPES = {
                "valueString",
                "valueUri",
                "valueUrl",
                "valueCode",
                "valueMarkdown",
                "valueId",
                "valueOid",
                "valueBase64Binary",
            }
            if (
                value_type not in _STRING_VALUE_TYPES
                and not value_source.element.startswith("TODO")
            ):
                value_source.condition = "$this != ''"
            value_rule.source = [value_source]

            value_target = StructureMapGroupRuleTarget.model_construct()
            value_target.context = f"ext-{var_suffix}"

            # Set appropriate transform based on value type
            if value_type.startswith("value") and not is_primitive_type(
                value_type[5:].lower()
            ):
                value_target.element = "value"
                value_target.transform = "create"
                value_target.parameter = [{"valueString": value_type[5:]}]
            elif type_mismatch:
                # non-boolean primitive mismatch (boolean returned above):
                # `copy` would keep the source type cast to the target type
                value_target.element = value_type
                value_target.transform = "cast"
                value_target.parameter = [
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueId=value_source.variable
                    ),
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueString=target_code
                    ),
                ]
            else:
                value_target.element = value_type
                value_target.transform = "copy"
                value_target.parameter = [
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueId=value_source.variable
                    )
                ]

            value_rule.target = [value_target]
            nested_rules.append(value_rule)

        if emits_only_url(nested_rules, f"ext-{var_suffix}") and rule_can_fire(source):
            self._report_url_only_extension(
                lookup_path,
                extension_url,
                "neither a sub-extension nor a value produced an executable rule "
                "(every sub-extension slice was unmapped, refused, or absent)",
            )
            return None

        rule.rule = nested_rules

        return rule
