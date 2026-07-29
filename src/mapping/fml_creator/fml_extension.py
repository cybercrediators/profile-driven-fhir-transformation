import re

import logging
from fhir.resources.R4B.structuremap import (
    StructureMapGroupRule,
    StructureMapGroupRuleTarget,
    StructureMapGroupRuleTargetParameter,
    StructureMapGroupRuleSource,
    StructureMapGroupRuleDependent,
)
from mapping.fml_creator.fml_helper import (
    infer_extension_value_type,
    is_primitive_type,
    parse_slice_info,
    clean_field_name,
    attr as _attr,
)
from parser.resource_parser.value_expander import expand_valueset
from data_handling.url_resolver.fhir_url_resolver import resolve_url

logger = logging.getLogger(__name__)

_DOC_TYPE_MAXLEN = 4096


class _ExtensionRulesMixin:
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

    @staticmethod
    def _extension_value_def(sd):
        """Return (value_type, value_set) for a standalone Extension SD's Extension.value[x]."""
        snapshot = _attr(sd, "snapshot")
        for el in (_attr(snapshot, "element", []) if snapshot else []) or []:
            if _attr(el, "id") == "Extension.value[x]":
                types = _attr(el, "type", []) or []
                vt = _attr(types[0], "code") if len(types) == 1 else None
                binding = _attr(el, "binding")
                vs = _attr(binding, "valueSet") if binding else None
                return vt, vs
        return None, None

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
                vt, vs = self._extension_value_def(sd)
                if vs:
                    value_type, value_set = vt or value_type, vs

        if not value_set and value_type is None and parent_extension_url and slice_name:
            base = parent_extension_url.rsplit("/", 1)[0]
            kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", slice_name).lower()
            candidate = f"{base}/{kebab}"
            sd = _safe_resolve(candidate)
            if sd:
                vt, vs = self._extension_value_def(sd)
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
        if is_slice and minimum > 0 and mapping_key is None:
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
        source.context = parent_source_context
        source.element = source_element
        source.variable = f"src-{var_suffix}"
        rule.source = [source]

        declared_modifier = self._extension_is_modifier(extension_url)
        target_element = (
            "modifierExtension"
            if path.endswith(".modifierExtension") or declared_modifier
            else "extension"
        )
        if declared_modifier and not path.endswith(".modifierExtension"):
            recorder = getattr(self, "record_diagnostic", None)
            if callable(recorder):
                recorder(
                    "modifier-extension-rerouted",
                    f"{path} references modifier extension {extension_url}; "
                    "the target was rerouted to modifierExtension.",
                    path=path,
                    extension_url=extension_url,
                    severity="information",
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
        url_rule.name = f"set-extension-url-{var_suffix}"
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
        sub_ext_slices = []
        for c in children or []:
            if c.get("path", "").endswith(".extension") and c.get("slices"):
                for sl in c["slices"]:
                    if isinstance(sl, dict) and sl.get("sliceName"):
                        sub_ext_slices.append(sl)

        if sub_ext_slices:
            specs = self._subextension_specs(extension_url)
            for sl in sub_ext_slices:
                sname = sl.get("sliceName")
                spec = specs.get(sname, {})
                sub_min = spec.get("min", sl.get("cardinality", {}).get("min", 0))
                is_mapped = bool(
                    automapped_mappings and sl.get("path") in automapped_mappings
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

            value_rules = self.create_field_rules(
                "",
                fields=non_url_children,
                parent_source_context=parent_source_context,
                parent_target_context=f"ext-{var_suffix}",
                automapped_mappings=child_mappings,
            )
            if value_rules:
                nested_rules.extend(value_rules)
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

            if not field.get("value_type") and extension_url != "TODO_EXTENSION_URL":
                try:
                    ext_sd = resolve_url(extension_url, self.app_state)
                except Exception:
                    ext_sd = None
                if ext_sd:
                    sd_value_type, _ = self._extension_value_def(ext_sd)
                    if sd_value_type:
                        field = {**field, "value_type": sd_value_type}
            value_type = infer_extension_value_type(field_type, field)

            reference_target = field.get("reference_target")
            if value_type == "valueReference" and not reference_target:
                rule.rule = nested_rules
                return rule
            if reference_target and value_type == "valueReference":
                value_rule = StructureMapGroupRule.model_construct()
                value_rule.name = f"set-extension-value-{clean_field_name(var_suffix)}"
                value_rule.documentation = (
                    f"Sets the extension value (Reference to {reference_target})"
                )

                value_source = StructureMapGroupRuleSource.model_construct()
                value_source.context = parent_source_context
                value_source.variable = "srcVal"
                value_rule.source = [value_source]

                value_target = StructureMapGroupRuleTarget.model_construct()
                value_target.context = f"ext-{var_suffix}"
                value_target.element = "valueReference"
                value_target.variable = "valRef"
                value_target.transform = "create"
                value_target.parameter = [{"valueString": "Reference"}]
                value_rule.target = [value_target]

                dependent = StructureMapGroupRuleDependent.model_construct(
                    name=f"Reference{reference_target}", variable=["srcVal", "valRef"]
                )
                value_rule.dependent = [dependent]

                nested_rules.append(value_rule)
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
                    bool_source.context = parent_source_context
                    bool_source.element = mapped_source_element
                    bool_source.variable = "srcValue"
                    bool_source.condition = cond
                    bool_rule.source = [bool_source]
                    bool_target = StructureMapGroupRuleTarget.model_construct()
                    bool_target.context = f"ext-{var_suffix}"
                    bool_target.element = "valueBoolean"
                    bool_target.transform = "copy"
                    bool_target.parameter = [
                        StructureMapGroupRuleTargetParameter.model_construct(
                            valueBoolean=literal
                        )
                    ]
                    bool_rule.target = [bool_target]
                    nested_rules.append(bool_rule)
                rule.rule = nested_rules
                return rule

            value_rule = StructureMapGroupRule.model_construct()
            value_rule.name = f"set-extension-value-{clean_field_name(var_suffix)}"
            value_rule.documentation = f"Sets the extension value (type: {value_type})"

            value_source = StructureMapGroupRuleSource.model_construct()
            value_source.context = parent_source_context
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
        rule.rule = nested_rules

        return rule
