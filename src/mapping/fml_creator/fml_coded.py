from typing import Optional

import logging

logger = logging.getLogger(__name__)
from fhir.resources.R4B.structuremap import (
    StructureMapGroupRule,
    StructureMapGroupRuleTarget,
    StructureMapGroupRuleTargetParameter,
    StructureMapGroupRuleSource,
)
from fhir.resources.R4B.conceptmap import (
    ConceptMap,
    ConceptMapGroup,
    ConceptMapGroupElement,
    ConceptMapGroupElementTarget,
)
from mapping.fml_creator.fml_helper import (
    clean_field_name,
    fhir_type_suffix,
    fixed_scalar_literal,
    type_codes,
    attr as _attr,
)
from data_handling.url_resolver.fhir_url_resolver import resolve_url
import json


class _CodedRulesMixin:
    @staticmethod
    def _canonical_coded_type(field):
        """Return one coded type from either parser or generation shape."""
        codes = type_codes(field.get("type", "code"))
        if len(codes) == 1 and codes[0] in ("code", "Coding", "CodeableConcept"):
            return codes[0]
        return None

    @staticmethod
    def _coded_source_keys(field) -> list:
        """Candidate ``automapped_mappings`` keys for a coded field's source.

        """
        path = field.get("path")
        if not path:
            return []
        ftype = _CodedRulesMixin._canonical_coded_type(field)
        keys = [path]
        if ftype == "CodeableConcept":
            keys += [f"{path}.coding.code", f"{path}.coding", f"{path}.text"]
        elif ftype == "Coding":
            keys += [f"{path}.code"]
        return keys

    def _resolve_coded_source(self, field, automapped_mappings):
        """Return the local source element for a coded field, or ``None``.

        """
        if not automapped_mappings:
            return None
        for key in self._coded_source_keys(field):
            if key in automapped_mappings:
                return self._as_local_element(automapped_mappings[key])
        return None

    def _coding_structure_rules(
        self,
        *,
        field_type,
        outer_target,
        outer_element,
        inner_name_base,
        system_name_base,
        system_uri,
        code_transform,
        code_params,
        guard_empty,
    ):
        """analyze nested rule list populating system and code rules"""

        def _src():
            s = StructureMapGroupRuleSource.model_construct(context="src")
            if guard_empty:
                s.condition = "$this != ''"
            return s

        def _leaf_rules():
            rules = []
            if system_uri:
                sys_rule = StructureMapGroupRule.model_construct(
                    name=f"set-{system_name_base}-system"
                )
                sys_rule.source = [_src()]
                sys_tgt = StructureMapGroupRuleTarget.model_construct(
                    context="tgt-coding", element="system", transform="copy"
                )
                sys_tgt.parameter = [
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueString=system_uri
                    )
                ]
                sys_rule.target = [sys_tgt]
                rules.append(sys_rule)
            code_rule = StructureMapGroupRule.model_construct(
                name=f"set-{inner_name_base}-code"
            )
            code_rule.source = [_src()]
            code_tgt = StructureMapGroupRuleTarget.model_construct(
                context="tgt-coding", element="code", transform=code_transform
            )
            code_tgt.parameter = code_params
            code_rule.target = [code_tgt]
            rules.append(code_rule)
            return rules

        outer_target.element = outer_element
        outer_target.transform = "create"
        if field_type == "Coding":
            outer_target.variable = "tgt-coding"
            outer_target.parameter = [
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString="Coding"
                )
            ]
            return _leaf_rules()

        # CodeableConcept: outer create, then a coding rule that creates Coding + leaf rules.
        outer_target.variable = "tgt-cc"
        outer_target.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(
                valueString="CodeableConcept"
            )
        ]
        coding_rule = StructureMapGroupRule.model_construct(
            name=f"add-{inner_name_base}-coding"
        )
        coding_rule.source = [_src()]
        coding_tgt = StructureMapGroupRuleTarget.model_construct(
            context="tgt-cc",
            element="coding",
            variable="tgt-coding",
            transform="create",
        )
        coding_tgt.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(valueString="Coding")
        ]
        coding_rule.target = [coding_tgt]
        coding_rule.rule = _leaf_rules()
        return [coding_rule]

    def _create_translate_rule(
        self,
        field,
        field_name,
        parent_source_context,
        parent_target_context,
        cm_url,
        automapped_mappings,
    ):
        field_type = self._canonical_coded_type(field)

        _CODED_VALUE_X_TYPES = ("code", "Coding", "CodeableConcept")
        is_value_x = field_name == "value" and field_type in _CODED_VALUE_X_TYPES
        if is_value_x:
            field_name = f"value{fhir_type_suffix(field_type)}"
        create_element = "value" if is_value_x else field_name

        rule = StructureMapGroupRule.model_construct()
        rule.name = f"map-{clean_field_name(field_name)}-translate"
        rule.documentation = f"Translate mapping for {field['path']} using {cm_url}"

        resolved = self._resolve_coded_source(field, automapped_mappings)
        source_element = resolved or f"TODO-SOURCE-FOR-{field_name.upper()}"

        guard_empty = field.get("cardinality", {}).get("min", 0) == 0

        source = StructureMapGroupRuleSource.model_construct()
        source.context = parent_source_context
        source.element = source_element
        source.variable = "src"
        if guard_empty:
            source.condition = "$this != ''"
        rule.source = [source]

        # Target
        target = StructureMapGroupRuleTarget.model_construct()
        target.context = parent_target_context

        # helper for translate param
        def _translate_params(src_var, url):
            return [
                StructureMapGroupRuleTargetParameter.model_construct(valueId=src_var),
                StructureMapGroupRuleTargetParameter.model_construct(valueString=url),
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString="code"
                ),
            ]

        if field_type == "code":
            target.element = field_name
            target.transform = "translate"
            target.parameter = _translate_params("src", cm_url)
            rule.target = [target]

        elif field_type in ("Coding", "CodeableConcept"):
            systems = {
                o.get("system") for o in (field.get("options") or []) if o.get("system")
            }
            if len(systems) > 1:
                rule.rule = self._coding_translate_rules(
                    field_type=field_type,
                    outer_target=target,
                    outer_element=create_element,
                    inner_name_base=field_name,
                    cm_url=cm_url,
                    src_var="src",
                    guard_empty=guard_empty,
                )
            else:
                rule.rule = self._coding_structure_rules(
                    field_type=field_type,
                    outer_target=target,
                    outer_element=create_element,
                    inner_name_base=field_name,
                    system_name_base=clean_field_name(field_name),
                    system_uri=self._system_from_options(field.get("options", [])),
                    code_transform="translate",
                    code_params=_translate_params("src", cm_url),
                    guard_empty=guard_empty,
                )
            rule.target = [target]

        return rule

    def _coding_translate_rules(
        self,
        *,
        field_type,
        outer_target,
        outer_element,
        inner_name_base,
        cm_url,
        src_var,
        guard_empty,
    ):
        """Multi-system variant of _coding_structure_rules"""

        def _coding_params():
            return [
                StructureMapGroupRuleTargetParameter.model_construct(valueId=src_var),
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString=cm_url
                ),
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString="Coding"
                ),
            ]

        outer_target.element = outer_element
        if field_type == "Coding":
            # The element itself is a Coding — translate straight into it.
            outer_target.transform = "translate"
            outer_target.parameter = _coding_params()
            return []

        # CodeableConcept: create the CC, then set .coding from the translate result.
        outer_target.transform = "create"
        outer_target.variable = "tgt-cc"
        outer_target.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(
                valueString="CodeableConcept"
            )
        ]
        coding_src = StructureMapGroupRuleSource.model_construct(context="src")
        if guard_empty:
            coding_src.condition = "$this != ''"
        coding_rule = StructureMapGroupRule.model_construct(
            name=f"set-{inner_name_base}-coding-translate"
        )
        coding_rule.source = [coding_src]
        coding_tgt = StructureMapGroupRuleTarget.model_construct(
            context="tgt-cc", element="coding", transform="translate"
        )
        coding_tgt.parameter = _coding_params()
        coding_rule.target = [coding_tgt]
        return [coding_rule]

    def create_conditional_coding_rules(
        self,
        field,
        field_name,
        parent_source_context,
        parent_target_context,
        is_slice=False,
        slice_info=None,
        automapped_mappings=None,
    ):
        field_type = self._canonical_coded_type(field)
        if field_type is None:
            logger.error(
                "Cannot create coded rule for %s with type %r",
                field.get("path"),
                field.get("type"),
            )
            return None
        if field.get("type") != field_type:
            field = dict(field)
            field["type"] = field_type
        options = field.get("options", [])
        fixed_value = field.get("fixed_value")

        fixed_codings = self._fixed_coding_slices(field)
        resolved_source = self._resolve_coded_source(field, automapped_mappings)
        has_source = resolved_source is not None

        if fixed_value:
            is_pattern = bool(field.get("is_pattern"))
            min_c = int((field.get("cardinality") or {}).get("min", 0) or 0)
            is_req = min_c > 0 or bool(field.get("is_required"))
            if is_pattern and not is_req and not has_source:
                logger.info(
                    "Optional pattern-only coded element %s has no source — emitting nothing.",
                    field.get("path"),
                )
                return None
            return self._create_fixed_value_rule(
                field_type,
                fixed_value,
                field_name,
                parent_target_context,
                parent_source_context,
            )
        fixed_system = self._fixed_system_from_field(field)

        if not fixed_codings and not has_source:
            leaf_coding = self._fixed_coding_from_leaves(field)
            if leaf_coding:
                fixed_codings = [leaf_coding]

        if fixed_system and has_source and not fixed_codings and not options:
            logger.info(
                f"Fixed coding.system for {field_name} ({fixed_system}) — generating direct copy rule."
            )
            return self._create_direct_copy_rule(
                field,
                field_name,
                parent_source_context,
                parent_target_context,
                fixed_system,
                automapped_mappings,
            )

        if fixed_codings:
            source_el = resolved_source
            system_uri = (
                (
                    fixed_system
                    or self._system_from_options(options)
                    or self._system_from_valueset(field)
                )
                if has_source
                else None
            )
            return self._create_fixed_codings_cc_rule(
                field,
                field_name,
                fixed_codings,
                parent_source_context,
                parent_target_context,
                source_element=source_el,
                source_system=system_uri,
            )

        if self._options_are_filter_based(
            options
        ) or self._bound_valueset_is_intensional(field):
            system_uri = (
                fixed_system
                or self._system_from_options(options)
                or self._system_from_valueset(field)
            )
            logger.info(
                f"Intensional/filter-based VS for {field_name} (system: {system_uri}) — generating direct copy rule."
            )
            return self._create_direct_copy_rule(
                field,
                field_name,
                parent_source_context,
                parent_target_context,
                system_uri,
                automapped_mappings,
            )

        # Enumerable VS: generate ConceptMap and use $translate.
        cm_url = "TODO-resolveConceptMap"
        if options:
            generated_url = self._generate_concept_map(
                field, options, field_name, automapped_mappings=automapped_mappings
            )
            if generated_url:
                cm_url = generated_url

        return self._create_translate_rule(
            field,
            field_name,
            parent_source_context,
            parent_target_context,
            cm_url,
            automapped_mappings,
        )

    def _options_are_filter_based(self, options: list) -> bool:
        """Return True when every option is a filter/open-system sentinel (not an enumerable code list)."""
        return bool(options) and all(
            opt.get("code", "").startswith("FILTER:") or opt.get("code") == "FROM_CS"
            for opt in options
        )

    def _system_from_options(self, options: list) -> Optional[str]:
        """Extract the first system URI found in the options list."""
        for opt in options:
            sys_uri = opt.get("system")
            if sys_uri:
                return sys_uri
        return None

    def _resolve_valueset_compose_includes(self, field):
        """Resolve a field's bound value set and return its compose.include list (or [])."""
        vs_url = field.get("valueSetUrl")
        if not vs_url:
            return []
        try:
            vs = resolve_url(vs_url, self.app_state)
        except Exception as e:
            logger.warning(
                "Could not resolve bound ValueSet %s (%s: %s) — no codes for this field.",
                vs_url,
                type(e).__name__,
                e,
            )
            return []
        if not vs:
            return []
        compose = _attr(vs, "compose")
        if not compose:
            return []
        return _attr(compose, "include") or []

    def _bound_valueset_is_intensional(self, field) -> bool:
        """check if the field's bound value set is intensional (any compose include carries a filter) """
        for inc in self._resolve_valueset_compose_includes(field):
            if _attr(inc, "filter"):
                return True
        return False

    def _system_from_valueset(self, field) -> Optional[str]:
        """First code-system URI declared by the field's bound value set compose"""
        for inc in self._resolve_valueset_compose_includes(field):
            sys_uri = _attr(inc, "system")
            if sys_uri:
                return sys_uri
        return None

    @staticmethod
    def _fixed_system_from_field(field):
        """Return the system URI a profile fixes on the fields .coding.system leaf
        (fixedUri/patternUri), or None"""

        for child in field.get("children") or []:
            if not isinstance(child, dict):
                continue
            if child.get("path", "").split(".")[-1] != "coding":
                continue
            for leaf in child.get("children") or []:
                if not isinstance(leaf, dict):
                    continue
                if leaf.get("path", "").split(".")[-1] != "system":
                    continue
                fv = leaf.get("fixed_value")
                if isinstance(fv, str) and fv:
                    return fv
        return None

    @staticmethod
    def _fixed_coding_from_leaves(field):
        """Build a fixed Coding dict {system, code, display} from a plain CodeableConcept's
        coding.system/.code/.display fixed leaves"""
        for child in field.get("children") or []:
            if not isinstance(child, dict):
                continue
            if child.get("path", "").split(".")[-1] != "coding":
                continue
            coding = {}
            for leaf in child.get("children") or []:
                if not isinstance(leaf, dict):
                    continue
                seg = leaf.get("path", "").split(".")[-1]
                fv = leaf.get("fixed_value")
                if isinstance(fv, str) and fv and seg in ("system", "code", "display"):
                    coding[seg] = fv
            if coding.get("system") and coding.get("code"):
                return coding
        return None

    @staticmethod
    def _fixed_coding_slices(field):
        """Return the fixed-pattern Coding dicts from a CodeableConcept field's sliced"""
        codings = []
        for child in field.get("children") or []:
            if not isinstance(child, dict):
                continue
            if child.get("path", "").split(".")[-1] != "coding":
                continue
            for sl in child.get("slices") or []:
                if not isinstance(sl, dict):
                    continue
                fv = sl.get("fixed_value")
                if hasattr(fv, "model_dump"):
                    try:
                        fv = fv.model_dump(exclude_none=True)
                    except Exception as e:
                        logger.warning(
                            "Dropping unreadable fixed coding on slice %s (%s: %s)",
                            sl.get("path"),
                            type(e).__name__,
                            e,
                        )
                        fv = None
                if isinstance(fv, dict) and fv.get("code"):
                    codings.append(fv)
        return codings

    @staticmethod
    def _source_coding_redundant(fixed_codings, source_system) -> bool:
        """check a source-derived coding would duplicate a fixed coding slice"""
        for fc in fixed_codings or []:
            if not isinstance(fc, dict):
                continue
            if fc.get("code") and (
                source_system is None or fc.get("system") == source_system
            ):
                return True
        return False

    def _create_fixed_codings_cc_rule(
        self,
        field,
        field_name,
        fixed_codings,
        parent_source_context,
        parent_target_context,
        source_element=None,
        source_system=None,
    ):
        """create a create rule for CodeableConcepts + fixed Coding per coding-slice"""
        nm = clean_field_name(field_name)
        cc_var = f"tgt-{nm}-cc"
        rule = StructureMapGroupRule.model_construct(name=f"map-{nm}-fixed-codings")
        rule.documentation = f"Fixed coding slices for {field.get('path','')}"
        rule.source = [
            StructureMapGroupRuleSource.model_construct(context=parent_source_context)
        ]
        tgt = StructureMapGroupRuleTarget.model_construct(
            context=parent_target_context, element=field_name, variable=cc_var
        )
        tgt.transform = "create"
        tgt.parameter = [
            StructureMapGroupRuleTargetParameter.model_construct(
                valueString="CodeableConcept"
            )
        ]
        if (field.get("cardinality") or {}).get("max") == "*":
            tgt.listMode = ["share"]
        rule.target = [tgt]

        nested = []
        for i, fc in enumerate(fixed_codings):
            cvar = f"{cc_var}-coding{i}"
            cr = StructureMapGroupRule.model_construct(name=f"add-{nm}-coding{i}")
            cr.source = [
                StructureMapGroupRuleSource.model_construct(
                    context=parent_source_context
                )
            ]
            ct = StructureMapGroupRuleTarget.model_construct(
                context=cc_var, element="coding", variable=cvar, transform="create"
            )
            ct.parameter = [
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString="Coding"
                )
            ]
            cr.target = [ct]
            cr.rule = self._fixed_pattern_rules(
                cvar, fc, parent_source_context, f"{nm}-coding{i}"
            )
            nested.append(cr)

        if source_element and not self._source_coding_redundant(
            fixed_codings, source_system
        ):
            svar = f"{cc_var}-codingsrc"
            scr = StructureMapGroupRule.model_construct(name=f"add-{nm}-coding-source")
            scr.source = [
                StructureMapGroupRuleSource.model_construct(
                    context=parent_source_context
                )
            ]
            sct = StructureMapGroupRuleTarget.model_construct(
                context=cc_var, element="coding", variable=svar, transform="create"
            )
            sct.parameter = [
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString="Coding"
                )
            ]
            scr.target = [sct]
            sub = []
            if source_system:
                sysr = StructureMapGroupRule.model_construct(
                    name=f"set-{nm}-codingsrc-system"
                )
                sysr.source = [
                    StructureMapGroupRuleSource.model_construct(
                        context=parent_source_context
                    )
                ]
                syst = StructureMapGroupRuleTarget.model_construct(
                    context=svar, element="system", transform="copy"
                )
                syst.parameter = [
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueString=source_system
                    )
                ]
                sysr.target = [syst]
                sub.append(sysr)
            codr = StructureMapGroupRule.model_construct(
                name=f"set-{nm}-codingsrc-code"
            )
            codr.source = [
                StructureMapGroupRuleSource.model_construct(
                    context=parent_source_context,
                    element=source_element,
                    variable="src-codingval",
                )
            ]
            codt = StructureMapGroupRuleTarget.model_construct(
                context=svar, element="code", transform="copy"
            )
            codt.parameter = [
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueId="src-codingval"
                )
            ]
            codr.target = [codt]
            sub.append(codr)
            scr.rule = sub
            nested.append(scr)

        rule.rule = nested
        return rule

    @staticmethod
    def _infer_pattern_type(value: dict) -> Optional[str]:
        """best-effort FHIR type for a nested fixed-pattern dict (so it can be create()d)"""
        if "coding" in value:
            return "CodeableConcept"
        if "system" in value and "code" in value:
            return "Coding"
        return None

    def _fixed_pattern_rules(self, parent_var, fixed, src_ctx, name_prefix):
        """create copy/create rules ppinning slice fixed-pattern values recursively"""
        rules = []
        if not isinstance(fixed, dict):
            return rules
        for key, val in fixed.items():
            if (
                not key
                or str(key).startswith("_")
                or str(key).endswith("__ext")
                or key == "fhir_comments"
            ):
                continue
            if isinstance(val, (bool, str, int, float)):
                r = StructureMapGroupRule.model_construct(
                    name=f"set-{name_prefix}-{clean_field_name(key)}"
                )
                r.source = [
                    StructureMapGroupRuleSource.model_construct(context=src_ctx)
                ]
                t = StructureMapGroupRuleTarget.model_construct(
                    context=parent_var, element=key, transform="copy"
                )
                t.parameter = [
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueString=fixed_scalar_literal(val)
                    )
                ]
                r.target = [t]
                rules.append(r)
            elif isinstance(val, list):
                for item in val:
                    if not isinstance(item, dict):
                        continue
                    cvar = f"{parent_var}-{clean_field_name(key)}"
                    elem_type = (
                        "Coding" if key == "coding" else self._infer_pattern_type(item)
                    )
                    if not elem_type:
                        continue
                    cr = StructureMapGroupRule.model_construct(
                        name=f"add-{name_prefix}-{clean_field_name(key)}"
                    )
                    cr.source = [
                        StructureMapGroupRuleSource.model_construct(context=src_ctx)
                    ]
                    ct = StructureMapGroupRuleTarget.model_construct(
                        context=parent_var,
                        element=key,
                        variable=cvar,
                        transform="create",
                    )
                    ct.parameter = [
                        StructureMapGroupRuleTargetParameter.model_construct(
                            valueString=elem_type
                        )
                    ]
                    cr.target = [ct]
                    cr.rule = self._fixed_pattern_rules(
                        cvar, item, src_ctx, f"{name_prefix}-{clean_field_name(key)}"
                    )
                    rules.append(cr)
            elif isinstance(val, dict):
                elem_type = self._infer_pattern_type(val)
                if not elem_type:
                    logger.info(
                        "Fixed nested pattern '%s' skipped (indeterminate type).", key
                    )
                    continue
                cvar = f"{parent_var}-{clean_field_name(key)}"
                cr = StructureMapGroupRule.model_construct(
                    name=f"set-{name_prefix}-{clean_field_name(key)}"
                )
                cr.source = [
                    StructureMapGroupRuleSource.model_construct(context=src_ctx)
                ]
                ct = StructureMapGroupRuleTarget.model_construct(
                    context=parent_var, element=key, variable=cvar, transform="create"
                )
                ct.parameter = [
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueString=elem_type
                    )
                ]
                cr.target = [ct]
                cr.rule = self._fixed_pattern_rules(
                    cvar, val, src_ctx, f"{name_prefix}-{clean_field_name(key)}"
                )
                rules.append(cr)
        return rules

    def _create_direct_copy_rule(
        self,
        field,
        field_name,
        parent_source_context,
        parent_target_context,
        system_uri,
        automapped_mappings,
    ):
        """direct copy rule for filter-based or open-system ValueSets"""
        field_type = self._canonical_coded_type(field)

        rule = StructureMapGroupRule.model_construct()
        rule.name = f"map-{clean_field_name(field_name)}-copy"
        rule.documentation = (
            f"Direct copy for {field['path']} (filter-based/open VS"
            + (f", system: {system_uri}" if system_uri else "")
            + ")"
        )

        resolved = self._resolve_coded_source(field, automapped_mappings)
        source_element = resolved or f"TODO-SOURCE-FOR-{field_name.upper()}"

        source = StructureMapGroupRuleSource.model_construct()
        source.context = parent_source_context
        source.element = source_element
        source.variable = "src"
        rule.source = [source]

        target = StructureMapGroupRuleTarget.model_construct()
        target.context = parent_target_context

        if field_type == "code":
            target.element = field_name
            target.transform = "copy"
            target.parameter = [
                StructureMapGroupRuleTargetParameter.model_construct(valueId="src")
            ]
            rule.target = [target]

        elif field_type in ("Coding", "CodeableConcept"):
            rule.rule = self._coding_structure_rules(
                field_type=field_type,
                outer_target=target,
                outer_element=field_name,
                inner_name_base=clean_field_name(field_name),
                system_name_base=clean_field_name(field_name),
                system_uri=system_uri,
                code_transform="copy",
                code_params=[
                    StructureMapGroupRuleTargetParameter.model_construct(valueId="src")
                ],
                guard_empty=False,
            )
            rule.target = [target]

        return rule

    def _existing_concept_map_is_authored(self, cm_id) -> bool:
        """check if the on-disk ConceptMap ``cm_id`` carries non-identity arrows.

        """
        try:
            existing = self.app_state.dataIO.load_project_file(
                self.app_state.dataIO.ProjectFolders.CONCEPT_MAPS, f"{cm_id}.json"
            )
        except Exception as e:
            logger.warning(
                "Could not read existing ConceptMap %s (%s: %s) — treating as unauthored; "
                "regeneration may overwrite authored arrows.",
                cm_id,
                type(e).__name__,
                e,
            )
            return False
        if not isinstance(existing, dict):
            return False
        for group in existing.get("group", []) or []:
            if (
                group.get("source")
                and group.get("target")
                and group["source"] != group["target"]
            ):
                return True
            for el in group.get("element", []) or []:
                code = el.get("code")
                for tgt in el.get("target", []) or []:
                    if tgt.get("code") and tgt["code"] != code:
                        return True
        return False

    def _generate_concept_map(
        self, field, options, field_name, automapped_mappings=None
    ):
        """Helper to generate and store a ConceptMap for a field"""
        import hashlib

        # Create deterministic ID based on value set URL or options to allow reuse
        value_set_url = field.get("valueSetUrl", "")
        if value_set_url:
            hash_input = value_set_url.encode("utf-8")
        else:
            options_str = str([{k: v for k, v in opt.items()} for opt in options])
            hash_input = options_str.encode("utf-8")

        url_hash = hashlib.md5(hash_input).hexdigest()[:8]

        cm_id = f"cm-{clean_field_name(field_name)}-{url_hash}"
        cm_url = f"{self.map_url}/ConceptMap/{cm_id}"

        # Check if ConceptMap already exists and we are not overwriting
        file_exists = self.app_state.dataIO.project_file_exists(
            self.app_state.dataIO.ProjectFolders.CONCEPT_MAPS, f"{cm_id}.json"
        )

        if file_exists and self._existing_concept_map_is_authored(cm_id):
            logger.info(
                f"ConceptMap {cm_id} has authored (non-identity) arrows — preserving."
            )
            return cm_url

        if file_exists and not self.overwrite:
            logger.info(f"ConceptMap {cm_id} already exists for {field_name}, reusing.")
            return cm_url

        cm = ConceptMap.model_construct(
            id=cm_id,
            url=cm_url,
            name=f"ConceptMap_{field_name}",
            title=f"Automatically generated ConceptMap for {field_name}",
            status="draft",
            sourceUri="SourceData",
            targetUri=field.get("valueSetUrl", "TargetValueSet"),
        )

        # Group by system
        groups = {}
        for opt in options:
            sys = opt.get("system", "http://example.org/default-system")
            if sys not in groups:
                groups[sys] = ConceptMapGroup.model_construct(
                    source=sys, target=sys, element=[]
                )

            code = opt.get("code")
            if code and not code.startswith("FILTER:"):
                groups[sys].element.append(
                    ConceptMapGroupElement.model_construct(
                        code=code,
                        target=[
                            ConceptMapGroupElementTarget.model_construct(
                                code=code, equivalence="equivalent"
                            )
                        ],
                    )
                )

        cm.group = list(groups.values())

        # add plugin hook info
        cm_dict = cm.model_dump() if cm.group else None
        source_field_name = field_name  # fallback: FHIR field name
        if automapped_mappings:
            fhir_path = field.get("path", "")
            source_id = automapped_mappings.get(fhir_path, "")
            if source_id and "." in source_id:
                source_field_name = source_id.split(".")[-1]

        for plugin in self.plugins:
            try:
                field_meta = plugin.get_field_metadata(source_field_name)
                if field_meta:
                    updated = plugin.generate_concept_map(
                        source_field_name, field_meta, cm_dict
                    )
                    if updated:
                        cm_dict = updated
            except Exception as e:
                logger.warning(
                    "Plugin '%s' generate_concept_map failed for '%s': %s",
                    plugin.plugin_id,
                    source_field_name,
                    e,
                )

        # store it (use plugin-enriched dict if available, otherwise the original)
        if cm_dict:

            cm_dict["url"] = cm_url
            cm_dict["id"] = cm_id
            store_content = json.dumps(cm_dict, indent=2, ensure_ascii=False)
            logger.info(f"Generated ConceptMap {cm_id} for {field_name} options.")
            self.app_state.dataIO.store_project_file(
                self.app_state.dataIO.ProjectFolders.CONCEPT_MAPS,
                f"{cm_id}.json",
                store_content,
                mode="STR",
                overwrite=self.overwrite,
            )
            return cm_url
        elif len(cm.group) > 0:
            logger.info(f"Generated ConceptMap {cm_id} for {field_name} options.")
            self.app_state.dataIO.store_project_file(
                self.app_state.dataIO.ProjectFolders.CONCEPT_MAPS,
                f"{cm_id}.json",
                cm.model_dump_json(indent=2),
                mode="STR",
                overwrite=self.overwrite,
            )
            return cm_url
        return None

    def _create_fixed_value_rule(
        self,
        field_type,
        fixed_value,
        field_name,
        parent_target_context,
        parent_source_context="source",
    ):
        """create a rule for a fixed value"""
        codes = type_codes(field_type)
        if len(codes) == 1:
            field_type = codes[0]

        if "[x]" in field_name and isinstance(field_type, str):
            field_name = field_name.replace("[x]", fhir_type_suffix(field_type))

        if hasattr(fixed_value, "model_dump"):
            try:
                fixed_value = fixed_value.model_dump(exclude_none=True)
            except Exception as e:
                logger.warning(
                    "model_dump failed for fixed value of %s (%s: %s) — using the raw model.",
                    field_name,
                    type(e).__name__,
                    e,
                )

        rule = StructureMapGroupRule.model_construct()
        rule.name = f"set-fixed-{clean_field_name(field_name)}"
        rule.documentation = f"Sets fixed value for {field_name}"

        # Fixed-value rules fire unconditionally once — use a wildcard source on the parent context.
        wildcard_src = StructureMapGroupRuleSource.model_construct()
        wildcard_src.context = parent_source_context
        rule.source = [wildcard_src]

        target = StructureMapGroupRuleTarget.model_construct()
        target.context = parent_target_context
        target.element = field_name

        def _string_param(val):
            return StructureMapGroupRuleTargetParameter.model_construct(
                valueString=fixed_scalar_literal(val)
            )

        if field_type == "code":
            target.transform = "copy"
            target.parameter = [_string_param(fixed_value)]
            rule.target = [target]
            return rule

        if field_type in ("Coding", "CodeableConcept"):
            nm = clean_field_name(field_name)
            var = f"tgt-fixed-{nm}"
            target.variable = var
            target.transform = "create"
            target.parameter = [
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString=field_type
                )
            ]
            rule.target = [target]
            pattern = fixed_value if isinstance(fixed_value, dict) else {}
            rule.rule = self._fixed_pattern_rules(
                var, pattern, parent_source_context, f"fixed-{nm}"
            )
            return rule

        if isinstance(fixed_value, (dict, list)):
            logger.error(
                "Cannot emit fixed complex value for %s with type %r; omitting rule.",
                field_name,
                field_type,
            )
            return None

        target.transform = "copy"
        target.parameter = [_string_param(fixed_value)]
        rule.target = [target]
        return rule
