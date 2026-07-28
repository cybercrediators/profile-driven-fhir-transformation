from fhir.resources.R4B.structuremap import (
    StructureMapGroupRule,
    StructureMapGroupRuleTarget,
    StructureMapGroupRuleTargetParameter,
    StructureMapGroupRuleSource,
)
from mapping.fml_creator.fml_helper import (
    is_primitive_type,
    get_transform_for_type,
    clean_field_name,
    fixed_scalar_literal,
    type_codes,
    fhir_type_suffix,
)
from parser.resource_parser.fhir_type_introspection import get_complex_type_fields
import logging

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
    def _create_choice_populate_rule(
        self,
        base_name,
        choice_type,
        sub_field_maps,
        parent_source_context,
        parent_target_context,
        cardinality,
    ):
        """emit a single rule that creates and populates a choice-typed element (value[x] or similar)"""
        bn = clean_field_name(base_name)
        ct = clean_field_name(choice_type)
        rule = StructureMapGroupRule.model_construct(name=f"map-{bn}-{ct}")

        if is_primitive_type(choice_type):
            suffix = self._choice_suffix(choice_type)
            src = StructureMapGroupRuleSource.model_construct(
                context=parent_source_context
            )
            tgt = StructureMapGroupRuleTarget.model_construct()
            tgt.context = parent_target_context
            tgt.element = f"{base_name}{suffix}" if suffix else base_name
            if sub_field_maps:
                src.element = sub_field_maps[0][1]
                src.variable = "src-choiceval"
                info = get_transform_for_type(choice_type, "src-choiceval")
                tgt.transform = info.get("transform")
                if info.get("parameters"):
                    tgt.parameter = info["parameters"]
            rule.source = [src]
            rule.target = [tgt]
            return rule

        # A concrete complex choice can also be copied as a whole.  This is distinct
        # from descendant mappings, where the datatype must be created and populated.
        if sub_field_maps and sub_field_maps[0][0] == "":
            src = StructureMapGroupRuleSource.model_construct(
                context=parent_source_context,
                element=sub_field_maps[0][1],
                variable="src-choiceval",
            )
            tgt = StructureMapGroupRuleTarget.model_construct(
                context=parent_target_context,
                element=f"{base_name}{self._choice_suffix(choice_type)}",
                transform="copy",
                parameter=[
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueId="src-choiceval"
                    )
                ],
            )
            rule.source = [src]
            rule.target = [tgt]
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
        rule.rule = nested
        return rule

    def _resolve_leaf_type(self, complex_type, sub, children=None):
        """Resolve a direct sub-field's type on a complex datatype: a profile-declared child
        element wins (its type may be constrained), then generic type introspection."""
        for c in children or []:
            if (c.get("path", "") or "").split(".")[-1] == sub:
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
        slice_path = slice_field.get("path", "")  # e.g. Patient.name:name
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
        _sub_paths = {s for s, _ in subs}
        subs = [
            (s, v)
            for (s, v) in subs
            if not any(o != s and o.startswith(s + ".") for o in _sub_paths)
        ]

        min_c = (slice_field.get("cardinality") or {}).get("min", 0) or 0
        if not subs and min_c == 0:
            return None

        nm = f"{clean_field_name(base_element)}-{clean_field_name(slice_name)}"
        rule = StructureMapGroupRule.model_construct(name=f"map-{nm}")
        rule.documentation = (
            f"Slice {slice_name} of {parent_field.get('path','')} (type {slice_type})"
        )
        src = StructureMapGroupRuleSource.model_construct(context=parent_source_context)
        rule.source = [src]

        if is_primitive_type(slice_type):
            tgt = StructureMapGroupRuleTarget.model_construct(
                context=parent_target_context, element=base_element
            )
            tgt.listMode = ["share"]
            if subs:
                src.element = subs[0][1]
                src.variable = "src-slice"
                info = get_transform_for_type(slice_type, "src-slice")
                tgt.transform = info.get("transform")
                if info.get("parameters"):
                    tgt.parameter = info["parameters"]
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
        tgt.listMode = ["share"]
        rule.target = [tgt]

        nested = []
        if isinstance(fixed, dict):
            nested += self._fixed_pattern_rules(var, fixed, parent_source_context, nm)
        base_depth = len((slice_field.get("path", "") or "").split("."))
        slice_children = slice_field.get("children") or []
        direct_children = [
            c
            for c in slice_children
            if len((c.get("path", "") or "").split(".")) == base_depth + 1
        ]
        mapped_roots = {
            sub.split(".", 1)[0].split(":", 1)[0].replace("[x]", "") for sub, _ in subs
        }
        slice_eid = slice_path if ":" in slice_path else f"{slice_path}:{slice_name}"
        nested += self._slice_child_fixed_rules(
            direct_children,
            var,
            parent_source_context,
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
                    parent_source_context,
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
                    parent_source_context,
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
                ref_rule = StructureMapGroupRule.model_construct(
                    name=(
                        f"TODO-resolve-reference-{clean_field_name(res_type)}"
                        f"-{clean_field_name(sub)}"
                    )
                )
                ref_rule.source = [
                    StructureMapGroupRuleSource.model_construct(
                        context=parent_source_context,
                        variable=f"src-{nm}-{clean_field_name(sub)}",
                    )
                ]
                ref_rule.target = [
                    StructureMapGroupRuleTarget.model_construct(element=sub)
                ]
                ref_rule.documentation = (
                    f"Reference<{res_type}.{_rel_base}.{sub}> → {_ref_target} "
                    f"— resolve via bundle assembler"
                )
                nested.append(ref_rule)
                continue
            sub_var = f"src-{nm}-{clean_field_name(sub)}"
            s = StructureMapGroupRuleSource.model_construct(
                context=parent_source_context, element=src_local, variable=sub_var
            )
            sr = StructureMapGroupRule.model_construct(
                name=f"set-{nm}-{clean_field_name(sub)}"
            )
            sr.source = [s]
            # Type-aware transform, like _create_choice_populate_rule: a numeric/date leaf
            # needs a cast, a bare copy crashes the engine on non-string sources.
            leaf_type = self._resolve_leaf_type(
                slice_type, sub, children=direct_children
            )
            info = get_transform_for_type(leaf_type or "string", sub_var)
            stt = StructureMapGroupRuleTarget.model_construct(
                context=var, element=sub, transform=info.get("transform", "copy")
            )
            stt.parameter = info.get("parameters") or [
                StructureMapGroupRuleTargetParameter.model_construct(valueId=sub_var)
            ]
            sr.target = [stt]
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
        for el in elements:
            eid = _attr(el, "id") or ""
            if eid != exact_id and not (
                eid.startswith(base_path) and eid.endswith(f".{sub}")
            ):
                continue
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
            leaf_target.parameter = [
                StructureMapGroupRuleTargetParameter.model_construct(
                    valueString=fixed_scalar_literal(value)
                )
            ]
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
                    cvar, fv, src_ctx, f"{nm}-{clean_field_name(elem)}"
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
                t.parameter = [
                    StructureMapGroupRuleTargetParameter.model_construct(
                        valueString=fixed_scalar_literal(fv)
                    )
                ]
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
            for c in children or []:
                last = (c.get("path", "") or "").split(".")[-1]
                if last == seg or last == f"{seg}[x]":
                    tc = type_codes(c.get("type"))
                    ty = tc[0] if tc else None
                    name = seg
                    if last.endswith("[x]") and ty:
                        name = f"{seg}{ty[:1].upper()}{ty[1:]}"  # value + Integer -> valueInteger
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
            sid = clean_field_name(seg)
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
