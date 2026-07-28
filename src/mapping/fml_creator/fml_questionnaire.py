import re
from typing import Dict, List, Optional
from parser.resource_parser.value_expander import expand_valueset
from mapping.fml_creator.fml_helper import local_element_name
from mapping.rule_ir import mapping_target_path
from data_handling.registry.registry_object import RegistryObject

import logging
logger = logging.getLogger(__name__)

from fhir.resources.R4B.structuremap import (
    StructureMapGroup,
    StructureMapGroupRule,
    StructureMapGroupRuleSource,
    StructureMapGroupRuleTarget,
    StructureMapGroupRuleTargetParameter,
)
from fhir.resources.R4B.questionnaire import Questionnaire, QuestionnaireItem


# Source field-id → local FHIRPath element name; shared with the rule factory.
_local = local_element_name


class QuestionnaireMapCreator:
    def __init__(self, registry_object: RegistryObject, factory=None):
        self.registry_object = registry_object
        self.questionnaire: Questionnaire = registry_object.data
        self.factory = factory

    def _build_linkid_source_index(self, mapping_table: dict) -> Dict[str, List[str]]:
        """Build {linkId: [source_field_ids]} from QuestionnaireResponse.item[linkId] entries."""
        result: Dict[str, List[str]] = {}
        if not mapping_table:
            return result
        pattern = re.compile(r"^QuestionnaireResponse\.item\[([^\]]+)\]")
        for source_field_id, target_value in mapping_table.items():
            target_path = mapping_target_path(target_value)
            if not target_path:
                continue
            m = pattern.match(target_path)
            if m:
                link_id = m.group(1)
                result.setdefault(link_id, []).append(source_field_id)
        return result

    def _get_status_source_field(self, mapping_table: dict) -> Optional[str]:
        """Return the source field ID mapped to QuestionnaireResponse.status."""
        if not mapping_table:
            return None
        for source_field_id, target_value in mapping_table.items():
            target_path = mapping_target_path(target_value)
            if target_path == "QuestionnaireResponse.status":
                return source_field_id
        return None

    def generate_group(
        self, source_alias: str, target_alias: str, mapping_table: dict = None
    ) -> StructureMapGroup:
        """Generate a StructureMapGroup for this Questionnaire."""
        group_name = self.questionnaire.name or "QuestionnaireMap"
        group_name = group_name.replace(" ", "_").replace("-", "_")

        group = StructureMapGroup.model_construct(
            name=group_name,
            typeMode="none",
            input=[
                {"name": source_alias, "type": "Source", "mode": "source"},
                {
                    "name": target_alias,
                    "type": "QuestionnaireResponse",
                    "mode": "target",
                },
            ],
            rule=[],
        )

        linkid_to_sources = self._build_linkid_source_index(mapping_table)
        status_source = self._get_status_source_field(mapping_table)

        if status_source:
            group.rule.append(
                self._create_status_rule(source_alias, target_alias, status_source)
            )

        group.rule.append(self._create_subject_placeholder_rule(source_alias))

        # link the QR to its Questionnaire — the canonical is known at
        # generation time, so this is a deterministic fixed value
        if self.questionnaire.url:
            group.rule.append(
                self._create_assignment_rule(
                    target_alias,
                    "questionnaire",
                    str(self.questionnaire.url),
                    src_context=source_alias,
                )
            )

        if self.questionnaire.item:
            self._process_items(
                self.questionnaire.item,
                group.rule,
                source_alias,
                target_alias,
                linkid_to_sources,
            )

        return group

    def _concept_map_for(
        self,
        field_name: str,
        source_field_id: str,
        target_path: str,
        value_set: str = "",
        options: list = None,
    ) -> Optional[str]:
        """check plugin for a ConceptMap URL for a source field to target path mapping."""
        if not self.factory:
            return None
        field = {"path": target_path, "valueSetUrl": value_set or ""}
        opts = list(options or [])
        if not opts and value_set:
            try:
                opts = expand_valueset(value_set, [], self.factory.app_state) or []
            except Exception as e:
                logger.warning(
                    "Could not expand value set %s for %s: %s", value_set, field_name, e
                )
        try:
            return self.factory._generate_concept_map(
                field,
                opts,
                field_name,
                automapped_mappings={target_path: source_field_id},
            )
        except Exception as e:
            logger.warning("ConceptMap generation failed for %s: %s", field_name, e)
            return None

    def _extract_options(self, item: QuestionnaireItem) -> List[dict]:
        """extract answerOption values from a QuestionnaireItem, expanding answerValueSet if needed"""
        opts: List[dict] = []
        for ao in item.answerOption or []:
            vc = getattr(ao, "valueCoding", None)
            if vc is not None:
                opts.append(
                    {
                        "element": "valueCoding",
                        "code": vc.code,
                        "system": vc.system,
                        "display": vc.display,
                    }
                )
                continue
            for attr in ("valueString", "valueInteger", "valueDate", "valueTime"):
                v = getattr(ao, attr, None)
                if v is not None:
                    opts.append({"element": attr, "value": v})
                    break
        if not opts and getattr(item, "answerValueSet", None) and self.factory:
            try:
                for o in (
                    expand_valueset(item.answerValueSet, [], self.factory.app_state)
                    or []
                ):
                    if (
                        o.get("code")
                        and not str(o.get("code")).startswith("FILTER:")
                        and o.get("code") != "FROM_CS"
                    ):
                        opts.append(
                            {
                                "element": "valueCoding",
                                "code": o.get("code"),
                                "system": o.get("system"),
                                "display": o.get("display"),
                            }
                        )
            except Exception as e:
                logger.warning(
                    "Could not expand answerValueSet %s: %s", item.answerValueSet, e
                )
        return opts

    def _create_subject_placeholder_rule(self, source_alias: str) -> StructureMapGroupRule:
        """QuestionnaireResponse.subject cannot be derived from the Questionnaire —
        emit the standard reference placeholder so the bundle assembler wires it to
        the in-bundle subject (``Questionnaire.subjectType`` when declared, else
        Patient). Same shape as the profile maps' TODO-resolve-reference rules."""
        subject_types = "|".join(
            str(t) for t in (self.questionnaire.subjectType or [])
        ) or "Patient"
        return StructureMapGroupRule.model_construct(
            name="TODO-resolve-reference-QuestionnaireResponse-subject",
            source=[
                StructureMapGroupRuleSource.model_construct(
                    context=source_alias, variable="src-subject"
                )
            ],
            target=None,
            documentation=(
                f"Reference<QuestionnaireResponse.subject> → {subject_types} "
                f"— resolve via bundle assembler"
            ),
        )

    def _create_status_rule(
        self, source_alias: str, target_alias: str, source_field_id: str
    ) -> StructureMapGroupRule:
        """create a rule to map the source field to QuestionnaireResponse.status, using translate() if a ConceptMap is available"""
        cm_url = self._concept_map_for(
            "status", source_field_id, "QuestionnaireResponse.status"
        )
        target = StructureMapGroupRuleTarget(context=target_alias, element="status")
        if cm_url:
            target.transform = "translate"
            target.parameter = [
                StructureMapGroupRuleTargetParameter(valueId="srcStatus"),
                StructureMapGroupRuleTargetParameter(valueString=cm_url),
                StructureMapGroupRuleTargetParameter(valueString="code"),
            ]
        else:
            target.transform = "copy"
            target.parameter = [
                StructureMapGroupRuleTargetParameter(valueString="TODO-map-status-code")
            ]
        return StructureMapGroupRule(
            name="set-status",
            source=[
                StructureMapGroupRuleSource(
                    context=source_alias,
                    element=_local(source_field_id),
                    variable="srcStatus",
                )
            ],
            target=[target],
        )

    def _process_items(
        self,
        items: List[QuestionnaireItem],
        rules: List[StructureMapGroupRule],
        source_context: str,
        target_context: str,
        linkid_to_sources: Dict[str, List[str]],
   ):
        for item in items:
            self._create_item_rule(
                item, rules, source_context, target_context, linkid_to_sources
            )

    def _create_item_rule(
        self,
        item: QuestionnaireItem,
        rules: List[StructureMapGroupRule],
        source_context: str,
        target_context: str,
        linkid_to_sources: Dict[str, List[str]],
    ):
        if item.type == "display":
            # Display items are read-only but may have answer-able children
            if item.item:
                self._process_items(
                    item.item, rules, source_context, target_context, linkid_to_sources
                )
            return

        link_id = item.linkId
        safe_link_id = link_id.replace(".", "-").replace("_", "-")
        rule_name = f"rule-{safe_link_id}"

        source_fields = linkid_to_sources.get(link_id, [])
        is_multi = len(source_fields) > 1
        is_group = item.type == "group"
        is_unmapped = not is_group and not source_fields

        if is_group or is_unmapped:
            outer_source = StructureMapGroupRuleSource(
                context=source_context, variable="srcVal"
            )
        else:
            primary_source_element = _local(source_fields[0])
            outer_source = StructureMapGroupRuleSource(
                context=source_context,
                element=primary_source_element,
                variable="srcVal",
            )
        outer_rule = StructureMapGroupRule(
            name=rule_name,
            source=[outer_source],
            target=[
                StructureMapGroupRuleTarget(
                    context=target_context, element="item", variable="tItem"
                )
            ],
            rule=[],
        )

        outer_rule.rule.append(
            self._create_assignment_rule(
                "tItem", "linkId", link_id, src_context="srcVal"
            )
        )
        if item.text:
            outer_rule.rule.append(
                self._create_assignment_rule(
                    "tItem", "text", item.text, src_context="srcVal"
                )
            )

        if is_group:
            if item.item:
                self._process_items(
                    item.item,
                    outer_rule.rule,
                    source_context,
                    "tItem",
                    linkid_to_sources,
                )
        elif is_multi:
            options = self._extract_options(item)
            for idx, sf in enumerate(source_fields):
                option = options[idx] if idx < len(options) else None
                outer_rule.rule.append(
                    self._create_checkbox_answer_rule(
                        sf, source_context, "tItem", item.type, idx + 1, option
                    )
                )
        elif is_unmapped:
            outer_rule.rule.append(
                self._create_placeholder_answer_rule(
                    rule_name, safe_link_id, item.type, source_context
                )
            )
        else:
            options = (
                self._extract_options(item)
                if item.type in ("choice", "open-choice")
                else []
            )
            opt_element = options[0]["element"] if options else None
            if opt_element == "valueCoding":
                coded_opts = [
                    {
                        "code": o.get("code"),
                        "system": o.get("system"),
                        "display": o.get("display"),
                    }
                    for o in options
                    if o.get("code")
                ]
                cm_url = (
                    self._concept_map_for(
                        f"item-{safe_link_id}",
                        source_fields[0],
                        f"QuestionnaireResponse.item[{link_id}]",
                        options=coded_opts,
                    )
                    if coded_opts
                    else None
                )
                system = next(
                    (o.get("system") for o in options if o.get("system")), None
                )
                answer_rule = self._build_coded_answer_rule(
                    rule_name, cm_url or "TODO-resolveConceptMap", system
                )
            elif opt_element == "valueString":
                answer_rule = self._build_copy_answer_rule(rule_name, "valueString")
            else:
                answer_rule = self._build_copy_answer_rule(
                    rule_name, self._get_value_type(item.type)
                )
            outer_rule.rule.append(answer_rule)

        rules.append(outer_rule)

    def _build_copy_answer_rule(
        self, rule_name: str, value_type: Optional[str]
    ) -> StructureMapGroupRule:
        """answer.value[x] = copy(source) — for string options and plain primitive items."""
        answer_rule = StructureMapGroupRule(
            name=f"answer-{rule_name}",
            source=[StructureMapGroupRuleSource(context="srcVal")],
            target=[
                StructureMapGroupRuleTarget(
                    context="tItem", element="answer", variable="tAns"
                )
            ],
            rule=[],
        )
        if value_type:
            answer_rule.rule.append(
                StructureMapGroupRule(
                    name="set-val",
                    source=[
                        StructureMapGroupRuleSource(context="srcVal", variable="srcV")
                    ],
                    target=[
                        StructureMapGroupRuleTarget(
                            context="tAns",
                            element=value_type,
                            transform="copy",
                            parameter=[
                                StructureMapGroupRuleTargetParameter(valueId="srcV")
                            ],
                        )
                    ],
                )
            )
        return answer_rule

    def _build_coded_answer_rule(
        self, rule_name: str, cm_url: str, system: Optional[str]
    ) -> StructureMapGroupRule:
        """answer.valueCoding = translate(source, ConceptMap) for choice/open-choice items."""
        code_rule = StructureMapGroupRule(
            name="set-code",
            source=[StructureMapGroupRuleSource(context="srcVal", variable="srcV")],
            target=[
                StructureMapGroupRuleTarget(
                    context="tCoding",
                    element="code",
                    transform="translate",
                    parameter=[
                        StructureMapGroupRuleTargetParameter(valueId="srcV"),
                        StructureMapGroupRuleTargetParameter(valueString=cm_url),
                        StructureMapGroupRuleTargetParameter(valueString="code"),
                    ],
                )
            ],
        )
        coding_rule = StructureMapGroupRule(
            name="set-coding",
            source=[StructureMapGroupRuleSource(context="srcVal")],
            target=[
                StructureMapGroupRuleTarget(
                    context="tAns",
                    element="value",
                    variable="tCoding",
                    transform="create",
                    parameter=[
                        StructureMapGroupRuleTargetParameter(valueString="Coding")
                    ],
                )
            ],
            rule=[code_rule],
        )
        if system:
            coding_rule.rule.append(
                StructureMapGroupRule(
                    name="set-system",
                    source=[StructureMapGroupRuleSource(context="srcVal")],
                    target=[
                        StructureMapGroupRuleTarget(
                            context="tCoding",
                            element="system",
                            transform="copy",
                            parameter=[
                                StructureMapGroupRuleTargetParameter(valueString=system)
                            ],
                        )
                    ],
                )
            )
        return StructureMapGroupRule(
            name=f"answer-{rule_name}",
            source=[StructureMapGroupRuleSource(context="srcVal")],
            target=[
                StructureMapGroupRuleTarget(
                    context="tItem", element="answer", variable="tAns"
                )
            ],
            rule=[coding_rule],
        )

    def _create_placeholder_answer_rule(
        self, rule_name: str, safe_link_id: str, item_type: str, source_context: str
    ) -> StructureMapGroupRule:
        """emit a placeholder answer rule for an unmapped leaf item, with a TODO sentinel source element"""
        value_type = self._get_value_type(item_type)
        src_var = "srcV"
        return StructureMapGroupRule(
            name=f"answer-{rule_name}",
            source=[
                StructureMapGroupRuleSource(
                    context=source_context,
                    element=f"TODO-map-{safe_link_id}",
                    variable="srcVal",
                )
            ],
            target=[
                StructureMapGroupRuleTarget(
                    context="tItem", element="answer", variable="tAns"
                )
            ],
            rule=[
                StructureMapGroupRule(
                    name="set-val",
                    source=[
                        StructureMapGroupRuleSource(context="srcVal", variable=src_var)
                    ],
                    target=[
                        StructureMapGroupRuleTarget(
                            context="tAns",
                            element=value_type or "valueString",
                            transform="copy",
                            parameter=[
                                StructureMapGroupRuleTargetParameter(valueId=src_var)
                            ],
                        )
                    ],
                )
            ],
        )

    def _create_checkbox_answer_rule(
        self,
        source_field_id: str,
        source_context: str,
        item_var: str,
        item_type: str,
        option_index: int,
        option: Optional[dict] = None,
    ) -> StructureMapGroupRule:
        """emit a nested rule for a checkbox multi-select source field, keyed to the source flag element"""
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "-", source_field_id.split(".")[-1]).strip(
            "-"
        )
        return StructureMapGroupRule(
            name=f"cb-{safe_name}",
            source=[
                StructureMapGroupRuleSource(
                    context=source_context,
                    element=_local(source_field_id),
                    condition="$this = '1'",
                    variable="srcCb",
                )
            ],
            target=[
                StructureMapGroupRuleTarget(
                    context=item_var, element="answer", variable="tAns"
                )
            ],
            rule=self._fixed_option_value_rules(option, item_type, option_index),
        )

    def _fixed_option_value_rules(
        self, option: Optional[dict], item_type: str, option_index: int
    ) -> List[StructureMapGroupRule]:
        """Build the rule(s) that set a checkbox answer to its (fixed) answerOption value."""
        if option and option.get("element") == "valueCoding" and option.get("code"):
            nested = [
                StructureMapGroupRule(
                    name=f"set-code-{option_index}",
                    source=[StructureMapGroupRuleSource(context="srcCb")],
                    target=[
                        StructureMapGroupRuleTarget(
                            context="tCoding",
                            element="code",
                            transform="copy",
                            parameter=[
                                StructureMapGroupRuleTargetParameter(
                                    valueString=option["code"]
                                )
                            ],
                        )
                    ],
                )
            ]
            if option.get("system"):
                nested.append(
                    StructureMapGroupRule(
                        name=f"set-system-{option_index}",
                        source=[StructureMapGroupRuleSource(context="srcCb")],
                        target=[
                            StructureMapGroupRuleTarget(
                                context="tCoding",
                                element="system",
                                transform="copy",
                                parameter=[
                                    StructureMapGroupRuleTargetParameter(
                                        valueString=option["system"]
                                    )
                                ],
                            )
                        ],
                    )
                )
            return [
                StructureMapGroupRule(
                    name=f"set-coding-{option_index}",
                    source=[StructureMapGroupRuleSource(context="srcCb")],
                    target=[
                        StructureMapGroupRuleTarget(
                            context="tAns",
                            element="value",
                            variable="tCoding",
                            transform="create",
                            parameter=[
                                StructureMapGroupRuleTargetParameter(
                                    valueString="Coding"
                                )
                            ],
                        )
                    ],
                    rule=nested,
                )
            ]
        if option and option.get("element") in (
            "valueString",
            "valueInteger",
            "valueDate",
            "valueTime",
        ):
            return [
                StructureMapGroupRule(
                    name=f"set-val-{option_index}",
                    source=[StructureMapGroupRuleSource(context="srcCb")],
                    target=[
                        StructureMapGroupRuleTarget(
                            context="tAns",
                            element=option["element"],
                            transform="copy",
                            parameter=[
                                StructureMapGroupRuleTargetParameter(
                                    valueString=str(option["value"])
                                )
                            ],
                        )
                    ],
                )
            ]
        # Fallback: option not resolvable — typed, non-gating placeholder.
        return [
            StructureMapGroupRule(
                name=f"set-val-{option_index}",
                source=[StructureMapGroupRuleSource(context="srcCb")],
                target=[
                    StructureMapGroupRuleTarget(
                        context="tAns",
                        element=self._get_value_type(item_type) or "valueString",
                        transform="copy",
                        parameter=[
                            StructureMapGroupRuleTargetParameter(
                                valueString=f"TODO-set-option-{option_index}"
                            )
                        ],
                    )
                ],
            )
        ]

    def _create_assignment_rule(
        self, context: str, element: str, value: str, src_context: str = "srcVal"
    ) -> StructureMapGroupRule:
        return StructureMapGroupRule(
            name=f"assign-{element}",
            source=[StructureMapGroupRuleSource(context=src_context)],
            target=[
                StructureMapGroupRuleTarget(
                    context=context,
                    element=element,
                    transform="copy",
                    parameter=[StructureMapGroupRuleTargetParameter(valueString=value)],
                )
            ],
        )

    def _get_value_type(self, item_type: str) -> Optional[str]:
        mapping = {
            "boolean": "valueBoolean",
            "decimal": "valueDecimal",
            "integer": "valueInteger",
            "date": "valueDate",
            "dateTime": "valueDateTime",
            "time": "valueTime",
            "string": "valueString",
            "text": "valueString",
            "url": "valueUri",
            "choice": "valueCoding",
            "open-choice": "valueCoding",
            "attachment": "valueAttachment",
            "reference": "valueReference",
            "quantity": "valueQuantity",
        }
        return mapping.get(item_type, "valueString")
