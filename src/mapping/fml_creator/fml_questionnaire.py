import logging
from decimal import Decimal
import re
from typing import Any, Dict, List, Optional

from fhir.resources.R4B.questionnaire import Questionnaire, QuestionnaireItem
from fhir.resources.R4B.structuremap import (
    StructureMapGroup,
    StructureMapGroupRule,
    StructureMapGroupRuleSource,
    StructureMapGroupRuleTarget,
    StructureMapGroupRuleTargetParameter,
)

from data_handling.registry.registry_object import RegistryObject
from mapping.fml_creator.fml_helper import local_element_name
from mapping.rule_ir import mapping_target_path
from parser.resource_parser.value_expander import expand_valueset

logger = logging.getLogger(__name__)


# Source field-id → local FHIRPath element name; shared with the rule factory.
_local = local_element_name

_SDC_CALCULATED_EXPRESSION = "sdc-questionnaire-calculatedExpression"
_SDC_INITIAL_EXPRESSION = "sdc-questionnaire-initialExpression"
_SDC_ENABLE_WHEN_EXPRESSION = "sdc-questionnaire-enableWhenExpression"
_SDC_ANSWER_EXPRESSION = "sdc-questionnaire-answerExpression"
_QUESTIONNAIRE_MIN_OCCURS = "questionnaire-minOccurs"
_QUESTIONNAIRE_MAX_OCCURS = "questionnaire-maxOccurs"
_MIN_VALUE = "minValue"
_MAX_VALUE = "maxValue"
_FHIRPATH_LANGUAGES = {
    "text/fhirpath",
    "application/fhirpath",
    "text/x-fhirpath",
}


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
        self._source_root_alias = source_alias
        self._target_root_alias = target_alias
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

        group.rule.append(
            self._create_status_rule(
                source_alias,
                target_alias,
                status_source or "TODO_MAP_STATUS",
            )
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

    @staticmethod
    def _extension_matches(extension, suffix: str) -> bool:
        url = str(getattr(extension, "url", "") or "").rstrip("/")
        return url.rsplit("/", 1)[-1] == suffix

    @staticmethod
    def _extension_value(extension) -> Optional[tuple[str, Any]]:
        if hasattr(extension, "model_dump"):
            raw = extension.model_dump(exclude_none=True, by_alias=True)
        else:
            raw = extension if isinstance(extension, dict) else {}
        for key, value in raw.items():
            if key.startswith("value") and value is not None:
                return key, value
        return None

    def _extension(self, item: QuestionnaireItem, suffix: str):
        return next(
            (
                extension
                for extension in (item.extension or [])
                if self._extension_matches(extension, suffix)
            ),
            None,
        )

    def _record(self, code: str, message: str, **details) -> None:
        recorder = getattr(self.factory, "record_diagnostic", None)
        if callable(recorder):
            recorder(code, message, **details)

    def _expression(
        self, item: QuestionnaireItem, suffix: str
    ) -> Optional[str]:
        extension = self._extension(item, suffix)
        if extension is None:
            return None
        choice = self._extension_value(extension)
        value = choice[1] if choice and choice[0] == "valueExpression" else None
        if not isinstance(value, dict):
            self._record(
                "questionnaire-expression-invalid",
                "Questionnaire expression extension requires valueExpression.",
                link_id=str(item.linkId),
                extension=suffix,
            )
            return None
        language = str(value.get("language") or "")
        expression = value.get("expression")
        if language not in _FHIRPATH_LANGUAGES or not isinstance(expression, str):
            self._record(
                "questionnaire-expression-language-unsupported",
                "Only explicitly authored FHIRPath Questionnaire expressions "
                "can be emitted by the StructureMap generator.",
                link_id=str(item.linkId),
                extension=suffix,
                language=language,
            )
            return None
        unsupported_variables = sorted(
            set(re.findall(r"%([A-Za-z][A-Za-z0-9_-]*)", expression))
            - {"resource"}
        )
        if unsupported_variables:
            self._record(
                "questionnaire-expression-context-unavailable",
                "Questionnaire expression refers to context variables that are "
                "not available to the generated one-shot transform.",
                link_id=str(item.linkId),
                extension=suffix,
                variables=unsupported_variables,
            )
            return None
        return re.sub(r"%resource\b", "$this", expression)

    def _extension_scalar(
        self, item: QuestionnaireItem, suffix: str
    ) -> Optional[Any]:
        extension = self._extension(item, suffix)
        choice = self._extension_value(extension) if extension is not None else None
        return choice[1] if choice else None

    def _record_answer_expression_boundary(self, item: QuestionnaireItem) -> None:
        if self._extension(item, _SDC_ANSWER_EXPRESSION) is None:
            return
        expression = self._expression(item, _SDC_ANSWER_EXPRESSION)
        if expression is not None:
            self._record(
                "questionnaire-answer-expression-candidate-only",
                "Questionnaire answerExpression defines candidate answers; it "
                "does not select a QuestionnaireResponse answer. The generated "
                "map therefore retains the authored source mapping and leaves "
                "candidate-set membership to response validation.",
                link_id=str(item.linkId),
            )

    def _value_check(self, item: QuestionnaireItem) -> Optional[str]:
        checks = []
        if (
            getattr(item, "maxLength", None) is not None
            and item.type in {"string", "text", "url"}
        ):
            checks.append(
                f"$this.toString().length() <= {int(item.maxLength)}"
            )

        for suffix, operator in ((_MIN_VALUE, ">="), (_MAX_VALUE, "<=")):
            bound = self._extension_scalar(item, suffix)
            if bound is None:
                continue
            literal = self._fhirpath_literal(bound, str(item.type))
            if literal is None:
                self._record(
                    "questionnaire-answer-bound-unsupported",
                    "Questionnaire answer bound is complex and remains subject "
                    "to post-transform validation.",
                    link_id=str(item.linkId),
                    constraint=suffix,
                )
                continue
            checks.append(f"$this {operator} {literal}")

        if item.type == "choice" and item.answerOption:
            literals = []
            for option in self._extract_options(item):
                if option.get("element") not in {
                    "valueString",
                    "valueInteger",
                    "valueDate",
                    "valueTime",
                }:
                    literals = []
                    break
                value_kind = str(option.get("element") or "").removeprefix("value")
                if value_kind:
                    value_kind = value_kind[0].lower() + value_kind[1:]
                literal = self._fhirpath_literal(
                    option.get("value"), value_kind
                )
                if literal is None:
                    literals = []
                    break
                literals.append(literal)
            if literals:
                checks.append(
                    "("
                    + " or ".join(f"$this = {literal}" for literal in literals)
                    + ")"
                )
        return " and ".join(f"({check})" for check in checks) or None

    def _count_check(
        self,
        item: QuestionnaireItem,
        source_fields: List[str],
        *,
        checkbox_fields: bool,
    ) -> Optional[str]:
        if not source_fields:
            return None
        if checkbox_fields:
            terms = [
                f"{_local(field)}.where($this = '1').count()"
                for field in source_fields
            ]
            count = " + ".join(terms)
        else:
            count = f"{_local(source_fields[0])}.count()"

        minimum = self._extension_scalar(item, _QUESTIONNAIRE_MIN_OCCURS)
        maximum = self._extension_scalar(item, _QUESTIONNAIRE_MAX_OCCURS)
        conditional = bool(
            item.enableWhen
            or self._extension(item, _SDC_ENABLE_WHEN_EXPRESSION)
        )
        if conditional and (
            minimum is not None
            or maximum is not None
            or bool(getattr(item, "required", False))
            or (
                not bool(item.repeats)
                and (self._source_repeats(source_fields) or len(source_fields) > 1)
            )
        ):
            self._record(
                "questionnaire-conditional-cardinality-postvalidation",
                "Questionnaire answer cardinality is conditional on enablement "
                "and remains subject to QuestionnaireResponse validation.",
                link_id=str(item.linkId),
            )
            return None
        checks = []
        if minimum is None and bool(getattr(item, "required", False)):
            minimum = 1
        if minimum is not None:
            try:
                checks.append(f"({count}) >= {int(minimum)}")
            except (TypeError, ValueError):
                self._record(
                    "questionnaire-occurrence-bound-invalid",
                    "Questionnaire minimum answer occurrence is not an integer.",
                    link_id=str(item.linkId),
                    constraint=_QUESTIONNAIRE_MIN_OCCURS,
                )
        if maximum is not None:
            try:
                checks.append(f"({count}) <= {int(maximum)}")
            except (TypeError, ValueError):
                self._record(
                    "questionnaire-occurrence-bound-invalid",
                    "Questionnaire maximum answer occurrence is not an integer.",
                    link_id=str(item.linkId),
                    constraint=_QUESTIONNAIRE_MAX_OCCURS,
                )

        source_repeats = self._source_repeats(source_fields)
        if not bool(item.repeats) and (source_repeats or len(source_fields) > 1):
            checks.append(f"({count}) <= 1")
        return " and ".join(f"({check})" for check in checks) or None

    def _source_repeats(self, source_fields: List[str]) -> bool:
        source_max = getattr(self.factory, "source_field_max", {}) or {}
        return any(
            str(source_max.get(_local(field), "1")) not in {"0", "1"}
            for field in source_fields
        )

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
            vr = getattr(ao, "valueReference", None)
            if vr is not None:
                opts.append(
                    {
                        "element": "valueReference",
                        "value": vr.model_dump(exclude_none=True, by_alias=True),
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

    @staticmethod
    def _choice_value(value: Any, prefix: str = "value") -> Optional[tuple[str, Any]]:
        """Return the populated FHIR choice property and its plain value."""

        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True, by_alias=True)
        if not isinstance(value, dict):
            return None
        for key, item in value.items():
            if (
                key.startswith(prefix)
                and not key.startswith(f"_{prefix}")
                and item is not None
            ):
                return key, item
        return None

    def _initial_values(self, item: QuestionnaireItem) -> List[tuple[str, Any]]:
        """Collect explicit initial values and selected answer options."""

        result = []
        for initial in item.initial or []:
            value = self._choice_value(initial)
            if value:
                result.append(value)
        if result:
            return result
        for option in item.answerOption or []:
            if getattr(option, "initialSelected", False):
                value = self._choice_value(option)
                if value:
                    result.append(value)
        return result

    @staticmethod
    def _type_from_value_element(element: str) -> str:
        return {
            "valueAttachment": "Attachment",
            "valueBoolean": "boolean",
            "valueCoding": "Coding",
            "valueDate": "date",
            "valueDateTime": "dateTime",
            "valueDecimal": "decimal",
            "valueInteger": "integer",
            "valueQuantity": "Quantity",
            "valueReference": "Reference",
            "valueString": "string",
            "valueTime": "time",
            "valueUri": "uri",
        }.get(element, "string")

    def _create_initial_answer_rule(
        self,
        rule_name: str,
        index: int,
        value_element: str,
        value: Any,
        source_context: str,
    ) -> Optional[StructureMapGroupRule]:
        """Create one QuestionnaireResponse.answer from a deterministic default."""

        field_type = self._type_from_value_element(value_element)
        target_element = (
            "value"
            if field_type in {"Attachment", "Coding", "Quantity", "Reference"}
            else value_element
        )
        if self.factory:
            fixed_rule = self.factory._create_fixed_value_rule(
                field_type,
                value,
                target_element,
                "tAns",
                parent_source_context=source_context,
            )
        else:
            fixed_rule = None
        if fixed_rule is None:
            recorder = getattr(self.factory, "record_diagnostic", None)
            if callable(recorder):
                recorder(
                    "questionnaire-initial-not-emitted",
                    "Questionnaire initial value could not be represented as a "
                    "StructureMap fixed assignment.",
                    link_id=rule_name.removeprefix("rule-"),
                    value_element=value_element,
                )
            return None
        fixed_rule.name = f"set-initial-{index}"
        return StructureMapGroupRule(
            name=f"initial-{rule_name}-{index}",
            source=[StructureMapGroupRuleSource(context=source_context)],
            target=[
                StructureMapGroupRuleTarget(
                    context="tItem", element="answer", variable="tAns"
                )
            ],
            rule=[fixed_rule],
        )

    @staticmethod
    def _fhirpath_literal(
        value: Any, value_type: Optional[str] = None
    ) -> Optional[str]:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
            return str(value)
        if value_type in {"date", "dateTime"}:
            return f"@{value}"
        if value_type == "time":
            return f"@T{value}"
        if isinstance(value, str):
            return "'" + value.replace("'", "''") + "'"
        return None

    @staticmethod
    def _complex_enable_predicate(
        answer_element: str,
        answer_value: Any,
        operator: str,
    ) -> Optional[str]:
        original_operator = operator
        if hasattr(answer_value, "model_dump"):
            answer_value = answer_value.model_dump(
                exclude_none=True, by_alias=True
            )
        if not isinstance(answer_value, dict):
            return None

        def _clauses(keys):
            clauses = []
            for key in keys:
                if answer_value.get(key) is None:
                    continue
                literal = QuestionnaireMapCreator._fhirpath_literal(
                    answer_value[key]
                )
                if literal is None:
                    return None
                clauses.append(f"{key} = {literal}")
            return clauses

        if answer_element == "answerCoding":
            if operator not in {"=", "!="}:
                return None
            # Display is presentation metadata, not part of coded identity.
            clauses = _clauses(("system", "version", "code"))
        elif answer_element == "answerReference":
            if operator not in {"=", "!="}:
                return None
            clauses = _clauses(("reference",))
            identifier = answer_value.get("identifier")
            if identifier:
                if not isinstance(identifier, dict):
                    return None
                for key in ("system", "value"):
                    if identifier.get(key) is not None:
                        literal = QuestionnaireMapCreator._fhirpath_literal(
                            identifier[key]
                        )
                        if literal is None:
                            return None
                        clauses.append(f"identifier.{key} = {literal}")
        elif answer_element == "answerQuantity":
            clauses = _clauses(("system", "code", "unit"))
            quantity_value = answer_value.get("value")
            literal = QuestionnaireMapCreator._fhirpath_literal(quantity_value)
            if literal is None:
                return None
            value_operator = "=" if operator in {"=", "!="} else operator
            clauses.insert(0, f"value {value_operator} {literal}")
        else:
            return None
        if not clauses:
            return None
        predicate = " and ".join(clauses)
        return f"not({predicate})" if original_operator == "!=" else predicate

    def _enable_condition(
        self,
        item: QuestionnaireItem,
        linkid_to_sources: Dict[str, List[str]],
    ) -> Optional[tuple[str, str]]:
        """Compile core/SDC enablement against the target response built so far."""

        del linkid_to_sources  # enablement uses QuestionnaireResponse semantics.
        expression = self._expression(item, _SDC_ENABLE_WHEN_EXPRESSION)
        if expression and item.enableWhen:
            self._record(
                "questionnaire-enablewhen-conflict",
                "Questionnaire item declares both enableWhen and "
                "enableWhenExpression; core enableWhen takes precedence.",
                link_id=str(item.linkId),
            )
            expression = None
        if expression:
            return self._target_root_alias, expression

        conditions = []
        for enable_when in item.enableWhen or []:
            answer = self._choice_value(enable_when, prefix="answer")
            operator = str(enable_when.operator)
            if answer is None:
                conditions = []
                break
            answer_element, answer_value = answer
            question = str(enable_when.question).replace("'", "''")
            values = (
                "$this.repeat(item)"
                f".where(linkId = '{question}').answer.value"
            )
            if operator == "exists" and answer_element == "answerBoolean":
                conditions.append(
                    f"{values}.exists()"
                    if answer_value
                    else f"{values}.empty()"
                )
                continue
            if operator not in {"=", "!=", ">", "<", ">=", "<="}:
                conditions = []
                break
            value_kind = answer_element.removeprefix("answer")
            if value_kind:
                value_kind = value_kind[0].lower() + value_kind[1:]
            literal = self._fhirpath_literal(answer_value, value_kind)
            if literal is not None:
                conditions.append(
                    f"{values}.where($this {operator} {literal}).exists()"
                )
                continue
            predicate = self._complex_enable_predicate(
                answer_element, answer_value, operator
            )
            if predicate:
                conditions.append(f"{values}.where({predicate}).exists()")
                continue
            conditions = []
            break

        if conditions:
            behavior = str(item.enableBehavior or "all")
            joiner = " or " if behavior == "any" else " and "
            return (
                self._target_root_alias,
                joiner.join(f"({condition})" for condition in conditions),
            )
        if item.enableWhen:
            self._record(
                "questionnaire-enablewhen-authoring-required",
                "Questionnaire enableWhen could not be represented as a "
                "target-answer FHIRPath condition; the item remains ungated.",
                link_id=str(item.linkId),
            )
        return None

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
                StructureMapGroupRuleTargetParameter(valueId="srcStatus")
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

    def _attach_question_children(
        self,
        item: QuestionnaireItem,
        answer_rules: List[StructureMapGroupRule],
        source_context: str,
        linkid_to_sources: Dict[str, List[str]],
    ) -> None:
        """Place child questions under QR.answer.item when correlation is unambiguous."""

        if not item.item:
            return
        if bool(item.repeats) or len(answer_rules) != 1:
            self._record(
                "questionnaire-nested-answer-correlation-required",
                "Children of a question with multiple possible answers require "
                "an authored source-to-answer correlation rule.",
                link_id=str(item.linkId),
            )
            return
        self._process_items(
            item.item,
            answer_rules[0].rule,
            source_context,
            "tAns",
            linkid_to_sources,
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
        self._record_answer_expression_boundary(item)

        source_fields = linkid_to_sources.get(link_id, [])
        is_multi = len(source_fields) > 1
        is_group = item.type == "group"
        is_unmapped = not is_group and not source_fields
        source_repeats = self._source_repeats(source_fields)
        # FHIR repetition differs by item kind: a repeating group produces
        # repeated QR.item nodes, while a repeating question produces one
        # QR.item containing repeated answer nodes.  A source collection also
        # needs root scope for a non-repeating question so the cardinality guard
        # can reject more than one value before any answer is emitted.
        answer_collection = (
            not is_group
            and bool(source_fields)
            and (
                bool(item.repeats)
                or source_repeats
                or is_multi
                or bool(getattr(item, "required", False))
            )
        )

        mapped_group = is_group and bool(source_fields)
        if (is_group and not mapped_group) or is_unmapped or answer_collection:
            outer_source = StructureMapGroupRuleSource(
                context=source_context,
                variable="srcItemRoot" if answer_collection else "srcVal",
            )
        else:
            primary_source_element = _local(source_fields[0])
            outer_source = StructureMapGroupRuleSource(
                context=source_context,
                element=primary_source_element,
                variable="srcVal",
            )
            outer_source.check = self._value_check(item)

        count_check = self._count_check(
            item, source_fields, checkbox_fields=is_multi
        )
        if count_check and outer_source.context == source_context and not outer_source.element:
            outer_source.check = count_check
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
        if count_check and outer_source.check != count_check:
            outer_rule.source.append(
                StructureMapGroupRuleSource(
                    context=source_context,
                    check=count_check,
                )
            )
        enable_condition = self._enable_condition(item, linkid_to_sources)
        if enable_condition:
            enable_context, enable_expression = enable_condition
            outer_rule.source.append(
                StructureMapGroupRuleSource(
                    context=enable_context,
                    condition=enable_expression,
                )
            )

        metadata_source = "srcItemRoot" if answer_collection else "srcVal"
        outer_rule.rule.append(
            self._create_assignment_rule(
                "tItem", "linkId", link_id, src_context=metadata_source
            )
        )
        if item.text:
            outer_rule.rule.append(
                self._create_assignment_rule(
                    "tItem", "text", item.text, src_context=metadata_source
                )
            )

        if is_group:
            if item.item:
                self._process_items(
                    item.item,
                    outer_rule.rule,
                    "srcVal" if mapped_group else source_context,
                    "tItem",
                    linkid_to_sources,
                )
        elif is_multi:
            options = self._extract_options(item)
            checkbox_rules = []
            for idx, sf in enumerate(source_fields):
                option = options[idx] if idx < len(options) else None
                checkbox_rule = self._create_checkbox_answer_rule(
                    sf, source_context, "tItem", item.type, idx + 1, option
                )
                checkbox_rules.append(checkbox_rule)
                outer_rule.rule.append(checkbox_rule)
            self._attach_question_children(
                item, checkbox_rules, source_context, linkid_to_sources
            )
        elif is_unmapped:
            calculated_expression = self._expression(
                item, _SDC_CALCULATED_EXPRESSION
            )
            initial_expression = self._expression(item, _SDC_INITIAL_EXPRESSION)
            initial_rules = [
                self._create_initial_answer_rule(
                    rule_name,
                    index,
                    value_element,
                    value,
                    source_context,
                )
                for index, (value_element, value) in enumerate(
                    self._initial_values(item), start=1
                )
            ]
            emitted_initial_rules = [
                rule for rule in initial_rules if rule is not None
            ]
            if calculated_expression:
                if emitted_initial_rules or initial_expression:
                    self._record(
                        "questionnaire-expression-conflict",
                        "Questionnaire calculatedExpression takes precedence over "
                        "initial values and initialExpression in a one-shot transform.",
                        link_id=str(item.linkId),
                    )
                outer_rule.rule.append(
                    self._create_expression_answer_rule(
                        rule_name, item.type, calculated_expression, "calculated"
                    )
                )
            elif initial_expression:
                if emitted_initial_rules:
                    self._record(
                        "questionnaire-expression-conflict",
                        "Questionnaire initialExpression takes precedence over "
                        "fixed initial values in a one-shot transform.",
                        link_id=str(item.linkId),
                    )
                outer_rule.rule.append(
                    self._create_expression_answer_rule(
                        rule_name, item.type, initial_expression, "initial"
                    )
                )
            elif emitted_initial_rules:
                outer_rule.rule.extend(emitted_initial_rules)
            else:
                outer_rule.rule.append(
                    self._create_placeholder_answer_rule(
                        rule_name, safe_link_id, item.type, source_context
                    )
                )
            answer_rules = [
                rule
                for rule in outer_rule.rule
                if rule.name.startswith(
                    (
                        "answer-",
                        "initial-",
                        "calculated-expression-",
                        "initial-expression-",
                    )
                )
            ]
            self._attach_question_children(
                item, answer_rules, source_context, linkid_to_sources
            )
        else:
            if self._extension(item, _SDC_CALCULATED_EXPRESSION) or self._extension(
                item, _SDC_INITIAL_EXPRESSION
            ):
                self._record(
                    "questionnaire-expression-shadowed-by-mapping",
                    "An authored source mapping takes precedence over the "
                    "Questionnaire expression for this item.",
                    link_id=str(item.linkId),
                )
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
                    rule_name,
                    cm_url or "TODO-resolveConceptMap",
                    system,
                    source_context=(
                        "srcItemRoot" if answer_collection else "srcVal"
                    ),
                    source_element=(
                        _local(source_fields[0]) if answer_collection else None
                    ),
                    source_check=self._value_check(item),
                )
            elif opt_element == "valueString":
                answer_rule = self._build_copy_answer_rule(
                    rule_name,
                    "valueString",
                    source_context=(
                        "srcItemRoot" if answer_collection else "srcVal"
                    ),
                    source_element=(
                        _local(source_fields[0]) if answer_collection else None
                    ),
                    source_check=self._value_check(item),
                )
            elif opt_element in {
                "valueInteger",
                "valueDate",
                "valueTime",
                "valueReference",
            }:
                answer_rule = self._build_copy_answer_rule(
                    rule_name,
                    opt_element,
                    source_context=(
                        "srcItemRoot" if answer_collection else "srcVal"
                    ),
                    source_element=(
                        _local(source_fields[0]) if answer_collection else None
                    ),
                    source_check=self._value_check(item),
                )
            else:
                answer_rule = self._build_copy_answer_rule(
                    rule_name,
                    self._get_value_type(item.type),
                    source_context=(
                        "srcItemRoot" if answer_collection else "srcVal"
                    ),
                    source_element=(
                        _local(source_fields[0]) if answer_collection else None
                    ),
                    source_check=self._value_check(item),
                )
            outer_rule.rule.append(answer_rule)
            self._attach_question_children(
                item, [answer_rule], source_context, linkid_to_sources
            )

        rules.append(outer_rule)

    def _build_copy_answer_rule(
        self,
        rule_name: str,
        value_type: Optional[str],
        *,
        source_context: str = "srcVal",
        source_element: Optional[str] = None,
        source_check: Optional[str] = None,
    ) -> StructureMapGroupRule:
        """answer.value[x] = copy(source) — for string options and plain primitive items."""
        answer_source = StructureMapGroupRuleSource(
            context=source_context,
            element=source_element,
            variable="srcVal" if source_element else None,
            check=source_check,
        )
        answer_rule = StructureMapGroupRule(
            name=f"answer-{rule_name}",
            source=[answer_source],
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
        self,
        rule_name: str,
        cm_url: str,
        system: Optional[str],
        *,
        source_context: str = "srcVal",
        source_element: Optional[str] = None,
        source_check: Optional[str] = None,
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
            source=[
                StructureMapGroupRuleSource(
                    context=source_context,
                    element=source_element,
                    variable="srcVal" if source_element else None,
                    check=source_check,
                )
            ],
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

    def _create_expression_answer_rule(
        self,
        rule_name: str,
        item_type: str,
        expression: str,
        expression_kind: str,
    ) -> StructureMapGroupRule:
        """Evaluate an authored FHIRPath once against the response under construction."""

        value_element = self._get_value_type(item_type) or "valueString"
        if value_element in {
            "valueAttachment",
            "valueCoding",
            "valueQuantity",
            "valueReference",
        }:
            value_element = "value"
        return StructureMapGroupRule(
            name=f"{expression_kind}-expression-{rule_name}",
            source=[
                StructureMapGroupRuleSource(
                    context=self._target_root_alias,
                    variable="qrExpressionContext",
                )
            ],
            target=[
                StructureMapGroupRuleTarget(
                    context="tItem",
                    element="answer",
                    variable="tAns",
                )
            ],
            rule=[
                StructureMapGroupRule(
                    name=f"evaluate-{expression_kind}",
                    source=[
                        StructureMapGroupRuleSource(
                            context="qrExpressionContext"
                        )
                    ],
                    target=[
                        StructureMapGroupRuleTarget(
                            context="tAns",
                            element=value_element,
                            transform="evaluate",
                            parameter=[
                                StructureMapGroupRuleTargetParameter(
                                    valueId="qrExpressionContext"
                                ),
                                StructureMapGroupRuleTargetParameter(
                                    valueString=expression
                                ),
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
            value = option["value"]
            if (
                option["element"] == "valueInteger"
                and not isinstance(value, bool)
            ):
                parameter = StructureMapGroupRuleTargetParameter(
                    valueInteger=int(value)
                )
            else:
                parameter = StructureMapGroupRuleTargetParameter(
                    valueString=str(value)
                )
            return [
                StructureMapGroupRule(
                    name=f"set-val-{option_index}",
                    source=[StructureMapGroupRuleSource(context="srcCb")],
                    target=[
                        StructureMapGroupRuleTarget(
                            context="tAns",
                            element=option["element"],
                            transform="copy",
                            parameter=[parameter],
                        )
                    ],
                )
            ]
        if option and option.get("element") == "valueReference":
            fixed_rule = (
                self.factory._create_fixed_value_rule(
                    "Reference",
                    option.get("value"),
                    "value",
                    "tAns",
                    parent_source_context="srcCb",
                )
                if self.factory
                else None
            )
            if fixed_rule:
                fixed_rule.name = f"set-reference-{option_index}"
                return [fixed_rule]
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
