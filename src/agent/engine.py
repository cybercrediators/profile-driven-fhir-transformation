"""Handles validation through matchbox including parsing error messages and define report msgs"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from agent.fixtures import FixtureKind, SourceFixture, fixtures_digest
from agent.validation import (
    CLASSIFICATION,
    ActionOwner,
    GateStatus,
    Producer,
    Stage,
    ValidationFinding,
    declared_target_paths,
    deferred_reference_contracts,
    deferred_reference_paths,
    normalize_target_path,
    target_path_spellings,
    required_gap_owner,
)

logger = logging.getLogger(__name__)

_MAX_FATAL_OUTPUT_CHARS = 80000

_REPORTABLE_SEVERITIES = frozenset({"error", "fatal", "warning"})

_RULE_HINT = re.compile(r'on Rule "([^"]+)"')

#: Message shapes that name a target element
_PATH_HINTS = (
    re.compile(r"^([A-Za-z][\w\[\]:.\-]*): minimum required = "),
    re.compile(r"^Slice '([A-Za-z][\w.\-]*(?:\[x\])?)(?::[\w\-]+)?'"),
    re.compile(r"for slice ([A-Za-z][\w.\-]*(?:\[x\])?)(?::[\w\-]+)?"),
)

# define message responses for "this server does not have the definition that URL names"
_UNRESOLVABLE_CANONICAL = (
    re.compile(r"Unable to find definition '([^']+)'"),
    re.compile(r"Unable to resolve profile CanonicalType\[([^\]]+)\]"),
    re.compile(r"Unable to resolve profile ([^\s,]+)"),
    re.compile(r"No definition could be found for URL value '([^']+)'"),
    re.compile(r"ValueSet '([^']+)' not found"),
    
    re.compile(r"Profile reference '([^']+)' has not been checked"),
)

_UNDERSPECIFIED_DISCRIMINATOR = re.compile(
    r"the discriminator \[([^\]]+)\] does not have fixed value, binding or "
    r"existence assertions"
)

_UNRECOGNIZED_RESOURCE = re.compile(
    r"unknown or unrecognized resource name '([A-Za-z]+)'"
)

# max FHIR id length
_FHIR_ID_MAX_LENGTH = 64


def known_fhir_resource(name: str) -> bool:
    """if name is a resource type this build the FHIR model defines"""

    if not name or not name[:1].isupper():
        return False
    try:
        from fhir.resources.R4B import get_fhir_model_class  # noqa: PLC0415

        get_fhir_model_class(name)
    except (ImportError, ValueError, KeyError, AttributeError):
        return False
    return True


def _container_spellings(relative: str) -> List[str]:
    """A path and the choice base it may be a concrete form of"""

    parent, separator, leaf = relative.rpartition(".")
    out = [relative]
    for index in range(len(leaf) - 1, 0, -1):
        if leaf[index].isupper():
            out.append(f"{parent}{separator}{leaf[:index]}")
    return out


def _concrete_choice_name(raw_path: str, type_name: str) -> str:
    """Observation.effective[x] and dateTime resulting and returns an Observation.effectiveDateTime"""

    if "[x]" not in raw_path or not type_name:
        return ""
    parent, _, leaf = normalize_target_path(raw_path).rpartition(".")
    typed = f"{leaf}{str(type_name)[:1].upper()}{str(type_name)[1:]}"
    return f"{parent}.{typed}" if parent else typed


def _choice_names_in_output(output: Mapping[str, Any], parent: str, base: str) -> List[str]:
    """Concrete choice names the *output* actually carries under parent"""

    from data_handling.instance_validation import extract_path  # noqa: PLC0415

    containers = extract_path(dict(output), parent) if parent else [output]
    found: List[str] = []
    for container in containers or []:
        if not isinstance(container, Mapping):
            continue
        for key in container:
            name = str(key)
            if (
                name.startswith(base)
                and len(name) > len(base)
                and name[len(base)].isupper()
            ):
                found.append(name)
    return list(dict.fromkeys(found))


def _output_path_candidates(raw_path: str, target_tree, output=None) -> List[str]:
    """every JSON path a conformant output may use for one required element"""

    candidates = [normalize_target_path(raw_path)]
    if target_tree is None or "[x]" not in raw_path:
        return candidates

    segments = raw_path.split(".")
    for index, segment in enumerate(segments):
        if not segment.endswith("[x]"):
            continue
        base = segment[: -len("[x]")]
        prefix = ".".join(segments[: index + 1])
        typed_names = [
            f"{base}{str(name)[:1].upper()}{str(name)[1:]}"
            for name in target_tree.effective_types(prefix) or []
        ]
        if not typed_names and output is not None:
            # The tree does not know this choice's types. The instance does.
            parent = normalize_target_path(".".join(segments[:index]))
            typed_names = _choice_names_in_output(output, parent, base)
        for typed in typed_names:
            variant = segments[:index] + [typed] + segments[index + 1 :]
            candidates.append(normalize_target_path(".".join(variant)))
    return list(dict.fromkeys(candidates))


def unrecognized_resource_name(message: str) -> Optional[str]:
    """A resource type the engine could not parse but this build knows"""

    found = _UNRECOGNIZED_RESOURCE.search(message)
    if not found:
        return None
    name = found.group(1)
    return name if known_fhir_resource(name) else None

_BINDING_ISSUE_CODES = frozenset({"code-invalid", "value", "invalid", "not-found"})

# matching slice is required, but not found
_REQUIRED_SLICE = re.compile(r"a matching slice is required")


class EngineResult(BaseModel):
    """What the engine said about one map."""

    model_config = ConfigDict(extra="forbid")

    findings: List[ValidationFinding] = Field(default_factory=list)
    executed_fixtures: List[str] = Field(default_factory=list)
    required_fixtures: List[str] = Field(default_factory=list)
    
    validated_fixtures: List[str] = Field(default_factory=list)
    validation_expected_fixtures: List[str] = Field(default_factory=list)
    evaluation_context_sha256: Optional[str] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    cached: bool = False


def scratch_identity(document: Mapping[str, Any]) -> Tuple[str, str]:
    """Content-derived (id, url) under which a map is uploaded for testing"""

    digest = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    base_id = str(document.get("id") or "map")
    base_url = str(document.get("url") or "http://example.org/StructureMap/agent")
    
    scratch_id = f"{base_id[: _FHIR_ID_MAX_LENGTH - len(digest) - 7]}-agent-{digest}"
    return scratch_id, f"{base_url}-agent-{digest}"


def _outcome_issues(body: Any) -> List[Dict[str, Any]]:
    """Reportable issues from an ``OperationOutcome``, or a synthetic one."""

    if isinstance(body, Mapping) and body.get("resourceType") == "OperationOutcome":
        return [
            dict(issue)
            for issue in body.get("issue") or []
            if isinstance(issue, Mapping)
            and str(issue.get("severity") or "") in _REPORTABLE_SEVERITIES
        ]
    if body is None:
        return []
    return [
        {
            "severity": "error",
            "code": "processing",
            "diagnostics": str(body)[:2000],
        }
    ]


def _issue_message(issue: Mapping[str, Any]) -> str:
    """Constructing an issue message"""
    details = issue.get("details")
    text = ""
    if isinstance(details, Mapping):
        text = str(details.get("text") or "")
    return str(issue.get("diagnostics") or text or issue.get("code") or "unknown")


def _hints(message: str) -> Dict[str, str]:
    """best-effort locators pulled out of engine prose"""

    hints: Dict[str, str] = {}
    rule = _RULE_HINT.search(message)
    if rule:
        # "<map>|<group>|<rule>" — the leaf is the rule that threw.
        hints["rule_hint"] = rule.group(1).rsplit("|", 1)[-1]
    stripped = message.strip()
    for pattern in _PATH_HINTS:
        found = pattern.search(stripped)
        if found:
            hints["path_hint"] = found.group(1)
            break
    return hints


def normalize_expression(expression: str) -> str:
    """A FHIRPath ``expression`` reduced to the element definition it addresses"""

    collapsed = re.sub(r"\[\d+\]", "", str(expression))
    # `a.value.ofType(Quantity)` -> `a.valueQuantity`
    collapsed = re.sub(
        r"\.(\w+)\.ofType\((\w+)\)",
        lambda m: f".{m.group(1)}{m.group(2)[0].upper()}{m.group(2)[1:]}",
        collapsed,
    )
    collapsed = re.sub(r"\.ofType\(\w+\)", "", collapsed)
    return normalize_target_path(collapsed)


def expression_path(expressions: Sequence[str]) -> Optional[str]:
    """The target element an issue's FHIRPath expression addresses"""

    for expression in expressions:
        path = normalize_expression(expression)
        if "." in path:
            return path
    return None


def _expression_node(output: Mapping[str, Any], expression: str) -> Any:
    """Resolve a simple resource-rooted OperationOutcome expression."""

    tokens = str(expression).split(".")
    if tokens and tokens[0] == str(output.get("resourceType") or ""):
        tokens = tokens[1:]
    node: Any = output
    for token in tokens:
        found = re.fullmatch(r"([^\[]+)(?:\[(\d+)\])?", token)
        if found is None or not isinstance(node, Mapping):
            return None
        node = node.get(found.group(1))
        if found.group(2) is not None:
            if not isinstance(node, list):
                return None
            index = int(found.group(2))
            if index >= len(node):
                return None
            node = node[index]
    return node


def _deferred_validation_path(
    expressions: Sequence[str],
    hints: Mapping[str, str],
    output: Mapping[str, Any],
    deferred: Set[str],
    contracts: Sequence[Mapping[str, Any]],
) -> Optional[str]:
    """deferred path responsible for an otherwise blocking validation issue"""

    located = re.sub(
        r"\[\d+\]",
        "",
        normalize_target_path(
            str(hints.get("path_hint") or expression_path(expressions) or "")
        ),
    )
    selector_paths = {
        re.sub(
            r"\[\d+\]",
            "",
            normalize_target_path(str(contract.get("fullPath") or "")),
        )
        for contract in contracts
        if any(
            isinstance(selector, Mapping)
            for selector in (contract.get("selectors") or [])
        )
    }
    if located:
        for path in deferred:
            normalized = re.sub(r"\[\d+\]", "", normalize_target_path(path))
            if normalized in selector_paths:
                continue
            if located == normalized or located.startswith(normalized + "."):
                return path

    for contract in contracts:
        full_path = str(contract.get("fullPath") or "")
        normalized = re.sub(r"\[\d+\]", "", normalize_target_path(full_path))
        if not located or not (
            located == normalized
            or normalized.startswith(located + ".")
            or located.startswith(normalized + ".")
        ):
            continue
        selectors = [
            selector
            for selector in contract.get("selectors") or []
            if isinstance(selector, Mapping)
        ]
        if not selectors:
            continue
        for expression in expressions:
            candidate_expression = str(expression)
            node = _expression_node(output, candidate_expression)
            while node is None and "." in candidate_expression:
                candidate_expression = candidate_expression.rsplit(".", 1)[0]
                node = _expression_node(output, candidate_expression)
            candidates = node if isinstance(node, list) else [node]
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    continue
                if any(
                    selector.get("discriminator") == "url"
                    and candidate.get("url") == selector.get("value")
                    for selector in selectors
                ):
                    return full_path
    return None


def _fatal_output_evidence(
    output: Mapping[str, Any], fixture: SourceFixture, issue: Mapping[str, Any]
) -> Dict[str, Any]:
    """bounded evidence for a fatal validator response"""

    encoded = json.dumps(output, sort_keys=True, separators=(",", ":"), default=str)
    evidence: Dict[str, Any] = {
        "raw_issue": copy.deepcopy(dict(issue)),
        "output_resource_type": output.get("resourceType"),
        "output_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "output_top_level_keys": sorted(str(key) for key in output),
    }
    if (
        fixture.kind is FixtureKind.SYNTHETIC_FILLED
        and len(encoded) <= _MAX_FATAL_OUTPUT_CHARS
    ):
        evidence["synthetic_output"] = copy.deepcopy(dict(output))
    elif fixture.kind is FixtureKind.SYNTHETIC_FILLED:
        evidence["synthetic_output_omitted"] = "size-limit"
        evidence["synthetic_output_chars"] = len(encoded)
    return evidence


def _root_relative(path: str) -> str:
    """drop the leading segment of a dotted target path"""

    _root, _, rest = path.partition(".")
    return rest or path


def unresolvable_canonical(message: str) -> Optional[str]:
    """The canonical the engine said it could not resolve, when it said so.

    :param message: the engines own prose for one issue.
    :return: the URL, or None when the message is not about a missing definition.
    """

    for pattern in _UNRESOLVABLE_CANONICAL:
        found = pattern.search(message)
        if found:
            return found.group(1)
    return None


def _environment_owner(
    message: str, override: Optional[ActionOwner]
) -> Tuple[Optional[ActionOwner], Dict[str, str]]:
    """Ownership and evidence for one engine issue, before classification"""

    if override is not None:
        return override, {}
    canonical = unresolvable_canonical(message)
    if canonical is not None:
        return ActionOwner.ENVIRONMENT, {"unresolvable_canonical": canonical}
    
    resource = unrecognized_resource_name(message)
    if resource is not None:
        return ActionOwner.ENVIRONMENT, {"unrecognized_resource": resource}
    
    discriminator = _UNDERSPECIFIED_DISCRIMINATOR.search(message)
    if discriminator is not None:
        return (
            ActionOwner.ENVIRONMENT,
            {"underspecified_discriminator": discriminator.group(1)},
        )
    return None, {}


def _expansion_codes(response: Any) -> Optional[List[str]]:
    """Codes out of a ``ValueSet/$expand`` response, or None when it has none."""

    if not isinstance(response, Mapping):
        return None
    codes = [
        str(item["code"])
        for item in ((response.get("expansion") or {}).get("contains") or [])
        if isinstance(item, Mapping) and item.get("code")
    ]
    return codes or None


class EngineSession:
    """manage Matchbox connection/session, reused across baseline and candidate"""

    def __init__(
        self,
        matchbox_controller,
        *,
        validate_output: bool = True,
        terminology_url: Optional[str] = None,
    ) -> None:
        """
        :param matchbox_controller: an existing
            :class:`controller.external_services.matchbox_controller.MatchboxController`.
        :param validate_output: run layer 6 (``$validate``) on transform output.
        :param terminology_url: the project's ``terminology_server_uri``. Asked
            for value-set expansions before Matchbox, which generally cannot
            expand at all.
        """

        self.matchbox = matchbox_controller
        self.validate_output = validate_output
        self.terminology_url = str(terminology_url).strip() if terminology_url else None
        
        self._session_id = uuid.uuid4().hex
        self._available: Optional[bool] = None
        
        self._known: Set[str] = set()

        self._uploaded_profiles: Set[str] = set()

        self._bootstrap_digest = "none"
        self._bootstrapped = False
        self._verdicts: Dict[str, EngineResult] = {}

    def available(self) -> bool:
        """True when the server answers its CapabilityStatement."""

        if self._available is None:
            try:
                self._available = bool(self.matchbox.get_capability_statement())
            except Exception as exc:
                logger.info("Matchbox is not reachable (%s: %s).", type(exc).__name__, exc)
                self._available = False
        return self._available

    def bootstrap(
        self,
        *,
        source_model: Optional[Mapping[str, Any]] = None,
        profiles: Iterable[Mapping[str, Any]] = (),
        concept_maps: Iterable[Mapping[str, Any]] = (),
        questionnaires: Iterable[Mapping[str, Any]] = (),
    ) -> List[ValidationFinding]:
        """Put all run-related resources on the (matchbox) server.

        :param questionnaires: Questionnaires a QuestionnaireResponse map is
            validated against

        :return: environment findings for anything the server refused. Empty
            means the engine is usable.
        """

        findings: List[ValidationFinding] = []
        if not self.available():
            return [self._unreachable_finding()]

        digest_parts: List[str] = []
        resources: List[Tuple[str, Mapping[str, Any]]] = []
        if source_model is not None:
            resources.append(("source model", source_model))
        resources.extend(("profile", profile) for profile in profiles or ())
        resources.extend(("ConceptMap", concept_map) for concept_map in concept_maps or ())
        resources.extend(
            ("Questionnaire", questionnaire) for questionnaire in questionnaires or ()
        )

        for label, resource in resources:
            if not isinstance(resource, Mapping) or not resource.get("url"):
                continue
            payload = dict(resource)
            payload.setdefault("id", str(payload["url"]).rsplit("/", 1)[-1])
            status, body = self.matchbox.upsert_resource(payload)
            digest_parts.append(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
            )
            if status is None or status >= 400:
                message = "; ".join(
                    _issue_message(issue) for issue in _outcome_issues(body)
                ) or f"HTTP {status}"
                findings.append(
                    ValidationFinding.build(
                        Producer.ENGINE,
                        Stage.BOOTSTRAP,
                        "dependency-missing",
                        f"The server refused the {label} {payload['url']}: {message}",
                        path=str(payload["url"]),
                        evidence={"status": status, "resource_type": payload.get("resourceType")},
                    )
                )
                if label == "source model":
                    logger.warning(
                        "Matchbox refused the source logical model — engine validation "
                        "is unusable for this session."
                    )
                    self._available = False
                    return findings
                continue
            self._known.add(str(payload["url"]))
            if payload.get("resourceType") == "StructureDefinition":
                self._uploaded_profiles.add(str(payload["url"]))

        self._bootstrap_digest = hashlib.sha256(
            "".join(sorted(digest_parts)).encode("utf-8")
        ).hexdigest()[:16]
        self._bootstrapped = True
        logger.info(
            "Engine bootstrap complete (%d resource(s), %d refused).",
            len(resources),
            len(findings),
        )
        return findings

    def _unreachable_finding(self) -> ValidationFinding:
        return ValidationFinding.build(
            Producer.ENGINE,
            Stage.BOOTSTRAP,
            "engine-unreachable",
            "Matchbox is not reachable, so no engine evidence exists for this "
            "map. This is an environment failure and never a pass.",
        )

    def declared_dependencies(self, document: Mapping[str, Any]) -> List[str]:
        """canonicals the map needs at runtime, read from its own structure"""

        canonicals: List[str] = []
        for entry in document.get("structure") or []:
            if isinstance(entry, Mapping) and entry.get("url"):
                canonicals.append(str(entry["url"]))
        for entry in document.get("import") or []:
            if entry:
                canonicals.append(str(entry))

        def walk(rules: Any) -> None:
            for rule in rules or []:
                if not isinstance(rule, Mapping):
                    continue
                for target in rule.get("target") or []:
                    if not isinstance(target, Mapping):
                        continue
                    if target.get("transform") != "translate":
                        continue
                    for parameter in target.get("parameter") or []:
                        if isinstance(parameter, Mapping) and parameter.get("valueString"):
                            canonicals.append(str(parameter["valueString"]))
                walk(rule.get("rule"))

        for group in document.get("group") or []:
            if isinstance(group, Mapping):
                walk(group.get("rule"))
        return sorted(set(canonicals))

    def _missing_dependencies(self, document: Mapping[str, Any]) -> List[str]:
        missing = []
        for canonical in self.declared_dependencies(document):
            if canonical in self._known:
                continue
            if canonical.startswith("http://hl7.org/fhir/StructureDefinition/"):
                self._known.add(canonical)
                continue
            found = None
            for resource_type in ("StructureDefinition", "ConceptMap", "StructureMap"):
                found = self.matchbox.get_resource_by_url(resource_type, canonical)
                if found:
                    break
            if found:
                self._known.add(canonical)
            else:
                missing.append(canonical)
        return missing

    def validate_map(
        self,
        document: Mapping[str, Any],
        fixtures: Sequence[SourceFixture],
        *,
        target_tree=None,
        profile_url: Optional[str] = None,
        profile_id: Optional[str] = None,
        mapping_table: Optional[Mapping[str, Any]] = None,
    ) -> EngineResult:
        """takes a StructureMap, uploads to Matchbox for $validate and retrieves
        validation results.

        :param document: the StructureMap as plain JSON.
        :param fixtures: inputs to transform
        :param target_tree: target profile tree
        :param profile_url: target profile for $validate
        :param mapping_table: used to decide who owns a required-output gap.
        :return: an :class:EngineResult
        """

        map_url = document.get("url")
        map_id = document.get("id")
        required = sorted(
            fixture.fixture_id for fixture in fixtures if fixture.gate_relevant
        )
        validation_expected = sorted(
            fixture.fixture_id
            for fixture in fixtures
            if fixture.gate_relevant
            and fixture.kind is not FixtureKind.SYNTHETIC_EMPTY_OPTIONAL
            and bool(profile_url)
            and self.validate_output
        )
        evaluation_context = self._evaluation_context_digest(
            fixtures, profile_url, target_tree, mapping_table
        )
        if not self.available():
            return EngineResult(
                findings=[self._unreachable_finding()],
                required_fixtures=required,
                validation_expected_fixtures=validation_expected,
                evaluation_context_sha256=evaluation_context,
            )
        if not self._bootstrapped:
            logger.debug("Validating without an explicit bootstrap — nothing uploaded.")

        cache_key = self._cache_key(document, evaluation_context)
        cached = self._verdicts.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"cached": True})

        missing = self._missing_dependencies(document)
        findings: List[ValidationFinding] = [
            ValidationFinding.build(
                Producer.ENGINE,
                Stage.BOOTSTRAP,
                "dependency-missing",
                f"The map declares {canonical}, which the server cannot resolve.",
                map_url=map_url,
                map_id=map_id,
                path=canonical,
            )
            for canonical in missing
        ]

        owner_override = ActionOwner.ENVIRONMENT if missing else None

        engine_id, engine_url = scratch_identity(document)
        engine_copy = copy.deepcopy(dict(document))
        engine_copy["id"] = engine_id
        engine_copy["url"] = engine_url

        status, body = self.matchbox.upsert_resource(engine_copy)
        if status is None:
            self._available = False
            result = EngineResult(
                findings=[
                    self._transport_finding(
                        Stage.UPLOAD,
                        "The StructureMap upload did not reach Matchbox.",
                        map_url=map_url,
                        map_id=map_id,
                    )
                ],
                required_fixtures=required,
                validation_expected_fixtures=validation_expected,
                evaluation_context_sha256=evaluation_context,
            )
            self._verdicts[cache_key] = result
            return result
        if status >= 400:
            for issue in _outcome_issues(body) or [
                {"severity": "error", "code": "processing", "diagnostics": f"HTTP {status}"}
            ]:
                message = _issue_message(issue)
                owner, unresolvable = _environment_owner(message, owner_override)
                findings.append(
                    ValidationFinding.build(
                        Producer.ENGINE,
                        Stage.UPLOAD,
                        f"upload:{issue.get('code') or 'processing'}",
                        message,
                        owner=owner,
                        map_url=map_url,
                        map_id=map_id,
                        severity=str(issue.get("severity") or ""),
                        issue_code=str(issue.get("code") or ""),
                        evidence={"status": status, **unresolvable, **_hints(message)},
                    )
                )
            result = EngineResult(
                findings=findings,
                required_fixtures=required,
                validation_expected_fixtures=validation_expected,
                evaluation_context_sha256=evaluation_context,
            )
            self._verdicts[cache_key] = result
            return result

        outputs: Dict[str, Any] = {}
        executed: List[str] = []
        for fixture in fixtures:
            fixture_findings, output = self._run_fixture(
                fixture,
                engine_url,
                map_url=map_url,
                map_id=map_id,
                owner_override=owner_override,
            )
            findings.extend(fixture_findings)
            if any(item.code == "engine-unreachable" for item in fixture_findings):
                break
            executed.append(fixture.fixture_id)
            if output is not None:
                outputs[fixture.fixture_id] = output

        validated: List[str] = []
        if self.validate_output and profile_url:
            validate_findings, validated = self._validate_outputs(
                outputs,
                fixtures,
                profile_url=profile_url,
                map_url=map_url,
                map_id=map_id,
                target_tree=target_tree,
                mapping_table=mapping_table,
                owner_override=owner_override,
                deferred=deferred_reference_paths(document),
                deferred_contracts=deferred_reference_contracts(document),
                profile_id=profile_id,
            )
            findings.extend(validate_findings)

        result = EngineResult(
            findings=findings,
            executed_fixtures=executed,
            required_fixtures=required,
            validated_fixtures=validated,
            validation_expected_fixtures=validation_expected,
            evaluation_context_sha256=evaluation_context,
            outputs=outputs,
        )
        self._verdicts[cache_key] = result
        return result

    def _evaluation_context_digest(
        self,
        fixtures: Sequence[SourceFixture],
        profile_url: Optional[str],
        target_tree,
        mapping_table: Optional[Mapping[str, Any]],
    ) -> str:
        """evaluate the identity of a complete verdict"""

        table = json.dumps(
            dict(mapping_table or {}), sort_keys=True, separators=(",", ":"), default=str
        )
        manifest = json.dumps(
            target_tree.required_manifest() if target_tree is not None else None,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        payload = "|".join(
            [
                self._session_id,
                fixtures_digest(fixtures),
                self._bootstrap_digest,
                str(profile_url),
                str(self.validate_output),
                hashlib.sha256(table.encode("utf-8")).hexdigest()[:16],
                hashlib.sha256(manifest.encode("utf-8")).hexdigest()[:16],
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _cache_key(document: Mapping[str, Any], evaluation_context: str) -> str:
        _engine_id, engine_url = scratch_identity(document)
        return hashlib.sha256(
            f"{engine_url}|{evaluation_context}".encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _transport_finding(
        stage: Stage,
        message: str,
        *,
        map_url=None,
        map_id=None,
        fixture_id=None,
    ) -> ValidationFinding:
        """A request that never reached Matchbox is environmental, never a map edit."""

        return ValidationFinding.build(
            Producer.ENGINE,
            stage,
            "engine-unreachable",
            message,
            map_url=map_url,
            map_id=map_id,
            fixture_id=fixture_id,
        )

    def _run_fixture(
        self,
        fixture: SourceFixture,
        engine_url: str,
        *,
        map_url,
        map_id,
        owner_override: Optional[ActionOwner],
    ) -> Tuple[List[ValidationFinding], Any]:
        status, body = self.matchbox.transform_data_detailed(fixture.instance, engine_url)
        findings: List[ValidationFinding] = []

        if status is None:
            self._available = False
            return [
                self._demote(
                    self._transport_finding(
                        Stage.TRANSFORM,
                        "The fixture transform did not reach Matchbox.",
                        map_url=map_url,
                        map_id=map_id,
                        fixture_id=fixture.fixture_id,
                    ),
                    fixture,
                )
            ], None

        if status is not None and 200 <= status < 400:
            if isinstance(body, Mapping) and body.get("resourceType"):
                return findings, dict(body)
            findings.append(
                self._demote(
                    ValidationFinding.build(
                        Producer.ENGINE,
                        Stage.TRANSFORM,
                        "transform:no-output",
                        "The transform succeeded but produced no resource.",
                        owner=owner_override,
                        map_url=map_url,
                        map_id=map_id,
                        fixture_id=fixture.fixture_id,
                        evidence={"fixture": fixture.label},
                    ),
                    fixture,
                )
            )
            return findings, None

        for issue in _outcome_issues(body) or [
            {"severity": "error", "code": "processing", "diagnostics": f"HTTP {status}"}
        ]:
            message = _issue_message(issue)
            owner, unresolvable = _environment_owner(message, owner_override)
            findings.append(
                self._demote(
                    ValidationFinding.build(
                        Producer.ENGINE,
                        Stage.TRANSFORM,
                        f"transform:{issue.get('code') or 'processing'}",
                        message,
                        owner=owner,
                        map_url=map_url,
                        map_id=map_id,
                        fixture_id=fixture.fixture_id,
                        severity=str(issue.get("severity") or ""),
                        issue_code=str(issue.get("code") or ""),
                        evidence={
                            "fixture": fixture.label,
                            "status": status,
                            **unresolvable,
                            **_hints(message),
                        },
                    ),
                    fixture,
                )
            )
        return findings, None

    def _validate_outputs(
        self,
        outputs: Mapping[str, Any],
        fixtures: Sequence[SourceFixture],
        *,
        profile_url: str,
        map_url,
        map_id,
        target_tree,
        mapping_table: Optional[Mapping[str, Any]],
        owner_override: Optional[ActionOwner],
        deferred: Optional[Set[str]] = None,
        deferred_contracts: Optional[Sequence[Mapping[str, Any]]] = None,
        profile_id: Optional[str] = None,
    ) -> Tuple[List[ValidationFinding], List[str]]:
        """validate outputs including required-output check

        :return: (findings, validated_fixture_ids)
        """

        from agent.fixtures import FixtureKind  # noqa: PLC0415

        findings: List[ValidationFinding] = []
        validated: List[str] = []
        by_id = {fixture.fixture_id: fixture for fixture in fixtures}
        unverified, verified = self._target_value_provenance(fixtures, mapping_table)
        # The same view `_required_output_findings` uses
        required_gaps = self._required_gap_owners(
            target_tree, mapping_table, profile_id
        )
        for fixture_id, output in outputs.items():
            fixture = by_id.get(fixture_id)
            if fixture is None or fixture.kind is FixtureKind.SYNTHETIC_EMPTY_OPTIONAL:
                continue

            findings.extend(
                self._required_output_findings(
                    output,
                    fixture,
                    target_tree=target_tree,
                    mapping_table=mapping_table,
                    profile_url=profile_url,
                    map_url=map_url,
                    map_id=map_id,
                    deferred=deferred,
                    profile_id=profile_id,
                )
            )

            outcome = self.matchbox.validate_fhir_resources(output, profile_url)
            if outcome is None:
                # $validate answers 200 even when the resource is invalid
                findings.append(
                    self._demote(
                        ValidationFinding.build(
                            Producer.ENGINE,
                            Stage.VALIDATE,
                            "validate-unavailable",
                            "The server returned no outcome for $validate, so no "
                            "conformance evidence exists for this output.",
                            map_url=map_url,
                            map_id=map_id,
                            profile_url=profile_url,
                            fixture_id=fixture.fixture_id,
                            evidence={"fixture": fixture.label},
                        ),
                        fixture,
                    )
                )
                continue

            validated.append(fixture.fixture_id)
            for issue in _outcome_issues(outcome):
                findings.append(
                    self._validate_finding(
                        issue,
                        fixture,
                        profile_url=profile_url,
                        map_url=map_url,
                        map_id=map_id,
                        owner_override=owner_override,
                        unverified=unverified,
                        verified=verified,
                        required_gaps=required_gaps,
                        output=output,
                        deferred=deferred or set(),
                        deferred_contracts=deferred_contracts or (),
                    )
                )
        return findings, validated

    def _validate_finding(
        self,
        issue: Mapping[str, Any],
        fixture: SourceFixture,
        *,
        profile_url: str,
        map_url,
        map_id,
        owner_override: Optional[ActionOwner],
        unverified: Set[str],
        verified: Optional[Set[str]] = None,
        required_gaps: Optional[Mapping[str, ActionOwner]] = None,
        output: Optional[Mapping[str, Any]] = None,
        deferred: Optional[Set[str]] = None,
        deferred_contracts: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> ValidationFinding:
        """Turn one $validate issue into a classified finding."""

        message = _issue_message(issue)
        code = f"validate:{issue.get('code') or 'processing'}"
        severity = str(issue.get("severity") or "")
        expressions = [str(item) for item in (issue.get("expression") or [])]

        gate: Optional[GateStatus] = None
        demotion: Optional[str] = None

        if severity == "warning" and (Producer.ENGINE, code) in CLASSIFICATION:
            gate = GateStatus.NON_BLOCKING
            demotion = "engine-severity:warning"


        placeholder = self._placeholder_expression(
            expressions, unverified, verified or set()
        )

        slice_required = bool(_REQUIRED_SLICE.search(message))
        demotable = str(issue.get("code")) in _BINDING_ISSUE_CODES or (
            slice_required and str(issue.get("code")) == "structure"
        )
        if placeholder is not None and demotable:
            gate = GateStatus.NON_BLOCKING
            demotion = f"unverified-synthetic-value:{placeholder}"

        owner, unresolvable = _environment_owner(message, owner_override)

        hints = _hints(message)
        from_expression = expression_path(expressions)
        if from_expression:
            hints["path_hint"] = from_expression

        output = output or {}
        deferred_path = _deferred_validation_path(
            expressions,
            hints,
            output,
            deferred or set(),
            deferred_contracts or (),
        )
        if owner is None and deferred_path:
            owner = ActionOwner.ADVISORY
            gate = GateStatus.NON_BLOCKING
            demotion = f"deferred-reference:{deferred_path}"

        if (
            owner is None
            and severity == "fatal"
            and code == "validate:structure"
            and from_expression is None
        ):
            owner = ActionOwner.UNCLASSIFIED

        if owner is None and code == "validate:structure" and required_gaps:
            path = normalize_target_path(str(hints.get("path_hint") or ""))
            owner = required_gaps.get(path)

        evidence = {
            "fixture": fixture.label,
            "expression": expressions,
            **unresolvable,
            **hints,
        }
        if severity == "fatal":
            evidence.update(_fatal_output_evidence(output, fixture, issue))

        finding = ValidationFinding.build(
            Producer.ENGINE,
            Stage.VALIDATE,
            code,
            message,
            owner=owner,
            gate=gate,
            map_url=map_url,
            map_id=map_id,
            profile_url=profile_url,
            fixture_id=fixture.fixture_id,
            severity=severity,
            issue_code=str(issue.get("code") or ""),
            discriminator=self._issue_discriminator(expressions, message),
            evidence=evidence,
        )
        if demotion is not None:
            finding = finding.model_copy(update={"demoted_by": demotion})
        return self._demote(finding, fixture)

    @staticmethod
    def _issue_discriminator(expressions: Sequence[str], message: str) -> str:
        """identify differences between two same-coded $validate issues"""

        payload = json.dumps(
            {"expression": sorted(expressions), "message": message},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _required_gap_owners(
        target_tree,
        mapping_table: Optional[Mapping[str, Any]],
        profile_id: Optional[str],
    ) -> Dict[str, ActionOwner]:
        """Who owns an unmet requirement, per normalized target path"""

        if target_tree is None:
            return {}
        declared = set(
            declared_target_paths(
                mapping_table,
                res_type=getattr(target_tree, "res_type", None),
                profile_id=profile_id,
            )
        )
        owners: Dict[str, ActionOwner] = {}
        for entry in target_tree.required_manifest():
            raw = str(entry.get("path") or "")
            if not raw:
                continue
            owner = required_gap_owner(entry, declared)

            for spelling in target_path_spellings(raw) | {
                _concrete_choice_name(raw, name) for name in (
                    target_tree.effective_types(raw) or []
                )
            }:
                if spelling:
                    owners.setdefault(spelling, owner)
        return owners

    @staticmethod
    def _target_value_provenance(
        fixtures: Sequence[SourceFixture],
        mapping_table: Optional[Mapping[str, Any]],
    ) -> Tuple[Set[str], Set[str]]:
        """target paths partitioned by synthetic-value provenance"""

        from mapping.rule_ir import mapping_target_path  # noqa: PLC0415

        verified = {
            field for fixture in fixtures for field in fixture.verified_fields
        }
        unverified: Set[str] = set()
        verified_targets: Set[str] = set()
        for source_field, value in (mapping_table or {}).items():
            leaf = str(source_field).rsplit(".", 1)[-1]
            target = mapping_target_path(value)
            if target:
                spellings = {
                    _root_relative(item)
                    for item in target_path_spellings(str(target))
                }
                if leaf in verified:
                    verified_targets |= spellings
                else:
                    unverified |= spellings
        return unverified, verified_targets

    @staticmethod
    def _unverified_target_paths(
        fixtures: Sequence[SourceFixture],
        mapping_table: Optional[Mapping[str, Any]],
    ) -> Set[str]:
        """Compatibility view of placeholder-fed targets only."""

        return EngineSession._target_value_provenance(fixtures, mapping_table)[0]

    @staticmethod
    def _placeholder_expression(
        expressions: Sequence[str],
        unverified: Set[str],
        verified: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """get placeholder-fed target an issue's expression points at, if any"""

        verified = verified or set()
        for expression in expressions:
            path = normalize_expression(expression)
            for relative in _container_spellings(_root_relative(path)):
                if relative in unverified:
                    return path
                placeholder_descendants = {
                    item for item in unverified if item.startswith(relative + ".")
                }
                verified_overlap = any(
                    item == relative
                    or item.startswith(relative + ".")
                    or relative.startswith(item + ".")
                    for item in verified
                )
                if placeholder_descendants and not verified_overlap:
                    return path
        return None

    def _required_output_findings(
        self,
        output: Mapping[str, Any],
        fixture: SourceFixture,
        *,
        target_tree,
        mapping_table: Optional[Mapping[str, Any]],
        profile_url: str,
        map_url,
        map_id,
        deferred: Optional[Set[str]] = None,
        profile_id: Optional[str] = None,
    ) -> List[ValidationFinding]:
        """required elements the profile demands and the output does not have"""

        if target_tree is None:
            return []

        from data_handling.instance_validation import extract_path  # noqa: PLC0415

        declared = set(
            declared_target_paths(
                mapping_table,
                res_type=getattr(target_tree, "res_type", None),
                profile_id=profile_id,
            )
        )

        findings: List[ValidationFinding] = []
        for entry in target_tree.required_manifest():
            if not entry.get("active"):
                continue
            raw = str(entry.get("path") or "")
            path = normalize_target_path(raw)
            if not path or "." not in path:
                continue
            if any(
                extract_path(dict(output), candidate)
                for candidate in _output_path_candidates(raw, target_tree, output)
            ):
                continue
            if path in (deferred or set()):
                findings.append(
                    self._demote(
                        ValidationFinding.build(
                            Producer.ENGINE,
                            Stage.VALIDATE,
                            "output-required-deferred",
                            f"The transform output has no {path}; this map "
                            "leaves that reference to the bundle assembler.",
                            map_url=map_url,
                            map_id=map_id,
                            profile_url=profile_url,
                            path=path,
                            fixture_id=fixture.fixture_id,
                            evidence={
                                "fixture": fixture.label,
                                "element_id": entry.get("id"),
                                "resolved_by": "bundle-assembler",
                            },
                        ),
                        fixture,
                    )
                )
                continue
            findings.append(
                self._demote(
                    ValidationFinding.build(
                        Producer.ENGINE,
                        Stage.VALIDATE,
                        "output-required-missing",
                        f"The transform output has no {path}, which the profile "
                        f"requires (min = {entry.get('min')}).",
                        owner=required_gap_owner(entry, declared),
                        gate=GateStatus.BLOCKING,
                        map_url=map_url,
                        map_id=map_id,
                        profile_url=profile_url,
                        path=path,
                        fixture_id=fixture.fixture_id,
                        evidence={
                            "fixture": fixture.label,
                            "element_id": entry.get("id"),
                            "provider": entry.get("provider"),
                        },
                    ),
                    fixture,
                )
            )
        return findings

    @staticmethod
    def _demote(finding: ValidationFinding, fixture: SourceFixture) -> ValidationFinding:
        """Strip gate relevance from findings produced by an experimental fixture"""

        if fixture.gate_relevant:
            return finding
        return finding.demote(f"experimental-fixture:{fixture.kind.value}")

    def expand(self, value_set_url: str) -> Optional[List[str]]:
        """expand Codes for a required binding, or None when nothing can expand it.

        :meth:`_unverified_target_paths` already accounts for.
        """

        codes = self._expand_via_terminology(value_set_url)
        if codes:
            return codes
        try:
            response = self.matchbox.mc.send_request(
                "ValueSet/$expand", "GET", params={"url": value_set_url}
            )
        except Exception as exc:
            logger.debug("$expand failed for %s (%s)", value_set_url, exc)
            return None
        return _expansion_codes(response)

    def _expand_via_terminology(self, value_set_url: str) -> Optional[List[str]]:
        """run $expand against the configured terminology server"""

        if not self.terminology_url:
            return None
        import requests  # noqa: PLC0415

        endpoint = self.terminology_url.rstrip("/") + "/ValueSet/$expand"
        try:
            response = requests.get(
                endpoint,
                params={"url": value_set_url},
                headers={"Accept": "application/fhir+json"},
                timeout=15,
            )
            response.raise_for_status()
            return _expansion_codes(response.json())
        except Exception as exc:
            logger.debug(
                "terminology $expand failed for %s at %s (%s)",
                value_set_url,
                endpoint,
                exc,
            )
            return None
