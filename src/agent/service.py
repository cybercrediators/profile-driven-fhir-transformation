"""controller to handle connections and ingestion from projects and langgraph
loop"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.context import AgentContext
from agent.engine import EngineSession
from agent.fixtures import SourceFixture, build_example_fixtures, build_shared_fixtures
from agent.loop import FEATURE, LoopLimits, Proposal
from agent.validation import (
    ActionOwner,
    GateStatus,
    Producer,
    Stage,
    ValidationFinding,
    ValidationReport,
    build_target_tree,
    declared_target_paths,
    declared_target_structure,
    findings_from_diagnostics,
    merge_engine_result,
    target_path_spellings,
    prohibited_target_path,
    validate_offline,
)

logger = logging.getLogger(__name__)

# threshold for serialization of structuremaps in prompts
PRETTY_MAP_BUDGET_CHARS = 20000


@dataclass
class ProjectContext:
    """the project artifacts agent mode reads, resolved once (read only)"""

    project_dir: Path
    conf: Dict[str, Any]
    structure_maps: List[Tuple[Path, Dict[str, Any]]] = field(default_factory=list)
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    concept_maps: List[Dict[str, Any]] = field(default_factory=list)

    questionnaires: List[Dict[str, Any]] = field(default_factory=list)
    source_model: Optional[Dict[str, Any]] = None
    mapping_table: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    coverage_report: Optional[Any] = None
    matchbox_controller: Optional[Any] = None
    
    _external_profiles: Dict[str, Optional[Dict[str, Any]]] = field(
        default_factory=dict, repr=False
    )

    def select(self, selector: Optional[str]) -> Tuple[Path, Dict[str, Any]]:
        """resolve given maps based on the given selector (e.g. map url or id)"""

        if not selector:
            if len(self.structure_maps) == 1:
                return self.structure_maps[0]
            raise AgentSetupError(
                f"The project has {len(self.structure_maps)} StructureMaps; name "
                "one by path, canonical URL, or id."
            )

        matches = [
            (path, document)
            for path, document in self.structure_maps
            if str(path) == selector
            or path.name == selector
            or document.get("url") == selector
            or document.get("id") == selector
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise AgentSetupError(f"No StructureMap in the project matches {selector!r}.")
        raise AgentSetupError(
            f"{selector!r} matches {len(matches)} StructureMaps: "
            + ", ".join(str(path) for path, _ in matches)
        )

    def profile_for(self, document: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """target profile a map declares (e.g. in targetProfile)"""

        for entry in document.get("structure") or []:
            if not isinstance(entry, Mapping) or entry.get("mode") != "target":
                continue
            url = str(entry.get("url") or "").split("|", 1)[0]
            profile = self.profiles.get(url) or self._resolve_external_profile(url)
            if profile is not None:
                return profile
        return None

    def _resolve_external_profile(self, url: str) -> Optional[Dict[str, Any]]:
        """resolve/loasd StructureDefinition from the disk cache or an installed package"""

        if url in self._external_profiles:
            return self._external_profiles[url]

        candidates: List[Path] = []
        cache_root = str(self.conf.get("resource_cache_path") or "")
        if cache_root:
            flattened = url.replace("://", "_").replace("/", "_")
            candidates.append(Path(cache_root) / f"{flattened}.json")
            packages = Path(cache_root) / "local_packages" / "node_modules"
            leaf = url.rsplit("/", 1)[-1]
            if packages.is_dir():
                candidates.extend(packages.glob(f"*/StructureDefinition-{leaf}.json"))
                candidates.extend(
                    packages.glob(f"*/package/StructureDefinition-{leaf}.json")
                )

        found: Optional[Dict[str, Any]] = None
        for path in candidates:
            document = _load_json(path) if path.exists() else None
            if (
                document
                and document.get("resourceType") == "StructureDefinition"
                and str(document.get("url") or "").split("|", 1)[0] == url
            ):
                found = document
                logger.debug("Resolved target profile %s from %s", url, path)
                break
        if found is None:
            logger.info(
                "No StructureDefinition available for target profile %s; this map "
                "cannot be graded against a profile.",
                url,
            )
        self._external_profiles[url] = found
        return found


class AgentSetupError(RuntimeError):
    """the project or configuration cannot support an agent run: error definition"""


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Skipping unreadable %s (%s: %s)", path, type(exc).__name__, exc)
        return None
    return data if isinstance(data, dict) else None


def build_project_context(conf: Mapping[str, Any]) -> ProjectContext:
    """resolve a project through the same DataIO the pipeline uses

    :raises AgentSetupError: when the project has no StructureMaps to repair.
    """

    from data_handling.data_io import DataIO  # noqa: PLC0415

    project_path = conf.get("project_path") or conf.get("profile_path") or ""
    data_io = DataIO(str(project_path))

    structure_maps: List[Tuple[Path, Dict[str, Any]]] = []
    for path in sorted(data_io.get_structure_map_files() or [], key=str):
        document = _load_json(Path(path))
        if document and document.get("resourceType") == "StructureMap":
            structure_maps.append((Path(path), document))
    if not structure_maps:
        raise AgentSetupError(
            f"No StructureMaps found under {data_io.project_dir}; run the pipeline "
            "before agent mode."
        )

    profiles: Dict[str, Dict[str, Any]] = {}
    questionnaires: List[Dict[str, Any]] = []
    for path in data_io.get_json_profile_files(str(conf.get("profile_path", ""))) or []:
        document = _load_json(Path(path))
        if not document:
            continue
        if document.get("resourceType") == "StructureDefinition":
            url = str(document.get("url") or "")
            if url:
                profiles[url] = document
        elif document.get("resourceType") == "Questionnaire":
            questionnaires.append(document)

    source_model = None
    for path in sorted(data_io.get_source_helper_map_files() or [], key=str):
        document = _load_json(Path(path))
        if document and document.get("resourceType") == "StructureDefinition":
            source_model = document
            break

    concept_maps = [
        document
        for path in sorted(data_io.get_concept_map_files() or [], key=str)
        if (document := _load_json(Path(path)))
        and document.get("resourceType") == "ConceptMap"
    ]

    examples = [
        document
        for path in sorted(data_io.get_example_files() or [], key=str)
        if (document := _load_json(Path(path)))
        and document.get("resourceType")
        and document.get("resourceType") != "StructureDefinition"
    ]

    return ProjectContext(
        project_dir=Path(data_io.project_dir),
        conf=dict(conf),
        structure_maps=structure_maps,
        profiles=profiles,
        concept_maps=concept_maps,
        questionnaires=questionnaires,
        source_model=source_model,
        mapping_table=_load_mapping_table(conf, data_io),
        examples=examples,
        coverage_report=_load_coverage_report(data_io),
        matchbox_controller=_build_matchbox(conf),
    )


def _load_mapping_table(conf: Mapping[str, Any], data_io) -> Dict[str, Any]:
    """load project source-to-target table, from config object or path"""

    value = conf.get("mapping_table") or conf.get("custom_mapping_table")
    if isinstance(value, dict):
        return dict(value)
    path = conf.get("mapping_table_path") or conf.get("custom_mapping_table_path")
    if path:
        loaded = _load_json(Path(str(path)))
        if loaded:
            return loaded
    default = Path(data_io.project_dir) / "source_data" / "mapping_table.json"
    if default.is_file():
        return _load_json(default) or {}
    return {}


def _load_coverage_report(data_io):
    """load the coverage report (if it exists)"""

    from mapping.generation_result import CoverageReport  # noqa: PLC0415

    source_data = Path(data_io.project_dir) / "source_data"
    for path in sorted(source_data.glob("*_coverage.json")):
        raw = _load_json(path)
        if raw:
            try:
                return CoverageReport.from_raw(raw)
            except Exception as exc:
                logger.warning(
                    "Coverage report %s could not be read (%s: %s).",
                    path,
                    type(exc).__name__,
                    exc,
                )
    return None


def _build_matchbox(conf: Mapping[str, Any]):
    """create/load matchbox connection through matchbox controller"""
    from controller.external_services.matchbox_controller import (  # noqa: PLC0415
        MatchboxController,
    )

    connection = conf.get("matchbox_connection") or {}
    if not connection:
        logger.warning(
            "No matchbox_connection in configuration — engine validation will be "
            "unavailable, which agent mode treats as a failure, not a pass."
        )
        return None
    try:
        return MatchboxController(connection)
    except Exception as exc:
        logger.warning(
            "Matchbox controller unavailable (%s: %s).", type(exc).__name__, exc
        )
        return None


class MapEvaluator:
    """handle map validation for one maps baseline and corresponding candidates"""

    def __init__(
        self,
        project: ProjectContext,
        document: Mapping[str, Any],
        *,
        engine: Optional[EngineSession] = None,
        use_examples: bool = False,
        require_engine: bool = True,
    ) -> None:
        self.project = project
        self.profile = project.profile_for(document)
        self.profile_url = str(self.profile.get("url")) if self.profile else None
        
        self.profile_id = str(self.profile.get("id")) if self.profile else None
        self.target_tree = build_target_tree(self.profile) if self.profile else None
        
        self._setup_findings: List[Any] = []
        declared_target = declared_target_structure(document)
        if self.profile is None and declared_target:
            self._setup_findings.append(
                ValidationFinding.build(
                    Producer.PATH_RESOLUTION,
                    Stage.PATHS,
                    "target-profile-unavailable",
                    f"The target profile {declared_target} is not available from "
                    "the project, its resource cache, or any installed package, "
                    "so this map was not graded against a profile.",
                    owner=ActionOwner.ENVIRONMENT,
                    gate=GateStatus.BLOCKING,
                    map_url=document.get("url"),
                    map_id=document.get("id"),
                )
            )
        self.require_engine = require_engine
        self.use_examples = use_examples
        self.engine = engine
        self._bootstrap_findings: List[Any] = []
        if self.engine is None and require_engine and project.matchbox_controller:
            self.engine = EngineSession(
                project.matchbox_controller,
                
                terminology_url=project.conf.get("terminology_server_uri"),
            )
            self._bootstrap_findings = self.engine.bootstrap(
                source_model=project.source_model,
                profiles=[self.profile] if self.profile else [],
                concept_maps=project.concept_maps,
                questionnaires=getattr(project, "questionnaires", ()),
            )
        self._fixtures: List[SourceFixture] = []
        self._fixture_documents: List[Mapping[str, Any]] = []

    @property
    def source_field_specs(self) -> List[Dict[str, Any]]:
        from agent.fixtures import source_field_specs  # noqa: PLC0415

        if not self.project.source_model:
            return []
        return source_field_specs(self.project.source_model)

    def _fixtures_for(self, documents: Sequence[Mapping[str, Any]]) -> List[SourceFixture]:
        if not self.project.source_model:
            return []
        fixtures = build_shared_fixtures(
            self.project.source_model,
            list(documents),
            concept_maps=self.project.concept_maps,
            target_tree=self.target_tree,
            mapping_table=self.project.mapping_table,
            expand=self.engine.expand if self.engine is not None else None,
        )
        if self.use_examples:
            fixtures.extend(
                build_example_fixtures(
                    self.project.examples, self.project.mapping_table
                )
            )
        return fixtures

    def _offline(self, document: Mapping[str, Any]) -> ValidationReport:
        """retrieve offline validation report"""
        report = validate_offline(
            document,
            target_tree=self.target_tree,
            mapping_table=self.project.mapping_table,
            source_fields={spec["name"] for spec in self.source_field_specs},
            profile_url=self.profile_url,
            profile_id=self.profile_id,
        )
        
        report.findings.extend(self._setup_findings)
        return report

    def _with_engine(
        self, document: Mapping[str, Any], fixtures: Sequence[SourceFixture]
    ) -> ValidationReport:
        """check which validation were requested"""
        report = self._offline(document)
        if self.engine is None:
            if self.require_engine:
                report.engine_requested = True
                report.engine_available = False
            return report
        report.findings.extend(self._bootstrap_findings)
        return merge_engine_result(
            report,
            self.engine.validate_map(
                document,
                list(fixtures),
                target_tree=self.target_tree,
                profile_url=self.profile_url,
                profile_id=self.profile_id,
                mapping_table=self.project.mapping_table,
            ),
        )

    def evaluate(self, document: Mapping[str, Any]) -> ValidationReport:
        """execute validation based on selected methods and given document"""
        self._fixture_documents = [document]
        self._fixtures = self._fixtures_for(self._fixture_documents)
        return self._with_engine(document, self._fixtures)

    def evaluate_pair(
        self, baseline: Mapping[str, Any], candidate: Mapping[str, Any]
    ) -> Tuple[ValidationReport, ValidationReport]:
        """return evaluation for baseline vs. candidate pairs"""
        self._fixture_documents = [baseline, candidate]
        self._fixtures = self._fixtures_for(self._fixture_documents)
        return (
            self._with_engine(baseline, self._fixtures),
            self._with_engine(candidate, self._fixtures),
        )


class LLMProposer:
    """turns an :class:`~agent.context.AgentContext` into an AgentPatch"""

    def __init__(self, client, model_settings) -> None:
        self.client = client
        self.model_settings = model_settings

    @staticmethod
    def _messages(context: AgentContext):
        import json as _json  # noqa: PLC0415

        from llm.models import LLMMessage  # noqa: PLC0415
        from llm.prompts import AGENT_FIX_SYSTEM, AGENT_FIX_USER, render  # noqa: PLC0415

        excerpt = _json.dumps(context.map_excerpt, indent=2, default=str)
        if len(excerpt) > PRETTY_MAP_BUDGET_CHARS:
            excerpt = _json.dumps(
                context.map_excerpt, separators=(",", ":"), default=str
            )
            logger.debug(
                "Map excerpt serialized compactly (%d chars).", len(excerpt)
            )
        user = render(AGENT_FIX_USER, context=context, map_json=excerpt)
        messages = [
            LLMMessage.system(render(AGENT_FIX_SYSTEM)),
            LLMMessage.user(user),
        ]
        return user, messages

    def prompt_chars(self, context: AgentContext) -> int:
        """Render locally so the loop can enforce its bound before any call."""

        user, _messages = self._messages(context)
        return len(user)

    def propose(self, context: AgentContext) -> Proposal:
        from llm.errors import LLMError, LLMResponseValidationError  # noqa: PLC0415
        from pydantic import ValidationError  # noqa: PLC0415

        from agent.models import ProposedPatch  # noqa: PLC0415

        user, messages = self._messages(context)

        try:
            response = self.client.complete(
                messages=messages,
                response_model=ProposedPatch,
                settings=self.model_settings,
                feature=FEATURE,
                context=context.cache_context(),
            )
        except (LLMResponseValidationError, ValidationError) as exc:
            return Proposal(
                patch=None,
                prompt_chars=len(user),
                error=f"The response did not satisfy the patch envelope: {exc}",
            )
        except LLMError as exc:
            logger.error("Provider failure during agent mode: %s", exc)
            return Proposal(
                patch=None,
                prompt_chars=len(user),
                error=f"The provider could not be used: {exc}",
                fatal=True,
            )

        return Proposal(
            patch=response.value.as_agent_patch(),
            cache_hit=response.cache_hit,
            prompt_chars=len(user),
        )


@dataclass
class AgentFixService:
    """Agent (fix) mode for one project"""

    project: ProjectContext
    proposer: Any
    limits: LoopLimits = field(default_factory=LoopLimits)
    use_examples: bool = False
    require_engine: bool = True

    llm_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        conf: Mapping[str, Any],
        *,
        limits: Optional[LoopLimits] = None,
        offline: bool = False,
        use_examples: bool = False,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "AgentFixService":
        """build the service, fails on a missing dependency or key.

        :param offline: run the deterministic layers only
        :param overrides: per-run LLMSettings field overrides from the CLI
        :raises AgentSetupError: for a project problem.
        :raises llm.errors.LLMError: for a provider problem.
        """

        from dataclasses import replace as _replace  # noqa: PLC0415

        from llm.backend import build_client  # noqa: PLC0415
        from llm.models import LLMSettings  # noqa: PLC0415

        project = build_project_context(conf)
        settings = LLMSettings.from_env(project_path=conf.get("project_path"))
        if overrides:
            settings = _replace(
                settings, **{key: value for key, value in overrides.items() if value}
            )
        client = build_client(settings)
        logger.info(
            "Agent mode ready (provider=%s, model=%s, engine=%s).",
            settings.provider,
            settings.model,
            "off" if offline else "on",
        )
        return cls(
            project=project,
            proposer=LLMProposer(client, settings.model_settings()),
            limits=limits or LoopLimits(),
            use_examples=use_examples,
            require_engine=not offline,
            llm_config={
                "provider": settings.provider,
                "model": settings.model,
                "base_url": settings.base_url,
                "api_key_env": settings.api_key_env,
                "temperature": settings.temperature,
                "max_output_tokens": settings.max_output_tokens,
                "timeout_s": settings.timeout_s,
                "seed": settings.seed,
                "structured_output": settings.structured_output.value,
                "cache_enabled": settings.cache_enabled,
            },
        )

    def evaluator_for(self, document: Mapping[str, Any]) -> MapEvaluator:
        return MapEvaluator(
            self.project,
            document,
            use_examples=self.use_examples,
            require_engine=self.require_engine,
        )

    def generator_findings(self, document: Mapping[str, Any]):
        """Retrieve diagnostics for a given map"""

        report = self.project.coverage_report
        if report is None:
            return []
        profile = self.project.profile_for(document) or {}
        identities = {
            str(value)
            for value in (profile.get("id"), profile.get("url"))
            if value
        }
        identities.update(
            str(entry.get("alias"))
            for entry in document.get("structure") or []
            if isinstance(entry, Mapping)
            and entry.get("mode") == "target"
            and entry.get("alias")
        )
        findings = findings_from_diagnostics(
            report.mapping_diagnostics,
            map_url=document.get("url"),
            map_id=document.get("id"),
            profile=identities or None,
        )
        res_type = str(profile.get("type") or "") or None
        declared = set(
            declared_target_paths(
                self.project.mapping_table,
                res_type=res_type,
                profile_id=str(profile.get("id") or "") or None,
            )
        )
        findings = [
            finding
            for finding in findings
            if not (
                finding.code == "target-modifier-element"
                and finding.path
                and any(
                    candidate == spelling
                    or candidate.startswith(spelling.rstrip(".") + ".")
                    for candidate in declared
                    for spelling in target_path_spellings(finding.path)
                )
            )
        ]
        target_tree = build_target_tree(profile, introspect=False) if profile else None
        return [
            finding.model_copy(
                update={
                    "action_owner": ActionOwner.MAPPING_INPUT_REQUIRED,
                    "evidence": {
                        **finding.evidence,
                        "prohibited_ancestor": prohibited,
                    },
                }
            )
            if (prohibited := prohibited_target_path(target_tree, finding.path or ""))
            else finding
            for finding in findings
        ]

    def targets(
        self, selector: Optional[str] = None, *, all_maps: bool = False
    ) -> Dict[str, Tuple[Path, Dict[str, Any]]]:
        """identify/select the maps which the current run covers, identifiable by stable ids"""

        entries = (
            list(self.project.structure_maps)
            if all_maps
            else [self.project.select(selector)]
        )
        keyed: Dict[str, Tuple[Path, Dict[str, Any]]] = {}
        for path, document in entries:
            path = Path(path)
            key = path.stem
            if key in keyed:
                key = f"{path.parent.name}-{path.stem}"
            if key in keyed:
                raise AgentSetupError(
                    f"Two selected StructureMaps resolve to the same run key {key!r}; "
                    "rename one before repairing the project."
                )
            keyed[key] = (path, document)
        return keyed
