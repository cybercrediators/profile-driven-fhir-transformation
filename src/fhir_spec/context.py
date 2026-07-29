"""FHIR-release selection and data-driven StructureMap capabilities.

The project deliberately keeps the mature ``fhir.resources.R4B`` Pydantic
classes as its in-memory carrier for both supported releases.  This module is
the release boundary: it reads the normative code systems from the selected
FHIR core package and prevents that carrier choice from silently changing the
declared FHIR release.
"""

from __future__ import annotations

from collections.abc import Set
from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator


_SUPPORTED_RELEASES = {
    "4.0": ("R4", "hl7.fhir.r4.core", "4.0.1"),
    "4.3": ("R4B", "hl7.fhir.r4b.core", "4.3.0"),
}

_CAPABILITY_FILES = {
    "source_list_modes": "CodeSystem-map-source-list-mode.json",
    "target_list_modes": "CodeSystem-map-target-list-mode.json",
    "transforms": "CodeSystem-map-transform.json",
    "structure_modes": "CodeSystem-map-model-mode.json",
    "input_modes": "CodeSystem-map-input-mode.json",
    "context_types": "CodeSystem-map-context-type.json",
    "group_type_modes": "CodeSystem-map-group-type-mode.json",
}


def _concept_codes(concepts: list[dict]) -> frozenset[str]:
    codes: set[str] = set()
    pending = list(concepts or [])
    while pending:
        concept = pending.pop()
        code = concept.get("code")
        if isinstance(code, str) and code:
            codes.add(code)
        pending.extend(concept.get("concept") or [])
    return frozenset(codes)


@dataclass(frozen=True)
class StructureMapCapabilities:
    """Normative StructureMap codes loaded from one FHIR core package."""

    source_list_modes: frozenset[str]
    target_list_modes: frozenset[str]
    transforms: frozenset[str]
    structure_modes: frozenset[str]
    input_modes: frozenset[str]
    context_types: frozenset[str]
    group_type_modes: frozenset[str]
    parameter_value_types: frozenset[str]


@dataclass(frozen=True)
class FHIRSpecContext:
    """Immutable selected FHIR release and its normative capability data."""

    release: str
    fhir_version: str
    core_package_name: str
    core_package_version: str
    core_package_path: Path
    capabilities: StructureMapCapabilities

    @classmethod
    def from_core_package(
        cls,
        package_path: str | Path,
        *,
        declared_fhir_version: str | None = None,
    ) -> "FHIRSpecContext":
        package_path = Path(package_path).resolve()
        if package_path.name != "package" and (package_path / "package").is_dir():
            package_path = package_path / "package"
        metadata_path = package_path / "package.json"
        if not metadata_path.is_file():
            raise ValueError(
                f"FHIR core package metadata not found at {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        package_name = metadata.get("name")
        package_version = str(metadata.get("version") or "")
        versions = metadata.get("fhirVersions") or []
        package_fhir_version = str(versions[0]) if versions else package_version
        requested = declared_fhir_version or package_fhir_version
        release, expected_name, canonical_version = _release_details(requested)
        if package_name != expected_name:
            raise ValueError(
                f"FHIR {release} requires core package {expected_name}, "
                f"got {package_name!r}"
            )
        if _release_key(package_fhir_version) != _release_key(requested):
            raise ValueError(
                f"Declared FHIR version {requested} is incompatible with "
                f"{package_name}#{package_version} ({package_fhir_version})"
            )

        values: dict[str, frozenset[str]] = {}
        for field, filename in _CAPABILITY_FILES.items():
            path = package_path / filename
            if not path.is_file():
                raise ValueError(
                    f"FHIR core package is missing StructureMap capability file {filename}"
                )
            resource = json.loads(path.read_text(encoding="utf-8"))
            values[field] = _concept_codes(resource.get("concept") or [])

        structure_map_path = package_path / "StructureDefinition-StructureMap.json"
        if not structure_map_path.is_file():
            raise ValueError(
                "FHIR core package is missing StructureDefinition-StructureMap.json"
            )
        structure_map = json.loads(
            structure_map_path.read_text(encoding="utf-8")
        )
        parameter_types = frozenset(
            type_ref.get("code")
            for element in structure_map.get("snapshot", {}).get("element", [])
            if element.get("path")
            == "StructureMap.group.rule.target.parameter.value[x]"
            for type_ref in element.get("type", [])
            if type_ref.get("code")
        )
        if not parameter_types:
            raise ValueError(
                "FHIR core StructureMap snapshot does not define parameter.value[x]"
            )

        return cls(
            release=release,
            fhir_version=requested or canonical_version,
            core_package_name=expected_name,
            core_package_version=package_version,
            core_package_path=package_path,
            capabilities=StructureMapCapabilities(
                parameter_value_types=parameter_types,
                **values,
            ),
        )

    @classmethod
    def for_project(cls, project_dir: str | Path, conf: dict | None = None):
        """Select R4 by default, or the single release declared by package.json."""

        conf = conf or {}
        project_dir = Path(project_dir)
        package_metadata = project_dir / "input_profile" / "package.json"
        declared = conf.get("fhir_version")
        if package_metadata.is_file():
            metadata = json.loads(package_metadata.read_text(encoding="utf-8"))
            versions = metadata.get("fhirVersions") or []
            release_keys = {_release_key(str(value)) for value in versions}
            if len(release_keys) > 1:
                raise ValueError(
                    "The input package declares multiple incompatible FHIR releases: "
                    + ", ".join(map(str, versions))
                )
            if versions:
                package_declared = str(versions[0])
                if declared and _release_key(str(declared)) != _release_key(
                    package_declared
                ):
                    raise ValueError(
                        f"Configured FHIR version {declared} conflicts with input "
                        f"package version {package_declared}"
                    )
                declared = package_declared
        declared = str(declared or "4.0.1")
        release, package_name, canonical_version = _release_details(declared)

        explicit = conf.get("fhir_core_package_path")
        if explicit:
            return cls.from_core_package(
                explicit, declared_fhir_version=declared
            )

        cache_root = Path.home() / ".fhir" / "packages"
        candidates = [
            cache_root / f"{package_name}#{canonical_version}" / "package",
            cache_root / f"{package_name}#{declared}" / "package",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return cls.from_core_package(
                    candidate, declared_fhir_version=declared
                )
        raise ValueError(
            f"FHIR {release} core package {package_name}#{canonical_version} was "
            "not found. Install it with SUSHI/Firely or set "
            "'fhir_core_package_path'."
        )

    def assert_profile_compatible(self, resource: dict) -> None:
        """Reject explicit metadata or element types from another FHIR release."""

        declared = resource.get("fhirVersion")
        declared_versions = (
            declared if isinstance(declared, list) else [declared]
        )
        for declared_version in declared_versions:
            if declared_version and _release_key(
                str(declared_version)
            ) != _release_key(self.fhir_version):
                raise ValueError(
                    f"{resource.get('url') or resource.get('id') or 'FHIR resource'} "
                    f"declares FHIR {declared_version}, but the project uses "
                    f"{self.fhir_version} ({self.release})"
                )
        if resource.get("resourceType") != "StructureDefinition":
            return
        identity = resource.get("url") or resource.get("id") or "StructureDefinition"
        for element in resource.get("snapshot", {}).get("element", []):
            # Snapshot publishers sometimes retain later-release choice members
            # on a prohibited element.  max=0 makes those types uninhabitable and
            # therefore irrelevant to generated R4 content.
            if element.get("max") == "0":
                continue
            for type_ref in element.get("type") or []:
                code = type_ref.get("code")
                if (
                    not isinstance(code, str)
                    or not code
                    or code.startswith("http://hl7.org/fhirpath/System.")
                ):
                    continue
                definition = self.core_package_path / f"StructureDefinition-{code}.json"
                if not definition.is_file():
                    raise ValueError(
                        f"{identity} uses element type {code!r}, which is not "
                        f"defined by {self.core_package_name}#"
                        f"{self.core_package_version} ({self.release})"
                    )


def _release_key(version: str) -> str:
    parts = version.strip().split(".")
    if len(parts) < 2:
        raise ValueError(f"Unsupported FHIR version {version!r}")
    return ".".join(parts[:2])


def _release_details(version: str) -> tuple[str, str, str]:
    key = _release_key(version)
    try:
        return _SUPPORTED_RELEASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported FHIR version {version!r}; supported releases are "
            "R4 (4.0.x) and R4B (4.3.x)"
        ) from exc


_ACTIVE_CONTEXT: ContextVar[FHIRSpecContext | None] = ContextVar(
    "fhir_spec_context", default=None
)


def activate_fhir_spec_context(context: FHIRSpecContext) -> FHIRSpecContext:
    _ACTIVE_CONTEXT.set(context)
    return context


def get_fhir_spec_context() -> FHIRSpecContext | None:
    return _ACTIVE_CONTEXT.get()


def clear_fhir_spec_context() -> None:
    """Clear the active context (primarily for isolated tests)."""

    _ACTIVE_CONTEXT.set(None)


def ensure_fhir_spec_context(app_state) -> FHIRSpecContext:
    """Select, store, and activate the context for one pipeline state."""

    context = getattr(app_state, "fhir_spec_context", None)
    if context is None:
        context = FHIRSpecContext.for_project(
            app_state.dataIO.project_dir, app_state.conf
        )
        app_state.fhir_spec_context = context
    return activate_fhir_spec_context(context)


def bundled_capabilities(release: str = "R4") -> StructureMapCapabilities:
    """Load the packaged, pinned fallback used before a project is activated."""

    filename = f"structuremap-capabilities-{release.lower()}.json"
    path = Path(__file__).with_name("data") / filename
    values = json.loads(path.read_text(encoding="utf-8"))
    return StructureMapCapabilities(
        **{
            key: frozenset(value)
            for key, value in values.items()
        }
    )


class ActiveCapabilitySet(Set[str]):
    """Set view that follows the active release without module-level constants."""

    def __init__(self, attribute: str):
        self.attribute = attribute
        self.fallback = getattr(bundled_capabilities(), attribute)

    def _values(self) -> frozenset[str]:
        context = get_fhir_spec_context()
        if context is None:
            return self.fallback
        return getattr(context.capabilities, self.attribute)

    def __contains__(self, value: object) -> bool:
        return value in self._values()

    def __iter__(self) -> Iterator[str]:
        return iter(self._values())

    def __len__(self) -> int:
        return len(self._values())

    def __repr__(self) -> str:
        return repr(self._values())

    @classmethod
    def _from_iterable(cls, values):
        # ``collections.abc.Set`` uses this hook for reflected set operations.
        # Their result is an ordinary snapshot, not another active release view.
        return frozenset(values)
