"""Set up a bridge project from a Simplifier/npm FHIR package or project endpoint.

- point at package (or simplifier project slug) and project name
- create conf/<project>.json
- creates a project folder and initializes it
- possible sources: published npm package, or simplifier project

Usage:
    python scripts/setup_simplifier_project.py --package de.basisprofil.r4 --version 1.5.3 \
        --project basisprofil
    python scripts/setup_simplifier_project.py --endpoint-project menstrual-bleeding \
        --project hmb
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_PREFIX = _REPO / "data" / "resource_cache" / "local_packages"
_CONF_TEMPLATE = _REPO / "conf" / "testing_uscore.json"
_DEFAULT_VALIDATOR = _REPO / "data" / "tools" / "validator_cli.jar"
_SIMPLIFIER_FHIR = "https://fhir.simplifier.net"

# message envelopes — profiled containers, not standalone validatable resources
_WRAPPER_TYPES = {"Bundle", "MessageHeader", "Parameters", "OperationOutcome"}

# resource types that are package apparatus, not example instances
_APPARATUS = {
    "StructureDefinition",
    "ValueSet",
    "CodeSystem",
    "ConceptMap",
    "ImplementationGuide",
    "CapabilityStatement",
    "SearchParameter",
    "OperationDefinition",
    "NamingSystem",
    "CompartmentDefinition",
    "MessageDefinition",
    "GraphDefinition",
    "TerminologyCapabilities",
}


def find_package_dir(package: str, prefix: Path) -> Path:
    """Locate the installed package root (the folder holding package.json)."""
    base = prefix / "node_modules" / package
    if (base / "package" / "package.json").exists():
        return base / "package"
    if (base / "package.json").exists():
        return base
    raise FileNotFoundError(f"Package not installed under {base}")


def install_package(package: str, version: str, prefix: Path, registry: str) -> None:
    """Install the package via npm from the given registry (idempotent)."""
    prefix.mkdir(parents=True, exist_ok=True)
    command = [
        "npm",
        "--registry",
        registry,
        "install",
        f"{package}@{version}",
        "--prefix",
        str(prefix),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print("Error: 'npm' command was not found. Install it!")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error while installing {package}@{version}!")
        print(e.stderr[-2000:] if e.stderr else e)
        sys.exit(1)


def load_resources(package_path: Path) -> list:
    """Read every JSON resource in the package root as (path, dict) pairs."""
    out = []
    for f in sorted(package_path.glob("*.json")):
        if f.name in ("package.json", ".index.json"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and data.get("resourceType"):
            out.append((f, data))
    return out


def load_examples(package_path: Path) -> list:
    """Example instances — packages ship them at the root or in example[s]/."""
    out = [
        (f, r)
        for f, r in load_resources(package_path)
        if r.get("resourceType") not in _APPARATUS
    ]
    for sub in ("example", "examples"):
        if (package_path / sub).is_dir():
            out.extend(load_resources(package_path / sub))
    return out


def resource_profiles(resources: list, include_wrappers: bool = False) -> list:
    """Constraint StructureDefinitions on resources, sorted by canonical URL."""
    hits = [
        (f, r)
        for f, r in resources
        if r.get("resourceType") == "StructureDefinition"
        and r.get("kind") == "resource"
        and r.get("derivation") == "constraint"
        and (include_wrappers or r.get("type") not in _WRAPPER_TYPES)
    ]
    return sorted(hits, key=lambda pair: pair[1].get("url", ""))


def referenced_canonicals(resource: dict) -> set:
    """Every canonical URL a StructureDefinition points at, anywhere inside it.

    """
    keys = {"baseDefinition", "profile", "targetProfile", "valueSet", "url"}
    found = set()

    def walk(node, key=None):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for item in node:
                walk(item, key)
        elif isinstance(node, str) and key in keys and node.startswith("http"):
            found.add(node.split("|", 1)[0])

    # `url` is a canonical only on a reference; on the resource itself it is the
    # identity, and including it would make every profile depend on itself.
    walk({k: v for k, v in resource.items() if k != "url"})
    return found


def is_mappable_profile(resource: dict) -> bool:
    """Whether `resource` would become a mapping target if it were staged.
    """
    return (
        resource.get("resourceType") == "StructureDefinition"
        and resource.get("kind") == "resource"
        and resource.get("derivation") == "constraint"
    )


def dependency_closure(staged: list, resources: list) -> list:
    """Supporting definitions inside the same package that *staged* needs.

    """
    by_url = {
        r.get("url"): (f, r)
        for f, r in resources
        if r.get("resourceType") in ("StructureDefinition", "ValueSet", "CodeSystem")
        and r.get("url")
        and not is_mappable_profile(r)
    }
    staged_urls = {r.get("url") for _, r in staged}

    closure: dict = {}
    frontier = [r for _, r in staged]
    while frontier:
        nxt = []
        for resource in frontier:
            for url in referenced_canonicals(resource):
                if url in staged_urls or url in closure or url not in by_url:
                    continue
                f, dep = by_url[url]
                closure[url] = (f, dep)
                nxt.append(dep)
        frontier = nxt
    return [closure[url] for url in sorted(closure)]


def fetch_endpoint_package(slug: str, workdir: Path) -> Path:
    """Materialize a Simplifier project's profiles as a package-shaped folder.

    """
    workdir.mkdir(parents=True, exist_ok=True)
    written = 0
    for rtype in ("StructureDefinition", "ValueSet", "CodeSystem"):
        url = f"{_SIMPLIFIER_FHIR}/{slug}/{rtype}?_count=200"
        req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            bundle = json.load(resp)
        for entry in bundle.get("entry", []):
            res = entry.get("resource") or {}
            if not res.get("id"):
                continue
            (workdir / f"{rtype}-{res['id']}.json").write_text(
                json.dumps(res, indent=2), encoding="utf-8"
            )
            written += 1
    (workdir / "package.json").write_text(
        json.dumps(
            {
                "name": f"{slug}.local",
                "version": "0.0.0",
                "fhirVersions": ["4.0.1"],
                "description": f"Pulled from {_SIMPLIFIER_FHIR}/{slug} (no published package)",
                "dependencies": {"hl7.fhir.r4.core": "4.0.1"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Fetched {written} resources from {_SIMPLIFIER_FHIR}/{slug}", file=sys.stderr)
    return workdir


def generate_snapshots(
    stage: Path,
    validator_jar: Path,
    context_igs: Optional[list] = None,
    fhir_version: str = "4.0.1",
) -> dict:
    """Fill in snapshots for differential-only SDs via the reference validator.

    """
    # match on content, not filename — packages name their SD files freely
    pending = []
    for f in sorted(stage.glob("*.json")):
        if f.name == "package.json":
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("resourceType") == "StructureDefinition" and not data.get("snapshot"):
            pending.append(f)
    if not pending:
        return {"generated": 0, "pending": 0, "validator": None}
    if not validator_jar.is_file():
        raise FileNotFoundError(
            f"validator jar not found at {validator_jar} — download a pinned "
            "validator_cli.jar release or pass --validator-jar"
        )
    total = len(pending)
    version_line, failures = "", {}
    remaining, generated = list(pending), 0
    while remaining:
        cmd = ["java", "-jar", str(validator_jar), "snapshot"]
        cmd += [f.name for f in remaining]
        for ig in context_igs or []:
            cmd += ["-ig", str(ig)]
        cmd += ["-version", fhir_version, "-tx", "n/a", "-outputSuffix", "snap.json"]
        proc = subprocess.run(cmd, cwd=stage, capture_output=True, text=True)
        version_line = version_line or next(
            (ln for ln in proc.stdout.splitlines() if "Validation tool Version" in ln), ""
        )
        progressed = []
        for f in remaining:
            produced = stage / f"{f.name}.snap.json"
            if produced.is_file() and json.loads(produced.read_text(encoding="utf-8")).get(
                "snapshot"
            ):
                shutil.move(str(produced), str(f))
                generated += 1
                progressed.append(f)
        remaining = [f for f in remaining if f not in progressed]
        if not progressed:
            reason = next(
                (
                    ln.strip()
                    for ln in proc.stdout.splitlines()
                    if "Exception generating snapshot" in ln or "No base profile" in ln
                ),
                "unknown",
            )
            # the head of the batch is the one that aborted it; drop and retry the rest
            failures[remaining[0].name] = reason
            remaining = remaining[1:]
    return {
        "generated": generated,
        "pending": total,
        "failed": failures,
        "validator": version_line.strip() or None,
    }


def write_conf(project: str, force: bool) -> Path:
    """Write conf/<project>.json from the corpus template with project paths set."""
    conf_path = _REPO / "conf" / f"{project}.json"
    if conf_path.exists() and not force:
        raise FileExistsError(f"{conf_path} exists (use --force)")
    conf = json.loads(_CONF_TEMPLATE.read_text(encoding="utf-8"))
    conf["project_path"] = f"{_REPO / 'projects' / project}/"
    conf["resource_cache_path"] = f"{_REPO / 'data' / 'resource_cache'}/"
    conf_path.write_text(json.dumps(conf, indent=2) + "\n", encoding="utf-8")
    return conf_path


def setup(
    package_path: Path,
    project: str,
    force: bool,
    include_wrappers: bool = False,
    validator_jar: Path = _DEFAULT_VALIDATOR,
) -> dict:
    """Create conf + project folder, populate input_profile/ and examples/."""
    resources = load_resources(package_path)
    staged = resource_profiles(resources, include_wrappers)
    if not staged:
        raise RuntimeError(
            f"no resource profiles found in {package_path} "
            "(constraint StructureDefinitions with kind='resource')"
        )

    conf_path = write_conf(project, force)

    supporting = dependency_closure(staged, resources)

    # stage the profiles so the real `init` code path does the copying
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy(package_path / "package.json", stage / "package.json")
        for f, _ in list(staged) + supporting:
            shutil.copy(f, stage / f.name)
        deps = [
            find_package_dir(name, _DEFAULT_PREFIX)
            for name in json.loads(
                (package_path / "package.json").read_text(encoding="utf-8")
            ).get("dependencies", {})
            if (_DEFAULT_PREFIX / "node_modules" / name).exists()
        ]
        snapshots = generate_snapshots(stage, validator_jar, context_igs=deps)
        subprocess.run(
            [sys.executable, "src/main.py", "-c", str(conf_path), "init", str(stage)],
            cwd=_REPO,
            check=True,
            env={"PYTHONPATH": ".:src", "PATH": "/usr/bin:/bin"},
        )

    examples_dir = _REPO / "projects" / project / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    staged_urls = {r.get("url") for _, r in staged}
    all_examples = load_examples(package_path)
    matching = [
        (f, r)
        for f, r in all_examples
        # meta.profile may carry a |version suffix (us.core does, MII does not)
        if staged_urls
        & {p.split("|")[0] for p in r.get("meta", {}).get("profile", [])}
    ]
    for f, _ in matching:
        shutil.copy(f, examples_dir / f.name)

    return {
        "conf": str(conf_path),
        "project_dir": str(_REPO / "projects" / project),
        "profiles_staged": [r.get("url") for _, r in staged],
        "supporting_definitions": [r.get("url") for _, r in supporting],
        "examples_copied": len(matching),
        "examples_in_package": len(all_examples),
        "snapshots": snapshots,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Set up a bridge project from an npm/Simplifier FHIR package."
    )
    parser.add_argument("--package", help="npm package name")
    parser.add_argument("--version", help="package version")
    parser.add_argument(
        "--endpoint-project",
        help="Simplifier project slug to pull profiles from (for projects with no npm package)",
    )
    parser.add_argument("--project", required=True, help="project name")
    parser.add_argument("--prefix", type=Path, default=_DEFAULT_PREFIX)
    parser.add_argument("--registry-url", default="https://packages.simplifier.net")
    parser.add_argument("--force", action="store_true", help="overwrite an existing conf")
    parser.add_argument(
        "--include-wrappers",
        action="store_true",
        help="also stage Bundle/MessageHeader/Parameters/OperationOutcome message envelopes",
    )
    parser.add_argument("--validator-jar", type=Path, default=_DEFAULT_VALIDATOR)
    args = parser.parse_args()

    if args.endpoint_project:
        cache = _DEFAULT_PREFIX / "_endpoint" / args.endpoint_project
        package_path = fetch_endpoint_package(args.endpoint_project, cache)
    else:
        if not (args.package and args.version):
            parser.error("--package and --version are required without --endpoint-project")
        try:
            package_path = find_package_dir(args.package, args.prefix)
        except FileNotFoundError:
            print(f"Installing {args.package}@{args.version} …", file=sys.stderr)
            install_package(args.package, args.version, args.prefix, args.registry_url)
            package_path = find_package_dir(args.package, args.prefix)

    report = setup(
        package_path,
        args.project,
        args.force,
        args.include_wrappers,
        args.validator_jar,
    )
    report["source"] = str(package_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
