"""Report canonicals that more than one installed package defines.

Matchbox resolves a canonical to whichever package it currently considers
authoritative, not to the version a reference pins. So two packages defining the
same URL is not a tie broken in the reference's favour — it is a silent choice,
and the losing project's maps fail at `$transform` for a reason that has nothing
to do with them.

This is how the IPS results were invalidated twice. `testing_ips` ships a
14-profile subset registered as `hl7.fhir.uv.ips@2.0.1`; installing
`ca.on.oh.patient-summary`, which declares `hl7.fhir.uv.ips: 1.0.0` as a
dependency, pulled a competing copy back in and `CodeableConcept-uv-ips`
resolved to the wrong one.

Two collision sources are reported:

`package`   two locally installed npm packages define the same canonical.
`project`   a project's own `input_profile/` defines a canonical that an
            installed package also defines — the subset-under-an-upstream-identity
            case.

Usage:
    python scripts/check_package_conflicts.py
    python scripts/check_package_conflicts.py --project testing_ips
    python scripts/check_package_conflicts.py --matchbox   # also list installed IGs
"""

import argparse
import collections
import json

from pathlib import Path
from typing import Dict, Set

_REPO = Path(__file__).resolve().parent.parent
_PACKAGES = _REPO / "data" / "resource_cache" / "local_packages" / "node_modules"

#: Conformance resources whose canonical is what a map or profile resolves.
_CANONICAL_TYPES = {"StructureDefinition", "ValueSet", "CodeSystem", "ConceptMap"}


def _index(directory: Path) -> Dict[str, str]:
    """``{canonical: resourceType}`` for one package or profile folder."""
    found: Dict[str, str] = {}
    for path in directory.glob("*.json"):
        if path.name in ("package.json", ".index.json"):
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        if document.get("resourceType") in _CANONICAL_TYPES and document.get("url"):
            found[str(document["url"]).split("|", 1)[0]] = str(document["resourceType"])
    return found


def package_index() -> Dict[str, Dict[str, str]]:
    """``{package name: {canonical: type}}`` over every installed package."""
    packages: Dict[str, Dict[str, str]] = {}
    if not _PACKAGES.is_dir():
        return packages
    for base in sorted(_PACKAGES.iterdir()):
        if not base.is_dir():
            continue
        root = base / "package" if (base / "package").is_dir() else base
        entries = _index(root)
        if entries:
            packages[base.name] = entries
    return packages


def project_index(projects: list) -> Dict[str, Dict[str, str]]:
    """``{project: {canonical: type}}`` over each project's `input_profile/`."""
    out: Dict[str, Dict[str, str]] = {}
    for name in projects:
        folder = _REPO / "projects" / name / "input_profile"
        if folder.is_dir():
            out[name] = _index(folder)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", default=[])
    parser.add_argument(
        "--matchbox", action="store_true", help="also list IGs installed in Matchbox"
    )
    args = parser.parse_args()

    packages = package_index()
    names = args.project or sorted(
        p.name for p in (_REPO / "projects").glob("testing_*") if p.is_dir()
    )
    projects = project_index(names)

    owners: Dict[str, Set[str]] = collections.defaultdict(set)
    for package, entries in packages.items():
        for url in entries:
            owners[url].add(package)

    package_clashes = {u: o for u, o in owners.items() if len(o) > 1}
    print(f"installed packages : {len(packages)}")
    print(f"projects inspected : {len(projects)}")
    print(f"\ncanonicals defined by more than one installed package: {len(package_clashes)}")
    for url, who in sorted(package_clashes.items())[:20]:
        print(f"  {url}\n      {', '.join(sorted(who))}")

    print("\nproject definitions that an installed package also defines:")
    total = 0
    for project, entries in sorted(projects.items()):
        overlap = sorted(url for url in entries if url in owners)
        if not overlap:
            continue
        total += len(overlap)
        print(f"  {project}: {len(overlap)} canonical(s) also in "
              f"{', '.join(sorted({p for url in overlap for p in owners[url]}))}")
        for url in overlap[:5]:
            print(f"      {url}")
    if not total:
        print("  none")

    if args.matchbox:
        import urllib.request  # noqa: PLC0415

        url = "http://localhost:8080/matchboxv3/fhir/ImplementationGuide?_count=300"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                body = json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 - a probe, not a dependency
            print(f"\nMatchbox not reachable ({exc}).")
            return
        igs = sorted(
            (e["resource"].get("packageId"), e["resource"].get("version"))
            for e in body.get("entry") or []
        )
        duplicated = collections.Counter(pid for pid, _ in igs)
        print(f"\nIGs installed in Matchbox: {len(igs)}")
        for pid, count in duplicated.most_common():
            if count > 1:
                versions = [v for p, v in igs if p == pid]
                print(f"  {pid} installed {count}x: {', '.join(versions)}  <-- ambiguous")


if __name__ == "__main__":
    main()
