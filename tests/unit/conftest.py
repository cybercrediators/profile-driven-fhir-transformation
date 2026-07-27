"""Shared fakes/fixtures for controller unit tests.

These keep unit tests free of any running matchbox server, cache backend, or real
project IO. Services under test take a real ``AppState`` + ``PipelineState`` with
fakes injected for the cache, dataIO and matchbox controller.
"""

import json
from types import SimpleNamespace

import pytest

from data_handling.app_state import AppState
from data_handling.data_io import DataIO
from controller.pipeline_controller.pipeline_state import PipelineState


class FakeCache:
    """In-memory stand-in for the cache connector, keyed by resource url."""

    def __init__(self, initial=None):
        self.store = dict(initial or {})

    def get_resource_from_cache(self, url):
        return self.store.get(url)

    def add_resource_to_cache(self, resource):
        if isinstance(resource, dict) and resource.get("url"):
            self.store[resource["url"]] = resource


class FakeDataIO:
    """Configurable stand-in for DataIO returning preset file lists."""

    ProjectFolders = DataIO.ProjectFolders

    def __init__(
        self,
        project_dir,
        *,
        profile_files=None,
        sm_files=None,
        cm_files=None,
        helper_files=None,
        package_name="my.test.ig",
        package_version="1.0.0",
        tarred="/tmp/pkg.tgz",
        processed=True,
    ):
        self.project_dir = project_dir
        self._profile_files = profile_files or []
        self._sm_files = sm_files or []
        self._cm_files = cm_files or []
        self._helper_files = helper_files or []
        self._package_name = package_name
        self._package_version = package_version
        self._tarred = tarred
        self._processed = processed
        self.stored = []  # records store_project_file calls

    def get_json_profile_files(self, profile_path):
        return list(self._profile_files)

    def get_structure_map_files(self):
        return list(self._sm_files)

    def get_concept_map_files(self):
        return list(self._cm_files)

    def get_source_helper_map_files(self):
        return list(self._helper_files)

    def get_package_name(self):
        return self._package_name

    def get_package_version(self):
        return self._package_version

    def find_tarred_profile(self):
        return self._tarred

    def store_project_file(self, folder, name, content, mode="STR", overwrite=False):
        self.stored.append((folder, name, content, mode, overwrite))

    def check_processed_resources(self):
        return self._processed

    def check_processed_helper_maps(self):
        return self._processed

    def check_processed_structure_maps(self):
        return self._processed


class FakeMatchbox:
    """Stand-in for MatchboxController that records uploads and returns preset data."""

    def __init__(
        self,
        *,
        present_urls=None,
        ig_installed=True,
        transform_result=None,
        validate_outcome=None,
        capability=True,
    ):
        self.present = set(present_urls or [])
        self.ig_installed = ig_installed
        self.transform_result = transform_result
        self.validate_outcome = validate_outcome
        self.capability = capability
        self.uploaded = []  # list of (resource_type, payload)
        self.installed_packages = []
        self.transform_calls = []
        self.validate_calls = []

    def get_capability_statement(self):
        return {"resourceType": "CapabilityStatement"} if self.capability else None

    def check_implementation_guide_installed(self, ig_url=None, ig_id=None):
        return self.ig_installed

    def install_npm_package(self, **kwargs):
        self.installed_packages.append(kwargs)
        return {"ok": True}

    def get_resource_by_url(self, resource_type, url):
        if url in self.present:
            return {"resourceType": resource_type, "url": url}
        return None

    def upload_structure_definition(self, sd):
        self.uploaded.append(("StructureDefinition", sd))

    def upload_structure_map(self, sm):
        self.uploaded.append(("StructureMap", sm))

    def upload_concept_map(self, cm):
        self.uploaded.append(("ConceptMap", cm))

    def transform_data(self, source_obj, structure_map_url):
        self.transform_calls.append((source_obj, structure_map_url))
        if callable(self.transform_result):
            return self.transform_result(source_obj, structure_map_url)
        return self.transform_result

    def validate_fhir_resources(self, res_obj, profile_url):
        self.validate_calls.append((res_obj, profile_url))
        if callable(self.validate_outcome):
            return self.validate_outcome(res_obj, profile_url)
        return self.validate_outcome


@pytest.fixture
def fake_cache():
    return FakeCache()


@pytest.fixture
def fake_dataio(tmp_path):
    """Factory for a FakeDataIO rooted at tmp_path."""

    def _make(**kwargs):
        return FakeDataIO(tmp_path, **kwargs)

    return _make


@pytest.fixture
def fake_matchbox():
    """Factory for a FakeMatchbox."""

    def _make(**kwargs):
        return FakeMatchbox(**kwargs)

    return _make


@pytest.fixture
def state():
    return PipelineState()


@pytest.fixture
def make_app_state(tmp_path):
    """Factory: build a real AppState with fakes injected."""

    def _make(*, conf=None, dataIO=None, cache=None, registry=None):
        return AppState(
            conf=conf if conf is not None else {},
            registry=registry or SimpleNamespace(registry_objects={}),
            cache=cache or FakeCache(),
            dataIO=dataIO or FakeDataIO(tmp_path),
        )

    return _make


@pytest.fixture
def write_json(tmp_path):
    """Factory: write a dict to tmp as JSON and return its Path (so utils.get_json reads it)."""

    def _write(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return path

    return _write
