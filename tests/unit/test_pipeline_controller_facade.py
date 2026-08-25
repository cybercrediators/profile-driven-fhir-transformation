"""Facade tests for PipelineController.

The controller is a thin composition root + facade: it wires the services and
delegates public methods to them, and exposes shared runtime state via properties
backed by PipelineState. These tests bypass __init__ (which needs config/matchbox)
and inject stub services to verify only the wiring/delegation/state contracts.
"""

from types import SimpleNamespace

import pytest

from controller.pipeline_controller.pipeline_controller import PipelineController
from controller.pipeline_controller.pipeline_build_service import BuildOptions
from controller.pipeline_controller.pipeline_state import PipelineState


pytestmark = pytest.mark.unit


def bare_controller():
    """A PipelineController with stub services, without running __init__."""
    pc = PipelineController.__new__(PipelineController)
    pc.state = PipelineState()
    pc.build_service = SimpleNamespace(
        options=BuildOptions(project_name="proj"),
        last_generation_result=None,
    )
    pc.matchbox_sync = SimpleNamespace()
    pc.transform_service = SimpleNamespace()
    pc.validation_service = SimpleNamespace()
    pc.instance_validator = SimpleNamespace()
    return pc


# --------------------------------------------------------------------------- #
# delegation: method on controller -> method on the right service
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "service_attr, method, call",
    [
        ("build_service", "initial_processing", lambda pc: pc.initial_processing()),
        ("build_service", "run_process", lambda pc: pc.run_process()),
        ("build_service", "run_source_def", lambda pc: pc.run_source_def()),
        ("build_service", "run_static_gen_sm", lambda pc: pc.run_static_gen_sm()),
        ("build_service", "_ensure_processed", lambda pc: pc._ensure_processed()),
        ("matchbox_sync", "sync_changed_resources", lambda pc: pc.sync_changed_resources()),
        ("matchbox_sync", "check_matchbox_connection", lambda pc: pc.check_matchbox_connection()),
        ("transform_service", "transform_data", lambda pc: pc.transform_data({"r": 1})),
        ("transform_service", "transform_data_batch", lambda pc: pc.transform_data_batch([])),
        ("validation_service", "validate_data", lambda pc: pc.validate_data({}, "u")),
        ("validation_service", "validate_local_files", lambda pc: pc.validate_local_files()),
        ("validation_service", "run_validate", lambda pc: pc.run_validate("f")),
        ("instance_validator", "run_instance_validation", lambda pc: pc.run_instance_validation()),
    ],
)
def test_delegates_to_service(service_attr, method, call):
    pc = bare_controller()
    sentinel = object()
    setattr(getattr(pc, service_attr), method, lambda *a, **k: sentinel)
    assert call(pc) is sentinel


def test_prepare_matchbox_setup_forwards_flags():
    pc = bare_controller()
    seen = {}
    pc.matchbox_sync.prepare_matchbox_setup = lambda force_upload=False, read_only=False: seen.update(
        force_upload=force_upload, read_only=read_only
    ) or "ok"
    assert pc.prepare_matchbox_setup(force_upload=True, read_only=True) == "ok"
    assert seen == {"force_upload": True, "read_only": True}


def test_validate_setup_forwards_read_only():
    pc = bare_controller()
    seen = {}
    pc.validation_service.validate_setup = lambda read_only=True: seen.update(read_only=read_only) or True
    assert pc.validate_setup(read_only=False) is True
    assert seen == {"read_only": False}


def test_run_instance_validation_forwards_kwargs():
    pc = bare_controller()
    seen = {}
    pc.instance_validator.run_instance_validation = lambda **kwargs: seen.update(kwargs) or "rep"
    pc.run_instance_validation(examples_dir="d", direct_only=True, from_element_examples=True, report_path="r")
    assert seen == {
        "examples_dir": "d",
        "direct_only": True,
        "from_element_examples": True,
        "report_path": "r",
    }


# --------------------------------------------------------------------------- #
# shared-state properties (backed by PipelineState)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prop", ["source_helper_urls", "structure_map_urls", "custom_mapping_table"])
def test_state_properties_proxy_pipeline_state(prop):
    pc = bare_controller()
    # getter reflects state
    assert getattr(pc, prop) == getattr(pc.state, prop)
    # setter writes through to state
    setattr(pc, prop, ["sentinel"])
    assert getattr(pc.state, prop) == ["sentinel"]


def test_input_source_example_proxies_build_options():
    pc = bare_controller()
    assert pc.input_source_example == ""  # BuildOptions default
    # this is exactly what `pipeline source-def --source-data` does
    pc.input_source_example = "override.json"
    assert pc.build_service.options.input_source_example == "override.json"
    assert pc.input_source_example == "override.json"


def test_last_generation_result_proxies_build_service():
    result = object()
    pc = bare_controller()
    pc.build_service.last_generation_result = result

    assert pc.last_generation_result is result
