"""Unit tests for the MatchboxCLI view layer (view/matchbox_cli.py).

MatchboxController and MatchboxTransformService are monkeypatched at the module
level so no network/matchbox server is ever contacted.
"""

import json

import pytest

import view.matchbox_cli as matchbox_cli_module
from view.matchbox_cli import MatchboxCLI

pytestmark = pytest.mark.unit


class FakeMatchboxController:
    """Stand-in for MatchboxController; records calls, returns configurable results."""

    def __init__(self, conn):
        self.conn = conn
        self.capability_ok = True
        self.install_result = {"resourceType": "OperationOutcome"}
        self.ig_installed = True
        self.upload_result = True
        self.resource_by_id = {"resourceType": "Patient", "id": "1"}
        self.resource_by_url = {"resourceType": "Patient", "id": "2"}
        self.validation_result = {"resourceType": "OperationOutcome"}
        self.calls = []

    def get_capability_statement(self):
        return self.capability_ok

    def install_npm_package(self, **kwargs):
        self.calls.append(("install_npm_package", kwargs))
        return self.install_result

    def check_implementation_guide_installed(self, ig_url=None, ig_id=None):
        self.calls.append(("check_installed_ig", ig_url, ig_id))
        return self.ig_installed

    def upload_structure_definition(self, resource_json):
        self.calls.append(("upload_sd", resource_json))
        return self.upload_result

    def upload_structure_map(self, resource_json):
        self.calls.append(("upload_sm", resource_json))
        return self.upload_result

    def upload_concept_map(self, resource_json):
        self.calls.append(("upload_cm", resource_json))
        return self.upload_result

    def get_resource_by_id(self, resource_type=None, resource_id=None):
        self.calls.append(("get_by_id", resource_type, resource_id))
        return self.resource_by_id

    def get_resource_by_url(self, resource_type=None, resource_url=None):
        self.calls.append(("get_by_url", resource_type, resource_url))
        return self.resource_by_url

    def validate_fhir_resources(self, res_obj=None, profile_url=None):
        self.calls.append(("validate", res_obj, profile_url))
        return self.validation_result


class FakeMatchboxTransformService:
    def __init__(
        self, mc, project_path, plugins=None, external_reference_defaults=None
    ):
        self.mc = mc
        self.project_path = project_path
        self.plugins = plugins or []
        self.external_reference_defaults = external_reference_defaults or []
        self.transform_result = {"resourceType": "Bundle"}

    def transform(self, input_data, structure_map_url, bundle=False, batch=False):
        self.last_call = (input_data, structure_map_url, bundle, batch)
        return self.transform_result


@pytest.fixture
def fake_mc(monkeypatch):
    controller_holder = {}

    def _make(conn):
        controller = FakeMatchboxController(conn)
        controller_holder["instance"] = controller
        return controller

    monkeypatch.setattr(matchbox_cli_module, "MatchboxController", _make)
    return controller_holder


@pytest.fixture
def fake_transform_service(monkeypatch):
    service_holder = {}

    def _make(
        mc, project_path, plugins=None, external_reference_defaults=None
    ):
        service = FakeMatchboxTransformService(
            mc,
            project_path,
            plugins=plugins,
            external_reference_defaults=external_reference_defaults,
        )
        service_holder["instance"] = service
        return service

    monkeypatch.setattr(matchbox_cli_module, "MatchboxTransformService", _make)
    return service_holder


def make_cli(conf=None):
    conf = conf if conf is not None else {"matchbox_connection": {"url": "http://mb"}}
    return MatchboxCLI(conf)


# --------------------------------------------------------------------------- #
# handle_matchbox_command: connection / ping / dispatch guards
# --------------------------------------------------------------------------- #


def test_handle_matchbox_command_no_connection_config_exits():
    cli = make_cli(conf={})
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("install-package")
    assert exc.value.code == 1


def test_handle_matchbox_command_ping_failure_exits(monkeypatch):
    class FailingController(FakeMatchboxController):
        def __init__(self, conn):
            super().__init__(conn)
            self.capability_ok = False

    monkeypatch.setattr(matchbox_cli_module, "MatchboxController", FailingController)
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("install-package")
    assert exc.value.code == 1


def test_handle_matchbox_command_unknown_action_exits(fake_mc):
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("not-a-real-action")
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# install-package
# --------------------------------------------------------------------------- #


def test_install_package_success_prints_response(fake_mc, capsys):
    cli = make_cli()
    cli.handle_matchbox_command(
        "install-package", package_name="my.package", package_version="1.0.0"
    )
    controller = fake_mc["instance"]
    assert controller.calls[0][0] == "install_npm_package"
    assert controller.calls[0][1]["package_name"] == "my.package"
    assert '"resourceType": "OperationOutcome"' in capsys.readouterr().out


def test_install_package_failure_exits(monkeypatch):
    class FailingInstallController(FakeMatchboxController):
        def __init__(self, conn):
            super().__init__(conn)
            self.install_result = None

    monkeypatch.setattr(matchbox_cli_module, "MatchboxController", FailingInstallController)

    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("install-package")
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# check-installed-ig
# --------------------------------------------------------------------------- #


def test_check_installed_ig_installed_exits_zero(fake_mc):
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("check-installed-ig", ig_url="http://ig")
    assert exc.value.code == 0
    assert fake_mc["instance"].calls == [("check_installed_ig", "http://ig", None)]


def test_check_installed_ig_not_installed_exits_zero(monkeypatch):
    class NotInstalledController(FakeMatchboxController):
        def __init__(self, conn):
            super().__init__(conn)
            self.ig_installed = False

    monkeypatch.setattr(matchbox_cli_module, "MatchboxController", NotInstalledController)
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("check-installed-ig", ig_id="my.ig")
    assert exc.value.code == 0


# --------------------------------------------------------------------------- #
# upload-sd / upload-sm / upload-cm
# --------------------------------------------------------------------------- #


def test_upload_sd_success(fake_mc, tmp_path):
    resource = {"resourceType": "StructureDefinition", "id": "x"}
    path = tmp_path / "sd.json"
    path.write_text(json.dumps(resource))

    cli = make_cli()
    cli.handle_matchbox_command("upload-sd", sd_path=str(path))

    controller = fake_mc["instance"]
    assert controller.calls[0] == ("upload_sd", resource)


def test_upload_sm_missing_path_exits(fake_mc):
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("upload-sm", sm_path=None)
    assert exc.value.code == 1


def test_upload_cm_failure_exits(monkeypatch, tmp_path):
    class FailingUploadController(FakeMatchboxController):
        def __init__(self, conn):
            super().__init__(conn)
            self.upload_result = False

    monkeypatch.setattr(matchbox_cli_module, "MatchboxController", FailingUploadController)
    resource = {"resourceType": "ConceptMap"}
    path = tmp_path / "cm.json"
    path.write_text(json.dumps(resource))

    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("upload-cm", cm_path=str(path))
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# transform-data
# --------------------------------------------------------------------------- #


def test_transform_data_success_prints_output(fake_mc, fake_transform_service, tmp_path, capsys):
    input_data = {"resourceType": "Patient"}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(input_data))

    cli = make_cli()
    cli.handle_matchbox_command(
        "transform-data",
        input_file=str(path),
        structure_map_url="http://sm",
        bundle=True,
        batch=False,
    )

    service = fake_transform_service["instance"]
    assert service.last_call == (input_data, "http://sm", True, False)
    assert '"resourceType": "Bundle"' in capsys.readouterr().out


def test_transform_data_passes_external_reference_defaults(
    fake_mc, fake_transform_service, tmp_path
):
    input_data = {"resourceType": "Observation"}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(input_data))
    defaults = [
        {
            "path": "Observation.subject",
            "reference": "Patient/eval-patient-1",
        }
    ]

    cli = make_cli(
        {
            "matchbox_connection": {"url": "http://mb"},
            "external_reference_defaults": defaults,
        }
    )
    cli.handle_matchbox_command(
        "transform-data",
        input_file=str(path),
        structure_map_url="http://sm",
        bundle=True,
    )

    assert fake_transform_service["instance"].external_reference_defaults == defaults


def test_transform_data_writes_output_file(fake_mc, fake_transform_service, tmp_path):
    input_data = {"resourceType": "Patient"}
    in_path = tmp_path / "input.json"
    in_path.write_text(json.dumps(input_data))
    out_path = tmp_path / "output.json"

    cli = make_cli()
    cli.handle_matchbox_command(
        "transform-data",
        input_file=str(in_path),
        structure_map_url="http://sm",
        output_file=str(out_path),
    )

    assert json.loads(out_path.read_text()) == {"resourceType": "Bundle"}


def test_transform_data_missing_input_file_exits(fake_mc):
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command(
            "transform-data", input_file=None, structure_map_url="http://sm"
        )
    assert exc.value.code == 1


def test_transform_data_none_result_exits(monkeypatch, tmp_path):
    class NoneTransformService:
        def __init__(
            self, mc, project_path, plugins=None, external_reference_defaults=None
        ):
            pass

        def transform(self, *a, **kw):
            return None

    monkeypatch.setattr(matchbox_cli_module, "MatchboxTransformService", NoneTransformService)

    class OKController(FakeMatchboxController):
        pass

    monkeypatch.setattr(matchbox_cli_module, "MatchboxController", OKController)

    input_data = {"resourceType": "Patient"}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(input_data))

    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command(
            "transform-data", input_file=str(path), structure_map_url="http://sm"
        )
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# get-resource
# --------------------------------------------------------------------------- #


def test_get_resource_by_id(fake_mc, capsys):
    cli = make_cli()
    cli.handle_matchbox_command(
        "get-resource", resource_type="Patient", resource_id="1"
    )
    controller = fake_mc["instance"]
    assert controller.calls == [("get_by_id", "Patient", "1")]
    assert '"id": "1"' in capsys.readouterr().out


def test_get_resource_by_url(fake_mc, capsys):
    cli = make_cli()
    cli.handle_matchbox_command(
        "get-resource", resource_type="Patient", resource_url="http://x/p"
    )
    controller = fake_mc["instance"]
    assert controller.calls == [("get_by_url", "Patient", "http://x/p")]


def test_get_resource_missing_type_exits(fake_mc):
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("get-resource", resource_id="1")
    assert exc.value.code == 1


def test_get_resource_missing_identifier_exits(fake_mc):
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("get-resource", resource_type="Patient")
    assert exc.value.code == 1


def test_get_resource_not_found_prints_nothing(monkeypatch, capsys):
    class EmptyResourceController(FakeMatchboxController):
        def __init__(self, conn):
            super().__init__(conn)
            self.resource_by_id = None

    monkeypatch.setattr(matchbox_cli_module, "MatchboxController", EmptyResourceController)
    cli = make_cli()
    cli.handle_matchbox_command("get-resource", resource_type="Patient", resource_id="1")
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# validate-data
# --------------------------------------------------------------------------- #


def test_validate_data_success(fake_mc, tmp_path, capsys):
    input_data = {"resourceType": "Patient"}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(input_data))

    cli = make_cli()
    cli.handle_matchbox_command(
        "validate-data", input_file=str(path), profile_url="http://profile"
    )
    controller = fake_mc["instance"]
    assert controller.calls == [("validate", input_data, "http://profile")]
    assert '"resourceType": "OperationOutcome"' in capsys.readouterr().out


def test_validate_data_missing_input_file_exits(fake_mc):
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("validate-data", input_file=None, profile_url="http://x")
    assert exc.value.code == 1


def test_validate_data_missing_profile_url_exits(fake_mc, tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"resourceType": "Patient"}))
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command("validate-data", input_file=str(path), profile_url=None)
    assert exc.value.code == 1


def test_validate_data_failure_exits(monkeypatch, tmp_path):
    class FailingValidateController(FakeMatchboxController):
        def __init__(self, conn):
            super().__init__(conn)
            self.validation_result = None

    monkeypatch.setattr(matchbox_cli_module, "MatchboxController", FailingValidateController)
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"resourceType": "Patient"}))

    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli.handle_matchbox_command(
            "validate-data", input_file=str(path), profile_url="http://profile"
        )
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# _read_json_or_exit / _emit direct tests
# --------------------------------------------------------------------------- #


def test_read_json_or_exit_missing_path_exits():
    cli = make_cli()
    with pytest.raises(SystemExit) as exc:
        cli._read_json_or_exit(None, "input data")
    assert exc.value.code == 1


def test_emit_prints_when_no_output_file(capsys):
    cli = make_cli()
    cli._emit({"a": 1}, None)
    assert '"a": 1' in capsys.readouterr().out


def test_emit_writes_file_when_output_file_given(tmp_path):
    cli = make_cli()
    out_path = tmp_path / "out.json"
    cli._emit({"a": 1}, str(out_path))
    assert json.loads(out_path.read_text()) == {"a": 1}
