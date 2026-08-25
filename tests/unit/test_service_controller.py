import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

import controller.service_controller as service_controller_module
from controller.service_controller import ServiceController


def make_controller(conf=None, cache_cli=None, matchbox_cli=None):
    controller = ServiceController.__new__(ServiceController)
    controller.conf = conf or {}
    controller._cache_cli = cache_cli or SimpleNamespace()
    controller._matchbox_cli = matchbox_cli or SimpleNamespace()
    return controller


def test_handle_command_delegates_cache_requests():
    calls = []

    def handle_cache_command(action, **kwargs):
        calls.append((action, kwargs))

    controller = make_controller(
        cache_cli=SimpleNamespace(handle_cache_command=handle_cache_command)
    )

    args = Namespace(
        command="cache",
        cache_action="list",
        force=True,
        url="http://example.org/StructureDefinition/patient",
        output="out.json",
        filter="patient",
        file="input.json",
    )

    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)

    assert exc_info.value.code == 0
    assert calls == [
        (
            "list",
            {
                "force": True,
                "res_url": "http://example.org/StructureDefinition/patient",
                "output_path": "out.json",
                "filter_pattern": "patient",
                "file_path": "input.json",
            },
        )
    ]


def test_stop_service_uses_configured_socket(monkeypatch, capsys):
    observed = {}

    class FakeSocketClient:
        def __init__(self, socket_path="/tmp/fsh_nifi_bridge.sock"):
            observed["socket_path"] = socket_path

        def send_request(self, method, params=None, data=None):
            observed["method"] = method
            observed["params"] = params
            observed["data"] = data
            return {"status": "stopped"}

    monkeypatch.setattr(service_controller_module, "SocketClient", FakeSocketClient)

    controller = make_controller(
        conf={"socket_connection": {"path": "/tmp/test.sock"}}
    )

    controller.stop_service()

    captured = capsys.readouterr()
    assert observed == {
        "socket_path": "/tmp/test.sock",
        "method": "stop",
        "params": None,
        "data": None,
    }
    assert "stopped" in captured.out


def test_process_fsh_command_invokes_sushi_controller(tmp_path, monkeypatch):
    """The `process-fsh` command builds a SushiController and runs process_fsh with
    the project path derived from config (sibling of project_path) + the project name.
    (The actual SUSHI run + file copy is covered in test_fsh_connector.)"""
    calls = {}

    class FakeSushi:
        def __init__(self, *args, **kwargs):
            calls["constructed"] = True

        def process_fsh(self, fsh_dir, project_path, project_name=None, snapshots=True):
            calls["args"] = (fsh_dir, project_path, project_name)
            calls["snapshots"] = snapshots

        resolve_fsh_root = staticmethod(lambda p: p)

    monkeypatch.setattr(service_controller_module, "SushiController", FakeSushi)

    projects_root = tmp_path / "projects"
    configured_project = projects_root / "seed-project"
    controller = make_controller(conf={"project_path": str(configured_project)})

    args = Namespace(command="process-fsh", fsh_dir="some/waves-profile", name="my-project")
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)

    assert exc_info.value.code == 0
    assert calls.get("constructed") is True
    assert calls["args"] == (
        str(Path("some/waves-profile").resolve()),
        str(projects_root / "my-project"),
        "my-project",
    )
    assert calls["snapshots"] is True


def test_process_fsh_command_derives_name_from_resolved_root(tmp_path, monkeypatch):
    """passing <root>/input/fsh resolves to the sushi project root, incl. the default name"""
    calls = {}

    class FakeSushi:
        def __init__(self, *args, **kwargs):
            pass

        def process_fsh(self, fsh_dir, project_path, project_name=None, snapshots=True):
            calls["args"] = (fsh_dir, project_path, project_name)

    # real resolver: exercised against an FSH project layout built on disk
    from controller.connector.fsh_connector import SushiController as RealSushi

    FakeSushi.resolve_fsh_root = staticmethod(RealSushi.resolve_fsh_root)
    monkeypatch.setattr(service_controller_module, "SushiController", FakeSushi)

    fsh_root = tmp_path / "waves-profile"
    sources = fsh_root / "input" / "fsh"
    sources.mkdir(parents=True)
    (fsh_root / "sushi-config.yaml").write_text("canonical: http://example.org\n")

    projects_root = tmp_path / "projects"
    controller = make_controller(conf={"project_path": str(projects_root / "seed")})

    args = Namespace(command="process-fsh", fsh_dir=str(sources), name=None)
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)

    assert exc_info.value.code == 0
    assert calls["args"] == (str(fsh_root), str(projects_root / "waves-profile"), "waves-profile")


def test_process_fsh_command_errors_without_sushi_config(tmp_path, monkeypatch):
    class FakeSushi:
        def __init__(self, *args, **kwargs):
            raise AssertionError("SushiController must not be constructed without a project root")

    from controller.connector.fsh_connector import SushiController as RealSushi

    FakeSushi.resolve_fsh_root = staticmethod(RealSushi.resolve_fsh_root)
    monkeypatch.setattr(service_controller_module, "SushiController", FakeSushi)

    bare_dir = tmp_path / "not-a-project"
    bare_dir.mkdir()
    controller = make_controller(conf={"project_path": str(tmp_path / "projects" / "seed")})

    args = Namespace(command="process-fsh", fsh_dir=str(bare_dir), name=None)
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)
    assert exc_info.value.code == 1


# --------------------------------------------------------------------------- #
# __init__
# --------------------------------------------------------------------------- #


def test_init_loads_config_and_builds_clis(monkeypatch, tmp_path):
    conf_path = tmp_path / "config.json"
    conf_path.write_text(json.dumps({"project_path": str(tmp_path / "proj")}))

    built = {}

    class FakeCacheCLI:
        def __init__(self, conf):
            built["cache_conf"] = conf

    class FakeMatchboxCLI:
        def __init__(self, conf):
            built["matchbox_conf"] = conf

    monkeypatch.setattr(service_controller_module, "CacheCLI", FakeCacheCLI)
    monkeypatch.setattr(service_controller_module, "MatchboxCLI", FakeMatchboxCLI)

    controller = ServiceController(conf_path=str(conf_path))

    assert controller.conf == {"project_path": str(tmp_path / "proj")}
    assert built["cache_conf"] is controller.conf
    assert built["matchbox_conf"] is controller.conf
    assert isinstance(controller._cache_cli, FakeCacheCLI)
    assert isinstance(controller._matchbox_cli, FakeMatchboxCLI)


# --------------------------------------------------------------------------- #
# handle_command: matchbox-cli
# --------------------------------------------------------------------------- #


def test_handle_command_matchbox_cli_delegates():
    calls = []

    def handle_matchbox_command(**kwargs):
        calls.append(kwargs)

    controller = make_controller(
        matchbox_cli=SimpleNamespace(handle_matchbox_command=handle_matchbox_command)
    )

    args = Namespace(command="matchbox-cli", matchbox_action="install-package")

    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)

    assert exc_info.value.code == 0
    assert len(calls) == 1
    assert calls[0]["matchbox_action"] == "install-package"
    # all other kwargs default to None/False since not present on args
    assert calls[0]["package_name"] is None
    assert calls[0]["bundle"] is False
    assert calls[0]["batch"] is False


# --------------------------------------------------------------------------- #
# handle_command: client
# --------------------------------------------------------------------------- #


def test_handle_command_client_sends_request_and_prints(monkeypatch, capsys):
    observed = {}

    class FakeSocketClient:
        def __init__(self, socket_path="/tmp/fsh_nifi_bridge.sock"):
            observed["socket_path"] = socket_path

        def send_request(self, method, params=None, data=None):
            observed["method"] = method
            observed["params"] = params
            observed["data"] = data
            return {"status": "ok"}

    monkeypatch.setattr(service_controller_module, "SocketClient", FakeSocketClient)

    controller = make_controller()
    args = Namespace(
        command="client",
        socket_path="/tmp/custom.sock",
        method="status",
        params={"a": 1},
        data=None,
    )

    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)

    assert exc_info.value.code == 0
    assert observed == {
        "socket_path": "/tmp/custom.sock",
        "method": "status",
        "params": {"a": 1},
        "data": None,
    }
    assert "'status': 'ok'" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# handle_command: server
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "action,expected_method",
    [("start", "start_service"), ("stop", "stop_service"), ("status", "service_status")],
)
def test_handle_command_server_dispatches_to_matching_method(action, expected_method):
    controller = make_controller()
    calls = []
    setattr(controller, expected_method, lambda: calls.append(expected_method))

    args = Namespace(command="server", action=action)
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)

    assert exc_info.value.code == 0
    assert calls == [expected_method]


def test_handle_command_server_unknown_action_is_noop():
    controller = make_controller()
    args = Namespace(command="server", action="nonsense")
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)
    assert exc_info.value.code == 0


# --------------------------------------------------------------------------- #
# handle_command: init
# --------------------------------------------------------------------------- #


def test_handle_command_init_copies_json_files_from_directory(tmp_path, caplog):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "StructureDefinition-a.json").write_text("{}")
    (source_dir / "StructureDefinition-b.json").write_text("{}")
    (source_dir / "package.json").write_text("{}")

    project_path = tmp_path / "myproj"
    controller = make_controller(conf={"project_path": str(project_path)})

    args = Namespace(command="init", source=str(source_dir))
    with caplog.at_level("INFO"):
        with pytest.raises(SystemExit) as exc_info:
            controller.handle_command(args)

    assert exc_info.value.code == 0
    input_profile_dir = project_path / "input_profile"
    copied = sorted(p.name for p in input_profile_dir.glob("*.json"))
    assert copied == ["StructureDefinition-a.json", "StructureDefinition-b.json", "package.json"]
    assert "Copied 3 files" in caplog.text


def test_handle_command_init_warns_when_package_json_missing(tmp_path, caplog):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "StructureDefinition-a.json").write_text("{}")

    project_path = tmp_path / "myproj"
    controller = make_controller(conf={"project_path": str(project_path)})

    args = Namespace(command="init", source=str(source_dir))
    with caplog.at_level("WARNING"):
        with pytest.raises(SystemExit):
            controller.handle_command(args)

    assert "No package.json found" in caplog.text


def test_handle_command_init_extracts_tarred_profile(tmp_path, monkeypatch):
    tar_path = tmp_path / "profile.tgz"
    tar_path.write_bytes(b"fake-tar-contents")

    project_path = tmp_path / "myproj"
    controller = make_controller(conf={"project_path": str(project_path)})

    observed = {}

    def fake_add_tarred_profile(self, output_path):
        observed["output_path"] = output_path
        # simulate extraction creating package.json so no warning branch triggers
        (self.project_dir / "input_profile" / "package.json").write_text("{}")

    monkeypatch.setattr(
        service_controller_module.DataIO, "add_tarred_profile", fake_add_tarred_profile
    )

    args = Namespace(command="init", source=str(tar_path))
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)

    assert exc_info.value.code == 0
    assert observed["output_path"] == tar_path.resolve()


def test_handle_command_init_invalid_source_exits(tmp_path):
    bad_source = tmp_path / "notes.txt"
    bad_source.write_text("not a profile")

    project_path = tmp_path / "myproj"
    controller = make_controller(conf={"project_path": str(project_path)})

    args = Namespace(command="init", source=str(bad_source))
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)
    assert exc_info.value.code == 1


# --------------------------------------------------------------------------- #
# _make_pipeline_controller
# --------------------------------------------------------------------------- #


def test_make_pipeline_controller_passes_through_args(monkeypatch):
    captured = {}

    class FakePipelineController:
        def __init__(
            self,
            conf,
            force_overwrite,
            create_references,
            minimal_mode,
            automapping,
            auto_mapping_mode,
        ):
            captured.update(
                conf=conf,
                force_overwrite=force_overwrite,
                create_references=create_references,
                minimal_mode=minimal_mode,
                automapping=automapping,
                auto_mapping_mode=auto_mapping_mode,
            )

    monkeypatch.setattr(
        service_controller_module, "PipelineController", FakePipelineController
    )

    controller = make_controller(conf={"project_path": "/tmp/proj"})
    args = Namespace(
        mapping_table_path="tables/custom.csv",
        force_overwrite=True,
        create_references_in_structure_map=False,
        minimal_structure_map=True,
        auto_mapping=True,
    )
    controller._make_pipeline_controller(args)

    assert captured["conf"]["mapping_table_path"] == "tables/custom.csv"
    assert captured["force_overwrite"] is True
    assert captured["create_references"] is False
    assert captured["minimal_mode"] is True
    assert captured["automapping"] is True
    assert captured["auto_mapping_mode"] == "deterministic"


def test_make_pipeline_controller_defaults_when_args_missing(monkeypatch):
    captured = {}

    class FakePipelineController:
        def __init__(
            self,
            conf,
            force_overwrite,
            create_references,
            minimal_mode,
            automapping,
            auto_mapping_mode,
        ):
            captured.update(
                force_overwrite=force_overwrite,
                create_references=create_references,
                minimal_mode=minimal_mode,
                automapping=automapping,
                auto_mapping_mode=auto_mapping_mode,
            )

    monkeypatch.setattr(
        service_controller_module, "PipelineController", FakePipelineController
    )

    controller = make_controller(conf={})
    controller._make_pipeline_controller(Namespace())

    assert captured == {
        "force_overwrite": False,
        "create_references": True,
        "minimal_mode": False,
        "automapping": False,
        "auto_mapping_mode": "deterministic",
    }


# --------------------------------------------------------------------------- #
# handle_pipeline
# --------------------------------------------------------------------------- #


class FakePipelineController:
    def __init__(self):
        self.calls = []

    def initial_processing(self):
        self.calls.append(("initial_processing",))

    def prepare_matchbox_setup(self, force_upload=False):
        self.calls.append(("prepare_matchbox_setup", force_upload))

    def run_process(self):
        self.calls.append(("run_process",))

    def run_source_def(self):
        self.calls.append(("run_source_def",))

    def run_static_gen_sm(self):
        self.calls.append(("run_static_gen_sm",))

    def run_export_fields(self, output_file=None, roots_only=False):
        self.calls.append(("run_export_fields", output_file, roots_only))

    def run_validate(self, input_file=None, profile_url=None):
        self.calls.append(("run_validate", input_file, profile_url))

    def run_instance_validation(
        self, examples_dir=None, direct_only=False, from_element_examples=False, report_path=None
    ):
        self.calls.append(
            ("run_instance_validation", examples_dir, direct_only, from_element_examples, report_path)
        )


@pytest.fixture
def controller_with_fake_pc(monkeypatch):
    fake_pc = FakePipelineController()
    controller = make_controller()
    monkeypatch.setattr(controller, "_make_pipeline_controller", lambda args: fake_pc)
    return controller, fake_pc


def test_handle_pipeline_run_action_default(controller_with_fake_pc):
    controller, fake_pc = controller_with_fake_pc
    controller.handle_pipeline(Namespace(pipeline_action=None))
    assert fake_pc.calls == [("initial_processing",)]


def test_handle_pipeline_run_action_prepares_matchbox(controller_with_fake_pc):
    controller, fake_pc = controller_with_fake_pc
    controller.handle_pipeline(
        Namespace(pipeline_action="run", prepare_matchbox=True, force_overwrite=True)
    )
    assert fake_pc.calls == [
        ("initial_processing",),
        ("prepare_matchbox_setup", True),
    ]


def test_handle_pipeline_process_action(controller_with_fake_pc):
    controller, fake_pc = controller_with_fake_pc
    controller.handle_pipeline(Namespace(pipeline_action="process"))
    assert fake_pc.calls == [("run_process",)]


def test_handle_pipeline_source_def_action_sets_input_source_example(controller_with_fake_pc):
    controller, fake_pc = controller_with_fake_pc
    controller.handle_pipeline(Namespace(pipeline_action="source-def", source_data="my_source"))
    assert fake_pc.input_source_example == "my_source"
    assert fake_pc.calls == [("run_source_def",)]


def test_handle_pipeline_static_gen_sm_action(controller_with_fake_pc):
    controller, fake_pc = controller_with_fake_pc
    controller.handle_pipeline(Namespace(pipeline_action="static-gen-sm"))
    assert fake_pc.calls == [("run_static_gen_sm",)]


def test_handle_pipeline_prepare_matchbox_action(controller_with_fake_pc):
    controller, fake_pc = controller_with_fake_pc
    controller.handle_pipeline(
        Namespace(pipeline_action="prepare-matchbox", force_overwrite=True)
    )
    assert fake_pc.calls == [("prepare_matchbox_setup", True)]


def test_handle_pipeline_export_fields_action(controller_with_fake_pc):
    controller, fake_pc = controller_with_fake_pc
    controller.handle_pipeline(
        Namespace(pipeline_action="export-fields", output_file="out.csv", roots_only=True)
    )
    assert fake_pc.calls == [("run_export_fields", "out.csv", True)]


def test_handle_pipeline_validate_action(controller_with_fake_pc):
    controller, fake_pc = controller_with_fake_pc
    controller.handle_pipeline(
        Namespace(pipeline_action="validate", input_file="in.json", profile_url="http://x")
    )
    assert fake_pc.calls == [("run_validate", "in.json", "http://x")]


def test_handle_pipeline_validate_instances_action(controller_with_fake_pc):
    controller, fake_pc = controller_with_fake_pc
    controller.handle_pipeline(
        Namespace(
            pipeline_action="validate-instances",
            examples_dir="examples/",
            direct_only=True,
            from_element_examples=False,
            report="report.json",
        )
    )
    assert fake_pc.calls == [
        ("run_instance_validation", "examples/", True, False, "report.json")
    ]


def test_handle_command_pipeline_dispatches(monkeypatch):
    controller = make_controller()
    calls = []
    monkeypatch.setattr(controller, "handle_pipeline", lambda args: calls.append(args))
    args = Namespace(command="pipeline", pipeline_action="process")
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)
    assert exc_info.value.code == 0
    assert calls == [args]


# --------------------------------------------------------------------------- #
# handle_command: development / show_config / fallback
# --------------------------------------------------------------------------- #


def test_handle_command_development_flag_runs_pipeline_setup(monkeypatch, capsys):
    controller = make_controller()

    class FakePC:
        def initial_processing(self):
            pass

        def validate_setup(self):
            return "ok"

    monkeypatch.setattr(controller, "_make_pipeline_controller", lambda args: FakePC())
    args = Namespace(command="something-else", development=True)
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)
    assert exc_info.value.code == 0
    assert "Setup validation: ok" in capsys.readouterr().out


def test_handle_command_show_config_flag_displays_config(monkeypatch):
    controller = make_controller()
    calls = []

    class FakePC:
        def display_config(self):
            calls.append("displayed")

    monkeypatch.setattr(controller, "_make_pipeline_controller", lambda args: FakePC())
    args = Namespace(command="something-else", development=False, show_config=True)
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)
    assert exc_info.value.code == 0
    assert calls == ["displayed"]


def test_handle_command_unknown_command_logs_and_exits():
    controller = make_controller()
    args = Namespace(command="bogus", development=False, show_config=False)
    with pytest.raises(SystemExit) as exc_info:
        controller.handle_command(args)
    assert exc_info.value.code == 0


# --------------------------------------------------------------------------- #
# start_service / service_status
# --------------------------------------------------------------------------- #


def test_start_service_starts_socket_server(monkeypatch):
    observed = {}

    class FakeSocketServer:
        def __init__(self, conf):
            observed["conf"] = conf

        def start_server(self):
            observed["started"] = True

    monkeypatch.setattr(service_controller_module, "SocketServer", FakeSocketServer)

    controller = make_controller(conf={"foo": "bar"})
    controller.start_service()

    assert observed == {"conf": {"foo": "bar"}, "started": True}


def test_service_status_uses_configured_socket(monkeypatch, capsys):
    observed = {}

    class FakeSocketClient:
        def __init__(self, socket_path="/tmp/fsh_nifi_bridge.sock"):
            observed["socket_path"] = socket_path

        def send_request(self, method, params=None, data=None):
            observed["method"] = method
            return {"status": "running"}

    monkeypatch.setattr(service_controller_module, "SocketClient", FakeSocketClient)

    controller = make_controller(conf={"socket_connection": {"path": "/tmp/test.sock"}})
    controller.service_status()

    assert observed == {"socket_path": "/tmp/test.sock", "method": "status"}
    assert "running" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# __getattr__ delegation
# --------------------------------------------------------------------------- #


def test_getattr_delegates_to_matchbox_cli_when_not_on_cache_cli():
    controller = make_controller(
        matchbox_cli=SimpleNamespace(some_matchbox_method=lambda: "mb-result")
    )
    assert controller.some_matchbox_method() == "mb-result"


def test_getattr_raises_when_not_found_anywhere():
    controller = make_controller()
    with pytest.raises(AttributeError):
        controller.totally_unknown_attribute


# --------------------------------------------------------------------------- #
# Agent mode (WP8)
# --------------------------------------------------------------------------- #


def test_handle_agent_routes_fix_to_the_agent_cli(monkeypatch):
    """The agent package is imported inside the branch, never at module scope.

    Anything else would put the optional LLM dependencies into the import graph
    of every deterministic command.
    """
    import sys
    from types import ModuleType

    seen = {}

    def fake_run(args, conf):
        seen["args"] = args
        seen["conf"] = conf
        return 0

    module = ModuleType("agent.cli")
    module.run_agent_fix = fake_run
    monkeypatch.setitem(sys.modules, "agent.cli", module)

    controller = make_controller(conf={"project_path": "projects/demo"})
    args = Namespace(agent_action="fix", mapping_table_path="")

    assert controller.handle_agent(args) == 0
    assert seen["conf"]["project_path"] == "projects/demo"


def test_handle_agent_rejects_an_unknown_subcommand():
    controller = make_controller()
    assert controller.handle_agent(Namespace(agent_action="invent")) == 2


def test_handle_agent_honours_an_overridden_mapping_table(monkeypatch):
    import sys
    from types import ModuleType

    captured = {}
    module = ModuleType("agent.cli")
    module.run_agent_fix = lambda args, conf: captured.setdefault("conf", conf) and 0
    monkeypatch.setitem(sys.modules, "agent.cli", module)

    controller = make_controller(conf={})
    controller.handle_agent(
        Namespace(agent_action="fix", mapping_table_path="/tmp/table.json")
    )
    assert controller.conf["mapping_table_path"] == "/tmp/table.json"


def test_ordinary_commands_load_no_agent_or_provider_package():
    """Import isolation, proven in a subprocess rather than asserted.

    A stale ``sys.modules`` entry left by an earlier test in this process would
    make an in-process check pass while the real CLI still imported the world.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    probe = (
        "import sys; sys.path.insert(0, 'src');\n"
        "sys.argv = ['prog', 'pipeline', 'run'];\n"
        "from view.options import get_args; get_args();\n"
        "from controller.service_controller import ServiceController;\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'agent', 'openai', 'jinja2', 'jsonpatch', 'langchain', 'langgraph',"
        " 'langgraph_checkpoint_sqlite', 'langgraph_sdk', 'aiosqlite'});\n"
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"leaked modules: {result.stdout.strip()}"


def test_the_deterministic_pipeline_imports_on_a_core_only_install():
    """WP9: no optional dependency may be needed to run a normal pipeline.

    The isolation probe above shows the extras are not *loaded*; this shows they
    are not needed. A module that imports `jinja2` at the top of a file the
    pipeline touches would pass the first check on a machine where the extra
    happens to be installed, and fail on the user's.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    probe = (
        "import sys;\n"
        "blocked = {'openai', 'jinja2', 'jsonpatch', 'langgraph',"
        " 'langgraph_checkpoint_sqlite'};\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in blocked:\n"
        "            raise ImportError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker());\n"
        "sys.path.insert(0, 'src');\n"
        "sys.argv = ['prog', 'pipeline', 'run'];\n"
        "from view.options import get_args; get_args();\n"
        "from controller.pipeline_controller.pipeline_controller import "
        "PipelineController;\n"
        "from mapping.fml_map import StructureMapGenerator;\n"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_no_production_code_depends_on_the_copied_prototype():
    """WP9: the ``agent-on-fhir/`` copy is reference material, never a runtime.

    Checked as a property of the tree rather than trusted to review, because the
    whole point of the copy is that it can be deleted. A single import left
    behind would turn its removal into a broken installation, and nothing else
    in the suite would notice — the prototype is on nobody's import path today,
    so an accidental dependency fails only where the copy is absent.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    offenders = []
    for path in sorted((repo_root / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if "agent-on-fhir" in line or "agent_on_fhir" in line:
                offenders.append(f"{path.relative_to(repo_root)}:{number}: {line.strip()}")
    assert offenders == [], "production code references the prototype:\n" + "\n".join(
        offenders
    )
