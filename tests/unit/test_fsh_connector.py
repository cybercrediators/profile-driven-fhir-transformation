from types import SimpleNamespace

import docker.errors
import pytest

import controller.connector.fsh_connector as fsh_mod
from controller.connector.fsh_connector import SushiController


pytestmark = pytest.mark.unit


def _no_default_docker(monkeypatch):
    """make the default-image probe fail so the local fallback is exercised"""
    def raise_unavailable():
        raise docker.errors.DockerException("docker daemon unavailable")

    monkeypatch.setattr(fsh_mod.docker, "from_env", raise_unavailable)


def _force_local_sushi(monkeypatch):
    """resolve to the local binary: default docker image unavailable, local fsh-sushi verified"""
    _no_default_docker(monkeypatch)
    monkeypatch.setattr(SushiController, "_verify_local_sushi", lambda self: True)


def _bare_controller(command="sushi"):
    """instance without running __init__'s setup check (for testing the check helpers themselves)"""
    controller = SushiController.__new__(SushiController)
    controller.sushi_command = command
    controller.docker_image = None
    controller._docker_client = None
    return controller


def test_init_raises_when_sushi_missing(monkeypatch):
    _no_default_docker(monkeypatch)
    monkeypatch.setattr(fsh_mod.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError):
        SushiController()


def test_process_fsh_runs_sushi_and_copies_generated_json(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    generated = fsh_dir / "fsh-generated" / "resources"
    generated.mkdir(parents=True)
    (generated / "Patient-test.json").write_text('{"resourceType": "Patient"}')
    (generated / "Observation-test.json").write_text('{"resourceType": "Observation"}')

    _force_local_sushi(monkeypatch)
    observed = {}

    def fake_run(command, cwd=None, check=None):
        observed.update(command=command, cwd=cwd, check=check)

    monkeypatch.setattr(fsh_mod.subprocess, "run", fake_run)

    project_path = tmp_path / "projects" / "my-project"
    project_path.parent.mkdir(parents=True)  # DataIO.mkdir doesn't create parents
    SushiController().process_fsh(str(fsh_dir), str(project_path), "my-project")

    assert observed == {"command": ["sushi", "build", "."], "cwd": fsh_dir.resolve(), "check": True}

    target_dir = project_path / "input_profile"
    copied = sorted(p.name for p in target_dir.glob("*.json"))
    assert copied == ["Observation-test.json", "Patient-test.json"]
    assert (target_dir / "Patient-test.json").read_text() == '{"resourceType": "Patient"}'


def test_process_fsh_aborts_when_no_output(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    fsh_dir.mkdir()
    _force_local_sushi(monkeypatch)
    monkeypatch.setattr(fsh_mod.subprocess, "run", lambda *a, **k: None)

    project_path = tmp_path / "projects" / "my-project"
    # no fsh-generated/resources or output -> returns early, no project created
    SushiController().process_fsh(str(fsh_dir), str(project_path), "my-project")
    assert not (project_path / "input_profile").exists()


def test_process_fsh_missing_dir_returns_early(tmp_path, monkeypatch):
    _force_local_sushi(monkeypatch)

    def fail_run(*a, **k):
        raise AssertionError("sushi should not run when the fsh dir is missing")

    monkeypatch.setattr(fsh_mod.subprocess, "run", fail_run)

    project_path = tmp_path / "projects" / "my-project"
    SushiController().process_fsh(str(tmp_path / "does-not-exist"), str(project_path))
    assert not project_path.exists()


def test_process_fsh_called_process_error_returns_early(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    fsh_dir.mkdir()
    _force_local_sushi(monkeypatch)

    def fake_run(*a, **k):
        raise fsh_mod.subprocess.CalledProcessError(1, ["sushi", "build", "."])

    monkeypatch.setattr(fsh_mod.subprocess, "run", fake_run)

    project_path = tmp_path / "projects" / "my-project"
    SushiController().process_fsh(str(fsh_dir), str(project_path))
    assert not project_path.exists()


def test_process_fsh_generic_exception_returns_early(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    fsh_dir.mkdir()
    _force_local_sushi(monkeypatch)

    def fake_run(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(fsh_mod.subprocess, "run", fake_run)

    project_path = tmp_path / "projects" / "my-project"
    SushiController().process_fsh(str(fsh_dir), str(project_path))
    assert not project_path.exists()


def test_process_fsh_defaults_project_name_from_fsh_dir(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    generated = fsh_dir / "fsh-generated" / "resources"
    generated.mkdir(parents=True)
    (generated / "Patient-test.json").write_text('{"resourceType": "Patient"}')

    _force_local_sushi(monkeypatch)
    monkeypatch.setattr(fsh_mod.subprocess, "run", lambda *a, **k: None)

    project_path = tmp_path / "projects" / "waves-profile"
    project_path.parent.mkdir(parents=True)
    # no project_name given -> derived from the fsh dir's resolved name
    SushiController().process_fsh(str(fsh_dir), str(project_path))

    target_dir = project_path / "input_profile"
    assert (target_dir / "Patient-test.json").exists()


def test_process_fsh_clears_stale_files_in_input_profile(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    generated = fsh_dir / "fsh-generated" / "resources"
    generated.mkdir(parents=True)
    (generated / "Patient-new.json").write_text('{"resourceType": "Patient"}')

    _force_local_sushi(monkeypatch)
    monkeypatch.setattr(fsh_mod.subprocess, "run", lambda *a, **k: None)

    project_path = tmp_path / "projects" / "my-project"
    project_path.parent.mkdir(parents=True)
    # pre-populate the project's input_profile/ with a stale file that must be removed
    from data_handling.data_io import DataIO

    DataIO(project_path)
    stale_file = project_path / "input_profile" / "Old-stale.json"
    stale_file.write_text("{}")

    SushiController().process_fsh(str(fsh_dir), str(project_path), "my-project")

    target_dir = project_path / "input_profile"
    copied = sorted(p.name for p in target_dir.glob("*.json"))
    assert copied == ["Patient-new.json"]
    assert not stale_file.exists()


def test_find_sushi_output_prefers_fsh_generated_over_output(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    fsh_generated = fsh_dir / "fsh-generated" / "resources"
    output_dir = fsh_dir / "output"
    fsh_generated.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (fsh_generated / "a.json").write_text("{}")
    (output_dir / "b.json").write_text("{}")

    _force_local_sushi(monkeypatch)
    controller = SushiController()
    assert controller._find_sushi_output(fsh_dir) == fsh_generated


def test_find_sushi_output_falls_back_to_output_dir(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    output_dir = fsh_dir / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "b.json").write_text("{}")

    _force_local_sushi(monkeypatch)
    controller = SushiController()
    assert controller._find_sushi_output(fsh_dir) == output_dir


def test_find_sushi_output_returns_none_when_nothing_found(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    fsh_dir.mkdir()
    _force_local_sushi(monkeypatch)
    controller = SushiController()
    assert controller._find_sushi_output(fsh_dir) is None


# --------------------------------------------------------------------------- #
# docker-backed sushi execution
# --------------------------------------------------------------------------- #


class FakeDockerClient:
    def __init__(self):
        self.images = SimpleNamespace(get=self._get_image)
        self.containers = SimpleNamespace(run=self._run_container)
        self.existing_images = set()
        self.run_calls = []
        self.run_logs = b"build ok\n"

    def _get_image(self, image):
        if image not in self.existing_images:
            raise docker.errors.ImageNotFound(f"no such image: {image}")
        return SimpleNamespace(id=image)

    def _run_container(self, **kwargs):
        self.run_calls.append(kwargs)
        return self.run_logs


def test_init_raises_when_docker_image_missing(monkeypatch):
    fake_client = FakeDockerClient()
    monkeypatch.setattr(fsh_mod.docker, "from_env", lambda: fake_client)

    with pytest.raises(FileNotFoundError):
        SushiController(docker_image="my/sushi:latest")


def test_init_succeeds_when_docker_image_present(monkeypatch):
    fake_client = FakeDockerClient()
    fake_client.existing_images.add("my/sushi:latest")
    monkeypatch.setattr(fsh_mod.docker, "from_env", lambda: fake_client)

    controller = SushiController(docker_image="my/sushi:latest")
    assert controller.check_fsh_setup() is True


def test_process_fsh_runs_via_docker_and_copies_output(tmp_path, monkeypatch):
    fake_client = FakeDockerClient()
    fake_client.existing_images.add("my/sushi:latest")
    monkeypatch.setattr(fsh_mod.docker, "from_env", lambda: fake_client)
    monkeypatch.setattr(fsh_mod.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(fsh_mod.os, "getgid", lambda: 1000, raising=False)

    fsh_dir = tmp_path / "waves-profile"
    generated = fsh_dir / "fsh-generated" / "resources"
    generated.mkdir(parents=True)
    (generated / "Patient-test.json").write_text('{"resourceType": "Patient"}')

    project_path = tmp_path / "projects" / "my-project"
    project_path.parent.mkdir(parents=True)

    controller = SushiController(docker_image="my/sushi:latest")
    controller.process_fsh(str(fsh_dir), str(project_path), "my-project")

    assert len(fake_client.run_calls) == 1
    call = fake_client.run_calls[0]
    assert call["image"] == "my/sushi:latest"
    assert call["command"] == ["sushi", "build", "."]
    assert call["working_dir"] == "/opt/fsh_dir"
    assert (project_path / "input_profile" / "Patient-test.json").exists()


# --------------------------------------------------------------------------- #
# setup resolution: configured image > default image > verified local binary
# --------------------------------------------------------------------------- #


def test_default_docker_image_adopted_when_present(monkeypatch):
    fake_client = FakeDockerClient()
    fake_client.existing_images.add(fsh_mod._DEFAULT_DOCKER_IMAGE)
    monkeypatch.setattr(fsh_mod.docker, "from_env", lambda: fake_client)

    controller = SushiController()
    assert controller.docker_image == fsh_mod._DEFAULT_DOCKER_IMAGE
    assert controller._docker_client is fake_client


def test_falls_back_to_local_when_default_image_missing(monkeypatch):
    fake_client = FakeDockerClient()  # docker reachable, but no default image built
    monkeypatch.setattr(fsh_mod.docker, "from_env", lambda: fake_client)
    monkeypatch.setattr(SushiController, "_verify_local_sushi", lambda self: True)

    controller = SushiController()
    assert controller.docker_image is None
    assert controller._docker_client is None


def test_explicit_image_missing_never_falls_back(monkeypatch):
    fake_client = FakeDockerClient()
    fake_client.existing_images.add(fsh_mod._DEFAULT_DOCKER_IMAGE)  # default IS available
    monkeypatch.setattr(fsh_mod.docker, "from_env", lambda: fake_client)

    with pytest.raises(FileNotFoundError):
        SushiController(docker_image="my/sushi:latest")


def _version_result(returncode, stdout="", stderr=""):
    return fsh_mod.subprocess.CompletedProcess(
        args=["sushi", "--version"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_verify_local_sushi_accepts_fsh_sushi(monkeypatch):
    monkeypatch.setattr(fsh_mod.shutil, "which", lambda name: "/usr/bin/sushi")
    monkeypatch.setattr(
        fsh_mod.subprocess,
        "run",
        lambda *a, **k: _version_result(0, stdout="SUSHI v3.19.0 (implements FHIR Shorthand specification v3.0.0)\n"),
    )
    assert _bare_controller()._verify_local_sushi() is True


def test_verify_local_sushi_rejects_gnome_sushi(monkeypatch):
    # GNOME Sushi (nautilus previewer): --version exits 1 with a file-not-found message
    monkeypatch.setattr(fsh_mod.shutil, "which", lambda name: "/usr/bin/sushi")
    monkeypatch.setattr(
        fsh_mod.subprocess,
        "run",
        lambda *a, **k: _version_result(1, stderr="The file at file:///--version does not exist.\n"),
    )
    assert _bare_controller()._verify_local_sushi() is False


def test_verify_local_sushi_missing_binary(monkeypatch):
    monkeypatch.setattr(fsh_mod.shutil, "which", lambda name: None)
    assert _bare_controller()._verify_local_sushi() is False


def test_verify_local_sushi_run_failure(monkeypatch):
    monkeypatch.setattr(fsh_mod.shutil, "which", lambda name: "/usr/bin/sushi")

    def raise_oserror(*a, **k):
        raise OSError("exec failed")

    monkeypatch.setattr(fsh_mod.subprocess, "run", raise_oserror)
    assert _bare_controller()._verify_local_sushi() is False


# --------------------------------------------------------------------------- #
# snapshot stage: genonce + temp/pages overlay
# --------------------------------------------------------------------------- #


def _write_sd(path, name, with_snapshot):
    sd = {"resourceType": "StructureDefinition", "name": name, "differential": {"element": []}}
    if with_snapshot:
        sd["snapshot"] = {"element": []}
    path.write_text(fsh_mod.json.dumps(sd))


def test_process_fsh_runs_genonce_and_overlay_by_default(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    generated = fsh_dir / "fsh-generated" / "resources"
    generated.mkdir(parents=True)
    (generated / "Patient-test.json").write_text('{"resourceType": "Patient"}')

    _force_local_sushi(monkeypatch)
    monkeypatch.setattr(fsh_mod.subprocess, "run", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(SushiController, "_run_genonce", lambda self, p: calls.append(("genonce", p)))
    monkeypatch.setattr(
        SushiController, "_overlay_snapshots", lambda self, p, d: calls.append(("overlay", p, d))
    )

    project_path = tmp_path / "projects" / "my-project"
    project_path.parent.mkdir(parents=True)
    SushiController().process_fsh(str(fsh_dir), str(project_path), "my-project")
    assert [c[0] for c in calls] == ["genonce", "overlay"]


def test_process_fsh_skips_snapshot_stage_when_disabled(tmp_path, monkeypatch):
    fsh_dir = tmp_path / "waves-profile"
    generated = fsh_dir / "fsh-generated" / "resources"
    generated.mkdir(parents=True)
    (generated / "Patient-test.json").write_text('{"resourceType": "Patient"}')

    _force_local_sushi(monkeypatch)
    monkeypatch.setattr(fsh_mod.subprocess, "run", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(SushiController, "_run_genonce", lambda self, p: calls.append("genonce"))
    monkeypatch.setattr(SushiController, "_overlay_snapshots", lambda self, p, d: calls.append("overlay"))

    project_path = tmp_path / "projects" / "my-project"
    project_path.parent.mkdir(parents=True)
    SushiController().process_fsh(str(fsh_dir), str(project_path), "my-project", snapshots=False)
    assert calls == []


def test_run_genonce_skips_when_script_missing(tmp_path, monkeypatch):
    def fail_run(*a, **k):
        raise AssertionError("genonce must not run without _genonce.sh")

    monkeypatch.setattr(fsh_mod.subprocess, "run", fail_run)
    _bare_controller()._run_genonce(tmp_path)  # no script -> warning only, no crash


def test_run_genonce_local_invokes_script_and_tolerates_failure(tmp_path, monkeypatch):
    (tmp_path / "_genonce.sh").write_text("#!/bin/bash\n")
    observed = {}

    def fake_run(command, cwd=None, check=None):
        observed.update(command=command, cwd=cwd, check=check)
        raise fsh_mod.subprocess.CalledProcessError(1, command)  # Jekyll-step failure

    monkeypatch.setattr(fsh_mod.subprocess, "run", fake_run)
    _bare_controller()._run_genonce(tmp_path)  # tolerated, no exception
    assert observed == {"command": ["bash", "_genonce.sh"], "cwd": tmp_path, "check": True}


def test_overlay_snapshots_prefers_pages_and_warns_on_missing(tmp_path):
    fsh_dir = tmp_path / "fsh"
    pages = fsh_dir / "temp" / "pages"
    pages.mkdir(parents=True)
    input_profile = tmp_path / "input_profile"
    input_profile.mkdir()

    # SD with a snapshot-bearing publisher counterpart -> gets replaced
    _write_sd(input_profile / "StructureDefinition-a.json", "A", with_snapshot=False)
    _write_sd(pages / "StructureDefinition-a.json", "A", with_snapshot=True)
    # SD without publisher counterpart -> stays differential-only (warned)
    _write_sd(input_profile / "StructureDefinition-b.json", "B", with_snapshot=False)
    # publisher counterpart itself lacks a snapshot -> not copied
    _write_sd(input_profile / "StructureDefinition-c.json", "C", with_snapshot=False)
    _write_sd(pages / "StructureDefinition-c.json", "C", with_snapshot=False)

    _bare_controller()._overlay_snapshots(fsh_dir, input_profile)

    a = fsh_mod.json.loads((input_profile / "StructureDefinition-a.json").read_text())
    b = fsh_mod.json.loads((input_profile / "StructureDefinition-b.json").read_text())
    c = fsh_mod.json.loads((input_profile / "StructureDefinition-c.json").read_text())
    assert "snapshot" in a
    assert "snapshot" not in b
    assert "snapshot" not in c


def test_has_snapshot_handles_missing_and_invalid_files(tmp_path):
    assert SushiController._has_snapshot(tmp_path / "nope.json") is False
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert SushiController._has_snapshot(bad) is False


def test_resolve_fsh_root_accepts_root_and_walks_up_from_sources(tmp_path):
    root = tmp_path / "waves-profile"
    sources = root / "input" / "fsh"
    sources.mkdir(parents=True)
    (root / "sushi-config.yaml").write_text("canonical: http://example.org\n")

    assert SushiController.resolve_fsh_root(root) == root
    assert SushiController.resolve_fsh_root(sources) == root
    assert SushiController.resolve_fsh_root(root / "input") == root


def test_resolve_fsh_root_returns_none_without_config(tmp_path):
    bare = tmp_path / "just-a-dir"
    bare.mkdir()
    assert SushiController.resolve_fsh_root(bare) is None
