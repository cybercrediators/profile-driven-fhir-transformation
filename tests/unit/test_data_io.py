"""Unit tests for DataIO — filesystem operations within a project directory."""

import json

import pytest

from data_handling.data_io import DataIO

pytestmark = pytest.mark.unit


@pytest.fixture
def data_io(tmp_path):
    return DataIO(tmp_path / "project")


# --------------------------------------------------------------------------- #
# construction & structure
# --------------------------------------------------------------------------- #


def test_construction_creates_all_standard_folders(data_io):
    for folder in DataIO.ProjectFolders:
        assert (data_io.project_dir / folder.value).is_dir()


def test_project_name_matches_dir(tmp_path):
    io = DataIO(tmp_path / "my_project")
    assert io.project_name == "my_project"


# --------------------------------------------------------------------------- #
# create_folder / delete_folder / update_folder_name
# --------------------------------------------------------------------------- #


def test_create_folder_creates_nested_path(data_io):
    path = data_io.create_folder("custom/nested/folder")
    assert path.is_dir()


def test_delete_folder_removes_directory(data_io):
    data_io.create_folder("to_delete")
    data_io.delete_folder("to_delete")
    assert not (data_io.project_dir / "to_delete").exists()


def test_delete_folder_missing_is_noop(data_io):
    data_io.delete_folder("nonexistent")  # must not raise


def test_update_folder_name_renames(data_io):
    data_io.create_folder("old_name")
    data_io.update_folder_name("old_name", "new_name")
    assert not (data_io.project_dir / "old_name").exists()
    assert (data_io.project_dir / "new_name").is_dir()


# --------------------------------------------------------------------------- #
# read_file / update_file / rename_file / delete_file
# --------------------------------------------------------------------------- #


def test_read_file_returns_content(data_io):
    (data_io.project_dir / "hello.txt").write_text("hello world", encoding="utf-8")
    assert data_io.read_file("hello.txt") == "hello world"


def test_read_file_missing_returns_none(data_io):
    assert data_io.read_file("missing.txt") is None


def test_update_file_append(data_io):
    p = data_io.project_dir / "append.txt"
    p.write_text("line1\n", encoding="utf-8")
    data_io.update_file("append.txt", "line2\n", append=True)
    assert p.read_text(encoding="utf-8") == "line1\nline2\n"


def test_update_file_overwrite(data_io):
    p = data_io.project_dir / "over.txt"
    p.write_text("old content", encoding="utf-8")
    data_io.update_file("over.txt", "new content", append=False)
    assert p.read_text(encoding="utf-8") == "new content"


def test_update_file_missing_is_noop(data_io):
    data_io.update_file("missing.txt", "x")  # must not raise


def test_rename_file(data_io):
    p = data_io.project_dir / "old.txt"
    p.write_text("data", encoding="utf-8")
    data_io.rename_file("old.txt", "new.txt")
    assert not (data_io.project_dir / "old.txt").exists()
    assert (data_io.project_dir / "new.txt").read_text(encoding="utf-8") == "data"


def test_rename_file_missing_is_noop(data_io):
    data_io.rename_file("missing.txt", "other.txt")  # must not raise


def test_delete_file(data_io):
    p = data_io.project_dir / "to_del.txt"
    p.write_text("x", encoding="utf-8")
    data_io.delete_file("to_del.txt")
    assert not p.exists()


def test_delete_file_missing_is_noop(data_io):
    data_io.delete_file("missing.txt")  # must not raise


# --------------------------------------------------------------------------- #
# store_project_file / load_project_file / load_project_files
# --------------------------------------------------------------------------- #


def test_store_and_load_project_file(data_io):
    payload = {"key": "value", "n": 42}
    data_io.store_project_file(DataIO.ProjectFolders.PROCESSED_RESOURCES, "res.json", payload)
    loaded = data_io.load_project_file(DataIO.ProjectFolders.PROCESSED_RESOURCES, "res.json")
    assert loaded == payload


def test_store_project_file_skip_if_exists(data_io):
    folder = DataIO.ProjectFolders.PROCESSED_RESOURCES
    data_io.store_project_file(folder, "res.json", {"v": 1})
    data_io.store_project_file(folder, "res.json", {"v": 2}, overwrite=False)
    loaded = data_io.load_project_file(folder, "res.json")
    assert loaded["v"] == 1


def test_store_project_file_overwrite(data_io):
    folder = DataIO.ProjectFolders.PROCESSED_RESOURCES
    data_io.store_project_file(folder, "res.json", {"v": 1})
    data_io.store_project_file(folder, "res.json", {"v": 2}, overwrite=True)
    loaded = data_io.load_project_file(folder, "res.json")
    assert loaded["v"] == 2


def test_load_project_file_missing_returns_none(data_io):
    assert data_io.load_project_file(DataIO.ProjectFolders.PROCESSED_RESOURCES, "nope.json") is None


def test_load_project_files_returns_only_json(data_io):
    folder_path = data_io.project_dir / DataIO.ProjectFolders.PROCESSED_RESOURCES.value
    (folder_path / "a.json").write_text(json.dumps({"id": "a"}), encoding="utf-8")
    (folder_path / "b.json").write_text(json.dumps({"id": "b"}), encoding="utf-8")
    (folder_path / "ignore.txt").write_text("not json", encoding="utf-8")
    results = data_io.load_project_files(DataIO.ProjectFolders.PROCESSED_RESOURCES)
    names = {name for name, _ in results}
    assert names == {"a.json", "b.json"}


def test_load_project_files_empty_folder(data_io):
    results = data_io.load_project_files(DataIO.ProjectFolders.PROCESSED_RESOURCES)
    assert results == []


# --------------------------------------------------------------------------- #
# project_file_exists
# --------------------------------------------------------------------------- #


def test_project_file_exists_true(data_io):
    folder = DataIO.ProjectFolders.STRUCTURE_MAPS
    data_io.store_project_file(folder, "sm.json", {})
    assert data_io.project_file_exists(folder, "sm.json") is True


def test_project_file_exists_false(data_io):
    assert data_io.project_file_exists(DataIO.ProjectFolders.STRUCTURE_MAPS, "nope.json") is False


# --------------------------------------------------------------------------- #
# check_processed_* helpers
# --------------------------------------------------------------------------- #


def test_check_processed_resources_false_when_empty(data_io):
    assert data_io.check_processed_resources() is False


def test_check_processed_resources_true_after_store(data_io):
    data_io.store_project_file(DataIO.ProjectFolders.PROCESSED_RESOURCES, "r.json", {})
    assert data_io.check_processed_resources() is True


def test_check_processed_structure_maps_false_when_empty(data_io):
    assert data_io.check_processed_structure_maps() is False


def test_check_processed_helper_maps_false_when_empty(data_io):
    assert data_io.check_processed_helper_maps() is False


# --------------------------------------------------------------------------- #
# get_json_profile_files
# --------------------------------------------------------------------------- #


def test_get_json_profile_files_returns_only_json(data_io):
    profile_dir = data_io.project_dir / DataIO.ProjectFolders.INPUT_PROFILE.value
    (profile_dir / "sd.json").write_text("{}", encoding="utf-8")
    (profile_dir / "readme.txt").write_text("x", encoding="utf-8")
    files = list(data_io.get_json_profile_files(None))
    assert len(files) == 1
    assert files[0].name == "sd.json"


def test_get_json_profile_files_custom_path(data_io, tmp_path):
    custom = tmp_path / "custom_profiles"
    custom.mkdir()
    (custom / "p.json").write_text("{}", encoding="utf-8")
    (custom / "p.xml").write_text("<xml/>", encoding="utf-8")
    files = list(data_io.get_json_profile_files(custom))
    assert len(files) == 1


# --------------------------------------------------------------------------- #
# _get_json_files_from_folder / get_structure_map_files etc.
# --------------------------------------------------------------------------- #


def test_get_structure_map_files_only_json(data_io):
    sm_dir = data_io.project_dir / DataIO.ProjectFolders.STRUCTURE_MAPS.value
    (sm_dir / "sm.json").write_text("{}", encoding="utf-8")
    (sm_dir / "other.txt").write_text("x", encoding="utf-8")
    files = data_io.get_structure_map_files()
    assert len(files) == 1
    assert files[0].name == "sm.json"


def test_get_example_files_returns_json_only(data_io):
    ex_dir = data_io.project_dir / DataIO.ProjectFolders.EXAMPLES.value
    (ex_dir / "ex.json").write_text("{}", encoding="utf-8")
    files = data_io.get_example_files()
    assert len(files) == 1


# --------------------------------------------------------------------------- #
# find_tarred_profile / clear_project_folders
# --------------------------------------------------------------------------- #


def test_find_tarred_profile_finds_tgz(data_io):
    tgz = data_io.project_dir / "profile.tgz"
    tgz.write_bytes(b"dummy")
    found = data_io.find_tarred_profile()
    assert found == tgz


def test_find_tarred_profile_custom_path(data_io, tmp_path):
    tgz = tmp_path / "external.tgz"
    tgz.write_bytes(b"dummy")
    found = data_io.find_tarred_profile(str(tgz))
    assert found == tgz


def test_find_tarred_profile_missing_custom_path_returns_none(data_io):
    assert data_io.find_tarred_profile("/nonexistent/profile.tgz") is None


def test_find_tarred_profile_returns_none_when_absent(data_io):
    assert data_io.find_tarred_profile() is None


def test_clear_project_folders_removes_files(data_io):
    folder_path = data_io.project_dir / DataIO.ProjectFolders.STRUCTURE_MAPS.value
    (folder_path / "sm.json").write_text("{}", encoding="utf-8")
    data_io.clear_project_folders()
    assert list(folder_path.iterdir()) == []
