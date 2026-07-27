import sys

import pytest

from view import options


@pytest.fixture(autouse=True)
def reset_argv():
    original = sys.argv[:]
    yield
    sys.argv = original


def parse_args(argv):
    sys.argv = ["prog", *argv]
    return options.get_args()


# ---------------------------------------------------------------------------
# Global / top-level options
# ---------------------------------------------------------------------------


def test_no_command_leaves_command_none():
    args = parse_args([])

    assert args.command is None
    assert args.config == "conf/default.json"
    assert args.show_config is False
    assert args.development is False


def test_show_config_flag():
    args = parse_args(["--show-config"])

    assert args.show_config is True


def test_development_flag():
    args = parse_args(["--development"])

    assert args.development is True


def test_config_short_and_long_flag():
    short = parse_args(["-c", "conf/custom.json"])
    long = parse_args(["--config", "conf/other.json"])

    assert short.config == "conf/custom.json"
    assert long.config == "conf/other.json"


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_clear_defaults_force_false():
    args = parse_args(["cache", "clear"])

    assert args.command == "cache"
    assert args.cache_action == "clear"
    assert args.force is False


def test_cache_clear_force_flag():
    args = parse_args(["cache", "clear", "--force"])

    assert args.force is True


def test_cache_get_parses_url_and_output_path():
    args = parse_args([
        "cache",
        "get",
        "http://example.org/StructureDefinition/patient-profile",
        "--output",
        "patient.json",
    ])

    assert args.command == "cache"
    assert args.cache_action == "get"
    assert args.url == "http://example.org/StructureDefinition/patient-profile"
    assert args.output == "patient.json"


def test_cache_get_output_defaults_none():
    args = parse_args(["cache", "get", "http://example.org/x"])

    assert args.url == "http://example.org/x"
    assert args.output is None


def test_cache_delete_parses_url():
    args = parse_args(["cache", "delete", "http://example.org/x"])

    assert args.cache_action == "delete"
    assert args.url == "http://example.org/x"


def test_cache_list_parses_filter():
    args = parse_args([
        "cache",
        "list",
        "--filter",
        "patient",
    ])

    assert args.command == "cache"
    assert args.cache_action == "list"
    assert args.filter == "patient"


def test_cache_list_filter_defaults_none():
    args = parse_args(["cache", "list"])

    assert args.filter is None


def test_cache_stats_action():
    args = parse_args(["cache", "stats"])

    assert args.cache_action == "stats"


def test_cache_add_parses_file():
    args = parse_args(["cache", "add", "resource.json"])

    assert args.cache_action == "add"
    assert args.file == "resource.json"


def test_cache_get_missing_url_exits():
    with pytest.raises(SystemExit):
        parse_args(["cache", "get"])


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------


def test_pipeline_run_defaults():
    args = parse_args(["pipeline", "run"])

    assert args.command == "pipeline"
    assert args.pipeline_action == "run"
    assert args.force_overwrite is False
    assert args.auto_mapping is False
    assert args.minimal_structure_map is False
    # -crm defaults to True
    assert args.create_references_in_structure_map is True
    assert args.mapping_table_path == ""
    assert args.prepare_matchbox is False


def test_pipeline_run_all_sm_flags():
    args = parse_args([
        "pipeline",
        "run",
        "-f",
        "-am",
        "-msm",
        "-crm",
        "-mt",
        "table.json",
        "--prepare-matchbox",
    ])

    assert args.force_overwrite is True
    assert args.auto_mapping is True
    assert args.minimal_structure_map is True
    assert args.create_references_in_structure_map is True
    assert args.mapping_table_path == "table.json"
    assert args.prepare_matchbox is True


def test_pipeline_run_long_flags():
    args = parse_args([
        "pipeline",
        "run",
        "--force-overwrite",
        "--auto-mapping",
        "--minimal-structure-map",
        "--mapping-table-path",
        "t.json",
    ])

    assert args.force_overwrite is True
    assert args.auto_mapping is True
    assert args.minimal_structure_map is True
    assert args.mapping_table_path == "t.json"


def test_pipeline_process_action():
    args = parse_args(["pipeline", "process"])

    assert args.pipeline_action == "process"
    assert args.force_overwrite is False


def test_pipeline_source_def_parses_source_data():
    args = parse_args(["pipeline", "source-def", "-s", "example.json"])

    assert args.pipeline_action == "source-def"
    assert args.source_data == "example.json"


def test_pipeline_source_def_source_data_default_empty():
    args = parse_args(["pipeline", "source-def"])

    assert args.source_data == ""


def test_pipeline_static_gen_sm_flags():
    args = parse_args([
        "pipeline",
        "static-gen-sm",
        "-am",
        "-msm",
        "-mt",
        "map.json",
    ])

    assert args.pipeline_action == "static-gen-sm"
    assert args.auto_mapping is True
    assert args.minimal_structure_map is True
    assert args.mapping_table_path == "map.json"


def test_pipeline_prepare_matchbox_action():
    args = parse_args(["pipeline", "prepare-matchbox", "-f"])

    assert args.pipeline_action == "prepare-matchbox"
    assert args.force_overwrite is True


def test_pipeline_export_fields_defaults():
    args = parse_args(["pipeline", "export-fields"])

    assert args.pipeline_action == "export-fields"
    assert args.output_file == ""
    assert args.roots_only is False


def test_pipeline_export_fields_options():
    args = parse_args([
        "pipeline",
        "export-fields",
        "-o",
        "fields.json",
        "--roots-only",
    ])

    assert args.output_file == "fields.json"
    assert args.roots_only is True


def test_pipeline_validate_parses_input_and_profile():
    args = parse_args([
        "pipeline",
        "validate",
        "-i",
        "transformed.json",
        "-p",
        "http://example.org/StructureDefinition/patient",
    ])

    assert args.pipeline_action == "validate"
    assert args.input_file == "transformed.json"
    assert args.profile_url == "http://example.org/StructureDefinition/patient"


def test_pipeline_validate_profile_defaults_empty():
    args = parse_args(["pipeline", "validate", "-i", "transformed.json"])

    assert args.input_file == "transformed.json"
    assert args.profile_url == ""


def test_pipeline_validate_requires_input_file():
    with pytest.raises(SystemExit):
        parse_args(["pipeline", "validate"])


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["start", "stop", "status"])
def test_server_action_choices(action):
    args = parse_args(["server", action])

    assert args.command == "server"
    assert args.action == action


def test_server_invalid_action_exits():
    with pytest.raises(SystemExit):
        parse_args(["server", "restart"])


# ---------------------------------------------------------------------------
# client
# ---------------------------------------------------------------------------


def test_client_send_request_parses_method_and_payload():
    args = parse_args([
        "client",
        "send-request",
        "--method",
        "status",
        "--params",
        '{"a":1}',
        "--data",
        '{"b":2}',
        "--socket-path",
        "/tmp/custom.sock",
    ])

    assert args.command == "client"
    assert args.action == "send-request"
    assert args.method == "status"
    assert args.params == '{"a":1}'
    assert args.data == '{"b":2}'
    assert args.socket_path == "/tmp/custom.sock"


def test_client_send_request_short_flags_and_socket_default():
    args = parse_args([
        "client",
        "send-request",
        "-m",
        "transform_data",
        "-p",
        "{}",
        "-d",
        "[]",
    ])

    assert args.method == "transform_data"
    assert args.params == "{}"
    assert args.data == "[]"
    assert args.socket_path == "/tmp/fsh_nifi_bridge.sock"


def test_client_invalid_action_exits():
    with pytest.raises(SystemExit):
        parse_args(["client", "receive"])


# ---------------------------------------------------------------------------
# matchbox-cli
# ---------------------------------------------------------------------------


def test_matchbox_transform_data_parses_input_and_structure_map_url():
    args = parse_args([
        "matchbox-cli",
        "transform-data",
        "--input-file",
        "input.json",
        "--structure-map-url",
        "http://example.org/StructureMap/test-map",
    ])

    assert args.command == "matchbox-cli"
    assert args.matchbox_action == "transform-data"
    assert args.input_file == "input.json"
    assert args.structure_map_url == "http://example.org/StructureMap/test-map"


def test_matchbox_transform_data_defaults():
    args = parse_args(["matchbox-cli", "transform-data", "-i", "input.json"])

    assert args.structure_map_url is None
    assert args.bundle is False
    assert args.batch is False
    assert args.output_file is None


def test_matchbox_transform_data_bundle_batch_output():
    args = parse_args([
        "matchbox-cli",
        "transform-data",
        "-i",
        "input.json",
        "--bundle",
        "--batch",
        "-o",
        "out.json",
    ])

    assert args.bundle is True
    assert args.batch is True
    assert args.output_file == "out.json"


def test_matchbox_validate_data_parses_input_and_profile():
    args = parse_args([
        "matchbox-cli",
        "validate-data",
        "-i",
        "input.json",
        "-p",
        "http://example.org/StructureDefinition/patient",
    ])

    assert args.matchbox_action == "validate-data"
    assert args.input_file == "input.json"
    assert args.profile_url == "http://example.org/StructureDefinition/patient"


def test_matchbox_install_package_by_name_and_version():
    args = parse_args([
        "matchbox-cli",
        "install-package",
        "-n",
        "hl7.fhir.r4.core",
        "-v",
        "4.0.1",
        "--package-url",
        "http://registry.example.org",
    ])

    assert args.matchbox_action == "install-package"
    assert args.package_name == "hl7.fhir.r4.core"
    assert args.package_version == "4.0.1"
    assert args.package_url == "http://registry.example.org"


def test_matchbox_install_package_by_path():
    args = parse_args([
        "matchbox-cli",
        "install-package",
        "--package-path",
        "package.tgz",
    ])

    assert args.package_path == "package.tgz"


def test_matchbox_check_installed_ig_url_and_id():
    args = parse_args([
        "matchbox-cli",
        "check-installed-ig",
        "-u",
        "http://hl7.org/fhir/us/core/ImplementationGuide/hl7.fhir.us.core",
        "-i",
        "hl7.fhir.us.core",
    ])

    assert args.matchbox_action == "check-installed-ig"
    assert args.ig_url == "http://hl7.org/fhir/us/core/ImplementationGuide/hl7.fhir.us.core"
    assert args.ig_id == "hl7.fhir.us.core"


def test_matchbox_upload_sd_path():
    args = parse_args(["matchbox-cli", "upload-sd", "-p", "sd.json"])

    assert args.matchbox_action == "upload-sd"
    assert args.sd_path == "sd.json"


def test_matchbox_upload_sm_path():
    args = parse_args(["matchbox-cli", "upload-sm", "-p", "sm.json"])

    assert args.matchbox_action == "upload-sm"
    assert args.sm_path == "sm.json"


def test_matchbox_upload_cm_path():
    args = parse_args(["matchbox-cli", "upload-cm", "-p", "cm.json"])

    assert args.matchbox_action == "upload-cm"
    assert args.cm_path == "cm.json"


def test_matchbox_get_resource_by_type_url_id():
    args = parse_args([
        "matchbox-cli",
        "get-resource",
        "-t",
        "StructureDefinition",
        "-u",
        "http://example.org/sd",
        "-i",
        "patient-profile",
    ])

    assert args.matchbox_action == "get-resource"
    assert args.resource_type == "StructureDefinition"
    assert args.resource_url == "http://example.org/sd"
    assert args.resource_id == "patient-profile"


# ---------------------------------------------------------------------------
# init / process-fsh
# ---------------------------------------------------------------------------


def test_init_parses_source():
    args = parse_args(["init", "./profiles"])

    assert args.command == "init"
    assert args.source == "./profiles"


def test_init_requires_source():
    with pytest.raises(SystemExit):
        parse_args(["init"])


def test_process_fsh_parses_directory_and_name():
    args = parse_args([
        "process-fsh",
        "./data/fsh-profile",
        "--name",
        "tutorial-project",
    ])

    assert args.command == "process-fsh"
    assert args.fsh_dir == "./data/fsh-profile"
    assert args.name == "tutorial-project"


def test_process_fsh_name_defaults_none():
    args = parse_args(["process-fsh", "./data/fsh-profile"])

    assert args.fsh_dir == "./data/fsh-profile"
    assert args.name is None


def test_process_fsh_requires_directory():
    with pytest.raises(SystemExit):
        parse_args(["process-fsh"])
