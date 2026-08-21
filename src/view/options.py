import argparse


def get_args():
    """
    Read arguments from the command line
    """
    parser = argparse.ArgumentParser(
        prog="FHIR Bridge",
        description="""Client for processing data using
            FHIR profiles and a corresponding mapping table""",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Main processing command (default behavior)
    # subparsers.add_parser('pipeline', help="""Process FHIR profile
    #                                        Default behaviour (when fully configured):
    #                                        - Process/parse input FHIR profile
    #                                        - Create source data mapping (if source data provided)
    #                                        - Generate StructureMap(s) based on mappable fields (automapping optional)
    #                                        - TBA: start a socket manager for data transfers
    #                                        - Upload/check matchbox for profile definitions
    #                                        - Upload/check matchbox for source data definitions (if source data provided)
    #                                        - Upload/check matchbox for generated StructureMaps
    #                                        - Socket manager waits for data transfer request
    #                                        """)

    # argument to display config
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Display the loaded configuration and exit",
    )

    # Cache management command
    cache_parser = subparsers.add_parser("cache", help="Manage cache operations")
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_action", help="Cache actions"
    )

    # Cache clear
    clear_parser = cache_subparsers.add_parser(
        "clear", help="Clear all cached resources"
    )
    clear_parser.add_argument(
        "--force", default=False, action="store_true", help="Skip confirmation prompt"
    )

    # Cache get
    get_parser = cache_subparsers.add_parser(
        "get", help="Get a resource from cache by URL"
    )
    get_parser.add_argument(
        "url", type=str, help="Canonical URL of the resource to retrieve"
    )
    get_parser.add_argument(
        "--output",
        type=str,
        help="Output file path (if not specified, prints to console)",
    )

    # Cache delete
    delete_parser = cache_subparsers.add_parser(
        "delete", help="Delete a resource from cache by URL"
    )
    delete_parser.add_argument(
        "url", type=str, help="Canonical URL of the resource to delete"
    )

    # Cache list
    list_parser = cache_subparsers.add_parser(
        "list", help="List all cached resource URLs"
    )
    list_parser.add_argument(
        "--filter",
        type=str,
        help="Filter URLs by pattern (case-insensitive substring match)",
    )

    # Cache stats
    cache_subparsers.add_parser("stats", help="Show cache statistics")

    # Cache add
    add_parser = cache_subparsers.add_parser(
        "add", help="Add a resource to cache from file"
    )
    add_parser.add_argument("file", type=str, help="Path to FHIR resource JSON file")

    # Global config argument
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="conf/default.json",
        action="store",
        help="Path to configuration file",
    )

    # Shared parent parser that adds -f to individual pipeline actions
    _force_parent = argparse.ArgumentParser(add_help=False)
    _force_parent.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        help="Force re-generation/re-upload for this step, overwriting existing output",
    )

    # Pipeline processing command
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run pipeline processing steps"
    )
    pipeline_sub = pipeline_parser.add_subparsers(
        dest="pipeline_action", help="Pipeline step"
    )

    def _sm_flags(p):
        p.add_argument(
            "-am",
            "--auto-mapping",
            action="store_true",
            help="Enable simple field auto-mapping",
        )
        p.add_argument(
            "--auto-mapping-mode",
            choices=["deterministic", "llm"],
            default="deterministic",
            help=(
                "Auto-mapping strategy (default: deterministic). 'llm' reranks the "
                "deterministic candidates with a language model and requires the "
                "optional 'llm' extra plus provider configuration"
            ),
        )
        p.add_argument(
            "-msm",
            "--minimal-structure-map",
            action="store_true",
            help="Generate minimal StructureMap (only mapped fields)",
        )
        p.add_argument(
            "-crm",
            "--create-references-in-structure-map",
            action="store_true",
            default=True,
            help="Generate reference mappings in StructureMap",
        )
        p.add_argument(
            "-mt",
            "--mapping-table-path",
            type=str,
            default="",
            help="Path to source→target mapping table JSON",
        )

    run_p = pipeline_sub.add_parser(
        "run",
        parents=[_force_parent],
        help="Execute all steps: parse → source-def → static-gen-sm (+ optionally prepare-matchbox)",
    )
    _sm_flags(run_p)
    run_p.add_argument(
        "--prepare-matchbox",
        action="store_true",
        help="Upload to matchbox after generation (use -f to force re-upload)",
    )

    pipeline_sub.add_parser(
        "process",
        parents=[_force_parent],
        help="Step 1: Process profile files into registry",
    )
    source_def_p = pipeline_sub.add_parser(
        "source-def",
        parents=[_force_parent],
        help="Step 2: Generate source helper StructureDefinition from source data example",
    )
    source_def_p.add_argument(
        "-s",
        "--source-data",
        type=str,
        default="",
        help="Path to source data example JSON file (overrides input_source_example from config)",
    )

    sm_p = pipeline_sub.add_parser(
        "static-gen-sm",
        parents=[_force_parent],
        help="Step 3: Generate StructureMap(s) statically",
    )
    _sm_flags(sm_p)

    pipeline_sub.add_parser(
        "prepare-matchbox",
        parents=[_force_parent],
        help="Step 4: Upload profile and StructureMap(s) to matchbox (-f forces re-upload)",
    )

    export_fields_p = pipeline_sub.add_parser(
        "export-fields",
        help="Export all mappable fields from the processed registry as JSON",
    )
    export_fields_p.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="Path to write the JSON output (default: stdout)",
    )
    export_fields_p.add_argument(
        "--roots-only",
        action="store_true",
        default=False,
        help="Only export fields from root resources",
    )

    validate_p = pipeline_sub.add_parser(
        "validate",
        help="Validate a transformed FHIR resource against its target profile via matchbox $validate",
    )
    validate_p.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Path to the already-transformed FHIR resource JSON file to validate",
    )
    validate_p.add_argument(
        "-p",
        "--profile-url",
        type=str,
        default="",
        help="Profile URL to validate against. If omitted, derived from the loaded StructureMaps by matching the resource type.",
    )

    validate_instances_p = pipeline_sub.add_parser(
        "validate-instances",
        help="Validate the IG's own example instances: direct $validate + reverse-extract→transform→compare round-trip",
    )
    validate_instances_p.add_argument(
        "--examples-dir",
        type=str,
        default="",
        help="Directory of example instance JSONs (e.g. fsh-generated/resources). "
        "Adds to examples/ and input_profile/ auto-discovery.",
    )
    validate_instances_p.add_argument(
        "-mt",
        "--mapping-table-path",
        type=str,
        default="",
        help="Path to source→target mapping table JSON (required for the round-trip).",
    )
    validate_instances_p.add_argument(
        "--direct-only",
        action="store_true",
        default=False,
        help="Only $validate the raw examples against their profiles (skip the round-trip).",
    )
    validate_instances_p.add_argument(
        "--from-element-examples",
        action="store_true",
        default=False,
        help="Also synthesize instances from ElementDefinition.example values in the profiles.",
    )
    validate_instances_p.add_argument(
        "-o",
        "--report",
        type=str,
        default="",
        help="Path to write the JSON report (default: converted_data/instance_validation_report.json).",
    )

    reverse_map_p = pipeline_sub.add_parser(
        "reverse-map",
        help="PoC: extract flat source-shaped records back out of transformed FHIR "
        "output, using the mapping table as extraction spec (+ inverse ConceptMaps). "
        "Offline — no matchbox needed.",
    )
    reverse_map_p.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Path to transformed FHIR output JSON (Bundle, list of Bundles, or resource(s))",
    )
    reverse_map_p.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="Path to write the extracted flat records JSON (default: stdout)",
    )
    reverse_map_p.add_argument(
        "-s",
        "--source-data",
        type=str,
        default="",
        help="Original flat source data JSON to compare against (writes a round-trip report)",
    )
    reverse_map_p.add_argument(
        "--report",
        type=str,
        default="",
        help="Path for the round-trip report (default: converted_data/reverse_mapping_report.json)",
    )
    reverse_map_p.add_argument(
        "-mt",
        "--mapping-table-path",
        type=str,
        default="",
        help="Path to source→target mapping table JSON (default: from config)",
    )

    agent_parser = subparsers.add_parser(
        "agent",
        help=(
            "Agent mode: LLM-assisted StructureMap repair (requires optional 'llm' extra)"
        ),
    )
    agent_sub = agent_parser.add_subparsers(dest="agent_action", help="Agent command")

    agent_fix_p = agent_sub.add_parser(
        "fix",
        help=(
            "Propose and validate repairs for one StructureMap. Writes an "
            "auditable run directory; leaves the map untouched unless --apply"
        ),
    )
    agent_fix_p.add_argument(
        "map",
        type=str,
        nargs="?",
        default=None,
        help=(
            "StructureMap to repair: file path, canonical URL, or resource id. "
            "Optional when the project has exactly one map"
        ),
    )
    agent_fix_p.add_argument(
        "--all",
        dest="all_maps",
        action="store_true",
        help=(
            "Repair every StructureMap in the project, validate the assembled "
            "set, and requeue the maps its findings name"
        ),
    )
    agent_fix_p.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Maximum repair attempts per map before stopping unresolved (default: 4)",
    )
    agent_fix_p.add_argument(
        "--max-project-rounds",
        type=int,
        default=3,
        help=(
            "Maximum project rounds: repair every pending map, validate the "
            "assembled set, requeue what it names (default: 3)"
        ),
    )
    agent_fix_p.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help=(
            "Continue an interrupted run from its checkpoint. Refuses if the "
            "configuration, provider, or any selected map changed since it started"
        ),
    )
    agent_fix_p.add_argument(
        "--llm-provider",
        choices=["openai-compatible", "openai"],
        default=None,
        help="Override the configured LLM provider for this agent run",
    )
    agent_fix_p.add_argument(
        "--llm-model",
        default=None,
        help="Override the configured LLM model for this agent run",
    )
    agent_fix_p.add_argument(
        "--llm-base-url",
        default=None,
        help="Override the OpenAI-compatible endpoint for this agent run",
    )
    agent_fix_p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="",
        help="Where to write agent_output/<run-id>/ (default: the project directory)",
    )
    agent_fix_p.add_argument(
        "--offline",
        action="store_true",
        help=(
            "Run deterministic validation only, without Matchbox. Produces "
            "diagnostic candidates that can never be applied"
        ),
    )
    agent_fix_p.add_argument(
        "--use-examples",
        action="store_true",
        help=(
            "Additionally exercise candidates with fixtures reverse-extracted "
            "from project examples (experimental; never gate-relevant)"
        ),
    )
    agent_fix_p.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Replace the source map with an accepted candidate. Refuses "
            "rejected, offline-only, and stale candidates"
        ),
    )

    # Server management command
    server_parser = subparsers.add_parser(
        "server", help="Manage the background socket server"
    )
    server_parser.add_argument(
        "action",
        choices=["start", "stop", "status"],
        help="Action to perform on the server",
    )

    # Socket client commands
    client_parser = subparsers.add_parser("client", help="Manage the socket client")
    client_parser.add_argument(
        "action", choices=["send-request"], help="Action to perform with the client"
    )
    client_parser.add_argument(
        "-m",
        "--method",
        type=str,
        help='Method to send to the server (e.g., "status", "validate_setup", "transform_data")',
    )
    client_parser.add_argument(
        "-p", "--params", type=str, help="JSON string for params object"
    )
    client_parser.add_argument(
        "-d", "--data", type=str, help="JSON string for data payload"
    )
    client_parser.add_argument(
        "--socket-path",
        type=str,
        default="/tmp/fsh_nifi_bridge.sock",
        help="UNIX socket path",
    )

    parser.add_argument(
        "--development", action="store_true", help="Dev command -> manual testing"
    )

    # matchbox connector commands
    matchbox_connector_parser = subparsers.add_parser(
        "matchbox-cli", help="Test matchbox connector operations"
    )
    # subparser for matchbox connector parser
    matchbox_connector_subparsers = matchbox_connector_parser.add_subparsers(
        dest="matchbox_action", help="Matchbox actions"
    )

    # transform data using matchbox
    mb_transform_parser = matchbox_connector_subparsers.add_parser(
        "transform-data", help="Transform input FHIR data using matchbox"
    )
    mb_transform_parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Path to input FHIR resource JSON file to transform",
    )
    mb_transform_parser.add_argument(
        "-s",
        "--structure-map-url",
        type=str,
        default=None,
        help="URL of a specific StructureMap to use. If omitted, all StructureMaps from the project are used in order.",
    )
    mb_transform_parser.add_argument(
        "--bundle",
        action="store_true",
        default=False,
        help="Wrap transformed resources in a FHIR Bundle. Default returns a plain array.",
    )
    mb_transform_parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Treat the input file as a JSON array of source objects and transform each one independently.",
    )
    mb_transform_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Path to write the transformed output JSON (default: stdout)",
    )

    # validate data using matchbox
    mb_validate_parser = matchbox_connector_subparsers.add_parser(
        "validate-data", help="Validate input FHIR data using matchbox"
    )
    mb_validate_parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Path to input FHIR resource JSON file to validate",
    )
    mb_validate_parser.add_argument(
        "-p", "--profile-url", type=str, help="URL of the profile to use for validation"
    )

    # install local package
    mb_install_parser = matchbox_connector_subparsers.add_parser(
        "install-package",
        help="Install a FHIR npm package to matchbox. Usage: either provide a package name and version"
        "(with optional package registry uri) or a local package path to a tar archive",
    )

    mb_install_parser.add_argument(
        "-n",
        "--package-name",
        type=str,
        help='Provide the npm/package name (e.g., "hl7.fhir.r4.core")',
    )
    mb_install_parser.add_argument(
        "-v",
        "--package-version",
        type=str,
        help='Provide the npm/package version (e.g., "4.0.1")',
    )
    mb_install_parser.add_argument(
        "--package-url",
        type=str,
        help="URL to fetch the package from (overrides default npm registry)",
    )
    mb_install_parser.add_argument(
        "--package-path",
        type=str,
        help="Path to local FHIR package to install to matchbox -> must be a gzipped tar archive (.tgz or .tar.gz file)",
    )

    mb_check_parser = matchbox_connector_subparsers.add_parser(
        "check-installed-ig",
        help="Check if a given Implementation Guide (by package name and version) is installed in matchbox",
    )
    mb_check_parser.add_argument(
        "-u",
        "--ig-url",
        type=str,
        help='The Implementation Guide URL to check (e.g., "http://hl7.org/fhir/us/core/ImplementationGuide/hl7.fhir.us.core")',
    )

    mb_check_parser.add_argument(
        "-i",
        "--ig-id",
        type=str,
        help='The Implementation Guide ID to check (e.g., "hl7.fhir.us.core")',
    )

    upload_sd_parser = matchbox_connector_subparsers.add_parser(
        "upload-sd",
        help="Upload a StructureDefinition resource to matchbox (development mode only!)",
    )
    upload_sd_parser.add_argument(
        "-p",
        "--sd-path",
        type=str,
        help="Path to the StructureDefinition JSON file to upload",
    )

    upload_sd_parser = matchbox_connector_subparsers.add_parser(
        "upload-sm",
        help="Upload a StructureMap resource to matchbox (development mode only!)",
    )
    upload_sd_parser.add_argument(
        "-p", "--sm-path", type=str, help="Path to the StructureMap JSON file to upload"
    )
    upload_cm_parser = matchbox_connector_subparsers.add_parser(
        "upload-cm",
        help="Upload a ConceptMap resource to matchbox (development mode only!)",
    )
    upload_cm_parser.add_argument(
        "-p", "--cm-path", type=str, help="Path to the ConceptMap JSON file to upload"
    )

    # get matchbox resource commands
    mb_resource_parser = matchbox_connector_subparsers.add_parser(
        "get-resource", help="Get a FHIR resource from matchbox by type and id"
    )
    mb_resource_parser.add_argument(
        "-t",
        "--resource_type",
        type=str,
        help='The FHIR resource type (e.g., "StructureDefinition")',
    )
    # resource url
    mb_resource_parser.add_argument(
        "-u", "--resource_url", type=str, help="The FHIR resource url"
    )
    # resource id
    mb_resource_parser.add_argument(
        "-i", "--resource_id", type=str, help="The FHIR resource id"
    )

    # Project init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize project folder and populate input_profile/ from a folder or .tgz",
    )
    init_parser.add_argument(
        "source",
        type=str,
        help="Path to a profile folder (containing .json files) or a .tgz/.tar.gz npm package archive (must be defined in the provided config)",
    )

    # FSH Processor command
    fsh_parser = subparsers.add_parser(
        "process-fsh", help="Compile FSH project and initialize new bridge project"
    )
    fsh_parser.add_argument(
        "fsh_dir", type=str, help="Path to the FSH project directory"
    )
    fsh_parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="Name of the new project (default: name of FSH project directory)",
    )
    fsh_parser.add_argument(
        "--no-snapshots",
        dest="snapshots",
        action="store_false",
        default=True,
        help="Skip the IG-publisher (genonce) run; StructureDefinitions stay differential-only",
    )

    args = parser.parse_args()

    # A mode without --auto-mapping would silently do nothing, which reads as
    # "LLM mapping ran and found nothing" rather than "it never ran".
    if getattr(args, "auto_mapping_mode", "deterministic") != "deterministic" and not (
        getattr(args, "auto_mapping", False)
    ):
        parser.error("--auto-mapping-mode requires --auto-mapping")

    if args.command == "agent":
        if not getattr(args, "agent_action", None):
            agent_parser.error("agent requires a subcommand (currently: fix)")
        # Caught here rather than at the end of a run: an offline candidate can
        # never be applied, so asking for both is a mistake worth naming before
        # any provider call is paid for.
        if getattr(args, "apply", False) and getattr(args, "offline", False):
            agent_fix_p.error(
                "--apply cannot be combined with --offline: an offline run has "
                "no engine evidence, so its candidates are diagnostic only"
            )
        if getattr(args, "all_maps", False) and getattr(args, "map", None):
            agent_fix_p.error(
                "--all repairs every map in the project; do not also name one"
            )

    return args
