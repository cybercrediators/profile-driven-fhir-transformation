from argparse import Namespace
import logging
from pathlib import Path
from pprint import pprint
import shutil
import sys

from controller.connector.fsh_connector import SushiController
from controller.pipeline_controller.pipeline_controller import PipelineController
from controller.socket_controller.socket_controller import SocketServer
from data_handling.data_io import DataIO
from helpers import config
from view.cache_cli import CacheCLI
from view.matchbox_cli import MatchboxCLI
from view.socket_client import SocketClient

logger = logging.getLogger(__name__)


class ServiceController:
    """
    Sub-layer for managing inputs and outputs between app functions and requests/user-inputs.
    Uses the Socket Service for interactions.
    """

    def __init__(self, conf_path: str = "conf/config.json"):
        self.conf = config.load_config(conf_path)
        self._cache_cli = CacheCLI(self.conf)
        self._matchbox_cli = MatchboxCLI(self.conf)

    def handle_command(self, args: Namespace):
        """Handle service commands based on provided arguments"""
        if args.command == "cache":
            self.handle_cache_command(
                getattr(args, "cache_action", None),
                force=getattr(args, "force", False),
                res_url=getattr(args, "url", None),
                output_path=getattr(args, "output", None),
                filter_pattern=getattr(args, "filter", None),
                file_path=getattr(args, "file", None),
            )
        elif args.command == "matchbox-cli":
            self.handle_matchbox_command(
                matchbox_action=getattr(args, "matchbox_action", None),
                package_name=getattr(args, "package_name", None),
                package_version=getattr(args, "package_version", None),
                package_url=getattr(args, "package_url", None),
                package_path=getattr(args, "package_path", None),
                ig_url=getattr(args, "ig_url", None),
                ig_id=getattr(args, "ig_id", None),
                sd_path=getattr(args, "sd_path", None),
                sm_path=getattr(args, "sm_path", None),
                cm_path=getattr(args, "cm_path", None),
                input_file=getattr(args, "input_file", None),
                structure_map_url=getattr(args, "structure_map_url", None),
                bundle=getattr(args, "bundle", False),
                batch=getattr(args, "batch", False),
                resource_type=getattr(args, "resource_type", None),
                resource_id=getattr(args, "resource_id", None),
                resource_url=getattr(args, "resource_url", None),
                profile_url=getattr(args, "profile_url", None),
                output_file=getattr(args, "output_file", None),
            )

        elif args.command == "client":
            client = SocketClient(
                socket_path=getattr(args, "socket_path", "/tmp/fsh_nifi_bridge.sock"),
            )
            response = client.send_request(
                method=getattr(args, "method", None),
                params=getattr(args, "params", None),
                data=getattr(args, "data", None),
            )
            pprint(response)

        elif args.command == "server":
            if args.action == "start":
                self.start_service()
            elif args.action == "stop":
                self.stop_service()
            elif args.action == "status":
                self.service_status()

        elif args.command == "init":
            source = Path(getattr(args, "source")).resolve()
            project_path = Path(self.conf.get("project_path"))
            data_io = DataIO(project_path)
            input_profile_dir = project_path / DataIO.ProjectFolders.INPUT_PROFILE.value

            if source.is_dir():
                json_files = list(source.glob("*.json"))
                for f in json_files:
                    shutil.copy(f, input_profile_dir / f.name)
                logger.info(
                    f"Copied {len(json_files)} files from {source} to {input_profile_dir}"
                )
            elif source.suffix in (".tgz", ".gz") or str(source).endswith(".tar.gz"):
                data_io.add_tarred_profile(source)
                logger.info(f"Extracted profile from {source} to {input_profile_dir}")
            else:
                logger.error(
                    f"Source must be a folder or a .tgz/.tar.gz archive: {source}"
                )
                sys.exit(1)

            pkg_json = input_profile_dir / "package.json"
            if not pkg_json.exists():
                logger.warning(
                    "No package.json found in input_profile/ — matchbox IG installation will fail."
                )
            sys.exit(0)

        elif args.command == "process-fsh":
            fsh_dir = Path(getattr(args, "fsh_dir")).resolve()
            fsh_root = SushiController.resolve_fsh_root(fsh_dir)
            if fsh_root is None:
                logger.error(
                    f"No sushi-config.yaml found in {fsh_dir} or its parents — pass the FSH "
                    "project root (the folder containing sushi-config.yaml and input/fsh/)."
                )
                sys.exit(1)
            project_name = getattr(args, "name", None) or fsh_root.name
            projects_root = Path(self.conf.get("project_path")).parent
            sushi = SushiController(docker_image=self.conf.get("sushi_docker_image"))
            sushi.process_fsh(
                str(fsh_root),
                str(projects_root / project_name),
                project_name,
                snapshots=getattr(args, "snapshots", True),
            )

        elif args.command == "pipeline":
            self.handle_pipeline(args)

        elif args.command == "agent":
            sys.exit(self.handle_agent(args))

        elif getattr(args, "development", False):
            pc = self._make_pipeline_controller(args)
            pc.initial_processing()
            print(f"Setup validation: {pc.validate_setup()}")

        elif getattr(args, "show_config", False):
            pc = self._make_pipeline_controller(args)
            pc.display_config()

        else:
            #   pc = self._make_pipeline_controller(args)
            #   pc.initial_processing()
            logger.info(
                "No valid command provided. Enter a valid command or use --help for available commands."
            )
        sys.exit(0)

    def _make_pipeline_controller(self, args) -> PipelineController:
        """Instantiate PipelineController from parsed args, using force_overwrite as the global overwrite flag."""
        mapping_table_path = getattr(args, "mapping_table_path", "")
        if mapping_table_path:
            self.conf["mapping_table_path"] = mapping_table_path
        force = getattr(args, "force_overwrite", False)
        return PipelineController(
            conf=self.conf,
            force_overwrite=force,
            create_references=getattr(args, "create_references_in_structure_map", True),
            minimal_mode=getattr(args, "minimal_structure_map", False),
            automapping=getattr(args, "auto_mapping", False),
            auto_mapping_mode=getattr(args, "auto_mapping_mode", "deterministic"),
        )

    def handle_agent(self, args) -> int:
        """handle the agent command"""

        action = getattr(args, "agent_action", None)
        if action != "fix":
            logger.error("Unknown agent command: %s", action)
            return 2

        from agent.cli import run_agent_fix

        mapping_table_path = getattr(args, "mapping_table_path", "")
        if mapping_table_path:
            self.conf["mapping_table_path"] = mapping_table_path
        return run_agent_fix(args, self.conf)

    def handle_pipeline(self, args):
        """Route pipeline subcommands to PipelineController step methods."""
        action = getattr(args, "pipeline_action", None)
        force = getattr(args, "force_overwrite", False)

        if action in (None, "run"):
            pc = self._make_pipeline_controller(args)
            pc.initial_processing()
            if getattr(args, "prepare_matchbox", False):
                pc.prepare_matchbox_setup(force_upload=force)

        elif action == "process":
            self._make_pipeline_controller(args).run_process()

        elif action == "source-def":
            pc = self._make_pipeline_controller(args)
            source_data = getattr(args, "source_data", "")
            if source_data:
                pc.input_source_example = source_data
            pc.run_source_def()

        elif action == "static-gen-sm":
            self._make_pipeline_controller(args).run_static_gen_sm()

        elif action == "prepare-matchbox":
            self._make_pipeline_controller(args).prepare_matchbox_setup(
                force_upload=force
            )

        elif action == "export-fields":
            pc = self._make_pipeline_controller(args)
            pc.run_export_fields(
                output_file=getattr(args, "output_file", "") or None,
                roots_only=getattr(args, "roots_only", False),
            )

        elif action == "validate":
            pc = self._make_pipeline_controller(args)
            pc.run_validate(
                input_file=getattr(args, "input_file"),
                profile_url=getattr(args, "profile_url", None) or None,
            )

        elif action == "reverse-map":
            pc = self._make_pipeline_controller(args)
            pc.run_reverse_map(
                input_file=getattr(args, "input_file"),
                output_file=getattr(args, "output_file", "") or None,
                source_data=getattr(args, "source_data", "") or None,
                report_path=getattr(args, "report", "") or None,
            )

        elif action == "validate-instances":
            pc = self._make_pipeline_controller(args)
            pc.run_instance_validation(
                examples_dir=getattr(args, "examples_dir", "") or None,
                direct_only=getattr(args, "direct_only", False),
                from_element_examples=getattr(args, "from_element_examples", False),
                report_path=getattr(args, "report", "") or None,
            )

    def start_service(self):
        """Start the service controller"""
        server = SocketServer(self.conf)
        server.start_server()

    def stop_service(self):
        """Stop the service controller"""
        socket_conf = self.conf.get("socket_connection", {})
        client = SocketClient(
            socket_path=socket_conf.get("path", "/tmp/fsh_nifi_bridge.sock"),
        )
        response = client.send_request("stop")
        print(response)

    def service_status(self):
        """Get the service status"""
        socket_conf = self.conf.get("socket_connection", {})
        client = SocketClient(
            socket_path=socket_conf.get("path", "/tmp/fsh_nifi_bridge.sock"),
        )
        response = client.send_request("status")
        pprint(response)

    # delegate for external services (cache / matchbox CLIs)
    def __getattr__(self, name):
        if hasattr(self._cache_cli, name):
            return getattr(self._cache_cli, name)

        if hasattr(self._matchbox_cli, name):
            return getattr(self._matchbox_cli, name)

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )
