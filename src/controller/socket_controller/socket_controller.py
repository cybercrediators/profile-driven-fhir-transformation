import socket
import json
import os
import sys
import threading
import multiprocessing
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import logging
logger = logging.getLogger(__name__)
from controller.pipeline_controller.pipeline_controller import PipelineController
from controller.bundle_service import BundleService
from controller.socket_controller.socket_controller_process import serve_forever_process


class SocketServer:
    def __init__(
        self,
        conf: dict,
        shared_validation_status=None,
        shared_validation_lock=None,
        run_validation_scheduler: bool = True,
    ):
        self.pipeline_controller = PipelineController(conf=conf)
        self.app_state = self.pipeline_controller.app_state
        # Populate registry so BundleService auto-wiring and status reporting work correctly.
        self.pipeline_controller._ensure_processed()
        self.socket_conf = self.app_state.conf.get("socket_connection", {})
        # process_count controls number of forked workers; thread_count controls per-process thread pool
        self.worker_processes = int(self.socket_conf.get("worker_processes", 1))
        self.thread_workers = int(self.socket_conf.get("max_workers", 8))
        self.validation_interval = int(self.socket_conf.get("validation_interval", 0))
        self.running = True
        self._stop_event = threading.Event()
        self._mp_stop_event = (
            multiprocessing.Event() if self.worker_processes > 1 else None
        )
        self._executor = ThreadPoolExecutor(max_workers=self.thread_workers)
        self._state_lock = threading.Lock()
        self._mutating_methods = {
            "prepare_matchbox",
            "sync_resources",
            "transform_data",
            "transform_batch",
            "validate_data",
        }
        self._worker_pool = []
        self._validation_thread = None
        self._run_validation_scheduler = run_validation_scheduler

        # shared validation state across workers
        self._validation_status = (
            shared_validation_status
            if shared_validation_status is not None
            else {
                "ok": None,
                "timestamp": None,
                "error": None,
            }
        )
        self._validation_status_lock = (
            shared_validation_lock
            if shared_validation_lock is not None
            else threading.Lock()
        )

        self.dispatcher = {
            "status": self.get_status,
            "stop": self.stop_server,
            "transform_data": self.transform_data,
            "transform_batch": self.transform_batch,
            "validate_data": self.validate_data,
            "validate_setup": self.validate_setup,
            "prepare_matchbox": self.prepare_matchbox_setup,
            "sync_resources": self.sync_resources,
        }

    def start_server(self):
        """Create and start the UNIX socket server (path from socket_connection.path)."""
        socket_path = self.socket_conf.get("path", "/tmp/fsh_nifi_bridge.sock")
        # Ensure the socket does not already exist
        if os.path.exists(socket_path):
            os.remove(socket_path)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        os.chmod(socket_path, 0o600) # set socket permissions
        logger.info(f"Socket server listening on UNIX socket: {socket_path}")

        server.listen(5)
        server.settimeout(1.0)

        if self.worker_processes > 1:
            server_fd = server.fileno()
            os.set_inheritable(server_fd, True)

            manager = multiprocessing.Manager()
            shared_validation_status = manager.dict(
                {"ok": None, "timestamp": None, "error": None}
            )
            shared_validation_lock = manager.Lock()

            for idx in range(self.worker_processes):
                logger.info(
                    f"Starting worker process {idx + 1}/{self.worker_processes}..."
                )
                run_validator = (
                    idx == 0
                )  # only the first worker runs periodic validation
                p = multiprocessing.Process(
                    target=serve_forever_process,
                    args=(
                        server_fd,
                        self.app_state.conf,
                        self._mp_stop_event,
                        shared_validation_status,
                        shared_validation_lock,
                        run_validator,
                        self.validation_interval,
                    ),
                    daemon=True,
                )
                p.start()
                self._worker_pool.append(p)

            # Parent monitors stop event
            while not self._mp_stop_event.is_set():
                time.sleep(0.5)

            logger.info("Stopping worker processes...")
            for p in self._worker_pool:
                p.join(timeout=2)
            server.close()
            sys.exit(0)
        else:
            logger.info("Starting worker process 1/1...")

        # Start optional periodic validation in single-process mode only
        if self.validation_interval > 0 and self._run_validation_scheduler:
            self._start_validation_scheduler()

        try:
            while not self._stop_event.is_set():
                try:
                    conn, _ = server.accept()
                    self._executor.submit(self.handle_connection_safe, conn)
                except socket.timeout:
                    continue
                except OSError as e:
                    if self._stop_event.is_set():
                        break
                    logger.error(f"Server socket error: {e}")
                    break
                except KeyboardInterrupt:
                    logger.info("Received interrupt, shutting down server loop...")
                    break
                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.error(f"Error in server loop: {e}")
        finally:
            self._stop_event.set()

        logger.info("Server shutting down.")
        server.close()
        self._executor.shutdown(wait=True)
        sys.exit(0)

    def handle_connection_safe(self, conn):
        try:
            with conn:
                self._handle_connection(conn)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unhandled connection error: {e}")

    def _handle_connection(self, conn):
        """Handle an incoming client connection"""
        try:
            # Read data in a loop until newline
            buffer = ""
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8")
                if "\n" in buffer:
                    break

            if not buffer:
                return

            request = json.loads(buffer)
            logger.info(f"Received request: {request}")

            method = request.get("method")
            params = request.get("params", {})
            params = json.loads(params) if isinstance(params, str) else params
            request_data = request.get("data")

            if not isinstance(params, dict):
                response = {
                    "status": "error",
                    "message": "params must be a JSON object",
                }
            elif not method:
                response = {"status": "error", "message": "Missing method in request"}
            elif method in self.dispatcher:
                validation_error = self._get_validation_error(method)
                if validation_error:
                    response = {"status": "error", "message": validation_error}
                else:
                    call_params = dict(params)
                    if request_data is not None and "data" not in call_params:
                        call_params["data"] = request_data
                    if method in self._mutating_methods:
                        with self._state_lock:
                            response_data = self.dispatcher[method](**call_params)
                    else:
                        response_data = self.dispatcher[method](**call_params)
                    response = {"status": "success", "data": response_data}
            else:
                response = {"status": "error", "message": f"Unknown method: {method}"}

        except json.JSONDecodeError:
            response = {"status": "error", "message": "Invalid JSON request"}
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            response = {"status": "error", "message": str(e)}

        conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

    def prepare_matchbox_setup(self):
        """Prepare the matchbox setup for the application"""
        return self.pipeline_controller.prepare_matchbox_setup()

    def validate_setup(self):
        """Check-only validation: verifies local files and matchbox presence without uploading."""
        result = bool(self.pipeline_controller.validate_setup(read_only=True))
        self._update_validation_status(
            ok=result, error=None if result else "setup check failed"
        )
        if result:
            logger.info("Setup validation succeeded")
        else:
            logger.error("Setup validation failed")
        return result

    def sync_resources(self, force: bool = False):
        """Re-upload resources that have changed on disk since the last upload.

        force=True: re-upload all resources regardless of whether they changed
                    (equivalent to prepare_matchbox with force_upload=True).
        """
        if force:
            logger.info("Force-syncing all resources to matchbox...")
            changed = self.pipeline_controller.prepare_matchbox_setup(force_upload=True)
        else:
            logger.info("Syncing changed resources to matchbox...")
            changed = self.pipeline_controller.sync_changed_resources()
        return {"changed": changed}

    def validate_data(self, data: str, profile_url: Optional[str] = None):
        """
        Validate a FHIR resource against an optional profile via matchbox.
        If profile_url is omitted the bridge derives it from the resource type,
        or matchbox falls back to base-type validation.
        """
        matchbox_con = self.app_state.conf.get("matchbox_connection", {})
        if not matchbox_con:
            raise ValueError("No matchbox connection defined in config!")

        parsed = self._small_json_conv(data)
        if not parsed:
            raise RuntimeError("Failed to parse input data!")
        response = self.pipeline_controller.validate_data(
            resource_obj=parsed,
            profile_url=profile_url or "",
        )
        if not response:
            logger.error("No response from matchbox for validation request!")
            return RuntimeError("Failed execution of the data validation!")

        return response

    def transform_data(
        self, data: str, structure_map_url: str = None, bundle: bool = False
    ):
        """
        Transform input resource data based on given profile through matchbox.
        :param data: FHIR resource to be transformed as dict (json string)
        :param structure_map_url: (Optional) specific structure map to use. If None, uses all.
        :param bundle: If True, wrap results in a FHIR Bundle. Default False (returns list).
        """
        matchbox_con = self.app_state.conf.get("matchbox_connection", {})
        if not matchbox_con:
            raise ValueError("No matchbox connection defined in config!")

        parsed = self._small_json_conv(data)
        if not parsed:
            raise RuntimeError("Failed to parse input data!")

        map_outputs = {}
        resources = self.pipeline_controller.transform_data(
            input_data=parsed,
            structure_map_url=structure_map_url,
            map_outputs=map_outputs,
        )
        if not resources:
            logger.error("No response from matchbox for transformation request!")
            raise RuntimeError("Failed execution of the data transformation!")

        if bundle:
            return self._build_bundle_with_context(
                resources,
                source_record=parsed if isinstance(parsed, dict) else None,
                map_outputs=map_outputs,
            )
        return resources

    def transform_batch(
        self, data: str, structure_map_url: str = None, bundle: bool = False
    ):
        """
        Transform a JSON array of source objects through matchbox.
        :param data: JSON array of source objects (json string)
        :param structure_map_url: (Optional) specific structure map to use. If None, uses all.
        :param bundle: If True, wrap each result in a FHIR Bundle. Default False (returns list of lists).
        """
        matchbox_con = self.app_state.conf.get("matchbox_connection", {})
        if not matchbox_con:
            raise ValueError("No matchbox connection defined in config!")

        parsed = self._small_json_conv(data)
        if not isinstance(parsed, list):
            raise ValueError("transform_batch requires a JSON array as input!")

        batch = []
        for item in parsed:
            map_outputs = {}
            resources = self.pipeline_controller.transform_data(
                input_data=item,
                structure_map_url=structure_map_url,
                map_outputs=map_outputs,
            )
            if resources is not None:
                batch.append((item, resources, map_outputs))
        if not batch:
            logger.error("Batch transformation produced no results!")
            raise RuntimeError("Failed execution of the batch data transformation!")

        if bundle:
            return [
                self._build_bundle_with_context(
                    resources, source_record=item, map_outputs=map_outputs
                )
                for item, resources, map_outputs in batch
            ]
        return [resources for _, resources, _ in batch]

    def _build_bundle_with_context(
        self, resources: list, source_record: dict = None, map_outputs: dict = None
    ) -> dict:
        """Create a FHIR Bundle with reference resolution using loaded registry and structure maps."""
        structure_maps = []
        for url in self.pipeline_controller.structure_map_urls:
            sm = self.app_state.cache.get_resource_from_cache(url)
            if sm:
                structure_maps.append(sm)
        return BundleService.create_bundle(
            resources,
            registry=self.app_state.registry,
            structure_maps=structure_maps,
            plugins=self.pipeline_controller.plugins,
            source_record=source_record,
            map_outputs=map_outputs,
            external_reference_defaults=self.app_state.conf.get(
                "external_reference_defaults", []
            ),
        )

    def get_status(self):
        """Return the current status of the application"""
        with self._validation_status_lock:
            validation_snapshot = dict(self._validation_status)
        return {
            "service": "FSH-NiFi Bridge",
            "status": "running" if self.running else "stopped",
            "processed_resources": len(self.app_state.registry.registry_objects),
            "root_resources": [
                obj.data.url
                for obj in self.app_state.registry.registry_objects.values()
                if obj.is_root
            ],
            "validate_setup": validation_snapshot,
        }

    def stop_server(self):
        """Stop the socket server"""
        self.running = False
        self._stop_event.set()
        if self._mp_stop_event:
            self._mp_stop_event.set()
        if self._validation_thread and self._validation_thread.is_alive():
            self._validation_thread.join(timeout=1.0)
        return {"message": "Server is shutting down."}

    def _small_json_conv(self, data: str) -> dict:
        """Convert small JSON strings to dict safely"""
        try:
            return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to parse JSON data: {e}")
            return {}

    def _start_validation_scheduler(self):
        def _loop():
            while not self._stop_event.is_set():
                with self._state_lock:
                    try:
                        self.pipeline_controller.sync_changed_resources()
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Periodic resource sync failed: {e}")
                    try:
                        self.validate_setup()
                    except Exception as e:  # noqa: BLE001
                        logger.error(f"Periodic validation failed: {e}")
                self._stop_event.wait(timeout=self.validation_interval)

        # run once immediately to populate shared state
        try:
            with self._state_lock:
                self.pipeline_controller.sync_changed_resources()
                self.validate_setup()
        except Exception as e:
            logger.error(f"Initial validation failed: {e}")

        self._validation_thread = threading.Thread(
            target=_loop, daemon=True, name="validation-scheduler"
        )
        self._validation_thread.start()

    def _update_validation_status(self, ok: bool, error: Optional[str] = None):
        with self._validation_status_lock:
            self._validation_status["ok"] = bool(ok)
            self._validation_status["timestamp"] = time.time()
            self._validation_status["error"] = error if not ok else None

    def _get_validation_error(self, method: str) -> Optional[str]:
        """Return a validation error message for blocked methods, if any."""
        if method in {"status", "validate_setup", "sync_resources", "stop"}:
            return None
        with self._validation_status_lock:
            ok = self._validation_status.get("ok")
            error = self._validation_status.get("error")
        if ok is False:
            return error or "Validation failed; request blocked."
        return None
