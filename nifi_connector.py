"""
NiFi integration for fsh-nifi-bridge.

Deploy this file alongside the bridge source into NiFi's Python processor directory.
Requires the nifiapi package provided by the NiFi Python processor framework.
"""
import json
import os
import sys
from typing import Any, Dict, List, Optional

from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult  # type: ignore[import]
from nifiapi.properties import PropertyDescriptor, StandardValidators  # type: ignore[import]

# make source importable for NiFi BEFORE importing the bridge — NiFi runs each Python
# processor in an isolated venv that does NOT inherit the container's PYTHONPATH, so the
# bridge src dir must be put on sys.path here, ahead of the `controller.*` imports.
_BRIDGE_SRC = os.environ.get("BRIDGE_SRC", "/opt/bridge-src")
if os.path.isdir(_BRIDGE_SRC) and _BRIDGE_SRC not in sys.path:
    sys.path.insert(0, _BRIDGE_SRC)

from controller.etl_controller.etl_controller import ETLConnector
from controller.etl_controller.validation_result import ValidationResult

class NiFiETLConnector(ETLConnector):
    """
    ETLConnector implementation that translates bridge calls to NiFi idioms.
    Instantiated once per processor instance and shared across FlowFiles.
    """

    def __init__(self, socket_path: str, nifi_logger):
        super().__init__({"path": socket_path})
        self.nifi_logger = nifi_logger

    def validate_setup(self) -> bool:
        """Check that the bridge is reachable and its matchbox setup is valid."""
        result = self._socket_request("validate_setup")
        if isinstance(result, dict) and result.get("error"):
            return False
        return bool(result)

    def get_status(self) -> Dict[str, Any]:
        """Return the current bridge status dict (processed_resources, validation state, etc.)."""
        result = self._socket_request("status")
        return result if isinstance(result, dict) else {}

    def ingest_source_data(self, raw_payload: bytes) -> Any:
        """Parse raw bytes from the ETL tool into a source dict ready for transformation."""
        if not raw_payload:
            return self.emit_error("Empty FlowFile content")
        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except Exception as exc:
            return self.emit_error(f"Failed to parse FlowFile as JSON: {exc}")
        if isinstance(payload, list):
            return self.emit_error(
                "FlowFile contains a JSON array — the transformer expects one source "
                "record per FlowFile. Split upstream (e.g. SplitJson with $[*])."
            )
        if isinstance(payload, dict):
            payload = {k: v for k, v in payload.items() if v not in ("", None)}
        return payload

    def transform(
        self,
        payload: Any,
        bundle: bool = True,
        structure_map_url: Optional[str] = None,
    ) -> Any:
        """Transform a single source record via the bridge."""
        params: Dict[str, Any] = {"bundle": bundle}
        if structure_map_url:
            params["structure_map_url"] = structure_map_url
        return self._socket_request("transform_data", params=params, data=json.dumps(payload))

    def transform_batch(self, payloads: List[Any], bundle: bool = True) -> List[Any]:
        """Transform a list of source records via the bridge."""
        result = self._socket_request(
            "transform_batch",
            params={"bundle": bundle},
            data=json.dumps(payloads),
        )
        return result if isinstance(result, list) else []

    def validate_resource(
        self,
        resource: Dict[str, Any],
        profile_url: Optional[str] = None,
    ) -> ValidationResult:
        """Validate a resource dict against its profile via the bridge."""
        params: Dict[str, Any] = {}
        if profile_url:
            params["profile_url"] = profile_url
        raw = self._socket_request("validate_data", params=params, data=json.dumps(resource))
        if isinstance(raw, dict) and raw.get("error"):
            return ValidationResult(
                is_valid=False,
                resource_type=resource.get("resourceType"),
                profile_url=profile_url,
                issues=[{"severity": "error", "diagnostics": raw["error"]}],
            )
        issues = raw.get("issue", []) if isinstance(raw, dict) else []
        is_valid = not any(i.get("severity") in ("error", "fatal") for i in issues)
        return ValidationResult(
            is_valid=is_valid,
            resource_type=resource.get("resourceType"),
            profile_url=profile_url,
            issues=issues,
        )

    def return_data(self, data: Any, _metadata: Optional[Dict[str, Any]] = None) -> Any:  # noqa: ARG002
        """Serialize / format data for emission back to Nifi"""
        return data if data is not None else {}

    def emit_log(self, level: str, message: str, **details) -> None:
        """Route a log message to Nifi logging"""
        full_msg = message if not details else f"{message} {details}"
        lvl = level.lower()
        if lvl == "debug":
            self.nifi_logger.debug(full_msg)
        elif lvl in ("warn", "warning"):
            self.nifi_logger.warn(full_msg)
        elif lvl == "error":
            self.nifi_logger.error(full_msg)
        else:
            self.nifi_logger.info(full_msg)

    def emit_error(self, message: str, **details) -> Dict[str, Any]:
        """Report an error to Nifi and return a structured error payload."""
        self.nifi_logger.error(message)
        return {"error": message, **details}

class FHIRTransformerProcessor(FlowFileTransform):
    """
    NiFi Python processor that transforms source data to FHIR via fsh-nifi-bridge.

    Relationships:
      success          — transform (and optional validation) succeeded
      failure          — parse or transform error; original FlowFile content passed through
      validation_failure — transform succeeded but one or more resources failed validation
                          (only active when 'Validate Output' is enabled)
    """

    class Java:
        implements = ["org.apache.nifi.python.processor.FlowFileTransform"]

    class ProcessorDetails:
        version = "1.0.0"
        description = (
            "Transforms incoming source data to FHIR using the fsh-nifi-bridge "
            "socket server and matchbox. Optionally validates the output against "
            "the configured profiles."
        )
        tags = ["FHIR", "StructureMap", "matchbox", "bridge", "transform"]
        # set external dependencies
        dependencies = ["logger==1.4"]

    SOCKET_PATH = PropertyDescriptor(
        name="Socket Path",
        description="Path to the fsh-nifi-bridge UNIX domain socket.",
        default_value="/tmp/fsh_nifi_bridge.sock",
        required=True,
        validators=[StandardValidators.NON_EMPTY_VALIDATOR],
    )

    BUNDLE_OUTPUT = PropertyDescriptor(
        name="Bundle Output",
        description=(
            "When enabled, wrap transformed resources in a FHIR transaction Bundle "
            "before passing to the success relationship. "
            "Disable to receive a raw JSON array of resource objects instead."
        ),
        allowable_values=["true", "false"],
        default_value="true",
        required=True,
    )

    VALIDATE_OUTPUT = PropertyDescriptor(
        name="Validate Output",
        description=(
            "When enabled, each resource in the transform output is validated against "
            "its profile via matchbox before routing. Resources that fail validation "
            "are routed to the validation_failure relationship. "
            "Adds one matchbox round-trip per resource per FlowFile."
        ),
        allowable_values=["true", "false"],
        default_value="false",
        required=True,
    )

    property_descriptors = [SOCKET_PATH, BUNDLE_OUTPUT, VALIDATE_OUTPUT]

    def getPropertyDescriptors(self):
        return self.property_descriptors

    def __init__(self, **kwargs):
        super().__init__()
        self._connector: Optional[NiFiETLConnector] = None
        self._setup_valid: Optional[bool] = None

    def _get_connector(self, context) -> NiFiETLConnector:
        """Lazily create connector from current context properties."""
        if self._connector is None:
            socket_path = context.getProperty(self.SOCKET_PATH).getValue()
            self._connector = NiFiETLConnector(
                socket_path=socket_path,
                nifi_logger=self.logger,
            )
        return self._connector

    def _ensure_setup_valid(self, connector: NiFiETLConnector) -> bool:
        """Check bridge setup, caching the result for the processor lifecycle."""
        if self._setup_valid is None:
            self._setup_valid = connector.validate_setup()
        return bool(self._setup_valid)

    def transform(self, context, flowfile):
        """NiFi entry point for processing a FlowFile through the bridge."""
        connector = self._get_connector(context)
        bundle_output = context.getProperty(self.BUNDLE_OUTPUT).getValue() == "true"
        validate_output = context.getProperty(self.VALIDATE_OUTPUT).getValue() == "true"

        # Read incoming FlowFile content
        try:
            incoming_bytes = flowfile.getContentsAsBytes()
        except Exception:
            incoming_bytes = b""

        # Verify bridge availability (cached per processor lifecycle; invalidated on failure)
        if not self._ensure_setup_valid(connector):
            self._setup_valid = None  # retry next FlowFile
            return FlowFileTransformResult(
                relationship="failure",
                contents=incoming_bytes,
                attributes={"fsh_bridge_error": "bridge setup validation failed"},
            )

        # Parse source data
        source_payload = connector.ingest_source_data(incoming_bytes)
        if source_payload is None or (
            isinstance(source_payload, dict) and source_payload.get("error")
        ):
            err = (
                source_payload.get("error", "ingest failed")
                if isinstance(source_payload, dict)
                else "empty or unparseable payload"
            )
            return FlowFileTransformResult(
                relationship="failure",
                contents=incoming_bytes,
                attributes={"fsh_bridge_error": err},
            )

        # Transform
        transformed = connector.transform(source_payload, bundle=bundle_output)
        if transformed is None or (
            isinstance(transformed, dict) and transformed.get("error")
        ):
            err = (
                transformed.get("error", "transform failed")
                if isinstance(transformed, dict)
                else "no transform output"
            )
            return FlowFileTransformResult(
                relationship="failure",
                contents=incoming_bytes,
                attributes={"fsh_bridge_error": err},
            )

        # Optional per-resource validation
        if validate_output:
            resources = self._extract_resources(transformed, bundle_output)
            validation_errors = []
            for resource in resources:
                result = connector.validate_resource(resource)
                if not result.is_valid:
                    rt = resource.get("resourceType", "unknown")
                    error_count = sum(
                        1 for i in result.issues if i.get("severity") in ("error", "fatal")
                    )
                    validation_errors.append(f"{rt}: {error_count} error(s)")

            if validation_errors:
                try:
                    outbound_bytes = json.dumps(connector.return_data(transformed)).encode("utf-8")
                except Exception:
                    outbound_bytes = incoming_bytes
                return FlowFileTransformResult(
                    relationship="validation_failure",
                    contents=outbound_bytes,
                    attributes={
                        "fsh_bridge_status": "validation_failed",
                        "fsh_bridge_resource_count": str(len(resources)),
                        "fsh_bridge_validation_errors": "; ".join(validation_errors),
                    },
                )

        # Success
        resource_count = len(self._extract_resources(transformed, bundle_output))
        try:
            outbound_bytes = json.dumps(connector.return_data(transformed)).encode("utf-8")
        except Exception:
            outbound_bytes = incoming_bytes

        return FlowFileTransformResult(
            relationship="success",
            contents=outbound_bytes,
            attributes={
                "fsh_bridge_status": "ok",
                "fsh_bridge_resource_count": str(resource_count),
            },
        )

    @staticmethod
    def _extract_resources(transformed: Any, bundle_output: bool) -> List[Dict[str, Any]]:
        """Pull resource dicts out of a Bundle or raw list for per-resource operations."""
        if bundle_output and isinstance(transformed, dict):
            return [
                e["resource"]
                for e in transformed.get("entry", [])
                if isinstance(e.get("resource"), dict)
            ]
        if isinstance(transformed, list):
            return [r for r in transformed if isinstance(r, dict)]
        return []
