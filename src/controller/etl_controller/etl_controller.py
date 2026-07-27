from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from controller.etl_controller.validation_result import ValidationResult
from view.socket_client import SocketClient

import logging
logger = logging.getLogger(__name__)

class ETLConnector(ABC):
    """abstract base class for ETL tool integrations (e.g NiFi, Airflow, etc.), through socket"""

    def __init__(self, socket_conf: Dict[str, Any]):
        self.socket_conf = socket_conf
        self.socket_client = SocketClient(socket_conf)

    @abstractmethod
    def validate_setup(self) -> bool:
        """
        Check if service is reachable and its matchbox setup is validated
        """
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """return the current status of the connector"""
        raise NotImplementedError

    @abstractmethod
    def ingest_source_data(self, raw_payload: bytes) -> Any:
        """parse payload from the input tool into source dict to prepare for transformation"""
        raise NotImplementedError

    @abstractmethod
    def transform(
        self,
        payload: Any,
        bundle: bool = True,
        structure_map_url: Optional[str] = None,
    ) -> Any:
        """transform single resource via the connector tool

        bundle=True (default): returns a FHIR transaction Bundle dict
        bundle=False: returns a list of resource dicts
        structure_map_url: optional override to run a specific map only
        """
        raise NotImplementedError

    @abstractmethod
    def transform_batch(
        self,
        payloads: List[Any],
        bundle: bool = True,
    ) -> List[Any]:
        """transform list of input objects via connector tool"""
        raise NotImplementedError

    @abstractmethod
    def validate_resource(
        self,
        resource: Dict[str, Any],
        profile_url: Optional[str] = None,
    ) -> "ValidationResult":
        """validate single resource against optional profile via matchbox"""
        raise NotImplementedError

    @abstractmethod
    def return_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """serialize data for passing back to ETL tool"""
        raise NotImplementedError

    @abstractmethod
    def emit_log(self, level: str, message: str, **details) -> None:
        """route logger message to ETL tool logging"""
        raise NotImplementedError

    @abstractmethod
    def emit_error(self, message: str, **details) -> Any:
        """report errors back to ETL tool"""
        raise NotImplementedError

    def _socket_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
    ) -> Any:
        """send a request to the connector socket server and return the response data"""
        try:
            response = self.socket_client.send_request(
                method=method, params=params, data=data
            )
            if not response or response.get("status") != "success":
                msg = (
                    response.get("message")
                    if isinstance(response, dict)
                    else "Unknown socket error"
                )
                raise RuntimeError(f"Socket call '{method}' failed: {msg}")
            return response.get("data")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Socket request failed for method '{method}': {exc}")
            return self.emit_error(str(exc))
