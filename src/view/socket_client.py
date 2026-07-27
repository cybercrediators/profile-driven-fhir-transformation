import socket
import json
from typing import Any, Dict, Optional, Union
import logging
logger = logging.getLogger(__name__)


class SocketClient:
    """client for the fsh-nifi-bridge UNIX socket server"""

    def __init__(
        self,
        conf_or_type: Union[Dict[str, Any], str] = "UNIX",
        socket_path: str = "/tmp/fsh_nifi_bridge.sock",
    ):
        if isinstance(conf_or_type, dict):
            conf = conf_or_type
            self.socket_path = conf.get("path", "/tmp/fsh_nifi_bridge.sock")
        else:
            self.socket_path = socket_path

    def _connect(self) -> socket.socket:
        """establish a UNIX socket connection to the bridge server"""
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(self.socket_path)
        except FileNotFoundError:
            raise ConnectionError(
                f"UNIX socket not found at {self.socket_path}. Is the bridge server running?"
            )
        except ConnectionRefusedError:
            raise ConnectionError(
                f"Connection refused at {self.socket_path}. Is the bridge server running?"
            )
        return client

    def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        data: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """send a JSON request to the bridge server and return the parsed response"""
        request = {
            "method": method,
            "params": params or {},
            "data": data,
        }
        logger.debug(f"Socket request: method={method} params={params}")
        try:
            client = self._connect()
            with client:
                client.sendall((json.dumps(request) + "\n").encode("utf-8"))
                buffer = ""
                while True:
                    chunk = client.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8")
                    if "\n" in buffer:
                        break
                if not buffer:
                    logger.error("Empty response from bridge server.")
                    return None
                return json.loads(buffer)
        except ConnectionError as e:
            logger.error(f"Socket connection failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Error communicating with bridge server: {e}")
            return None
