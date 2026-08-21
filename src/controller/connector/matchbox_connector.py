import json
import requests

import logging
logger = logging.getLogger(__name__)

"""
Connect and exchange data with a matchbox server, handle matchbox requests and responses, and log errors.
"""
class MatchboxConnector:
    def __init__(self, matchbox_con: dict, headers=None):
        """Handle connections from/to the matchbox server"""
        # use the configured matchbox connection string
        if not matchbox_con:
            logger.error("No matchbox connection defined in config!")
            raise ValueError("No matchbox connection defined in config!")
        matchbox_uri = matchbox_con.get("url", "")
        if matchbox_uri and not matchbox_uri.endswith("/"):
            matchbox_uri += "/"
        self.matchbox_uri = matchbox_uri + "fhir/"
        if headers:
            self.headers = headers
        else:
            self.headers = {"Content-Type": "application/fhir+json"}

    def send_request(
        self, endpoint: str, method: str, params=None, headers=None, **kwargs
    ):
        """Send request to the matchbox server"""
        _status, body = self.send_request_detailed(
            endpoint, method, params=params, headers=headers, **kwargs
        )
        return body if _status is not None and _status < 400 else None

    def send_request_detailed(
        self, endpoint: str, method: str, params=None, headers=None, **kwargs
    ):
        """Send request and keep the response (if possible) (otherwise 4xx/5xx won't
        answer anything/dropping the response)"""

        response = None
        url = f"{self.matchbox_uri}{endpoint}"

        # Merge default and provided headers
        request_headers = self.headers.copy()
        if headers:
            request_headers.update(headers)

        try:
            response = requests.request(
                method, url, headers=request_headers, params=params, **kwargs
            )
        except requests.exceptions.RequestException as e:
            logger.error("[MATCHBOX Request] Error fetching %s: %s", url, e)
            return None, None

        if response.status_code >= 400:
            logger.error(
                "[MATCHBOX Request] %s %s returned %s", method, url, response.status_code
            )
            # Kept at error level: this is the only place the server's reason
            # appears for the callers that discard the body.
            logger.error("Response content: %s", response.text)
        if not response.content:
            return response.status_code, None
        try:
            return response.status_code, response.json()
        except json.JSONDecodeError as e:
            logger.warning(
                "Error decoding JSON from %s: %s. Response text: %s",
                url,
                e,
                response.text,
            )
            return response.status_code, response.text
