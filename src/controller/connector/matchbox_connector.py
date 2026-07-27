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
            response.raise_for_status()
            # Check for empty response body
            if response.content:
                return response.json()
            return None
        except requests.exceptions.RequestException as e:
            logger.error("[MATCHBOX Request] Error fetching %s: %s", url, e)
            if response is not None:
                logger.error(f"Response content: {response.text}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(
                "Error decoding JSON from %s: %s. Response text: %s",
                url,
                e,
                response.text,
            )
            return response.text  # Return text if not json
