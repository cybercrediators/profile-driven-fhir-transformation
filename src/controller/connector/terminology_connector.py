import requests
from data_handling.app_state import AppState

import logging
logger = logging.getLogger(__name__)

class TerminologyConnector:
    """Connector for querying a FHIR terminology server for ValueSet expansions."""
    def __init__(self, app_state: AppState):
        self.app_state = app_state
        self.endpoint = (
            f"{app_state.conf.get("terminology_server_uri")}ValueSet/$expand"
        )
        self.headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }
    
    def query_terminology_server(
        self, vs_uri: str, filter_str=None, count=None, offset=None
    ):
        """query the terminology server, try to expand the value set"""
        params = {"url": vs_uri}
        if filter_str:
            params["filter"] = filter_str
        if count is not None:
            params["count"] = count
        if offset is not None:
            params["offset"] = offset

        try:
            response = requests.get(self.endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            vs = response.json()
            return vs.get("expansion")

        except requests.exceptions.HTTPError as errh:
            logger.warning(f"Http Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            logger.warning(f"Error Connecting: {errc}")
        except requests.exceptions.Timeout as errt:
            logger.warning(f"Terminology query: Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            logger.warning(f"Terminology query failed!: {err}")
        return None

    def extract_codes(self, vs_expanded):
        options = []
        for val in (vs_expanded or {}).get("contains", []):
            options.append(
                {
                    "code": val.get("code"),
                    "display": val.get("display"),
                    "system": val.get("system"),
                }
            )
        return options

    extract_values = extract_codes
