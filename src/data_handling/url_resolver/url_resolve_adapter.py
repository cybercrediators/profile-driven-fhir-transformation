from abc import ABC, abstractmethod
from enum import Enum
import json

import logging
logger = logging.getLogger(__name__)
import requests


class URLResolveAdapterBase(ABC):
    """Base class for the Resolve adapter"""

    def __init__(self, conf: dict):
        self.conf = conf

    @abstractmethod
    def query(self, url, **kwargs):
        """Query the implemented adapter"""
        raise NotImplementedError("No query function defined for this adapter!")

    def http_json_call(self, url, headers=None, params=None):
        """Send a query to the given url with the given req headers using requests"""
        response = None
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            res = response.json()
        except requests.exceptions.RequestException as e:
            logger.warning("Error fetching %s: %s", url, e)
            return None
        except json.JSONDecodeError as e:
            logger.warning("Error decoding JSON from %s: %s", url, e)
            return None

        return res


class StandardURLResolveAdapter(URLResolveAdapterBase):
    """Query an URL and request the fhir json definition"""

    def query(self, url, **kwargs):
        res = None
        headers = {"Accept": "application/fhir+json"}
        res = self.http_json_call(url, headers=headers)
        # print(res)
        if not res:
            logger.warning("Could not resolve %s", url)
            return None
        return res


class SimplifierURLResolveAdapter(URLResolveAdapterBase):
    """Query an URL and request the resource from the simplifier platform"""

    def query(self, url, **kwargs):
        api_url = kwargs.get("api_url") or self.conf.get("simplifier_api_url")
        if not api_url:
            raise ValueError("No API url in class/function params for simplifier API!")
        # split url into url and profile version
        url = url.split("|")
        headers = {"Accept": "application/fhir+json"}
        params = {"url": url[0], "_sort": "lastUpdated"}
        res = self.http_json_call(api_url, params=params, headers=headers)
        if not res:
            logger.warning("Could not resolve %s", url)
            return None
        entries = res.get("entry")
        if not entries:
            logger.warning("No entries returned for %s", url)
            return None
        if len(url) > 1:
            # both branches must return the unwrapped resource, never the entry
            return next(
                (
                    x.get("resource")
                    for x in entries
                    if (x.get("resource") or {}).get("version") == url[1]
                ),
                entries[0].get("resource"),
            )
        return entries[0].get("resource")


class Resolvers(Enum):
    """List of available resolvers"""

    STANDARD = StandardURLResolveAdapter
    SIMPLIFIER = SimplifierURLResolveAdapter
