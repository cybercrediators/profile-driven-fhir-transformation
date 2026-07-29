from data_handling.registry.registry import Registry
from data_handling.caching.cache_connector import CacheConnector
from dataclasses import dataclass
from typing import Any

from data_handling.data_io import DataIO


@dataclass
class AppState:
    """Container for config, registry, and cache connection"""

    conf: dict
    registry: Registry
    cache: CacheConnector
    dataIO: DataIO
    fhir_spec_context: Any = None
