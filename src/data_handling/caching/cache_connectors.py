from enum import Enum
from .valkey_cache import ValkeyConnector
from .disk_cache import DiskResourceCache


class CacheConnectors(Enum):
    """Mapping of available cache connectors"""

    VALKEY = ValkeyConnector
    DISK = DiskResourceCache
