import json
import valkey

from .cache_connector import CacheConnector
import logging
logger = logging.getLogger(__name__)


class ValkeyConnector(CacheConnector):
    """Use valkey as a cache for resource definitions"""

    def __init__(self, valkey_host, valkey_port, valkey_db, valkey_url=None):
        self.host = valkey_host
        self.port = valkey_port
        self.valkey_db = valkey_db
        self.valkey_url = valkey_url
        self.connection = self.connect()

    def connect(self):
        """Connect to the valkey instance"""
        try:
            if self.valkey_url is not None:
                response = valkey.Valkey.from_url(self.valkey_url)
            else:
                response = valkey.Valkey(self.host, self.port, self.valkey_db)
            response.ping()
            logger.info("Connected to valkey cache")
            return response
        except valkey.exceptions.ConnectionError as e:
            print(f"Could not connect to Valkey: {e}")
            return None

    def get_resource_from_cache(self, url):
        """Get a resource from the cache by its canonical url"""
        logger.info(f"Try to get resource: {url}")
        res = self.connection.get(url)
        if res is None:
            logger.warning(f"Could not get resource with: {url}")
            return None
        logger.info("Retrieved resource successfully!")
        return json.loads(res)

    def add_resource_to_cache(self, resource):
        """Add a resource to the cache"""
        if "url" not in resource:
            logger.warning("No canonical url in definition! was not created")
            return None

        url = resource["url"]
        logger.info(f"Try to add resource {url} to the valkey cache")

        self.connection.set(url, json.dumps(resource))
        logger.info(f"Added {url} to the valkey cache!")
        return resource

    def remove_resource_from_cache(self, url):
        """Remove a resource from the cache by its canonical url"""
        logger.info(f"Deleting the resource {url} from the valkey cache!")
        if self.connection.delete(url) > 0:
            logger.info(f"The resource {url} was deleted from the valkey cache!")
        else:
            logger.warning(f"Key {url} was not found. Nothing to do!")

    def update_resource_in_cache(self, resource):
        """Update an existing resource in the cache"""
        if "url" not in resource:
            logger.warning("No canonical url in definition! was not created")
            return None
        url = resource["url"]
        logger.info(f"Try to update resource {url} to the valkey cache")

        self.connection.set(url, json.dumps(resource))
        logger.info(f"Updated {url} to the valkey cache!")
        return resource

    def clear_cache(self, force=False):
        """Clear all resources from the cache.
        """
        self.connection.flushdb()
        logger.info("Cleared valkey cache successfully!")

    def list_cached_resources(self, filter_pattern=None):
        """List all resources in the cache, optionally filtered by a pattern"""
        logger.info("Listing cached resources from valkey")
        keys = []
        for key in self.connection.scan_iter(match=filter_pattern):
            keys.append(key.decode("utf-8"))
        return keys

    def show_stats(self):
        """Show statistics about the valkey cache"""
        if hasattr(self.connection, "dbsize"):
            size = self.connection.dbsize()
            info = self.connection.info()

            print(f"Info\nTotal keys: {size}")
            print(f"Memory used: {info.get('used_memory_human', 'N/A')}")
            print(f"Peak memory: {info.get('used_memory_peak_human', 'N/A')}")
            print(f"Connected clients: {info.get('connected_clients', 'N/A')}")
            print(f"Uptime: {info.get('uptime_in_seconds', 'N/A')} seconds")
        else:
            print("️Stats operation not supported for this cache type")
