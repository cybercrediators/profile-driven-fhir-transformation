from pathlib import Path

from .cache_connector import CacheConnector
from helpers import utils
import logging
logger = logging.getLogger(__name__)


class DiskResourceCache(CacheConnector):
    """Resource Cache on disk"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file_path(self, url: str) -> Path:
        """Generate a file path for the cached resource based on its URL"""
        safe_filename = url.replace("://", "_").replace("/", "_")
        return self.cache_dir / f"{safe_filename}.json"

    def get_resource_from_cache(self, url: str):
        """Retrieve a resource from the disk cache by its URL"""
        cache_file = self._get_cache_file_path(url)
        if not cache_file.exists():
            logger.warning(f"Resource not found in disk cache: {url}")
            return None
        logger.info(f"Retrieving resource from disk cache: {url}")
        return utils.get_json(cache_file)

    def add_resource_to_cache(self, resource: dict):
        """Add a resource to the disk cache"""
        if "url" not in resource:
            logger.warning("No canonical url in definition! Resource was not cached")
            return None
        url = resource["url"]
        cache_file = self._get_cache_file_path(url)
        utils.store_json(resource, cache_file)
        logger.info(f"Added resource to disk cache: {url}")
        return resource

    def remove_resource_from_cache(self, url: str):
        """Remove a resource from the disk cache by its URL"""
        cache_file = self._get_cache_file_path(url)
        if cache_file.exists():
            cache_file.unlink()
            logger.info(f"Removed resource from disk cache: {url}")
        else:
            logger.warning(f"Resource not found in disk cache for removal: {url}")

    def update_resource_in_cache(self, resource: dict):
        """Update an existing resource in the disk cache"""
        return self.add_resource_to_cache(resource)

    def clear_cache(self, force: bool = False):
        """Clear all resources from the disk cache"""
        for file in self.cache_dir.glob("*.json"):
            file.unlink()
        logger.info("Cleared disk resource cache successfully!")

    def list_cached_resources(self, filter_pattern=None):
        """List all resource URLs in the disk cache, optionally filtered by a pattern.
        """
        logger.info("Listing cached resources from disk cache")
        keys = []
        for file in self.cache_dir.glob("*.json"):
            resource = utils.get_json(file)
            url = resource.get("url") if isinstance(resource, dict) else None
            if not url:
                continue
            if filter_pattern is None or filter_pattern.lower() in url.lower():
                keys.append(url)
        return keys

    def show_stats(self):
        """Show statistics about the disk cache"""
        total_files = len(list(self.cache_dir.glob("*.json")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        print(f"Disk Cache Stats\nTotal cached resources: {total_files}")
        print(f"Total cache size: {total_size / 1024:.2f} KB")
