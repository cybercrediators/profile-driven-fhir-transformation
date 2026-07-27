"""Cache management utilities for command-line operations"""

import sys
import json
from pathlib import Path
from data_handling.caching.cache_connectors import CacheConnectors

import logging
logger = logging.getLogger(__name__)


class CacheController:
    """Handles cache operations from command line"""

    def __init__(self, conf):
        """Initialize cache controller with configuration"""
        self.conf = conf
        try:
            # check if specified cache service is supported
            if conf.get("external_cache_service") not in CacheConnectors._member_names_:
                raise KeyError(
                    f"Cache service {conf.get('external_cache_service')} not supported"
                )
            # check if cache service is
            self.cache = CacheConnectors[conf.get("external_cache_service")].value(
                **conf.get("cache_args", {})
            )
        except KeyError as e:
            logger.error(
                "Cache %s does not have an implementation ready: %s",
                conf.get("external_cache_service"),
                e,
            )
            sys.exit(1)

    def clear_cache(self, force=False):
        """Clear all resources from cache"""
        if not force:
            response = input(
                "Are you sure you want to clear ALL cached resources? (yes/no): "
            )
            if response.lower() not in ["yes", "y"]:
                print("Cache clear cancelled.")
                return

        try:
            self.cache.clear_cache(force=force)
            print("All cached resources have been removed.")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            print(f"Error clearing cache: {e}")
            sys.exit(1)

    def get_resource(self, url, output_file=None):
        """Get a resource from cache by URL"""
        try:
            resource = self.cache.get_resource_from_cache(url)

            if resource is None:
                print(f"Resource not found: {url}")
                return

            # Output to file or console
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(resource, f, indent=2)
                print(f"Resource saved to: {output_file}")
            else:
                print(json.dumps(resource, indent=2))

        except Exception as e:
            logger.error(f"Error retrieving resource: {e}")
            print(f"Error retrieving resource: {e}")
            sys.exit(1)

    def delete_resource(self, url):
        """Delete a resource from cache by URL"""
        try:
            # Check if resource exists first
            resource = self.cache.get_resource_from_cache(url)
            if resource is None:
                print(f"Resource not found: {url}")
                return

            self.cache.remove_resource_from_cache(url)
            print(f"Resource deleted: {url}")

        except Exception as e:
            logger.error(f"Error deleting resource: {e}")
            print(f"Error deleting resource: {e}")
            sys.exit(1)

    def list_resources(self, filter_pattern=None):
        """List all cached resource URLs"""
        try:
            urls = self.cache.list_cached_resources(filter_pattern=filter_pattern)
            if not urls:
                if filter_pattern:
                    print(f"No cached resources matching '{filter_pattern}'")
                else:
                    print("No cached resources found")
                return

            print(f"Found {len(urls)} cached resource(s):\n")
            urls = [u.decode("utf-8") if isinstance(u, bytes) else u for u in urls]
            for url in sorted(urls):
                print(f"{url}")

        except Exception as e:
            logger.error(f"Error listing resources: {e}")
            print(f"Error listing resources: {e}")
            sys.exit(1)

    def show_stats(self):
        """Show cache statistics"""
        try:
            self.cache.show_stats()
        except Exception as e:
            logger.error(f"Error showing cache stats: {e}")
            print(f"Error showing cache stats: {e}")
            sys.exit(1)

    def add_resource(self, file_path):
        """Add a resource to cache from file"""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                print(f"File not found: {file_path}")
                sys.exit(1)

            # Load resource from file
            with open(file_path, "r", encoding="utf-8") as f:
                resource = json.load(f)

            # Validate resource has URL
            if "url" not in resource:
                print("Resource does not have a canonical 'url' field")
                sys.exit(1)

            # Add to cache
            result = self.cache.add_resource_to_cache(resource)

            if result:
                print(f"Resource added to cache: {resource['url']}")
            else:
                print("Failed to add resource to cache")

        except json.JSONDecodeError as e:
            print(f"Invalid JSON file: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error adding resource: {e}")
            print(f"Error adding resource: {e}")
            sys.exit(1)
