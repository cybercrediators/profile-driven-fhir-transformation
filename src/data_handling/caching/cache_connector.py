from abc import ABC, abstractmethod


class CacheConnector(ABC):
    """Base class for an external cache service"""

    @abstractmethod
    def get_resource_from_cache(self, url: str):
        """Retrieve a resource from the cache"""
        raise NotImplementedError(
            "No add to cache function implemented in cache connector!"
        )

    @abstractmethod
    def add_resource_to_cache(self, resource: dict):
        """Add a resource to the cache"""
        raise NotImplementedError(
            "No add to cache function implemented in cache connector!"
        )

    @abstractmethod
    def remove_resource_from_cache(self, url: str):
        """Remove a resource from the cache (if applicable)"""
        raise NotImplementedError(
            "No remove from cache function implemented in cache connector!"
        )

    @abstractmethod
    def update_resource_in_cache(self, resource: dict):
        """Update a resource in the cache"""
        raise NotImplementedError(
            "No update resource in cache function implemented in cache connector!"
        )

    @abstractmethod
    def clear_cache(self, force: bool = False):
        """Clear all resources from the cache"""
        raise NotImplementedError(
            "No clear cache function implemented in cache connector!"
        )

    @abstractmethod
    def list_cached_resources(self, filter_pattern: str = None):
        """List all cached resource keys"""
        raise NotImplementedError(
            "No list cached resources function implemented in cache connector!"
        )

    @abstractmethod
    def show_stats(self):
        """Show cache statistics"""
        raise NotImplementedError(
            "No show stats function implemented in cache connector!"
        )
