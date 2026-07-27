"""Unit tests for DiskResourceCache."""

import pytest

from data_handling.caching.disk_cache import DiskResourceCache

pytestmark = pytest.mark.unit


@pytest.fixture
def cache(tmp_path):
    return DiskResourceCache(tmp_path / "cache")


# --------------------------------------------------------------------------- #
# add / get roundtrip
# --------------------------------------------------------------------------- #


def test_add_and_get_resource(cache):
    resource = {"resourceType": "StructureDefinition", "url": "http://example.org/sd/Patient"}
    cache.add_resource_to_cache(resource)
    result = cache.get_resource_from_cache("http://example.org/sd/Patient")
    assert result == resource


def test_add_resource_without_url_returns_none(cache):
    result = cache.add_resource_to_cache({"resourceType": "StructureDefinition"})
    assert result is None


def test_add_resource_without_url_does_not_store(cache):
    cache.add_resource_to_cache({"resourceType": "StructureDefinition"})
    assert list(cache.cache_dir.glob("*.json")) == []


def test_get_missing_resource_returns_none(cache):
    assert cache.get_resource_from_cache("http://example.org/missing") is None


# --------------------------------------------------------------------------- #
# update
# --------------------------------------------------------------------------- #


def test_update_resource_overwrites_existing(cache):
    url = "http://example.org/sd/X"
    cache.add_resource_to_cache({"url": url, "version": "1"})
    cache.update_resource_in_cache({"url": url, "version": "2"})
    result = cache.get_resource_from_cache(url)
    assert result["version"] == "2"


# --------------------------------------------------------------------------- #
# remove
# --------------------------------------------------------------------------- #


def test_remove_resource(cache):
    url = "http://example.org/sd/ToRemove"
    cache.add_resource_to_cache({"url": url})
    cache.remove_resource_from_cache(url)
    assert cache.get_resource_from_cache(url) is None


def test_remove_missing_resource_is_noop(cache):
    cache.remove_resource_from_cache("http://example.org/nonexistent")  # must not raise


# --------------------------------------------------------------------------- #
# clear_cache
# --------------------------------------------------------------------------- #


def test_clear_cache_removes_all_files(cache):
    for i in range(3):
        cache.add_resource_to_cache({"url": f"http://example.org/sd/{i}"})
    cache.clear_cache()
    assert list(cache.cache_dir.glob("*.json")) == []


def test_clear_cache_on_empty_is_noop(cache):
    cache.clear_cache()  # must not raise


# --------------------------------------------------------------------------- #
# list_cached_resources
# --------------------------------------------------------------------------- #


def test_list_cached_resources_returns_exact_urls(cache):
    # The on-disk filename encoding is lossy, so the canonical URL must be read
    # back from each resource's own "url" field, not reconstructed from the name.
    urls = ["http://example.org/sd/A", "http://example.org/sd/B"]
    for url in urls:
        cache.add_resource_to_cache({"url": url})
    assert sorted(cache.list_cached_resources()) == sorted(urls)


def test_list_cached_resources_empty(cache):
    assert cache.list_cached_resources() == []


def test_list_cached_resources_filter_pattern(cache):
    cache.add_resource_to_cache({"url": "http://example.org/sd/Patient"})
    cache.add_resource_to_cache({"url": "http://example.org/cm/gender"})
    matches = cache.list_cached_resources(filter_pattern="sd")
    assert matches == ["http://example.org/sd/Patient"]


def test_list_cached_resources_skips_files_without_url(cache):
    # A stray JSON file with no "url" field must not appear in the listing.
    cache.add_resource_to_cache({"url": "http://example.org/sd/Real"})
    (cache.cache_dir / "stray.json").write_text('{"resourceType": "Patient"}')
    assert cache.list_cached_resources() == ["http://example.org/sd/Real"]


# --------------------------------------------------------------------------- #
# _get_cache_file_path — URL encoding
# --------------------------------------------------------------------------- #


def test_cache_file_path_replaces_slashes(cache):
    url = "http://example.org/sd/Patient"
    path = cache._get_cache_file_path(url)
    assert "/" not in path.name
    assert path.suffix == ".json"


def test_cache_file_path_is_deterministic(cache):
    url = "http://example.org/sd/Patient"
    assert cache._get_cache_file_path(url) == cache._get_cache_file_path(url)


# --------------------------------------------------------------------------- #
# show_stats smoke test
# --------------------------------------------------------------------------- #


def test_show_stats_runs_without_error(cache, capsys):
    cache.add_resource_to_cache({"url": "http://example.org/sd/X"})
    cache.show_stats()
    out = capsys.readouterr().out
    assert "1" in out
