import pytest

# Skeletons for future FML automapper tests. Skipped until fixtures are added.


@pytest.mark.skip(reason="TODO: add fixtures for source/target fields and embeddings")
def test_automapper_prefers_exact_matches():
    """Exact field name/path matches should score highest."""
    ...


@pytest.mark.skip(reason="TODO: add fixtures for weighted semantic matches")
def test_automapper_weights_name_path_description():
    """Semantic scoring should respect configured weights (name > path > description)."""
    ...


@pytest.mark.skip(reason="TODO: add fixtures for virtual path expansion")
def test_automapper_handles_virtual_paths():
    """Composite types should be flattened into virtual paths during matching."""
    ...
