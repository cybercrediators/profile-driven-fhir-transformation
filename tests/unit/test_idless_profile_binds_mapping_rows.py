"""A StructureDefinition without `id` must still bind its mapping-table rows.

`Resource.id` is 0..1 and published IGs do ship profiles without one — every
StructureDefinition in `fhir.bfarm.de` (DiGA/DiPA) lacks it. The mapping-table lookup
was keyed on `id` alone, so for such a profile the key was `None.<path>`, no row could
ever match, and every rule fell back to its `TODO_MAP_*` placeholder. The transform
then emitted a well-formed, profile-stamped, **empty** resource and reported zero
errors — the one failure mode in this pipeline that produces no signal at all.

The identity used here is the same one the map name, the on-disk file name and the
target alias already derive from the canonical url.
"""

import pytest

from helpers.utils import resource_identity

pytestmark = pytest.mark.unit


class _IdlessProfile:
    """Shaped like the bfarm profiles: a canonical url, no `id`."""

    id = None
    url = "https://fhir.bfarm.de/StructureDefinition/HealthAppCatalogEntry"
    name = "Profile-HealthAppCatalogEntry"
    type = "CatalogEntry"


class _NormalProfile:
    id = "UKCore-Immunization"
    url = "https://fhir.hl7.org.uk/StructureDefinition/UKCore-Immunization"
    name = "UKCoreImmunization"
    type = "Immunization"


def test_idless_profile_gets_a_usable_identity():
    """Not None — a None identity is what made every row unbindable."""
    identity = resource_identity(_IdlessProfile())
    assert identity, "an id-less profile must still yield a mapping-table key"
    assert identity == "HealthAppCatalogEntry"


def test_identity_prefers_id_when_present():
    """Profiles that do carry an id keep it, so existing tables are unaffected."""
    assert resource_identity(_NormalProfile()) == "UKCore-Immunization"


def test_lookup_key_for_an_idless_profile_is_not_none_prefixed():
    """The factory builds `f"{profile}.{path}"`; `None.` can never match a row."""
    identity = resource_identity(_IdlessProfile())
    key = f"{identity}.status"
    assert not key.startswith("None."), key
    assert key == "HealthAppCatalogEntry.status"


def test_group_creation_binds_the_profile_identity_not_the_missing_id():
    """Guards the assignment itself, which is what regressed.

    The tests above only pin `resource_identity`, which was already correct — the
    defect was that group creation read `data.id` directly. Only the prefix
    assignment is exercised: generation is stubbed out and whatever it raises
    afterwards is irrelevant to what is being asserted.
    """
    from types import SimpleNamespace

    from mapping.fml_map import StructureMapGenerator

    captured = {}

    class _Factory:
        collection_rules: dict = {}

        def __setattr__(self, key, value):
            captured[key] = value
            object.__setattr__(self, key, value)

        def record_unindexed_snapshot_slices(self, *args, **kwargs):
            return None

        def create_field_rules(self, *args, **kwargs):
            return []

    generator = object.__new__(StructureMapGenerator)
    generator.factory = _Factory()
    generator.custom_mapping_table = {}
    generator.mapping_diagnostics = []

    try:
        generator.create_res_group(
            "CatalogEntry",
            SimpleNamespace(data=_IdlessProfile(), mappable_fields=[]),
            create_references=False,
        )
    except Exception:  # noqa: BLE001 - later stages are deliberately unstubbed
        pass

    assert captured.get("_current_profile_id") == "HealthAppCatalogEntry", (
        "group creation must key the mapping table on the profile's identity; "
        f"got {captured.get('_current_profile_id')!r}"
    )
