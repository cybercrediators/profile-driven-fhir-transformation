"""Unit tests for the E15 comparison's list indexing (eval/e15_kfdm_compare.py).

The comparison decides what counts as a difference between the two pipelines, so a
flaw in it shows up as a fabricated content gap. These cover the identity-vs-position
choice specifically: keying `item` arrays positionally made one pipeline omitting an
item report every later item as mismatched.
"""

import pytest

from eval.e15_kfdm_compare import _list_keys, flatten

pytestmark = pytest.mark.unit


def test_items_with_link_ids_are_keyed_by_identity():
    assert _list_keys([{"linkId": "2.1"}, {"linkId": "2.3"}]) == [
        "[linkId=2.1]",
        "[linkId=2.3]",
    ]


def test_lists_without_link_ids_stay_positional():
    assert _list_keys([{"valueString": "a"}, {"valueString": "b"}]) == ["[0]", "[1]"]


def test_a_partially_identified_list_stays_positional():
    # Mixing the two would pair some entries by identity and some by offset, which is
    # worse than either; fall back to the predictable one.
    assert _list_keys([{"linkId": "1"}, {"text": "no id"}]) == ["[0]", "[1]"]


def test_scalars_and_empty_lists_are_positional():
    assert _list_keys([]) == []
    assert _list_keys(["a", "b"]) == ["[0]", "[1]"]


def test_duplicate_link_ids_are_disambiguated_not_merged():
    assert _list_keys([{"linkId": "2.8"}, {"linkId": "2.8"}]) == [
        "[linkId=2.8]",
        "[linkId=2.8#1]",
    ]


def test_an_omitted_item_no_longer_shifts_the_ones_after_it():
    # The real case: the reference pipeline drops group 2.2, so positional keying
    # reported 2.3/2.4 as mismatched linkIds and their answers as only-on-one-side.
    original = {"item": [{"linkId": "2.1", "answer": [{"valueString": "a"}]},
                         {"linkId": "2.3", "answer": [{"valueString": "c"}]}]}
    ours = {"item": [{"linkId": "2.1", "answer": [{"valueString": "a"}]},
                     {"linkId": "2.2", "answer": [{"valueString": "b"}]},
                     {"linkId": "2.3", "answer": [{"valueString": "c"}]}]}
    fo, fm = flatten(original), flatten(ours)
    # 2.1 and 2.3 line up; only the genuinely extra 2.2 is reported.
    assert not [k for k in set(fo) & set(fm) if fo[k] != fm[k]]
    assert set(fo) - set(fm) == set()
    assert all("2.2" in k for k in set(fm) - set(fo))
