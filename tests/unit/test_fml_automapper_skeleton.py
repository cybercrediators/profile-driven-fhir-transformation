"""Placeholder retired — the automapper now has real coverage (WP3).

The three skipped skeletons that used to live here are implemented:

* exact name/path matching and its scoring
  -> ``test_fml_automapper_legacy.py::test_exact_name_match_is_selected``
* the weighted name/path/description strategy
  -> ``test_fml_automapper_candidates.py``
     ``::test_weighted_components_reconstruct_the_legacy_weighted_score``
* virtual-path expansion of composite types
  -> ``test_fml_automapper_legacy.py``
     ``::test_nested_virtual_path_is_synthesised_from_the_parent_chain``

``test_fml_automapper_legacy.py`` characterizes the frozen deterministic path;
``test_fml_automapper_candidates.py`` covers the candidate API layered over it.
This module is kept as a signpost so the old filename still leads somewhere.
"""
