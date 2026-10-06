# Automapping (experimental)

The current automapping option derives the mapping table (source field -> target element) instead of authoring it by hand. A user-authored mapping table is never overwritten + is skipped if a mapping table exists already.

## Usage

- the deterministic matcher (TF-IDF) is the default and makes no LLM call
- `llm` mode only reranks the candidates
- for `llm` mode you have to configure the provider in `.env` (see `.env.example`) and install the optional `llm` packages from the `pyproject.toml`
- `--auto-mapping-mode llm` without `--auto-mapping` is a CLI error

```
python src/main.py -c conf/project.json pipeline static-gen-sm --auto-mapping
python src/main.py -c conf/project.json pipeline static-gen-sm --auto-mapping --auto-mapping-mode llm
```

Optional `.env` settings for the llm mode:

```
LLM_AUTOMAPPING_TOP_K=5              # candidates offered per source field
LLM_AUTOMAPPING_SECOND_PASS_TOP_K=15 # wider retry, or leave unset to disable
```

## Workflow

Both modes share candidate discovery, scoring and pruning; only the selection step differs. In `llm` mode, per source field:

- collect candidates that resolve uniquely to one element of the target profile tree
- pool them across all profiles of the project, dedupe by target, truncate to `top_k`
- each profile's deterministic rank-1 candidate is offered first, the rest fill up by score
- enumerate them as `c1..cN` and ask the provider once per source field
- the answer must be one of the offered ids or `unmapped`
- a target is allocated once: after a field selects it, it is removed from every later field's offers
- a declined field gets exactly one retry against `second_pass_top_k` (skipped if the wider set holds nothing new)
- the resulting table is compiled and validated by the ordinary custom-table path

Notes on behaviour:

- a source field with no offerable candidate costs no provider call
- nothing falls back to the deterministic matcher — a missing credential, transport error or provider rejection aborts the run
- bad model output only costs that one field, since model output is treated as an untrusted proposal
- the reported `confidence` and `reason` are provenance for a human reader, never an acceptance criterion
- the model never sees or edits a StructureMap — that is the [feedback-guided repair workflow](feedback_guided_repair.md)

## Output

Both modes write the compiled table to the same file, so downstream tooling does not need to know which strategy ran:

```
source_data/
  <map_name>_automapping.json               the compiled mapping table
  <map_name>_llm_automapping_report.json    llm mode only
```

The report is written only after every generated map has been compiled and structurally validated. Per source field and pass it records every offered candidate (element id, emitted target, types, cardinality, deterministic score, whether it was the deterministic rank-1 pick), the model's selection with confidence and reason, allocation and rejection reasons, and whether the answer came from cache. `mapped` in its summary means compiler-applied, not merely selected.
