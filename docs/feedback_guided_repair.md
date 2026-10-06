# Feedback-guided repair workflow

You can use an optional pipeline for repairing generated FHIR StructureMaps by passing them through an error classification and LLM-assisted patching workflow editing the resulting JSON.

## Usage

- first you have to configure the LLM provider you want to use in `.env` (you can use the `.env.example` as a template)
- make sure to have the optional agent packages installed as defined in the `pyproject.toml`

```
python -m src.main agent fix <map>                    # diagnose (read-only!)
python -m src.main agent fix <map> --apply            # -> and replace the map on disk
python -m src.main agent fix <map> --offline          # only run classification without LLM calls
python -m src.main agent fix --all                    # fix every map from the configured project
python -m src.main agent fix --all --resume <run-id>  # continue an interrupted run
```

## Workflow
The loop is implemented using `langgraph`. After the baseline is validated, the following langgraph nodes get passed (also, each node are identified as a checkpoint):

- worklist = map-fixable findings only
- build prompt context (pointers pre-resolved)
- ask provider for a JSON patch
- apply in memory, guards injected by the tool
- validate baseline + candidate on identical fixtures
- accept | retry with feedback | stop

There are multiple possible outcomes: `clean`, `blocked`, `accepted`, `exhausted`, `no-progress`, `limit-reached`, `provider-error` (`clean` and `blocked` are not passed to the llm provider, since `blocked` means a finding remains but is not map-fixable).
Currently there are a max attempts of `4` configured, with a maximum findings per attempt of `6`. There are further configuration options available to limit the attempts (`max_provider_calls`, `max_seconds`, `max_prompt_chars`, `max_operations`, `max_project_rounds`).

## Output

Each run creates a new directory in the corresponding projects `agent_output` folder which includes :

```
agent_output/<run-id>/
  run_manifest.json     resume identity: versions, provider, map digests
  checkpoints.sqlite    graph state
  project_report.json   rounds, routing, global findings, apply transaction
  agent_report.json     single-map runs (--all puts per-map artifacts under maps/<key>/)
```