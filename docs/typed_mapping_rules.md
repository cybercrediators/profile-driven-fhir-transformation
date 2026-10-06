# Typed StructureMap rules and reference policies

- The original mapping-table contract still holds: ordinary entries map one
  source path to one target path, and typed collection values configure repeated
  backbones.
- For mappings needing more of the R4/R4B Mapping Language, the same JSON
  document may carry these reserved top-level sections:
  - `$imports` — canonical StructureMap URLs;
  - `$structures` — additional `source`, `queried`, `target`, or `produced`
    structure declarations;
  - `$rules` — rules appended to the generated profile's primary transform group;
  - `$groups` — additional named groups;
  - `$references` — explicit direct, bundled, conditional, contained, or
    canonical reference policies.
- Every declaration can be restricted with `profiles` and/or `resourceTypes`.
- Invalid declarations are omitted individually and reported as
  `invalid-typed-rule` diagnostics; they never fall through to the legacy
  mapping parser.

## Rules and groups

- The typed form mirrors the R4B `StructureMap` resource.
- Supported: multiple sources and targets, source `condition`, `check`,
  `logMessage`, default values and list modes, target list modes, nested rules,
  and dependent group calls.
```json
{
  "$imports": ["http://example.org/StructureMap/common"],
  "$structures": [
    {
      "url": "http://example.org/StructureDefinition/Lookup",
      "mode": "queried",
      "alias": "Lookup"
    }
  ],
  "$groups": [
    {
      "name": "Correlate",
      "typeMode": "types",
      "inputs": [
        {"name": "left", "mode": "source", "type": "Observation"},
        {"name": "right", "mode": "source", "type": "Patient"},
        {"name": "out", "mode": "target", "type": "Observation"}
      ],
      "rules": [
        {
          "name": "copy-correlated-id",
          "sources": [
            {
              "context": "left",
              "element": "identifier",
              "variable": "left-id",
              "condition": "left-id.exists()",
              "listMode": "first"
            },
            {
              "context": "right",
              "element": "identifier",
              "variable": "right-id",
              "listMode": "only_one"
            }
          ],
          "targets": [
            {
              "context": "out",
              "element": "identifier",
              "transform": "copy",
              "parameters": [{"valueId": "left-id"}],
              "targetListMode": "share",
              "listRuleId": "correlated-id"
            }
          ],
          "dependent": [
            {
              "name": "Finalize",
              "variables": ["left", "right", "out"]
            }
          ]
        }
      ]
    }
  ]
}
```

- Target parameters use exactly one of `valueId`, `valueString`, `valueBoolean`,
  `valueInteger`, or `valueDecimal`.
- Accepted transforms are the R4/R4B codes `create`, `copy`, `truncate`,
  `escape`, `cast`, `append`, `translate`, `reference`, `dateOp`, `uuid`,
  `pointer`, `evaluate`, `cc`, `c`, `qty`, `id`, `cp`.
- The generator preserves declared parameter order; the compiler validates
  transform-specific arity, literal parameter roles, variable references, and
  local group calls.
- The author remains responsible for the meaning of FHIRPath expressions and for
  runtime-specific behavior not defined by the StructureMap capability
  CodeSystems.

## References

- `$references` prevents an inferred TODO reference rule from competing with an
  explicit policy for the same path.
```json
{
  "$references": [
    {
      "name": "subject-by-identifier",
      "path": "Observation.subject",
      "strategy": "conditional",
      "source": "patientId",
      "targetType": "Patient",
      "identifierSystem": "http://hospital.example/mrn"
    },
    {
      "name": "all-performers",
      "path": "Observation.performer",
      "strategy": "bundled",
      "targetTypes": ["Practitioner", "Organization"],
      "match": "all",
      "referenceMode": "urn"
    }
  ]
}
```

### Strategies

- `direct` — create `Reference` and copy an authored source or literal into
  `Reference.reference`; `displaySource` may also populate `display`;
- `conditional` — build `Type?identifier=system|value`;
- `contained` — create the contained resource and its local `#id` reference;
  `dependentGroup` can populate it using the source and contained target
  variables;
- `canonical` — copy a canonical value directly to the target element;
- `bundled` — emit a machine-readable contract consumed by `BundleService`.

### Bundled policies

- Require `targetType` or `targetTypes`. `match` is one of:
  - `byOrder` — deterministic modulo pairing, preserving legacy behavior;
  - `singleton` — wire only when exactly one eligible target exists;
  - `identifier` — require exactly one equality match between `sourceKey` and
    `targetKey`;
  - `all` — wire every eligible target, de-duplicated, to a repeating reference.
- `referenceMode` is `urn` (the bundle entry's `fullUrl`) or `relative`
  (`ResourceType/id`, falling back to the URN when the target has no id).
- Candidate types are tried in declared order, except `all`, which gathers all
  declared types.
- Optional `targetProfile` further restricts same-base-type candidates via
  `meta.profile`.
- Profile scoping, prohibited (`max=0`) paths, absent optional backbones, and
  self-reference exclusion retain the existing BundleService safeguards.
- Bundled references use `match` instead of StructureMap source/target list
  modes.

## Base and special elements

- `id`, `meta`, `implicitRules`, `language`, and narrative `text` stay absent
  from automatically inferred skeletons, but are emitted when a legacy table
  explicitly targets them.
- XHTML is copied from an authored source value; it is neither synthesized nor
  sanitized.
- Rejected with actionable diagnostics: unsliced `modifierExtension`, and a
  legacy mapping directly to `contained`.
- Use instead: a profile-defined modifier-extension slice, a contained
  `$references` policy, or an explicit `$rules` declaration.
