# Collection mappings

Either use plain `"source.path": "Target.path"` mapping to map a field from input file to output resource or declare collection rules if a source backbone is repeating and several children must populate the same repeatedly, e.g.:

```json
{
  "Source.identifiers": {
    "target": "Patient.identifier",
    "sourceListMode": "not_first",
    "targetListMode": ["share", "collate"],
    "listRuleId": "patient-identifiers",
    "correlation": {
      "sourceKey": "system",
      "targetKey": "system"
    }
  },
  "Source.identifiers.system": "Patient.identifier.system",
  "Source.identifiers.value": "Patient.identifier.value"
}
```
- Binds `Source.identifiers` and runs once per selected source repetition.
- Each execution creates or selects one `Patient.identifier`, then runs all descendant mappings with that source item and target item as their contexts.
- Zero selected source items produce no target repetition; one produces one correlated repetition; multiple are processed independently.
- Optional, heterogeneous children populate only the target repetition belonging to their containing source item.
- Nestable: declare another typed parent rule below the first source and target parents.
- `sourceListMode` accepts the R4/R4B codes `first`, `not_first`, `last`,
  `not_last`, `only_one`.
- `targetListMode` accepts `first`, `share`, `last`, `collate`, either as one string or an ordered list.
- `share` receives a deterministic `listRuleId` when none is supplied; a
  supplied `listRuleId` is valid only with `share`.
- Optional: `sourceKey` must identify one child value on every selected source repetition.
- If `targetKey` is also supplied, that exact source key must have a normal mapping to the target key below the declared collection.
- String shorthand such as `"correlation": "system"` declares the same relative key on both sides.
- Invalid modes, non-repeating parents, missing keys, or mismatched key mappings produce structured diagnostics (invalid collections are omitted)