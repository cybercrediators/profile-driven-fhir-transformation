def _to_plain_dict(v):
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump(exclude_none=True)
        except Exception:
            return v
    return v


def set_value_in_temp_dict(temp_dict, path, value, structure_def, _consumed=None):
    """set a pre-defined value in a template dictionary"""
    if not path:
        return
    if _consumed is None:
        _consumed = [structure_def.data.type]
    key = path[0]
    remaining_keys = path[1:]
    cur_path = ".".join(_consumed + [key])
    elem_def = next(
        (e for e in structure_def.data.snapshot.element if e.path == cur_path), None
    )
    is_list = elem_def.max == "*" if elem_def else False
    next_consumed = _consumed + [key]

    if not remaining_keys:
        if is_list:
            existing = temp_dict.get(key) if isinstance(temp_dict, dict) else None
            if isinstance(existing, list):
                existing.append(_to_plain_dict(value))
            elif existing is None:
                temp_dict[key] = [_to_plain_dict(value)]
            else:
                # element resolves as a list but a scalar/dict was
                # already written here (or vice versa)
                temp_dict[key] = _to_plain_dict(value)
        else:
            temp_dict[key] = _to_plain_dict(value)
    else:
        # slices sharing the path merge into the same first entry
        if is_list:
            existing = temp_dict.get(key) if isinstance(temp_dict, dict) else None
            if not existing or not isinstance(existing, list):
                temp_dict[key] = [{}]
            elif not isinstance(existing[0], dict):
                coerced = _to_plain_dict(existing[0])
                temp_dict[key][0] = coerced if isinstance(coerced, dict) else {}
            set_value_in_temp_dict(
                temp_dict[key][0], remaining_keys, value, structure_def, next_consumed
            )
        else:
            existing = temp_dict.get(key) if isinstance(temp_dict, dict) else None
            if not isinstance(existing, dict):
                coerced = _to_plain_dict(existing) if existing is not None else {}
                temp_dict[key] = coerced if isinstance(coerced, dict) else {}
            set_value_in_temp_dict(
                temp_dict[key], remaining_keys, value, structure_def, next_consumed
            )
    return temp_dict
