"""handle options and operations for patching documents based on model findings"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

PATCH_SCHEMA_VERSION = 1


class PatchOp(str, Enum):
    """define operations agent mode may propose"""

    TEST = "test"
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


# define operations on how they'd change the maps/docs
MUTATING_OPS = frozenset({PatchOp.ADD, PatchOp.REMOVE, PatchOp.REPLACE})
GUARDED_OPS = frozenset({PatchOp.REMOVE, PatchOp.REPLACE})


# define element types directly
_JsonLeaf = Union[str, bool, int, float, None]
_JsonElement = Union[_JsonLeaf, Dict[str, Any], List[Any]]
JsonPatchValue = Union[_JsonLeaf, Dict[str, Any], List[_JsonElement]]


class _Operation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str

    def as_rfc6902(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class TestOperation(_Operation):
    """Assert a value. The guard that makes positional addressing safe."""

    op: Literal["test"]
    value: JsonPatchValue


class AddOperation(_Operation):
    """Insert into an array, or set an object member."""

    op: Literal["add"]
    value: JsonPatchValue


class ReplaceOperation(_Operation):
    op: Literal["replace"]
    value: JsonPatchValue


class RemoveOperation(_Operation):
    """no value field anywhere"""

    op: Literal["remove"]


PatchOperation = Annotated[
    Union[TestOperation, AddOperation, ReplaceOperation, RemoveOperation],
    Field(discriminator="op"),
]

ProposedOperation = Annotated[
    Union[AddOperation, ReplaceOperation, RemoveOperation],
    Field(discriminator="op"),
]

_OPERATION_ADAPTER: TypeAdapter = TypeAdapter(PatchOperation)

_MISSING = object()


def operation(op: Union[PatchOp, str], path: str, value: Any = _MISSING):
    """Build the operation model matching op(eration)

    :raises pydantic.ValidationError: for an unsupported op
    """

    payload: Dict[str, Any] = {
        "op": op.value if isinstance(op, PatchOp) else op,
        "path": path,
    }
    if value is not _MISSING:
        payload["value"] = value
    return _OPERATION_ADAPTER.validate_python(payload)


class PatchProvenance(BaseModel):
    """Where a proposal came from"""

    model_config = ConfigDict(extra="allow")

    provider: Optional[str] = None
    model: Optional[str] = None
    cache_hit: Optional[bool] = None
    attempt: Optional[int] = None


class AgentPatch(BaseModel):
    """contains one proposal bound to the exact document"""

    model_config = ConfigDict(extra="forbid")

    # internal schema version
    schema_version: Literal[1]

    map_url: str
    map_id: str
    
    base_sha256: str
    # internal diagnostic IDs
    diagnostic_ids: List[str]
    # list of found patch proposals
    patch: List[PatchOperation]

    rationale: Optional[str] = None
    provenance: Optional[PatchProvenance] = None

    def as_rfc6902(self) -> List[Dict[str, Any]]:
        """The payload as a plain RFC 6902 document."""

        return [op.as_rfc6902() for op in self.patch]


class ProposedPatch(BaseModel):
    """The format a provider is asked for"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    map_url: str
    map_id: str
    base_sha256: str
    diagnostic_ids: List[str]
    patch: List[ProposedOperation]
    rationale: Optional[str] = None

    def as_agent_patch(self) -> "AgentPatch":
        """widen into the internal model"""

        return AgentPatch.model_validate(self.model_dump(mode="python"))


class PatchRejection(BaseModel):
    """Why a proposal was refused, in a form a retry prompt can consume."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    operation_index: Optional[int] = None
    path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class PatchApplication(BaseModel):
    """The outcome of evaluating one proposal against one base document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    applied: bool
    candidate: Optional[Dict[str, Any]] = None
    candidate_sha256: Optional[str] = None

    normalized_patch: List[Dict[str, Any]] = Field(default_factory=list)
    rejections: List[PatchRejection] = Field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.applied

    def as_report_dict(self) -> Dict[str, Any]:
        return {
            "applied": self.applied,
            "candidate_sha256": self.candidate_sha256,
            "normalized_patch": list(self.normalized_patch),
            "rejections": [rejection.as_dict() for rejection in self.rejections],
        }


class RejectionCode(str, Enum):
    """define identifiers for every refusal reason"""

    STALE_BASE = "stale-base"
    WRONG_MAP = "wrong-map"
    NOT_A_STRUCTURE_MAP = "not-a-structure-map"
    EMPTY_PATCH = "empty-patch"
    UNSUPPORTED_OP = "unsupported-op"
    MALFORMED_OP = "malformed-operation"
    UNGUARDED_MUTATION = "unguarded-mutation"
    ROOT_REPLACEMENT = "root-replacement"
    PROTECTED_FIELD = "protected-field"
    OUT_OF_SCOPE_POINTER = "out-of-scope-pointer"
    SHIFTED_POINTER = "shifted-pointer"
    TOO_MANY_OPERATIONS = "too-many-operations"
    PAYLOAD_TOO_LARGE = "payload-too-large"
    TEST_FAILED = "test-failed"
    INVALID_POINTER = "invalid-pointer"
    APPLICATION_FAILED = "application-failed"
    INVALID_CANDIDATE = "invalid-candidate"
    NO_EFFECT = "no-effect"
