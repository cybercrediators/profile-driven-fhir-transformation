"""Handle field automapping using word embeddings (e.g., GloVe via torchtext)

- find_mapping: simple automapping (TF-IDF ad GloVe)
- find_candidates: re-rank and LLM automapping targets
"""

from dataclasses import dataclass, replace
import hashlib
import logging
import math
import re
import threading
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from mapping.target_tree import (
    RESOLUTION_NOT_FOUND,
    RESOLVED_STATUSES,
    TargetTree,
)

logger = logging.getLogger(__name__)

try:
    from torchtext.vocab import GloVe

    TORCHTEXT_AVAILABLE = True
except ImportError:
    TORCHTEXT_AVAILABLE = False


RESOLUTION_NOT_ATTEMPTED = "not-attempted"

EXCLUDED_UNRESOLVED = "unresolved-target"
EXCLUDED_PROHIBITED = "prohibited-target"
EXCLUDED_CARDINALITY = "incompatible-cardinality"
EXCLUDED_DATATYPE = "incompatible-datatype"

_PRIMITIVE_FAMILY = {
    "boolean": "boolean",
    "decimal": "number",
    "integer": "number",
    "integer64": "number",
    "positiveInt": "number",
    "unsignedInt": "number",
    "date": "temporal",
    "dateTime": "temporal",
    "instant": "temporal",
    "time": "temporal",
    "base64Binary": "text",
    "canonical": "text",
    "code": "text",
    "id": "text",
    "markdown": "text",
    "oid": "text",
    "string": "text",
    "uri": "text",
    "url": "text",
    "uuid": "text",
}
_CONVERTIBLE_FAMILY = "text"


@dataclass(frozen=True)
class CandidateScores:
    """Component scores behind one candidate, as the legacy matcher computes them."""

    tfidf: float = 0.0
    name_sim: float = 0.0
    path_sim: float = 0.0
    desc_sim: float = 0.0
    weighted: float = 0.0

    @property
    def legacy(self) -> float:
        """The score this candidate contributes to the legacy comparison"""
        return max(self.tfidf, self.weighted)


@dataclass(frozen=True)
class AutomapCandidate:
    """One compiler-known target the deterministic matcher considered"""

    candidate_id: str
    scoring_path: str
    legacy_emitted_path: str
    legacy_virtual_path: str
    scores: CandidateScores
    flat_index: int
    is_legacy_selection: bool = False
    meets_legacy_threshold: bool = False

    element_id: Optional[str] = None
    element_path: Optional[str] = None
    types: Tuple[str, ...] = ()
    min: Optional[int] = None
    max: Optional[str] = None
    slice_name: Optional[str] = None
    profile_url: Optional[str] = None

    canonical_target_id: Optional[str] = None
    canonical_target_path: Optional[str] = None
    mapping_target_path: Optional[str] = None
    resolution_status: str = RESOLUTION_NOT_ATTEMPTED
    resolution_key: Optional[str] = None
    llm_visible: bool = False
    exclusion_reason: Optional[str] = None


class _TorchtextVectors:
    """Wrapper for torchtext vectors replacing gensim KeyedVectors interface."""

    def __init__(self, glove: GloVe):
        self._glove = glove

    def __contains__(self, word: str) -> bool:
        return word in self._glove.stoi

    def __getitem__(self, word: str) -> np.ndarray:
        vec = self._glove.vectors[self._glove.stoi[word]]
        return vec.detach().cpu().numpy()


class FMLAutomapper:
    """
    Simple automapper using a weighted multi-component strategy (TF/IDF) with optional Word2Vec embeddings
    """

    def __init__(self, threshold: float = 0.69, use_word2vec: bool = False) -> None:
        self.threshold = threshold
        self.use_word2vec = use_word2vec
        self.model = None
        self.model_ready = False

        if self.use_word2vec:
            if TORCHTEXT_AVAILABLE:
                logger.info(
                    "Starting background loading of GloVe vectors (torchtext)..."
                )
                self.load_thread = threading.Thread(target=self._load_model)
                self.load_thread.daemon = True
                self.load_thread.start()
            else:
                logger.warning(
                    "torchtext not installed. Falling back to simple vectorized approach."
                )
                self.use_word2vec = False

    def _load_model(self) -> None:
        try:
            logger.info("Loading GloVe vectors (glove.6B.50d via torchtext)...")
            glove = GloVe(name="6B", dim=50)
            self.model = _TorchtextVectors(glove)
            self.model_ready = True
            logger.info("GloVe vectors loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load GloVe vectors: {e}")
            self.use_word2vec = False

    def _await_embedding_model(self) -> None:
        """Block until the background GloVe load has settled"""

        if (
            self.use_word2vec
            and not self.model_ready
            and hasattr(self, "load_thread")
            and self.load_thread.is_alive()
        ):
            logger.info("Waiting for Word2Vec model to finish loading...")
            self.load_thread.join()

    def find_mapping(
        self, source_field: Dict[str, Any], target_fields: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        (try to) find the best matching target field for a given source field, based on:
        1. Vectorized Approach (TF-IDF + Cosine Similarity)
        2. Weighted Strategy (Name/Path/Desc) (weighted score = 0.5 * name_sim + 0.3 * path_sim + 0.2 * desc_sim)
        (Optional): Word2Vec embeddings for semantic similarity.

        :param self: Description
        :param source_field: Description
        :type source_field: Dict[str, Any]
        :param target_fields: Description
        :type target_fields: List[Dict[str, Any]]
        :return: Description
        :rtype: Tuple[Dict[str, Any] | None, float]
        """
        # Flatten target fields to include complex types
        flat_targets = self._flatten_target_fields(target_fields)

        # check if word2vec model is ready
        self._await_embedding_model()

        best_match = None
        best_score = 0.0

        # TF-IDF
        vec_match, vec_score = self._find_match_vectorized(source_field, flat_targets)
        if vec_score > best_score:
            best_score = vec_score
            best_match = vec_match

        # tokenize source field features
        source_feats = self._extract_features(source_field)

        # weighted
        for target in flat_targets:
            # tokenize target field features
            target_feats = self._extract_features(target)
            score = self._calculate_weighted_score(source_feats, target_feats)

            if score > best_score:
                best_score = score
                best_match = target

        print(
            "Field {} best match score: {:.4f} with {}".format(
                source_field.get("id", "unknown"), best_score, best_match
            )
        )
        if best_score >= self.threshold:
            return best_match, best_score
        return None, 0.0

    def find_candidates(
        self,
        source_field: Dict[str, Any],
        target_fields: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        target_tree: Optional[TargetTree] = None,
    ) -> List["AutomapCandidate"]:
        """Rank the targets ``find_mapping`` considered, as typed candidates.

        :param source_field: the source field being mapped.
        :param target_fields: targets exactly as ``find_mapping`` receives them,
            i.e. already through ``flatten_profile_fields`` and
            ``_flatten_fields_recursively``.
        :param top_k: how many candidates to return; ``None`` returns all.
        :param target_tree: profile target tree used to annotate and prune. When
            omitted, every candidate stays ``llm_visible=False`` — an
            unannotated candidate is never offered to a model.
        :return: candidates ordered with the legacy selection first.
        """

        pairs = self._flatten_target_candidates(target_fields)
        if not pairs:
            return []

        self._await_embedding_model()

        flat_targets = [flat for flat, _ in pairs]
        scores = self._score_table(source_field, flat_targets)
        winner_index = self._replay_legacy_selection(scores)

        candidates = []
        seen_ids: Dict[str, int] = {}
        for index, ((flat, origin), row) in enumerate(zip(pairs, scores)):
            virtual_path = flat.get("_virtual_path") or ""
            real_path = flat.get("path") or ""
            candidates.append(
                AutomapCandidate(
                    candidate_id=self._candidate_id(flat, seen_ids),
                    scoring_path=virtual_path or real_path,
                    legacy_emitted_path=real_path or virtual_path,
                    legacy_virtual_path=virtual_path,
                    scores=row,
                    flat_index=index,
                    is_legacy_selection=index == winner_index,
                    meets_legacy_threshold=row.legacy >= self.threshold,
                    **_validation_context(origin),
                )
            )

        if target_tree is not None:
            candidates = [
                self._annotate_candidate(candidate, source_field, target_tree)
                for candidate in candidates
            ]

        ordered = self._order_candidates(candidates, winner_index)
        return ordered if top_k is None else ordered[:top_k]

    def find_llm_candidates(
        self,
        source_field: Dict[str, Any],
        target_fields: List[Dict[str, Any]],
        target_tree: TargetTree,
        top_k: Optional[int] = 5,
    ) -> List["AutomapCandidate"]:
        """The candidates a model may actually be offered, at most *top_k* of them.

        :param target_tree: required — resolution is what makes a candidate
            offerable, so there is no unannotated form of this call.
        """

        candidates = self.find_candidates(
            source_field, target_fields, top_k=None, target_tree=target_tree
        )

        selected: List["AutomapCandidate"] = []
        seen_targets = set()
        for candidate in candidates:
            if not candidate.llm_visible:
                continue
            identity = candidate.mapping_target_path or candidate.canonical_target_id
            if identity in seen_targets:
                continue
            seen_targets.add(identity)
            selected.append(candidate)
            if top_k is not None and len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _order_candidates(
        candidates: List["AutomapCandidate"], winner_index: int
    ) -> List["AutomapCandidate"]:
        """Legacy selection first, then descending score with index as tie-break"""

        tail = sorted(
            (c for c in candidates if c.flat_index != winner_index),
            key=lambda c: (-c.scores.legacy, c.flat_index),
        )
        head = [c for c in candidates if c.flat_index == winner_index]
        return head + tail

    def _replay_legacy_selection(self, scores: List["CandidateScores"]) -> int:
        """Return the index ``find_mapping`` would select, or -1 for no match"""

        best_index, best_score = -1, 0.0

        tfidf_index, tfidf_best = -1, 0.0
        for index, row in enumerate(scores):
            if row.tfidf > tfidf_best:
                tfidf_best, tfidf_index = row.tfidf, index
        if tfidf_index != -1 and tfidf_best > best_score:
            best_index, best_score = tfidf_index, tfidf_best

        for index, row in enumerate(scores):
            if row.weighted > best_score:
                best_index, best_score = index, row.weighted

        return best_index

    def _score_table(
        self, source_field: Dict[str, Any], flat_targets: List[Dict[str, Any]]
    ) -> List["CandidateScores"]:
        """Per-target component scores, using the same helpers as find_mapping"""

        tfidf = self._tfidf_scores(source_field, flat_targets)
        source_feats = self._extract_features(source_field)

        rows = []
        for index, target in enumerate(flat_targets):
            target_feats = self._extract_features(target)
            name_sim = self._compute_similarity(source_feats["name"], target_feats["name"])
            path_sim = self._compute_similarity(source_feats["path"], target_feats["path"])
            desc_sim = self._compute_similarity(source_feats["name"], target_feats["desc"])
            rows.append(
                CandidateScores(
                    tfidf=tfidf[index],
                    name_sim=name_sim,
                    path_sim=path_sim,
                    desc_sim=desc_sim,
                    weighted=(0.5 * name_sim) + (0.3 * path_sim) + (0.2 * desc_sim),
                )
            )
        return rows

    def _tfidf_scores(
        self, source_field: Dict[str, Any], target_fields: List[Dict[str, Any]]
    ) -> List[float]:
        """Per-target TF-IDF cosine scores"""

        source_id = source_field.get("id", "")
        source_path = source_field.get("path", "")
        source_name = source_path.split(".")[-1] if source_path else source_id

        target_docs = []
        for target in target_fields:
            t_id = target.get("id", "")
            t_path = target.get("_virtual_path", target.get("path", ""))
            t_desc = (
                target.get("description", "")
                or target.get("short", "")
                or target.get("definition", "")
            )
            t_name = t_path.split(".")[-1]
            target_docs.append(self._tokenize(f"{t_name} {t_id} {t_path} {t_desc}"))

        if not target_docs:
            return []

        query_tokens = self._tokenize(f"{source_name} {source_id}")

        vocab = set(query_tokens)
        for doc in target_docs:
            vocab.update(doc)
        vocab_list = sorted(list(vocab))
        vocab_index = {word: i for i, word in enumerate(vocab_list)}

        if not vocab_list:
            return [0.0] * len(target_docs)

        n_docs = len(target_docs)
        idf = {}
        for word in vocab_list:
            df = sum(1 for doc in target_docs if word in doc)
            idf[word] = math.log((n_docs + 1) / (df + 1)) + 1

        query_vec = np.zeros(len(vocab_list))
        for word in query_tokens:
            if word in vocab_index:
                tf = query_tokens.count(word) / len(query_tokens)
                query_vec[vocab_index[word]] = tf * idf[word]

        scores = []
        for doc in target_docs:
            if not doc:
                scores.append(0.0)
                continue
            doc_vec = np.zeros(len(vocab_list))
            for word in doc:
                if word in vocab_index:
                    tf = doc.count(word) / len(doc)
                    doc_vec[vocab_index[word]] = tf * idf[word]
            scores.append(float(self._cosine_similarity(query_vec, doc_vec)))
        return scores

    def _flatten_target_candidates(
        self,
        fields: List[Dict[str, Any]],
        parent_path: str = "",
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """``_flatten_target_fields`` with provenance retained"""

        flat: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for field in fields:
            copy = {
                "id": field.get("id"),
                "path": field.get("path"),
                "description": field.get("description"),
                "short": field.get("short"),
                "definition": field.get("definition"),
            }

            current_name = field.get("path", "").split(".")[-1]
            if parent_path:
                virtual_path = f"{parent_path}.{current_name}"
            else:
                virtual_path = field.get("path", "")

            copy["_virtual_path"] = virtual_path
            flat.append((copy, field))

            if "type_structure" in field:
                flat.extend(
                    self._flatten_target_candidates(
                        field["type_structure"], parent_path=virtual_path
                    )
                )
        return flat

    @staticmethod
    def _candidate_id(flat: Dict[str, Any], seen: Dict[str, int]) -> str:
        """Content-derived, stable, and unique within one call"""

        material = "|".join(
            str(flat.get(key) or "") for key in ("id", "path", "_virtual_path")
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
        occurrence = seen.get(digest, 0)
        seen[digest] = occurrence + 1
        return f"cand-{digest}" if occurrence == 0 else f"cand-{digest}-{occurrence + 1}"

    def _annotate_candidate(
        self,
        candidate: "AutomapCandidate",
        source_field: Dict[str, Any],
        tree: TargetTree,
    ) -> "AutomapCandidate":
        """Resolve a candidate against the main target tree and decide visibility."""

        node, status, key_kind = _resolve_against_tree(candidate, tree)
        if node is None or status not in RESOLVED_STATUSES:
            return replace(
                candidate,
                resolution_status=status,
                llm_visible=False,
                exclusion_reason=_unresolved_reason(candidate, tree),
            )

        annotated = replace(
            candidate,
            canonical_target_id=node.eid,
            canonical_target_path=node.path,
            mapping_target_path=node.eid,
            resolution_status=status,
            resolution_key=key_kind,
            types=tuple(node.types) or candidate.types,
            min=node.min if node.min is not None else candidate.min,
            max=node.max if node.max is not None else candidate.max,
            slice_name=node.slice_name or candidate.slice_name,
        )

        reason = _prune_reason(annotated, node, tree, source_field)
        return replace(
            annotated, llm_visible=reason is None, exclusion_reason=reason
        )

    def _extract_features(self, field: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract tokenized features from a field"""
        path = field.get("_virtual_path", field.get("path", ""))
        name = path.split(".")[-1] if path else field.get("id", "")

        desc = f"{field.get('description', '')} {field.get('short', '')} {field.get('definition', '')}"

        return {
            "name": self._tokenize(name),
            "path": self._tokenize(path),
            "desc": self._tokenize(desc),
        }

    def _calculate_weighted_score(
        self, src: Dict[str, List[str]], tgt: Dict[str, List[str]]
    ) -> float:
        """
        :param self: Description
        :param src: Description
        :type src: Dict[str, List[str]]
        :param tgt: Description
        :type tgt: Dict[str, List[str]]
        :return: Description
        :rtype: float
        """

        name_sim = self._compute_similarity(src["name"], tgt["name"])
        path_sim = self._compute_similarity(src["path"], tgt["path"])
        desc_sim = self._compute_similarity(src["name"], tgt["desc"])

        return (0.5 * name_sim) + (0.3 * path_sim) + (0.2 * desc_sim)

    def _compute_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        :param self: Description
        :param tokens1: Description
        :type tokens1: List[str]
        :param tokens2: Description
        :type tokens2: List[str]
        :return: Description
        :rtype: float
        """

        if not tokens1 or not tokens2:
            return 0.0

        if self.use_word2vec and self.model_ready:
            return self._calculate_w2v_similarity(tokens1, tokens2)

        # jaccard similarity as fallback
        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _flatten_target_fields(
        self, fields: List[Dict[str, Any]], parent_path: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Flatten target fields to include nested complex types with virtual paths.

        :param self: Description
        :param fields: Description
        :type fields: List[Dict[str, Any]]
        :param parent_path: Description
        :type parent_path: str
        :return: Description
        :rtype: List[Dict[str, Any]]
        """
        flat = []
        for f in fields:
            # Create a copy to avoid modifying the original
            # Only keep necessary fields
            f_copy = {
                "id": f.get("id"),
                "path": f.get("path"),
                "description": f.get("description"),
                "short": f.get("short"),
                "definition": f.get("definition"),
            }

            current_name = f.get("path", "").split(".")[-1]

            if parent_path:
                virtual_path = f"{parent_path}.{current_name}"
            else:
                virtual_path = f.get("path", "")  # e.g. Patient.name

            f_copy["_virtual_path"] = virtual_path
            flat.append(f_copy)

            if "type_structure" in f:
                # Pass the virtual path as the parent path for the next level
                sub_fields = self._flatten_target_fields(
                    f["type_structure"], parent_path=virtual_path
                )
                flat.extend(sub_fields)
        return flat

    def _find_match_vectorized(
        self, source_field: Dict[str, Any], target_fields: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Match using TF-IDF vectorization and cosine similarity.

        :param self: Description
        :param source_field: Description
        :type source_field: Dict[str, Any]
        :param target_fields: Description
        :type target_fields: List[Dict[str, Any]]
        :return: Description
        :rtype: Tuple[Dict[str, Any] | None, float]
        """

        source_id = source_field.get("id", "")
        source_path = source_field.get("path", "")
        source_name = source_path.split(".")[-1] if source_path else source_id

        target_docs = []
        for t in target_fields:
            t_id = t.get("id", "")
            t_path = t.get("_virtual_path", t.get("path", ""))
            t_desc = (
                t.get("description", "")
                or t.get("short", "")
                or t.get("definition", "")
            )
            t_name = t_path.split(".")[-1]

            # combined text for TF-IDF
            text = f"{t_name} {t_id} {t_path} {t_desc}"
            target_docs.append(self._tokenize(text))

        if not target_docs:
            return None, 0.0

        query_text = f"{source_name} {source_id}"
        query_tokens = self._tokenize(query_text)

        # creat vocabulary
        vocab = set(query_tokens)
        for doc in target_docs:
            vocab.update(doc)
        vocab_list = sorted(list(vocab))
        vocab_index = {word: i for i, word in enumerate(vocab_list)}

        if not vocab_list:
            return None, 0.0

        N = len(target_docs)
        idf = {}
        for word in vocab_list:
            df = sum(1 for doc in target_docs if word in doc)
            idf[word] = math.log((N + 1) / (df + 1)) + 1

        query_vec = np.zeros(len(vocab_list))
        for word in query_tokens:
            if word in vocab_index:
                tf = query_tokens.count(word) / len(query_tokens)
                query_vec[vocab_index[word]] = tf * idf[word]

        # compute similarities between query and target docs
        best_idx = -1
        best_score = 0.0

        for i, doc in enumerate(target_docs):
            if not doc:
                continue
            doc_vec = np.zeros(len(vocab_list))
            for word in doc:
                if word in vocab_index:
                    tf = doc.count(word) / len(doc)
                    doc_vec[vocab_index[word]] = tf * idf[word]

            score = self._cosine_similarity(query_vec, doc_vec)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx != -1:
            return target_fields[best_idx], float(best_score)

        return None, 0.0

    def _calculate_w2v_similarity(
        self, source_tokens: List[str], target_tokens: List[str]
    ) -> float:
        if not source_tokens or not target_tokens:
            return 0.0

        # Simple average embedding comparison
        source_vecs = [self.model[word] for word in source_tokens if word in self.model]
        target_vecs = [self.model[word] for word in target_tokens if word in self.model]

        if not source_vecs or not target_vecs:
            return 0.0

        source_avg = np.mean(source_vecs, axis=0)
        target_avg = np.mean(target_vecs, axis=0)

        return self._cosine_similarity(source_avg, target_avg)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        :param self: Description
        :param vec1: Description
        :type vec1: np.ndarray
        :param vec2: Description
        :type vec2: np.ndarray
        :return: Description
        :rtype: float
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _get_field_text(self, field: Dict[str, Any]) -> str:
        """
        :param self: Description
        :param field: Description
        :type field: Dict[str, Any]
        :return: Description
        :rtype: str
        """
        path = field.get("_virtual_path", field.get("path", ""))
        name = path.split(".")[-1]

        description = field.get("description", "")
        short = field.get("short", "")

        return f"{name} {description} {short}".strip()

    def _tokenize(self, text: str) -> List[str]:
        """
        :param self: Description
        :param text: Description
        :type text: str
        :return: Description
        :rtype: List[str]
        """
        if not text:
            return []
        # split camelCase
        text = re.sub("([a-z0-9])([A-Z])", r"\1 \2", text)
        # replace dots and other separators with spaces
        text = text.replace(".", " ").replace("_", " ")
        return re.findall(r"\w+", text.lower())


def _field_types(field: Dict[str, Any]) -> Tuple[str, ...]:
    """Type codes of a field, covering both the raw and `conv_mappable` shapes."""

    if field.get("choice_types"):
        return tuple(str(code) for code in field["choice_types"] if code)

    raw = field.get("type")
    if isinstance(raw, str):
        return () if raw == "choice" else (raw,)
    if isinstance(raw, list):
        codes = []
        for item in raw:
            code = item.get("code") if isinstance(item, dict) else item
            if code:
                codes.append(str(code))
        return tuple(codes)
    return ()


def _cardinality(field: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """``(min, max)`` from either the nested ``cardinality`` dict or flat keys."""

    card = field.get("cardinality")
    if isinstance(card, dict):
        minimum, maximum = card.get("min"), card.get("max")
    else:
        minimum, maximum = field.get("min"), field.get("max")
    try:
        minimum = int(minimum) if minimum is not None else None
    except (TypeError, ValueError):
        minimum = None
    return minimum, str(maximum) if maximum is not None else None


def _is_repeating(maximum: Optional[str]) -> Optional[bool]:
    """Whether a max cardinality allows more than one value; ``None`` if unknown."""

    if maximum is None:
        return None
    if maximum == "*":
        return True
    try:
        return int(maximum) > 1
    except (TypeError, ValueError):
        return None


def _validation_context(origin: Dict[str, Any]) -> Dict[str, Any]:
    """Recover the identity and constraints the third flatten discarded."""

    minimum, maximum = _cardinality(origin)
    return {
        "element_id": origin.get("id"),
        "element_path": origin.get("path"),
        "types": _field_types(origin),
        "min": minimum,
        "max": maximum,
        "slice_name": origin.get("sliceName") or origin.get("slice_name"),
        "profile_url": origin.get("reference_target_profile") or origin.get("extension_url"),
    }


def _candidate_keys(candidate: "AutomapCandidate") -> Tuple[Tuple[str, str], ...]:
    """Keys a candidate may be identified by, strongest identity first."""

    attempts = (
        ("element-id", candidate.element_id),
        ("virtual-path", candidate.legacy_virtual_path),
        ("element-path", candidate.element_path),
    )
    return tuple((kind, key) for kind, key in attempts if key)


def _resolve_against_tree(
    candidate: "AutomapCandidate", tree: TargetTree
) -> Tuple[Any, str, Optional[str]]:
    """Find the one target-tree element a candidate denotes"""

    last_status = RESOLUTION_NOT_FOUND
    for key_kind, key in _candidate_keys(candidate):
        node, status = tree.resolve(key)
        if node is not None and status in RESOLVED_STATUSES:
            return node, status, key_kind
        if status != RESOLUTION_NOT_FOUND:
            last_status = status
    return None, last_status, None


def _unresolved_reason(candidate: "AutomapCandidate", tree: TargetTree) -> str:
    """Distinguish a forbidden element from an unknown one"""

    if any(tree.is_prohibited(key) for _, key in _candidate_keys(candidate)):
        return EXCLUDED_PROHIBITED
    return EXCLUDED_UNRESOLVED


def _prune_reason(
    candidate: "AutomapCandidate",
    node: Any,
    tree: TargetTree,
    source_field: Dict[str, Any],
) -> Optional[str]:
    """Why a resolved candidate must not be offered to a model, or None"""

    if node.prohibited or tree.is_prohibited(node.eid):
        return EXCLUDED_PROHIBITED

    source_repeating = _is_repeating(_cardinality(source_field)[1])
    target_repeating = _is_repeating(candidate.max)
    if source_repeating and target_repeating is False:
        return EXCLUDED_CARDINALITY

    if _clearly_incompatible_types(_field_types(source_field), candidate.types):
        return EXCLUDED_DATATYPE

    return None


def _clearly_incompatible_types(
    source_types: Tuple[str, ...], target_types: Tuple[str, ...]
) -> bool:
    """True only when no source type could plausibly supply any target type"""

    if not source_types or not target_types:
        return False

    source_families = {_PRIMITIVE_FAMILY.get(code) for code in source_types}
    target_families = {_PRIMITIVE_FAMILY.get(code) for code in target_types}
    if None in source_families or None in target_families:
        return False
    if _CONVERTIBLE_FAMILY in source_families or _CONVERTIBLE_FAMILY in target_families:
        return False
    return not (source_families & target_families)
