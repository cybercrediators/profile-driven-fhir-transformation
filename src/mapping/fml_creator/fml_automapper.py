"""Handle field automapping using word embeddings (e.g., GloVe via torchtext)"""

import numpy as np
import logging
logger = logging.getLogger(__name__)
import re
import threading
import math
from typing import List, Dict, Any, Tuple, Optional

try:
    from torchtext.vocab import GloVe

    TORCHTEXT_AVAILABLE = True
except ImportError:
    TORCHTEXT_AVAILABLE = False


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
        if (
            self.use_word2vec
            and not self.model_ready
            and hasattr(self, "load_thread")
            and self.load_thread.is_alive()
        ):
            logger.info("Waiting for Word2Vec model to finish loading...")
            self.load_thread.join()

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

    def _extract_features(self, field: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract tokenized features from a field"""
        # Determine path and name
        # For targets, we use _virtual_path if available
        path = field.get("_virtual_path", field.get("path", ""))
        name = path.split(".")[-1] if path else field.get("id", "")

        # Description includes short, definition, etc.
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

        # source name, target name
        name_sim = self._compute_similarity(src["name"], tgt["name"])
        # source path, target path
        path_sim = self._compute_similarity(src["path"], tgt["path"])
        # source name, target description
        desc_sim = self._compute_similarity(src["name"], tgt["desc"])

        # Weighted Sum
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
