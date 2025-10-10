"""Text similarity computation utilities."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from frame2kg_eval.utils.normalise import normalise_label, normalise_id


class TextSimilarityComputer:
    """Compute text similarities using various methods."""
    
    def __init__(self, mode: str = "tfidf", model_name: Optional[str] = None):
        """Initialize text similarity computer.
        
        Args:
            mode: Similarity mode - "tfidf", "semantic", or "hybrid"
            model_name: Sentence transformer model name (for semantic mode)
        """
        self.mode = mode
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._semantic_model = None
        self._embedding_cache = {}
    
    def _load_semantic_model(self):
        """Lazy load sentence transformer model."""
        if self._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._semantic_model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for semantic similarity. "
                    "Install with: pip install sentence-transformers"
                )
    
    def extract_node_text(self, node: Dict, fields: Tuple[str, ...] = ("id", "label")) -> str:
        """Extract and normalize text from node fields.
        
        Args:
            node: Node dictionary
            fields: Fields to extract text from
        
        Returns:
            Concatenated normalized text
        """
        parts = []

        def iter_leaf_values(value):
            """Yield terminal values from nested containers."""
            if isinstance(value, dict):
                for item in value.values():
                    yield from iter_leaf_values(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from iter_leaf_values(item)
            else:
                yield value

        for field in fields:
            value = node.get(field, "")
            normaliser = normalise_id if field == "id" else normalise_label

            for leaf in iter_leaf_values(value):
                normalized = normaliser(leaf)
                if normalized:
                    parts.append(normalized)

        # Also include attribute values if present
        if "attributes" not in fields:
            attrs = node.get("attributes", {})
            if isinstance(attrs, dict):
                for key in ["appearance", "size", "color"]:
                    if key in attrs:
                        for leaf in iter_leaf_values(attrs[key]):
                            normalized = normalise_label(leaf)
                            if normalized:
                                parts.append(normalized)

        return " ".join(parts)
    
    def compute_tfidf_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """Compute TF-IDF similarity matrix.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
        
        Returns:
            Similarity matrix of shape (len(texts1), len(texts2))
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Use character n-grams for robustness
            vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3, 5),
                min_df=1,
                max_features=10000
            )
            
            # Combine all texts for fitting
            all_texts = texts1 + texts2
            if not all_texts or all(not t for t in all_texts):
                return np.zeros((len(texts1), len(texts2)), dtype=np.float32)
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # Split back into two sets
            tfidf1 = tfidf_matrix[:len(texts1)]
            tfidf2 = tfidf_matrix[len(texts1):]
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf1, tfidf2)
            
            return similarity.astype(np.float32)
            
        except ImportError:
            # Fallback to Jaccard similarity
            return self._jaccard_similarity(texts1, texts2)
    
    def _jaccard_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """Fallback Jaccard similarity computation.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
        
        Returns:
            Similarity matrix
        """
        n1, n2 = len(texts1), len(texts2)
        similarity = np.zeros((n1, n2), dtype=np.float32)
        
        for i, t1 in enumerate(texts1):
            tokens1 = set(t1.split())
            for j, t2 in enumerate(texts2):
                tokens2 = set(t2.split())
                
                if not tokens1 or not tokens2:
                    continue
                
                intersection = len(tokens1 & tokens2)
                union = len(tokens1 | tokens2)
                
                if union > 0:
                    similarity[i, j] = intersection / union
        
        return similarity
    
    def compute_semantic_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """Compute semantic similarity using sentence embeddings.
        
        Args:
            texts1: First set of texts
            texts2: Second set of texts
        
        Returns:
            Similarity matrix of shape (len(texts1), len(texts2))
        """
        self._load_semantic_model()
        
        # Check cache for texts2 (ground truth)
        cache_key = (self.model_name, tuple(texts2))
        if cache_key in self._embedding_cache:
            embeddings2 = self._embedding_cache[cache_key]
        else:
            # Encode and cache ground truth embeddings
            embeddings2 = self._semantic_model.encode(
                texts2,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self._embedding_cache[cache_key] = embeddings2
        
        # Encode predictions
        embeddings1 = self._semantic_model.encode(
            texts1,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Cosine similarity (dot product for normalized embeddings)
        similarity = np.clip(embeddings1 @ embeddings2.T, -1.0, 1.0)
        
        return similarity.astype(np.float32)
    
    def compute_similarity_matrix(self, 
                                 pred_nodes: List[Dict], 
                                 gt_nodes: List[Dict],
                                 text_fields: Tuple[str, ...] = ("id", "label")) -> np.ndarray:
        """Compute text similarity matrix between node sets.
        
        Args:
            pred_nodes: Predicted nodes
            gt_nodes: Ground truth nodes
            text_fields: Fields to use for text extraction
        
        Returns:
            Similarity matrix of shape (len(pred_nodes), len(gt_nodes))
        """
        # Extract text from nodes
        pred_texts = [self.extract_node_text(n, text_fields) for n in pred_nodes]
        gt_texts = [self.extract_node_text(n, text_fields) for n in gt_nodes]
        
        # Compute similarity based on mode
        if self.mode == "tfidf":
            return self.compute_tfidf_similarity(pred_texts, gt_texts)
        
        elif self.mode == "semantic":
            return self.compute_semantic_similarity(pred_texts, gt_texts)
        
        elif self.mode == "hybrid":
            # Blend TF-IDF and semantic similarities
            tfidf_sim = self.compute_tfidf_similarity(pred_texts, gt_texts)
            semantic_sim = self.compute_semantic_similarity(pred_texts, gt_texts)
            # Equal weighting for hybrid
            return (tfidf_sim + semantic_sim) / 2
        
        else:
            raise ValueError(f"Unknown similarity mode: {self.mode}")
    
    def clear_cache(self):
        """Clear embedding cache to free memory."""
        self._embedding_cache.clear()
