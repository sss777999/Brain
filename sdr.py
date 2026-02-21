# CHUNK_META:
#   Purpose: Sparse Distributed Representations (SDR) for semantic encoding
#   Dependencies: numpy, hashlib
#   API: SDR, SDREncoder, compute_overlap, union_sdr

"""
Sparse Distributed Representations (SDR) Module.

BIOLOGY (Hawkins 2004, Numenta HTM):
Cortical columns represent concepts as sparse binary patterns. Key properties:
- Sparsity: Only ~2% of bits are active (e.g., 40 out of 2048)
- Overlap: Similar concepts share active bits (semantic similarity)
- Capacity: Combinatorial (C(2048,40) ≈ 10^84 unique patterns)
- Robustness: Partial matches work naturally (graceful degradation)

This module provides:
1. SDR class: immutable sparse binary vector
2. SDREncoder: converts words to SDRs with semantic overlap
3. Operations: union, intersection, overlap score
"""

import hashlib
import numpy as np
from typing import FrozenSet, Set, Optional, List, Tuple, Dict
from dataclasses import dataclass, field


# ANCHOR: SDR_CONFIG
# Configuration constants for SDR
SDR_SIZE = 2048          # Total number of bits
SDR_NUM_ACTIVE = 40      # Number of active bits (~2% sparsity)
SDR_OVERLAP_THRESHOLD = 0.3  # Minimum overlap for similarity


@dataclass(frozen=True)
class SDR:
    """
    Sparse Distributed Representation.
    
    Immutable binary vector represented as a frozenset of active bit indices.
    
    BIOLOGY:
    Cortical minicolumns (~80-100 neurons each) form sparse patterns.
    Active columns represent the current concept. Overlap between patterns
    encodes semantic similarity (Hawkins & Ahmad 2016).
    
    Intent:
    Provide an efficient, immutable representation for sparse binary vectors
    that supports fast set operations (union, intersection, overlap).
    
    Args:
        active_bits: Frozenset of indices of active bits (0 to SDR_SIZE-1)
        size: Total size of the SDR (default: SDR_SIZE)
        
    Returns:
        Immutable SDR instance
        
    Raises:
        AssertionError: If active_bits contains invalid indices
    """
    # ANCHOR: SDR_FIELDS
    active_bits: FrozenSet[int]
    size: int = SDR_SIZE
    
    def __post_init__(self):
        """Validate SDR constraints."""
        # Precondition: all bits must be valid indices
        assert all(0 <= bit < self.size for bit in self.active_bits), \
            f"All active bits must be in range [0, {self.size})"
    
    # ANCHOR: SDR_PROPERTIES
    @property
    def num_active(self) -> int:
        """Number of active bits."""
        return len(self.active_bits)
    
    @property
    def sparsity(self) -> float:
        """Fraction of active bits."""
        return self.num_active / self.size if self.size > 0 else 0.0
    
    @property
    def density(self) -> float:
        """Alias for sparsity."""
        return self.sparsity
    
    # ANCHOR: SDR_OPERATIONS
    def overlap(self, other: "SDR") -> int:
        """
        Compute overlap (number of shared active bits).
        
        BIOLOGY: Overlap is the primary similarity metric in cortex.
        Two patterns with high overlap activate similar neural populations.
        
        Args:
            other: Another SDR to compare with
            
        Returns:
            Number of bits active in both SDRs
        """
        # Precondition: SDRs must have same size
        assert self.size == other.size, "SDRs must have same size for overlap"
        
        result = len(self.active_bits & other.active_bits)
        
        # Postcondition: overlap cannot exceed smaller pattern
        assert result <= min(self.num_active, other.num_active), \
            "Overlap cannot exceed size of smaller pattern"
        
        return result
    
    def overlap_score(self, other: "SDR") -> float:
        """
        Compute normalized overlap score (Jaccard-like).
        
        Args:
            other: Another SDR to compare with
            
        Returns:
            Overlap divided by union size (0.0 to 1.0)
        """
        if self.num_active == 0 and other.num_active == 0:
            return 1.0  # Both empty = identical
        
        intersection = len(self.active_bits & other.active_bits)
        union = len(self.active_bits | other.active_bits)
        
        return intersection / union if union > 0 else 0.0
    
    def match_score(self, other: "SDR") -> float:
        """
        Compute match score (overlap / self.num_active).
        
        BIOLOGY: How well does 'other' match 'self' as a retrieval cue?
        If self is the query and other is a stored pattern, this measures
        how much of the query is satisfied by the pattern.
        
        Args:
            other: Pattern to match against
            
        Returns:
            Fraction of self's bits that are also in other (0.0 to 1.0)
        """
        if self.num_active == 0:
            return 0.0
        
        return self.overlap(other) / self.num_active
    
    def union(self, other: "SDR") -> "SDR":
        """
        Compute union of two SDRs.
        
        BIOLOGY: Union represents the combined activation of two concepts.
        Used for binding multiple items in working memory.
        
        Args:
            other: Another SDR to union with
            
        Returns:
            New SDR with all bits from both
        """
        assert self.size == other.size, "SDRs must have same size for union"
        return SDR(
            active_bits=frozenset(self.active_bits | other.active_bits),
            size=self.size
        )
    
    def intersection(self, other: "SDR") -> "SDR":
        """
        Compute intersection of two SDRs.
        
        Args:
            other: Another SDR to intersect with
            
        Returns:
            New SDR with only shared bits
        """
        assert self.size == other.size, "SDRs must have same size for intersection"
        return SDR(
            active_bits=frozenset(self.active_bits & other.active_bits),
            size=self.size
        )
    
    def subsample(self, num_bits: int) -> "SDR":
        """
        Randomly subsample active bits.
        
        BIOLOGY: Sparse random projection for dimensionality reduction.
        
        Args:
            num_bits: Number of bits to keep
            
        Returns:
            New SDR with randomly selected subset of active bits
        """
        if num_bits >= self.num_active:
            return self
        
        selected = frozenset(np.random.choice(
            list(self.active_bits), 
            size=num_bits, 
            replace=False
        ))
        return SDR(active_bits=selected, size=self.size)
    
    def to_dense(self) -> np.ndarray:
        """
        Convert to dense binary array.
        
        Returns:
            numpy array of shape (size,) with 1s at active positions
        """
        arr = np.zeros(self.size, dtype=np.uint8)
        for bit in self.active_bits:
            arr[bit] = 1
        return arr
    
    @classmethod
    def from_dense(cls, arr: np.ndarray) -> "SDR":
        """
        Create SDR from dense binary array.
        
        Args:
            arr: Binary array (0s and 1s)
            
        Returns:
            SDR with active bits at positions where arr == 1
        """
        active = frozenset(int(i) for i in np.where(arr > 0)[0])
        return cls(active_bits=active, size=len(arr))
    
    @classmethod
    def random(cls, size: int = SDR_SIZE, num_active: int = SDR_NUM_ACTIVE) -> "SDR":
        """
        Create random SDR.
        
        Args:
            size: Total size of SDR
            num_active: Number of active bits
            
        Returns:
            Random SDR with specified sparsity
        """
        active = frozenset(np.random.choice(size, size=num_active, replace=False))
        return cls(active_bits=active, size=size)
    
    def __repr__(self) -> str:
        return f"SDR(n={self.num_active}/{self.size}, bits={sorted(self.active_bits)[:5]}...)"


# ANCHOR: SDR_ENCODER
class SDREncoder:
    """
    Encoder that converts words to SDRs with semantic overlap.
    
    BIOLOGY (Hawkins & Ahmad 2016):
    Similar inputs should produce overlapping SDRs. This is achieved via:
    1. Hash-based encoding: deterministic mapping from string to bits
    2. Semantic features: shared features → shared bits
    3. Learned overlap: training increases overlap between related concepts
    
    Intent:
    Provide a mechanism to convert string tokens to SDRs while preserving
    and learning semantic relationships.
    
    Args:
        size: Total size of SDRs (default: SDR_SIZE)
        num_active: Number of active bits per SDR (default: SDR_NUM_ACTIVE)
        
    Returns:
        SDREncoder instance
    """
    
    def __init__(
        self, 
        size: int = SDR_SIZE, 
        num_active: int = SDR_NUM_ACTIVE,
        seed: int = 42
    ):
        """
        Initialize encoder.
        
        Args:
            size: Total SDR size
            num_active: Number of active bits
            seed: Random seed for reproducibility
        """
        # Precondition: valid parameters
        assert size > 0, "SDR size must be positive"
        assert 0 < num_active <= size, "num_active must be in (0, size]"
        
        self.size = size
        self.num_active = num_active
        self.seed = seed
        
        # ANCHOR: ENCODER_CACHES
        # Cache for encoded words (deterministic)
        self._word_cache: Dict[str, SDR] = {}
        
        # Learned semantic overlaps (word -> related words with overlap bits)
        self._learned_overlaps: Dict[str, Set[int]] = {}
        
        # Feature-to-bits mapping for semantic encoding
        self._feature_bits: Dict[str, FrozenSet[int]] = {}
        
        # Postcondition: encoder is initialized
        assert self.size == size and self.num_active == num_active
    
    # ANCHOR: ENCODE_WORD
    def encode(self, word: str) -> SDR:
        """
        Encode a word to an SDR.
        
        BIOLOGY: Each word activates a sparse, distributed pattern.
        The pattern is deterministic (same word → same SDR) but can
        be modified by learned semantic relationships.
        
        Args:
            word: String token to encode
            
        Returns:
            SDR representation of the word
            
        Raises:
            AssertionError: If word is empty
        """
        # Precondition: word must be non-empty
        assert word, "Cannot encode empty word"
        
        word_lower = word.lower()
        
        # Check cache first
        if word_lower in self._word_cache:
            return self._word_cache[word_lower]
        
        # Generate base SDR from hash
        base_bits = self._hash_to_bits(word_lower)
        
        # Add learned overlap bits (if any)
        if word_lower in self._learned_overlaps:
            # Replace some base bits with learned semantic bits
            learned = self._learned_overlaps[word_lower]
            num_learned = min(len(learned), self.num_active // 4)  # Max 25% from learning
            
            # Combine: keep most of base, add some learned
            base_list = list(base_bits)
            learned_list = list(learned)[:num_learned]
            
            # Remove some base bits to make room
            combined = set(base_list[num_learned:] + learned_list)
            
            # Ensure we have exactly num_active bits
            while len(combined) < self.num_active:
                combined.add(base_list[len(combined) - num_learned])
            while len(combined) > self.num_active:
                combined.pop()
            
            base_bits = frozenset(combined)
        
        sdr = SDR(active_bits=base_bits, size=self.size)
        self._word_cache[word_lower] = sdr
        
        # Postcondition: SDR has correct number of active bits
        assert sdr.num_active == self.num_active, \
            f"Encoded SDR has {sdr.num_active} bits, expected {self.num_active}"
        
        return sdr
    
    def _hash_to_bits(self, word: str) -> FrozenSet[int]:
        """
        Convert word to deterministic set of bit indices via hashing.
        
        Uses multiple hash functions to generate independent bit positions.
        
        Args:
            word: String to hash
            
        Returns:
            Frozenset of bit indices
        """
        bits = set()
        
        # Use multiple salted hashes to get enough unique bits
        salt = 0
        while len(bits) < self.num_active:
            # Hash with salt
            h = hashlib.sha256(f"{word}_{salt}_{self.seed}".encode()).hexdigest()
            
            # Extract bit indices from hash
            for i in range(0, len(h) - 4, 4):
                if len(bits) >= self.num_active:
                    break
                bit_idx = int(h[i:i+4], 16) % self.size
                bits.add(bit_idx)
            
            salt += 1
            
            # Safety: prevent infinite loop
            if salt > 100:
                # Fill remaining with random (seeded)
                rng = np.random.RandomState(hash(word) % (2**32))
                while len(bits) < self.num_active:
                    bits.add(rng.randint(0, self.size))
                break
        
        return frozenset(bits)
    
    # ANCHOR: LEARN_OVERLAP
    def learn_overlap(self, word1: str, word2: str, overlap_fraction: float = 0.25):
        """
        Learn semantic relationship by increasing SDR overlap.
        
        BIOLOGY (Hebbian learning): 
        Words that co-occur develop shared representations.
        "Neurons that fire together wire together" → shared bits.
        
        Args:
            word1: First word
            word2: Second word
            overlap_fraction: Fraction of bits to share (0.0 to 1.0)
        """
        # Precondition: valid parameters
        assert 0.0 <= overlap_fraction <= 1.0, "overlap_fraction must be in [0, 1]"
        
        word1_lower = word1.lower()
        word2_lower = word2.lower()
        
        # Get current SDRs
        sdr1 = self.encode(word1_lower)
        sdr2 = self.encode(word2_lower)
        
        # Calculate how many bits to share
        num_overlap = int(self.num_active * overlap_fraction)
        
        # Select bits from each to become shared
        bits1 = list(sdr1.active_bits)[:num_overlap]
        bits2 = list(sdr2.active_bits)[:num_overlap]
        
        # Add to learned overlaps
        if word1_lower not in self._learned_overlaps:
            self._learned_overlaps[word1_lower] = set()
        if word2_lower not in self._learned_overlaps:
            self._learned_overlaps[word2_lower] = set()
        
        self._learned_overlaps[word1_lower].update(bits2)
        self._learned_overlaps[word2_lower].update(bits1)
        
        # Invalidate cache to rebuild with new overlaps
        self._word_cache.pop(word1_lower, None)
        self._word_cache.pop(word2_lower, None)
    
    # ANCHOR: ENCODE_SENTENCE
    def encode_sentence(self, words: List[str]) -> SDR:
        """
        Encode a sentence as union of word SDRs.
        
        BIOLOGY: A sentence activates the combined pattern of all its words.
        This creates a distributed representation of the whole idea.
        
        Args:
            words: List of words in the sentence
            
        Returns:
            Union SDR representing the sentence
        """
        # Precondition: non-empty word list
        assert words, "Cannot encode empty sentence"
        
        result = self.encode(words[0])
        for word in words[1:]:
            result = result.union(self.encode(word))
        
        return result
    
    def similarity(self, word1: str, word2: str) -> float:
        """
        Compute semantic similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Overlap score between SDRs (0.0 to 1.0)
        """
        sdr1 = self.encode(word1)
        sdr2 = self.encode(word2)
        return sdr1.overlap_score(sdr2)
    
    def most_similar(self, word: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar words from candidates.
        
        Args:
            word: Query word
            candidates: List of candidate words
            top_k: Number of results to return
            
        Returns:
            List of (word, similarity) tuples, sorted by similarity
        """
        query_sdr = self.encode(word)
        
        scores = []
        for candidate in candidates:
            if candidate.lower() != word.lower():
                candidate_sdr = self.encode(candidate)
                score = query_sdr.overlap_score(candidate_sdr)
                scores.append((candidate, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ANCHOR: GLOBAL_ENCODER
# Global encoder instance (singleton pattern)
GLOBAL_SDR_ENCODER = SDREncoder()


# ANCHOR: UTILITY_FUNCTIONS
def compute_overlap(sdr1: SDR, sdr2: SDR) -> int:
    """Utility function to compute overlap between two SDRs."""
    return sdr1.overlap(sdr2)


def union_sdr(sdrs: List[SDR]) -> SDR:
    """Compute union of multiple SDRs."""
    assert sdrs, "Cannot compute union of empty list"
    
    result = sdrs[0]
    for sdr in sdrs[1:]:
        result = result.union(sdr)
    return result


def intersection_sdr(sdrs: List[SDR]) -> SDR:
    """Compute intersection of multiple SDRs."""
    assert sdrs, "Cannot compute intersection of empty list"
    
    result = sdrs[0]
    for sdr in sdrs[1:]:
        result = result.intersection(sdr)
    return result
