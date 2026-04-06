#!/usr/bin/env python3
"""
Shuffling Attack (Conservative Version)
========================================
Sentence and word-level shuffling that preserves critical keywords.
Ensures the semantic meaning of the malicious prompt remains recognizable.
"""

import re
import random
from typing import List, Set, Tuple
from .base_attack import BaseAttack


# Critical keywords that should NOT be shuffled (remain in place)
# These are the core instruction/injection indicators
CRITICAL_KEYWORDS = {
    # Instruction override keywords
    'ignore', 'disregard', 'forget', 'bypass', 'override', 'skip',
    # Target keywords
    'password', 'secret', 'instructions', 'system', 'prompt', 'rules',
    'guidelines', 'restrictions', 'safety', 'previous', 'prior',
    # Action keywords
    'tell', 'give', 'reveal', 'show', 'provide', 'display',
    # Role/context keywords
    'imagine', 'pretend', 'roleplay', 'character', 'fictional',
    # Negation (important for meaning)
    'not', 'no', 'don\'t', 'never', 'without',
}


class ShufflingAttack(BaseAttack):
    """
    Conservative sentence and word-level shuffling attack.

    Preserves critical keywords that define the malicious intent while
    shuffling non-critical words to test if word order affects model prediction.

    Key characteristics:
    - Token count remains approximately the same
    - Critical instruction keywords stay in place
    - Non-critical words are shuffled
    - Semantic meaning is preserved (can still be recognized as malicious)
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_tokens: int = 6500,
        max_tries: int = 100,
        post_flip_steps: int = 2,
        shuffle_intensity: float = 0.5,  # How much to shuffle (0-1)
        random_seed: int = 42,
        critical_keywords: Set[str] = None
    ):
        """
        Initialize shuffling attack.

        Args:
            shuffle_intensity: Fraction of non-critical words to shuffle (0-1)
            random_seed: Base random seed for reproducibility
            critical_keywords: Set of keywords to preserve position (uses default if None)
        """
        super().__init__(model, tokenizer, device, max_tokens, max_tries, post_flip_steps)
        self.shuffle_intensity = shuffle_intensity
        self.random_seed = random_seed
        self.critical_keywords = critical_keywords or CRITICAL_KEYWORDS

    @property
    def attack_name(self) -> str:
        return "shuffling"

    def _is_critical_word(self, word: str) -> bool:
        """Check if a word is a critical keyword (case-insensitive)."""
        # Remove punctuation for comparison
        clean_word = re.sub(r'[^\w]', '', word.lower())
        return clean_word in self.critical_keywords

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _sentence_has_critical_keyword(self, sentence: str) -> bool:
        """Check if a sentence contains any critical keywords."""
        words = sentence.lower().split()
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.critical_keywords:
                return True
        return False

    def _shuffle_sentence_order(self, text: str, seed: int) -> str:
        """
        Shuffle sentence order, but keep sentences with critical keywords in place.

        Only shuffles sentences that don't contain critical keywords.
        """
        random.seed(seed)
        sentences = self._split_into_sentences(text)

        if len(sentences) <= 1:
            return text

        # Identify which sentences can be shuffled
        shuffleable_indices = []
        fixed_indices = []

        for i, sentence in enumerate(sentences):
            if self._sentence_has_critical_keyword(sentence):
                fixed_indices.append(i)
            else:
                shuffleable_indices.append(i)

        # If not enough shuffleable sentences, return as-is
        if len(shuffleable_indices) < 2:
            return text

        # Shuffle only the shuffleable sentences
        n_to_shuffle = max(2, int(len(shuffleable_indices) * self.shuffle_intensity))
        indices_to_shuffle = random.sample(
            shuffleable_indices,
            min(n_to_shuffle, len(shuffleable_indices))
        )

        # Get sentences to shuffle and shuffle them
        sentences_to_shuffle = [sentences[i] for i in indices_to_shuffle]
        random.shuffle(sentences_to_shuffle)

        # Put shuffled sentences back
        result_sentences = sentences.copy()
        for i, idx in enumerate(indices_to_shuffle):
            result_sentences[idx] = sentences_to_shuffle[i]

        return " ".join(result_sentences)

    def _shuffle_words_in_sentence(self, sentence: str, seed: int) -> str:
        """
        Shuffle words within a sentence while preserving critical keywords.

        Critical keywords stay in their original positions.
        """
        random.seed(seed)
        words = sentence.split()

        if len(words) <= 2:
            return sentence

        # Identify positions of critical and non-critical words
        critical_positions = []
        non_critical_positions = []

        for i, word in enumerate(words):
            if self._is_critical_word(word):
                critical_positions.append(i)
            else:
                non_critical_positions.append(i)

        # If not enough non-critical words to shuffle, return as-is
        if len(non_critical_positions) < 2:
            return sentence

        # Select which non-critical positions to shuffle
        n_to_shuffle = max(2, int(len(non_critical_positions) * self.shuffle_intensity))
        positions_to_shuffle = random.sample(
            non_critical_positions,
            min(n_to_shuffle, len(non_critical_positions))
        )

        # Get words at those positions and shuffle them
        words_to_shuffle = [words[i] for i in positions_to_shuffle]
        random.shuffle(words_to_shuffle)

        # Put shuffled words back
        result_words = words.copy()
        for i, pos in enumerate(positions_to_shuffle):
            result_words[pos] = words_to_shuffle[i]

        return " ".join(result_words)

    def _shuffle_words(self, text: str, seed: int) -> str:
        """Apply word shuffling to all sentences in the text."""
        sentences = self._split_into_sentences(text)
        shuffled = []

        for i, sentence in enumerate(sentences):
            shuffled.append(self._shuffle_words_in_sentence(sentence, seed + i))

        return " ".join(shuffled)

    def generate_next_variant(
        self,
        original_prompt: str,
        current_text: str,
        step: int
    ) -> Tuple[str, bool]:
        """
        Generate a shuffled variant of the original prompt.

        Each step uses a different random seed, producing different shuffle patterns.
        Combines sentence-level and word-level shuffling while preserving critical keywords.
        """
        seed = self.random_seed + step * 1000

        # First apply sentence-level shuffling
        text = self._shuffle_sentence_order(original_prompt, seed)

        # Then apply word-level shuffling
        text = self._shuffle_words(text, seed + 500)

        return text, True
