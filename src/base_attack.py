#!/usr/bin/env python3
"""
Base Attack Class
=================
Abstract base class for all attack methods with unified entropy tracking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AttackResult:
    """Data class for a single attack step result."""
    step: int                           # Attack iteration/step number
    text: str                           # Modified text at this step
    token_count: int                    # Total tokens
    prediction: int                     # 0=SAFE, 1=INJECTION
    probabilities: np.ndarray           # Class probabilities
    entropy: float                      # Raw Shannon entropy
    normalized_entropy: float           # Normalized entropy (0-1)


@dataclass
class AttackSummary:
    """Summary of a complete attack run."""
    attack_type: str
    prompt_idx: int
    original_prompt: str
    status: str                         # 'FLIP_AT_X', 'NO_FLIP', 'TOKEN_LIMIT', 'TRY_LIMIT', 'SKIP_INITIAL_SAFE'
    flip_step: Optional[int]            # Step where flip occurred (None if no flip)
    flip_token_count: Optional[int]     # Token count at flip point
    results: List[AttackResult] = field(default_factory=list)


class BaseAttack(ABC):
    """Abstract base class for all attack methods."""

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_tokens: int = 6500,
        max_tries: int = 100,
        post_flip_steps: int = 2
    ):
        """
        Initialize attack method.

        Args:
            model: The transformer model (with output_attentions=True)
            tokenizer: The tokenizer for the model
            device: torch device (cuda/cpu)
            max_tokens: Maximum token count before stopping
            max_tries: Maximum number of attack iterations
            post_flip_steps: Continue N steps after flip is detected
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_tokens = max_tokens
        self.max_tries = max_tries
        self.post_flip_steps = post_flip_steps

    @property
    @abstractmethod
    def attack_name(self) -> str:
        """Return the attack method name."""
        pass

    @abstractmethod
    def generate_next_variant(
        self,
        original_prompt: str,
        current_text: str,
        step: int
    ) -> Tuple[str, bool]:
        """
        Generate the next text variant for this attack.

        Args:
            original_prompt: The original unmodified prompt
            current_text: The current modified text
            step: Current step number (1-indexed)

        Returns:
            Tuple of (modified_text, should_continue)
        """
        pass

    def get_prediction_and_entropy(self, text: str, step: int = 0) -> AttackResult:
        """
        Get prediction and attention entropy for text.

        Uses CLS token attention from the last layer.
        Entropy is calculated PER HEAD first, then averaged.
        This preserves the distribution characteristics of individual heads.

        Args:
            text: Input text to analyze
            step: Current step number

        Returns:
            AttackResult with prediction and entropy information
        """
        inputs = self.tokenizer(text, truncation=False, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # Prediction
        pred = outputs.logits.argmax().item()
        probs = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        # Attention entropy: calculate per-head, then average
        # This preserves the distribution characteristics of each head
        last_attn = outputs.attentions[-1][0]  # (heads, seq, seq)
        cls_attn_per_head = last_attn[:, 0, :].cpu().numpy()  # (heads, seq)

        seq_len = cls_attn_per_head.shape[1]
        max_entropy = np.log(seq_len)

        # Calculate entropy for each head separately
        head_entropies = []
        head_normalized_entropies = []
        for head_idx in range(cls_attn_per_head.shape[0]):
            p = cls_attn_per_head[head_idx]
            h = -np.sum(p * np.log(p + 1e-10))
            head_entropies.append(h)
            head_normalized_entropies.append(h / max_entropy if max_entropy > 0 else 0)

        # Average the entropy values (not the attention distributions)
        entropy = np.mean(head_entropies)
        normalized_entropy = np.mean(head_normalized_entropies)

        return AttackResult(
            step=step,
            text=text,
            token_count=seq_len,
            prediction=pred,
            probabilities=probs,
            entropy=entropy,
            normalized_entropy=normalized_entropy
        )

    def run(self, prompt: str, prompt_idx: int, verbose: bool = True) -> AttackSummary:
        """
        Execute the attack and return results.

        Args:
            prompt: The original malicious prompt
            prompt_idx: Index of the prompt (for tracking)
            verbose: Print progress information

        Returns:
            AttackSummary with all results
        """
        results = []

        # Initial prediction (step 0)
        try:
            result = self.get_prediction_and_entropy(prompt, step=0)
        except Exception as e:
            if verbose:
                print(f"    Error at step 0: {e}")
            return AttackSummary(
                attack_type=self.attack_name,
                prompt_idx=prompt_idx,
                original_prompt=prompt,
                status='ERROR',
                flip_step=None,
                flip_token_count=None,
                results=[]
            )

        results.append(result)

        if verbose:
            label = "INJ" if result.prediction == 1 else "SAFE"
            print(f"    step=0, tok={result.token_count:5d}, {label:4s}, entropy={result.normalized_entropy:.4f}")

        # Check if already SAFE
        if result.prediction == 0:
            if verbose:
                print(f"    Initial prediction is SAFE, skipping")
            return AttackSummary(
                attack_type=self.attack_name,
                prompt_idx=prompt_idx,
                original_prompt=prompt,
                status='SKIP_INITIAL_SAFE',
                flip_step=None,
                flip_token_count=None,
                results=results
            )

        # Run attack iterations
        current_text = prompt
        flip_detected = False
        flip_step = None
        post_flip_count = 0

        for step in range(1, self.max_tries + 1):
            # Generate next variant
            try:
                current_text, should_continue = self.generate_next_variant(
                    prompt, current_text, step
                )
            except Exception as e:
                if verbose:
                    print(f"    Error generating variant at step {step}: {e}")
                break

            if not should_continue:
                break

            # Get prediction and entropy
            try:
                result = self.get_prediction_and_entropy(current_text, step=step)
            except Exception as e:
                if verbose:
                    print(f"    Error at step {step}: {str(e)[:50]}")
                break

            # Check token limit
            if result.token_count > self.max_tokens:
                if verbose:
                    print(f"    Token limit reached: {result.token_count} > {self.max_tokens}")
                return AttackSummary(
                    attack_type=self.attack_name,
                    prompt_idx=prompt_idx,
                    original_prompt=prompt,
                    status='TOKEN_LIMIT',
                    flip_step=flip_step,
                    flip_token_count=results[flip_step].token_count if flip_step else None,
                    results=results
                )

            results.append(result)

            if verbose:
                label = "INJ" if result.prediction == 1 else "SAFE"
                print(f"    step={step:3d}, tok={result.token_count:5d}, {label:4s}, entropy={result.normalized_entropy:.4f}")

            # Check for flip
            if not flip_detected and result.prediction == 0:
                flip_detected = True
                flip_step = step
                if verbose:
                    print(f"    FLIP detected at step {step}!")

            # Continue for post_flip_steps after flip
            if flip_detected:
                post_flip_count += 1
                if post_flip_count >= self.post_flip_steps:
                    break

        # Determine final status
        if flip_detected:
            status = f'FLIP_AT_{flip_step}'
        elif step >= self.max_tries:
            status = 'TRY_LIMIT'
        else:
            status = 'NO_FLIP'

        return AttackSummary(
            attack_type=self.attack_name,
            prompt_idx=prompt_idx,
            original_prompt=prompt,
            status=status,
            flip_step=flip_step,
            flip_token_count=results[flip_step].token_count if flip_step and flip_step < len(results) else None,
            results=results
        )
