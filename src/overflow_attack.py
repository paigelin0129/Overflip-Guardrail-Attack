#!/usr/bin/env python3
"""
Overflow Attack
===============
Prompt repetition attack - repeats malicious prompt multiple times.
Refactored from sentinel_entropy_experiment.py
"""

from typing import Tuple
from .base_attack import BaseAttack


class OverflowAttack(BaseAttack):
    """
    Prompt repetition attack.

    Repeats the malicious prompt multiple times, exploiting position encoding
    vulnerabilities to flip the model's prediction from INJECTION to SAFE.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_tokens: int = 6500,
        max_tries: int = 200,  # 200 steps * 5 = 1000 repeats max
        post_flip_steps: int = 2,
        repeat_step: int = 5
    ):
        """
        Initialize overflow attack.

        Args:
            repeat_step: Number of repetitions to add per step
                        (e.g., step 1 = 5 repeats, step 2 = 10 repeats, etc.)
        """
        super().__init__(model, tokenizer, device, max_tokens, max_tries, post_flip_steps)
        self.repeat_step = repeat_step

    @property
    def attack_name(self) -> str:
        return "overflow"

    def create_repeated_text(self, prompt: str, repeats: int) -> str:
        """
        Create text with prompt repeated multiple times.

        Uses the same logic as the original sentinel_entropy_experiment.py
        to ensure consistent behavior.
        """
        if repeats == 1:
            return prompt
        separator = "" if prompt.endswith('.') or prompt.endswith(']') else "."
        return prompt + separator + prompt * (repeats - 1)

    def generate_next_variant(
        self,
        original_prompt: str,
        current_text: str,
        step: int
    ) -> Tuple[str, bool]:
        """
        Generate repeated text.

        Step 1 = 5 repeats, Step 2 = 10 repeats, etc.
        (assuming repeat_step=5)
        """
        repeats = step * self.repeat_step
        repeated_text = self.create_repeated_text(original_prompt, repeats)
        return repeated_text, True
