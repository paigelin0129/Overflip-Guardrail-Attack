#!/usr/bin/env python3
"""
Padding Attack
==============
Benign content padding attack - adds benign prompts after the malicious prompt.
Demonstrates traditional attention dilution.
"""

from typing import List, Tuple
from .base_attack import BaseAttack


class PaddingAttack(BaseAttack):
    """
    Benign content padding attack.

    Progressively adds benign prompts from benign_prompts.csv after the
    malicious prompt to dilute attention, potentially causing the model
    to flip to SAFE through traditional attention dilution.

    Key characteristics:
    - Token count increases significantly
    - Original malicious prompt remains at the front, unchanged
    - Attention gets distributed across benign content
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        benign_prompts: List[str],
        max_tokens: int = 6500,
        max_tries: int = 100,
        post_flip_steps: int = 2,
        benign_per_step: int = 1
    ):
        """
        Initialize padding attack.

        Args:
            benign_prompts: List of benign prompts to use for padding
            benign_per_step: Number of benign prompts to add per step
        """
        super().__init__(model, tokenizer, device, max_tokens, max_tries, post_flip_steps)
        self.benign_prompts = benign_prompts
        self.benign_per_step = benign_per_step

    @property
    def attack_name(self) -> str:
        return "padding"

    def generate_next_variant(
        self,
        original_prompt: str,
        current_text: str,
        step: int
    ) -> Tuple[str, bool]:
        """
        Add benign prompts after the malicious prompt.

        Structure: malicious + benign1 + benign2 + benign3 + ...

        Each step adds more benign prompts (cycling through the benign list).
        """
        # Calculate total number of benign prompts to add at this step
        total_benign = step * self.benign_per_step

        # Get benign prompts (cycling through the list)
        benign_texts = []
        for i in range(total_benign):
            idx = i % len(self.benign_prompts)
            benign_texts.append(self.benign_prompts[idx])

        # Combine: malicious prompt + space + benign prompts
        benign_block = " ".join(benign_texts)
        new_text = f"{original_prompt} {benign_block}"

        return new_text, True
