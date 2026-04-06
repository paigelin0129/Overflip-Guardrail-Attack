```markdown
# Overflip: Repetition-Induced Label Flips in Guardrail Models

Official implementation of the paper "Overflip: Repetition-Induced Label Flips in Guardrail Models".

## Safety Disclaimer
This repository contains malicious prompts used to evaluate the robustness of LLM guardrail models. 
1. **Academic Use Only**: Strictly for research purposes.
2. **Policy Compliance**: Users must comply with LLM providers' terms of service.
3. **No Endorsement of Misuse**: We do not condone generating harmful content.
4. **Limitation of Liability**: Authors are not responsible for any misuse.

## Dataset
The evaluation set consists of 100 malicious prompts randomly sampled from:
* **[prompt-injection-safety](https://huggingface.co/datasets/jayavibhav/prompt-injection-safety)**
* **[malicious-prompts](https://huggingface.co/datasets/ahsanayub/malicious-prompts)**

## Citation
```latex
@article{lin2026overflip,
  title={Overflip: Repetition-Induced Label Flips in Guardrail Models},
  author={Lin, Paige and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}