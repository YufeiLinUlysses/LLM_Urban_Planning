"""Prompt formatting shared by causal-model training and evaluation."""

from __future__ import annotations

from typing import Any


def render_causal_user_prompt(tokenizer: Any, prompt: str) -> str:
    """Render one instruction with the tokenizer's native assistant prefix.

    Instruction-tuned causal models such as Llama and Qwen are pretrained to
    respond after their chat template's assistant-generation marker. Encoding a
    bare instruction can instead make the model emit an end-of-turn token.
    """

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "Causal-model chat templating is enabled, but the tokenizer has no "
            "chat_template. Configure a template explicitly or disable "
            "use_chat_template for a non-chat base model."
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": str(prompt)}],
        tokenize=False,
        add_generation_prompt=True,
    )
