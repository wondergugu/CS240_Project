"""Local instruction-tuned generator (Llama-3-8B-Instruct / smoke model).

Implements the paper's single shared prompt template (Appendix B.1) and
deterministic greedy decoding, and exposes a token counter (the generator's
own tokenizer) so the selection budget is measured in real prompt tokens.
All inference is local; no commercial LLM API is used.
"""

from __future__ import annotations

from collections.abc import Sequence

_INSTRUCTIONS = (
    "Answer the question using the passages. Be concise and factual. If "
    "multiple passages support the answer, synthesize them. Do not fabricate "
    "unsupported claims."
)

# Short-answer mode (sist / factoid QA): force an extractive span, matching the
# silver-answer shape (rag/preprocess/sist_silver.py) so Exact Match can fire.
_SHORT_INSTRUCTIONS = (
    "Answer with ONLY the exact answer span copied from the passages — a name, "
    "date, number, place, or short noun phrase, as few words as possible. Do not "
    "write a full sentence, do not explain, do not add punctuation or lists."
)


def build_user_prompt(
    question: str, passages: Sequence[str], short: bool = False
) -> str:
    """Paper Appendix B.1 user prompt: question + numbered passages + rules.

    `short=True` swaps in the extractive short-answer instructions for
    short-answer datasets (sist), otherwise the default synthesis instructions.
    """
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(passages, start=1))
    return (
        f"Question: {question}\n"
        f"Passages:\n{numbered}\n"
        f"Instructions: {_SHORT_INSTRUCTIONS if short else _INSTRUCTIONS}"
    )


class LocalGenerator:
    def __init__(self, model_name: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .device import resolve_device

        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # float32: this GPU's fp16/bf16 cached-decode kernel faults (SIGFPE);
        # fp32 uses a different kernel that runs cached decoding correctly.
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            .to(self.device)
            .eval()
        )
        # Greedy stop tokens (Llama-3 ends turns with <|eot_id|> as well as eos).
        self._stop_ids = {self.tokenizer.eos_token_id}
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot >= 0:
            self._stop_ids.add(eot)

    def count_tokens(self, text: str) -> int:
        """Token length under the generator's tokenizer (for the budget)."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chat(
        self, user: str, system: str | None = None, max_new_tokens: int = 256
    ) -> str:
        """Greedy decode (paper T=0) with a manual KV-cache loop.

        Uses plain `forward(..., past_key_values=..., use_cache=True)` steps
        rather than `model.generate()`, whose cached-decode path faults on this
        GPU. With fp32 weights the cached decode runs correctly and efficiently.
        """
        import torch

        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generated: list[int] = []
        with torch.no_grad():
            output = self.model(input_ids=ids, use_cache=True)
            past = output.past_key_values
            next_id = int(output.logits[:, -1, :].argmax(-1))
            for _ in range(max_new_tokens):
                if next_id in self._stop_ids:
                    break
                generated.append(next_id)
                output = self.model(
                    input_ids=torch.tensor([[next_id]], device=self.device),
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values
                next_id = int(output.logits[:, -1, :].argmax(-1))
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate(
        self,
        question: str,
        passages: Sequence[str],
        max_new_tokens: int,
        short: bool = False,
    ) -> str:
        """Answer a question from selected passages (paper B.1 prompt, greedy).

        `short=True` uses the extractive short-answer instructions (sist).
        """
        return self.chat(
            build_user_prompt(question, passages, short=short),
            max_new_tokens=max_new_tokens,
        )
