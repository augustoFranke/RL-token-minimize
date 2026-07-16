import torch

from .tools import TOOL_SCHEMAS


class HFPolicy:
    """Wraps a HF causal LM as an episode policy. Records (prompt_ids, gen_ids)
    per turn so an RL trainer can recompute log-probs over generated tokens."""

    def __init__(self, model, tokenizer, thinking=False, max_new_tokens=1024,
                 do_sample=False, temperature=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.segments = []

    def token_counter(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @torch.no_grad()
    def __call__(self, messages: list[dict]) -> str:
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
            return_tensors="pt",
            return_dict=False,
        ).to(self.model.device)
        out = self.model.generate(
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0, prompt_ids.shape[1]:]
        self.segments.append((prompt_ids[0].cpu(), gen_ids.cpu()))
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)
