import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_and_tokenizer(model_name: str = MODEL_NAME, adapter_path: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # bf16 training on MPS NaN'd adapters in smoke tests; fp32 is the safe default there
    dtype = torch.float32 if device() == "mps" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model.to(device()), tokenizer


def lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
