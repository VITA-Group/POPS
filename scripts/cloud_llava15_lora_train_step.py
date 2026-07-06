import json
import subprocess
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def gpu_state(label: str) -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except Exception as exc:
        out = f"ERROR: {type(exc).__name__}: {exc}"
    return {"label": label, "nvidia_smi": out}


def main() -> None:
    root = Path.home() / "POPS_Code"
    image = Image.open(root / "asset" / "demo.jpg").convert("RGB")
    states = [gpu_state("start")]
    timings = {}

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    timings["load_seconds"] = round(time.perf_counter() - t0, 2)
    states.append(gpu_state("after_base_load"))

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, config)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    states.append(gpu_state("after_lora"))

    prompt = "USER: <image>\nWhat is shown in this image?\nASSISTANT: A diagram."
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = model(**inputs)
    loss = out.loss
    loss.backward()
    optim.step()
    optim.zero_grad(set_to_none=True)
    timings["train_step_seconds"] = round(time.perf_counter() - t0, 2)
    states.append(gpu_state("after_train_step"))

    result = {
        "model_id": MODEL_ID,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "targets": ["q_proj", "v_proj"]},
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": round(100 * trainable / total, 4),
        "loss": float(loss.detach().cpu()),
        "peak_cuda_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "timings": timings,
        "gpu_states": states,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
