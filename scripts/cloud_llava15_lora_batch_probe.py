import gc
import json
import subprocess
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def gpu_state(label: str) -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def run_step(model, processor, image, batch_size: int) -> dict:
    prompt = (
        "USER: <image>\n"
        "Answer the visual question with the correct option. "
        "This is a memory probe for the LLaVA-1.5 training path.\n"
        "Question: What is shown in this image?\n"
        "Options: A. A diagram B. A landscape C. A portrait D. A vehicle\n"
        "ASSISTANT: A. A diagram"
    )
    texts = [prompt] * batch_size
    images = [image] * batch_size

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = gpu_state(f"before_bs{batch_size}")
    t0 = time.perf_counter()
    try:
        out = model(**inputs)
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        ok = True
        err = None
    except RuntimeError as exc:
        ok = False
        loss = None
        err = str(exc).splitlines()[0][:240]
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    seconds = round(time.perf_counter() - t0, 2)
    after = gpu_state(f"after_bs{batch_size}")
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    del inputs, labels, optim
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "ok": ok,
        "error": err,
        "seconds": seconds,
        "loss": None if loss is None else float(loss.detach().cpu()),
        "peak_cuda_allocated_gb": peak,
        "gpu_before": before,
        "gpu_after": after,
    }


def main() -> None:
    root = Path.home() / "POPS_Code"
    image = Image.open(root / "asset" / "demo.jpg").convert("RGB")

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.train()
    results = {
        "model_id": MODEL_ID,
        "load_seconds": round(time.perf_counter() - t0, 2),
        "base_gpu": gpu_state("after_load"),
        "probes": [],
    }
    for batch_size in [1, 2, 4]:
        results["probes"].append(run_step(model, processor, image, batch_size))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
