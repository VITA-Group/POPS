import io
import json
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def gpu_state() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    return subprocess.check_output(cmd, text=True).strip()


def load_image(value) -> Image.Image:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(value)}")


def format_question(question: str, options) -> str:
    if not options:
        return question
    if isinstance(options, dict):
        lines = [question]
        for key in ["A", "B", "C", "D"]:
            if key in options:
                lines.append(f"{key}. {options[key]}")
        return "\n".join(lines)
    return question


def format_answer(answer, options) -> str:
    if isinstance(answer, str) and isinstance(options, dict) and answer in options:
        return f"{answer}. {options[answer]}"
    return str(answer)


def collect_samples(root: Path, limit: int = 4) -> list:
    parquet_files = sorted((root / "data" / "MLLMU-Bench" / "forget_10").glob("*.parquet"))
    rows = []
    for parquet_file in parquet_files:
        rows.extend(pq.read_table(parquet_file).to_pylist())
    samples = []
    for row in rows:
        image = load_image(row["image"])
        task = row.get("Classification_Task") or {}
        for qa in task.get("Image_Textual_Questions", []):
            q = qa.get("Question") or qa.get("question")
            options = qa.get("Options") or qa.get("options")
            ans = qa.get("Correct_Answer") or qa.get("answer")
            if q and ans:
                samples.append(
                    {
                        "image": image,
                        "text": "USER: <image>\n"
                        + format_question(q, options)
                        + "\nASSISTANT: "
                        + format_answer(ans, options),
                    }
                )
            if len(samples) >= limit:
                return samples
    return samples


def main() -> None:
    root = Path.home() / "POPS_Code"
    samples = collect_samples(root, limit=4)
    if len(samples) < 4:
        raise RuntimeError(f"Expected 4 samples, found {len(samples)}")

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
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)

    texts = [sample["text"] for sample in samples]
    images = [sample["image"] for sample in samples]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels

    warmup_steps = 2
    timed_steps = 20
    step_times = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(warmup_steps + timed_steps):
        t0 = time.perf_counter()
        out = model(**inputs)
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        if step >= warmup_steps:
            step_times.append(time.perf_counter() - t0)

    result = {
        "model_id": MODEL_ID,
        "dataset": "MLLMU-Bench forget_10 Image_Textual_Questions",
        "batch_size": len(samples),
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "avg_step_seconds": round(sum(step_times) / len(step_times), 4),
        "steps_per_hour": round(3600 / (sum(step_times) / len(step_times)), 2),
        "peak_cuda_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gpu_after": gpu_state(),
        "last_loss": float(loss.detach().cpu()),
        "first_prompt_chars": len(texts[0]),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
