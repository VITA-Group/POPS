import json
import subprocess
import time
from pathlib import Path

import torch
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
    image_path = root / "asset" / "demo.jpg"
    image = Image.open(image_path).convert("RGB")

    timings = {}
    states = [gpu_state("start")]

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    timings["processor_seconds"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    timings["model_load_seconds"] = round(time.perf_counter() - t0, 2)
    states.append(gpu_state("after_model_load"))

    prompt = "USER: <image>\nWhat is shown in this image?\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    timings["generate_seconds"] = round(time.perf_counter() - t0, 2)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    states.append(gpu_state("after_generate"))

    decoded = processor.decode(output[0], skip_special_tokens=True)
    result = {
        "model_id": MODEL_ID,
        "image": str(image_path),
        "timings": timings,
        "peak_cuda_allocated_gb": round(peak_gb, 2),
        "gpu_states": states,
        "decoded": decoded,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
