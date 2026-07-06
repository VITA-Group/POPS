import importlib
import sys


def main() -> None:
    print("python", sys.version.replace("\n", " "))
    for name in ["torch", "transformers", "peft", "datasets", "PIL", "pyarrow", "yaml", "accelerate"]:
        module_name = "PIL" if name == "PIL" else name
        try:
            module = importlib.import_module(module_name)
            print(name, getattr(module, "__version__", "ok"))
        except Exception as exc:
            print(name, "MISSING", type(exc).__name__, str(exc)[:160])


if __name__ == "__main__":
    main()
