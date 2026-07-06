from pathlib import Path


def main() -> None:
    root = Path.home() / "POPS_Code"
    moved = 0
    skipped = 0

                                                                      
                                                                            
    paths = sorted(root.rglob("*"), key=lambda path: len(path.parts))
    for src in paths:
        rel = src.relative_to(root).as_posix()
        if "\\" not in rel:
            continue
        dst = root.joinpath(*rel.split("\\"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if src.is_dir() and not any(src.iterdir()):
                src.rmdir()
                moved += 1
                continue
            skipped += 1
            continue
        src.replace(dst)
        moved += 1
    remaining = sum(1 for path in root.rglob("*") if "\\" in path.relative_to(root).as_posix())
    print({"moved": moved, "skipped": skipped, "remaining_flat": remaining})


if __name__ == "__main__":
    main()
