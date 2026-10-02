"""Check cloned model/RAG assets, optionally load the complete inference pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REQUIRED_WEIGHTS = (
    "yolo_weights.pt",
    "unet_weights.pt",
    "densenet169_weights.pth",
    "best_multimodal_model.pth",
)
REQUIRED_RAG_FILES = (
    "embedding_config.json",
    "child_embeddings.npy",
    "child_chunks_metadata.jsonl",
    "parent_chunks_lookup.json",
)


def check_assets(backend_dir: Path) -> list[str]:
    errors = []
    for relative_path in [
        *(Path("ai_core/weights") / name for name in REQUIRED_WEIGHTS),
        *(Path("rag/xai") / name for name in REQUIRED_RAG_FILES),
    ]:
        path = backend_dir / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty file: {relative_path.as_posix()}")
            continue
        with path.open("rb") as file:
            header = file.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            errors.append(f"Git LFS pointer instead of model: {relative_path.as_posix()}; run git lfs pull origin")
        elif path.suffix in {".pt", ".pth"} and path.stat().st_size < 1024:
            errors.append(f"Model file is too small: {relative_path.as_posix()}")
        elif path.suffix == ".npy" and not header.startswith(b"\x93NUMPY"):
            errors.append(f"Invalid NumPy embedding file: {relative_path.as_posix()}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--load-models", action="store_true", help="Load all models; requires ML dependencies and sufficient RAM")
    args = parser.parse_args()
    backend_dir = args.backend_dir.resolve()
    errors = check_assets(backend_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("OK: all four model files and four RAG assets are present (no LFS pointers).")
    if args.load_models:
        sys.path.insert(0, str(backend_dir))
        import torch
        from ai_core.pipeline import TumorAnalysisPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = TumorAnalysisPipeline(str(backend_dir / "ai_core/weights"), device=device)
        if pipeline.multimodal_model is None:
            print("ERROR: multimodal model was not loaded.")
            return 1
        print(f"OK: full MRI and multimodal pipeline loaded on {device}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
