from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai_core.architectures.densenet_classifier import DenseNetClassifier
from ai_core.architectures.xai_finer_cam import FinerCAMExplainer
from services.gemini_service import GeminiXaiExplanationService
from services.rag_service import get_xai_rag_service


def run(image_id: int, source_name: str | None = None, suffix: str = "rerun") -> dict:
    load_dotenv(REPO_ROOT / ".env", override=True)

    analysis_dir = BACKEND_DIR / "analysis_results" / str(image_id)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Missing analysis directory: {analysis_dir}")

    roi_path = analysis_dir / "step2_roi.png"
    if not roi_path.exists() and source_name:
        roi_path = analysis_dir / source_name
    if not roi_path.exists():
        raise FileNotFoundError(
            f"Missing ROI image. Expected {analysis_dir / 'step2_roi.png'} "
            "or pass --source-name."
        )

    roi_bgr = cv2.imread(str(roi_path))
    if roi_bgr is None:
        raise RuntimeError(f"Cannot read ROI/image: {roi_path}")

    classifier = DenseNetClassifier(device="cpu")
    classifier.load_weights(str(BACKEND_DIR / "ai_core" / "weights" / "densenet169_weights.pth"))

    explainer = FinerCAMExplainer(classifier)
    xai = explainer.generate(roi_bgr=roi_bgr)

    heatmap_path = analysis_dir / f"step9_classification_finer_cam_{suffix}.png"
    meta_path = analysis_dir / f"xai_classification_finer_cam_meta_{suffix}.json"
    explain_path = analysis_dir / f"xai_classification_llm_explanation_{suffix}.txt"
    explain_meta_path = analysis_dir / f"xai_classification_llm_rag_meta_{suffix}.json"

    if not cv2.imwrite(str(heatmap_path), xai.overlay_bgr):
        raise RuntimeError(f"Cannot write heatmap: {heatmap_path}")

    xai_metadata = xai.metadata or {}
    meta_path.write_text(json.dumps(xai_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    tumor_label = xai_metadata.get("target_class_name")
    classification_confidence = xai_metadata.get("target_probability")

    rag = get_xai_rag_service()
    contexts, rag_diagnostics = rag.retrieve_classification_context(
        tumor_label=tumor_label,
        classification_confidence=classification_confidence,
        xai_metadata=xai_metadata,
        top_k=3,
        candidate_k=10,
    )

    explanation = GeminiXaiExplanationService().generate_classification_explanation(
        tumor_label=tumor_label,
        classification_confidence=classification_confidence,
        heatmap_path=str(heatmap_path),
        xai_metadata=xai_metadata,
        contexts=contexts,
        rag_diagnostics=rag_diagnostics,
    )
    explain_path.write_text(explanation, encoding="utf-8")

    rag_meta = {
        "image_id": image_id,
        "source_name": source_name,
        "roi_path": str(roi_path),
        "heatmap_path": str(heatmap_path),
        "xai_metadata_path": str(meta_path),
        "tumor_label": tumor_label,
        "classification_confidence": classification_confidence,
        "rag_diagnostics": rag_diagnostics,
        "rag_contexts": [
            {
                "child_id": context.child_id,
                "parent_id": context.parent_id,
                "score": context.score,
                "source_title": context.source_title,
                "source_url": context.source_url,
                "labels": context.labels,
                "child_preview": context.child_text[:500],
            }
            for context in contexts
        ],
        "explanation_path": str(explain_path),
    }
    explain_meta_path.write_text(json.dumps(rag_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "heatmap_path": str(heatmap_path),
        "xai_metadata_path": str(meta_path),
        "explanation_path": str(explain_path),
        "rag_metadata_path": str(explain_meta_path),
        "tumor_label": tumor_label,
        "classification_confidence": classification_confidence,
        "rag_top_scores": [round(context.score, 4) for context in contexts],
        "explanation_preview": explanation[:700],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun Classification Finer-CAM + RAG + Gemini explanation.")
    parser.add_argument("--image-id", type=int, default=17)
    parser.add_argument("--source-name", default="Meningioma_12_9271.jpg")
    parser.add_argument("--suffix", default="rerun")
    args = parser.parse_args()

    result = run(image_id=args.image_id, source_name=args.source_name, suffix=args.suffix)
    print(json.dumps(result, ensure_ascii=False, indent=2).encode("unicode_escape").decode("ascii"))


if __name__ == "__main__":
    main()
