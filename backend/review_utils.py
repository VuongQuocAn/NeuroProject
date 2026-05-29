import os
from typing import Any

import models


CLASSIFICATION_REVIEW_THRESHOLD = float(os.getenv("CLASSIFICATION_REVIEW_THRESHOLD", "0.95"))


LABEL_MAP = {
    "class_0": "Glioma",
    "class_1": "Meningioma",
    "class_2": "Pituitary tumor",
}


def display_label(label: str | None) -> str | None:
    if not label:
        return None
    return LABEL_MAP.get(label, label)


def latest_classification_review(db: Any, image_id: int | None) -> models.ClassificationReview | None:
    if not image_id:
        return None
    return (
        db.query(models.ClassificationReview)
        .filter(models.ClassificationReview.image_id == image_id)
        .order_by(models.ClassificationReview.created_at.desc())
        .first()
    )


def classification_review_state(
    db: Any,
    image_id: int | None,
    ai_label: str | None,
    ai_confidence: float | None,
) -> dict:
    review = latest_classification_review(db, image_id)
    ai_display = display_label(ai_label)
    expert_display = display_label(review.expert_tumor_label) if review else None
    final_label = expert_display or ai_display

    if review:
        status = review.review_action if review.review_action in {"confirmed", "corrected"} else "confirmed"
        return {
            "ai_tumor_label": ai_display,
            "ai_confidence": ai_confidence,
            "final_tumor_label": final_label,
            "expert_tumor_label": expert_display,
            "expert_comment": review.expert_comment,
            "review_required": False,
            "review_status": status,
            "review_action": review.review_action,
            "reviewed_at": review.created_at,
            "review": review,
        }

    if not ai_display:
        status = "not_available"
        required = False
    elif ai_confidence is not None and ai_confidence < CLASSIFICATION_REVIEW_THRESHOLD:
        status = "needs_review"
        required = True
    else:
        status = "not_required"
        required = False

    return {
        "ai_tumor_label": ai_display,
        "ai_confidence": ai_confidence,
        "final_tumor_label": final_label,
        "expert_tumor_label": None,
        "expert_comment": None,
        "review_required": required,
        "review_status": status,
        "review_action": None,
        "reviewed_at": None,
        "review": None,
    }
