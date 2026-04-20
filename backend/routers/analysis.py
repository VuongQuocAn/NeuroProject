import base64
import os
import textwrap
from io import BytesIO
from typing import List

import cv2
import numpy as np
import pydicom
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps
from sqlalchemy.orm import Session

import models
import schemas
from database import get_db
from utils import get_current_user, minio_client

router = APIRouter(tags=["Analysis & XAI"])
BUCKET_NAME = "medical-data"
ImageFile.LOAD_TRUNCATED_IMAGES = True
LABEL_MAP = {
    "class_0": "Glioma",
    "class_1": "Meningioma",
    "class_2": "Pituitary tumor",
}

NAVY = (18, 42, 66)
TEAL = (38, 120, 133)
RED = (177, 58, 58)
ORANGE = (185, 117, 45)
LIGHT_GRAY = (242, 245, 247)
BORDER = (220, 226, 230)
TEXT = (33, 37, 41)
TEXT_MUTED = (108, 117, 125)


def _get_presigned_url(file_path: str | None) -> str | None:
    if not file_path:
        return None
    try:
        object_name = file_path.split("/", 2)[-1]
        return minio_client.presigned_get_object(
            bucket_name=BUCKET_NAME,
            object_name=object_name,
        )
    except Exception:
        return None


def _local_image_to_data_url(file_path: str | None) -> str | None:
    if not file_path or not os.path.exists(file_path):
        return None

    extension = os.path.splitext(file_path)[1].lower()
    mime_type = "image/png"
    if extension in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"

    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def _image_array_to_data_url(image_bgr: np.ndarray | None) -> str | None:
    if image_bgr is None or image_bgr.size == 0:
        return None

    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        return None

    encoded_text = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded_text}"


def _parse_minio_path(file_path: str) -> tuple[str, str]:
    normalized = file_path.lstrip("/")
    parts = normalized.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Duong dan MinIO khong hop le: {file_path}")
    return parts[0], parts[1]


def _load_image_from_minio(file_path: str | None) -> np.ndarray | None:
    if not file_path:
        return None

    bucket_name, object_name = _parse_minio_path(file_path)
    response = minio_client.get_object(bucket_name, object_name)
    try:
        file_bytes = response.read()
    finally:
        response.close()
        response.release_conn()

    return _decode_image_bytes(file_bytes)


def _decode_image_bytes(file_bytes: bytes) -> np.ndarray | None:
    dicom_image = _try_load_dicom(file_bytes)
    if dicom_image is not None:
        return dicom_image

    array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    try:
        with Image.open(BytesIO(file_bytes)) as pil_image:
            rgb_image = pil_image.convert("RGB")
            image_array = np.array(rgb_image, dtype=np.uint8)
            return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def _try_load_dicom(file_bytes: bytes) -> np.ndarray | None:
    try:
        try:
            dicom = pydicom.dcmread(BytesIO(file_bytes))
        except Exception:
            dicom = pydicom.dcmread(BytesIO(file_bytes), force=True)

        if not hasattr(dicom, "PixelData"):
            return None

        pixel_array = dicom.pixel_array.astype(np.float32)
        if pixel_array.ndim == 3:
            pixel_array = pixel_array[0]

        pixel_array -= pixel_array.min()
        max_value = pixel_array.max()
        if max_value > 0:
            pixel_array /= max_value

        image_u8 = (pixel_array * 255).astype(np.uint8)
        return cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    except Exception:
        return None


def _build_segmentation_overlays(
    original_image_bgr: np.ndarray | None,
    bbox: list[int] | None,
    seg_mask_path: str | None,
) -> tuple[str | None, str | None]:
    if original_image_bgr is None or bbox is None or not seg_mask_path or not os.path.exists(seg_mask_path):
        return None, None

    mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None

    x_min, y_min, x_max, y_max = [int(value) for value in bbox]
    height, width = original_image_bgr.shape[:2]
    x_min = max(0, min(x_min, width - 1))
    y_min = max(0, min(y_min, height - 1))
    x_max = max(x_min + 1, min(x_max, width))
    y_max = max(y_min + 1, min(y_max, height))

    roi_width = x_max - x_min
    roi_height = y_max - y_min
    if roi_width <= 0 or roi_height <= 0:
        return None, None

    if mask.shape[1] != roi_width or mask.shape[0] != roi_height:
        mask = cv2.resize(mask, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)

    binary_mask = ((mask > 127).astype(np.uint8)) * 255

    mask_overlay = original_image_bgr.copy()
    roi_mask_overlay = mask_overlay[y_min:y_max, x_min:x_max]
    green_fill = np.zeros_like(roi_mask_overlay)
    green_fill[:, :] = (0, 255, 0)
    blended = cv2.addWeighted(roi_mask_overlay, 0.65, green_fill, 0.35, 0)
    roi_mask_overlay[binary_mask > 0] = blended[binary_mask > 0]

    contour_overlay = original_image_bgr.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shifted_contours = [contour + np.array([[[x_min, y_min]]], dtype=np.int32) for contour in contours]
    cv2.drawContours(contour_overlay, shifted_contours, -1, (0, 255, 255), 2)

    return _image_array_to_data_url(mask_overlay), _image_array_to_data_url(contour_overlay)


def _display_tumor_label(label: str | None, no_tumor_detected: bool = False) -> str | None:
    if no_tumor_detected:
        return None
    if not label:
        return None
    return LABEL_MAP.get(label, label)


def _format_percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if os.name == "nt":
        if bold:
            candidates.extend(
                [
                    r"C:\Windows\Fonts\arialbd.ttf",
                    r"C:\Windows\Fonts\segoeuib.ttf",
                ]
            )
        else:
            candidates.extend(
                [
                    r"C:\Windows\Fonts\arial.ttf",
                    r"C:\Windows\Fonts\segoeui.ttf",
                ]
            )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _fit_report_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, (250, 251, 252))
    fitted = ImageOps.contain(image.convert("RGB"), size)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_spacing: int = 6,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = word if not current else f"{current} {word}"
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_spacing
    return y


def _draw_image_card(
    page: Image.Image,
    title: str,
    image: Image.Image | None,
    box: tuple[int, int, int, int],
    title_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    small_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
):
    draw = ImageDraw.Draw(page)
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=18, fill=(255, 255, 255), outline=BORDER, width=2)
    draw.text((x1 + 18, y1 + 14), title, font=title_font, fill=TEXT)

    image_area = (x1 + 18, y1 + 52, x2 - 18, y2 - 56)
    draw.rounded_rectangle(image_area, radius=12, fill=LIGHT_GRAY, outline=BORDER, width=1)
    if image is not None:
        fitted = _fit_report_image(image, (image_area[2] - image_area[0], image_area[3] - image_area[1]))
        page.paste(fitted, (image_area[0], image_area[1]))
    else:
        placeholder = "Image not available"
        bbox = draw.textbbox((0, 0), placeholder, font=small_font)
        text_x = image_area[0] + ((image_area[2] - image_area[0]) - (bbox[2] - bbox[0])) // 2
        text_y = image_area[1] + ((image_area[3] - image_area[1]) - (bbox[3] - bbox[1])) // 2
        draw.text((text_x, text_y), placeholder, font=small_font, fill=TEXT_MUTED)


def _bgr_path_to_pil(path: str | None) -> Image.Image | None:
    if not path or not os.path.exists(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _load_image_for_report(minio_file_path: str | None = None, local_path: str | None = None) -> Image.Image | None:
    if local_path:
        local_image = _bgr_path_to_pil(local_path)
        if local_image is not None:
            return local_image

    if minio_file_path:
        image_bgr = _load_image_from_minio(minio_file_path)
        if image_bgr is not None:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

    return None


def _build_summary_text(no_tumor_detected: bool, tumor_label: str | None, bbox_confidence: float | None) -> str:
    if no_tumor_detected:
        return (
            "AI did not detect a convincing intracranial lesion on this MRI study. "
            "No tumor type was assigned by the classification stage."
        )
    confidence_text = _format_percent(bbox_confidence)
    if tumor_label:
        return (
            f"AI detected an intracranial lesion with detection confidence {confidence_text}. "
            f"Predicted class is {tumor_label}."
        )
    return "AI detected a suspicious intracranial region. Final interpretation should be made by a physician."


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _build_simple_pdf(lines: list[str]) -> bytes:
    y = 780
    content_lines = ["BT", "/F1 12 Tf", "50 780 Td"]
    first_line = True
    for line in lines:
        safe_line = _escape_pdf_text(line)
        if not first_line:
            content_lines.append(f"0 -18 Td")
        content_lines.append(f"({safe_line}) Tj")
        first_line = False
        y -= 18
        if y < 80:
            break
    content_lines.append("ET")
    content_stream = "\n".join(content_lines).encode("latin-1", errors="replace")

    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj",
        f"4 0 obj << /Length {len(content_stream)} >> stream\n".encode("latin-1") + content_stream + b"\nendstream endobj",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
    ]

    pdf = BytesIO()
    pdf.write(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(pdf.tell())
        pdf.write(obj)
        pdf.write(b"\n")

    xref_position = pdf.tell()
    pdf.write(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    pdf.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.write(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.write(
        (
            f"trailer << /Size {len(offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_position}\n%%EOF"
        ).encode("latin-1")
    )
    return pdf.getvalue()


def _build_professional_report_pdf(
    report_id: str,
    patient_code: str,
    image_id: int,
    status: str,
    processing_date: str,
    tumor_label: str | None,
    classification_confidence: float | None,
    bbox_confidence: float | None,
    bbox: list[int] | None,
    class_probabilities: list[float] | None,
    no_tumor_detected: bool,
    multimodal_available: bool,
    original_image: Image.Image | None,
    bbox_image: Image.Image | None,
    cropped_roi_image: Image.Image | None,
    mask_image: Image.Image | None,
    overlay_image: Image.Image | None,
) -> bytes:
    page_size = (1240, 1754)
    page1 = Image.new("RGB", page_size, "white")
    page2 = Image.new("RGB", page_size, "white")
    draw1 = ImageDraw.Draw(page1)
    draw2 = ImageDraw.Draw(page2)

    font_title = _load_font(40, bold=True)
    font_h1 = _load_font(28, bold=True)
    font_h2 = _load_font(22, bold=True)
    font_body = _load_font(20)
    font_small = _load_font(16)
    font_label = _load_font(18, bold=True)

    draw1.rectangle((0, 0, page_size[0], 140), fill=NAVY)
    draw1.text((60, 38), "NeuroDiagnosis AI", font=font_h1, fill="white")
    draw1.text((60, 78), "MRI Brain Analysis Report", font=font_title, fill="white")
    draw1.text((page_size[0] - 340, 42), f"Report ID: {report_id}", font=font_small, fill=(220, 232, 240))
    draw1.text((page_size[0] - 340, 72), f"Generated: {processing_date}", font=font_small, fill=(220, 232, 240))

    draw1.text((60, 175), "Patient and Study Information", font=font_h2, fill=TEXT)
    info_box = (60, 215, page_size[0] - 60, 425)
    draw1.rounded_rectangle(info_box, radius=18, fill=LIGHT_GRAY, outline=BORDER, width=2)

    info_rows = [
        ("Patient ID", patient_code),
        ("Image ID / Study ID", str(image_id)),
        ("Modality", "MRI Brain"),
        ("Status", status.upper()),
        ("Processing date", processing_date),
        ("Referring physician", "Not provided"),
        ("Institution / Department", "NeuroDiagnosis AI Lab"),
    ]

    left_x = 90
    right_x = 650
    row_y = 250
    for idx, (label, value) in enumerate(info_rows):
        column_x = left_x if idx < 4 else right_x
        offset_y = row_y + (idx % 4) * 40
        draw1.text((column_x, offset_y), label, font=font_small, fill=TEXT_MUTED)
        draw1.text((column_x, offset_y + 18), value, font=font_label, fill=TEXT)

    draw1.text((60, 465), "AI Result Summary", font=font_h2, fill=TEXT)
    summary_box = (60, 505, page_size[0] - 60, 705)
    draw1.rounded_rectangle(summary_box, radius=18, fill=(247, 250, 251), outline=BORDER, width=2)

    predicted_display = tumor_label or "No lesion detected"
    detection_display = _format_percent(bbox_confidence)
    classification_display = _format_percent(classification_confidence)
    summary_text = _build_summary_text(no_tumor_detected, tumor_label, bbox_confidence)

    draw1.text((90, 540), "Predicted tumor type", font=font_small, fill=TEXT_MUTED)
    draw1.text((90, 565), predicted_display, font=font_h1, fill=RED if not no_tumor_detected else TEXT)
    draw1.text((520, 540), "Classification confidence", font=font_small, fill=TEXT_MUTED)
    draw1.text((520, 565), classification_display, font=font_h1, fill=TEXT)
    draw1.text((860, 540), "Detection confidence", font=font_small, fill=TEXT_MUTED)
    draw1.text((860, 565), detection_display, font=font_h1, fill=TEXT)
    _draw_wrapped_text(draw1, summary_text, (90, 625), font_body, TEXT, max_width=1040)

    draw1.text((60, 745), "Imaging Review", font=font_h2, fill=TEXT)
    _draw_image_card(page1, "Original MRI", original_image, (60, 785, 585, 1225), font_label, font_small)
    _draw_image_card(page1, "Detected Bounding Box", bbox_image, (655, 785, 1180, 1225), font_label, font_small)
    _draw_image_card(page1, "Cropped ROI", cropped_roi_image, (60, 1265, 585, 1705), font_label, font_small)
    _draw_image_card(page1, "Segmentation Mask", mask_image, (655, 1265, 1180, 1705), font_label, font_small)

    draw2.rectangle((0, 0, page_size[0], 110), fill=NAVY)
    draw2.text((60, 38), "NeuroDiagnosis AI - Continued Report", font=font_h1, fill="white")
    draw2.text((page_size[0] - 260, 44), f"Report ID: {report_id}", font=font_small, fill=(220, 232, 240))

    draw2.text((60, 150), "Overlay Review", font=font_h2, fill=TEXT)
    _draw_image_card(page2, "Overlay / Masked ROI", overlay_image, (60, 190, 1180, 760), font_label, font_small)

    draw2.text((60, 810), "Technical Model Details", font=font_h2, fill=TEXT)
    tech_box = (60, 850, 1180, 1260)
    draw2.rounded_rectangle(tech_box, radius=18, fill=LIGHT_GRAY, outline=BORDER, width=2)
    tech_rows = [
        ("Bounding box coordinates", str(bbox) if bbox else "N/A"),
        ("Bounding box confidence", detection_display),
        ("Class probabilities", ", ".join(f"{value * 100:.2f}%" for value in (class_probabilities or [])) or "N/A"),
        ("Predicted label", predicted_display),
        ("Segmentation available", "Yes" if not no_tumor_detected and mask_image is not None else "No"),
        ("Multimodal fusion status", "Available" if multimodal_available else "Not available"),
    ]
    y = 885
    for label, value in tech_rows:
        draw2.text((90, y), label, font=font_small, fill=TEXT_MUTED)
        _draw_wrapped_text(draw2, value, (440, y), font_small, TEXT, max_width=680, line_spacing=4)
        draw2.line((90, y + 42, 1150, y + 42), fill=BORDER, width=1)
        y += 60

    draw2.text((60, 1320), "Disclaimer", font=font_h2, fill=TEXT)
    disclaimer_box = (60, 1360, 1180, 1630)
    draw2.rounded_rectangle(disclaimer_box, radius=18, fill=(252, 248, 247), outline=(230, 214, 210), width=2)
    disclaimer_lines = [
        "This AI-generated report is intended for decision support only and must not replace clinical judgment or radiologist interpretation.",
        "Final diagnosis should be made by a qualified physician.",
    ]
    current_y = 1405
    for line in disclaimer_lines:
        current_y = _draw_wrapped_text(draw2, line, (90, current_y), font_body, TEXT, max_width=1040, line_spacing=8)
        current_y += 10

    buffer = BytesIO()
    page1.save(buffer, format="PDF", save_all=True, append_images=[page2], resolution=150)
    return buffer.getvalue()


def _get_latest_mri_task(db: Session, image_id: int) -> models.InferenceTask | None:
    return (
        db.query(models.InferenceTask)
        .filter(
            models.InferenceTask.task_type == "mri_pipeline",
            models.InferenceTask.target_id == image_id,
        )
        .order_by(models.InferenceTask.created_at.desc())
        .first()
    )


@router.get("/records/analysis/{patient_id}", response_model=List[schemas.AnalysisResultResponse])
def get_patient_analysis(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Khong tim thay benh nhan")

    results = db.query(models.AnalysisResult).filter(models.AnalysisResult.patient_id == patient_id).all()
    if not results:
        raise HTTPException(status_code=404, detail="Chua co ket qua phan tich cho benh nhan nay.")

    return results


@router.get("/records/analysis/image/{image_id}", response_model=schemas.ImageAIResultResponse)
def get_image_analysis_detail(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Khong tim thay anh MRI")

    latest_task = _get_latest_mri_task(db, image_id)
    analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_id).first()

    status = "done" if analysis else "ready"
    error_message = None
    result_payload = {}
    task_id = None
    created_at = analysis.created_at if analysis else None
    updated_at = latest_task.updated_at if latest_task else None

    if latest_task:
        task_id = latest_task.id
        status = latest_task.status
        error_message = latest_task.error_message
        result_payload = latest_task.result or {}
        created_at = latest_task.created_at
        updated_at = latest_task.updated_at

    bbox = result_payload.get("bbox")
    seg_mask_path = result_payload.get("seg_mask_path")
    no_tumor_detected = bool(result_payload.get("no_tumor_detected"))
    mask_overlay_data_url = _local_image_to_data_url(result_payload.get("mask_overlay_path"))
    contour_overlay_data_url = _local_image_to_data_url(result_payload.get("contour_overlay_path"))

    if mask_overlay_data_url is None or contour_overlay_data_url is None:
        original_image_bgr = _load_image_from_minio(image.file_path)
        fallback_mask_overlay_data_url, fallback_contour_overlay_data_url = _build_segmentation_overlays(
            original_image_bgr=original_image_bgr,
            bbox=bbox,
            seg_mask_path=seg_mask_path,
        )
        if mask_overlay_data_url is None:
            mask_overlay_data_url = fallback_mask_overlay_data_url
        if contour_overlay_data_url is None:
            contour_overlay_data_url = fallback_contour_overlay_data_url

    return schemas.ImageAIResultResponse(
        image_id=image_id,
        patient_id=image.patient_id,
        task_id=task_id,
        status=status,
        no_tumor_detected=no_tumor_detected,
        error_message=error_message,
        bbox=bbox,
        bbox_confidence=result_payload.get("bbox_confidence"),
        tumor_label=_display_tumor_label(
            result_payload.get("tumor_label") or (analysis.tumor_label if analysis else None),
            no_tumor_detected=no_tumor_detected,
        ),
        classification_confidence=result_payload.get("classification_confidence")
        if result_payload.get("classification_confidence") is not None
        else (analysis.classification_confidence if analysis else None),
        class_probabilities=result_payload.get("class_probabilities"),
        bbox_overlay_data_url=_local_image_to_data_url(result_payload.get("bbox_image_path")),
        mask_data_url=_local_image_to_data_url(result_payload.get("seg_mask_path")),
        mask_overlay_data_url=mask_overlay_data_url,
        contour_overlay_data_url=contour_overlay_data_url,
        created_at=created_at,
        updated_at=updated_at,
    )


@router.get("/records/analysis/image/{image_id}/report")
def download_image_report(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Khong tim thay anh MRI")

    latest_task = _get_latest_mri_task(db, image_id)
    analysis = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_id).first()
    if not latest_task and not analysis:
        raise HTTPException(status_code=404, detail="Chua co ket qua de xuat bao cao")

    result_payload = latest_task.result if latest_task and latest_task.result else {}
    no_tumor_detected = bool(result_payload.get("no_tumor_detected"))
    tumor_label = _display_tumor_label(
        result_payload.get("tumor_label") or (analysis.tumor_label if analysis else None),
        no_tumor_detected=no_tumor_detected,
    )
    classification_confidence = result_payload.get("classification_confidence")
    if classification_confidence is None and analysis:
        classification_confidence = analysis.classification_confidence
    patient = db.query(models.Patient).filter(models.Patient.id == image.patient_id).first()
    patient_code = patient.patient_external_id if patient and patient.patient_external_id else str(image.patient_id)
    report_id = f"RPT-{patient_code}-{image_id}"
    processing_date = (
        latest_task.updated_at.strftime("%Y-%m-%d %H:%M") if latest_task and latest_task.updated_at else "N/A"
    )
    pdf_bytes = _build_professional_report_pdf(
        report_id=report_id,
        patient_code=patient_code,
        image_id=image_id,
        status=(latest_task.status if latest_task else "done"),
        processing_date=processing_date,
        tumor_label=tumor_label,
        classification_confidence=classification_confidence,
        bbox_confidence=result_payload.get("bbox_confidence"),
        bbox=result_payload.get("bbox"),
        class_probabilities=result_payload.get("class_probabilities"),
        no_tumor_detected=no_tumor_detected,
        multimodal_available=bool(analysis and analysis.risk_score is not None),
        original_image=_load_image_for_report(minio_file_path=image.file_path, local_path=result_payload.get("original_image_path")),
        bbox_image=_bgr_path_to_pil(result_payload.get("bbox_image_path")),
        cropped_roi_image=_bgr_path_to_pil(result_payload.get("cropped_roi_path")),
        mask_image=_bgr_path_to_pil(result_payload.get("seg_mask_path")),
        overlay_image=_bgr_path_to_pil(result_payload.get("mask_overlay_path"))
        or _bgr_path_to_pil(result_payload.get("masked_roi_path")),
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="mri_report_{image_id}.pdf"'},
    )


@router.get("/records/analysis/{image_id}/xai-overlay", response_model=schemas.XAIOverlayResponse)
def get_xai_overlay(
    image_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    result = db.query(models.AnalysisResult).filter(models.AnalysisResult.image_id == image_id).first()
    if not result:
        raise HTTPException(status_code=404, detail=f"Chua co ket qua XAI cho image_id={image_id}.")

    return schemas.XAIOverlayResponse(
        image_id=image_id,
        gradcam_url=_get_presigned_url(result.gradcam_path),
        mask_url=_get_presigned_url(result.mask_path),
    )


@router.get("/analytics/survival/{patient_id}", response_model=schemas.SurvivalCurveResponse)
def get_survival_curve(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Khong tim thay benh nhan")

    result = (
        db.query(models.AnalysisResult)
        .filter(
            models.AnalysisResult.patient_id == patient_id,
            models.AnalysisResult.survival_curve_data.isnot(None),
        )
        .order_by(models.AnalysisResult.created_at.desc())
        .first()
    )

    if not result or not result.survival_curve_data:
        raise HTTPException(status_code=404, detail="Chua co du lieu duong cong song con.")

    curve_points = [
        schemas.SurvivalPoint(
            time=point["time"],
            survival_probability=point["survival_probability"],
        )
        for point in result.survival_curve_data
    ]

    return schemas.SurvivalCurveResponse(
        patient_id=patient_id,
        risk_group=result.risk_group,
        curve=curve_points,
    )
