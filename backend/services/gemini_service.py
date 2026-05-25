from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .rag_service import RetrievedContext


class GeminiXaiExplanationService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    def generate_classification_explanation(
        self,
        tumor_label: str | None,
        classification_confidence: float | None,
        heatmap_path: str | None,
        xai_metadata: dict[str, Any] | None,
        contexts: list[RetrievedContext],
        rag_diagnostics: dict[str, Any],
    ) -> str:
        prompt = self._build_prompt(
            tumor_label=tumor_label,
            classification_confidence=classification_confidence,
            xai_metadata=xai_metadata,
            contexts=contexts,
            rag_diagnostics=rag_diagnostics,
        )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return self._format_for_display(
                self._fallback_text(
                    tumor_label=tumor_label,
                    classification_confidence=classification_confidence,
                    xai_metadata=xai_metadata,
                    contexts=contexts,
                    rag_diagnostics=rag_diagnostics,
                )
            )

        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)

        content: list[Any] = [prompt]
        image_part = self._load_image_part(heatmap_path)
        if image_part:
            content.append(image_part)

        response = model.generate_content(
            content,
            generation_config={
                "temperature": 0.15,
                "top_p": 0.9,
                "max_output_tokens": 2000,
            },
        )

        text = (getattr(response, "text", None) or "").strip()

        ok, _reason = self._validate_explanation(
            text=text,
            tumor_label=tumor_label,
            classification_confidence=classification_confidence,
            xai_metadata=xai_metadata,
        )
        if ok:
            return self._format_for_display(text)

        repair_prompt = self._build_repair_prompt(
            original_prompt=prompt,
            bad_answer=text,
            tumor_label=tumor_label,
            classification_confidence=classification_confidence,
            xai_metadata=xai_metadata,
        )

        repair_response = model.generate_content(
            [repair_prompt],
            generation_config={
                "temperature": 0.05,
                "top_p": 0.8,
                "max_output_tokens": 2000,
            },
        )

        repaired_text = (getattr(repair_response, "text", None) or "").strip()

        ok, _reason = self._validate_explanation(
            text=repaired_text,
            tumor_label=tumor_label,
            classification_confidence=classification_confidence,
            xai_metadata=xai_metadata,
        )
        if ok:
            return self._format_for_display(repaired_text)

        return self._format_for_display(
            self._fallback_text(
                tumor_label=tumor_label,
                classification_confidence=classification_confidence,
                xai_metadata=xai_metadata,
                contexts=contexts,
                rag_diagnostics=rag_diagnostics,
            )
        )

    def _load_image_part(self, heatmap_path: str | None) -> dict[str, Any] | None:
        if not heatmap_path:
            return None

        path = Path(heatmap_path)
        if path.exists():
            data = path.read_bytes()
            suffix = path.suffix.lower()
        else:
            try:
                from utils import minio_client

                normalized = heatmap_path.lstrip("/")
                bucket_name, object_name = normalized.split("/", 1)
                response = minio_client.get_object(bucket_name, object_name)
                try:
                    data = response.read()
                finally:
                    response.close()
                    response.release_conn()
                suffix = Path(object_name).suffix.lower()
            except Exception:
                return None

        mime_type = "image/png"
        if suffix in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"

        return {
            "mime_type": mime_type,
            "data": data,
        }

    @staticmethod
    def _format_for_display(text: str) -> str:
        clean = (text or "").strip()
        clean = re.sub(r"^##\s+", "", clean, flags=re.MULTILINE)
        clean = re.sub(
            r"\n\s*4\.\s+L[^\n]*\n(?:- .*(?:\n|$))*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        return clean.strip()

    @staticmethod
    def _safe_float(value: Any, default: float | None = None) -> float | None:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @classmethod
    def _pct(cls, value: Any, ndigits: int = 2) -> str:
        number = cls._safe_float(value)
        if number is None:
            return "không rõ"
        return f"{number * 100:.{ndigits}f}%".replace(".", ",")

    @classmethod
    def _num(cls, value: Any, ndigits: int = 2) -> str:
        number = cls._safe_float(value)
        if number is None:
            return "không rõ"
        return f"{number:.{ndigits}f}".replace(".", ",")

    @staticmethod
    def _tumor_vi_name(name: str | None) -> str:
        if not name:
            return "không rõ"

        key = str(name).strip().lower()
        mapping = {
            "glioma": "u thần kinh đệm",
            "meningioma": "u màng não",
            "pituitary": "u tuyến yên",
            "pituitary tumor": "u tuyến yên",
            "u mang nao": "u màng não",
            "u màng não": "u màng não",
            "u than kinh dem": "u thần kinh đệm",
            "u thần kinh đệm": "u thần kinh đệm",
            "u tuyen yen": "u tuyến yên",
            "u tuyến yên": "u tuyến yên",
        }
        return mapping.get(key, str(name))

    @classmethod
    def _tumor_display(cls, name: str | None) -> str:
        if not name:
            return "không rõ"

        raw = str(name).strip()
        key = raw.lower()

        english_to_vi = {
            "glioma": "u thần kinh đệm",
            "meningioma": "u màng não",
            "pituitary": "u tuyến yên",
            "pituitary tumor": "u tuyến yên",
        }

        if key in english_to_vi:
            return f"{english_to_vi[key]} ({raw})"

        return raw

    @staticmethod
    def _localization_text(localization_strength: str | None) -> str:
        if localization_strength == "strongly_focal":
            return "tập trung khá rõ"
        if localization_strength == "moderately_focal":
            return "tập trung ở mức vừa phải"
        if localization_strength == "diffuse":
            return "còn lan rộng, chưa tập trung mạnh"
        return "không rõ"

    @staticmethod
    def _shape_text(shape: Any) -> str:
        if isinstance(shape, list) and len(shape) == 2:
            return f"{shape[0]} x {shape[1]}"
        if isinstance(shape, tuple) and len(shape) == 2:
            return f"{shape[0]} x {shape[1]}"
        return "không rõ"

    @staticmethod
    def _relative_zone_text(x_norm: float, y_norm: float) -> str:
        if x_norm < 0.33:
            horizontal = "lệch trái"
        elif x_norm <= 0.66:
            horizontal = "gần trung tâm theo chiều ngang"
        else:
            horizontal = "lệch phải"

        if y_norm < 0.33:
            vertical = "phía trên"
        elif y_norm <= 0.66:
            vertical = "vùng giữa theo chiều dọc"
        else:
            vertical = "phía dưới"

        return f"{vertical}, {horizontal}"

    @classmethod
    def _xy_position_text(
        cls,
        xy: Any,
        xy_norm: Any,
        label: str,
    ) -> str:
        """
        Trả về mô tả tọa độ dạng:
        Vùng sáng nhất của heatmap có tọa độ ROI khoảng (x=32, y=48)
        (vị trí tương đối: khoảng 35,6% chiều ngang từ trái sang phải
        và 64,0% chiều dọc từ trên xuống; nằm ở vùng giữa theo chiều dọc, lệch trái).
        """
        if not isinstance(xy, list) or len(xy) != 2:
            return f"{label}: không rõ tọa độ ROI."

        x = cls._safe_float(xy[0])
        y = cls._safe_float(xy[1])
        if x is None or y is None:
            return f"{label}: không rõ tọa độ ROI."

        if isinstance(xy_norm, list) and len(xy_norm) == 2:
            x_norm = cls._safe_float(xy_norm[0])
            y_norm = cls._safe_float(xy_norm[1])
        else:
            x_norm = None
            y_norm = None

        x_text = cls._num(x, 1) if abs(x - round(x)) > 1e-6 else str(int(round(x)))
        y_text = cls._num(y, 1) if abs(y - round(y)) > 1e-6 else str(int(round(y)))

        if x_norm is None or y_norm is None:
            return (
                f"{label} có tọa độ ROI khoảng (x={x_text}, y={y_text}) "
                f"(vị trí tương đối: không rõ)."
            )

        zone_text = cls._relative_zone_text(x_norm, y_norm)

        return (
            f"{label} có tọa độ ROI khoảng (x={x_text}, y={y_text}) "
            f"(vị trí tương đối: khoảng {cls._pct(x_norm, 1)} chiều ngang từ trái sang phải "
            f"và {cls._pct(y_norm, 1)} chiều dọc từ trên xuống; nằm ở {zone_text})."
        )

    @classmethod
    def _bbox_position_text(cls, bbox_xyxy: Any, roi_shape_hw: Any) -> str:
        """
        bbox_xyxy: [x1, y1, x2, y2]
        roi_shape_hw: [H, W]
        """
        if (
            not isinstance(bbox_xyxy, list)
            or len(bbox_xyxy) != 4
            or not isinstance(roi_shape_hw, list)
            or len(roi_shape_hw) != 2
        ):
            return "Vùng nóng nổi bật: không rõ tọa độ."

        x1 = cls._safe_float(bbox_xyxy[0])
        y1 = cls._safe_float(bbox_xyxy[1])
        x2 = cls._safe_float(bbox_xyxy[2])
        y2 = cls._safe_float(bbox_xyxy[3])
        h = cls._safe_float(roi_shape_hw[0])
        w = cls._safe_float(roi_shape_hw[1])

        if None in {x1, y1, x2, y2, h, w} or h == 0 or w == 0:
            return "Vùng nóng nổi bật: không rõ tọa độ."

        x1_text = str(int(round(x1)))
        y1_text = str(int(round(y1)))
        x2_text = str(int(round(x2)))
        y2_text = str(int(round(y2)))

        x1p = x1 / w
        x2p = x2 / w
        y1p = y1 / h
        y2p = y2 / h

        center_xp = (x1p + x2p) / 2
        center_yp = (y1p + y2p) / 2
        zone_text = cls._relative_zone_text(center_xp, center_yp)

        return (
            f"Vùng nóng nổi bật có hộp bao ROI khoảng "
            f"(x1={x1_text}, y1={y1_text}, x2={x2_text}, y2={y2_text}) "
            f"(vị trí tương đối: trải từ {cls._pct(x1p, 1)} đến {cls._pct(x2p, 1)} chiều ngang "
            f"và từ {cls._pct(y1p, 1)} đến {cls._pct(y2p, 1)} chiều dọc ROI; "
            f"tâm vùng này nằm ở {zone_text}). "
            f"Đây là vùng mô hình dựa vào nhiều hơn khi phân loại, không phải ranh giới chính xác của khối u."
        )

    def _extract_prompt_facts(
        self,
        tumor_label: str | None,
        classification_confidence: float | None,
        xai_metadata: dict[str, Any] | None,
    ) -> dict[str, str]:
        meta = xai_metadata or {}
        heatmap_summary = meta.get("heatmap_summary", {}) or {}

        pred_name = tumor_label or meta.get("target_class_name") or "không rõ"
        pred_display = self._tumor_display(pred_name)

        confidence = classification_confidence
        if confidence is None:
            confidence = meta.get("target_probability")

        ref_name = meta.get("reference_class_name") or "không rõ"
        ref_display = self._tumor_display(ref_name)
        ref_prob = meta.get("reference_probability")

        margin = meta.get("target_minus_reference_logit_margin")

        heatmap_scope = meta.get("heatmap_scope")
        if heatmap_scope == "classification_roi_after_detection":
            scope_text = "vùng ROI nghi ngờ có tổn thương sau bước phát hiện, không phải toàn bộ ảnh MRI"
        else:
            scope_text = "không rõ"

        roi_shape_text = self._shape_text(meta.get("roi_shape_hw"))
        heatmap_shape_text = self._shape_text(meta.get("heatmap_shape_hw"))

        localization = heatmap_summary.get("localization_strength")
        localization_text = self._localization_text(localization)

        top10_energy_ratio = heatmap_summary.get("top10_energy_ratio")
        top20_area_ratio = heatmap_summary.get("top20_area_ratio")

        peak_position_text = self._xy_position_text(
            xy=heatmap_summary.get("peak_xy"),
            xy_norm=heatmap_summary.get("peak_xy_normalized"),
            label="Vùng sáng nhất của heatmap",
        )

        center_position_text = self._xy_position_text(
            xy=heatmap_summary.get("energy_center_xy"),
            xy_norm=heatmap_summary.get("energy_center_xy_normalized"),
            label="Tâm chú ý trung bình của heatmap",
        )

        bbox_position_text = self._bbox_position_text(
            bbox_xyxy=heatmap_summary.get("top20_bbox_xyxy"),
            roi_shape_hw=meta.get("roi_shape_hw"),
        )

        required_result_line = (
            f"Mô hình dự đoán: {pred_display}; "
            f"độ tin cậy phân loại: {self._pct(confidence)}."
        )

        required_xai_line = (
            f"So với lớp so sánh {ref_display}, độ tin cậy của lớp so sánh khoảng "
            f"{self._pct(ref_prob, ndigits=4)}, và điểm chênh lệch giữa hai lớp khoảng "
            f"{self._num(margin)}."
        )

        required_heatmap_line = (
            f"Heatmap được tạo trên {scope_text}; ROI có kích thước khoảng "
            f"{roi_shape_text} pixel; heatmap gốc có độ phân giải {heatmap_shape_text}; "
            f"mức độ tập trung của heatmap: {localization_text}."
        )

        required_region_lines = [
            peak_position_text,
            center_position_text,
            bbox_position_text,
        ]

        required_energy_line = (
            f"10% vùng nóng nhất chiếm khoảng {self._pct(top10_energy_ratio)} tổng tín hiệu heatmap; "
            f"nhóm vùng nóng nổi bật chiếm khoảng {self._pct(top20_area_ratio)} diện tích heatmap."
        )

        return {
            "pred_display": pred_display,
            "confidence_pct": self._pct(confidence),
            "required_result_line": required_result_line,
            "required_xai_line": required_xai_line,
            "required_heatmap_line": required_heatmap_line,
            "required_region_line": " ".join(required_region_lines),
            "required_region_bullets": "\n  - ".join(required_region_lines),
            "required_energy_line": required_energy_line,
        }

    def _build_prompt(
        self,
        tumor_label: str | None,
        classification_confidence: float | None,
        xai_metadata: dict[str, Any] | None,
        contexts: list[RetrievedContext],
        rag_diagnostics: dict[str, Any],
    ) -> str:
        facts = self._extract_prompt_facts(
            tumor_label=tumor_label,
            classification_confidence=classification_confidence,
            xai_metadata=xai_metadata,
        )

        context_blocks: list[str] = []
        for idx, context in enumerate(contexts, start=1):
            score = self._safe_float(getattr(context, "score", None), 0.0) or 0.0
            text = context.parent_text or context.child_text or ""
            context_blocks.append(
                f"[Nguồn {idx}] score={score:.4f}\n"
                f"Tiêu đề: {context.source_title or 'unknown'}\n"
                f"URL: {context.source_url or 'unknown'}\n"
                f"Nội dung: {text[:1800]}"
            )

        context_text = "\n\n".join(context_blocks).strip()
        if not context_text:
            context_text = "Không có ngữ cảnh RAG."

        xai_metadata_text = json.dumps(xai_metadata or {}, ensure_ascii=False, indent=2)
        rag_diagnostics_text = json.dumps(rag_diagnostics or {}, ensure_ascii=False, indent=2)

        return f"""
Bạn là trợ lý AI tiếng Việt trong hệ thống hỗ trợ phân tích MRI u não có XAI.

Nhiệm vụ của bạn là giải thích kết quả phân loại u não, heatmap Finer-CAM, thông tin y khoa và rủi ro lâm sàng dựa trên RAG.

Câu trả lời dành cho người dùng cuối, bao gồm bác sĩ lâm sàng, kỹ thuật viên y tế hoặc bệnh nhân.
Vì vậy, phải dễ hiểu nhưng không được nói chung chung. Bắt buộc phải có các thông số quan trọng đã được diễn giải.

# DỮ LIỆU BẮT BUỘC ĐÃ TÍNH SẴN

Dòng kết quả bắt buộc:
- {facts["required_result_line"]}

Dòng so sánh AI bắt buộc:
- {facts["required_xai_line"]}

Dòng heatmap bắt buộc:
- {facts["required_heatmap_line"]}

Dòng vị trí heatmap bắt buộc:
- {facts["required_region_bullets"]}

Dòng năng lượng heatmap bắt buộc:
- {facts["required_energy_line"]}

# DỮ LIỆU GỐC

## Metadata XAI
{xai_metadata_text}

## Thông tin truy xuất RAG
{rag_diagnostics_text}

## Ngữ cảnh y khoa từ RAG
Các nguồn bên dưới là top 3 context sau bước reranking bằng cross-encoder. Ở mục 3, không được chép một nguồn duy nhất; hãy tổng hợp/fusion cả 3 nguồn đã rerank, ưu tiên điểm score cao hơn nhưng vẫn giữ nhất quán với toàn bộ ngữ cảnh.
{context_text}

# FORMAT BẮT BUỘC

Bạn PHẢI trả lời đúng 4 mục sau, đúng thứ tự và đúng tiêu đề:

## 1. Kết quả tóm tắt
## 2. Giải thích AI/XAI và vùng heatmap mô hình dựa vào
## 3. Thông tin y khoa về loại u này
## 4. Lưu ý an toàn

CẤM:
- Không viết lời chào.
- Không viết câu mở đầu kiểu “Dưới đây là...”.
- Không thêm mục thứ 5.
- Không đổi tên tiêu đề.
- Không bỏ mục nào.
- Không dùng bảng.
- Không viết một đoạn văn dài trong mỗi mục; phải tách thành các gạch đầu dòng.
- Không nói chung chung kiểu “độ tin cậy cao” nếu đã có số phần trăm.
- Không nói “heatmap chứng minh mô hình đúng”.
- Không nói “độ tin cậy phân loại là mức độ nguy hiểm”.
- Không nói “vùng sáng chắc chắn là khối u”.
- Không đưa phác đồ điều trị.
- Không thay thế bác sĩ.

BẮT BUỘC:
- Câu trả lời phải bắt đầu ngay bằng: ## 1. Kết quả tóm tắt
- Mỗi mục phải viết bằng gạch đầu dòng.
- Mỗi ý là một gạch đầu dòng riêng.
- Không được dồn nhiều ý trong cùng một đoạn văn dài.
- Mỗi mục phải có từ 2 đến 7 gạch đầu dòng.
- Mỗi gạch đầu dòng chỉ nên dài 1 câu, tối đa 2 câu nếu cần giải thích.
- Trong mục 1, gạch đầu dòng đầu tiên PHẢI chép lại nguyên văn:
  - {facts["required_result_line"]}
- Trong mục 2, PHẢI có đủ các dòng sau, không được bỏ số:
  - {facts["required_xai_line"]}
  - {facts["required_heatmap_line"]}
  - {facts["required_region_bullets"]}
  - {facts["required_energy_line"]}
- Trong mục 2, PHẢI in tọa độ theo dạng có ngoặc giải thích vị trí tương đối, ví dụ:
  tọa độ ROI khoảng (x=..., y=...) (vị trí tương đối: ...).
- Phải có số liệu cụ thể ở mục 1 và mục 2 nếu dữ liệu có cung cấp.
- Thông tin y khoa chỉ được dựa trên RAG.
- Nếu RAG không đủ thông tin, phải nói “chưa đủ thông tin”, không tự bịa.

# CÁCH DIỄN GIẢI

## 1. Về kết quả phân loại
- Độ tin cậy phân loại chỉ cho biết mô hình nghiêng về loại u nào trong bài toán phân loại ảnh.
- Không được diễn giải độ tin cậy là mức độ nguy hiểm của bệnh.
- Nếu có lớp so sánh, hãy nêu mô hình ưu tiên lớp dự đoán hơn lớp so sánh.

## 2. Về heatmap và vùng mô hình dựa vào
- Heatmap là bản đồ nhiệt cho biết vùng ảnh ảnh hưởng nhiều hơn đến quyết định phân loại của mô hình.
- Nếu heatmap được tạo trên ROI, hãy giải thích ROI là vùng ảnh nghi ngờ có tổn thương đã được hệ thống phát hiện.
- Phải nói rõ vùng sáng nhất nằm tại tọa độ nào trong ROI và ghi thêm trong ngoặc vị trí tương đối để người dùng dễ hiểu.
- Phải nói rõ tâm chú ý trung bình nằm tại tọa độ nào trong ROI và ghi thêm trong ngoặc vị trí tương đối.
- Phải nói rõ vùng nóng nổi bật trải từ đâu đến đâu trong ROI nếu có dữ liệu hộp bao.
- Nếu heatmap còn lan rộng, hãy nói rõ khả năng giải thích của XAI chưa mạnh và không nên xem heatmap là ranh giới chính xác của tổn thương.
- Nếu heatmap có độ phân giải thấp hơn ROI, hãy nói heatmap chỉ mang tính định hướng vùng quan trọng tương đối.

## 3. Về RAG
- Chỉ sử dụng thông tin trong ngữ cảnh RAG.
- Phần thông tin y khoa phải là bản tổng hợp/fusion từ top 3 nguồn sau reranking, không được chỉ lấy nguyên một chunk tốt nhất.
- Nếu RAG nói u nhỏ hoặc không triệu chứng có thể theo dõi, có thể nêu ở mức tổng quát.
- Nếu RAG nói u lớn, có triệu chứng hoặc tăng trưởng nhanh cần đánh giá/can thiệp chuyên khoa, có thể nêu ở mức tổng quát.
- Không đưa phác đồ điều trị cụ thể.

# NỘI DUNG BẮT BUỘC TRONG TỪNG MỤC

## 1. Kết quả tóm tắt
- Gạch đầu dòng đầu tiên bắt buộc là:
  - {facts["required_result_line"]}
- Các gạch đầu dòng tiếp theo phải nói rõ đây là kết quả hỗ trợ tham khảo, không phải chẩn đoán cuối cùng.
- Phải nhắc rõ độ tin cậy phân loại không phải mức độ nguy hiểm.

## 2. Giải thích AI/XAI và vùng heatmap mô hình dựa vào
- Mục này phải gộp toàn bộ phần giải thích kỹ thuật thành ngôn ngữ dễ hiểu.
- Bắt buộc có gạch đầu dòng về lớp so sánh, độ tin cậy lớp so sánh và điểm chênh lệch giữa hai lớp.
- Bắt buộc có gạch đầu dòng về phạm vi heatmap, kích thước ROI, độ phân giải heatmap và mức độ tập trung heatmap.
- Bắt buộc có gạch đầu dòng về vùng sáng nhất của heatmap, kèm tọa độ ROI và ngoặc giải thích vị trí tương đối.
- Bắt buộc có gạch đầu dòng về tâm chú ý trung bình, kèm tọa độ ROI và ngoặc giải thích vị trí tương đối.
- Bắt buộc có gạch đầu dòng về vùng nóng nổi bật trải rộng trong ROI như thế nào.
- Bắt buộc có gạch đầu dòng về tỷ lệ tín hiệu trong vùng nóng nhất.
- Phải giải thích vì sao mô hình nghiêng về loại u được dự đoán dựa trên độ tin cậy, điểm chênh lệch và đặc trưng hình ảnh trong ROI.
- Nếu heatmap lan rộng, phải nói rõ XAI chưa phải bằng chứng khu trú mạnh.
- Không khẳng định chắc chắn mô hình đúng.

## 3. Thông tin y khoa về loại u này
- Mục này phải viết bằng gạch đầu dòng.
- Phải tổng hợp/fusion từ top 3 nguồn RAG sau reranking; không được chỉ lấy một chunk tốt nhất.
- Chỉ dùng RAG để mô tả loại u được dự đoán.
- Nêu thông tin MRI, theo dõi hoặc yếu tố liên quan nếu RAG có.
- Không tự thêm kiến thức ngoài RAG.

## 4. Lưu ý an toàn
- Mục này phải viết bằng gạch đầu dòng.
- Nói rõ heatmap không phải bằng chứng giải phẫu bệnh.
- Kết quả AI không thay thế bác sĩ.
- Cần đối chiếu MRI đầy đủ, T1/T2/FLAIR/DWI, ảnh sau tiêm nếu có, triệu chứng lâm sàng và kết luận chuyên khoa.

Bây giờ hãy tạo câu trả lời cuối cùng theo đúng 4 mục đã quy định. Mỗi ý phải là một gạch đầu dòng riêng, không dồn thành đoạn văn dài.
""".strip()

    def _build_repair_prompt(
        self,
        original_prompt: str,
        bad_answer: str,
        tumor_label: str | None,
        classification_confidence: float | None,
        xai_metadata: dict[str, Any] | None,
    ) -> str:
        facts = self._extract_prompt_facts(
            tumor_label=tumor_label,
            classification_confidence=classification_confidence,
            xai_metadata=xai_metadata,
        )

        return f"""
Câu trả lời trước đã sai format hoặc thiếu số liệu bắt buộc.

Hãy viết lại câu trả lời theo đúng yêu cầu sau:
- Bắt đầu ngay bằng: ## 1. Kết quả tóm tắt
- Có đúng 4 mục.
- Không lời chào.
- Không mở bài.
- Không bảng.
- Không thêm mục khác.
- Mỗi mục phải viết bằng gạch đầu dòng.
- Mỗi ý là một gạch đầu dòng riêng.
- Không dồn nhiều ý thành một đoạn văn dài.
- Mục 1 gạch đầu dòng đầu tiên PHẢI chép nguyên văn:
  - {facts["required_result_line"]}
- Mục 2 PHẢI có đủ các dòng/số liệu:
  - {facts["required_xai_line"]}
  - {facts["required_heatmap_line"]}
  - {facts["required_region_bullets"]}
  - {facts["required_energy_line"]}
- Mục 2 PHẢI có tọa độ dạng:
  (x=..., y=...) (vị trí tương đối: ...)

Câu trả lời sai trước đó:
{bad_answer}

Yêu cầu gốc:
{original_prompt}
""".strip()

    def _validate_explanation(
        self,
        text: str,
        tumor_label: str | None,
        classification_confidence: float | None,
        xai_metadata: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        text = (text or "").strip()

        required_heads = [
            "## 1. Kết quả tóm tắt",
            "## 2. Giải thích AI/XAI và vùng heatmap mô hình dựa vào",
            "## 3. Thông tin y khoa về loại u này",
            "## 4. Lưu ý an toàn",
        ]

        if not text.startswith(required_heads[0]):
            return False, "Sai format: không bắt đầu bằng mục 1."

        for head in required_heads:
            if head not in text:
                return False, f"Thiếu tiêu đề: {head}"

        if re.search(r"^##\s*5\.", text, flags=re.MULTILINE):
            return False, "Có thêm mục thứ 5."

        # Mỗi mục nên có bullet.
        for idx, head in enumerate(required_heads):
            start = text.find(head)
            end = text.find(required_heads[idx + 1]) if idx + 1 < len(required_heads) else len(text)
            section = text[start:end]
            if "- " not in section:
                return False, f"Mục thiếu gạch đầu dòng: {head}"

        facts = self._extract_prompt_facts(
            tumor_label=tumor_label,
            classification_confidence=classification_confidence,
            xai_metadata=xai_metadata,
        )

        section1 = text.split("## 2. Giải thích AI/XAI và vùng heatmap mô hình dựa vào")[0]

        if facts["pred_display"] not in section1:
            return False, "Thiếu loại u ở mục 1."

        if facts["confidence_pct"] != "không rõ" and facts["confidence_pct"] not in section1:
            return False, "Thiếu độ tin cậy phần trăm ở mục 1."

        vague_patterns = [
            r"độ tin cậy\s+(khá cao|cao|rất cao)",
            r"confidence\s+(cao|khá cao|rất cao)",
        ]
        lowered_section1 = section1.lower()
        for pattern in vague_patterns:
            if re.search(pattern, lowered_section1):
                return False, "Mục 1 nói mơ hồ về độ tin cậy, thiếu số cụ thể."

        section2_start = text.find("## 2. Giải thích AI/XAI và vùng heatmap mô hình dựa vào")
        section3_start = text.find("## 3. Thông tin y khoa về loại u này")
        section2 = text[section2_start:section3_start] if section2_start != -1 and section3_start != -1 else ""

        meta = xai_metadata or {}
        heatmap_summary = meta.get("heatmap_summary", {}) or {}

        if meta.get("target_minus_reference_logit_margin") is not None:
            margin_text = self._num(meta.get("target_minus_reference_logit_margin"))
            if margin_text not in section2:
                return False, "Thiếu điểm chênh lệch giữa hai lớp ở mục 2."

        if heatmap_summary.get("top10_energy_ratio") is not None:
            energy_text = self._pct(heatmap_summary.get("top10_energy_ratio"))
            if energy_text not in section2:
                return False, "Thiếu tỷ lệ tín hiệu vùng nóng ở mục 2."

        if heatmap_summary.get("peak_xy") is not None:
            if "(x=" not in section2 or "y=" not in section2:
                return False, "Thiếu tọa độ heatmap dạng (x=..., y=...) ở mục 2."

        if heatmap_summary.get("peak_xy_normalized") is not None:
            if "vị trí tương đối" not in section2.lower():
                return False, "Thiếu mô tả vị trí tương đối trong ngoặc ở mục 2."

        return True, "OK"

    def _fallback_text(
        self,
        tumor_label: str | None,
        classification_confidence: float | None,
        xai_metadata: dict[str, Any] | None,
        contexts: list[RetrievedContext],
        rag_diagnostics: dict[str, Any],
    ) -> str:
        facts = self._extract_prompt_facts(
            tumor_label=tumor_label,
            classification_confidence=classification_confidence,
            xai_metadata=xai_metadata,
        )

        rag_snippets: list[str] = []
        for idx, context in enumerate(contexts[:3], start=1):
            text = (context.parent_text or context.child_text or "").replace("\n", " ").strip()
            if text:
                rag_snippets.append(f"Nguồn {idx}: {text[:450]}")

        rag_text = " ".join(rag_snippets) if rag_snippets else "Ngữ cảnh RAG hiện chưa đủ thông tin để giải thích sâu."

        warning = ""
        if rag_diagnostics.get("store_has_mojibake"):
            warning = (
                " Có cảnh báo kỹ thuật: kho RAG có dấu hiệu lỗi mã hóa tiếng Việt, "
                "nên cần tạo lại artifact UTF-8 trước khi dùng trong báo cáo chính thức."
            )

        return f"""
## 1. Kết quả tóm tắt
- {facts["required_result_line"]}
- Đây là kết quả hỗ trợ tham khảo, không phải chẩn đoán y khoa cuối cùng.
- Độ tin cậy phân loại chỉ cho biết mức mô hình nghiêng về loại u này, không phải mức độ nguy hiểm của bệnh.

## 2. Giải thích AI/XAI và vùng heatmap mô hình dựa vào
- {facts["required_xai_line"]}
- {facts["required_heatmap_line"]}
- {facts["required_region_bullets"]}
- {facts["required_energy_line"]}
- Vùng sáng hoặc nóng trên heatmap là vùng ảnh có ảnh hưởng nhiều hơn đến quyết định phân loại của mô hình.
- Nếu heatmap còn lan rộng hoặc chưa tập trung mạnh, khả năng giải thích của XAI chỉ nên được xem là định hướng, không phải bằng chứng khu trú chắc chắn.

## 3. Thông tin y khoa về loại u này
- {rag_text}{warning}
- Các thông tin này chỉ là ngữ cảnh tham khảo từ RAG, không phải kết luận chẩn đoán riêng cho bệnh nhân.

## 4. Lưu ý an toàn
- Heatmap không phải bằng chứng giải phẫu bệnh và không xác nhận chắc chắn bản chất mô học của khối u.
- Kết quả AI không thay thế bác sĩ chuyên khoa.
- Cần đối chiếu với MRI đầy đủ, các chuỗi T1/T2/FLAIR/DWI, ảnh sau tiêm nếu có, triệu chứng lâm sàng và kết luận chẩn đoán hình ảnh.
""".strip()
