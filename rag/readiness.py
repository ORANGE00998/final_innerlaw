# rag/readiness.py
# 역할: Phase 2 문서 진단 결과(outputs/ingestion_report.csv)를 읽어,
#       각 문서/페이지를 RAG 다음 단계로 넘겨도 되는지 판정하는 "Readiness Gate".
# - 판정 상태: Ready / Partial / Blocked
# - 결과는 outputs/readiness_report.csv 로 저장(덮어쓰기 스냅샷).
# - Chunking / Embedding / Vector DB / Retriever 는 다루지 않는다(다음 단계).

import os
import csv
from dataclasses import dataclass, asdict, fields


# ── 설정 상수 ────────────────────────────────────────────────
MIN_READY_CHARS = 200     # 이 이상이면 "텍스트 충분"
MIN_USABLE_CHARS = 50     # 이 미만이면 "텍스트 매우 짧음"
INPUT_PATH = os.path.join("outputs", "ingestion_report.csv")
OUTPUT_PATH = os.path.join("outputs", "readiness_report.csv")


# ── 판정 결과 한 줄 (CSV 13개 컬럼과 1:1) ────────────────────
@dataclass
class ReadinessRecord:
    source: str
    file_type: str
    parser_type: str
    page: int
    text_length: int
    scanned: bool
    warning: str
    readiness_status: str    # Ready / Partial / Blocked
    rag_ready: bool          # 다음 단계로 넘겨도 되는가 (Ready 일 때만 True)
    needs_ocr: bool          # OCR 이 필요한가
    needs_vision: bool       # Vision 이 필요한가
    needs_conversion: bool   # 형식 변환이 필요한가 (HWP 등)
    reason: str              # 판정 근거(사람이 읽는 문장)


# ── 작은 도우미: CSV 값 → 파이썬 타입 변환 ──────────────────
def _to_int(value) -> int:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return 0


def _to_bool(value) -> bool:
    return str(value).strip().lower() == "true"


# ── 진단 CSV 읽기 ────────────────────────────────────────────
def load_ingestion_report(path: str = INPUT_PATH) -> list:
    """진단 CSV(outputs/ingestion_report.csv)를 읽어 행(dict) 목록을 돌려준다.
    파일이 없으면 빈 목록을 돌려준다. (utf-8-sig: 저장 시 붙은 BOM 처리)"""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _dedup_keep_last(rows: list) -> list:
    """같은 (source, page) 가 여러 번 있으면 '마지막' 것만 남긴다 (재업로드 대비)."""
    latest = {}
    for row in rows:
        key = (row.get("source", ""), str(row.get("page", "")))
        latest[key] = row  # 뒤에 나온 행이 앞의 행을 덮어씀 → 마지막 것만 유지
    return list(latest.values())


# ── 판정 근거 문장 만들기 ────────────────────────────────────
def _build_reason(status, file_type, text_length, scanned, warning,
                  needs_ocr, needs_vision, needs_conversion) -> str:
    """사람이 읽을 수 있는 판정 근거 문장을 만든다."""
    if status == "Ready":
        return f"텍스트 충분({text_length:,}자)하고 경고가 없어 RAG 투입 가능"

    if status == "Partial":
        if warning:
            return f"텍스트는 확보({text_length:,}자)했으나 경고가 있어 검토 후 투입 권장: {warning}"
        return f"텍스트가 다소 짧아({text_length:,}자, 기준 {MIN_READY_CHARS}자) 검토 후 투입 권장"

    # Blocked — 가장 구체적인 원인부터 안내
    if file_type == "IMAGE":
        return "이미지 파일 → OCR 또는 Vision 필요, 다음 단계 투입 불가"
    if needs_conversion:
        return f"변환/구조 확인이 필요한 문서({file_type}) → 변환 후 재진단 필요, 투입 불가"
    if scanned or needs_ocr or needs_vision:
        return "스캔본으로 의심 → OCR 또는 Vision 필요, 투입 불가"
    return f"추출 텍스트가 매우 짧음({text_length:,}자, 기준 {MIN_USABLE_CHARS}자) → 투입 불가"


# ── 한 행 판정 ───────────────────────────────────────────────
def classify_row(row: dict) -> ReadinessRecord:
    """진단 CSV 한 행(dict)을 읽어 판정 결과(ReadinessRecord)를 만든다."""
    source = row.get("source", "")
    file_type = row.get("file_type", "")
    parser_type = row.get("parser_type", "")
    page = _to_int(row.get("page", 0))
    text_length = _to_int(row.get("text_length", 0))
    scanned = _to_bool(row.get("scanned", "False"))
    warning = (row.get("warning") or "").strip()

    # 파생 플래그: warning / 형식에서 도출
    needs_ocr = "OCR" in warning
    needs_vision = "Vision" in warning
    needs_conversion = (
        ("변환" in warning)
        or (file_type == "HWP")
        or (file_type == "HWPX" and "구조 확인" in warning)
    )

    # 상태 판정 (우선순위: Blocked → Partial → Ready)
    if file_type == "IMAGE" or scanned or text_length < MIN_USABLE_CHARS:
        status = "Blocked"
    elif warning or text_length < MIN_READY_CHARS:
        status = "Partial"
    else:
        status = "Ready"

    reason = _build_reason(status, file_type, text_length, scanned, warning,
                           needs_ocr, needs_vision, needs_conversion)

    return ReadinessRecord(
        source=source,
        file_type=file_type,
        parser_type=parser_type,
        page=page,
        text_length=text_length,
        scanned=scanned,
        warning=warning,
        readiness_status=status,
        rag_ready=(status == "Ready"),
        needs_ocr=needs_ocr,
        needs_vision=needs_vision,
        needs_conversion=needs_conversion,
        reason=reason,
    )


# ── 상태별 개수 요약 ─────────────────────────────────────────
def summarize(records) -> dict:
    """상태별(Ready/Partial/Blocked) 개수를 센다."""
    counts = {"Ready": 0, "Partial": 0, "Blocked": 0}
    for record in records:
        counts[record.readiness_status] = counts.get(record.readiness_status, 0) + 1
    return counts


# ── 판정 결과 CSV 저장 (덮어쓰기 스냅샷) ─────────────────────
def save_readiness_report(records, path: str = OUTPUT_PATH) -> str:
    """판정 결과를 CSV 로 덮어쓴다(매 판정마다 최신 스냅샷). 저장 경로를 돌려준다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    column_names = [field.name for field in fields(ReadinessRecord)]
    # utf-8-sig: Excel 에서 한글이 깨지지 않도록 BOM 을 붙인다
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


# ── 메인: 전체 판정 ──────────────────────────────────────────
def evaluate_readiness(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH):
    """진단 CSV 를 읽어 전체를 판정하고, 결과 CSV 저장 후 (records, summary, saved_path) 반환.
    같은 (source, page) 는 마지막 항목만 판정한다(재업로드 대비)."""
    rows = _dedup_keep_last(load_ingestion_report(input_path))
    records = [classify_row(row) for row in rows]
    saved_path = save_readiness_report(records, output_path)
    summary = summarize(records)
    return records, summary, saved_path
