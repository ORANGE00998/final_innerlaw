# rag/ingestion.py
# 역할: 업로드된 문서의 "형식 진단 + 텍스트 추출"을 담당하는 모듈.
# - 지원 형식: PDF, TXT, DOCX, HWP, HWPX, 이미지
# - 각 파일(또는 PDF 페이지)마다 진단 결과(IngestionRecord)를 만들고,
#   Q&A 에 쓸 전체 텍스트(IngestionResult.text)도 함께 돌려준다.
# - 진단 결과는 outputs/ingestion_report.csv 에 "누적(append)" 저장한다.
# - Chunking / Embedding / Vector DB / Retriever 는 다루지 않는다(다음 단계).

import os
import io
import re
import csv
import json
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict, fields, field

import docx            # python-docx
import fitz            # PyMuPDF
import pymupdf4llm     # PDF -> Markdown 추출


# ── 설정 상수 ────────────────────────────────────────────────
PREVIEW_CHARS = 300               # content_preview 로 보여줄 최대 글자 수
PDF_SCANNED_MIN_CHARS = 20        # PDF 페이지 텍스트가 이보다 짧으면 스캔본으로 의심
HWPX_MIN_CHARS = 10               # HWPX 추출 텍스트가 이보다 짧으면 구조 확인 필요로 간주
REPORT_PATH = os.path.join("outputs", "ingestion_report.csv")
EXTRACTED_DIR = os.path.join("outputs", "extracted")  # 페이지별 전체 텍스트 저장소(Phase 4 청킹용)

# 이미지로 취급할 확장자
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}


# ── 진단 결과 한 줄을 담는 구조 (CSV 컬럼과 1:1) ─────────────
@dataclass
class IngestionRecord:
    source: str            # 파일명
    file_type: str         # PDF / TXT / DOCX / HWP / HWPX / IMAGE / UNKNOWN
    parser_type: str       # 사용한 추출기 (pymupdf4llm / plain-text / python-docx / hwpx-zip / none)
    page: int              # 페이지 번호 (PDF 는 1..N, 그 외는 1)
    text_length: int       # 추출된 텍스트 길이(글자 수)
    scanned: bool          # 스캔본(=텍스트 추출이 어려운 상태) 의심 여부
    warning: str           # 사용자 안내 문구 (없으면 빈 문자열)
    content_preview: str   # 추출 텍스트 앞부분 미리보기


@dataclass
class IngestionResult:
    records: list          # list[IngestionRecord] — 진단 표/CSV 용 (페이지·파일별 1줄)
    text: str              # 추출된 전체 텍스트 — Q&A(모델 입력) 용
    page_texts: list = field(default_factory=list)  # 페이지별 전체 텍스트(청킹용). PDF=페이지수, 그 외=1개


# ── 작은 도우미 ─────────────────────────────────────────────
def _preview(text: str) -> str:
    """텍스트 앞부분을 잘라 미리보기 문자열을 만든다 (연속 공백은 하나로 정리)."""
    cleaned = " ".join(text.split())
    return cleaned[:PREVIEW_CHARS]


def detect_file_type(filename: str) -> str:
    """파일 확장자를 보고 형식 이름을 돌려준다."""
    ext = os.path.splitext(filename.lower())[1]
    mapping = {".pdf": "PDF", ".txt": "TXT", ".docx": "DOCX", ".hwp": "HWP", ".hwpx": "HWPX"}
    if ext in mapping:
        return mapping[ext]
    if ext in IMAGE_EXTS:
        return "IMAGE"
    return "UNKNOWN"


# ── 형식별 추출 함수 (각각 IngestionResult 를 돌려준다) ──────
def ingest_pdf(source: str, data: bytes) -> IngestionResult:
    """PDF: PyMuPDF4LLM 으로 페이지별 Markdown 텍스트를 추출한다.
    텍스트가 거의 없는 페이지는 스캔본(이미지 PDF)일 가능성이 높아 warning 을 남긴다."""
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        pages = pymupdf4llm.to_markdown(doc, page_chunks=True)
    finally:
        doc.close()

    records, texts = [], []
    for page_number, page in enumerate(pages, start=1):
        text = page.get("text", "") or ""
        texts.append(text)
        is_scanned = len(text.strip()) < PDF_SCANNED_MIN_CHARS
        records.append(IngestionRecord(
            source=source,
            file_type="PDF",
            parser_type="pymupdf4llm",
            page=page_number,
            text_length=len(text),
            scanned=is_scanned,
            warning="OCR 또는 Vision 필요 가능성" if is_scanned else "",
            content_preview=_preview(text),
        ))

    # 페이지가 하나도 없는(빈) PDF 도 진단 결과 한 줄은 남긴다
    if not records:
        records.append(IngestionRecord(
            source=source, file_type="PDF", parser_type="pymupdf4llm", page=1,
            text_length=0, scanned=True,
            warning="OCR 또는 Vision 필요 가능성", content_preview="",
        ))
        texts = [""]

    return IngestionResult(records=records, text="\n\n".join(texts), page_texts=texts)


def ingest_txt(source: str, data: bytes) -> IngestionResult:
    """TXT: 파일 내용을 그대로 읽는다 (utf-8 우선, 안 되면 cp949)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("cp949", errors="replace")
    record = IngestionRecord(
        source=source, file_type="TXT", parser_type="plain-text", page=1,
        text_length=len(text), scanned=False, warning="",
        content_preview=_preview(text),
    )
    return IngestionResult(records=[record], text=text, page_texts=[text])


def ingest_docx(source: str, data: bytes) -> IngestionResult:
    """DOCX: python-docx 로 문단 텍스트를 추출한다."""
    document = docx.Document(io.BytesIO(data))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    record = IngestionRecord(
        source=source, file_type="DOCX", parser_type="python-docx", page=1,
        text_length=len(text), scanned=False, warning="",
        content_preview=_preview(text),
    )
    return IngestionResult(records=[record], text=text, page_texts=[text])


def ingest_hwp(source: str, data: bytes) -> IngestionResult:
    """HWP: 직접 파싱하지 않고 변환을 권장한다 (구버전 바이너리 포맷)."""
    record = IngestionRecord(
        source=source, file_type="HWP", parser_type="none", page=1,
        text_length=0, scanned=False,
        warning="HWPX/PDF/DOCX 변환 권장", content_preview="",
    )
    return IngestionResult(records=[record], text="", page_texts=[""])


def ingest_hwpx(source: str, data: bytes) -> IngestionResult:
    """HWPX: zip 구조를 열어 Contents/section*.xml 의 텍스트 추출을 '시도'한다.
    구조가 예상과 다르거나 내용이 거의 없으면 '구조 확인 필요' 경고를 남긴다."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            section_names = sorted(
                name for name in zf.namelist()
                if name.startswith("Contents/section") and name.endswith(".xml")
            )
            if not section_names:
                raise ValueError("Contents/section*.xml 을 찾지 못함")

            texts = []
            for name in section_names:
                root = ET.fromstring(zf.read(name))
                # 네임스페이스를 무시하고, 지역 태그 이름이 't'(본문 글자) 인 요소를 모은다
                for elem in root.iter():
                    local_tag = elem.tag.split("}")[-1]  # '{namespace}t' -> 't'
                    if local_tag == "t" and elem.text:
                        texts.append(elem.text)
            text = " ".join(texts)
    except Exception as error:
        record = IngestionRecord(
            source=source, file_type="HWPX", parser_type="hwpx-zip", page=1,
            text_length=0, scanned=True,
            warning=f"HWPX 구조 확인 필요 ({error})", content_preview="",
        )
        return IngestionResult(records=[record], text="", page_texts=[""])

    # 추출은 됐지만 내용이 거의 없으면 구조 확인이 필요하다고 안내
    if len(text.strip()) < HWPX_MIN_CHARS:
        record = IngestionRecord(
            source=source, file_type="HWPX", parser_type="hwpx-zip", page=1,
            text_length=len(text), scanned=True,
            warning="HWPX 구조 확인 필요", content_preview=_preview(text),
        )
        return IngestionResult(records=[record], text=text, page_texts=[text])

    record = IngestionRecord(
        source=source, file_type="HWPX", parser_type="hwpx-zip", page=1,
        text_length=len(text), scanned=False, warning="",
        content_preview=_preview(text),
    )
    return IngestionResult(records=[record], text=text, page_texts=[text])


def ingest_image(source: str, data: bytes) -> IngestionResult:
    """이미지: 텍스트 추출을 시도하지 않고 OCR/Vision 이 필요하다고 안내한다."""
    record = IngestionRecord(
        source=source, file_type="IMAGE", parser_type="none", page=1,
        text_length=0, scanned=True,
        warning="OCR 또는 Vision 필요", content_preview="",
    )
    return IngestionResult(records=[record], text="", page_texts=[""])


# ── 메인 분기 ────────────────────────────────────────────────
_HANDLERS = {
    "PDF": ingest_pdf,
    "TXT": ingest_txt,
    "DOCX": ingest_docx,
    "HWP": ingest_hwp,
    "HWPX": ingest_hwpx,
    "IMAGE": ingest_image,
}


def ingest_document(source: str, data: bytes) -> IngestionResult:
    """파일 형식을 판별하고 알맞은 추출기를 호출한다.
    추출 도중 예상치 못한 오류가 나도 앱이 멈추지 않도록, 오류를 warning 레코드로 바꾼다."""
    file_type = detect_file_type(source)
    handler = _HANDLERS.get(file_type)

    if handler is None:  # 지원하지 않는 형식
        record = IngestionRecord(
            source=source, file_type="UNKNOWN", parser_type="none", page=1,
            text_length=0, scanned=False,
            warning="지원하지 않는 형식입니다 (PDF/TXT/DOCX/HWP/HWPX/이미지)",
            content_preview="",
        )
        return IngestionResult(records=[record], text="", page_texts=[""])

    try:
        return handler(source, data)
    except Exception as error:
        record = IngestionRecord(
            source=source, file_type=file_type, parser_type="none", page=1,
            text_length=0, scanned=False,
            warning=f"처리 중 오류가 발생했습니다: {error}", content_preview="",
        )
        return IngestionResult(records=[record], text="", page_texts=[""])


# ── 진단 결과 CSV 저장 (누적) ────────────────────────────────
def save_report(records, path: str = REPORT_PATH) -> str:
    """진단 결과를 CSV 에 누적(append) 저장한다.
    폴더와 헤더는 필요할 때 자동으로 만든다. 저장한 파일 경로를 돌려준다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    column_names = [field.name for field in fields(IngestionRecord)]
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)

    # utf-8-sig: Excel 에서 한글이 깨지지 않도록 BOM 을 붙인다
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        if need_header:
            writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


# ── 페이지별 전체 텍스트 저장/읽기 (Phase 4 청킹에서 사용) ───
def _safe_name(source: str) -> str:
    """파일명을 파일 시스템에 안전한 형태로 바꾼다 (경로 문제 방지)."""
    return re.sub(r"[^\w.\-]+", "_", source)


def save_extracted_text(source: str, page_texts: list, out_dir: str = EXTRACTED_DIR) -> str:
    """추출된 '페이지별 전체 텍스트'를 outputs/extracted/<source>.json 에 저장한다.
    형식: {"1": "1페이지 텍스트", "2": ...}. 다음 단계(청킹)가 이 파일을 읽는다."""
    os.makedirs(out_dir, exist_ok=True)
    mapping = {str(i): text for i, text in enumerate(page_texts, start=1)}
    path = os.path.join(out_dir, _safe_name(source) + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    return path


def load_extracted_text(source: str, out_dir: str = EXTRACTED_DIR) -> dict:
    """저장해 둔 페이지별 텍스트를 {페이지번호(str): 텍스트} 로 읽는다. 없으면 빈 dict."""
    path = os.path.join(out_dir, _safe_name(source) + ".json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
