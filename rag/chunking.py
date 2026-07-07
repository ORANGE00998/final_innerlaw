# rag/chunking.py
# 역할: Phase 3 판정 결과(readiness_report.csv)에서 Ready/Partial 문서만 골라,
#       페이지별 전체 텍스트를 검색 가능한 Chunk 로 나누고 출처 추적용 메타데이터를 붙인다.
# - 입력: outputs/readiness_report.csv + outputs/extracted/<source>.json (페이지별 전체 텍스트)
# - 출력: outputs/chunk_report.csv
# - Blocked 문서는 제외한다.
# - Embedding / Vector DB / Retriever 는 다루지 않는다(다음 단계).

import os
import csv
from dataclasses import dataclass, asdict, fields

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestion import load_extracted_text  # 페이지별 전체 텍스트 로더


# ── 설정 상수 ────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 800        # 토큰 기준 기본 chunk 크기
DEFAULT_CHUNK_OVERLAP = 100     # 토큰 기준 기본 overlap
ENCODING_NAME = "o200k_base"    # 토큰 계산·분할 기준 인코딩
READINESS_PATH = os.path.join("outputs", "readiness_report.csv")
CHUNK_REPORT_PATH = os.path.join("outputs", "chunk_report.csv")

# Chunking 대상 상태 (Blocked 등은 제외)
CHUNKABLE_STATUSES = {"Ready", "Partial"}


# ── 청크 한 개 (CSV 컬럼: 메타 10개 + text) ──────────────────
@dataclass
class ChunkRecord:
    source: str
    file_type: str
    parser_type: str
    page: int
    readiness_status: str
    warning: str
    chunk_id: str            # 출처 추적용 고유 ID
    chunk_index: int         # 같은 (source, page) 안에서의 순번(0부터)
    token_count: int         # 청크의 토큰 수
    char_count: int          # 청크의 글자 수
    chunk_text: str          # 청크 '전체' 본문 (다음 단계 Embedding 입력용) — preview 가 아닌 원문 전체


# ── 토큰 인코더 (한 번만 생성해 재사용) ──────────────────────
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(ENCODING_NAME)
    return _encoder


def count_tokens(text: str) -> int:
    """텍스트의 토큰 수를 센다."""
    return len(_get_encoder().encode(text))


# ── readiness 읽기 & 대상 선별 ───────────────────────────────
def load_readiness_report(path: str = READINESS_PATH) -> list:
    """판정 CSV(outputs/readiness_report.csv)를 읽어 행(dict) 목록을 돌려준다.
    파일이 없으면 빈 목록. (utf-8-sig: 저장 시 붙은 BOM 처리)"""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_chunkable(rows: list):
    """Ready/Partial 행만 남기고 Blocked 등은 제외한다.
    (선별된 행 목록, 제외된 행 수) 를 돌려준다."""
    chunkable = [r for r in rows if r.get("readiness_status") in CHUNKABLE_STATUSES]
    excluded = len(rows) - len(chunkable)
    return chunkable, excluded


# ── 토큰 기준 분할 ───────────────────────────────────────────
def split_text(text: str,
               chunk_size: int = DEFAULT_CHUNK_SIZE,
               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> list:
    """RecursiveCharacterTextSplitter 를 '토큰 기준'으로 사용해 텍스트를 나눈다.
    chunk_size / chunk_overlap 은 토큰 단위이며, size 는 400 / 800 / 1200 등으로 바꿔 비교할 수 있다."""
    if not text.strip():
        return []
    # overlap 이 size 보다 크면 라이브러리가 오류를 내므로 안전하게 줄인다
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=ENCODING_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


# ── 청크 + 메타데이터 생성 ──────────────────────────────────
def build_chunks(rows: list, chunk_size: int, chunk_overlap: int) -> list:
    """선별된 readiness 행마다 '해당 페이지'의 전체 텍스트를 읽어 청크와 메타데이터를 만든다."""
    records = []
    text_cache = {}  # source 별 페이지 텍스트 캐시 (같은 파일을 반복해 읽지 않도록)

    for row in rows:
        source = row.get("source", "")
        page = str(row.get("page", "1"))
        if source not in text_cache:
            text_cache[source] = load_extracted_text(source)
        page_text = text_cache[source].get(page, "")

        for index, piece in enumerate(split_text(page_text, chunk_size, chunk_overlap)):
            records.append(ChunkRecord(
                source=source,
                file_type=row.get("file_type", ""),
                parser_type=row.get("parser_type", ""),
                page=int(page) if page.isdigit() else 1,
                readiness_status=row.get("readiness_status", ""),
                warning=row.get("warning", ""),
                chunk_id=f"{source}|p{page}|c{index}",
                chunk_index=index,
                token_count=count_tokens(piece),
                char_count=len(piece),
                chunk_text=piece,   # 잘라내지 않은 청크 전체 본문
            ))
    return records


# ── 요약 (총 개수 + source 별 개수) ──────────────────────────
def summarize(records: list) -> dict:
    per_source = {}
    for record in records:
        per_source[record.source] = per_source.get(record.source, 0) + 1
    return {"total": len(records), "per_source": per_source}


# ── 청크 결과 CSV 저장 (덮어쓰기 스냅샷) ─────────────────────
def save_chunk_report(records: list, path: str = CHUNK_REPORT_PATH) -> str:
    """청킹 결과를 CSV 로 덮어쓴다. 저장 경로를 돌려준다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    column_names = [f.name for f in fields(ChunkRecord)]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


# ── 메인: 전체 청킹 실행 ─────────────────────────────────────
def run_chunking(chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
                 readiness_path: str = READINESS_PATH,
                 output_path: str = CHUNK_REPORT_PATH):
    """readiness 에서 Ready/Partial 만 골라 토큰 기준으로 청킹하고 CSV 저장.
    반환: (records, summary, excluded_count, saved_path)"""
    rows = load_readiness_report(readiness_path)
    chunkable, excluded = select_chunkable(rows)
    records = build_chunks(chunkable, chunk_size, chunk_overlap)
    saved_path = save_chunk_report(records, output_path)
    summary = summarize(records)
    return records, summary, excluded, saved_path
