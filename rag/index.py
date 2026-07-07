# rag/index.py
# 역할: Phase 4 청킹 결과(outputs/chunk_report.csv)를 읽어 각 Chunk 본문을
#       OpenAI Embedding 으로 변환하고, 본문 + Metadata 를 Chroma Vector DB 에 저장한다.
# - 저장 위치: chroma_db/ , collection: rag_docs , 거리 기준: cosine
# - Retriever 검색 / RAG 답변 / Source Citation 은 다루지 않는다(다음 단계).
# - API Key 는 .env 의 OPENAI_API_KEY 에서만 읽는다.

import os
import csv
from dataclasses import dataclass, asdict, fields

import chromadb
from dotenv import load_dotenv
from openai import OpenAI


# ── 상단 상수 (여기만 바꾸면 됨) ─────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"   # 기본 Embedding 모델
CHROMA_DIR = "chroma_db"                      # Chroma 저장 위치
COLLECTION_NAME = "rag_docs"                  # Chroma collection 이름
DISTANCE_METRIC = "cosine"                    # 거리 기준 (hnsw:space)

CHUNK_REPORT_PATH = os.path.join("outputs", "chunk_report.csv")
VECTOR_DB_REPORT_PATH = os.path.join("outputs", "vector_db_report.csv")

# Chunk 본문 컬럼 후보 (앞에서부터 우선). preview 계열은 본문으로 인정하지 않는다.
BODY_COLUMNS = ("chunk_text", "content", "text")
EMBED_BATCH = 100                            # Embedding 요청 배치 크기
CHROMA_UPSERT_BATCH = 5000                   # Chroma 한 번에 저장 개수 (최대 약 5461 보다 안전하게)

# Chroma metadata 로 함께 저장할 필드 (요구 메타데이터 10개)
METADATA_FIELDS = ["source", "file_type", "parser_type", "page", "readiness_status",
                   "warning", "chunk_id", "chunk_index", "token_count", "char_count"]
INT_FIELDS = {"page", "chunk_index", "token_count", "char_count"}


# ── vector_db_report.csv 한 줄 ───────────────────────────────
@dataclass
class VectorDBRecord:
    chunk_id: str
    source: str
    file_type: str
    page: int
    chunk_index: int
    token_count: int
    char_count: int
    embedding_model: str
    embedding_dim: int
    collection: str
    status: str


# ── 작은 도우미 ─────────────────────────────────────────────
def _to_int(value) -> int:
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return 0


def load_chunk_report(path: str = CHUNK_REPORT_PATH):
    """chunk_report.csv 를 읽어 (헤더, 행목록) 을 돌려준다. 없으면 (None, [])."""
    if not os.path.exists(path):
        return None, []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames, rows


def find_body_column(header):
    """헤더에서 Chunk 본문 컬럼명을 찾는다 (chunk_text → content → text). 없으면 None."""
    if not header:
        return None
    for name in BODY_COLUMNS:
        if name in header:
            return name
    return None


def validate_chunks(header, rows):
    """chunk_report.csv 가 인덱싱에 적합한지 확인한다.
    반환: (ok, message, body_column)"""
    if not rows:
        return False, "chunk_report.csv 가 없거나 비어 있습니다. 먼저 Chunking 을 실행해 주세요.", None
    if "chunk_id" not in (header or []):
        return False, "chunk_id 컬럼이 없습니다. Chunking 단계를 다시 실행해 주세요.", None

    body = find_body_column(header)
    if body is None:
        preview_like = [c for c in (header or []) if "preview" in c.lower()]
        if preview_like:
            return (False,
                    f"본문 컬럼(chunk_text/content)이 없고 미리보기 컬럼({', '.join(preview_like)})만 있습니다. "
                    "Embedding 은 전체 본문이 필요합니다. Chunking 을 다시 실행해 chunk_text 를 만들어 주세요.",
                    None)
        return False, "Chunk 본문 컬럼(chunk_text/content/text)을 찾을 수 없습니다.", None

    non_empty = sum(1 for r in rows if (r.get(body) or "").strip())
    if non_empty == 0:
        return False, f"본문 컬럼 '{body}' 이 모두 비어 있습니다. Chunking 결과를 확인해 주세요.", None

    return True, f"본문 컬럼 '{body}' 사용 (총 {len(rows)}개 청크)", body


# ── OpenAI Embedding ─────────────────────────────────────────
def get_openai_client():
    """.env 에서 OPENAI_API_KEY 를 읽어 OpenAI 클라이언트를 만든다. 키가 없으면 None."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def embed_texts(client, texts, model: str = EMBEDDING_MODEL, batch_size: int = EMBED_BATCH):
    """텍스트 목록을 배치로 Embedding 한다 (실제 Chunk 본문 대상). 벡터 리스트를 돌려준다."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        # response.data 는 입력 순서를 그대로 보존한다
        vectors.extend(item.embedding for item in response.data)
    return vectors


# ── Chroma ───────────────────────────────────────────────────
def get_collection(recreate: bool = False):
    """Chroma collection 을 준비한다. recreate=True 면 기존 것을 지우고 새로 만든다."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # 없으면 무시
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC},
    )


def build_metadata(row) -> dict:
    """행에서 Chroma 에 저장할 10개 메타데이터 dict 를 만든다 (int 변환 포함)."""
    meta = {}
    for field_name in METADATA_FIELDS:
        value = row.get(field_name, "")
        meta[field_name] = _to_int(value) if field_name in INT_FIELDS else (value or "")
    return meta


# ── vector_db_report.csv 저장 ────────────────────────────────
def save_vector_db_report(records, path: str = VECTOR_DB_REPORT_PATH) -> str:
    """저장 결과를 CSV 로 덮어쓴다. 저장 경로를 돌려준다."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    column_names = [f.name for f in fields(VectorDBRecord)]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


# ── 메인: 인덱스 생성 ────────────────────────────────────────
def build_index(recreate: bool = False, client=None) -> dict:
    """chunk_report.csv 를 읽어 Embedding 후 Chroma 에 저장한다.
    반환: 요약 dict (ok / message / read_count / stored_count / model / collection /
                     chroma_dir / report_path / embedding_dim / collection_count)"""
    header, rows = load_chunk_report()
    ok, message, body = validate_chunks(header, rows)

    result = {
        "ok": False, "message": message,
        "read_count": len(rows), "stored_count": 0,
        "model": EMBEDDING_MODEL, "collection": COLLECTION_NAME,
        "chroma_dir": CHROMA_DIR, "report_path": None,
        "embedding_dim": 0, "collection_count": None,
    }
    if not ok:
        return result

    # 본문이 실제로 있는 행만 인덱싱 (빈 본문 제외)
    usable = [r for r in rows if (r.get(body) or "").strip()]
    ids = [r.get("chunk_id", "") for r in usable]
    documents = [r.get(body) for r in usable]
    metadatas = [build_metadata(r) for r in usable]

    # OpenAI 클라이언트 (보안: .env 의 OPENAI_API_KEY 만 사용)
    if client is None:
        client = get_openai_client()
    if client is None:
        result["message"] = "OPENAI_API_KEY 를 찾을 수 없습니다. .env 를 확인해 주세요."
        return result

    # Embedding → Chroma 저장. 오류가 나도 화면에 raw 트레이스백 대신 안내를 보여주도록 여기서 잡는다.
    try:
        # 실제 Chunk 본문을 Embedding
        embeddings = embed_texts(client, documents)
        dim = len(embeddings[0]) if embeddings else 0

        # Chroma 는 한 번에 넣을 수 있는 최대 개수(약 5461)가 있어 배치로 나눠 저장한다.
        # (upsert 라 같은 chunk_id 로 다시 실행해도 안전)
        collection = get_collection(recreate=recreate)
        for start in range(0, len(ids), CHROMA_UPSERT_BATCH):
            end = start + CHROMA_UPSERT_BATCH
            collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
        collection_count = collection.count()

        # 결과 리포트 저장
        report_records = [VectorDBRecord(
            chunk_id=r.get("chunk_id", ""),
            source=r.get("source", ""),
            file_type=r.get("file_type", ""),
            page=_to_int(r.get("page", 0)),
            chunk_index=_to_int(r.get("chunk_index", 0)),
            token_count=_to_int(r.get("token_count", 0)),
            char_count=_to_int(r.get("char_count", 0)),
            embedding_model=EMBEDDING_MODEL,
            embedding_dim=dim,
            collection=COLLECTION_NAME,
            status="stored",
        ) for r in usable]
        report_path = save_vector_db_report(report_records)
    except Exception as error:
        result["message"] = f"Vector DB 생성 중 오류가 발생했습니다: {error}"
        return result

    result.update({
        "ok": True,
        "message": f"{len(usable)}개 청크를 Embedding 하여 Chroma 에 저장했습니다.",
        "stored_count": len(usable),
        "report_path": report_path,
        "embedding_dim": dim,
        "collection_count": collection_count,
    })
    return result
