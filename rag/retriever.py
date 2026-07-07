# rag/retriever.py
# 역할: Phase 5 에서 만든 Chroma Vector DB(chroma_db/, collection=rag_docs)를 불러와,
#       사용자 질문을 Embedding 으로 바꿔 Top-K 관련 Chunk 를 검색한다 (Retrieval Debug).
# - 최종 RAG 답변 / Source Citation / Reranker / Hybrid Search 는 다루지 않는다(다음 단계).
# - 보안: 검색된 Context(chunk 본문)는 '명령'이 아니라 '데이터'로만 취급한다.
#         (다음 단계에서 모델에 넘길 때도 지시문으로 해석하지 않도록 주의 — 프롬프트 인젝션 방지)
# - API Key 는 .env 의 OPENAI_API_KEY 에서만 읽는다 (index.py 의 get_openai_client 재사용).

import os
import csv
from dataclasses import dataclass, asdict, fields

import yaml
import chromadb

# Phase 5(index.py) 의 설정·임베딩 로직을 그대로 재사용 (index.py 는 수정하지 않음)
from rag.index import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL,
    get_openai_client, embed_texts,
)


# ── 설정 상수 ────────────────────────────────────────────────
DEFAULT_K = 4                                    # 기본 Top-K
PREVIEW_CHARS = 200                              # preview 로 보여줄 최대 글자 수
SEARCH_RESULTS_PATH = os.path.join("outputs", "vector_search_results.csv")
EVAL_QUESTIONS_PATH = os.path.join("eval", "questions.yaml")


# ── 검색 결과 한 줄 ──────────────────────────────────────────
@dataclass
class SearchResult:
    rank: int
    distance: float       # Chroma cosine 거리 (작을수록 유사)
    score: float          # 1 - distance (클수록 유사, 사람이 보기 편한 값)
    source: str
    file_type: str
    parser_type: str
    page: int
    chunk_id: str
    warning: str
    preview: str          # chunk 본문 앞부분 미리보기


# ── 도우미 ──────────────────────────────────────────────────
def _preview(text: str) -> str:
    return " ".join((text or "").split())[:PREVIEW_CHARS]


def get_collection():
    """chroma_db/ 의 rag_docs collection 을 불러온다. 없거나 열 수 없으면 None."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


# ── 검색 ─────────────────────────────────────────────────────
def search(question: str, k: int = DEFAULT_K, client=None) -> dict:
    """질문을 Embedding 으로 바꿔 Top-K Chunk 를 검색한다.
    반환: dict(ok, message, results, count, k, collection_count[, saved_path])"""
    result = {"ok": False, "message": "", "results": [], "count": 0,
              "k": k, "collection_count": 0}

    if not (question or "").strip():
        result["message"] = "질문을 입력해 주세요."
        return result

    collection = get_collection()
    if collection is None:
        result["message"] = (f"Vector DB(collection '{COLLECTION_NAME}')를 찾을 수 없습니다. "
                             "먼저 Phase 5 에서 Vector DB 를 생성해 주세요.")
        return result

    total = collection.count()
    result["collection_count"] = total
    if total == 0:
        result["message"] = "Vector DB 가 비어 있습니다. 먼저 Vector DB 를 생성해 주세요."
        return result

    if client is None:
        client = get_openai_client()
    if client is None:
        result["message"] = "OPENAI_API_KEY 를 찾을 수 없습니다. .env 를 확인해 주세요."
        return result

    # 질문 Embedding → Top-K 검색
    try:
        query_vec = embed_texts(client, [question])[0]
        n = max(1, min(k, total))  # k 가 저장된 개수보다 크면 줄인다
        res = collection.query(query_embeddings=[query_vec], n_results=n)
    except Exception as error:
        result["message"] = f"검색 중 오류가 발생했습니다: {error}"
        return result

    # Chroma 결과는 [[...]] 중첩 → [0] 으로 첫 질문의 결과를 꺼낸다
    distances = (res.get("distances") or [[]])[0]
    documents = (res.get("documents") or [[]])[0]
    metadatas = (res.get("metadatas") or [[]])[0]

    results = []
    for rank, (dist, doc, meta) in enumerate(zip(distances, documents, metadatas), start=1):
        meta = meta or {}
        results.append(SearchResult(
            rank=rank,
            distance=round(float(dist), 4),
            score=round(1.0 - float(dist), 4),
            source=str(meta.get("source", "")),
            file_type=str(meta.get("file_type", "")),
            parser_type=str(meta.get("parser_type", "")),
            page=int(meta.get("page", 0) or 0),
            chunk_id=str(meta.get("chunk_id", "")),
            warning=str(meta.get("warning", "")),
            preview=_preview(doc),
        ))

    result.update({
        "ok": True, "results": results, "count": len(results),
        "message": f"Top-{len(results)} 검색 완료 (collection 총 {total}개)",
    })
    return result


# ── 검색 결과 CSV 저장 (덮어쓰기, 최신 검색 스냅샷) ──────────
def save_search_results(question: str, results, path: str = SEARCH_RESULTS_PATH) -> str:
    """검색 결과를 CSV 로 저장한다 (요구 컬럼 + 어떤 질문이었는지 query). 저장 경로 반환."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = ["query"] + [f.name for f in fields(SearchResult)]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({"query": question, **asdict(r)})
    return path


# ── 평가 질문 ────────────────────────────────────────────────
def load_eval_questions(path: str = EVAL_QUESTIONS_PATH) -> list:
    """eval/questions.yaml 을 읽어 질문 목록(list of dict)을 돌려준다. 없으면 빈 목록."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("questions", []) or []


def expected_source_hit(results, expected_source: str) -> bool:
    """Top-K 결과의 source 중에 expected_source 가 포함되는지(Y/N) 확인한다."""
    if not expected_source:
        return False
    target = expected_source.strip().lower()
    for r in results:
        src = (r.source or "").strip().lower()
        # expected_source 를 짧게 적어도 매칭되도록 target 이 저장 source 에 포함되면 인정.
        # 반대 방향(src in target)은 오탐 위험이 있어 넣지 않는다.
        if src == target or target in src:
            return True
    return False
