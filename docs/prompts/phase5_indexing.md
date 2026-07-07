# Phase 5 — Indexing (Embedding + Chroma Vector DB) (수업용 프롬프트)

## 목표
`outputs/chunk_report.csv` 의 **실제 Chunk 본문**을 OpenAI Embedding 으로 바꿔
본문 + 메타데이터를 Chroma Vector DB 에 저장한다. 로직은 `rag/index.py` 로 분리한다.

## 원칙
- **먼저 계획과 파일 구조를 제안하고, 승인 후 구현한다.**
- `app.py` 는 화면·버튼만, 인덱싱은 `rag/index.py`.
- API Key 는 `python-dotenv` + `os.getenv("OPENAI_API_KEY")` 만 사용.

## 프롬프트 (예시)
> Phase 5 로, Chunk 를 Embedding 하여 Chroma 에 저장하는 Indexing 을 만듭니다.
>
> - 입력 `outputs/chunk_report.csv` (chunk_id / chunk_text 본문 컬럼 확인, preview 뿐이면 안내)
> - Vector DB = Chroma, 저장 위치 `chroma_db/`, collection=`rag_docs`, 거리=cosine
> - Embedding 모델명은 상단 상수로 분리(기본 `text-embedding-3-small`)
> - 기존 chroma_db 재생성 옵션(체크박스)
> - 메타데이터 함께 저장: source, file_type, parser_type, page, readiness_status, warning, chunk_id, chunk_index, token_count, char_count
> - 결과 `outputs/vector_db_report.csv`
> - 화면: 생성 버튼, 읽은/저장된 Chunk 수, 모델명, collection, 저장 위치, CSV 저장 안내, "검색은 다음 Phase" 안내
> - 보안: `chroma_db/` 는 `.gitignore` 에 포함
>
> **먼저 계획을 제안하고, 승인하면 구현해주세요.**

## 이번 Phase 에서 구현하지 않을 것
- Retriever 검색, RAG 답변 생성, Source Citation
