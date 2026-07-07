# Phase 4 — Chunking & Metadata (수업용 프롬프트)

## 목표
Ready/Partial 문서만 골라 **토큰 기준 Chunk** 로 나누고 출처 추적용 **메타데이터**를 붙인다.
청킹 로직을 `rag/chunking.py` 로 분리한다.

## 원칙
- **먼저 계획과 파일 구조를 제안하고, 승인 후 구현한다.**
- `app.py` 는 화면·버튼만, 청킹은 `rag/chunking.py`.

## 프롬프트 (예시)
> Phase 4 로, Readiness 결과에서 Ready/Partial 문서만 Chunk 로 나눕니다. (Blocked 제외)
>
> - `RecursiveCharacterTextSplitter` 사용, **토큰 기준** 분할
> - 기본 chunk_size=800, chunk_overlap=100, chunk_size 는 400/800/1200 을 비교할 수 있게 인자로 분리
> - 입력 `outputs/readiness_report.csv` → 출력 `outputs/chunk_report.csv`
> - 메타: source, file_type, parser_type, page, readiness_status, warning, chunk_id, chunk_index, token_count, char_count
> - Embedding 대상이 되는 **전체 Chunk 본문 컬럼(chunk_text)** 도 저장(미리보기만 저장하지 말 것)
> - 화면: 총 Chunk 수, source 별 개수, chunk_size/overlap 조정, Chunk Preview, Blocked 제외 안내, CSV 저장 안내
>
> **먼저 계획을 제안하고, 승인하면 구현해주세요.**

## 이번 Phase 에서 구현하지 않을 것
- Embedding, Vector DB, Retriever
