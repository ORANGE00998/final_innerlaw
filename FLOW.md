# FLOW — RAG Lab 단계별 흐름

각 Phase 의 **목표 / 입력 / 처리 / 출력 / 아직 하지 않는 것** 을 정리합니다.
전체 흐름: `업로드 → 진단 → 판정 → 청킹 → 임베딩·저장 → 검색`

---

## Phase 1 — Baseline Q&A
- **목표**: 문서 전체 텍스트를 프롬프트에 넣어 답하는 가장 단순한 Long-Context Q&A
- **입력**: 업로드한 문서(PDF/TXT/DOCX) 1개 + 사용자 질문
- **처리**: 텍스트 추출 → `[문서] + [질문]` 프롬프트 → OpenAI 답변, 토큰 사용량 서버 로깅
- **출력**: 대화형 답변 화면
- **아직 안 함**: Chunking / Embedding / Vector DB / Retriever

## Phase 2 — Ingestion
- **목표**: 문서 형식 진단 + 텍스트 추출
- **입력**: PDF / TXT / DOCX / HWP / HWPX / 이미지
- **처리**: 형식별 추출(PDF=PyMuPDF4LLM 페이지별, TXT/DOCX/HWPX), 스캔본·변환필요·OCR 필요를 `warning` 으로 표시
- **출력**: `outputs/ingestion_report.csv`(source·page·parser_type·text_length·scanned·warning·preview), `outputs/extracted/<source>.json`(페이지별 전체 텍스트)
- **아직 안 함**: Chunking / Embedding / Vector DB

## Phase 3 — Readiness Gate
- **목표**: 문서를 RAG 다음 단계로 넘겨도 되는지 판정
- **입력**: `outputs/ingestion_report.csv`
- **처리**: 규칙으로 Ready / Partial / Blocked 분류 + `needs_ocr` / `needs_vision` / `needs_conversion` / `reason`
- **출력**: `outputs/readiness_report.csv` (13 컬럼)
- **아직 안 함**: Chunking / Embedding / Vector DB

## Phase 4 — Chunking & Metadata
- **목표**: Ready/Partial 문서를 검색 단위 Chunk 로 분할
- **입력**: `outputs/readiness_report.csv` + `outputs/extracted/*.json`
- **처리**: Blocked 제외, RecursiveCharacterTextSplitter(**토큰 기준**, chunk_size 400/800/1200, overlap 100), 메타데이터 부착
- **출력**: `outputs/chunk_report.csv` (메타 10개 + `chunk_text` 본문)
- **아직 안 함**: Embedding / Vector DB / Retriever

## Phase 5 — Indexing
- **목표**: Chunk 를 Embedding 하여 Vector DB 에 저장
- **입력**: `outputs/chunk_report.csv`
- **처리**: 본문(`chunk_text`) OpenAI Embedding(`text-embedding-3-small`) → Chroma(collection=`rag_docs`, distance=cosine) upsert(배치 저장), 기존 DB 재생성 옵션
- **출력**: `chroma_db/`, `outputs/vector_db_report.csv`
- **아직 안 함**: Retriever 검색 / RAG 답변 / Citation

## Phase 6 — Retrieval Debug View
- **목표**: 질문에 관련된 Chunk 가 실제로 검색되는지 확인
- **입력**: 사용자 질문(또는 `eval/questions.yaml` 평가 질문) + `chroma_db/`
- **처리**: 질문 Embedding → Top-K 검색(기본 k=4, 화면에서 조정) → 순위·distance/score·메타데이터, `expected_source` 포함 여부 Y/N
- **출력**: 검색 결과 화면, `outputs/vector_search_results.csv`
- **아직 안 함**: 최종 RAG 답변 / Source Citation / Reranker / Hybrid Search

---

## 다음 단계 (예정)
- **Phase 7**: 검색 결과를 근거로 한 RAG 답변 생성 + Source Citation
