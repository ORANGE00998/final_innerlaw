# Phase 3 — Readiness Gate (RAG 투입 가능 여부 판정) (수업용 프롬프트)

## 목표
`outputs/ingestion_report.csv` 를 읽어 각 문서/페이지를 **Ready / Partial / Blocked** 로 판정한다.
판정 로직을 `rag/readiness.py` 로 분리한다.

## 원칙
- **먼저 계획과 파일 구조를 제안하고, 승인 후 구현한다.**
- `app.py` 는 화면·버튼만, 판정은 `rag/readiness.py`.

## 프롬프트 (예시)
> Phase 3 로, 진단 결과를 바탕으로 RAG 다음 단계로 넘겨도 되는지 판정하는 Readiness Gate 를 만듭니다.
>
> 판정 기준:
> - text_length 충분 + warning 없음 → Ready
> - text 어느 정도 있으나 warning 있음 → Partial
> - text 매우 짧거나 scanned=True → Blocked
> - 변환 필요 문서(HWP 등) → warning 에 따라 Partial/Blocked, 이미지 → Blocked
>
> 생성 컬럼: source, file_type, parser_type, page, text_length, scanned, warning,
> readiness_status, rag_ready, needs_ocr, needs_vision, needs_conversion, reason
> 입력 `outputs/ingestion_report.csv` → 출력 `outputs/readiness_report.csv`.
> 화면: Ready/Partial/Blocked 개수, 상태별 목록, Blocked 제외 안내, reason/warning, CSV 저장 안내.
>
> **먼저 계획을 제안하고, 승인하면 구현해주세요.**

## 이번 Phase 에서 구현하지 않을 것
- Chunking, Embedding, Vector DB, Retriever
