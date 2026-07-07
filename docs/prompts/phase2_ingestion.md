# Phase 2 — Ingestion (형식 진단 + 텍스트 추출) (수업용 프롬프트)

## 목표
업로드한 문서의 **형식을 진단**하고 텍스트 추출 가능 여부와 warning 을 보여준다.
문서 처리 로직을 `rag/ingestion.py` 로 분리한다.

## 원칙
- **먼저 계획과 파일 구조를 제안하고, 승인 후 구현한다.**
- `app.py` 는 화면·입력만, 문서 처리는 `rag/ingestion.py`.

## 프롬프트 (예시)
> Phase 2 로, 문서 형식 진단과 Ingestion 기능을 추가합니다.
> 지원 파일: PDF, TXT, DOCX, HWP, HWPX, 이미지.
>
> 처리 규칙:
> - PDF = PyMuPDF4LLM 으로 Markdown 추출(페이지별), 텍스트가 매우 짧으면 `scanned=True` + "OCR 또는 Vision 필요 가능성"
> - TXT = 그대로 읽기 / DOCX = python-docx
> - HWP = 직접 파싱하지 말고 "HWPX/PDF/DOCX 변환 권장"
> - HWPX = 구조 확인 시도, 복잡하면 "HWPX 구조 확인 필요"
> - 이미지 = 추출하지 말고 "OCR 또는 Vision 필요"
>
> 각 청크/페이지 메타: source, file_type, parser_type, page, text_length, scanned, warning, content_preview
> 결과를 `outputs/ingestion_report.csv` 에 저장. 화면에 파일명·형식·길이·warning·preview(접기)·CSV 저장 안내.
>
> **먼저 계획을 제안하고, 승인하면 구현해주세요.**

## 이번 Phase 에서 구현하지 않을 것
- Chunking, Embedding, Vector DB, Retriever
