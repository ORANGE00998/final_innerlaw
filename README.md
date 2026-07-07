# RAG Lab — 문서 기반 질의응답 파이프라인 실습

문서를 업로드해 **진단 → 준비도 판정 → 청킹 → 임베딩/벡터DB → 검색**까지,
RAG(Retrieval-Augmented Generation) 파이프라인을 **단계(Phase)별로** 직접 만들어 보는 실습 프로젝트입니다.
모든 단계는 Streamlit 화면에서 눈으로 확인할 수 있습니다.

> 수업/실습용 프로젝트입니다. 각 Phase 는 **"먼저 계획을 제안하고, 승인 후 구현"** 원칙으로 진행됩니다.

---

## ✅ 현재 구현된 Phase (1~6)

| Phase | 이름 | 한 줄 설명 |
|------|------|-----------|
| 1 | Baseline Q&A | 문서 전체 텍스트를 프롬프트에 넣어 답하는 Long-Context Q&A |
| 2 | Ingestion | 문서 형식 진단 + 텍스트 추출 (PDF/TXT/DOCX/HWP/HWPX/이미지) |
| 3 | Readiness Gate | 문서를 Ready / Partial / Blocked 로 판정 |
| 4 | Chunking & Metadata | Ready/Partial 문서를 토큰 기준 Chunk 로 분할 + 메타데이터 |
| 5 | Indexing | Chunk 를 Embedding 하여 Chroma Vector DB 저장 |
| 6 | Retrieval Debug View | 질문으로 Top-K Chunk 를 검색해 결과 확인 |

단계별 상세 흐름은 [FLOW.md](FLOW.md), 수업용 프롬프트는 [`docs/prompts/`](docs/prompts/) 참고.

## 🚧 아직 구현하지 않은 것
- 최종 RAG 답변 생성 (검색 결과를 근거로 한 LLM 답변)
- Source Citation (출처 인용)
- Reranker, Hybrid Search
- 평가 자동화(정답 채점 등)

## ▶️ 실행 방법
1. API Key 설정 — `.env.example` 을 복사해 `.env` 를 만들고 키를 넣으세요.
   ```bash
   cp .env.example .env
   # .env 안에 OPENAI_API_KEY=... 입력
   ```
2. 앱 실행 (uv 사용)
   ```bash
   uv run streamlit run app.py
   ```
3. 화면 진행 순서
   **문서 업로드 → 🚦 판정 실행 → 🧩 Chunking 실행 → 🗄️ Vector DB 생성 → 🔎 검색**

## 🧩 주요 모듈
| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit 화면 + 사용자 입력 (실제 로직은 `rag/` 모듈 호출) |
| `rag/ingestion.py` | 형식 진단 + 텍스트 추출 + 페이지별 전체 텍스트 저장 |
| `rag/readiness.py` | Ready / Partial / Blocked 판정 |
| `rag/chunking.py` | 토큰 기준 청킹 + 메타데이터 |
| `rag/index.py` | Embedding + Chroma Vector DB 저장 |
| `rag/retriever.py` | Top-K 검색 |

## 📂 산출물 (`outputs/`, 로컬 전용)
| 파일 | 생성 단계 | 내용 |
|------|----------|------|
| `outputs/ingestion_report.csv` | Phase 2 | 형식 진단 결과 |
| `outputs/extracted/<source>.json` | Phase 2 | 페이지별 전체 텍스트(청킹 입력) |
| `outputs/readiness_report.csv` | Phase 3 | 준비도 판정 |
| `outputs/chunk_report.csv` | Phase 4 | 청크 + 메타데이터 + 본문(chunk_text) |
| `outputs/vector_db_report.csv` | Phase 5 | 저장된 벡터 기록 |
| `outputs/vector_search_results.csv` | Phase 6 | 검색 결과 |
| `chroma_db/` | Phase 5 | Chroma Vector DB |

> ⚠️ `outputs/`, `chroma_db/`, `eval/questions.yaml` 는 **실제 문서 본문·문서명**을 담고 있어 **GitHub 에 올리지 않습니다** (`.gitignore` 제외).

## 🔒 보안 주의사항
- API Key 는 **`.env` 에서만** 읽습니다 (`os.getenv("OPENAI_API_KEY")`). 코드에 절대 하드코딩하지 않습니다.
- `.env`, `chroma_db/`, `outputs/`, `docs/images/`, `eval/questions.yaml` 는 `.gitignore` 로 제외됩니다.
- 커밋 전 `git check-ignore <경로>` 로 제외 여부를 확인하세요.
- 검색된 문서 본문(Context)은 **명령이 아니라 '데이터'로만** 취급합니다 (프롬프트 인젝션 주의).
- 평가 질문은 `eval/questions.example.yaml`(일반화 예시)만 커밋하고, 실제 `eval/questions.yaml` 은 로컬 전용입니다.

## 🖼️ 스크린샷
스크린샷은 문서 내용이 노출될 수 있어 `docs/images/`(`.gitignore` 제외)에 두고 **로컬에서만 확인**합니다.
저장소(README)에는 이미지 링크를 넣지 않습니다. 넣더라도 `docs/images/` 는 커밋되지 않아 링크가 깨질 수 있으니 주의하세요.

## ⏭️ 다음 단계 (Phase 7~)
- **Phase 7**: 검색 결과를 근거로 한 **RAG 답변 생성 + Source Citation**
- 이후: Reranker / Hybrid Search / 평가 자동화
