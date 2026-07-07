# Phase 1 — Baseline 문서 Q&A (수업용 프롬프트)

## 목표
문서를 업로드하면 전체 텍스트를 추출하고, 질문과 문서 내용을 함께 프롬프트에 넣어
OpenAI 모델이 답하는 **Long-Context Q&A** 앱을 만든다.

## 원칙
- **먼저 구현 계획과 파일 구조를 제안하고, 승인 후 구현한다.**
- `app.py` 하나로 시작 / Streamlit 사용 / API Key 는 `.env` 의 `OPENAI_API_KEY` 에서만 읽는다.

## 프롬프트 (예시)
> 이번 프로젝트는 RAG 실습 프로젝트입니다. Phase 1 로, Baseline 문서 업로드 Q&A 앱을 만들려고 합니다.
> 사용자가 문서를 업로드하면 텍스트를 추출하고, 질문과 문서 내용을 함께 프롬프트에 넣어 답하게 해주세요.
>
> 요구사항:
> - `app.py` 하나에서 동작 / Streamlit
> - 사이드바 파일 업로드(PDF, TXT, DOCX), PDF=pypdf, DOCX=python-docx, TXT=그대로 읽기
> - `st.chat_input` 으로 질문, `st.chat_message` 로 대화 표시
> - API Key 는 `.env` 의 `OPENAI_API_KEY` 에서만 읽기
> - 문서가 없으면 "먼저 문서를 업로드해주세요" 안내
> - 업로드한 파일명 / 형식 / 추출된 텍스트 길이 표시
> - 오류 시 사용자가 이해할 수 있는 안내 문구
>
> **먼저 계획을 제안하고, 승인하면 구현해주세요.**

## 이번 Phase 에서 구현하지 않을 것
- Chunking, Embedding, Vector DB, Retriever (문서 전체를 그대로 넣는 Long-Context 방식)
