# Phase 6.5 — 문서화 & 보안 체크포인트 (수업용 프롬프트)

## 목표
Phase 1~6 구현 상태를 **강의자료 / GitHub 커밋용**으로 정리하고, 민감정보가 올라가지 않도록 **보안 점검**한다.
(코드 기능 추가가 아니라 문서화 + 보안 정비 단계)

## 원칙
- **먼저 계획과 변경 예정 파일을 제안하고, 승인 후 생성한다.**
- **이 단계에서 `git commit` / `git push` 는 실행하지 않는다.** (커밋은 사용자가 별도로 진행)

## 프롬프트 (예시)
> Phase 1~6 구현 상태를 문서화하고 보안 점검을 합니다.
>
> 문서화:
> - `README.md`: 소개 / 구현된 Phase 1~6 / 미구현 / 실행법 / 주요 모듈 / 산출물 / 보안 / 다음 단계
> - `FLOW.md`: Phase 별 목표·입력·처리·출력·아직 안 함
> - `docs/prompts/` 에 Phase 별 수업용 프롬프트 정리
>
> 프롬프트 정리 규칙:
> - 실제 API Key·개인정보·내부 문서명 금지, 문서명은 `sample_document.pdf` 로 일반화
> - 실제 `eval/questions.yaml` 은 제외, `questions.example.yaml` 만 커밋
> - 각 프롬프트에 "먼저 계획 제안 → 승인 후 구현" 원칙과 "이번 Phase 에서 안 하는 것" 포함
>
> 보안 점검:
> - `.gitignore` 확인/보강, `git check-ignore` 로 `.env` / `chroma_db/` / `outputs/` / `docs/images/` / `eval/questions.yaml` 제외 확인
> - `.env.example` 은 커밋 가능(필요 시 `.env.*` + `!.env.example`)
> - 코드/문서에 API Key·`sk-`·개인정보·문서 원문 하드코딩 없는지 확인
> - 스크린샷(`docs/images/`)은 로컬 전용 안내
>
> 마지막으로 커밋 가능/불가 파일, 수정 파일, 추천 커밋 메시지를 표로 정리.
> **먼저 계획을 제안하고, 승인하면 파일을 생성해주세요. (commit/push 는 하지 않음)**

## 이번 단계에서 하지 않을 것
- `git commit` / `git push` 실행
- Phase 7 이후 기능(RAG 답변 생성, Citation 등)
