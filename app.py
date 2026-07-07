# app.py
# 역할: Streamlit 화면 표시와 사용자 입력을 담당하는 진입점.
# - 문서 업로드를 받아 rag/ingestion.py 로 "형식 진단 + 텍스트 추출"을 맡기고,
#   그 결과(진단 표 · 경고 · 미리보기)를 화면에 보여준다.
# - 추출된 텍스트로 기존 Baseline Q&A(문서 기반 질의응답)를 수행한다.
# - 문서 처리 로직 자체는 이 파일에 두지 않는다 (rag/ingestion.py 담당).
# - Chunking / Embedding / Vector DB / Retriever 는 아직 사용하지 않는다(다음 단계).

import os
import sys
import logging

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken  # 토큰 수 계산(추정)용

# 문서 처리(형식 진단 + 텍스트 추출)는 rag/ingestion.py 가 담당한다
from rag.ingestion import ingest_document, save_report, save_extracted_text, REPORT_PATH
# RAG 투입 가능 여부 판정은 rag/readiness.py 가 담당한다
from rag.readiness import evaluate_readiness
# 문서 Chunking + 메타데이터 설계는 rag/chunking.py 가 담당한다
from rag.chunking import run_chunking
# Embedding + Chroma Vector DB 저장은 rag/index.py 가 담당한다
from rag.index import build_index, EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_DIR
# Top-K 검색(Retrieval)은 rag/retriever.py 가 담당한다
from rag.retriever import (
    search, save_search_results, load_eval_questions, expected_source_hit, DEFAULT_K,
)


# ── 기본 설정 ────────────────────────────────────────────────
MODEL_NAME = "gpt-5.4-mini"  # 사용할 OpenAI 모델

load_dotenv()                       # .env 파일에서 환경변수를 읽어온다
API_KEY = os.getenv("OPENAI_API_KEY")  # 보안 규칙: API Key 는 .env 에서만 읽는다

# 서버(터미널) 로그 설정 — Streamlit 은 스크립트를 매번 위에서 아래로 다시 실행하므로,
# 핸들러가 중복 등록되지 않도록 이미 있으면 다시 추가하지 않는다.
logger = logging.getLogger("doc_qa")
if not logger.handlers:
    # 한글/특수문자가 깨지지 않도록 표준 에러를 UTF-8 로 맞춘다 (가능한 경우에만).
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()  # streamlit run 을 실행한 터미널(서버)로 출력
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.propagate = False


# ── 화면 기본 구성 ────────────────────────────────────────────
st.set_page_config(page_title="문서 Q&A (Ingestion)", page_icon="📄")
st.title("📄 문서 Q&A — Ingestion")
st.caption("문서를 업로드하면 형식을 진단하고 텍스트를 추출한 뒤, 그 내용을 바탕으로 답변합니다.")

# API Key 가 없으면 더 진행하지 않고 안내한다
if not API_KEY:
    st.error(
        "OPENAI_API_KEY 를 찾을 수 없습니다.\n\n"
        "프로젝트 폴더의 .env 파일에 아래 한 줄을 넣어주세요.\n"
        "OPENAI_API_KEY=여기에_본인_키"
    )
    st.stop()

client = OpenAI(api_key=API_KEY)


# ── 토큰 수 계산(추정) ───────────────────────────────────────
@st.cache_resource
def get_encoder():
    """토큰 수를 세기 위한 tiktoken 인코더를 준비한다 (한 번만 생성해 재사용)."""
    try:
        return tiktoken.encoding_for_model(MODEL_NAME)  # 모델 전용 인코딩이 있으면 사용
    except KeyError:
        return tiktoken.get_encoding("o200k_base")      # 없으면 최신 계열 기본값(추정치)


def count_tokens(text: str) -> int:
    """주어진 텍스트가 대략 몇 토큰인지 계산한다 (추정치). 계산 실패 시 -1 을 돌려준다."""
    try:
        return len(get_encoder().encode(text))
    except Exception:
        return -1


# ── OpenAI 에게 질문을 보내는 함수 ────────────────────────────
def ask_openai(document_text: str, question: str):
    """문서 내용과 사용자 질문을 함께 넣어 모델의 답변과 토큰 사용량을 받는다.

    반환값: (답변 문자열, usage 객체)  ← usage 로 실제 사용된 토큰을 알 수 있다.
    """
    system_prompt = (
        "너는 업로드된 문서를 근거로 사용자의 질문에 답하는 도우미다. "
        "문서에 있는 내용만 바탕으로 정확하게 답하고, "
        "문서에 없는 내용은 추측하지 말고 '문서에서 찾을 수 없습니다'라고 답하라. "
        "답변은 한국어로 한다."
    )
    # 요구사항: 문서 내용과 사용자 질문을 함께 프롬프트에 넣는다
    user_prompt = f"[문서]\n{document_text}\n\n[질문]\n{question}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = response.choices[0].message.content or "(빈 응답을 받았습니다.)"
    # 답변과 함께 토큰 사용량(usage)도 돌려준다 → 호출부에서 서버 로그로 남긴다
    return answer, response.usage


# ── 사이드바: 문서 업로드 & 진단 결과 표시 ────────────────────
# 업로드된 문서와 관련된 session_state 키들 (업로드 해제 시 함께 비운다)
DOC_STATE_KEYS = ("doc_key", "doc_text", "ingest_records", "saved_path")

with st.sidebar:
    st.header("📎 문서 업로드")
    uploaded_file = st.file_uploader(
        "PDF, TXT, DOCX, HWP, HWPX, 이미지 파일을 올려주세요",
        type=["pdf", "txt", "docx", "hwp", "hwpx",
              "png", "jpg", "jpeg", "gif", "bmp", "webp", "tif", "tiff"],
    )

    if uploaded_file is not None:
        file_key = (uploaded_file.name, uploaded_file.size)
        # 같은 파일을 매 입력마다 다시 처리/저장하지 않도록, "새 파일"일 때만 진단한다.
        if st.session_state.get("doc_key") != file_key:
            data = uploaded_file.getvalue()
            result = ingest_document(uploaded_file.name, data)  # 형식 진단 + 텍스트 추출
            saved_path = save_report(result.records)            # 진단 결과 CSV 누적 저장
            save_extracted_text(uploaded_file.name, result.page_texts)  # 전체 텍스트 저장(청킹용)

            st.session_state["doc_key"] = file_key
            st.session_state["doc_text"] = result.text          # Q&A 에 쓸 전체 텍스트
            st.session_state["ingest_records"] = result.records
            st.session_state["saved_path"] = saved_path

            # 서버 로그: 진단 요약 + 추정 토큰 + CSV 경로
            total_len = sum(r.text_length for r in result.records)
            est_tokens = count_tokens(result.text)
            logger.info(
                "문서 진단: %s | 형식=%s | 레코드=%d | 총 문자=%d | 추정 토큰 약 %s개 | CSV=%s",
                uploaded_file.name,
                result.records[0].file_type,
                len(result.records),
                total_len,
                (f"{est_tokens:,}" if est_tokens >= 0 else "추정 불가"),
                saved_path,
            )
            warned = list(dict.fromkeys(r.warning for r in result.records if r.warning))
            if warned:
                logger.info("경고 %d건: %s", len(warned), " / ".join(warned))

        # ── 진단 결과 화면 표시 ──
        records = st.session_state.get("ingest_records")
        if records:
            first = records[0]
            total_len = sum(r.text_length for r in records)

            st.success("문서 진단 완료")
            st.markdown(f"- **파일명**: {first.source}")
            st.markdown(f"- **형식**: {first.file_type}")
            st.markdown(f"- **추출 텍스트 길이**: {total_len:,} 자")
            if len(records) > 1:
                st.markdown(f"- **페이지 수**: {len(records)}")

            # 경고가 있으면 눈에 띄게 표시 (중복 제거)
            for warning in dict.fromkeys(r.warning for r in records if r.warning):
                st.warning(f"⚠️ {warning}")

            # CSV 저장 안내
            st.caption(f"🧾 진단 결과 저장됨 → `{st.session_state.get('saved_path', REPORT_PATH)}`")

            # content_preview 를 접어서 볼 수 있게 한다
            with st.expander("📑 페이지별 진단 · 내용 미리보기"):
                for r in records:
                    page_label = f"p.{r.page}" if r.file_type == "PDF" else "전체"
                    line = f"**[{page_label}]** {r.text_length:,}자 · `{r.parser_type}`"
                    if r.scanned:
                        line += " · 🔍 scanned"
                    st.markdown(line)
                    if r.content_preview:
                        st.text(r.content_preview)
                    else:
                        st.caption("(미리볼 텍스트 없음)")
                    st.divider()
    else:
        # 업로드를 해제하면 저장해 둔 문서 정보도 함께 비운다
        for key in DOC_STATE_KEYS:
            st.session_state.pop(key, None)


# ── RAG Readiness Gate (판정) ─────────────────────────────────
# 판정 로직은 rag/readiness.py 가 담당하고, 여기서는 "버튼 실행 + 결과 표시"만 한다.
st.header("🚦 RAG Readiness Gate")
st.caption("Phase 2 진단 결과(outputs/ingestion_report.csv)를 읽어 RAG 투입 가능 여부를 판정합니다.")

if st.button("📋 판정 실행 (ingestion_report.csv 평가)"):
    records, summary, saved_path = evaluate_readiness()
    st.session_state["readiness"] = {
        "records": records, "summary": summary, "saved_path": saved_path,
    }
    logger.info(
        "Readiness 판정: Ready=%d Partial=%d Blocked=%d | CSV=%s",
        summary["Ready"], summary["Partial"], summary["Blocked"], saved_path,
    )

readiness = st.session_state.get("readiness")
if readiness:
    records = readiness["records"]
    summary = readiness["summary"]

    if not records:
        st.info("판정할 문서가 없습니다. 먼저 문서를 업로드해 진단 결과를 만들어 주세요.")
    else:
        # (요구 1) Ready / Partial / Blocked 개수 요약
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Ready", summary["Ready"])
        col2.metric("🟡 Partial", summary["Partial"])
        col3.metric("⛔ Blocked", summary["Blocked"])

        # (요구 3) Blocked 문서는 다음 단계로 넘기지 않는다는 안내
        if summary["Blocked"] > 0:
            st.error(
                f"⛔ Blocked 문서 {summary['Blocked']}건은 다음 단계(RAG)로 넘기지 않습니다. "
                "OCR/Vision 또는 형식 변환 후 다시 진단해 주세요."
            )

        # (요구 5) readiness_report.csv 저장 안내
        st.caption(f"🧾 판정 결과 저장됨 → `{readiness['saved_path']}`")

        # (요구 2, 4) 상태별로 나누어 표시 + warning/reason 을 사람이 읽기 쉽게
        status_style = {
            "Ready": ("✅ Ready", "다음 단계로 진행 가능"),
            "Partial": ("🟡 Partial", "검토 후 진행 권장"),
            "Blocked": ("⛔ Blocked", "다음 단계로 넘기지 않음"),
        }
        for status in ("Ready", "Partial", "Blocked"):
            group = [r for r in records if r.readiness_status == status]
            if not group:
                continue
            label, note = status_style[status]
            # Partial/Blocked 는 눈에 잘 띄도록 기본으로 펼쳐 둔다
            with st.expander(f"{label} — {len(group)}건 · {note}", expanded=(status != "Ready")):
                for r in group:
                    st.markdown(f"**{r.source}** · p.{r.page} · `{r.file_type}` · {r.text_length:,}자")
                    st.markdown(f"↳ {r.reason}")
                    if r.warning:
                        st.caption(f"⚠️ warning: {r.warning}")
                    needs = [name for flag, name in (
                        (r.needs_ocr, "OCR"),
                        (r.needs_vision, "Vision"),
                        (r.needs_conversion, "변환"),
                    ) if flag]
                    if needs:
                        st.caption("필요: " + ", ".join(needs))
                    st.divider()

st.divider()


# ── Chunking & Metadata (Phase 4) ─────────────────────────────
# 청킹 로직은 rag/chunking.py 가 담당하고, 여기서는 "파라미터 + 버튼 실행 + 결과 표시"만 한다.
st.header("🧩 Chunking & Metadata")
st.caption("Readiness 결과에서 Ready/Partial 문서만 골라 토큰 기준으로 Chunk 를 만듭니다. (Blocked 제외)")

col_size, col_overlap = st.columns(2)
chunk_size = col_size.selectbox("chunk_size (토큰)", [400, 800, 1200], index=1)
chunk_overlap = col_overlap.number_input(
    "chunk_overlap (토큰)", min_value=0, max_value=500, value=100, step=10
)

if st.button("🧩 Chunking 실행"):
    records, summary, excluded, saved_path = run_chunking(
        chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
    )
    st.session_state["chunking"] = {
        "records": records, "summary": summary, "excluded": excluded,
        "saved_path": saved_path,
        "chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap),
    }
    logger.info(
        "Chunking: size=%d overlap=%d | chunks=%d | 제외=%d | CSV=%s",
        int(chunk_size), int(chunk_overlap), summary["total"], excluded, saved_path,
    )

chunking = st.session_state.get("chunking")
if chunking:
    summary = chunking["summary"]
    records = chunking["records"]

    # (요구 8) Blocked 등 제외 안내
    if chunking["excluded"] > 0:
        st.info(
            f"ℹ️ Blocked 등 대상이 아닌 문서/페이지 {chunking['excluded']}건은 "
            "Chunking 에서 제외되었습니다."
        )

    if not records:
        st.warning(
            "생성된 Chunk 가 없습니다. 문서를 업로드·진단·판정한 뒤, "
            "Ready/Partial 문서가 있는지 확인해 주세요. "
            "(이 기능 추가 이전에 업로드한 문서는 전체 텍스트가 없어 다시 업로드해야 합니다.)"
        )
    else:
        # (요구 2) 총 청크 수 + 설정
        st.markdown(
            f"**총 Chunk 수: {summary['total']:,}개** "
            f"(size={chunking['chunk_size']} · overlap={chunking['chunk_overlap']} 토큰)"
        )

        # (요구 3) source 별 청크 수
        st.markdown("**source 별 Chunk 수**")
        for src, count in summary["per_source"].items():
            st.markdown(f"- `{src}` : {count:,}개")

        # (요구 7) CSV 저장 안내
        st.caption(f"🧾 Chunk 결과 저장됨 → `{chunking['saved_path']}`")

        # (요구 5, 6) Chunk Preview — source·page·chunk_id·token_count·warning 함께 표시
        preview_count = min(20, len(records))
        with st.expander(f"🔎 Chunk Preview (앞 {preview_count}개)", expanded=True):
            for r in records[:preview_count]:
                st.markdown(
                    f"**{r.chunk_id}** · `{r.source}` p.{r.page} · "
                    f"{r.token_count} tok / {r.char_count:,}자"
                )
                if r.warning:
                    st.caption(f"⚠️ warning: {r.warning}")
                snippet = r.chunk_text[:200].replace("\n", " ")
                st.text(snippet + ("…" if len(r.chunk_text) > 200 else ""))
                st.divider()

st.divider()


# ── Vector DB Indexing (Phase 5) ──────────────────────────────
# Embedding + Chroma 저장 로직은 rag/index.py 가 담당하고, 여기서는 버튼/옵션/결과 표시만 한다.
st.header("🗄️ Vector DB Indexing (Embedding → Chroma)")
st.caption(f"chunk_report.csv 의 Chunk 본문을 Embedding 하여 Chroma collection '{COLLECTION_NAME}' 에 저장합니다.")

# (요구 8) 기존 DB 재생성 옵션 체크박스
recreate = st.checkbox("기존 chroma_db 재생성 (기존 collection 삭제 후 새로 생성)", value=False)

# (요구 1) Vector DB 생성 버튼
if st.button("🗄️ Vector DB 생성"):
    with st.spinner("Embedding 및 저장 중... (문서 양에 따라 시간이 걸릴 수 있어요)"):
        vdb_result = build_index(recreate=recreate)
    st.session_state["vdb"] = vdb_result
    logger.info(
        "Vector DB: ok=%s read=%d stored=%d model=%s collection=%s dir=%s",
        vdb_result.get("ok"), vdb_result.get("read_count", 0),
        vdb_result.get("stored_count", 0), vdb_result.get("model"),
        vdb_result.get("collection"), vdb_result.get("chroma_dir"),
    )

vdb = st.session_state.get("vdb")
if vdb:
    if not vdb.get("ok"):
        # (확인 3) 본문이 없거나 chunk_report 문제 시 안내
        st.warning(f"⚠️ {vdb.get('message')}")
    else:
        st.success(vdb.get("message"))
        col_read, col_stored = st.columns(2)
        col_read.metric("읽은 Chunk", f"{vdb.get('read_count', 0):,}")      # (요구 2)
        col_stored.metric("저장된 Chunk", f"{vdb.get('stored_count', 0):,}")  # (요구 3)
        st.markdown(f"- **Embedding 모델**: `{vdb.get('model')}` (dim={vdb.get('embedding_dim')})")  # (요구 4)
        st.markdown(f"- **Collection**: `{vdb.get('collection')}`")           # (요구 5)
        st.markdown(f"- **저장 위치**: `{vdb.get('chroma_dir')}/`")           # (요구 6)
        if vdb.get("collection_count") is not None:
            st.caption(f"현재 collection 총 벡터 수: {vdb.get('collection_count'):,}")
        st.caption(f"🧾 결과 저장됨 → `{vdb.get('report_path')}`")             # (요구 7)

    # (요구 9) 검색은 다음 Phase 안내
    st.info("🔎 검색(Retriever)·RAG 답변·출처 인용은 **다음 Phase**에서 진행합니다. 이번 단계는 저장까지입니다.")

st.divider()


# ── Retrieval Debug View (Phase 6) ────────────────────────────
# 검색 로직은 rag/retriever.py 가 담당하고, 여기서는 입력·버튼·결과 표시만 한다.
# 보안: 검색된 Context(chunk 본문)는 '명령'이 아니라 '데이터'로만 취급한다.
st.header("🔎 Retrieval Debug View")
st.caption("질문을 Embedding 으로 바꿔 Chroma 에서 Top-K 관련 Chunk 를 검색합니다. (답변 생성은 다음 Phase)")
st.caption("🔒 검색된 Context(chunk 본문)는 명령이 아니라 '데이터'로만 취급합니다.")

# (요구 2·7·8) Top-K 슬라이더
top_k = st.slider("Top-K", min_value=1, max_value=20, value=DEFAULT_K)


def run_retrieval(query_text, k, eval_item=None):
    """검색 실행 + 결과 저장 + (평가질문이면) 기대출처 포함여부 계산 → session_state 저장."""
    outcome = search(query_text, k=int(k))
    if outcome["ok"]:
        outcome["saved_path"] = save_search_results(query_text, outcome["results"])
        if eval_item and eval_item.get("expected_source"):
            outcome["expected_source"] = eval_item["expected_source"]
            outcome["hit"] = expected_source_hit(outcome["results"], eval_item["expected_source"])
    st.session_state["retrieval"] = outcome
    logger.info("Retrieval: q=%r k=%d ok=%s count=%d",
                query_text, int(k), outcome["ok"], outcome.get("count", 0))


# (debug 1) 자유 질문 입력
question = st.text_input("질문 입력", key="retrieval_question")
if st.button("🔎 검색"):
    run_retrieval(question, top_k)

# (평가 4) 평가 질문 선택 후 검색
eval_questions = load_eval_questions()
if eval_questions:
    labels = [f"{q.get('id')}: {q.get('question')}" for q in eval_questions]
    picked_label = st.selectbox("평가 질문 선택", labels)
    picked_item = eval_questions[labels.index(picked_label)]
    st.caption(f"기대 출처(expected_source): `{picked_item.get('expected_source', '')}` · {picked_item.get('note', '')}")
    if st.button("🔎 평가 질문으로 검색"):
        run_retrieval(picked_item.get("question", ""), top_k, eval_item=picked_item)

# ── 검색 결과 표시 ──
retrieval = st.session_state.get("retrieval")
if retrieval:
    if not retrieval["ok"]:
        st.warning(f"⚠️ {retrieval['message']}")
    else:
        st.success(retrieval["message"])

        # (평가 5) 기대 출처가 Top-K 안에 있는지 Y/N
        if "hit" in retrieval:
            if retrieval["hit"]:
                st.success(f"✅ 기대 출처 포함 (Y) — `{retrieval['expected_source']}` 가 Top-{retrieval['count']} 에 있음")
            else:
                st.error(f"❌ 기대 출처 미포함 (N) — `{retrieval['expected_source']}` 가 Top-{retrieval['count']} 에 없음")

        # (요구 8) CSV 저장 안내
        if retrieval.get("saved_path"):
            st.caption(f"🧾 검색 결과 저장됨 → `{retrieval['saved_path']}`")

        # (요구 3·4·5·6·7) 순위별 검색 결과
        for r in retrieval["results"]:
            st.markdown(f"**#{r.rank}** · `{r.source}` p.{r.page} · "
                        f"score={r.score} (distance={r.distance})")     # (요구 6)
            st.caption(f"chunk_id: `{r.chunk_id}`")                      # (요구 4)
            if r.warning:                                               # (요구 5) 눈에 띄게
                st.warning(f"⚠️ warning: {r.warning}")
            with st.expander("preview 보기"):                            # (요구 7) 접어서
                st.text(r.preview)
            st.divider()

    # (요구 9) 답변 생성은 다음 Phase 안내
    st.info("💬 최종 RAG 답변·출처 인용은 **다음 Phase**에서 진행합니다. 이번 단계는 검색 결과 확인까지입니다.")

st.divider()


# ── 대화 기록 표시 ────────────────────────────────────────────
# messages: [{"role": "user"/"assistant", "content": "..."}] 형태로 대화를 저장한다
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── 질문 입력 & 답변 생성 ─────────────────────────────────────
question = st.chat_input("문서에 대해 궁금한 점을 물어보세요")

if question:
    # 1) 사용자 질문을 기록하고 화면에 표시한다
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    logger.info("질문 수신: %s", question)  # 서버 로그

    doc_text = st.session_state.get("doc_text")
    has_upload = bool(st.session_state.get("ingest_records"))

    # 2) 사용할 텍스트가 없으면 API 를 호출하지 않고 안내만 한다 (토큰 사용 없음)
    if not doc_text:
        if has_upload:
            # 파일은 올렸지만 텍스트를 뽑지 못한 경우 (스캔 PDF·이미지·HWP 등)
            answer = ("이 문서에서는 텍스트를 추출하지 못했어요. "
                      "OCR 또는 Vision 이 필요한 파일(스캔 PDF·이미지·HWP 등)일 수 있습니다.")
            logger.info("업로드됨·텍스트 없음 -> 안내만 표시 (API 호출 안 함, 토큰 0)")
        else:
            answer = "먼저 문서를 업로드해주세요."
            logger.info("문서 없이 질문 -> 안내만 표시 (API 호출 안 함, 토큰 0)")
        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        # 3) 텍스트가 있으면 문서 + 질문을 함께 넣어 모델에게 답변을 받는다
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중..."):
                try:
                    answer, usage = ask_openai(doc_text, question)
                    # 서버 로그: 이번 호출에서 실제로 사용된 토큰 + 세션 누적 합계
                    if usage is not None:
                        st.session_state["total_tokens"] = (
                            st.session_state.get("total_tokens", 0) + usage.total_tokens
                        )
                        logger.info(
                            "토큰 사용 - prompt=%d, completion=%d, total=%d",
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            usage.total_tokens,
                        )
                        logger.info("세션 누적 토큰: %d", st.session_state["total_tokens"])
                except Exception as error:
                    answer = (
                        "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n"
                        f"(자세한 내용: {error})"
                    )
                    logger.exception("답변 생성 실패: %s", error)
            st.markdown(answer)

    # 4) AI 답변(또는 안내)을 대화 기록에 저장한다
    st.session_state["messages"].append({"role": "assistant", "content": answer})
