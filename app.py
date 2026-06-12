"""
도메인 특화 AI
=======================================================
문서 기반 도메인 특화 AI 가드레일

"""
import streamlit as st
import time
import os
import io
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="도메인 특화 AI",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Noto+Sans+KR:wght@300;400;700&display=swap');
:root {
  --bg:#0a0e1a; --surface:#111827; --surface2:#1a2235;
  --border:#1e3050; --accent:#00e5ff; --green:#00ff88;
  --yellow:#ffd600; --red:#ff4444; --text:#e2e8f0; --muted:#64748b;
}
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important;color:var(--text)!important;
  font-family:'Noto Sans KR',sans-serif;
}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
.title{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;color:var(--accent);}
.sub{font-family:'Space Mono',monospace;font-size:0.72rem;color:var(--muted);margin-bottom:2rem;}
.cluster-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:0.8rem 1rem;margin-bottom:0.5rem;}
.cluster-title{font-family:'Space Mono',monospace;font-size:0.62rem;letter-spacing:2px;color:var(--muted);}
.pass {color:#00ff88;font-family:'Space Mono',monospace;font-weight:700;}
.warn {color:#ffd600;font-family:'Space Mono',monospace;font-weight:700;}
.fatal{color:#ff4444;font-family:'Space Mono',monospace;font-weight:700;}
.bar-wrap{background:#1a2235;border-radius:4px;height:6px;margin-top:4px;overflow:hidden;}
[data-testid="stButton"] button{
  background:var(--accent)!important;color:#000!important;
  font-weight:700!important;font-family:'Space Mono',monospace!important;
  border:none!important;border-radius:6px!important;
}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ── 엔진 import ───────────────────────────────────────────────
try:
    from core_ai_v2_engine import CoreAIv2Engine
    ENGINE_OK = True
except Exception as e:
    ENGINE_OK = False
    st.error(f"엔진 로딩 실패: {e}")

try:
    from guardrail_loop import run_guardrail_loop
    LOOP_OK = True
except Exception as e:
    LOOP_OK = False

# ── 세션 초기화 ───────────────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = CoreAIv2Engine() if ENGINE_OK else None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "train_stats" not in st.session_state:
    st.session_state.train_stats = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "guideline_hint" not in st.session_state:
    st.session_state.guideline_hint = ""

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 도메인 특화 AI")
    st.markdown("---")

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY",""),
        type="password",
        placeholder="sk-...",
    )
    model = st.selectbox("모델", ["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"])

    st.markdown("---")
    st.markdown("### 📚 가이드라인 코퍼스")

    corpus_file = st.file_uploader(
        "코퍼스 업로드",
        type=["txt","pdf","docx","xlsx"],
        help="이 문서가 도메인 기준",
    )

    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.select_slider("클러스터 수", [3,4,5,6,7], value=5)
    with col2:
        emb_epochs = st.select_slider("학습 강도", [5,10,15,20], value=10)

    logp_thr = st.slider("감지 임계값", -15.0, -5.0, -11.5, 0.5)
    max_retry = st.slider("최대 재생성 횟수", 1, 5, 3)

    if corpus_file and st.button("🚀 v2 학습", use_container_width=True):
        with st.spinner("AI 학습 중..."):
            try:
                # 파일 읽기
                name = corpus_file.name.lower()
                if name.endswith(".pdf"):
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(corpus_file.read()))
                    text = "\n".join(p.extract_text() or "" for p in reader.pages)
                elif name.endswith(".docx"):
                    import docx
                    doc = docx.Document(io.BytesIO(corpus_file.read()))
                    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
                elif name.endswith(".xlsx"):
                    import pandas as pd
                    df = pd.read_excel(io.BytesIO(corpus_file.read()))
                    text = "\n".join(
                        " ".join(str(v) for v in row if str(v)!="nan")
                        for _,row in df.iterrows()
                    )
                else:
                    text = corpus_file.read().decode("utf-8", errors="ignore")

                st.session_state.guideline_hint = text[:500]

                # 코퍼스 정제
                from korean_tokenizer import clean_corpus
                text_clean = clean_corpus(text)
                n_before = len([l for l in text.split('\n') if l.strip()])
                n_after  = len([l for l in text_clean.split('\n') if l.strip()])
                if n_after < n_before:
                    st.caption(f"코퍼스 정제: {n_before}줄 → {n_after}줄 ({n_before-n_after}줄 제거)")
                text = text_clean if text_clean.strip() else text

                prog = st.progress(0)
                status_txt = st.empty()

                def cb(pct, msg):
                    prog.progress(pct)
                    status_txt.text(msg)

                st.session_state.engine = CoreAIv2Engine(n_clusters=n_clusters)
                stats = st.session_state.engine.train(
                    text, emb_epochs=emb_epochs, on_progress=cb
                )
                st.session_state.trained = True
                st.session_state.train_stats = stats
                prog.progress(100)
                st.success(
                    f"✓ 학습 완료 ({stats['vocab_size']}어휘 | "
                    f"{stats['total_ms']:.0f}ms)"
                )

                # 자동 저장
                try:
                    import pickle
                    engine_bytes = pickle.dumps({
                        "n_clusters":      st.session_state.engine.n_clusters,
                        "global_vocab":    st.session_state.engine.global_vocab,
                        "corpus_name":     corpus_file.name,
                        "train_stats":     stats,
                        "emb_emb":         st.session_state.engine.embedder.emb,
                        "emb_vocab":       st.session_state.engine.embedder.vocab,
                        "emb_dim":         st.session_state.engine.embedder.dim,
                        "cluster_sentences": dict(st.session_state.engine.decomposer.cluster_sentences),
                        "cluster_tokens":    dict(st.session_state.engine.decomposer.cluster_tokens),
                        "cluster_keywords":  st.session_state.engine.decomposer.cluster_keywords,
                        "decomp_vocab":      st.session_state.engine.decomposer.vocab,
                        "decomp_W":          st.session_state.engine.decomposer.W,
                        "markovs": {
                            k: {
                                "uni":   dict(m.uni),
                                "bi":    {k2: dict(v) for k2,v in m.bi.items()},
                                "tri":   {k2: dict(v) for k2,v in m.tri.items()},
                                "total": m.total,
                            }
                            for k, m in st.session_state.engine.markovs.items()
                        },
                    })
                    st.session_state.engine_bytes = engine_bytes
                    st.session_state.engine_filename = f"coreai_v2_{corpus_file.name.split('.')[0]}.pkl"
                except Exception:
                    pass
            except Exception as e:
                st.error(f"학습 실패: {e}")

    # 학습 완료 상태
    if st.session_state.trained:
        stats = st.session_state.train_stats
        st.success(f"✓ AI 학습됨")
        st.caption(f"{stats.get('n_sentences',0)}문장 | {stats.get('vocab_size',0)}어휘")

        # 다운로드 버튼
        if "engine_bytes" in st.session_state:
            st.download_button(
                label="💾 엔진 저장 (.pkl)",
                data=st.session_state.engine_bytes,
                file_name=st.session_state.get("engine_filename","coreai_v2.pkl"),
                mime="application/octet-stream",
                help="저장 후 다음 접속 시 업로드하면 재학습 불필요 (0ms)",
                use_container_width=True,
            )

        # 클러스터 요약
        st.markdown("**클러스터 구조**")
        for k, info in stats.get("clusters",{}).items():
            if info["n_sentences"] > 0:
                kw = " · ".join(info["keywords"][:3])
                st.markdown(f"""
<div class="cluster-card">
  <div class="cluster-title">CLUSTER {k} — {info['n_sentences']}문장</div>
  <div style="font-size:0.8rem;margin-top:0.2rem;">{kw}</div>
</div>""", unsafe_allow_html=True)

    # pkl 업로드로 즉시 로드
    st.markdown("---")
    st.markdown("### 💾 저장된 엔진 불러오기")
    pkl_file = st.file_uploader(
        "엔진 파일 (.pkl) — 모바일: 파일앱에서 선택",
        type=None,
        key="engine_pkl",
        help="이전에 저장한 엔진을 업로드하면 재학습 없이 즉시 사용",
    )
    if pkl_file and not st.session_state.trained:
        if not pkl_file.name.endswith('.pkl'):
            st.warning("⚠️ .pkl 파일만 지원해요.")
        else:
            try:
                import pickle, time as _t
                t0 = _t.perf_counter()
                data = pickle.loads(pkl_file.read())
                engine = CoreAIv2Engine.load_from_dict(data)
                elapsed = (_t.perf_counter()-t0)*1000
                st.session_state.engine = engine
                st.session_state.trained = True
                st.session_state.train_stats = data.get("train_stats", {})
                st.success(f"✓ 엔진 로드 완료 ({elapsed:.0f}ms)")
                st.rerun()
            except Exception as e:
                st.error(f"로드 실패: {e}")

    st.markdown("---")
    st.caption("도메인 특화 AI — GPU 0 | 오프라인")
    st.caption("GPU 0 | CPU only | numpy only")


# ── 메인 ─────────────────────────────────────────────────────
st.markdown('<div class="title">🧠 도메인 특화 AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">문서를 넣으면 그 도메인 전문가가 됩니다 | GPU 0 | 오프라인</div>',
    unsafe_allow_html=True
)

if not st.session_state.trained:
    st.info("← 사이드바에서 가이드라인 코퍼스를 업로드하고 학습하세요.")

tab1, tab2, tab3 = st.tabs(["💬 질문 답변", "🔍 가드레일 단독", "📊 클러스터 분석"])

# ── 탭 1: 질문 답변 (재생성 루프) ────────────────────────────
with tab1:
    question = st.text_area(
        "질문",
        placeholder="질문을 입력하세요...",
        height=80,
        label_visibility="collapsed",
    )
    run = st.button("▶ 실행", use_container_width=True,
                    disabled=not (api_key and st.session_state.trained))

    if run and question.strip():
        if not api_key:
            st.error("API Key를 입력하세요")
            st.stop()

        def llm_fn(prompt: str) -> str:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            msgs = []
            if st.session_state.guideline_hint:
                msgs.append({"role":"system","content":
                    f"당신은 다음 가이드라인을 참고하여 답하는 전문 AI입니다.\n\n"
                    f"가이드라인:\n{st.session_state.guideline_hint}\n\n"
                    f"가이드라인을 최대한 활용하여 답하되, "
                    f"가이드라인에 명시되지 않은 내용은 일반 지식으로 보완하여 답하세요. "
                    f"단, 가이드라인과 명백히 다른 내용은 그렇다고 밝혀주세요."
                })
            msgs.append({"role":"user","content":prompt})
            resp = client.chat.completions.create(
                model=model, messages=msgs, max_tokens=500
            )
            return resp.choices[0].message.content.strip()

        with st.spinner("AI 실행 중..."):
            response = run_guardrail_loop(
                question=question,
                llm_fn=llm_fn,
                engine=st.session_state.engine,
                max_attempts=max_retry,
                logp_thr=logp_thr,
                guideline_hint=st.session_state.guideline_hint[:600],
            )

        st.session_state.history.insert(0, {
            "question": question,
            "response": response,
        })

        # 판정 표시
        v = response.status
        v_cls = {"PASS":"pass","WARNING":"warn","FATAL":"fatal"}.get(v,"")
        v_icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(v,"⬜")

        st.markdown(f"""
<div style="background:#111827;border:1px solid #1e3050;border-radius:8px;
padding:1.5rem;margin:1rem 0;">
  <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
  color:#64748b;letter-spacing:3px;margin-bottom:0.5rem;">VERDICT</div>
  <div class="{v_cls}" style="font-size:2rem;">{v_icon} {v}</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
  color:#64748b;margin-top:0.4rem;">
    {response.attempts}회 시도 | {response.total_ms:.0f}ms | logP {response.final_logp:+.2f}
  </div>
</div>""", unsafe_allow_html=True)

        st.markdown("**최종 답변**")
        st.info(response.answer)

        if len(response.history) > 1:
            st.markdown("**시도별 가드레일 결과**")
            for h in response.history:
                s_icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(h.status,"⬜")
                bar_col = {"PASS":"#00ff88","WARNING":"#ffd600","FATAL":"#ff4444"}.get(h.status,"#64748b")
                pct = max(0, min(100, int((h.avg_logp+15)/15*100)))
                st.markdown(f"""
<div style="background:#0d1525;border:1px solid #1e3050;border-radius:6px;
padding:0.6rem 1rem;margin-bottom:0.4rem;font-family:'Space Mono',monospace;font-size:0.75rem;">
  {s_icon} 시도 {h.attempt} — <b>{h.status}</b> &nbsp;|&nbsp;
  logP: <b>{h.avg_logp:+.2f}</b> &nbsp;|&nbsp; {h.elapsed_ms:.2f}ms
  <div class="bar-wrap">
    <div style="width:{pct}%;height:6px;background:{bar_col};border-radius:4px;"></div>
  </div>
</div>""", unsafe_allow_html=True)


# ── 탭 2: 가드레일 단독 ───────────────────────────────────────
with tab2:
    text_input = st.text_area(
        "텍스트 입력",
        placeholder="검증할 텍스트를 입력하세요...",
        height=100,
        label_visibility="collapsed",
    )
    if st.button("🔍 가드레일 검증", use_container_width=True,
                 disabled=not st.session_state.trained):
        if text_input.strip():
            r = st.session_state.engine.evaluate(text_input, logp_thr=logp_thr)
            v = r["verdict"]
            v_icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴","SKIP":"⬜"}[v]
            v_cls = {"PASS":"pass","WARNING":"warn","FATAL":"fatal"}.get(v,"")

            st.markdown(f"""
<div style="background:#111827;border:1px solid #1e3050;border-radius:8px;
padding:1.2rem;margin:0.8rem 0;">
  <div class="{v_cls}" style="font-size:1.8rem;">{v_icon} {v}</div>
  <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
  color:#64748b;margin-top:0.3rem;">
    logP: {r['logp']:+.3f} | 클러스터: {r['cluster']} | {r['ms']:.2f}ms
    {'| ⚡의미확장' if r.get('expanded') else ''}
  </div>
</div>""", unsafe_allow_html=True)

            if r["cluster_keywords"]:
                st.caption(f"매칭 클러스터 핵심어: {' · '.join(r['cluster_keywords'])}")

            # XAI 토큰별 분석
            nm_r = st.session_state.engine.nm_engine.evaluate(text_input, logp_thr=logp_thr)
            per_token = nm_r.get("per_token", [])
            if per_token:
                with st.expander("🔍 XAI — 토큰별 분석", expanded=(v!="PASS")):
                    for pt in per_token:
                        tok = pt["token"]
                        lp  = pt["logp"]
                        ing = pt.get("in_graph", True)
                        col = "#00ff88" if lp>=-5 else "#ffd600" if lp>=-11.5 else "#ff4444"
                        oov = " ⚠️OOV" if not ing else ""
                        st.markdown(
                            f"<span style='font-family:monospace;color:{col};'>"
                            f"[{tok}{oov}] logP:{lp:+.2f}</span>",
                            unsafe_allow_html=True)

            # 클러스터별 점수
            with st.expander("클러스터별 logP", expanded=False):
                for k, info in sorted(r["per_cluster"].items(),
                                      key=lambda x:-x[1]["logp"]):
                    kw = " · ".join(info["keywords"][:3])
                    lp = info["logp"]
                    bar_col = "#00ff88" if lp >= -10 else "#ffd600" if lp >= logp_thr else "#ff4444"
                    pct = max(0, min(100, int((lp+20)/20*100)))
                    st.markdown(f"""
<div style="margin-bottom:0.4rem;font-size:0.8rem;">
  <span style="font-family:'Space Mono',monospace;color:#64748b;">C{k}</span>
  &nbsp; {kw} &nbsp;
  <span style="font-family:'Space Mono',monospace;color:{bar_col};">{lp:+.2f}</span>
  <div class="bar-wrap">
    <div style="width:{pct}%;height:4px;background:{bar_col};border-radius:4px;"></div>
  </div>
</div>""", unsafe_allow_html=True)


# ── 탭 3: 클러스터 분석 ───────────────────────────────────────
with tab3:
    if not st.session_state.trained:
        st.info("학습 후 클러스터 구조를 확인할 수 있어요.")
    else:
        stats = st.session_state.train_stats
        st.markdown("#### 학습 통계")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("문장 수", stats.get("n_sentences",0))
        c2.metric("어휘 수", stats.get("vocab_size",0))
        c3.metric("클러스터", n_clusters)
        c4.metric("학습 시간", f"{stats.get('total_ms',0):.0f}ms")

        st.markdown("---")
        st.markdown("#### 클러스터별 구조")
        cols = st.columns(min(n_clusters, 3))
        for k, info in stats.get("clusters",{}).items():
            if info["n_sentences"] == 0: continue
            with cols[k % len(cols)]:
                kw_html = " ".join(
                    f'<span style="background:#1a2235;border:1px solid #1e3050;'
                    f'border-radius:3px;padding:1px 6px;font-size:0.75rem;'
                    f'font-family:Space Mono,monospace;">{w}</span>'
                    for w in info["keywords"]
                )
                st.markdown(f"""
<div class="cluster-card">
  <div class="cluster-title">CLUSTER {k}</div>
  <div style="font-size:1.4rem;font-family:'Space Mono',monospace;
  color:var(--accent);font-weight:700;margin:0.3rem 0;">
    {info['n_sentences']}
  </div>
  <div style="font-size:0.7rem;color:var(--muted);">문장</div>
  <div style="margin-top:0.5rem;">{kw_html}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 학습 시간 분석")
        c1,c2,c3 = st.columns(3)
        c1.metric("TinyTransformer", f"{stats.get('emb_ms',0):.0f}ms")
        c2.metric("Hopfield 분해", f"{stats.get('decomp_ms',0):.0f}ms")
        c3.metric("마르코프", f"{stats.get('markov_ms',0):.0f}ms")

# ── 대화 기록 ─────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📋 대화 기록")
    for i, item in enumerate(st.session_state.history[:8]):
        r = item["response"]
        icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(r.status,"⬜")
        with st.expander(
            f"{icon} {item['question'][:50]}{'...' if len(item['question'])>50 else ''} ({r.attempts}회)",
            expanded=(i==0)
        ):
            st.info(r.answer)
            st.caption(f"logP: {r.final_logp:+.2f} | {r.total_ms:.0f}ms")
    if st.button("기록 초기화"):
        st.session_state.history = []
        st.rerun()
