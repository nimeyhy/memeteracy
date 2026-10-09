import html
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import pipeline as P

st.set_page_config(page_title="밈터러시", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.small-note {font-size: 0.78rem; color: #888; font-weight: 400;}
.kind {display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.78rem; margin-right:6px;}
.kind-핵심 {background:#1f6feb22; color:#1f6feb; border:1px solid #1f6feb55;}
.kind-범용 {background:#8b949e22; color:#6e7681; border:1px solid #8b949e55;}
.kind-기타 {background:#d2992222; color:#9a6700; border:1px solid #d2992255;}
.doc {padding:8px 10px; border-left:3px solid #1f6feb55; margin:6px 0; font-size:0.92rem;}
.doc .meta {font-size:0.75rem; color:#888;}
</style>
""", unsafe_allow_html=True)

st.title("밈터러시")
st.caption("밈이 실제로 어떤 맥락에서 쓰이는지 확인하고, 사용 여부를 스스로 판단하도록 돕는 도구")

tab_live, tab_study, tab_about = st.tabs(["실시간 용례 분석", "연구 결과", "도구 소개"])


# ------------------------------------------------------------------ 공용 키 사용량 제한
SHARED_LIMIT = 300  # 한 사람이 하루에 공용 키로 수집할 수 있는 최대 건수


@st.cache_resource
def usage_book():
    import threading
    return {"lock": threading.Lock(), "day": None, "used": {}}


def visitor_id():
    """접속자 구분용 ID. IP는 해시로만 보관하고 원문은 저장하지 않는다."""
    import hashlib
    ip = ""
    try:
        h = st.context.headers
        ip = (h.get("X-Forwarded-For") or h.get("X-Real-Ip") or "").split(",")[0].strip()
    except Exception:
        pass
    if not ip:  # 헤더가 없으면 브라우저 세션 단위로 센다
        ip = st.session_state.setdefault("_sid", os.urandom(8).hex())
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def _today():
    from datetime import datetime, timedelta, timezone
    return datetime.now(timezone(timedelta(hours=9))).date()


def shared_used(vid):
    book = usage_book()
    with book["lock"]:
        if book["day"] != _today():
            book["day"], book["used"] = _today(), {}
        return book["used"].get(vid, 0)


def add_shared_use(vid, n):
    book = usage_book()
    with book["lock"]:
        book["used"][vid] = book["used"].get(vid, 0) + n


@st.cache_resource(show_spinner="문장 임베딩 모델 불러오는 중 (처음 한 번만)")
def get_model():
    return P.load_model()


def render_cluster(c, idx):
    head = (f"<span class='kind kind-{c['kind']}'>{c['kind']} 군집</span>"
            f"<b>{c['name']}</b> "
            f"<span class='small-note'>※ c-TF-IDF 상위 키워드로 자동 생성한 임의 명칭</span>")
    st.markdown(head, unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("문서 수", c["size"])
    m2.metric("유의 키워드", c["n_sig"])
    m3.metric("평균 Log Ratio", f"{c['mean_log_ratio']:.2f}")
    m4.metric("응집도(중앙값)", f"{c['cohesion']:.3f}")
    kw = ", ".join(w for w, _ in c["ctfidf"][:10])
    st.markdown(f"**c-TF-IDF 키워드** · {kw or '—'}")
    if c["keyness"]:
        st.markdown("**Keyness 유의 어휘 (q ≤ 0.05)** · " +
                    ", ".join(f"{r['word']}({r['log_ratio']:.1f})" for r in c["keyness"][:10]))
    with st.expander(f"대표 문서 {len(c['rep'])}개 보기 (군집 중심에 가까운 순)", expanded=(idx == 0)):
        for d in c["rep"]:
            meta = " · ".join(x for x in [d.get("source"), d.get("origin"), d.get("date"),
                                          f"유사도 {d['sim']:.3f}"] if x)
            link = f" <a href='{d['link']}' target='_blank'>원문</a>" if d.get("link") else ""
            st.markdown(f"<div class='doc'>{html.escape(d['window'])}<div class='meta'>{meta}{link}</div></div>",
                        unsafe_allow_html=True)


# ------------------------------------------------------------------ 실시간
with tab_live:
    st.subheader("밈 검색")
    vid = visitor_id()
    remaining = max(0, SHARED_LIMIT - shared_used(vid))

    with st.form("search"):
        c1, c2 = st.columns([3, 1])
        q = c1.text_input("밈 (변형 표기는 쉼표로 구분)", placeholder="예: 샤갈, 쌰갈")
        total = c2.selectbox("수집량", [100, 200, 300], index=1)
        c3, c4, c5 = st.columns([2, 1, 2])
        sources = c3.multiselect("검색 대상", list(P.SOURCES), default=["blog", "cafearticle"],
                                 format_func=P.SOURCES.get)
        sort = c4.radio("정렬", ["sim", "date"], format_func={"sim": "정확도", "date": "최신"}.get)
        exclude = c5.text_input("제외할 표현 (선택)", placeholder="예: 밤티라미수, 밤티골")
        go = st.form_submit_button("분석하기", type="primary")

    st.caption("네이버 검색 결과의 미리보기만 수집하며, 본문에 들어가지 않습니다. "
               "300건 기준 1분 안팎이 걸립니다.")

    with st.expander("내 네이버 API 키로 검색하기" + ("" if remaining else " (공용 한도 소진)"),
                     expanded=(remaining == 0)):
        st.markdown(f"""
공용 키로는 한 사람당 하루 **{SHARED_LIMIT}건**까지 수집할 수 있습니다. 오늘 남은 공용 한도: **{remaining}건**

더 검색하려면 본인의 NAVER API HUB 키를 넣으세요. 넣은 키는 이 브라우저 창에서만 쓰이고 서버에 저장되지 않습니다.
<span class='small-note'>발급: 네이버 클라우드 플랫폼 콘솔 → Menu → All Services → Application Services → NAVER API HUB →
이용 신청 → Application 등록(검색 - 블로그·카페글·뉴스 선택) → 인증 정보</span>
""", unsafe_allow_html=True)
        k1, k2 = st.columns(2)
        own_id = k1.text_input("Client ID", key="own_id", type="password")
        own_sec = k2.text_input("Client Secret", key="own_sec", type="password")
        use_own = bool(own_id and own_sec)
        if use_own:
            st.success("내 키로 검색합니다. 공용 한도가 차감되지 않습니다.")

    if go:
        variants = [v.strip() for v in q.split(",") if v.strip()]
        excl = [v.strip() for v in exclude.split(",") if v.strip()]
        if use_own:
            cid, sec = own_id.strip(), own_sec.strip()
        else:
            cid, sec = P.credentials()
        if not variants:
            st.warning("밈을 입력하세요.")
        elif not sources:
            st.warning("검색 대상을 하나 이상 고르세요.")
        elif not use_own and total > remaining:
            st.warning(f"오늘 남은 공용 한도는 {remaining}건입니다. 수집량을 줄이거나, "
                       "아래 '내 네이버 API 키로 검색하기'에 본인 키를 넣어 주세요.")
        elif not (cid and sec):
            st.error("서버에 네이버 API 키가 설정되지 않았습니다. 본인 키를 넣어 검색해 주세요.")
        else:
            with st.status("분석 중", expanded=True) as status:
                try:
                    docs, stats = P.fetch_naver(variants, total, sources, cid, sec, sort=sort,
                                                exclude=excl, log=st.write)
                    if not use_own:
                        add_shared_use(vid, len(docs))
                    st.write(f"수집 완료: {len(docs)}건")
                    model = get_model()
                    res = P.analyze(docs, variants, lambda t: P.embed(model, t), log=st.write)
                    st.session_state["live"] = (variants, stats, res)
                    status.update(label="분석 완료", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="오류", state="error")
                    msg = str(e)
                    if use_own and ("401" in msg or "403" in msg):
                        msg = "입력한 API 키로 인증하지 못했습니다. Client ID·Secret과 검색 API 선택 여부를 확인하세요. " + msg
                    st.error(msg)

    if "live" in st.session_state:
        variants, stats, res = st.session_state["live"]
        st.divider()
        st.subheader(f"'{', '.join(variants)}' 용례 분석 결과")
        a, b, c, d = st.columns(4)
        a.metric("수집", res.n_input)
        b.metric("단독 사용 제외", res.n_standalone)
        c.metric("분석 대상", res.n_used)
        noise = int((res.labels == -1).sum()) if res.labels is not None else 0
        d.metric("어느 군집에도 속하지 않음", noise)
        st.caption(f"미리보기에 밈이 없는 결과 {stats['no_meme']}건, 제외 표현 {stats['excluded']}건, "
                   f"중복 {stats['duplicate']}건은 수집 단계에서 걸렀습니다.")

        if not res.clusters:
            st.info("군집이 만들어지지 않았습니다. 수집량을 늘리거나 변형 표기를 추가해 보세요.")
        else:
            core = [k for k in res.clusters if k["kind"] == "핵심"]
            gen = [k for k in res.clusters if k["kind"] == "범용"]
            etc = [k for k in res.clusters if k["kind"] == "기타"]

            df = pd.DataFrame({"x": res.coords[:, 0], "y": res.coords[:, 1],
                               "군집": [str(l) for l in res.labels],
                               "문장": [d["window"][:80] for d in res.docs]})
            name = {str(k["id"]): f"{k['kind']} · {k['name']}" for k in res.clusters}
            df["군집"] = df["군집"].map(lambda l: name.get(l, "미분류"))
            fig = px.scatter(df, x="x", y="y", color="군집", hover_data={"문장": True, "x": False, "y": False},
                             height=420)
            fig.update_traces(marker=dict(size=7, opacity=0.8))
            fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend_title_text="",
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, width="stretch")
            st.caption("점 하나가 게시글 하나입니다. 가까울수록 밈이 비슷한 맥락에서 쓰였다는 뜻입니다.")

            st.markdown(f"### 핵심 용례 {len(core)}개")
            if not core:
                st.info("세 가지 기준을 모두 충족한 핵심 군집이 없습니다. 아래 범용 군집을 참고하세요.")
            for i, k in enumerate(core):
                with st.container(border=True):
                    render_cluster(k, i)
            if gen:
                st.markdown(f"### 범용 용례 {len(gen)}개")
                st.caption("문서 수와 응집도 기준은 충족했지만 다른 군집과 구별되는 어휘가 부족한 군집")
                for k in gen:
                    with st.container(border=True):
                        render_cluster(k, 1)
            if etc:
                with st.expander(f"기준 미달 군집 {len(etc)}개"):
                    for k in etc:
                        failed = [n for n, ok in k["pass"].items() if not ok]
                        st.markdown(f"- **{k['name']}** ({k['size']}건) · 미충족: {', '.join(failed)}")

        p = res.params
        with st.expander("적용한 분석 기준"):
            st.markdown(f"""
- 임베딩: `{P.MODEL_NAME}` (밈이 쓰인 문장과 앞뒤 한 문장)
- 차원 축소: UMAP (cosine) · 군집화: HDBSCAN `min_cluster_size={p.min_cluster_size}`, `min_samples={p.min_samples}`
- **유의 군집 기준** (보고서 1.5절과 같은 논리, 소량 데이터에 맞게 최소 크기와 최소 빈도만 비례 축소)
  1. 문서 수: 군집 크기 상위 {100 - p.size_percentile:.0f}% (75백분위 이상)
  2. Keyness: G²·BH-FDR q ≤ 0.05 유의 어휘 {p.min_sig_keywords}개 이상, 평균 Log Ratio ≥ {p.min_mean_log_ratio}
     (어휘 최소 출현 {p.min_freq}회 · {p.min_docs}개 문서)
  3. 응집도: 군집 중심과의 코사인 유사도 중앙값 < {p.max_cohesion} (복사·붙여넣기 군집 제외)
- 1·3만 충족한 군집은 범용 군집으로 분류
""")

def show_image(target, rel, **kw):
    """그림이 없으면 오류 대신 안내만 표시한다."""
    fp = Path(__file__).parent / "data" / rel
    if fp.exists():
        target.image(str(fp), **kw)
    else:
        target.caption(f"(그림 파일 없음: data/{rel})")


# ------------------------------------------------------------------ 연구 결과
with tab_study:
    DATA = Path(__file__).parent / "data"
    path = DATA / "examples.json"
    if not path.exists():
        st.info("연구 결과 데이터를 준비 중입니다.")
    else:
        ex = json.loads(path.read_text(encoding="utf-8"))
        st.markdown("2026년 6월 28일부터 9월 26일까지(13주) 디시인사이드·X·유튜브·네이버 블로그에서 수집한 "
                    "텍스트 밈 5종의 분석 결과입니다. 실시간 검색과 같은 절차를 대규모 표본에 적용했습니다.")
        keys = list(ex["memes"])
        pick = st.radio("밈 선택", keys, horizontal=True, format_func=lambda k: ex["memes"][k]["label"])
        m = ex["memes"][pick]
        a, b, c, d = st.columns(4)
        a.metric("군집에 배정된 글", f"{m['n_docs_clustered']:,}")
        b.metric("전체 군집", m["n_clusters"])
        c.metric("핵심 용례", m["n_core"])
        d.metric("범용 용례", m["n_general"])

        sub_c, sub_p, sub_b = st.tabs(["용례 군집", "용례 × 플랫폼", "사용 분포 (브래드포드)"])
        with sub_c:
            kinds = st.multiselect("표시할 군집", ["핵심", "범용"], default=["핵심"], key=f"k_{pick}")
            shown = [k for k in m["clusters"] if k["kind"] in kinds]
            for i, k in enumerate(shown):
                with st.container(border=True):
                    st.markdown(f"<span class='kind kind-{k['kind']}'>{k['kind']} 군집</span>"
                                f"<b>{html.escape(k['name'])}</b> "
                                f"<span class='small-note'>※ BERTopic c-TF-IDF 상위 키워드로 붙인 임의 명칭 · 군집 {k['id']}</span>",
                                unsafe_allow_html=True)
                    plat = k.get("platform")
                    meta = f"문서 {k['size']:,}건 ({k['share']:.1f}%) · 응집도 {k['cohesion']:.3f}"
                    if k.get("mean_lr") is not None:
                        meta += f" · 유의 어휘 {k['n_sig']}개 · 평균 Log Ratio {k['mean_lr']:.2f}"
                    if plat:
                        meta += f" · 가장 많이 쓰인 플랫폼: {plat['top']} (조정 잔차 {plat['resid']:+.1f})"
                    st.caption(meta)
                    st.markdown("**c-TF-IDF 키워드** · " + ", ".join(k["keywords"]))
                    if k["keyness"]:
                        st.markdown("**Keyness 상위 어휘 (Log Ratio)** · " + ", ".join(k["keyness"]))
                    if plat and plat.get("share"):
                        st.markdown("**플랫폼 비율** · " + " · ".join(f"{p} {v:.0f}%" for p, v in plat["share"].items()))
                    with st.expander(f"대표 문서 {len(k['docs'])}개 (군집 중심에 가까운 순)", expanded=(i == 0)):
                        for t in k["docs"]:
                            st.markdown(f"<div class='doc'>{html.escape(t)}</div>", unsafe_allow_html=True)
            if m["images"].get("map"):
                with st.expander("전체 군집 지도 (UMAP 2차원)"):
                    show_image(st, m["images"]["map"])
        with sub_p:
            if m.get("platform_test"):
                t = m["platform_test"]
                x1, x2, x3 = st.columns(3)
                x1.metric("χ²", f"{float(t['카이제곱']):,.1f}", help=f"자유도 {t['자유도']}")
                x2.metric("p", "< .001" if float(t["p값"]) < 0.001 else f"{float(t['p값']):.3f}")
                x3.metric("Cramér's V", f"{float(t['크래머 V']):.3f}")
            for key in ["compare", "heatmap"]:
                if m["images"].get(key):
                    show_image(st, m["images"][key])
            if m.get("platform_text"):
                with st.expander("분석 결과문"):
                    st.text(m["platform_text"])
        with sub_b:
            b = m.get("bradford")
            if not b:
                st.info("브래드포드 분석 결과를 준비 중입니다.")
            else:
                scope = st.radio("범위", list(b["scopes"]), horizontal=True, key=f"bs_{pick}")
                s_ = b["scopes"][scope]
                g1, g2 = st.columns(2)
                if s_.get("curve"):
                    show_image(g1, s_["curve"], caption="브래드포드 곡선")
                if s_.get("zone_img"):
                    show_image(g2, s_["zone_img"], caption="구역표")
                if s_.get("zones"):
                    st.markdown(f"**{scope} · 구역별 출처**")
                    st.dataframe(pd.DataFrame(s_["zones"]), hide_index=True, width="stretch")
                st.divider()
                st.markdown("**범위별 비교**")
                st.dataframe(pd.DataFrame(b["compare"]), hide_index=True, width="stretch")
                st.markdown("세부 플랫폼(블로그 주제, 디시 갤러리, 유튜브 카테고리, X 작성자 소개글 군집)을 **출처**로, "
                            "밈이 쓰인 글을 **항목**으로 보고 브래드포드 법칙을 적용했습니다. 글 수가 많은 순으로 출처를 "
                            "정렬해 글 수가 1/3씩 되도록 세 구역으로 나누고, 구역별 출처 수가 1 : k : k²에 가까운지 봅니다.")
                st.caption("지니계수가 1에 가까울수록 소수의 출처에 집중. 플랫폼마다 출처를 나눈 단위가 달라 "
                           "(디시는 갤러리 수백 개, 나머지는 13~32개) 플랫폼 간 수치를 직접 비교하기는 어렵습니다. "
                           "모든 플랫폼에서 같은 양을 수집했으므로 비율은 실제 사용량이 아니라 표본 안의 분포입니다.")

# ------------------------------------------------------------------ 소개
with tab_about:
    st.markdown("""
#### 왜 이 도구가 필요한가
밈은 특정 커뮤니티의 맥락 속에서 만들어져 그 맥락을 모르는 사람들에게까지 퍼집니다.
같은 표현도 친구와의 대화에서는 자연스럽지만 공문서나 방송, 기업 홍보에서는 문제가 될 수 있습니다.
이 도구는 써도 되는 밈과 안 되는 밈을 정해 주지 않습니다. 대신 밈이 **실제로 어떤 맥락에서 쓰이는지**를
보여 주어, 이용자가 자신의 상황에 비추어 스스로 판단하도록 돕습니다.

#### 어떻게 분석하나
1. 네이버 검색 결과의 미리보기를 100·200·300건 수집합니다.
2. 밈이 쓰인 문장과 앞뒤 한 문장을 한국어 문장 임베딩 모델(KR-SBERT)로 벡터화합니다.
3. 의미가 비슷한 문장끼리 묶고(UMAP + HDBSCAN), 문서 수·변별 어휘·응집도 기준을 통과한 군집만 핵심 용례로 보여 줍니다.
4. 각 군집의 중심에 가까운 대표 문서 10개로 실제 쓰임을 확인할 수 있습니다.

#### 해석할 때 주의할 점
- 군집 이름은 c-TF-IDF 상위 키워드로 자동 생성한 것으로, 연구진이 붙인 해석이 아닙니다.
- 검색 결과 미리보기만 쓰므로 글 전체의 맥락과 다를 수 있습니다. 대표 문서의 원문 링크로 확인하세요.
- 네이버 검색 결과에 한정되며, 검색 시점과 정렬 방식에 따라 결과가 달라집니다.

<span class='small-note'>2026 사회과학대학 학술제 출품작</span>
""", unsafe_allow_html=True)
