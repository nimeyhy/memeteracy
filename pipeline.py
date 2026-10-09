"""밈터러시 실시간 용례 분석 파이프라인.

보고서 '연구 방법' 1.3~1.6절의 절차를 소량(100~300건) 실시간 데이터에 맞게 옮긴 것이다.
  수집(네이버 검색 API 미리보기) → 전처리 → 문맥 창 추출 → KR-SBERT 임베딩
  → UMAP(cosine) → HDBSCAN → 유의 군집 3기준 → c-TF-IDF 명칭 → 대표 문서
"""
from __future__ import annotations

import html
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import requests
from scipy.stats import chi2

# 2026년 NAVER API HUB 이관 후 주소 (기존 openapi.naver.com 키는 사용 불가)
NAVER_URL = "https://naverapihub.apigw.ntruss.com/search/v1/{source}"
SOURCES = {"blog": "블로그", "cafearticle": "카페", "news": "뉴스"}
MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"

# ---------------------------------------------------------------- 수집

TAG_RE = re.compile(r"<[^>]+>")


def clean_html(s: str) -> str:
    return html.unescape(TAG_RE.sub("", s or "")).strip()


def fetch_naver(variants, total, sources, client_id, client_secret,
                sort="sim", exclude=(), session=None, log=None):
    """검색 결과 미리보기를 total건까지 모은다.

    변형 표기 × 출처마다 할당량을 나누고, 밈이 실제로 들어 있는 글만 남긴다.
    한 검색어당 네이버 API 한도(start ≤ 1000)를 넘지 않는다.
    """
    session = session or requests.Session()
    headers = {"X-NCP-APIGW-API-KEY-ID": client_id, "X-NCP-APIGW-API-KEY": client_secret}
    combos = [(v, s) for v in variants for s in sources]
    quotas = [total // len(combos)] * len(combos)
    for i in range(total - sum(quotas)):
        quotas[i] += 1

    docs, seen_links, seen_texts = [], set(), set()
    stats = {"raw": 0, "no_meme": 0, "excluded": 0, "duplicate": 0}
    for (variant, source), quota in zip(combos, quotas):
        got, start = 0, 1
        while got < quota and start <= 1000:
            r = session.get(NAVER_URL.format(source=source), headers=headers, timeout=10,
                            params={"query": variant, "display": 100, "start": start, "sort": sort})
            if r.status_code != 200:
                raise RuntimeError(f"네이버 API 오류 {r.status_code}: {r.text[:200]}")
            items = r.json().get("items", [])
            if not items:
                break
            for it in items:
                stats["raw"] += 1
                title, desc = clean_html(it.get("title")), clean_html(it.get("description"))
                text = f"{title} {desc}".strip()
                link = it.get("link") or it.get("originallink") or ""
                if not any(v in text for v in variants):
                    stats["no_meme"] += 1
                    continue
                if any(x and x in text for x in exclude):
                    stats["excluded"] += 1
                    continue
                key = normalize(text)
                if link in seen_links or key in seen_texts:
                    stats["duplicate"] += 1
                    continue
                seen_links.add(link)
                seen_texts.add(key)
                docs.append({
                    "text": text, "link": link, "source": SOURCES[source],
                    "origin": it.get("bloggername") or it.get("cafename") or "",
                    "date": it.get("postdate") or it.get("pubDate") or "",
                })
                got += 1
                if got >= quota:
                    break
            start += 100
        if log:
            log(f"{SOURCES[source]} · '{variant}': {got}/{quota}건")
    return docs, stats


# ---------------------------------------------------------------- 전처리 (보고서 1.3절)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.]+")
SENT_SPLIT = re.compile(r"(?<=[.!?…~])\s+|\n+")


def normalize(text: str) -> str:
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub("@USER", text)
    text = text.replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def context_window(text: str, variants) -> str:
    """밈이 쓰인 문장과 앞뒤 한 문장만 남긴다 (임베딩 입력)."""
    sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
    hits = [i for i, s in enumerate(sents) if any(v in s for v in variants)]
    if not hits:
        return text
    keep = sorted({j for i in hits for j in (i - 1, i, i + 1) if 0 <= j < len(sents)})
    return " ".join(sents[j] for j in keep)


def is_standalone(text: str, variants) -> bool:
    """밈 단독 사용(밈 외에 실질 내용이 없음)이면 True — 보고서 1.4절 제외 기준."""
    rest = text
    for v in sorted(variants, key=len, reverse=True):
        rest = rest.replace(v, " ")
    rest = re.sub(r"[\W_ㅋㅎㅠㅜ]+", "", rest)
    return len(rest) < 2


# ---------------------------------------------------------------- 토큰화 (군집 해석용, kiwi)

KEEP_TAGS = {"NNG", "NNP", "XR", "SL"}
VERB_TAGS = {"VV", "VA"}
NEG_ADV = {"안", "못"}
ONE_CHAR_OK = {"돈", "집", "옷", "꿈", "밥", "술", "춤", "말", "책", "차", "일", "맛", "눈", "몸", "팬"}
EN_STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "for", "is", "are", "i", "you",
           "it", "my", "me", "we", "with", "at", "by", "this", "that", "be", "com", "www", "http", "https"}
KO_STOP = {"것", "수", "등", "때", "거", "더", "중", "좀", "그", "이", "저", "여기", "오늘", "진짜", "정말",
           "블로그", "카페", "포스팅", "게시", "글", "사진", "영상", "댓글"}


class Tokenizer:
    def __init__(self, variants):
        from kiwipiepy import Kiwi
        self.kiwi = Kiwi()
        self.variants = list(variants)
        for v in self.variants:
            self.kiwi.add_user_word(v, "NNP", 0)

    def __call__(self, text: str) -> list[str]:
        out = []
        for t in self.kiwi.tokenize(text):
            f, tag = t.form, t.tag
            if f in self.variants:
                out.append(f)
            elif tag in KEEP_TAGS:
                f = f.lower() if tag == "SL" else f
                if tag == "SL" and (f in EN_STOP or len(f) < 2):
                    continue
                if tag != "SL" and len(f) == 1 and f not in ONE_CHAR_OK:
                    continue
                if f not in KO_STOP:
                    out.append(f)
            elif tag in VERB_TAGS and len(f) >= 1:
                out.append(f + "다")
            elif tag == "MAG" and f in NEG_ADV:
                out.append(f)
        return out


# ---------------------------------------------------------------- 임베딩

def load_model(name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device="cpu")


def embed(model, texts):
    return np.asarray(model.encode(texts, batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=False), dtype=np.float32)


# ---------------------------------------------------------------- 군집 지표

def ctfidf(tokens_by_cluster: dict[int, list[str]], top_n=10):
    """BERTopic식 c-TF-IDF: tf_{t,c} · log(1 + A / f_t)."""
    vocab = sorted({w for toks in tokens_by_cluster.values() for w in toks})
    idx = {w: i for i, w in enumerate(vocab)}
    labels = list(tokens_by_cluster)
    tf = np.zeros((len(labels), len(vocab)))
    for r, c in enumerate(labels):
        for w in tokens_by_cluster[c]:
            tf[r, idx[w]] += 1
    if not vocab:
        return {c: [] for c in labels}
    A = tf.sum(1).mean()
    f_t = tf.sum(0)
    idf = np.log(1 + A / np.maximum(f_t, 1))
    tfn = tf / np.maximum(tf.sum(1, keepdims=True), 1)
    score = tfn * idf
    return {c: [(vocab[j], float(score[r, j])) for j in np.argsort(-score[r])[:top_n] if score[r, j] > 0]
            for r, c in enumerate(labels)}


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def keyness(target_docs: list[list[str]], ref_docs: list[list[str]], min_freq: int, min_docs: int):
    """로그우도비 G² + Log Ratio + BH-FDR (보고서 1.5절). 과대 사용 어휘만 반환."""
    from collections import Counter
    a_cnt = Counter(w for d in target_docs for w in d)
    b_cnt = Counter(w for d in ref_docs for w in d)
    df_cnt = Counter(w for d in target_docs for w in set(d))
    c, d = sum(a_cnt.values()), sum(b_cnt.values())
    rows = []
    if c == 0 or d == 0:
        return rows
    for w, a in a_cnt.items():
        if a < min_freq or df_cnt[w] < min_docs:
            continue
        b = b_cnt.get(w, 0)
        if a / c <= b / d:
            continue
        e1 = c * (a + b) / (c + d)
        e2 = d * (a + b) / (c + d)
        g2 = 2 * (a * math.log(a / e1) + (b * math.log(b / e2) if b > 0 else 0))
        lr = math.log2((a / c) / ((b if b > 0 else 0.5) / d))
        rows.append({"word": w, "freq": a, "g2": g2, "log_ratio": lr, "p": float(chi2.sf(g2, 1))})
    q = bh_fdr([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q"] = float(qq)
    return sorted(rows, key=lambda r: -r["g2"])


# ---------------------------------------------------------------- 메인 분석

@dataclass
class Params:
    min_cluster_size: int
    min_samples: int
    min_freq: int
    min_docs: int
    min_sig_keywords: int = 5
    min_mean_log_ratio: float = 2.0
    max_cohesion: float = 0.95
    size_percentile: float = 75.0
    n_rep: int = 10

    @classmethod
    def for_n(cls, n: int) -> "Params":
        """보고서 기준(min_cluster_size=30, 최소 출현 5회·3문서)을 수집량에 비례해 축소."""
        return cls(min_cluster_size=max(5, round(n * 0.05)),
                   min_samples=3,
                   min_freq=max(2, round(n / 75)),
                   min_docs=2)


@dataclass
class Result:
    n_input: int
    n_standalone: int
    n_used: int
    params: Params
    clusters: list = field(default_factory=list)  # dict per cluster
    coords: np.ndarray | None = None
    labels: np.ndarray | None = None
    docs: list = field(default_factory=list)


def analyze(docs, variants, embed_fn, seed=42, log=None) -> Result:
    import umap
    from sklearn.cluster import HDBSCAN

    n_input = len(docs)
    docs = [dict(d, clean=normalize(d["text"])) for d in docs]
    kept = [d for d in docs if not is_standalone(d["clean"], variants)]
    n_standalone = n_input - len(kept)
    for d in kept:
        d["window"] = context_window(d["clean"], variants)
    n = len(kept)
    params = Params.for_n(n)
    res = Result(n_input, n_standalone, n, params, docs=kept)
    if n < params.min_cluster_size * 2:
        return res

    if log: log("문장 임베딩 계산 중")
    X = embed_fn([d["window"] for d in kept])

    if log: log("UMAP 차원 축소 · HDBSCAN 군집화 중")
    nn = max(5, min(15, n // 10))
    Z = umap.UMAP(n_neighbors=nn, n_components=5, min_dist=0.0, metric="cosine",
                  random_state=seed).fit_transform(X)
    labels = HDBSCAN(min_cluster_size=params.min_cluster_size,
                     min_samples=params.min_samples).fit_predict(Z)
    coords = umap.UMAP(n_neighbors=nn, n_components=2, min_dist=0.1, metric="cosine",
                       random_state=seed).fit_transform(X)
    res.labels, res.coords = labels, coords

    cids = sorted(c for c in set(labels) if c >= 0)
    if not cids:
        return res

    if log: log("Keyness · c-TF-IDF 계산 중")
    tok = Tokenizer(variants)
    toks = [tok(d["clean"]) for d in kept]  # 해석에는 전문 사용 (보고서 1.4절)
    variant_set = set(variants)
    toks_kw = [[w for w in t if w not in variant_set] for t in toks]
    ct = ctfidf({c: [w for i in np.where(labels == c)[0] for w in toks_kw[i]] for c in cids})

    sizes = np.array([(labels == c).sum() for c in cids])
    size_cut = float(np.percentile(sizes, params.size_percentile))

    for c in cids:
        members = np.where(labels == c)[0]
        others = np.where(labels != c)[0]
        centroid = X[members].mean(0)
        centroid /= np.linalg.norm(centroid) or 1
        sims = X[members] @ centroid
        cohesion = float(np.median(sims))
        kw = keyness([toks_kw[i] for i in members], [toks_kw[i] for i in others],
                     params.min_freq, params.min_docs)
        sig = [r for r in kw if r["q"] <= 0.05]
        mean_lr = float(np.mean([r["log_ratio"] for r in sig])) if sig else 0.0

        pass_size = len(members) >= size_cut
        pass_cohesion = cohesion < params.max_cohesion
        pass_keyness = len(sig) >= params.min_sig_keywords and mean_lr >= params.min_mean_log_ratio
        if pass_size and pass_cohesion and pass_keyness:
            kind = "핵심"
        elif pass_size and pass_cohesion:
            kind = "범용"
        else:
            kind = "기타"

        order = members[np.argsort(-sims)]
        top_words = [w for w, _ in ct[c]]
        res.clusters.append({
            "id": int(c), "size": int(len(members)), "kind": kind,
            "name": " · ".join(top_words[:3]) or f"군집 {c}",
            "ctfidf": ct[c], "keyness": sig[:15], "n_sig": len(sig), "mean_log_ratio": mean_lr,
            "cohesion": cohesion, "pass": {"문서 수": pass_size, "Keyness": pass_keyness, "응집도": pass_cohesion},
            "rep": [dict(kept[i], sim=float(X[i] @ centroid)) for i in order[:params.n_rep]],
        })
    res.size_cut = size_cut
    rank = {"핵심": 0, "범용": 1, "기타": 2}
    res.clusters.sort(key=lambda k: (rank[k["kind"]], -k["size"]))
    return res


def credentials():
    """Streamlit secrets 또는 환경 변수에서 네이버 키를 읽는다."""
    cid = os.environ.get("NAVER_CLIENT_ID")
    sec = os.environ.get("NAVER_CLIENT_SECRET")
    if not (cid and sec):
        try:
            import streamlit as st
            cid = cid or st.secrets.get("NAVER_CLIENT_ID")
            sec = sec or st.secrets.get("NAVER_CLIENT_SECRET")
        except Exception:
            pass
    return cid, sec
