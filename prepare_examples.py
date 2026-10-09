"""팀 분석 결과 폴더 → data/examples.json 변환.

사용법: python prepare_examples.py <압축 푼 결과들이 있는 상위 폴더>
  - final_cluster_summary/ (밈 이름이 붙은 파일로 밈을 식별)
  - hdbscan_mcs30_ms5_eom/ (representative_documents.csv, 코드 접두어로 밈 식별)
"""
import glob
import json
import os
import re
import shutil
import sys

import pandas as pd

# 코드 = 플랫폼 글자(T=X, D=디시, Y=유튜브, B=블로그) + 밈 글자 + 번호
CODE2MEME = {"A": "야르", "B": "샤갈", "C": "미새", "D": "밤티", "E": "크크"}
DISPLAY = {"야르": "야르", "샤갈": "샤갈", "미새": "-미새", "밤티": "밤티", "크크": "-크크"}
ORDER = ["야르", "샤갈", "미새", "밤티", "크크"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def dec(s):
    return re.sub(r"#U([0-9a-f]{4})", lambda m: chr(int(m.group(1), 16)), s)


def split_list(s, n=10):
    if not isinstance(s, str):
        return []
    return [x.strip() for x in s.split("|") if x.strip()][:n]


def keyness_words(s, n=5):
    out = []
    for part in split_list(s, n):
        m = re.match(r"(.+?) \(LR=([\d.]+)", part)
        out.append(f"{m.group(1)}({float(m.group(2)):.1f})" if m else part)
    return out


def main(root):
    finals, reps = {}, {}
    for d in glob.glob(os.path.join(root, "**", "final_cluster_summary"), recursive=True):
        for f in os.listdir(d):
            name = dec(f)
            for m in ORDER:
                if name.startswith(m + "_"):
                    finals[m] = d
    for d in glob.glob(os.path.join(root, "**", "hdbscan_mcs30_ms5_eom"), recursive=True):
        rp = os.path.join(d, "representative_documents.csv")
        if os.path.exists(rp):
            code = pd.read_csv(rp)["코드"].astype(str).str[1].mode().iloc[0]
            if code in CODE2MEME:
                reps[CODE2MEME[code]] = rp

    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    memes = {}
    for m in ORDER:
        if m not in finals:
            continue
        F = finals[m]
        files = {dec(f): os.path.join(F, f) for f in os.listdir(F)}
        judge = pd.read_csv(files["cluster_selection_judgement.csv"])
        plat = None
        pk = f"{m}_군집별_플랫폼_비율과_잔차.csv"
        if pk in files:
            plat = pd.read_csv(files[pk]).set_index("군집")
        rep = None
        if m in reps:
            r = pd.read_csv(reps[m])
            # 중심 코사인 순위 우선, 부족하면 소속 확률 순위로 보충 (중복 제외, 최대 10개)
            r["_m"] = (r["Representative_Method"] != "Centroid_Cosine_Similarity").astype(int)
            rep = r.sort_values(["Cluster", "_m", "Rank"])

        clusters = []
        sel = judge[judge["Selection_Type"].isin(["핵심", "범용"])].copy()
        sel["_o"] = sel["Selection_Type"].map({"핵심": 0, "범용": 1})
        for _, row in sel.sort_values(["_o", "Document_Count"], ascending=[True, False]).iterrows():
            cid = int(row["Cluster"])
            bt = split_list(row.get("BERTopic_Top10"))
            c = {
                "id": cid, "kind": row["Selection_Type"], "size": int(row["Document_Count"]),
                "share": float(row["Document_Share_pct"]),
                "name": " · ".join(bt[:3]) or f"군집 {cid}",
                "keywords": bt, "keyness": keyness_words(row.get("Top5_Visual_Keyness")),
                "n_sig": int(row["Keyness_FDR_Significant_Count"]) if pd.notna(row["Keyness_FDR_Significant_Count"]) else 0,
                "mean_lr": float(row["Keyness_Log_Ratio_Mean"]) if pd.notna(row["Keyness_Log_Ratio_Mean"]) else None,
                "cohesion": float(row["Centroid_Cosine_Median"]),
                "docs": [],
            }
            if plat is not None and cid in plat.index:
                p = plat.loc[cid]
                c["platform"] = {"top": p["가장 쏠린 플랫폼"], "resid": float(p["최대 잔차"]),
                                 "share": {k: float(p[f"{k} 비율(%)"]) for k in ["디시", "X", "유튜브", "블로그"]
                                           if f"{k} 비율(%)" in p}}
            if rep is not None:
                rr = rep[rep["Cluster"] == cid]
                seen = []
                for t in rr["Embedding_Text"].fillna(rr["Text_Original"]):
                    t = re.sub(r"https?://\S+", "", str(t))
                    t = re.sub(r"@[\w.]+", "@USER", t).strip()  # 계정명 비식별
                    if t and t not in seen:
                        seen.append(t)
                c["docs"] = seen[:10]
            clusters.append(c)

        entry = {
            "label": DISPLAY[m], "n_clusters": int(len(judge)),
            "n_docs_clustered": int(judge["Document_Count"].sum()),
            "n_core": int((judge["Selection_Type"] == "핵심").sum()),
            "n_general": int((judge["Selection_Type"] == "범용").sum()),
            "clusters": clusters,
        }
        tk = f"{m}_플랫폼_분석_결과문.txt"
        if tk in files:
            entry["platform_text"] = open(files[tk], encoding="utf-8").read().strip()
        tt = f"{m}_플랫폼_연관성_검정.csv"
        if tt in files:
            entry["platform_test"] = pd.read_csv(files[tt]).iloc[0].to_dict()
        imgs = {}
        for key, fname in [("heatmap", f"{m}_용례_플랫폼_잔차_히트맵.png"),
                           ("compare", f"{m}_용례별_플랫폼_비교(본문).png"),
                           ("map", "cluster_map_all.png")]:
            if fname in files:
                dst = f"img/{m}_{key}.png"
                shutil.copy(files[fname], os.path.join(OUT, dst))
                imgs[key] = dst
        entry["images"] = imgs
        memes[m] = entry
        print(f"{m}: 군집 {entry['n_clusters']} · 핵심 {entry['n_core']} · 범용 {entry['n_general']}"
              f" · 대표문서 {'O' if m in reps else 'X'}")

    if len(sys.argv) > 2:
        add_bradford(memes, sys.argv[2])
    json.dump({"memes": memes}, open(os.path.join(OUT, "examples.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)


SCOPES = ["전체", "X", "디시", "유튜브", "블로그"]


def add_bradford(memes, root):
    """브래드포드 폴더(밈별 *_브래드포드_결과.xlsx, 곡선·구역표 PNG)를 붙인다."""
    for m in memes:
        hits = glob.glob(os.path.join(root, "**", f"*{m}_브래드포드_결과.xlsx"), recursive=True)
        if not hits:
            continue
        x = pd.read_excel(hits[0], sheet_name=None)
        folder = os.path.dirname(hits[0])
        prefix = os.path.basename(hits[0]).replace("_브래드포드_결과.xlsx", "")
        b = {"compare": x["범위별비교"].astype(str).to_dict("records"), "scopes": {}}
        for sc in SCOPES:
            sc_entry = {}
            if f"{sc}_구역표" in x:
                sc_entry["zones"] = x[f"{sc}_구역표"].astype(str).to_dict("records")
            for kind in ["곡선", "구역표"]:
                src = os.path.join(folder, f"{prefix}_브래드포드_{kind}_{sc}.png")
                if os.path.exists(src):
                    dst = f"img/{m}_bradford_{kind}_{sc}.png"
                    shutil.copy(src, os.path.join(OUT, dst))
                    sc_entry["curve" if kind == "곡선" else "zone_img"] = dst
            if sc_entry:
                b["scopes"][sc] = sc_entry
        memes[m]["bradford"] = b
        print(f"{m}: 브래드포드 범위 {list(b['scopes'])}")


if __name__ == "__main__":
    main(sys.argv[1])
