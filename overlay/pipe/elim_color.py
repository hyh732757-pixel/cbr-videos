"""마커 색으로 이름 채우기 (2026-10-10, m11 SPF전 사용자 정답 30+30개 기반)
기본(EC_ALL=1): 빈 우리 상자의 마커 색이 한 선수 기준색에만 가까우면(그 장면에 그 선수 없음) 붙임. 정답2: 30개 중 27 맞음, 틀린 3개 중 2개는 기준색이 흰색으로 잘못 잡힌 MandoolsLove → 퍼짐 기준 추가
EC_ALL=0: 아래 '남은 한 명' 장면에서만
장면 단위 '남은 한 명' 후보(라인업 9명 이름 있음 + 빈 우리 상자 1개)는 m11 정답 30개 중 15개만 맞음 → 마커 색으로 거름.
마커 색: tri3(연한 색도 찾는 새 방식, tri3.json). 선수별 기준색 = 이름표로 직접 읽힌 상자의 Lab 중앙값(5개 이상).
조건: 후보 색이 빠진 선수 기준색과 거리 ≤ DMAX 이고, 기준색이 있는 다른 라인업 선수들보다 MARGIN 이상 가까움.
결과: who_src="elim_c". 다시 돌려도 같은 결과"""
import json, os, collections, numpy as np
DMAX = float(os.environ.get("EC_DMAX", "14")); MARGIN = float(os.environ.get("EC_MARGIN", "6")); ALL = os.environ.get("EC_ALL", "1") == "1"
f = json.load(open("feat_labeled.json")); T = json.load(open("tri3.json")) if os.path.exists("tri3.json") else {}
for v in f.values():
    for o in v or []:
        if o.get("who_src") == "elim_c": o["who"] = None; o.pop("who_src")
lu = json.load(open("lineup.json"))
L = {("DUNK(나)" if w == "Gamm_jA" else w) for r, w in lu.items() if r != "GK"}
Llow = {w.lower() for w in L} | {"gamm_ja"}
kit_ok = not os.path.exists("kit_skip.json")
tri = lambda k, j: (T.get(str(k)) or {}).get(str(j))
ref = collections.defaultdict(list)
for k, v in f.items():
    for j, o in enumerate(v or []):
        t = tri(k, j)
        if t and o.get("who") in L and not o.get("who_src") and o.get("tag"): ref[o["who"]].append(t[3:6])
R = {}
for w, x in ref.items():   # 기준색: 5개 이상 + 퍼짐(중앙값 거리) 10 이하만 — 마커를 잘못 읽는 선수(m11 MandoolsLove 갈색→흰색 오인, 퍼짐 16.8)는 기준색 없음
    if len(x) < 5: continue
    a = np.array(x); m = np.median(a, 0)
    if np.median(np.linalg.norm(a - m, axis=1)) <= 10: R[w] = m
n = 0; st = collections.Counter()
for k, v in f.items():
    us = [(j, o) for j, o in enumerate(v or []) if o.get("real", True) and o.get("team2") == "us"]
    named = {o["who"] for _, o in us if o.get("who")}; miss = L - named; un = [(j, o) for j, o in us if not o.get("who")]
    if len(miss) != 1 or len(un) != 1: continue
    w = next(iter(miss)); j, o = un[0]; nm = o.get("name")
    if ALL: continue
    if nm and nm.lower() not in Llow: st["상대 이름"] += 1; continue
    if kit_ok and o.get("kit_p") is not None and o["kit_p"] < 0.5: st["유니폼 상대쪽"] += 1; continue
    t = tri(k, j)
    if not t or w not in R: st["색 정보 없음"] += 1; continue
    c = np.array(t[3:6]); d = {p: float(np.linalg.norm(c - q)) for p, q in R.items()}
    if d[w] > DMAX: st["색 다름"] += 1; continue
    if any(d[p] < d[w] + MARGIN for p in d if p != w): st["다른 선수 색과 비슷"] += 1; continue
    o["who"] = w; o["who_src"] = "elim_c"; n += 1; st[w] += 1
if ALL:   # 실험: 남은 한 명 조건 없이, 빈 우리 상자 색이 한 선수 기준색에만 가까우면(그 장면에 그 선수 없음) 붙임
    for k, v in f.items():
        have = {o.get("who") for o in v or [] if o.get("who")}
        for j, o in enumerate(v or []):
            if not (o.get("real", True) and o.get("team2") == "us" and not o.get("who")): continue
            nm = o.get("name")
            if nm and nm.lower() not in Llow: continue
            if kit_ok and o.get("kit_p") is not None and o["kit_p"] < 0.5: continue
            t = tri(k, j)
            if not t: continue
            c = np.array(t[3:6]); d = sorted((float(np.linalg.norm(c - q)), p) for p, q in R.items() if p in L)
            if len(d) < 2 or d[0][0] > DMAX or d[1][0] < d[0][0] + MARGIN or d[0][1] in have: continue
            o["who"] = d[0][1]; o["who_src"] = "elim_c"; have.add(d[0][1]); n += 1; st["색만:" + d[0][1]] += 1
json.dump(f, open("feat_labeled.json", "w"))
print(f"남은 한 명(색 확인) 채움 {n}개 · 기준색 {sorted(R)} · {dict(st)}")
