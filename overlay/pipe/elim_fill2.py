"""남은 한 명 채우기 v2 (2026-10-10): 장면 단위 '남은 한 명' 표를 추적 줄기(tid)별로 모아,
같은 줄기에서 MINV번 이상 같은 이름으로 나오고(동의율 AGR 이상) 그 줄기에 다른 이름표가 없을 때만 줄기의 빈 상자 전부에 붙임"""
import json, os, collections, sys
MINV=int(os.environ.get("ELIM_MINV","3")); AGR=float(os.environ.get("ELIM_AGR","1.0")); STRICT=os.environ.get("ELIM_STRICT","1")=="1"
f = json.load(open("feat_labeled.json"))
for v in f.values():
    for o in v or []:
        if o.get("who_src") == "elim": o["who"] = None; o.pop("who_src")
lu = json.load(open("lineup.json"))
L = {("DUNK(나)" if w == "Gamm_jA" else w) for r, w in lu.items() if r != "GK"}
Llow = {w.lower() for w in L} | {"gamm_ja"}
kit_ok = not os.path.exists("kit_skip.json")
vote = collections.defaultdict(collections.Counter); tagged = collections.defaultdict(collections.Counter)
for k, v in f.items():
    us = [o for o in v or [] if o.get("real", True) and o.get("team2") == "us"]
    for o in us:
        if o.get("who") and o.get("tid") is not None: tagged[o["tid"]][o["who"]] += 1
    named = {o["who"] for o in us if o.get("who")}
    miss = L - named; un = [o for o in us if not o.get("who")]
    if len(miss) != 1 or len(un) != 1: continue
    if STRICT and any(x.get("who_src") not in (None, "tag") for x in us if x.get("who")): continue   # 9명 모두 이름표로 직접 읽은 경우만
    o = un[0]; nm = o.get("name")
    if nm and nm.lower() not in Llow: continue
    if kit_ok and o.get("kit_p") is not None and o["kit_p"] < 0.5: continue
    if o.get("tid") is not None: vote[o["tid"]][next(iter(miss))] += 1
ok = {}
for t, c in vote.items():
    w, nv = c.most_common(1)[0]; tot = sum(c.values())
    if nv < MINV or nv / tot < AGR: continue
    if any(x != w for x in tagged[t]): continue   # 같은 줄기에 다른 이름이 붙은 적 있으면 제외
    ok[t] = w
n = collections.Counter()
for k, v in f.items():
    have = {o.get("who") for o in v or [] if o.get("who")}
    for o in v or []:
        if o.get("real", True) and o.get("team2") == "us" and not o.get("who") and o.get("tid") in ok and ok[o["tid"]] not in have:
            nm = o.get("name")
            if nm and nm.lower() not in Llow: continue
            o["who"] = ok[o["tid"]]; o["who_src"] = "elim"; have.add(o["who"]); n[o["who"]] += 1
json.dump(f, open("feat_labeled.json", "w"))
print(f"남은 한 명(줄기) 채움 {sum(n.values())}개 · 줄기 {len(ok)}개 · {dict(n)}")
