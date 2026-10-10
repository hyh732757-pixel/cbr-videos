"""상대 골키퍼가 우리팀으로 분류된 경우 바로잡기 (2026-10-10)
m11 SPF전 Alpha_Fox9(45번 중 45번 우리팀 분류), VIP전 Mom923(54·96번 전부 우리팀) — 둘 다 상대 GK(어두운 유니폼), 위치 100% 골대 20m 안.
반대로 상대 이름이 '우리' 상자에 읽힌 다른 경우(NC전 NEMOTO_HARUMI 등)는 정답 확인 결과 실제 우리 선수(이름표가 옆 상대 것) → 건드리지 않음.
조건: 라인업과 비슷하지 않은 이름(유사도<0.6)이 20번 이상 읽혔고, 그중 90% 이상이 '우리' 상자, 90% 이상이 골대 20m 안(x<20 또는 x>85)
→ 그 이름이 읽힌 골대 20m 안 상자만 상대로(같은 추적 줄기까지 넓히면 m11 표본 6개 중 4개가 실제 우리 공격수 SAMBAEK·Gareth 여서 뺌)(team2='opp', who 지움). 되돌리기용 team2_was·who_was 기록"""
import json, difflib, collections
f = json.load(open("feat_labeled.json"))
for v in f.values():
    for o in v or []:
        if "team2_was" in o:
            o["team2"] = o.pop("team2_was"); w = o.pop("who_was", None)
            if w: o["who"] = w
L = [w.lower() for r, w in json.load(open("lineup.json")).items()] + ["gamm_ja", "dunk(나)"]
far = lambda nm: max(difflib.SequenceMatcher(None, nm.lower(), w).ratio() for w in L) < 0.6
near_goal = lambda o: o.get("xy") and (o["xy"][0] < 20 or o["xy"][0] > 85)
cnt = collections.defaultdict(collections.Counter)
for v in f.values():
    for o in v or []:
        nm = o.get("name")
        if not nm or not o.get("real", True) or not far(nm): continue
        cnt[nm][o.get("team2")] += 1
        if o.get("team2") == "us" and o.get("xy"): cnt[nm]["goal" if near_goal(o) else "field"] += 1
G = [nm for nm, c in cnt.items() if c["us"] >= 20 and c["us"] >= 0.9 * (c["us"] + c["opp"]) and c["goal"] >= 0.9 * (c["goal"] + c["field"])]
tids = {o.get("tid") for v in f.values() for o in v or [] if o.get("name") in G and o.get("team2") == "us" and o.get("tid") is not None}
bad = {o.get("tid") for v in f.values() for o in v or [] if o.get("who") and not o.get("who_src") and o.get("name") and not far(o["name"])}
tids -= bad   # 우리 선수 이름표가 직접 읽힌 적 있는 줄기는 제외(추적이 다른 사람으로 넘어간 경우)
n = 0
for v in f.values():
    for o in v or []:
        if o.get("team2") == "us" and o.get("name") in G and near_goal(o):
            o["team2_was"] = "us"
            if o.get("who"): o["who_was"] = o["who"]
            o["team2"] = "opp"; o["who"] = None; o.pop("who_src", None); n += 1
json.dump(f, open("feat_labeled.json", "w"))
print(f"상대 GK 로 보이는 이름 {G} → 상자 {n}개 상대로")
