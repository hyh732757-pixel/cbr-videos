"""골키퍼 판별 + 상대 유니폼 위 우리 이름 제거 (2026-10-08)
배경(8경기 검토): 우리 GK 유니폼이 주황 → 빨간 상대(VIP·AP)·주황 상대 GK(NC)와 같은 군집으로 섞이고,
GK 상자에 필드 선수 이름(GarethBale0413·ufomg123 등)이 붙어 패스맵·히트맵 위치가 틀어짐(우리 골문 앞 표본 15/15 GK).
방법
1) 유니폼 색 기준: 이름표가 직접 읽힌 상자로 GK(myman6)·우리 필드·상대 색 평균을 만듦(team_feat.npz 색 히스토그램)
2) 진영: 이름표로 확인된 GK 위치(시간순)로 전·후반 우리 골대 쪽을 정함(한 번 바뀌는 시점을 가장 잘 나누는 곳)
3) 매 장면: GK 색에 가장 가깝고(우리·상대 색보다) 우리 골라인에서 20m 안에(35m 로 했을 때 AP전 표본 18개 중 오답 2개가 28~34m 지점의 빨간 상대 선수) 있는 상자 중 우리 골대 중앙에 가장 가까운 1명 = 우리 GK
   → who=myman6, team2=us_gk. 그 장면의 다른 myman6 이름은 지움. 원래 us_gk 로 분류됐지만 GK 가 아닌 상자는 색으로 우리/상대 다시 정함
4) team2=us 인데 유니폼 판별(kit_p)이 0.5 미만(상대 유니폼 쪽)인 상자의 이름은 지움(모르면 비움). 0.3→0.5: 공 접촉 표본에서 0.3~0.5 구간 2개 모두 오답(상대 선수), 정답 표본은 GK 외 0.74 미만 없음(2026-10-08 15시)
다시 돌려도 같은 결과(처음 실행 때 feat_labeled_before_gk.json 백업, 이후 항상 그 백업에서 시작)"""
import os as _os_k, json, os, shutil, collections, numpy as np
if not os.path.exists("feat_labeled_before_gk.json"): shutil.copy("feat_labeled.json", "feat_labeled_before_gk.json")
F = json.load(open("feat_labeled_before_gk.json"))
Z = np.load("team_feat.npz"); X = Z["X"]; idx = {(int(k), int(i)): j for j, (k, i) in enumerate(zip(Z["k"], Z["i"]))}
def hist(k, i):
    j = idx.get((int(k), i)); return None if j is None else X[j]
seed = collections.defaultdict(list); gk_obs = []
for k, v in F.items():
    for i, o in enumerate(v or []):
        h = hist(k, i)
        if h is None or not o.get("tag"): continue
        if o.get("who") == "myman6" and not o.get("who_src"): seed["gk"].append(h); (gk_obs.append((int(k), o["xy"][0])) if o.get("xy") else None)
        elif o.get("team2") == "us" and o.get("who"): seed["us"].append(h)
        elif o.get("team2") == "opp": seed["opp"].append(h)
# 4경기(404·NC·VIP1·AP) 이름표 GK 평균 — 경기 안 GK 이름표가 20개 미만일 때만 사용
GK_DEFAULT = np.array([.52, .19, .08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .01, .11, 0, .02, .035, .03], np.float32)
P = {"gk": np.mean(seed["gk"], 0) if len(seed["gk"]) >= 20 else GK_DEFAULT, "us": np.mean(seed["us"], 0), "opp": np.mean(seed["opp"], 0)}
def dist(h): return {c: float(np.abs(h - p).sum()) for c, p in P.items()}
# 진영: 경기 구간마다(matches.json — 한 영상에 경기가 여러 개일 수 있음, 2026-10-09 GAP 영상 2경기 확인) GK 이름표 위치가 왼쪽(x<52.5)인지 시간순으로 보고 한 번 바뀌는 지점 찾기
_R = [(x["t"], x["clock"]) for x in json.load(open("scene_raw.json")) if x["clock"] is not None]
_M = json.load(open("matches.json")) if os.path.exists("matches.json") else [[0, 10 ** 9]]
SIDE = []   # (장면 시작, 끝, 바뀌는 장면, 전반 우리 골대 왼쪽?)
for a_, b_ in _M:
    obs = sorted(o for o in gk_obs if a_ * 5 <= o[0] < b_ * 5)
    if len(obs) < 2: continue
    side = np.array([x < 52.5 for _, x in obs]); best = (-1, None, None)
    for s_ in range(1, len(obs)):
        for a in (True, False):
            sc = (side[:s_] == a).sum() + (side[s_:] == (not a)).sum()
            if sc > best[0]: best = (sc, obs[s_][0], a)
    score, KS, left_first = best
    R = [(t, c) for t, c in _R if a_ <= t < b_]
    _t45 = None
    for i, (t, c) in enumerate(R):
        w = [cc for tt, cc in R[i:] if tt < t + 30]
        if c >= 2700 and np.median(w) >= 2700: _t45 = i; break
    if _t45 is not None:   # (2026-10-13) 전반 추가시간 뒤 시계가 45:00 근처로 돌아가는 지점(후반 킥오프)이 있으면 그곳
        _mx = -1
        for i2 in range(_t45, len(R)):
            t2, c2 = R[i2]
            if c2 < _mx - 20 and abs(c2 - 2700) < 120 and np.median([cc for _, cc in R[i2:i2 + 5]]) < _mx - 20: _t45 = i2; break   # 글자 오인 1번은 무시
            _mx = max(_mx, c2)
    for i, (t, c) in enumerate(R):
        if i != _t45: continue
        if True:
            lo = max([kk for kk, _ in obs if kk < KS], default=a_ * 5); hi = KS
            if lo <= t * 5 <= hi: KS = t * 5
            print(f"[{a_}~{b_}초] 경기 시계 45분 시점 영상 {t}초 → 진영 바뀜 장면 {KS}"); break
    print(f"[{a_}~{b_}초] GK 이름표 {len(obs)}개 · 진영 바뀌는 장면 {KS} · 일치 {score}/{len(obs)} · 전반 우리 골대 {'왼쪽' if left_first else '오른쪽'}")
    SIDE.append((a_ * 5, b_ * 5, KS, left_first))
def own_x(k, x):
    for a, b, KS, lf in SIDE:
        if a <= k < b: left = lf if k < KS else (not lf); return x if left else 105 - x
    a, b, KS, lf = min(SIDE, key=lambda q: min(abs(k - q[0]), abs(k - q[1])))   # 경기 구간 밖 장면은 가까운 경기 기준
    left = lf if k < KS else (not lf); return x if left else 105 - x
st = collections.Counter()
for k, v in F.items():
    k_ = int(k); cand = []
    for i, o in enumerate(v or []):
        if not o.get("xy") or not o.get("real", True): continue
        h = hist(k, i)
        if h is None: continue
        d = dist(h); o["_d"] = d
        if d["gk"] < min(d["us"], d["opp"]) and own_x(k_, o["xy"][0]) < 20: cand.append((np.hypot(own_x(k_, o["xy"][0]), o["xy"][1] - 34), i))
    g = min(cand)[1] if cand else None
    for i, o in enumerate(v or []):
        d = o.pop("_d", None)
        if i == g:
            if o.get("who") and o["who"] != "myman6": st["GK 상자의 필드선수 이름 교체"] += 1
            if o.get("team2") != "us_gk": st["GK 팀 바로잡음"] += 1
            o["who"] = "myman6"; o["who_src"] = "gk_rule"; o["team2"] = "us_gk"; st["GK 지정"] += 1; continue
        if o.get("who") == "myman6": o["who"] = None; o.pop("who_src", None); st["GK 아닌 상자 myman6 이름 지움"] += 1
        if o.get("team2") == "us_gk" and d:
            o["team2"] = "us" if d["us"] < d["opp"] else "opp"; st["us_gk 군집 → " + o["team2"]] += 1
        # DUNK(나)는 마커가 보이면 항상 별도 보임(6경기 이름표 확인 DUNK 379개 중 379개 별 있음) → 마커 보이는데 별 없으면 DUNK 이름 지움
        # (위치 채움·마커 모델이 같은 노란 계열 Gareth 등에 DUNK 를 붙인 경우, VIP2 RCB 접촉 3건 확인)
        if o.get("who") == "DUNK(나)" and o.get("mk") and not o.get("star") and o["box"][1] >= 230:   # 화면 위쪽은 별이 상단 표시 영역에 가려질 수 있어 제외
            o["who"] = None; o.pop("who_src", None); st["별 없는 DUNK 이름 지움"] += 1
        if not _os_k.path.exists("kit_skip.json") and o.get("team2") == "us" and o.get("who") and o.get("kit_p") is not None and o["kit_p"] < 0.5:   # 유니폼 모델을 안 쓴 경기(kit_skip)는 이 규칙도 안 씀
            o["who"] = None; o.pop("who_src", None); st["상대 유니폼 위 우리 이름 지움"] += 1
json.dump(F, open("feat_labeled.json", "w"))
print("GK 판별", dict(st))
