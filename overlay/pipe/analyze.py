"""feat.json → 팀 분류, 우리 팀 선수 식별(마커 색), 포지션 배정, 공 받은 위치, 패스.
출력: analysis.json, 중간 진단 출력"""
import json, numpy as np
from collections import Counter, defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

FPS = 5
feat = {int(k): v for k, v in json.load(open("feat.json")).items()}
cam = {o["k"]: o for o in json.load(open("cam_track.json"))}
lab = json.load(open("scene_lab.json"))
clock = lab["clock"]

# ---------- 1. 팀 분류: 상체 어두운 비율 vs 흰 비율 (CBR 검정 / LIV 흰색) ----------
for k, objs in feat.items():
    for o in objs:
        d, w = o["j"]["dark"], o["j"]["white"]
        o["team"] = "us" if d > w + 0.05 else ("opp" if w > d + 0.05 else "other")
# (2026-10-09) 유니폼이 검정이 아닌 경기: team_orient.py 가 이름표 기준 색 군집으로 만든 팀(team_override.json)을 씀. 파일 없으면 위 규칙 그대로
import os as _os
if _os.path.exists("team_override.json"):
    _ov = json.load(open("team_override.json")); _n = 0
    for k, objs in feat.items():
        for i, o in enumerate(objs):
            t = _ov.get(f"{k}:{i}")
            if t: o["team"] = t; _n += 1
    print("팀 재지정(team_override)", _n)
st = Counter(o["team"] for objs in feat.values() for o in objs if o["star"])
print("star team", dict(st))
# (2026-10-11) 우리가 흰 유니폼(원정)인 경기: 별(DUNK, 항상 우리 선수)이 대부분 'opp' 로 분류됨(m12 FCH전 opp 2794 · us 41) → us/opp 맞바꿈.
# 기존 11경기는 별이 'us' 가 압도적(opp 의 5~50배) → 이 규칙 안 걸림
if not _os.path.exists("team_override.json") and st["opp"] >= 100 and st["opp"] > 3 * st["us"]:
    for objs in feat.values():
        for o in objs:
            o["team"] = {"us": "opp", "opp": "us"}.get(o["team"], o["team"])
    st = Counter(o["team"] for objs in feat.values() for o in objs if o["star"])
    print("흰 유니폼 경기로 판단 → 팀 맞바꿈, star team", dict(st))
# ---------- 2. 우리 팀 마커 색 → 개인 ID ----------
def mk_vec(m):
    h = m[0] / 180 * 2 * np.pi
    return [np.cos(h) * 3, np.sin(h) * 3, m[1] / 255 * 2, m[2] / 255 * 2]
mrows, mrefs = [], []
for k, objs in feat.items():
    for i, o in enumerate(objs):
        if o["team"] == "us" and not o["star"] and o["mk"] and o["mk"][1] > 100 and o["mk"][2] > 185:
            mrows.append(mk_vec(o["mk"])); mrefs.append((k, i))
M = np.array(mrows)
print("colored markers on our team:", len(M))
kmm = KMeans(n_clusters=8, n_init=10, random_state=0).fit(M)
sizes = Counter(kmm.labels_)
small = {c for c, n in sizes.items() if n < 200}
NID = 8 - len(small)
print("marker clusters", {c: (n, "H%.0f S%.0f V%.0f" % tuple(np.median(np.array([feat[k][i]["mk"][:3] for (k, i), l in zip(mrefs, kmm.labels_) if l == c]), 0))) for c, n in sorted(sizes.items())})
print("chosen identities:", NID)
for n, (k, i) in enumerate(mrefs):
    feat[k][i]["pid"] = int(kmm.labels_[n])
    # 너무 멀리 떨어진 색은 미확정
    d = np.linalg.norm(M[n] - kmm.cluster_centers_[kmm.labels_[n]])
    if d > 0.6 or kmm.labels_[n] in small: feat[k][i]["pid"] = None
# 내 선수(노란 별)는 어떤 ID 인가
star_mk = [mk_vec(o["mk"]) for objs in feat.values() for o in objs if o["star"] and o["mk"] and o["mk"][1] > 100 and o["mk"][2] > 185 and 60 <= o["mk"][0] <= 90]
star_pid = Counter(kmm.predict(np.array(star_mk)).tolist()) if star_mk else Counter()
print("star → pid", dict(star_pid))
me = star_pid.most_common(1)[0][0] if star_pid else None
for objs in feat.values():
    for o in objs:
        if o["star"] and o["team"] == "us": o["pid"] = me

# ---------- 3. 공격 방향: 각 프레임에서 가장 왼쪽 선수가 우리 팀인 비율 ----------
def half(k): return 1 if k / FPS < 500 else 2
dirs = {}
for hf in (1, 2):
    lo = []
    for k, objs in feat.items():
        if half(k) != hf: continue
        u = sorted(o["xy"][0] for o in objs if o["team"] == "us"); p = sorted(o["xy"][0] for o in objs if o["team"] == "opp")
        if len(u) >= 4 and len(p) >= 4: lo.append(u[0] < p[0])
    dirs[hf] = 1 if np.mean(lo) > 0.5 else -1
    print(f"half {hf}: our player is leftmost in {np.mean(lo):.2f} of frames → we attack {'right' if dirs[hf]==1 else 'left'}")
def norm(xy, k):
    x, y = xy
    return (x, y) if dirs[half(k)] == 1 else (105 - x, 68 - y)

# ---------- 4. 포지션 배정 (ID 평균 위치 → 4-1-2-1-2) ----------
# 4-1-2-1-2 평균 위치 템플릿(공격 방향 오른쪽, 경기 전체 평균은 중앙으로 압축됨)
TEMPLATE = {"GK": (8, 34), "LB": (48, 10), "LCB": (36, 26), "RCB": (36, 42), "RB": (48, 58), "CDM": (46, 34),
            "LCM": (55, 22), "RCM": (55, 46), "CAM": (63, 34), "LS": (70, 27), "RS": (70, 41)}
pos = defaultdict(list)
for k, objs in feat.items():
    for o in objs:
        if o["team"] == "us" and o.get("pid") is not None:
            pos[o["pid"]].append(norm(o["xy"], k))
pids = sorted(pos, key=lambda p: -len(pos[p]))
pids = [p for p in pids if len(pos[p]) >= 100]
avg = {p: np.mean(pos[p], 0) for p in pids}
print("pid counts", {p: len(pos[p]) for p in pids})
roles = list(TEMPLATE)
T = np.array([TEMPLATE[r] for r in roles], float)
P = np.array([avg[p] for p in pids])
shift = np.zeros(2)
for _ in range(3):   # 배정 → 평균 오차만큼 평행이동 → 재배정
    C = np.linalg.norm((P + shift)[:, None] - T[None], axis=2)
    ri, ci = linear_sum_assignment(C)
    shift = (T[ci] - P[ri]).mean(0)
role_of = {pids[a]: roles[b] for a, b in zip(ri, ci)}
print("role assignment", {role_of[p]: (p, np.round(avg[p], 1).tolist(), len(pos[p])) for p in role_of})

json.dump(dict(dirs=dirs, NID=NID, me=me, role_of={str(k): v for k, v in role_of.items()},
               avg={str(p): avg[p].tolist() for p in pids}, counts={str(p): len(pos[p]) for p in pids}),
          open("analysis_ids.json", "w"), ensure_ascii=False, indent=1)
json.dump({str(k): v for k, v in feat.items()}, open("feat_labeled.json", "w"))
