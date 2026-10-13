"""한 영상 안의 경기 하나를 분석해 웹앱용 데이터 묶음(match_<id>.json)을 만든다.
입력(현재 폴더): feat_labeled.json(이름 OCR 포함), cam_track.json, scene_lab.json, s1/*.jpg
사용: python3 match_pipeline.py <id> <video_s0> <video_s1> <opp_formation> [opp_abbr]
"""
import json, sys, difflib, re
import numpy as np, cv2, pytesseract
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment

FPS = 5
MID, S0, S1, OPP_FORM = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
K0, K1 = S0 * FPS, S1 * FPS

from lineup import OUR_ROLE, GUESTS
OUR_ORDER = ["GK", "LB", "LCB", "RCB", "RB", "CDM", "LCM", "RCM", "CAM", "LS", "RS"]
TEMPLATES = {
    "3-5-2": {"GK": (8, 34), "LCB": (30, 22), "CB": (26, 34), "RCB": (30, 46), "LM": (52, 8), "LCM": (52, 24),
              "CM": (46, 34), "RCM": (52, 44), "RM": (52, 60), "LS": (70, 27), "RS": (70, 41)},
}
OPP_ORDER = {"3-5-2": ["GK", "LCB", "CB", "RCB", "LM", "LCM", "CM", "RCM", "RM", "LS", "RS"]}

feat_all = {int(k): v for k, v in json.load(open("feat_labeled.json")).items()}
feat = {k: v for k, v in feat_all.items() if K0 <= k < K1}
cam = {o["k"]: o for o in json.load(open("cam_track.json"))}
lab = json.load(open("scene_lab.json")); CLOCK = lab["clock"]; LABEL = lab["labels"]

# ---------- 시계 정리: 중앙값 필터, 전·후반 경계 ----------
secs = list(range(S0, min(S1, len(CLOCK))))
raw = {s: CLOCK[s] for s in secs if CLOCK[s] is not None and LABEL[s] in ("play", "stopped")}
clean = {}
ks = sorted(raw)
for i, s in enumerate(ks):
    win = [raw[t] for t in ks[max(0, i - 4):i + 5]]
    med = float(np.median(win))
    clean[s] = raw[s] if abs(raw[s] - med) < 90 else med
def clock_at(s):
    if s in clean: return clean[s]
    near = [t for t in ks if abs(t - s) <= 3]
    return clean[near[0]] if near else None
HB = next((s for s in ks if clean[s] >= 2700 and np.median([clean[t] for t in ks if s <= t < s + 30]) >= 2700), S1)
# (2026-10-13) 전반 추가시간(45:00~47:xx)은 전반: 45분 넘은 뒤 시계가 20초 이상 뒤로 가서 45:00 근처가 되는 첫 지점 = 후반 킥오프
_mx = -1
for _s in ks:
    if _s < HB: continue
    if clean[_s] < _mx - 20 and abs(clean[_s] - 2700) < 120: HB = _s; break
    _mx = max(_mx, clean[_s])
def half(k): return 1 if k / FPS < HB else 2
print("half boundary video s", HB)

# ---------- 이름 → 팀/우리 선수 ----------
def canon_ours(n):
    if not n: return None
    best = max(OUR_ROLE, key=lambda c: difflib.SequenceMatcher(None, n.lower(), c.lower()).ratio())
    return best if difflib.SequenceMatcher(None, n.lower(), best.lower()).ratio() >= 0.62 else None
votes = defaultdict(Counter)
for objs in feat_all.values():
    for o in objs:
        n = o.get("name")
        if n and o["tag"]:
            tc = (o["tag"][0] + o["tag"][2]) / 2
            crowd = [p for p in objs if p is not o and abs((p["box"][0] + p["box"][2]) / 2 - tc) < 45 and abs(p["box"][1] - o["box"][1]) < 70]
            if not crowd: votes[n][o["team"]] += 1
team_of = {}
for n, c in votes.items():
    co = canon_ours(n)
    if co: team_of[n] = "us"
    elif sum(c.values()) >= 8 and c["opp"] >= c["us"]: team_of[n] = "opp"
for n in OUR_ROLE: team_of[n] = "us"
def who_of(o):
    # (2026-10-08) 별 표시(star)만으로 Gamm_jA 로 보던 규칙 제거: 별 감지가 주황·노랑 마커(ufomg123·Gareth·KR_Gang)에도 켜져
    # 경기당 300~500개 상자가 DUNK 로 바뀌어 있었음(VIP1 RCB 위치 오류 원인). 별은 star_fix 단계에서 이미 who 에 반영됨
    n = o.get("name")
    if n and team_of.get(n) == "us": return canon_ours(n) or n
    if n and team_of.get(n) == "opp": return n
    w = o.get("who")
    if w in ("DUNK(나)",): return "Gamm_jA"
    if w and team_of.get(w) == "us": return canon_ours(w) or w
    if w and team_of.get(w) == "opp": return w
    return None
def team_obj(o):
    w = who_of(o)
    if w: return "us" if w in OUR_ROLE else "opp"
    return o["team"] if o["team"] in ("us", "opp") else None

# ---------- 공격 방향 ----------
dirs = {}; shares = {}
for hf in (1, 2):
    lo = []
    for k, objs in feat.items():
        if half(k) != hf: continue
        u = sorted(o["xy"][0] for o in objs if team_obj(o) == "us"); p = sorted(o["xy"][0] for o in objs if team_obj(o) == "opp")
        if len(u) >= 4 and len(p) >= 4: lo.append(u[0] < p[0])
    dirs[hf] = 1 if (np.mean(lo) if lo else 0.5) > 0.5 else -1
    print("half", hf, "our leftmost share", round(float(np.mean(lo)) if lo else -1, 2), "→ attack", "right" if dirs[hf] == 1 else "left")
    shares[hf] = float(np.mean(lo)) if lo else 0.5
# 전후반은 항상 진영을 바꿈: 한쪽 판단이 애매하면(0.4~0.6) 더 확실한 쪽의 반대로 (2026-10-08 VIP전 후반 0.50 → 영상 확인 결과 오른쪽 공격)
a1, a2 = abs(shares[1] - 0.5), abs(shares[2] - 0.5)
if dirs[1] == dirs[2] or min(a1, a2) < 0.1:
    if a1 >= a2: dirs[2] = -dirs[1]
    else: dirs[1] = -dirs[2]
    print("진영 보정 → 전반", "right" if dirs[1] == 1 else "left", "후반", "right" if dirs[2] == 1 else "left")
def own(xy, team, k):
    x, y = xy; d = dirs[half(k)]
    if team == "us": return (x, y) if d == 1 else (105 - x, 68 - y)
    return (105 - x, 68 - y) if d == 1 else (x, y)

# ---------- 상대 이름 정리(비슷한 철자 병합) ----------
opp_names = Counter()
for objs in feat.values():
    for o in objs:
        w = who_of(o)
        if w and w not in OUR_ROLE: opp_names[w] += 1
canon_opp = {}
reps = []
for n, c in opp_names.most_common():
    m = next((r for r in reps if difflib.SequenceMatcher(None, n.lower(), r.lower()).ratio() >= 0.7), None)
    if m: canon_opp[n] = m
    else: reps.append(n); canon_opp[n] = n
def W(o):
    w = who_of(o)
    if w is None: return None
    return w if w in OUR_ROLE else canon_opp.get(w, w)

# ---------- 공 받음 에피소드 ----------
tagged = []
for k in sorted(feat):
    for o in feat[k]:
        if o["tag"] and o["box"][3] < 900:
            w = W(o); t = ("us" if w in OUR_ROLE else "opp") if w else (o["team"] if o["team"] in ("us", "opp") else None)
            if t: tagged.append(dict(k=k, team=t, pid=w, xy=o["xy"], seg=cam[k]["seg"]))
eps = []
for t in tagged:
    merged = False
    for e in reversed(eps[-6:]):
        if e["seg"] == t["seg"] and e["team"] == t["team"] and t["k"] - e["k_end"] <= 3:
            same = (t["pid"] is not None and t["pid"] == e["pid"]) or np.hypot(t["xy"][0] - e["xy_end"][0], t["xy"][1] - e["xy_end"][1]) < 3.5
            if same:
                e["k_end"] = t["k"]; e["xy_end"] = t["xy"]; e["n"] += 1; e["pids"].append(t["pid"]); merged = True; break
    if not merged:
        eps.append(dict(k=t["k"], k_end=t["k"], team=t["team"], pid=t["pid"], pids=[t["pid"]], xy=t["xy"], xy_end=t["xy"], seg=t["seg"], n=1))
for e in eps:
    c = Counter(p for p in e["pids"] if p); e["pid"] = c.most_common(1)[0][0] if c else None; del e["pids"]
    e["name"] = e["pid"]; e["clock"] = clock_at(int(e["k"] / FPS))
eps = [e for e in eps if e["n"] >= 2]
# ---------- (선택) 공 접촉 기준 에피소드: BALL_EVENTS=1 이고 ball_events.json 이 있으면 이름표 기준 대신 사용 ----------
import os as _os
BALL_MODE = _os.environ.get("BALL_EVENTS", "1") != "0" and _os.path.exists("ball_events.json")   # 기본: 공 기준(끄려면 BALL_EVENTS=0)
if BALL_MODE:
    _be = json.load(open(_os.environ.get("BE_FILE", "ball_events.json")))["segs"]; eps = []
    for s_ in _be:
        if not (K0 <= s_["k0"] < K1) or s_["k0"] not in cam: continue
        pid_ = s_["who"] if s_["team"] == "us" and s_["who"] in OUR_ROLE else None
        eps.append(dict(k=s_["k0"], k_end=s_["k1"], team=s_["team"], pid=pid_, name=pid_, xy=s_["xy0"], xy_end=s_["xy"], seg=cam[s_["k0"]]["seg"], n=s_["n"], clock=clock_at(int(s_["k0"] / FPS))))
    print("공 접촉 기준 에피소드 사용", len(eps))
json.dump([dict(k=e["k"], k_end=e["k_end"], team=e["team"], pid=e["pid"], xy=e["xy"], clock=e["clock"]) for e in eps], open(f"eps_{MID}.json", "w"))   # 검증용
print("receptions us", sum(e["team"] == "us" for e in eps), "opp", sum(e["team"] == "opp" for e in eps))

# ---------- 상대 포지션 배정 ----------
pos = defaultdict(list)
for k, objs in feat.items():
    for o in objs:
        w = W(o)
        if w and w not in OUR_ROLE: pos[w].append(own(o["xy"], "opp", k))
cand = [n for n in pos if len(pos[n]) >= 60]
cand = sorted(cand, key=lambda n: -len(pos[n]))[:11]
T = TEMPLATES[OPP_FORM]; roles = list(T); TT = np.array([T[r] for r in roles], float)
opp_role = {}
if cand:   # 상대 이름을 못 읽은 영상(저화질)은 상대 개인 배정 생략
    P = np.array([np.mean(pos[n], 0) for n in cand]); sh = np.zeros(2)
    for _ in range(3):
        C = np.linalg.norm((P + sh)[:, None] - TT[None], axis=2); ri, ci = linear_sum_assignment(C); sh = (TT[ci] - P[ri]).mean(0)
    opp_role = {cand[a]: roles[b] for a, b in zip(ri, ci)}
role_of = dict(OUR_ROLE); role_of.update(opp_role)
print("opp roles", opp_role)

# ---------- 패스/손실 ----------
passes = {"us": Counter(), "opp": Counter()}
for a, b in zip(eps, eps[1:]):
    if a["seg"] != b["seg"] or (b["k"] - a["k_end"]) / FPS > 6 or a["team"] != b["team"]: continue
    if a["pid"] and a["pid"] == b["pid"]: continue
    ra, rb = role_of.get(a["pid"]), role_of.get(b["pid"])
    if ra and rb and ra != rb: passes[a["team"]][tuple(sorted((ra, rb)))] += 1

# 패스맵 위치용 공 접촉 순간: 공(검출된 장면)에서 1.5m 안 가장 가까운 선수가 이름 붙은 우리 선수인 장면, 연속 장면은 한 번(첫 장면 위치)
# 근거(2026-10-08): VIP2 Gareth 정답 17터치 평균 x70.8 — 공 받음(0.4초↑)+패스 기준 42.9, 접촉 1.2m 62.7, 1.5m 67.4, 2.0m 69.1
CONTACT = defaultdict(list)
# 공 접촉 장면 마커 색 점검(2026-10-08): 이름표로 확인된 상자로 선수별 마커 색 평균 → 접촉 상자 마커 색이 다른 선수 쪽에 0.15 이상 더 가까우면 그 접촉은 패스맵 위치에서 뺌
# 근거: VIP1 RCB 접촉 중 ufomg123·Gareth 오인 4건 모두 걸러짐. 이름 자체는 지우지 않음(무작위 이름 정답 357개에선 맞는 이름도 12% 지워져 전체 적용은 손해)
def _mv(m): h, s_, v = m[:3]; return np.array([np.cos(h / 90 * np.pi) * s_ / 255 * 1.5, np.sin(h / 90 * np.pi) * s_ / 255 * 1.5, v / 255])
_mc = defaultdict(list)
for _v in feat.values():
    for _o in _v:
        if _o.get("team2") == "us" and _o.get("who") and _o.get("tag") and _o.get("mk"): _mc[W(_o)].append(_mv(_o["mk"]))
MKC = {p_: np.mean(c_, 0) for p_, c_ in _mc.items() if len(c_) >= 10}
MK_DROP = Counter()
if _os.path.exists("ball.json"):
    _bb = json.load(open("ball.json"))["ball"]; _last = {}
    _EXT_R = float(_os.environ.get("EXT_R", "0")); _EXT_T = float(_os.environ.get("EXT_T", "2"))
    # 0.1초 공 추적(2026-10-09 적용, ball_mid.json 있으면 사용 — 끄려면 MID_BALL=0. 근거: 공격수 위치 정답 4명 차이 평균 3.8→2.1m) — 0.2초 장면 사이 중간 장면(k/5+0.1초) 공으로도 접촉 판정
    #   중간 장면 선수 위치 = 같은 이름이 k·k+1 장면에 다 있으면 가운데, 아니면 k 장면 위치. 마커 색 점검은 원래 0.2초 장면에서만
    _BM = {int(a): b for a, b in json.load(open("ball_mid.json"))["ball"].items()} if _os.environ.get("MID_BALL", "1") == "1" and _os.path.exists("ball_mid.json") else {}
    def _steps():
        for k in sorted(feat):
            b_ = _bb.get(str(k))
            if b_ and b_.get("xy") and b_.get("src") == "det":
                yield 2 * k, k, b_["xy"], [(o["xy"], o) for o in feat[k] if o.get("xy") and o.get("real", True) and o.get("team2") in ("us", "us_gk", "opp")], True
            m_ = _BM.get(k)
            if m_ and m_.get("xy"):
                nx = feat.get(k + 1, []); O2 = []
                for o in feat[k]:
                    if not o.get("xy") or not o.get("real", True) or o.get("team2") not in ("us", "us_gk", "opp"): continue
                    w2 = W(o) if o.get("team2") != "opp" else None
                    mt = [q for q in nx if w2 and q.get("xy") and W(q) == w2]
                    O2.append((((o["xy"][0] + mt[0]["xy"][0]) / 2, (o["xy"][1] + mt[0]["xy"][1]) / 2) if mt else o["xy"], o))
                yield 2 * k + 1, k, m_["xy"], O2, False
    for t_, k, bx, O_, even_ in _steps():
        if not O_: continue
        xy0, o0 = min(O_, key=lambda q: np.hypot(q[0][0] - bx[0], q[0][1] - bx[1]))
        _d0 = np.hypot(xy0[0] - bx[0], xy0[1] - bx[1])
        if _EXT_R and 1.5 < _d0 <= _EXT_R:   # 시험용(기본 꺼짐): 공 1.5~R m 이라도 상대가 공에서 T m 넘게 떨어져 있으면 접촉
            if min([np.hypot(q[0][0] - bx[0], q[0][1] - bx[1]) for q in O_ if q[1].get("team2") == "opp"] + [99]) <= _EXT_T: continue
        elif _d0 > 1.5: continue
        if o0.get("team2") not in ("us", "us_gk"): continue
        w_ = W(o0)
        if not w_ or w_ not in OUR_ROLE: continue
        bad_ = False
        # (2026-10-09 적용, 기본 켜짐 — 끄려면 STAR_OK=0) 별 표시(DUNK 전용, 이름표 확인 379/379)가 있는 상자는 마커 색 점검 안 함 —
        #   마커 색 점검이 DUNK(Gamm_jA) 접촉을 대량으로 빼고 있었음: 7경기 빠진 DUNK 접촉 35개 전부 별 있음(LIV 13개 중 11개, 404 17개 중 8개 빠짐)
        if even_ and o0.get("mk") and w_ in MKC and not (_os.environ.get("STAR_OK", "1") == "1" and o0.get("star") and w_ == "Gamm_jA"):
            _v = _mv(o0["mk"]); _ds = np.linalg.norm(_v - MKC[w_])
            bad_ = any(np.linalg.norm(_v - c_) < _ds - 0.15 for p_, c_ in MKC.items() if p_ != w_)
        x, y = own(xy0, "us", k)
        if _last.get(w_, -99) >= t_ - 4:   # 같은 접촉(0.4초 안 연속): 한 장면이라도 마커 불일치면 그 접촉 전체를 뺌
            _last[w_] = t_; CONTACT[w_][-1][2] = CONTACT[w_][-1][2] or bad_; CONTACT[w_][-1][4] += 1; continue
        _last[w_] = t_; CONTACT[w_].append([round(x, 1), round(y, 1), bad_, k, 1])
    # (2026-10-09) [기본 꺼짐, FAST_DROP=1 일 때만] 공이 빠르게 지나간 한 장면짜리 접촉 제외 — 접촉 정확도는 오르지만 Gareth 위치 정답 67.1→57.0(0.1초 병용 69.8→61.2)으로 나빠져 미적용: 1.5m 안에 딱 한 장면 + 앞뒤 장면 공 속도 초속 10m 초과
    # 근거: 공 접촉 정답 150개 중 현재 접촉 66개 — 틀린 9개 중 6개가 이 조건(진짜 접촉은 57개 중 3개만 해당), 정확도 86→92%
    #       0.1초 접촉 정답 39개 79→86%, 별도 Gareth 정답 36순간: 안 닿음 오인 3→1, 터치 11→10
    def _spd(k):
        a_, c_ = _bb.get(str(k - 1)), _bb.get(str(k + 1))
        if not a_ or not c_ or not a_.get("xy") or not c_.get("xy") or a_.get("src") != "det" or c_.get("src") != "det": return 0.0
        return float(np.hypot(c_["xy"][0] - a_["xy"][0], c_["xy"][1] - a_["xy"][1]) / 0.4)
    def _ang(k):   # 공 방향 변화(앞 0.4초 → 뒤 0.4초, 없으면 0.2초). 공이 느리면(초속 3m 이하) 판단 안 함
        def bf(kk):
            b = _bb.get(str(kk)); return np.array(b["xy"]) if b and b.get("xy") and b.get("src") == "det" else None
        for dk, dt in ((2, 0.4), (1, 0.2)):
            a, c, e = bf(k - dk), bf(k), bf(k + dk)
            if a is not None and c is not None and e is not None:
                v1, v2 = (c - a) / dt, (e - c) / dt; s1, s2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if s1 <= 3 or s2 <= 3: return None
                return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (s1 * s2), -1, 1))))
        return None
    ANG_DROP = Counter()
    if _os.environ.get("ANG_DROP", "0") == "1":   # 시험용(기본 꺼짐): 한 장면짜리 접촉인데 공 방향이 45도 미만으로만 바뀜(그냥 지나감) → 제외
        for w_ in list(CONTACT):
            keep_ = []
            for c_ in CONTACT[w_]:
                a_ = _ang(c_[3]) if c_[4] == 1 else None
                if a_ is not None and a_ < 45: ANG_DROP[w_] += 1; continue
                keep_.append(c_)
            CONTACT[w_] = keep_
        print("방향 안 바뀐 한 장면 접촉 제외", dict(ANG_DROP))
    FAST_DROP = Counter()
    for w_ in list(CONTACT):
        keep_ = []
        for c_ in CONTACT[w_]:
            if _os.environ.get("FAST_DROP", "0") == "1" and c_[4] == 1 and _spd(c_[3]) > 10: FAST_DROP[w_] += 1; print("  제외", w_, c_[3], c_[0], c_[1]) if w_ == "GarethBale0413" else None; continue
            keep_.append(c_)
        CONTACT[w_] = keep_
    print("빠른 공 한 장면 접촉 제외", dict(FAST_DROP))
    for w_ in list(CONTACT):
        MK_DROP[w_] = sum(1 for c_ in CONTACT[w_] if c_[2])
    json.dump({w_: [[c_[3], c_[0], c_[1], bool(c_[2])] for c_ in v_] for w_, v_ in CONTACT.items()}, open(f"contacts_{MID}.json", "w"))   # 검증용
    for w_ in list(CONTACT): CONTACT[w_] = [c_[:2] for c_ in CONTACT[w_] if not c_[2]]
print("마커 색 불일치로 뺀 접촉 장면", dict(MK_DROP))
# 띠(cr) 표본 — A안(2026-10-08 20시): 우리 공 소유 구간(ball_events team=us)에서 공(검출) 3m 안 이름 붙은 우리 선수 위치, 선수당 1초에 1개
# 근거(6경기 404~AP): 접촉 1.5m 대비 이름표 직접 확인 표본 19%→33%, 띠 길이 흔들림 ±8.0→±6.5m, 이름표 확인 표본만의 띠와 경계 차이 4.1→4.2m(같은 수준)
NEAR = defaultdict(dict)
if _os.path.exists("ball.json") and _os.path.exists("ball_events.json"):
    _ours = set()
    for s_ in json.load(open(_os.environ.get("BE_FILE", "ball_events.json")))["segs"]:
        if s_["team"] == "us": _ours.update(range(s_["k0"], s_["k1"] + 1))
    for k in sorted(feat):
        if k not in _ours: continue
        b_ = _bb.get(str(k))
        if not b_ or not b_.get("xy") or b_.get("src") != "det": continue
        for o in feat[k]:
            if not o.get("xy") or not o.get("real", True) or o.get("team2") not in ("us", "us_gk"): continue
            w_ = W(o)
            if not w_ or w_ not in OUR_ROLE or np.hypot(o["xy"][0] - b_["xy"][0], o["xy"][1] - b_["xy"][1]) > 3: continue
            NEAR[w_].setdefault(k // FPS, own(o["xy"], "us", k))
print("띠 표본(우리 소유 중 공 3m, 1초 1개)", {w_: len(v_) for w_, v_ in NEAR.items()})
def players_block(team):
    order = OUR_ORDER if team == "us" else OPP_ORDER[OPP_FORM]
    names = {r: n for n, r in (OUR_ROLE.items() if team == "us" else opp_role.items())}
    out = []
    for r in order:
        n = names.get(r)
        if not n: continue
        pts = []
        for e in eps:
            if e["team"] == team and e["pid"] == n:
                x, y = own(e["xy"], team, e["k"]); pts.append([round(x, 1), round(y, 1), None if e["clock"] is None else int(e["clock"])])
        # 패스맵 위치 = 공 받은 지점 + 패스한 지점(받는 사람 이름이 확인된 같은 팀 패스의 출발점) 평균
        # 근거: Soccermatics 패스 네트워크 "average location of passes made and receptions" (2026-10-08)
        rel = []
        for a, b in zip(eps, eps[1:]):
            if a["team"] == team and a["pid"] == n and b["team"] == team and b["pid"] and b["pid"] != n and a["seg"] == b["seg"] and (b["k"] - a["k_end"]) / FPS <= 6:
                x, y = own(a["xy_end"], team, a["k_end"]); rel.append([round(x, 1), round(y, 1)])
        allp = CONTACT.get(n) if team == "us" and CONTACT.get(n) else [q[:2] for q in pts] + rel
        node = [round(float(np.mean([q[0] for q in allp])), 1), round(float(np.mean([q[1] for q in allp])), 1), len(allp)] if allp else None
        if node and len(allp) >= 2:   # 위치 흔들림(재표본 500번 평균 이동 거리, m) — 웹앱에서 ±5m 초과·10개 미만이면 흐리게 표시
            A_ = np.array(allp, float); rg = np.random.default_rng(0)
            bs = np.array([A_[rg.integers(0, len(A_), len(A_))].mean(0) for _ in range(500)])
            node.append(round(float(np.sqrt(((bs - A_.mean(0)) ** 2).sum(1)).mean()), 1))
        cr = None   # 공 접촉 위치 범위(중간 50%): [x25, x75, y25, y75] — 패스맵에서 선수 누르면 반투명 띠로 표시
        if team == "us" and len(NEAR.get(n, {})) >= 10:   # 10개 미만은 경계 흔들림이 커서 표시 안 함
            A_ = np.array(list(NEAR[n].values()), float); cr = [round(float(np.percentile(A_[:, i], q)), 1) for i, q in ((0, 25), (0, 75), (1, 25), (1, 75))]
        out.append(dict(role=r, name=n, me=(n == "Gamm_jA"), pts=pts, rel=rel, node=node, cr=cr))
    return out

# ---------- 골 ----------
def read_score_rows(s):
    im = cv2.imread(f"s1/{s+1:04d}.jpg")
    def ocr(y0, y1, x0, x1, wl, psm):
        c = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY); c = cv2.resize(c, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        _, b = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return pytesseract.image_to_string(b, config=f"--psm {psm} -c tessedit_char_whitelist={wl}").strip()
    names = [ocr(58, 82, 160, 240, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 7), ocr(82, 106, 160, 240, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", 7)]
    digs = [ocr(58, 82, 258, 292, "0123456789", 10), ocr(82, 106, 258, 292, "0123456789", 10)]
    digs = [int(d) if d.isdigit() and int(d) <= 9 else None for d in digs]
    return names, digs
play_secs = [s for s in secs if LABEL[s] == "play"]
import os as _os
from multiprocessing import Pool as _Pool
_SC = "score_rows_cache.json"
_c = json.load(open(_SC)) if _os.path.exists(_SC) else {}
_todo = [s for s in play_secs if str(s) not in _c]
if _todo:
    with _Pool(int(__import__("os").environ.get("NPROC", "2"))) as _p:
        for s, r in zip(_todo, _p.map(read_score_rows, _todo, chunksize=10)): _c[str(s)] = r
    json.dump(_c, open(_SC, "w"))
rows = [tuple(_c[str(s)]) for s in play_secs]
nm = Counter()
for names, _ in rows:
    for i, t in enumerate(names):
        if "CBR" in t: nm[i] += 1
OUR_ROW = nm.most_common(1)[0][0] if nm else 0
opp_abbr = Counter(re.sub("[^A-Z]", "", r[0][1 - OUR_ROW])[-3:] for r in rows if r[0][1 - OUR_ROW]).most_common(1)[0][0]
if len(sys.argv) > 5: opp_abbr = sys.argv[5]
def smooth(i):
    v = [r[1][i] for r in rows]; out = []
    for j in range(len(v)):
        w = [x for x in v[max(0, j - 7):j + 8] if x is not None]
        out.append(Counter(w).most_common(1)[0][0] if w else None)
    return out
sa, sb = smooth(0), smooth(1)
goals = []; prev = [0, 0]
for j, s in enumerate(play_secs):
    cur = [sa[j], sb[j]]
    for i in (0, 1):
        if cur[i] is not None and cur[i] == prev[i] + 1:
            team = "us" if i == OUR_ROW else "opp"
            c = clock_at(s); goals.append(dict(vs=s, team=team, clock=c)); prev[i] = cur[i]
# 경기 막판 골 보정: 마지막 플레이 이후(정지·종료 직전) 스코어판도 읽어 점수가 오른 경우 골 추가
try:
    _raw = json.load(open("scene_raw.json")); _tail = [s for s in range(max(S0, S1 - 40), min(S1 + 20, len(_raw))) if _raw[s]["box"] > 180 and s not in set(play_secs)]
    _rd = [(s, read_score_rows(s)[1]) for s in _tail]
    for i in (0, 1):
        vals = [(s, d[i]) for s, d in _rd if d[i] is not None]
        for j, (s, v) in enumerate(vals):
            nxt = [w for _, w in vals[j:j + 3]]
            if v == prev[i] + 1 and nxt.count(v) >= 2:
                c = clock_at(s) if s in clean else CLOCK[s]
                goals.append(dict(vs=s, team="us" if i == OUR_ROW else "opp", clock=c)); prev[i] = v
except FileNotFoundError: pass
final = [prev[OUR_ROW], prev[1 - OUR_ROW]]
print("our row", OUR_ROW, "opp", opp_abbr, "final us:opp", final, [(g["team"], g["clock"]) for g in goals])
json.dump(goals, open(f"goals_{MID}.json", "w"))   # 검증용(골 시점: 영상 초)
goal_list = []
for g in goals:
    t = g["team"]
    before = [e for e in eps if e["k"] / FPS < g["vs"] - 4 and e["k"] / FPS > g["vs"] - 40]
    last = next((e for e in reversed(before) if e["team"] == t), None)
    sc = last["pid"] if last else ""
    asst = ""
    if last:
        idx = before.index(last)
        prv = before[idx - 1] if idx > 0 else None
        if prv and prv["team"] == t and prv["pid"] and prv["pid"] != sc and (last["k"] - prv["k_end"]) / FPS <= 6: asst = prv["pid"]
    minute = int(g["clock"] // 60) + 1 if g["clock"] is not None else None
    goal_list.append(["CBR" if t == "us" else opp_abbr, minute, sc or "미확인", asst, "est"])

# ---------- 통계 ----------
def inbox(xy): return xy[0] >= 88.5 and 13.84 <= xy[1] <= 54.16
own_frame = {}
for a, b in zip(eps, eps[1:]):
    if a["seg"] == b["seg"] and (b["k"] - a["k"]) / FPS <= 8:
        for k in range(a["k"], b["k"]): own_frame[k] = a["team"]
pct = lambda a, b: round(a / b * 100) if b else 0
HOW = dict(
 poss="이름표가 뜬 선수의 팀을 공 소유 팀으로 보고, 다음 이름표까지(8초 이내)의 시간을 합산",
 tilt="공격 1/3 지역(상대 골문에서 35m 안)에서 공을 받은 횟수의 양 팀 비율",
 rate="공이 같은 팀 다른 선수에게 이어지면 성공, 상대에게 넘어가면 실패. 드리블 중 뺏긴 경우도 실패에 포함",
 ok="같은 팀 선수 사이에 공이 이어진 횟수", rec="사람 선수 머리 위 이름표가 0.4초 이상 뜬 횟수. AI 선수는 포함되지 않음",
 lost="공을 가진 선수 다음에 상대 팀 선수가 공을 받은 횟수", own3="자기 골문 쪽 1/3 구역(35m 이내)에서 공을 잃은 횟수",
 danger="공을 잃은 뒤 10초 안에 상대가 우리 페널티 박스 안에서 공을 받은 횟수", cp="공을 잃은 뒤 5초 안에 다시 공을 받은 횟수(카운터프레싱 성공, StatsBomb 5초 기준)",
 cpr="5초 안에 되찾음 ÷ 볼 손실", box="상대 페널티 박스 안에서 이름표가 뜬 횟수",
 press="공을 받는 순간 가장 가까운 상대 선수가 4.6m(약 5야드, StatsBomb 압박 기준) 안에 있던 비율")
if BALL_MODE:
    HOW.update(poss="공에 가장 가까운 선수(1.2m 이내, 상대와 0.5m 이상 차이)의 팀을 공 소유 팀으로 보고, 다음 소유까지(8초 이내)의 시간을 합산",
               rec="공이 선수 발 1.2m 안에 0.4초 이상 머문 횟수(공 접촉 기준)", box="상대 페널티 박스 안에서 공을 가진(공 접촉 기준) 횟수")
def calc(H):
    inH = lambda k: H is None or half(k) == H
    S = {t: dict(poss=0.0, ok=0, lost=0, rec=0, third=[0, 0, 0], box=0, own3=0, cp=0, danger=0, tilt=0, pr=[0, 0]) for t in ("us", "opp")}
    for e in eps:
        if not inH(e["k"]): continue
        t = e["team"]; S[t]["rec"] += 1; x, y = own(e["xy"], t, e["k"])
        S[t]["third"][0 if x < 35 else (1 if x < 70 else 2)] += 1
        if x >= 70: S[t]["tilt"] += 1
        if inbox((x, y)): S[t]["box"] += 1
        objs = feat.get(e["k"], [])
        opp = [o for o in objs if team_obj(o) and team_obj(o) != t]
        if opp:
            S[t]["pr"][1] += 1
            if min(np.hypot(o["xy"][0] - e["xy"][0], o["xy"][1] - e["xy"][1]) for o in opp) <= 4.6: S[t]["pr"][0] += 1
    for i, (a, b) in enumerate(zip(eps, eps[1:])):
        if a["seg"] != b["seg"] or not inH(a["k"]): continue
        g = (b["k"] - a["k"]) / FPS
        if g <= 8: S[a["team"]]["poss"] += g
        if (b["k"] - a["k_end"]) / FPS > 6 or (a["pid"] and a["pid"] == b["pid"]): continue
        if a["team"] == b["team"]: S[a["team"]]["ok"] += 1; continue
        S[a["team"]]["lost"] += 1
        if own(a["xy"], a["team"], a["k"])[0] < 35: S[a["team"]]["own3"] += 1
        for c in eps[i + 2:i + 8]:
            if c["seg"] != a["seg"] or (c["k"] - b["k"]) / FPS > 5: break
            if c["team"] == a["team"]: S[a["team"]]["cp"] += 1; break
        for c in eps[i + 1:i + 12]:
            if c["seg"] != a["seg"] or (c["k"] - b["k"]) / FPS > 10: break
            if c["team"] == b["team"] and inbox(own(c["xy"], b["team"], c["k"])): S[a["team"]]["danger"] += 1; break
    # 팀 간격
    shape = defaultdict(list)
    for k, objs in feat.items():
        if k not in own_frame or not inH(k): continue
        for t in ("us", "opp"):
            P_ = [own(o["xy"], t, k) for o in objs if team_obj(o) == t]
            P_ = np.array([p for p in P_ if p[0] > 12])
            if len(P_) < 8: continue
            xs = np.sort(P_[:, 0]); st = "in" if own_frame[k] == t else "out"
            shape[(t, st)].append((xs[:4].mean(), xs[-1] - xs[0], P_[:, 1].max() - P_[:, 1].min()))
    u, o = S["us"], S["opp"]; tp = u["poss"] + o["poss"]; tt = u["tilt"] + o["tilt"]
    rows = [dict(k="점유율", u=pct(u["poss"], tp), o=pct(o["poss"], tp), unit="%", how=HOW["poss"]),
            dict(k="필드 틸트", u=pct(u["tilt"], tt), o=pct(o["tilt"], tt), unit="%", how=HOW["tilt"] + f". 이 구간 CBR {u['tilt']}회 · {opp_abbr} {o['tilt']}회"),
            dict(k="패스 연결 성공률", u=pct(u["ok"], u["ok"] + u["lost"]), o=pct(o["ok"], o["ok"] + o["lost"]), unit="%", how=HOW["rate"]),
            dict(k="연결된 패스", u=u["ok"], o=o["ok"], unit="", how=HOW["ok"]),
            dict(k="공 받은 횟수", u=u["rec"], o=o["rec"], unit="", how=HOW["rec"]),
            dict(k="압박 속 공 받기", u=pct(*u["pr"]), o=pct(*o["pr"]), unit="%", how=HOW["press"] + f". 이 구간 CBR {u['pr'][0]}/{u['pr'][1]}회 · {opp_abbr} {o['pr'][0]}/{o['pr'][1]}회"),
            dict(k="볼 손실", u=u["lost"], o=o["lost"], unit="", how=HOW["lost"], low=True),
            dict(k="자기 진영 1/3 볼 손실", u=u["own3"], o=o["own3"], unit="", how=HOW["own3"], low=True),
            dict(k="결정적 볼 손실", u=u["danger"], o=o["danger"], unit="", how=HOW["danger"], low=True),
            dict(k="5초 안에 되찾음", u=u["cp"], o=o["cp"], unit="", how=HOW["cp"]),
            dict(k="되찾음 비율", u=pct(u["cp"], u["lost"]), o=pct(o["cp"], o["lost"]), unit="%", how=HOW["cpr"]),
            dict(k="상대 박스 안에서 공 받음", u=u["box"], o=o["box"], unit="", how=HOW["box"])]
    names = [("수비 라인 높이", 0, "가장 뒤에 있는 필드 플레이어 4명의 평균 위치를 자기 골라인부터 잰 거리. 클수록 라인을 높이 올린 것"),
             ("팀 길이", 1, "가장 앞 선수와 가장 뒤 필드 플레이어 사이 거리. 작을수록 앞뒤 간격이 촘촘함"),
             ("팀 폭", 2, "가장 왼쪽과 가장 오른쪽 선수 사이 거리")]
    for st, lb in (("in", "공 소유 시"), ("out", "상대 소유 시")):
        for nm_, i, how in names:
            a_ = np.array(shape[("us", st)]); b_ = np.array(shape[("opp", st)])
            if len(a_) == 0 or len(b_) == 0: continue
            rows.append(dict(k=f"{nm_} · {lb}", u=round(float(a_[:, i].mean())), o=round(float(b_[:, i].mean())), unit="m",
                             how=how + f". 그 팀 필드 플레이어 8명 이상이 화면에 잡힌 장면만 사용(CBR {len(a_)//5}초 · {opp_abbr} {len(b_)//5}초)", shape=True, neutral=True))
    th = lambda s_: [pct(v, sum(s_["third"])) for v in s_["third"]]
    return dict(rows=rows, thirds=dict(us=th(u), opp=th(o)))
halves = {"all": calc(None), "h1": calc(1), "h2": calc(2)}
pp = defaultdict(lambda: [0, 0])
for e in eps:
    objs = feat.get(e["k"], []); opp = [o for o in objs if team_obj(o) and team_obj(o) != e["team"]]
    if not opp or e["pid"] not in role_of: continue
    p = int(min(np.hypot(o["xy"][0] - e["xy"][0], o["xy"][1] - e["xy"][1]) for o in opp) <= 4.6)
    pp[(e["team"], e["pid"])][0] += p; pp[(e["team"], e["pid"])][1] += 1
press_players = {t: sorted([dict(role=role_of[n], name=n, p=v[0], n=v[1]) for (tt, n), v in pp.items() if tt == t], key=lambda d: -d["n"]) for t in ("us", "opp")}

# ---------- 대형(소유 상태별) ----------
F = {"us": {"in": defaultdict(list), "out": defaultdict(list)}, "opp": {"in": defaultdict(list), "out": defaultdict(list)}}
heat = defaultdict(lambda: np.zeros((2, 53, 34), np.int32))
for k, objs in feat.items():
    for o in objs:
        w = W(o)
        if not w or w not in role_of: continue
        t = "us" if w in OUR_ROLE else "opp"; X, Y = own(o["xy"], t, k)
        if 0 <= X <= 105 and 0 <= Y <= 68:
            heat[w][half(k) - 1, min(52, int((105 - X) / 2)), min(33, int(Y / 2))] += 1
        if k in own_frame: F[t]["in" if own_frame[k] == t else "out"][role_of[w]].append((X, Y))
FORM = {t: {st: {r: [round(float(np.mean([p[0] for p in v])), 1), round(float(np.mean([p[1] for p in v])), 1), len(v)] for r, v in F[t][st].items()} for st in ("in", "out")} for t in F}
# 화면 밖 보정(offscreen.py 설명 참고): 우리 필드 선수 앞뒤(x)만. 근거·검증은 offscreen.py 머리말
from offscreen import view_edges, x_at, fit as _off_fit
_bb2 = json.load(open("ball.json"))["ball"] if _os.path.exists("ball.json") else {}
_flip = lambda k: own((0, 0), "us", k)[0] != 0
_frames = {"in": [], "out": []}
for k in sorted(feat):
    if k not in own_frame or k not in cam or not cam[k].get("p") or not cam[k].get("ok", True): continue   # LIV·GAP 예전 카메라 파일엔 ok 항목 없음
    b_ = _bb2.get(str(k))
    if not b_ or not b_.get("xy") or b_.get("src") != "det": continue
    st = "in" if own_frame[k] == "us" else "out"
    E = np.array([own(e, "us", k) for e in view_edges(cam[k]["p"])])
    if _flip(k): E = E[[2, 3, 0, 1]]
    seen = {}
    for o in feat[k]:
        w = W(o)
        if w in OUR_ROLE and o.get("xy") and OUR_ROLE[w] not in seen: seen[OUR_ROLE[w]] = own(o["xy"], "us", k)
    _frames[st].append((own(b_["xy"], "us", k)[0], E, seen))
OFF_LOG = {}
for st in ("in", "out"):
    fr_ = _frames[st]
    for r, v in list(FORM["us"][st].items()):
        if r == "GK" or not fr_: continue
        ym = v[1]; bx = np.array([f[0] for f in fr_])
        L_ = np.array([x_at(f[1][0], f[1][1], ym) for f in fr_]); R_ = np.array([x_at(f[1][2], f[1][3], ym) for f in fr_])
        xs = np.array([f[2][r][0] if r in f[2] else np.nan for f in fr_])
        res = _off_fit(bx, L_, R_, xs)
        if res is None: OFF_LOG[f"{st} {r}"] = "유지(추정 실패/표본 부족)"; continue
        ex, prm = res
        if not (0 <= prm[1] <= 1.2) or abs(ex - v[0]) > 8: OFF_LOG[f"{st} {r}"] = f"유지(b={prm[1]:.2f}, 보정량 {ex - v[0]:+.1f})"; continue
        OFF_LOG[f"{st} {r}"] = f"{v[0]} → {ex:.1f}"; v[0] = round(ex, 1)
print("화면 밖 보정(대형 x)", OFF_LOG)
HEAT = dict(cell=2.0, nx=34, ny=53, players=[])
for t in ("us", "opp"):
    for p in players_block(t):
        g = heat[p["name"]]
        HEAT["players"].append(dict(team=t, role=p["role"], name=p["name"], me=p["me"], n=[int(g[0].sum()), int(g[1].sum())],
                                    h=[[[int(i), int(v)] for i, v in enumerate(g[h].ravel()) if v] for h in (0, 1)]))

# ---------- 턴오버 ----------
GOALS = [(g["vs"], g["team"]) for g in goals]
TURN = []
for i, (a, b) in enumerate(zip(eps, eps[1:])):
    if a["seg"] != b["seg"] or (b["k"] - a["k_end"]) / FPS > 6 or a["team"] == b["team"]: continue
    t, o = a["team"], b["team"]; tb = b["k"] / FPS
    later = [c for c in eps[i + 2:i + 30] if c["seg"] == a["seg"]]
    sg = next((g for g, tm in GOALS if tm == t and 0 <= g - a["k_end"] / FPS <= 20), None)
    if sg and not any(c["team"] == t and c["k"] / FPS < sg - 6 for c in later): continue
    regain = next((c for c in later if c["team"] == t and (c["k"] - b["k"]) / FPS <= 5), None)
    boxe = next((c for c in [b] + later if c["team"] == o and (c["k"] - b["k"]) / FPS <= 10 and inbox(own(c["xy"], o, c["k"]))), None)
    firstback = next((c for c in later if c["team"] == t), None)
    goal = next((g for g, tm in GOALS if tm == o and 0 <= g - tb <= 25 and (firstback is None or firstback["k"] / FPS > g - 6)), None)
    res = "실점" if goal else "5초 안에 되찾음" if regain else "박스 침투 허용" if boxe else "15초 안에 되찾음" if firstback and (firstback["k"] - b["k"]) / FPS <= 15 else "상대 공격 지속"
    x, y = own(a["xy_end"], t, a["k"])
    TURN.append(dict(team=t, name=a["pid"], role=role_of.get(a["pid"]), by=b["pid"], x=round(x, 1), y=round(y, 1), half=half(a["k"]),
                     clock=None if a["clock"] is None else int(a["clock"]), vt=round(a["k_end"] / FPS, 1), res=res,
                     rg=regain["pid"] if regain else None))

us_g, op_g = final
DATA = dict(match=f"CBR vs {opp_abbr}", score=f"{us_g} : {op_g}", teams=dict(
    us=dict(team="CBR", formation="4-1-2-1-2", players=players_block("us"), passes=[[a, b, n] for (a, b), n in passes["us"].most_common()]),
    opp=dict(team=opp_abbr, formation=OPP_FORM, players=players_block("opp"), passes=[[a, b, n] for (a, b), n in passes["opp"].most_common()])))
bundle = dict(id=MID, opp=opp_abbr, us_score=us_g, opp_score=op_g, home=(OUR_ROW == 0), opp_form=OPP_FORM,
              DATA=DATA, STATS=dict(goals=goal_list, halves=halves, press_players=press_players), TURN=TURN, HEAT=HEAT, FORM=FORM)
_js = json.dumps(bundle, ensure_ascii=False)
for _g, _lab in GUESTS.items(): _js = _js.replace(f'"{_g}"', f'"{_lab}"')   # 게스트는 Guest1, Guest2로 표시
open(f"match_{MID}.json", "w").write(_js)
print("saved", MID, "score CBR", us_g, ":", op_g, opp_abbr, "turnovers", len(TURN), "passes", sum(passes["us"].values()), sum(passes["opp"].values()))
