"""처리 결과 가정 점검 (2026-10-13)
지금까지 새 영상마다 터진 버그는 대부분 '입력에 대한 가정'이 깨진 경우였음(노란·흰 유니폼, 게스트, 상대 GK 어두운 유니폼,
연한 마커, 경기마다 바뀌는 마커 색, 라인업 화면 없음, 전반 추가시간, 영상 하나에 경기 둘).
→ 결과를 웹에 넣기 전에 가정을 하나씩 데이터로 확인. 멈춤(STOP)이 하나라도 있으면 종료코드 2.
사용: (경기 작업 폴더에서) python3 preflight.py MID
출력: preflight.json, 화면에 표"""
import json, os, sys, re, collections, numpy as np
MID = sys.argv[1] if len(sys.argv) > 1 else "?"
R = []   # (단계, 항목, 값, 판정 OK/WARN/STOP, 설명)
def add(item, val, lvl, why): R.append(dict(item=item, val=val, lvl=lvl, why=why))
def rd(p):
    try: return open(p, encoding="utf-8", errors="ignore").read()
    except Exception: return ""
f = json.load(open("feat_labeled.json"))
lu = json.load(open("lineup.json")) if os.path.exists("lineup.json") else None
M = json.load(open("matches.json")) if os.path.exists("matches.json") else None
S0, S1 = (M[0] if M else (0, 10 ** 9))
inm = lambda k: S0 * 5 <= int(k) < S1 * 5

# 1. 라인업
lg = rd("l_lineup.log") + rd("timeline.txt")
if lu is None: add("라인업", "없음", "STOP", "lineup.json 없음 → 이름 판별 불가. 라인업을 받아서 다시 처리")
elif "로비 화면 없음" in lg and "라인업 지정" not in lg: add("라인업", "기본값 사용", "STOP", "영상에 로비 화면이 없고 라인업 지정도 없음 → 기본 라인업은 틀릴 수 있음")
else: add("라인업", "지정" if "라인업 지정" in lg else "영상에서 읽음", "OK", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from lineup import ROSTER
except Exception: ROSTER = set()
if lu:
    newn = [n for r, n in lu.items() if n not in ROSTER]
    add("라인업 밖 이름(게스트·새 팀원)", newn or "-", "WARN" if newn else "OK", "팀원인지 게스트인지 확인 필요(표시 이름이 Guest 로 바뀜)" if newn else "")

# 2. 경기 수·구간
if M is None: add("경기 구간", "없음", "STOP", "matches.json 없음")
elif len(M) > 1: add("경기 구간", f"{len(M)}경기 {M}", "WARN", "영상 하나에 경기가 여러 개 → 첫 경기만 처리됨")
else: add("경기 구간", f"{S0}~{S1}초", "OK", "")

# 3. 팀(유니폼) 방향: 별(DUNK)은 항상 우리 선수
st = collections.Counter(o.get("team2") for k, v in f.items() if inm(k) for o in v or [] if o.get("star"))
us_star = (st["us"] + st["us_gk"]) / max(1, sum(st.values()))
add("팀 구분(별 표시 DUNK 가 '우리')", f"{us_star:.0%} ({dict(st)})", "OK" if us_star >= 0.85 else "STOP", "" if us_star >= 0.85 else "유니폼 색 가정이 틀렸을 가능성 → 팀이 뒤바뀌었는지 확인")
L = {("DUNK(나)" if n == "Gamm_jA" else n) for r, n in (lu or {}).items() if r != "GK"}
Llow = {n.lower() for n in L} | {"gamm_ja"}
tg = collections.Counter(o.get("team2") for k, v in f.items() if inm(k) for o in v or [] if o.get("name") and o["name"].lower() in Llow)
r_us = (tg["us"]) / max(1, sum(tg.values()))
add("팀 구분(라인업 이름표가 '우리' 상자에)", f"{r_us:.0%} ({sum(tg.values())}개)", "OK" if r_us >= 0.8 else ("WARN" if r_us >= 0.6 else "STOP"), "" if r_us >= 0.8 else "이름표 상당수가 우리팀 아닌 상자에 붙음 → 팀 구분 오류 의심")

# 4. 선수별 이름표 수(직접 읽힌 것) — 적으면 그 선수 자료 빈약
lowmap = {n.lower(): n for n in L}; lowmap["gamm_ja"] = "DUNK(나)"
cnt = collections.Counter(lowmap[o["name"].lower()] for k, v in f.items() if inm(k) for o in v or [] if o.get("name") and o["name"].lower() in lowmap and o.get("team2") == "us")   # 이름표 글자가 직접 읽힌 상자
low = {w: cnt.get(w, 0) for w in sorted(L) if cnt.get(w, 0) < 30}
add("선수별 이름표 직접 확인 수", dict(cnt.most_common()), "WARN" if low else "OK", f"이름표 30개 미만 선수 {low} → 이 선수 위치·패스 자료가 적거나 다른 방법(색·위치)에 의존" if low else "")
named_by = collections.Counter(o.get("who_src", "tag") for k, v in f.items() if inm(k) for o in v or [] if o.get("team2") == "us" and o.get("who"))
tot_us = sum(1 for k, v in f.items() if inm(k) for o in v or [] if o.get("team2") == "us" and o.get("real", True))
add("우리 상자 이름 비율·출처", f"{sum(named_by.values()) / max(1, tot_us):.0%} · {dict(named_by)}", "OK", "")

# 5. 전·후반 경계: 경계 직후 시계가 45:00 근처(후반 킥오프)인지
mp = rd("l_mp.log"); m = re.search(r"half boundary video s (\d+)", mp)
if os.path.exists("scene_lab.json") and m:
    HB = int(m.group(1)); lab = json.load(open("scene_lab.json")); C = lab["clock"]
    after = [C[s] for s in range(HB, min(HB + 25, len(C))) if C[s] is not None]
    before = [C[s] for s in range(max(0, HB - 40), HB) if C[s] is not None]
    a = float(np.median(after)) if after else None; b = max(before) if before else None
    # 경계 뒤 150초 안에 시계가 20초 넘게 뒤로 가면(= 아직 전반 추가시간이었음) 경계가 너무 이름
    seq = [C[s] for s in range(HB, min(HB + 150, len(C))) if C[s] is not None]
    mx = -1; back = False
    for c in seq:
        if c < mx - 20 and abs(c - 2700) < 120: back = True; break
        mx = max(mx, c)
    ok = a is not None and 2690 <= a <= 2780 and not back
    mmss = lambda c: "-" if c is None else f"{int(c)//60}:{int(c)%60:02d}"
    add("전·후반 경계", f"{HB}초 (직후 시계 {mmss(a)}, 직전 40초 최대 {mmss(b)})", "OK" if ok else "STOP",
        "" if ok else ("경계 뒤에 시계가 다시 45:00 으로 돌아감 → 전반 추가시간을 후반으로 처리 중" if back else "경계 직후 시계가 후반 시작(45:00 근처)이 아님 → 전·후반 방향이 섞였을 수 있음"))
else: add("전·후반 경계", "확인 불가", "WARN", "scene_lab.json 또는 l_mp.log 없음")

# 6. GK 진영 일관성: myman6 이름표 위치가 전반·후반 각각 한쪽 골대에 있는지
if m:
    HB5 = int(m.group(1)) * 5; side = collections.Counter()
    for k, v in f.items():
        if not inm(k): continue
        for o in v or []:
            if o.get("who") == "myman6" and o.get("xy") and o.get("tag"):
                side[(int(k) >= HB5, o["xy"][0] < 52.5)] += 1
    h1 = side[(False, True)] / max(1, side[(False, True)] + side[(False, False)]); h2 = side[(True, True)] / max(1, side[(True, True)] + side[(True, False)])
    cons = max(h1, 1 - h1) * 0.5 + max(h2, 1 - h2) * 0.5; flip = (h1 > 0.5) != (h2 > 0.5)
    n = sum(side.values())
    lvl = "OK" if (cons >= 0.9 and flip) or n < 10 else "STOP"
    add("GK 진영(전·후반 반대쪽)", f"일관성 {cons:.0%} · 전반 왼쪽 {h1:.0%} 후반 왼쪽 {h2:.0%} (이름표 {n}개)", lvl, "" if lvl == "OK" else "GK 위치로 본 진영이 전·후반에 안 바뀌거나 섞임 → 공격 방향 오류 의심")

# 7. 상대 GK 가 우리로 분류되는지(라인업 밖 이름이 우리 상자·골대 근처에 반복)
L2 = Llow | {"myman6", "dunk(나)"}
odd = collections.Counter(o["name"] for k, v in f.items() if inm(k) for o in v or [] if o.get("team2") == "us" and o.get("name") and o["name"].lower() not in L2 and o.get("xy") and (o["xy"][0] < 20 or o["xy"][0] > 85))
big = {n: c for n, c in odd.items() if c >= 20}
add("상대 이름이 우리 상자(골대 근처)에", big or "-", "WARN" if big else "OK", "상대 GK 가 우리로 분류됐을 수 있음(opp_name_fix 가 처리했는지 확인)" if big else "")

# 8. 마커 색 기준(경기 안에서 배운 색)
if os.path.exists("tri3.json"):
    T = json.load(open("tri3.json")); ref = collections.defaultdict(list)
    for k, v in f.items():
        for j, o in enumerate(v or []):
            t = (T.get(k) or {}).get(str(j))
            if t and o.get("who") in L and not o.get("who_src") and o.get("tag"): ref[o["who"]].append(t[3:6])
    good = [w for w, x in ref.items() if len(x) >= 5 and np.median(np.linalg.norm(np.array(x) - np.median(x, 0), axis=1)) <= 10]
    miss = sorted(L - set(good))
    add("마커 색 기준 있는 선수", f"{len(good)}/{len(L)}", "OK" if len(miss) <= 3 else "WARN", f"기준색 없음 {miss} → 이 선수들은 색으로 이름을 못 채움" if miss else "")

# 9. 점수: 점수판 최종 vs 득점 목록
mj = f"match_{MID}.json"
if os.path.exists(mj):
    J = json.load(open(mj)); g = J["STATS"]["goals"]; us = sum(1 for x in g if x[0] == "CBR"); op = len(g) - us
    ok = (us, op) == (J["us_score"], J["opp_score"])
    add("점수 vs 득점 목록", f"점수 {J['us_score']}:{J['opp_score']} · 득점 {us}:{op}", "OK" if ok else "WARN", "" if ok else "득점 시점·득점자 확인 필요")

# 10. 공 추적·위치 튐(자동 점검 결과 재사용)
au = rd(f"audit_{MID}.log") or rd("audit_hb.log")
mm = re.search(r"위치 튐\(0.2초 10m\+\): ([\d.]+)%", au)
if mm: v = float(mm.group(1)); add("위치 튐(0.2초 10m+)", f"{v}%", "OK" if v < 0.6 else "WARN", "" if v < 0.6 else "선수 위치가 순간적으로 크게 튀는 비율이 높음(카메라·추적 문제 의심)")

lv = collections.Counter(r["lvl"] for r in R)
json.dump(dict(mid=MID, result=R, summary=dict(lv)), open("preflight.json", "w"), ensure_ascii=False, indent=1)
mark = {"OK": "✅", "WARN": "⚠️", "STOP": "⛔"}
for r in R: print(f"{mark[r['lvl']]} {r['item']}: {r['val']}" + (f"  → {r['why']}" if r["why"] else ""))
print(f"=== 가정 점검 {MID}: 멈춤 {lv['STOP']} · 주의 {lv['WARN']} · 정상 {lv['OK']} ===")
sys.exit(2 if lv["STOP"] else 0)
