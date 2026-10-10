"""마커 판별 모델(mknet.pt)로 이름 없는 우리 선수에 이름 붙이기 (2026-10-07)
- 대상: 새 팀 구분이 '우리'이고, 마커가 보이고, 이름이 없는 상자
- 모델 확신 0.8 이상 + 예측이 우리 선수 + 그 장면에 같은 이름이 아직 없을 때만 붙임(같은 장면 후보 여럿이면 확신 높은 쪽 하나)
- 검증(경기 하나씩 빼고 학습): 지금 방식이 못 붙인 175명 중 101명에 붙임, 100명 정확(99.0%)
결과: who 채움 + who_src="marker_net". 다시 돌려도 같은 결과(이전 결과 제거 후 계산)"""
import json, sys, cv2, numpy as np, torch, collections
sys.path.insert(0, "/home/claude/mknet"); torch.set_num_threads(2)
from net_def import Net, prep   # 학습 자료 없이 정의만(train_eval 과 같은 구조)
from build_ds import CLS, crop
TH = 0.8
net = Net(); net.load_state_dict(torch.load("/home/claude/mknet/mknet.pt")); net.eval()
f = json.load(open("feat_labeled.json"))
for v in f.values():
    for o in v or []:
        if o.get("who_src") == "marker_net": o["who"] = None; o.pop("who_src")
import os
L = {w for r, w in json.load(open("lineup.json")).items() if r != "GK"} if os.path.exists("lineup.json") else set()
new = sorted(w for w in L if ("Gamm_jA" if w == "DUNK(나)" else w) not in CLS)
if new:   # 학습에 없던 선수가 뛴 경기: 그 선수 상자를 다른 이름으로 착각함(2026-10-08 새 경기에서 Hoiibe→Proclub_DD 236개 확인) → 이 단계 건너뜀
    json.dump(f, open("feat_labeled.json", "w")); print(f"마커 모델 건너뜀: 학습에 없던 선수 {new}"); raise SystemExit
# 마커 색 점검(2026-10-10): 마커 색은 경기마다 다시 배정됨(KR_Gang 404·ATM·NC 분홍 → VIP2·AP 노랑 → SPF 연한 크림).
# 이 모델은 404·ATM·NC(색 같음)로만 학습 → 이번 경기 선수 색(tri3, 이름표 상자)이 학습 때 색과 다르면 건너뜀. 색 정보가 없어도 건너뜀
_ok = False
if os.path.exists("tri3.json") and os.path.exists("/home/claude/mknet/train_colors.json"):
    _TR = json.load(open("/home/claude/mknet/train_colors.json")); _T = json.load(open("tri3.json")); _R = collections.defaultdict(list)
    for _k, _v in f.items():
        for _j, _o in enumerate(_v or []):
            _t = (_T.get(_k) or {}).get(str(_j))
            if _t and _o.get("who") in _TR and not _o.get("who_src") and _o.get("tag"): _R[_o["who"]].append(_t[3:6])
    _d = {}
    for w, x in _R.items():
        a = np.array(x); m = np.median(a, 0)
        if len(x) >= 5 and np.median(np.linalg.norm(a - m, axis=1)) <= 10: _d[w] = float(np.linalg.norm(m - np.array(_TR[w])))   # 퍼짐 큰(잘못 읽은) 선수는 비교 안 함
    _ok = len(_d) >= 3 and max(_d.values()) <= 15
    print("마커 색 학습 때와 비교(Lab 거리)", {w: round(v, 1) for w, v in _d.items()})
if not _ok:
    json.dump(f, open("feat_labeled.json", "w")); print("마커 모델 건너뜀: 이번 경기 마커 색이 학습 때와 다르거나 확인 불가"); raise SystemExit
name_of = lambda c: "DUNK(나)" if c == "Gamm_jA" else c
n = 0; st = collections.Counter()
for k, v in f.items():
    if not v: continue
    tgt = [j for j, o in enumerate(v) if o.get("real", True) and o.get("team2") == "us" and o.get("mk") and not o.get("who")]
    if not tgt: continue
    img = cv2.imread(f"f5/{int(k) + 1:05d}.jpg"); xs, js = [], []
    for j in tgt:
        c = crop(img, v[j])
        if c is not None: xs.append(c); js.append(j)
    if not xs: continue
    with torch.no_grad(): P = torch.softmax(net(prep(np.array(xs))), 1).numpy()
    have = {o.get("who") for o in v if o.get("who")}
    cand = sorted(((float(p.max()), CLS[int(p.argmax())], j) for p, j in zip(P, js)), reverse=True)
    for p, c, j in cand:
        if p < TH: st["낮은 확신"] += 1; continue
        if c == "opp": st["상대로 예측(이름 안 붙임)"] += 1; continue
        w = name_of(c)
        if w in have: st["같은 장면에 이미 있음"] += 1; continue
        v[j]["who"] = w; v[j]["who_src"] = "marker_net"; have.add(w); n += 1
json.dump(f, open("feat_labeled.json", "w"))
print(f"마커 모델 이름 붙임 {n}개 · {dict(st)}")
