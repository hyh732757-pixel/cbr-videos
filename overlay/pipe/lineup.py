"""경기별 라인업: 작업폴더의 lineup.json({"GK": "myman6", ...})이 있으면 사용, 없으면 고정 라인업.
고정 명단(이름 모델이 아는 11명) 밖의 이름은 게스트 → 웹앱 표시는 Guest1, Guest2..."""
import json, os
DEFAULT = {"GK": "myman6", "LB": "GAETERLING", "LCB": "KR_Gang", "RCB": "Gamm_jA", "RB": "MandoolsLove",
           "CDM": "ufomg123", "LCM": "JANG_BUTTER", "RCM": "Durayun_z", "CAM": "Kidz43", "LS": "GarethBale0413", "RS": "comba3"}
ROSTER = set(DEFAULT.values()) | {"Proclub_DD", "SAMBAEK", "Hoiibe", "Osumane_dembele1"}   # 팀원(이름 모델 학습 전). SAMBAEK 은 팀원(2026-10-13 사용자 확인)
ROLE_ORDER = ["GK", "LB", "LCB", "RCB", "RB", "CDM", "LCM", "RCM", "CAM", "LS", "RS"]
LINEUP = json.load(open("lineup.json")) if os.path.exists("lineup.json") else DEFAULT
OUR_ROLE = {n: r for r, n in LINEUP.items()}                 # 이름 → 포지션
NAMES = [LINEUP[r] for r in ROLE_ORDER if r in LINEUP]       # 오늘 뛴 11명
OTHERS = [n for n in NAMES if n != "Gamm_jA"]                # 나(별 표시) 제외
GUESTS = {n: f"Guest{i+1}" for i, n in enumerate(n for n in NAMES if n not in ROSTER)}
