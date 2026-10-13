"""EA 프로클럽 API(비공식)에서 클럽 기록 받기: python3 fetch_ea.py 클럽이름 출력폴더"""
import json, sys, os, time, urllib.request, urllib.parse
NAME, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
B = "https://proclubs.ea.com/api/fc"
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
     "Accept": "application/json", "Referer": "https://www.ea.com/", "Origin": "https://www.ea.com", "Accept-Language": "en-US,en;q=0.9"}
log = []
def get(path, **q):
    url = f"{B}/{path}?{urllib.parse.urlencode(q)}"
    for t in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=30) as r:
                d = json.loads(r.read().decode("utf-8")); log.append(f"OK {url}"); return d
        except Exception as e:
            log.append(f"ERR {url} {e}"); time.sleep(2)
    return None
found = {}
for plat in ("common-gen5", "common-gen4"):
    for ep in ("allTimeLeaderboard/search", "currentSeasonLeaderboard/search"):
        d = get(ep, platform=plat, clubName=NAME)
        json.dump(d, open(f"{OUT}/search_{plat}_{ep.split('/')[0]}.json", "w"), ensure_ascii=False, indent=1)
        for c in (d or []):
            nm = c.get("clubName") or (c.get("clubInfo") or {}).get("name")
            if nm and nm.lower() == NAME.lower(): found[(plat, str(c.get("clubId")))] = nm
log.append(f"찾은 클럽 {found}")
for (plat, cid), nm in found.items():
    p = f"{OUT}/{plat}_{cid}"; os.makedirs(p, exist_ok=True)
    for name, ep, q in (("info", "clubs/info", dict(clubIds=cid)), ("overall", "clubs/overallStats", dict(clubIds=cid)),
                        ("members", "members/stats", dict(clubId=cid)), ("career", "members/career/stats", dict(clubId=cid))):
        json.dump(get(ep, platform=plat, **q), open(f"{p}/{name}.json", "w"), ensure_ascii=False, indent=1)
    for mt in ("leagueMatch", "friendlyMatch", "playoffMatch"):
        d = get("clubs/matches", platform=plat, clubIds=cid, matchType=mt, maxResultCount=10)
        json.dump(d, open(f"{p}/matches_{mt}.json", "w"), ensure_ascii=False, indent=1)
        log.append(f"{plat} {cid} {mt}: {len(d) if isinstance(d, list) else type(d).__name__}")
open(f"{OUT}/log.txt", "w").write("\n".join(log)); print("\n".join(log))
