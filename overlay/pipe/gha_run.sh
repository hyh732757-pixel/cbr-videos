#!/bin/bash
# GitHub Actions 에서 경기 영상 하나 처리: gha_run.sh <영상URL> <경기ID> [라인업JSON]
# 결과: $OUT 에 경기 결과 파일·자동 점검·로그·시간
set -o pipefail
URL="$1"; MID="$2"; LINEUP="$3"; W="$GITHUB_WORKSPACE/run"; OUT="$GITHUB_WORKSPACE/out"; mkdir -p "$W" "$OUT"
T0=$(date +%s); log(){ echo "[$(date +%H:%M:%S)] $1 (누적 $(( $(date +%s)-T0 ))초)" | tee -a "$OUT/timeline.txt"; }
curl -sSL --retry 3 -o "$GITHUB_WORKSPACE/v.mp4" "$URL" || { echo "영상 받기 실패" | tee -a "$OUT/timeline.txt"; exit 1; }
log "영상 받기 $(du -h "$GITHUB_WORKSPACE/v.mp4" | cut -f1)"
if [ -n "$LINEUP" ]; then echo "$LINEUP" > "$W/lineup.json"; log "라인업 지정 $(cat "$W/lineup.json")"; fi
export NPROC=$(nproc)
bash /home/claude/pipe/run_video.sh "$GITHUB_WORKSPACE/v.mp4" "$W" 2>&1 | tee "$OUT/run_video.log"
log "영상 처리 끝"
cd "$W"
python3 - "$MID" > "$OUT/matches.txt" <<'P'
import json, sys
ms = json.load(open("matches.json")); mid = sys.argv[1]
for i, (a, b) in enumerate(ms): print(mid if len(ms) == 1 else f"{mid}_{i+1}", a, b)
P
while read -r M A B; do
  bash /home/claude/pipe/finish_match.sh "$M" "$A" "$B" "" > "$OUT/audit_$M.log" 2>&1
  cp "match_$M.json" "$OUT/" 2>/dev/null; cp "goals_$M.json" "$OUT/" 2>/dev/null
  log "결과 계산 $M ($A~$B초) $(grep '===' "$OUT/audit_$M.log")"
  cp "$OUT/audit_$M.log" "$OUT/timeline.txt" . 2>/dev/null; cp l_*.log . 2>/dev/null
  python3 /home/claude/pipe/preflight.py "$M" > "$OUT/preflight_$M.txt" 2>&1; cp preflight.json "$OUT/preflight_$M.json" 2>/dev/null
  log "가정 점검 $M $(tail -1 "$OUT/preflight_$M.txt")"
done < "$OUT/matches.txt"
cp lineup.json matches.json l_*.log "$OUT/" 2>/dev/null
tar czf "$OUT/key_json.tgz" feat_labeled.json feat_labeled_before_gk.json ball.json ball_mid.json ball_events.json cam_track.json scene_lab.json scene_raw.json matches.json lineup.json eps_*.json contacts_*.json goals_*.json tri3.json 2>/dev/null
log "완료"
