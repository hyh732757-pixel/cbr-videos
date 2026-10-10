#!/bin/bash
# 사용: run_from_analyze.sh <작업폴더> — 특징 추출까지 끝난 폴더에서 analyze(팀)부터 공 소유 계산까지 다시(공 검출 ball.json·0.1초 ball_mid.json 은 그대로 사용)
export OMP_THREAD_LIMIT=1; W="$1"; P=/home/claude/pipe; cd "$W"
T0=$(date +%s); s(){ echo "[$(date +%H:%M:%S)] $1 (누적 $(( $(date +%s)-T0 ))초)"; }
rm -f feat_labeled_before_gk.json.before_star2 feat_labeled_before_starfix.json feat_labeled_before_posfill.json feat_labeled_before_gk.json team_override.json kit_skip.json
python3 $P/analyze.py > l_an.log 2>&1; python3 $P/tracklets.py >> l_an.log 2>&1; s "팀·추적"
python3 $P/name_cls.py /home/claude/models/name_cls.pt 0.8 > l_names.log 2>&1
python3 $P/names_step.py >> l_names.log 2>&1; s "이름 판별 $(tail -2 l_names.log | tr '\n' ' ')"
python3 $P/team_orient.py > l_orient0.log 2>&1; if [ $? -eq 3 ]; then
  python3 $P/analyze.py > l_an.log 2>&1; python3 $P/tracklets.py >> l_an.log 2>&1
  python3 $P/name_cls.py /home/claude/models/name_cls.pt 0.8 > l_names.log 2>&1; python3 $P/names_step.py >> l_names.log 2>&1
  s "팀 재지정 후 이름 판별 다시 $(cat l_orient0.log | tr '\n' ' ') $(tail -2 l_names.log | tr '\n' ' ')"; else s "$(head -2 l_orient0.log | tr '\n' ' ')"; fi
python3 $P/identify.py 3 > l_id.log 2>&1; python3 $P/who_filter.py >> l_id.log 2>&1; s "선수 판별"
python3 $P/apply_marker2.py > l_mk.log 2>&1; python3 $P/marker_id.py >> l_mk.log 2>&1; python3 $P/dedupe.py >> l_mk.log 2>&1; python3 $P/star_fix.py >> l_mk.log 2>&1; s "마커 $(grep -m1 marker l_mk.log)"
python3 $P/team_feat.py > l_team.log 2>&1; python3 $P/team_cls.py >> l_team.log 2>&1; python3 $P/kit_apply.py >> l_team.log 2>&1; python3 $P/lineup_check.py | tee -a l_team.log; s "팀 구분·라인업 점검 $(grep CNN l_team.log)"
python3 $P/marker_net.py | tee -a l_team.log; python3 $P/pos_fill.py | tee -a l_team.log; python3 $P/opp_name_fix.py | tee -a l_team.log; python3 $P/elim_fill2.py | tee -a l_team.log; python3 $P/pos_fill.py | tee -a l_team.log; python3 $P/gk_fix.py | tee -a l_team.log; python3 $P/star_recheck.py >> l_team.log 2>&1 && python3 $P/gk_fix.py >> l_team.log 2>&1; s "마커 모델·위치 이어붙이기 이름 채움·골키퍼 판별"
python3 $P/ball_events.py > l_be.log 2>&1; s "공 소유 $(tail -1 l_be.log)"
s "완료"
