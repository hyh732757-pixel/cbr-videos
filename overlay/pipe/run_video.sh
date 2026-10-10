#!/bin/bash
# 사용: run_video.sh <영상경로> <작업폴더>
export OMP_THREAD_LIMIT=1   # tesseract 스레드 경합 방지
V="$1"; W="$2"; P=/home/claude/pipe; mkdir -p "$W/s1" "$W/f5"; cd "$W"
T0=$(date +%s); s(){ echo "[$(date +%H:%M:%S)] $1 (누적 $(( $(date +%s)-T0 ))초)"; }
CROP=""; WD=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$V"); [ "$WD" -gt 2000 ] && CROP="crop=1920:1080:210:0,"; echo "crop=$CROP"
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$V"); echo "영상 높이 $H"
mkdir -p f10m
# 0.2초 장면(f5) + 그 사이 중간 장면(f10m, k/5+0.1초 → f10m/{k}.jpg). 중간 장면은 fps=10:round=down 홀수 장면(따로 뽑은 것과 픽셀 일치 확인, 2026-10-09)
ffmpeg -v error -i "$V" -filter_complex "[0:v]${CROP}split=3[a][b][c];[a]fps=1,scale=1920:1080[s];[b]fps=5,scale=1920:1080[f];[c]fps=fps=10:round=down,select='mod(n\,2)',scale=1920:1080[m]" \
  -map "[s]" -q:v 3 s1/%04d.jpg -map "[f]" -q:v 3 f5/%05d.jpg -map "[m]" -fps_mode passthrough -q:v 3 -start_number 0 f10m/%05d.jpg; s "프레임 추출 $(ls s1|wc -l)초/$(ls f5|wc -l)장/중간 $(ls f10m|wc -l)장"
python3 $P/scene.py > l_scene.log 2>&1 && python3 $P/classify.py >> l_scene.log 2>&1; python3 $P/clean_clock.py >> l_scene.log 2>&1; s "장면 구분 $(tail -1 l_scene.log)"
if [ ! -f lineup.json ]; then python3 $P/lineup_detect.py > l_lineup.log 2>&1; fi; s "라인업 $( [ -f lineup.json ] && cat lineup.json || echo 없음-캡처필요)"
python3 $P/detect_new.py /home/claude/models/det1_int8_openvino_model > l_det.log 2>&1; s "선수·공 검출"
python3 $P/camnet_only.py 0 999999 cam_track.json > l_cam.log 2>&1; s "카메라 $(tail -1 l_cam.log)"
python3 $P/cam_rescue.py cam_track.json > l_rescue.log 2>&1; s "카메라 보정 $(tail -1 l_rescue.log)"
python3 $P/features.py > l_feat.log 2>&1; python3 $P/tag_resolve.py >> l_feat.log 2>&1; s "특징 추출 $(tail -1 l_feat.log)"
python3 $P/analyze.py > l_an.log 2>&1; python3 $P/tracklets.py >> l_an.log 2>&1; s "팀·추적"
if [ "$H" -lt 1000 ]; then python3 $P/retag720.py > l_tag.log 2>&1; s "720p 이름표 재검출"; fi
python3 $P/name_cls.py /home/claude/models/name_cls.pt 0.8 > l_names.log 2>&1
python3 $P/names_step.py >> l_names.log 2>&1; s "이름 판별 $(tail -2 l_names.log | tr '\n' ' ')"
python3 $P/team_orient.py > l_orient0.log 2>&1; if [ $? -eq 3 ]; then   # 유니폼이 검정이 아닌 경기: 팀 재지정 후 analyze~이름 판별 다시(2026-10-09)
  python3 $P/analyze.py > l_an.log 2>&1; python3 $P/tracklets.py >> l_an.log 2>&1
  python3 $P/name_cls.py /home/claude/models/name_cls.pt 0.8 > l_names.log 2>&1; python3 $P/names_step.py >> l_names.log 2>&1
  s "팀 재지정 후 이름 판별 다시 $(cat l_orient0.log | tr '\n' ' ') $(tail -2 l_names.log | tr '\n' ' ')"; else s "$(head -1 l_orient0.log)"; fi
python3 $P/identify.py 3 > l_id.log 2>&1; python3 $P/who_filter.py >> l_id.log 2>&1; s "선수 판별"
python3 $P/apply_marker2.py > l_mk.log 2>&1; python3 $P/marker_id.py >> l_mk.log 2>&1; python3 $P/dedupe.py >> l_mk.log 2>&1; python3 $P/star_fix.py >> l_mk.log 2>&1; s "마커 $(grep -m1 marker l_mk.log)"
python3 $P/team_feat.py > l_team.log 2>&1; python3 $P/team_cls.py >> l_team.log 2>&1; python3 $P/kit_apply.py >> l_team.log 2>&1; python3 $P/lineup_check.py | tee -a l_team.log; s "팀 구분·라인업 점검"
python3 $P/marker_net.py | tee -a l_team.log; python3 $P/pos_fill.py | tee -a l_team.log; python3 $P/opp_name_fix.py | tee -a l_team.log; python3 $P/elim_fill2.py | tee -a l_team.log; python3 $P/pos_fill.py | tee -a l_team.log; python3 $P/gk_fix.py | tee -a l_team.log; s "마커 모델·위치 이어붙이기 이름 채움·골키퍼 판별"
python3 $P/ball_track.py > l_ball.log 2>&1; python3 $P/ball_events.py >> l_ball.log 2>&1; s "공 추적 $(grep 공 l_ball.log | tr "\n" " ")"
python3 $P/ball_mid2.py 0 99999999 > l_ballmid.log 2>&1 && rm -rf f10m; s "0.1초 공 추적 $(tail -1 l_ballmid.log)"   # 결과 계산에서 ball_mid.json 사용(2026-10-09 적용)
python3 $P/find_matches.py; s "완료"
