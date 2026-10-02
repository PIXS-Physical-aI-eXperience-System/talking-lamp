#!/bin/bash
# 램프 전체 상태를 한 번에 본다. Jetson 에서 실행한다.
#
# 모션 데몬은 먹통이 돼도 프로세스가 살아 있고 ROS 의 connected 는 true 로
# 남는다. 그래서 "살아 있나" 는 sent_ticks 가 늘어나는지로만 판단할 수 있다.
# 두 번 읽어 비교하는 이유다.
set +u
PI=192.168.100.2
source /opt/ros/jazzy/setup.bash 2>/dev/null
source ~/talking-lamp-integration/jetson_ws/install/setup.bash 2>/dev/null

say() { printf '\n== %s\n' "$1"; }
field() { timeout 5 ros2 topic echo "$1" --once 2>/dev/null | grep -oP "$2:\s*\K.*" | head -1; }

say "Pi 서비스"
timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 pixs@$PI \
  'systemctl is-active talking-lamp-motion.service talking-lamp-device.service | paste -sd" " -' \
  2>/dev/null || echo "  Pi 접속 실패"

say "Pi 포트"
for p in 8765 8766; do
  timeout 3 bash -c "echo > /dev/tcp/$PI/$p" 2>/dev/null \
    && echo "  $p 열림" || echo "  $p 닫힘  <-- 모션 데몬 먹통"
done

say "제어 루프 (늘어나야 정상)"
t1=$(field /lamp/motion_status sent_ticks); sleep 3; t2=$(field /lamp/motion_status sent_ticks)
if [ -z "$t1$t2" ]; then echo "  상태 토픽 없음 (젯슨 브리지 꺼짐?)"
elif [ "$t1" = "$t2" ]; then echo "  sent_ticks $t1 -> $t2   멈춤  <-- 데몬 먹통"
else echo "  sent_ticks $t1 -> $t2   정상"; fi
echo "  state=$(field /lamp/motion_status '^state') busy=$(field /lamp/motion_status busy) fault=$(field /lamp/motion_status fault)"

say "고개 방향"
echo "  yaw=$(field /lamp/orientation_status current_yaw) state=$(field /lamp/orientation_status '^state')"
echo "  (speech_id 가 계속 바뀌면 마이크 소리에 반응해 정렬 중이라는 뜻)"

say "비전 노드"
n=$(pgrep -cf '[p]ython -m lamp_vision')
[ "$n" -gt 0 ] && echo "  실행 중 ($n)" || echo "  꺼짐"

say "서보 (시리얼이 비어 있을 때만 읽힌다)"
timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=5 pixs@$PI \
  '/home/pixs/talking-lamp/lelamp_runtime/.venv/bin/python /tmp/servo_diag.py' 2>/dev/null \
  | sed 's/^/  /' || echo "  못 읽음 (데몬이 시리얼을 쓰는 중이면 정상)"

say "먹통 포착 기록"
ls -1t /tmp/wedge/*.txt 2>/dev/null | head -3 | sed 's/^/  /' || echo "  없음"
echo
