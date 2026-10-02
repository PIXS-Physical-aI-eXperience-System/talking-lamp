#!/bin/bash
# Pi 모션 데몬이 멈추는 순간을 잡는다.
#
# 데몬은 먹통이 돼도 프로세스가 살아 있고 아무 로그도 남기지 않는다.
# systemd 의 Restart=on-failure 도 걸리지 않으니 main 이 반환조차 못 한 것이다.
# 그래서 밖에서 지켜보다 제어 루프가 멈추면 그 자리에서 상태를 떠낸다.
#
# 판정은 sent_ticks 다. 늘어나지 않으면 루프가 죽은 것이다.
set -u
set +u  # ROS setup.bash 가 미설정 변수를 참조한다
PI=192.168.100.2
OUT=${1:-/tmp/wedge}
mkdir -p "$OUT"
source /opt/ros/jazzy/setup.bash
source ~/talking-lamp-integration/jetson_ws/install/setup.bash

ticks() {
  timeout 5 ros2 topic echo /lamp/motion_status --once 2>/dev/null \
    | grep -oP 'sent_ticks:\s*\K[0-9]+'
}

prev=""
stuck=0
echo "감시 시작 $(date +%H:%M:%S) -> $OUT"
while true; do
  now=$(ticks)
  if [ -z "$now" ]; then
    echo "$(date +%H:%M:%S) 상태 못 읽음"
  elif [ "$now" = "$prev" ]; then
    stuck=$((stuck + 1))
    echo "$(date +%H:%M:%S) sent_ticks 멈춤 ($now) x$stuck"
    if [ "$stuck" -ge 3 ]; then
      ts=$(date +%H%M%S)
      echo "=== 먹통 포착, 상태 떠내는 중 -> $OUT/$ts"
      ssh -o BatchMode=yes pixs@$PI bash /tmp/wedge_capture.sh > "$OUT/$ts.txt" 2>&1
      echo "저장: $OUT/$ts.txt"
      stuck=0
    fi
  else
    stuck=0
  fi
  prev=$now
  sleep 5
done
