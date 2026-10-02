#!/bin/bash
# Pi 에서 실행. 모션 데몬이 멈춘 순간의 상태를 떠낸다. wedge_watch.sh 가 부른다.
pid=$(pgrep -f motion.middleware_server | head -1)
echo "pid ${pid:-없음}  $(date '+%F %T')"
if [ -z "$pid" ]; then
  echo '프로세스 없음. 스스로 종료했다면 systemd 가 재시작 중이다'
  journalctl -u talking-lamp-motion.service --since '-2 min' --no-pager 2>/dev/null | tail -20
  exit 0
fi
# 열린 파일(fuser, /proc/PID/fd)과 커널 대기 지점(wchan)은 여기서 보이지 않는다.
# 데몬이 Group=talking-lamp 로 떠서 같은 pixs 라도 ptrace 권한 검사에 걸린다.
# 그래서 "시리얼을 놓았다" 같은 판단을 거기서 내리면 안 된다(정상일 때도 비어 있다).
# 믿을 수 있는 내부 정보는 데몬이 스스로 찍는 아래 파이썬 스택뿐이다.
echo "--- 스레드 수 $(ls /proc/$pid/task 2>/dev/null | wc -l)"
echo '--- 리스닝'
ss -ltn | grep 8765 || echo '8765 리스닝 없음'
echo '--- 파이썬 스택 (SIGUSR1)'
kill -USR1 "$pid" 2>/dev/null && sleep 1
journalctl -u talking-lamp-motion.service --since '-15 sec' --no-pager 2>/dev/null \
  | sed -E 's/^.*python\[[0-9]+\]: //' | grep -vE '^-- '
