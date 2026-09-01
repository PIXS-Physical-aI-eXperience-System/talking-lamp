#!/usr/bin/env bash
# 후보별 venv를 따로 만든다. 한 환경에 다 넣으면 의존성이 충돌한다.
#
#   ./setup.sh record   녹음 도구만 (제일 먼저)
#   ./setup.sh fw       faster-whisper (small+base 공용)
#   ./setup.sh vosk     Vosk + 한국어 모델 다운로드
#   ./setup.sh melo     MeloTTS
#   ./setup.sh all
#
# 파이썬 버전을 바꾸려면:  PY=python3.11 ./setup.sh melo
set -euo pipefail
cd "$(dirname "$0")/.."   # voice-bench/ 기준으로 venv·models 를 만든다

PY="${PY:-python3}"
mkdir -p venvs models

mkvenv() {  # $1=이름  $2...=pip 패키지
  local name="$1"; shift
  echo "── venv: $name  ($PY)"
  [ -d "venvs/$name" ] || "$PY" -m venv "venvs/$name"
  "venvs/$name/bin/pip" install --quiet --upgrade pip
  "venvs/$name/bin/pip" install "$@"
  echo "   완료: venvs/$name"
}

case "${1:-all}" in
  record)
    mkvenv record sounddevice soundfile numpy
    ;;

  fw)
    mkvenv fw faster-whisper
    echo "   ※ 모델 가중치는 첫 실행 때 자동 다운로드된다 (small ~500MB, base ~150MB)"
    ;;

  vosk)
    mkvenv vosk vosk
    if [ ! -d models/vosk-model-small-ko-0.22 ]; then
      echo "── Vosk 한국어 모델 다운로드 (82MB)"
      curl -L -o models/ko.zip \
        https://alphacephei.com/vosk/models/vosk-model-small-ko-0.22.zip
      unzip -q models/ko.zip -d models/ && rm models/ko.zip
      echo "   완료: models/vosk-model-small-ko-0.22"
    else
      echo "   이미 있음: models/vosk-model-small-ko-0.22"
    fi
    ;;

  melo)
    # MeloTTS는 의존성이 까다롭다. 3.13에서 안 붙으면 PY=python3.11 로 재시도할 것.
    mkvenv melo soundfile
    "venvs/melo/bin/pip" install git+https://github.com/myshell-ai/MeloTTS.git
    "venvs/melo/bin/python" -m unidic download || true
    echo "   ※ 첫 실행 때 한국어 모델을 받는다. 이때 잡히는 RSS가 예산 판단의 핵심 숫자다."
    ;;

  piper)
    echo "piper-plus 는 배포 형태가 포크마다 달라 자동 설치를 넣지 않았다."
    echo "  1) https://github.com/ayutaz/piper-plus 에서 설치 (MIT)"
    echo "  2) 한국어 .onnx 를 models/ 에 두기"
    echo "  3) candidates.json 의 piper-plus-ko 에 경로 채우고 enabled:true"
    mkvenv piper soundfile
    ;;

  kokoro)
    mkvenv kokoro kokoro soundfile numpy
    echo "   ※ 한국어 lang_code/voice 이름을 확인해서 candidates.json 에 채울 것"
    ;;

  melo-onnx)
    mkvenv melo-onnx onnxruntime numpy soundfile transformers g2pkk python-mecab-ko onnx
    echo "   ※ torch 는 일부러 설치하지 않는다. ONNX 파이프라인의 전제다."
    ;;

  all)
    for t in record fw vosk melo; do "$0" "$t"; done
    echo
    echo "기본 후보 준비 완료. piper/kokoro 는 수동:  ./setup.sh piper"
    ;;

  *)
    echo "사용법: ./setup.sh [record|fw|vosk|melo|piper|kokoro|all]"; exit 1
    ;;
esac
