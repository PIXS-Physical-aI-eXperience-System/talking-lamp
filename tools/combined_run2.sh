#!/usr/bin/env bash
set -uo pipefail
# Force real overlap: start VLM, wait until it is well into its run
# (past torch/CUDA import, into generation), THEN fire STT+TTS so all
# three are genuinely active memory-wise at the same time.

VLM_VENV=/home/asdf/vlm-venv/bin/python
VOICE_VENV=/home/asdf/talking-lamp/voice-bench/venvs/melo-onnx/bin/python
RAM_BENCH=/home/asdf/ram-bench
VOICE_BENCH=/home/asdf/talking-lamp/voice-bench

cd "$RAM_BENCH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

$VLM_VENV tools/wake_latency.py --model-dir /mnt/ssd/internvl3_5-2b-nf4 \
    --image lamp_test.jpg > /tmp/combo2_vlm.log 2>&1 &
VLM_PID=$!

sleep 15

cd "$VOICE_BENCH"
$VOICE_VENV /tmp/stt_quick.py --wav ref/01.wav > /tmp/combo2_stt.log 2>&1 &
STT_PID=$!

$VOICE_VENV runners/tts_melo_onnx.py --out-dir out/tts/combo2 --label combo2 \
    --normalize --bert-int8 --warmup 1 --quiet-ort \
    --providers CUDAExecutionProvider,CPUExecutionProvider > /tmp/combo2_tts.log 2>&1 &
TTS_PID=$!

wait $VLM_PID; VLM_RC=$?
wait $STT_PID; STT_RC=$?
wait $TTS_PID; TTS_RC=$?

echo "exit codes: vlm=$VLM_RC stt=$STT_RC tts=$TTS_RC"
