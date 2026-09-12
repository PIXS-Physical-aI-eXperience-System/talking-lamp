#!/usr/bin/env bash
set -euo pipefail

# Jetson dialogue memory harness. Each heavyweight stage must exit before the
# next starts so unified CPU/GPU RAM is returned at state transitions.
# Paths can be overridden without editing this file.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VLM_PY=${VLM_PY:-/home/asdf/vlm-venv/bin/python}
VOICE_PY=${VOICE_PY:-/home/asdf/talking-lamp/voice-bench/venvs/melo-onnx/bin/python}
VLM_MODEL_DIR=${VLM_MODEL_DIR:-/mnt/ssd/internvl3_5-1b-nf4}
VLM_IMAGE=${VLM_IMAGE:-/home/asdf/ram-bench/lamp_test.jpg}
STT_SCRIPT=${STT_SCRIPT:-$script_dir/stt_quick.py}
STT_WAV=${STT_WAV:-/home/asdf/talking-lamp/voice-bench/ref/01.wav}
TTS_SCRIPT=${TTS_SCRIPT:-/home/asdf/talking-lamp/voice-bench/runners/tts_melo_onnx.py}
TTS_OUT=${TTS_OUT:-/tmp/talking-lamp-tts}
VLM_RESULT_DIR=${VLM_RESULT_DIR:-/tmp/talking-lamp-vlm-results}
MT_MODEL_DIR=${MT_MODEL_DIR:-}
SCENE_RENDERER=${SCENE_RENDERER:-1}
VOICE_ROOT=${VOICE_ROOT:-/home/asdf/talking-lamp/voice-bench}
if [[ "$SCENE_RENDERER" == 1 ]]; then
    default_prompt='List only the two most important visible objects in simple English, maximum 8 words.'
    default_tokens=16
else
    default_prompt='Describe this image in one short English sentence.'
    default_tokens=32
fi
PROMPT=${PROMPT:-$default_prompt}
VLM_MAX_NEW_TOKENS=${VLM_MAX_NEW_TOKENS:-$default_tokens}
ATTEMPT=${RAM_HARNESS_ATTEMPT:-1}

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
mkdir -p -- "$VLM_RESULT_DIR"

"$VOICE_PY" "$STT_SCRIPT" --wav "$STT_WAV"
"$VLM_PY" "$script_dir/wake_latency.py" \
    --model-dir "$VLM_MODEL_DIR" --image "$VLM_IMAGE" \
    --max-new-tokens "$VLM_MAX_NEW_TOKENS" --prompt "$PROMPT" \
    --out "$VLM_RESULT_DIR/attempt-$ATTEMPT.json"
if [[ "$SCENE_RENDERER" == 1 ]]; then
    "$VLM_PY" "$script_dir/render_scene_ko.py" \
        --input "$VLM_RESULT_DIR/attempt-$ATTEMPT.json" \
        --out "$VLM_RESULT_DIR/attempt-$ATTEMPT-ko.json"
    "$VOICE_PY" "$script_dir/tts_single.py" \
        --voice-root "$VOICE_ROOT" \
        --input "$VLM_RESULT_DIR/attempt-$ATTEMPT-ko.json" \
        --wav "$TTS_OUT/attempt-$ATTEMPT.wav" \
        --out "$TTS_OUT/attempt-$ATTEMPT.json"
elif [[ -n "$MT_MODEL_DIR" ]]; then
    "$VLM_PY" "$script_dir/translate_vlm_result.py" \
        --model-dir "$MT_MODEL_DIR" \
        --input "$VLM_RESULT_DIR/attempt-$ATTEMPT.json" \
        --out "$VLM_RESULT_DIR/attempt-$ATTEMPT-ko.json"
    "$VOICE_PY" "$script_dir/tts_single.py" \
        --voice-root "$VOICE_ROOT" \
        --input "$VLM_RESULT_DIR/attempt-$ATTEMPT-ko.json" \
        --wav "$TTS_OUT/attempt-$ATTEMPT.wav" \
        --out "$TTS_OUT/attempt-$ATTEMPT.json"
else
    "$VOICE_PY" "$TTS_SCRIPT" --out-dir "$TTS_OUT" --label sequential \
        --normalize --bert-int8 --warmup 1 --quiet-ort \
        --providers CUDAExecutionProvider,CPUExecutionProvider
fi
