# ctranslate2 (sm_87) 빌드

PyPI 의 ctranslate2 aarch64 휠은 CUDA 없이 빌드돼 있다. Jetson 에서
`device="cuda"` 를 주면 이렇게 죽는다:

```
ValueError: This CTranslate2 package was not compiled with CUDA support
```

## 왜 필요한가

Jetson 실측 (faster-whisper small, 참조 오디오 18.8s):

| 스레드 | 추론 | RTF | CER |
|---|---|---|---|
| 2 | 37.6s | 2.00 | 0.000 |
| 4 | 22.1s | 1.17 | 0.000 |
| 6 | 20.1s | 1.07 | 0.000 |

**정확도는 완벽한데(CER 0.000) 속도가 안 된다.** 6코어를 전부 써도 RTF 1.07 이고,
실제로는 비전·움직임과 나눠 써야 하니 4코어 기준 1.17 로 봐야 한다.
모델이 아니라 실행 경로의 문제라, 모델은 그대로 두고 GPU 로 옮긴다.

base 로 내리는 선택지는 없다. 심사 문장 6개 중 4개가 틀렸고
"승원아, 3시에 회의 있어" 를 "뭐 나 센시也可以 있어" 로 냈다.

## 실행

```bash
./build.sh              # 빌드 (실패해도 다시 실행하면 이어서 간다)
./build.sh shell        # 컨테이너 안에서 직접 확인
PARALLEL=4 ./build.sh   # 병렬도 조절
```

## 확인되지 않은 것

**CTranslate2 가 CUDA 13 에서 컴파일되는지는 모른다.** onnxruntime 은 CUDA 13.2 의
CCCL 헤더 충돌로 98% 에서 죽어 13.0 으로 내려가야 했다. 여기서도 비슷한 게 나오면
Dockerfile 의 `CT2_VERSION` 을 올려서 다시 시도할 것.

CUDA 주 버전은 바꿀 수 없다. Jetson 이 13.2 이므로 12.x 로 빌드한 휠은 아예 안 돈다.
마이너 버전 차이(13.0 으로 빌드 → 13.2 에서 실행)는 onnxruntime 으로 확인됐다.
