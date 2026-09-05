#!/usr/bin/env bash
# 컨테이너 안에서 실제 컴파일을 수행한다.
# 빌드 디렉터리(/build)는 볼륨이므로, 실패 후 다시 실행하면 cmake 가
# 이미 만든 오브젝트를 재사용해 이어서 진행한다.
set -euo pipefail
PARALLEL="${PARALLEL:-4}"
ARCH="${CUDA_ARCH_LIST:-8.7}"

echo "== CTranslate2 빌드 시작 (${ARCH}, 병렬 ${PARALLEL}) =="

# WITH_MKL=OFF  — MKL 은 x86 전용이다. ARM 에서는 Ruy(int8)와 OpenBLAS(float)를 쓴다.
# BUILD_CLI=OFF — 파이썬에서만 쓰므로 명령줄 도구는 필요 없다.
cmake -S /src/CTranslate2 -B /build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DWITH_CUDA=ON -DWITH_CUDNN=ON \
  -DCUDA_ARCH_LIST="$ARCH" \
  -DWITH_MKL=OFF -DWITH_RUY=ON -DWITH_OPENBLAS=ON \
  -DOPENMP_RUNTIME=COMP \
  -DBUILD_CLI=OFF

cmake --build /build --parallel "$PARALLEL"
cmake --install /build

echo "== 파이썬 휠 =="
# 빌드한 libctranslate2 를 링크해야 한다. 이걸 안 잡아주면 PyPI 소스에서
# CPU 판을 다시 받아 빌드해버려 CUDA 없는 휠이 또 나온다.
export CTRANSLATE2_ROOT=/usr/local
cd /src/CTranslate2/python
pip3 wheel . --no-deps --no-build-isolation -w /out

# pip wheel 은 libctranslate2.so.4 를 담지 않는다. 링크만 하고 파일은 두고 가서
# Jetson 에서 ImportError: libctranslate2.so.4: cannot open shared object file
# 로 죽는다. PyPI 휠은 auditwheel 로 이걸 안에 넣어 배포한다.
#
# CUDA 라이브러리는 제외한다. Jetson 에 이미 있고, 담으면 휠이 수백 MB 가 된다.
echo "== 공유 라이브러리 포장 (auditwheel) =="
RAW=$(ls /out/ctranslate2-*.whl | head -1)
if LD_LIBRARY_PATH=/usr/local/lib auditwheel repair "$RAW" \
     --plat linux_aarch64 -w /out/repaired \
     --exclude 'libcu*' --exclude 'libnv*' --exclude 'libcudnn*'; then
  mv "$RAW" "/tmp/$(basename "$RAW").raw"
  mv /out/repaired/*.whl /out/
  rmdir /out/repaired 2>/dev/null || true
  echo "  포장 완료"
else
  echo "  ! auditwheel 실패 — 공유 라이브러리를 직접 넣는다"
  cp -a /usr/local/lib/libctranslate2.so* /out/
  echo "  dist/libctranslate2.so* 를 Jetson 의 venv 안 ctranslate2/ 로 함께 복사할 것"
fi

echo "== 결과 =="
ls -lh /out/
echo
echo "== 휠에 담긴 공유 라이브러리 =="
python3 -c "
import glob, zipfile
w = sorted(glob.glob('/out/ctranslate2-*.whl'))[-1]
names = [n for n in zipfile.ZipFile(w).namelist() if n.endswith('.so') or '.so.' in n]
print('\n'.join('  ' + n for n in names) or '  (없음 — 이러면 Jetson 에서 또 죽는다)')
"
