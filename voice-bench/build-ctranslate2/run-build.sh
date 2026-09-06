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
echo "== 공유 라이브러리 포장 =="
RAW=$(ls /out/ctranslate2-*.whl | head -1)
# --plat 에 linux_aarch64 는 못 준다. auditwheel 은 manylinux_* 또는 auto 만 받는다.
if LD_LIBRARY_PATH=/usr/local/lib auditwheel repair "$RAW" \
     --plat auto -w /out/repaired \
     --exclude 'libcu*' --exclude 'libnv*' --exclude 'libcudnn*'; then
  rm -f "$RAW"
  mv /out/repaired/*.whl /out/
  rmdir /out/repaired 2>/dev/null || true
  echo "  auditwheel 포장 완료"
else
  # auditwheel 이 정책을 못 고르는 경우가 있어서 직접 넣는다.
  # .so 를 패키지 안에 복사하는 것만으로는 부족하다. _ext.so 는 SONAME 으로
  # 찾는데 패키지 디렉터리는 로더의 검색 경로가 아니다. RPATH 를 $ORIGIN 으로
  # 바꿔줘야 자기 옆에 있는 라이브러리를 본다.
  echo "  ! auditwheel 실패 — RPATH 를 직접 손봐서 포장한다"
  rm -rf /tmp/whl && mkdir -p /tmp/whl && cd /tmp/whl
  python3 -m zipfile -e "$RAW" .
  cp -L /usr/local/lib/libctranslate2.so.4 ctranslate2/
  patchelf --set-rpath '$ORIGIN' ctranslate2/_ext*.so
  python3 - "$RAW" <<'REPACK'
import os, sys, zipfile
out = sys.argv[1]
os.remove(out)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk("."):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, "."))
print("  다시 압축:", out)
REPACK
  cd /src/CTranslate2/python
fi
echo "== 결과 =="
ls -lh /out/
echo
echo "== 휠 검증 =="
# 여기서 막지 않으면 반쪽짜리 휠이 Jetson 까지 가서 ImportError 로 죽는다.
# 이미 두 번 그랬다. 실패는 옮기기 전에 드러나야 한다.
python3 - <<'VERIFY'
import glob, sys, zipfile
w = sorted(glob.glob("/out/ctranslate2-*.whl"))[-1]
names = [n for n in zipfile.ZipFile(w).namelist() if n.endswith(".so") or ".so." in n]
print("\n".join("  " + n for n in names) or "  (없음)")
# auditwheel 은 담으면서 이름에 해시를 붙인다:
#   libctranslate2.so.4.6.0 -> ctranslate2.libs/libctranslate2-38d35ec6.so.4.6.0
# 그래서 "libctranslate2.so" 를 찾으면 멀쩡한 휠도 실패로 잡힌다.
if not any("libctranslate2" in n for n in names):
    print("\n실패: 휠 안에 libctranslate2.so 가 없다. 이대로는 Jetson 에서 못 쓴다.")
    sys.exit(1)
print("\n  확인: libctranslate2.so 가 휠 안에 있다")
VERIFY
