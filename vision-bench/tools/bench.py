"""검출기 후보 1차 측정 — 속도와 피크 메모리.

모델·프로바이더 조합마다 별도 프로세스로 실행한다. 한 프로세스에서 여러 개를
띄우면 피크 RSS가 누적되어 개별 예산 판단에 쓸 수 없기 때문이다.
입력은 고정 시드 난수다. 이 단계는 정확도가 아니라 연산량만 보므로 내용은
무관하고, 모든 조합이 같은 배열을 받는다.
"""
import sys, time, statistics, numpy as np

def peak_rss_mb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) / 1024

WARMUP, ITERS = 8, 30

def main(path, ep):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(path, opts, providers=[ep])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    rng = np.random.default_rng(1217)
    x = rng.standard_normal(shape).astype(np.float32)
    feed = {inp.name: x}

    for _ in range(WARMUP):
        sess.run(None, feed)
    lat = []
    for _ in range(ITERS):
        t = time.perf_counter()
        sess.run(None, feed)
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    print(f"{statistics.mean(lat):.1f}\t{lat[len(lat)//2]:.1f}\t{lat[int(len(lat)*0.95)]:.1f}\t{peak_rss_mb():.0f}\t{'x'.join(map(str,shape[2:]))}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
