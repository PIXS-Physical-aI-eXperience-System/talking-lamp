"""ONNX 모델을 int8로 동적 양자화한다.

torch 동적 양자화와 달리 이건 onnxruntime 그래프 수준에서 이뤄지므로
Jetson(CUDA/TensorRT 백엔드)에서도 그대로 쓸 수 있다.

BERT는 MatMul 위주라 양자화가 잘 듣는다. VITS는 합성곱 기반이라
음질이 상할 수 있어서 따로 만들고 귀로 비교한다.
"""
import os
import sys

from onnxruntime.quantization import QuantType, quantize_dynamic

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "melo-ko-onnx")

TARGETS = [
    ("bert-kor-base.onnx", "bert-kor-base.int8.onnx"),
    ("melo-ko-vits.onnx", "melo-ko-vits.int8.onnx"),
]


def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"{'모델':<26}{'원본':>10}{'int8':>10}{'감소':>9}")
    print("-" * 56)
    for src, dst in TARGETS:
        if only and only not in src:
            continue
        sp, dp = os.path.join(D, src), os.path.join(D, dst)
        quantize_dynamic(sp, dp, weight_type=QuantType.QInt8)
        a, b = os.path.getsize(sp) / 1e6, os.path.getsize(dp) / 1e6
        print(f"{src:<26}{a:>8.1f}MB{b:>8.1f}MB{(1-b/a)*100:>8.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
