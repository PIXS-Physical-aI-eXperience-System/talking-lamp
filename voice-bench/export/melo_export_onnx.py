"""MeloTTS 한국어 VITS를 ONNX로 내보낸다.

목적: torch 런타임을 제거해 피크 메모리를 낮추는 것.
오늘 실측상 melo 사용량 약 2 GB 중 모델 가중치는 652 MB뿐이고,
나머지는 torch 런타임과 순간 할당이다. onnxruntime으로 바꾸면 그 몫이 사라진다.

BERT는 별도 모델이므로 이 스크립트는 VITS 부분만 다룬다.
"""
import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from melo.api import TTS
from melo import utils

OUT = os.path.join(HERE, "models", "melo-ko-onnx")
os.makedirs(OUT, exist_ok=True)


class VitsWrapper(torch.nn.Module):
    """infer()의 키워드 인자를 고정해 ONNX가 추적할 수 있는 forward로 만든다."""

    def __init__(self, model, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, length_scale=1.0):
        super().__init__()
        self.model = model
        self.sdp_ratio, self.noise_scale = sdp_ratio, noise_scale
        self.noise_scale_w, self.length_scale = noise_scale_w, length_scale

    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert):
        return self.model.infer(
            x, x_lengths, sid, tone, language, bert, ja_bert,
            sdp_ratio=self.sdp_ratio, noise_scale=self.noise_scale,
            noise_scale_w=self.noise_scale_w, length_scale=self.length_scale)[0]


def main():
    tts = TTS(language="KR", device="cpu")
    net = tts.model.eval()
    spk = tts.hps.data.spk2id["KR"]

    # 실제 문장으로 예시 입력을 만든다 (모양·dtype을 정확히 맞추기 위해)
    bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
        "왼쪽 좀 더 밝게 비춰줘.", "KR", tts.hps, "cpu", tts.symbol_to_id)
    args = (phones.unsqueeze(0), torch.LongTensor([phones.size(0)]),
            torch.LongTensor([spk]), tones.unsqueeze(0), lang_ids.unsqueeze(0),
            bert.unsqueeze(0), ja_bert.unsqueeze(0))
    print("입력 모양:", [tuple(a.shape) for a in args])

    wrapper = VitsWrapper(net).eval()
    with torch.no_grad():
        ref = wrapper(*args)
    print("torch 출력:", tuple(ref.shape))

    path = os.path.join(OUT, "melo-ko-vits.onnx")
    torch.onnx.export(
        wrapper, args, path,
        input_names=["phones", "phone_lengths", "sid", "tones", "lang_ids", "bert", "ja_bert"],
        output_names=["audio"],
        dynamic_axes={"phones": {1: "T"}, "tones": {1: "T"}, "lang_ids": {1: "T"},
                      "bert": {2: "T"}, "ja_bert": {2: "T"}, "audio": {2: "N"}},
        opset_version=17, do_constant_folding=True,
        # torch 2.x의 기본 dynamo 익스포터는 VITS의 스플라인 분기
        # (transforms.rational_quadratic_spline 안의 데이터 의존 if)에서 막힌다.
        # 레거시 TorchScript 경로는 그 분기를 추적 시점 값으로 굳혀서 통과한다.
        dynamo=False,
    )
    print(f"내보내기 성공: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
