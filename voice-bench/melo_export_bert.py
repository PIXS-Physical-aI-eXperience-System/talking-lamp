"""한국어 BERT(kykim/bert-kor-base)를 ONNX로 내보낸다.

melo는 이 모델의 hidden_states[-3] 만 쓴다 (MLM 헤드는 안 씀).
그래서 AutoModel로 로드하고 해당 층만 출력하는 래퍼를 내보낸다.
"""
import os, torch
from transformers import AutoModel, AutoTokenizer

MID = "kykim/bert-kor-base"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "melo-ko-onnx")
os.makedirs(OUT, exist_ok=True)


class BertFeature(torch.nn.Module):
    """melo가 실제로 쓰는 중간층 하나만 내보낸다."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.m(input_ids=input_ids, token_type_ids=token_type_ids,
                     attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[-3]


def main():
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModel.from_pretrained(MID).eval()
    w = BertFeature(model).eval()
    enc = tok("눈부신데 각도 십오 도만 내려줄래?", return_tensors="pt")
    args = (enc["input_ids"], enc["token_type_ids"], enc["attention_mask"])
    with torch.no_grad():
        print("torch 출력:", tuple(w(*args).shape))

    path = os.path.join(OUT, "bert-kor-base.onnx")
    torch.onnx.export(
        w, args, path,
        input_names=["input_ids", "token_type_ids", "attention_mask"],
        output_names=["hidden"],
        dynamic_axes={k: {0: "B", 1: "T"} for k in
                      ("input_ids", "token_type_ids", "attention_mask", "hidden")},
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"내보내기 성공: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    tok.save_pretrained(os.path.join(OUT, "tokenizer"))
    print("토크나이저 저장 완료")


if __name__ == "__main__":
    main()
