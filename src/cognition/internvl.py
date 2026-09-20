"""Persistent one-tile NF4 backend for the voice adapter; load on Jetson worker."""
from .contract import SYSTEM_PROMPT


class InternVLCognition:
    def __init__(self, model_id="OpenGVLab/InternVL3_5-1B-HF", *,
                 revision="main", max_new_tokens=96):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.processor.image_processor.crop_to_patches = False
        self.processor.image_processor.min_patches = 1
        self.processor.image_processor.max_patches = 1
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, revision=revision, device_map="cuda", low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            ),
        ).eval()

    def __call__(self, image, question):
        import torch
        from PIL import ImageOps

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": ImageOps.exif_transpose(image).convert("RGB")},
                {"type": "text", "text": question},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        if int(inputs["pixel_values"].shape[0]) != 1:
            raise ValueError("expected exactly one image tile")
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        return self.processor.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        ).strip()
