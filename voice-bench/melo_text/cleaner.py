from . import cleaned_text_to_sequence
import copy
import importlib

# [Talking Lamp 패치] 언어 모듈을 지연 로드한다.
# 원본은 7개 언어를 한 줄로 전부 import 하는데, 일본어 모듈은 mecab-python3(모듈명 MeCab)를,
# 한국어 모듈은 python-mecab-ko(모듈명 mecab)를 요구한다. macOS는 파일시스템이 대소문자를
# 구분하지 않아 두 패키지가 같은 자리를 놓고 충돌해 공존할 수 없다.
# 실제로 쓰는 언어만 불러오면 이 문제를 피한다. 리눅스(Jetson)에서는 이 패치가 없어도 된다.
_MODULE_NAMES = {"ZH": "chinese", "JP": "japanese", "EN": "english",
                 "ZH_MIX_EN": "chinese_mix", "KR": "korean",
                 "FR": "french", "SP": "spanish", "ES": "spanish"}


class _LazyLanguageModules:
    def __init__(self):
        self._cache = {}

    def __getitem__(self, lang):
        if lang not in self._cache:
            self._cache[lang] = importlib.import_module(f".{_MODULE_NAMES[lang]}", __package__)
        return self._cache[lang]


language_module_map = _LazyLanguageModules()


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    
    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)
    
    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass