"""支持的翻译语种(与中文互译)。

新增 / 删减语种只改这一个文件:侧边栏「双语翻译」、结果页「目标语言」、
实时字幕「边听边译」都从这里取选项,不用改界面代码。

每个语种四个字段:
    label   中文名,界面显示用(如「日文」)
    prompt  写进提示词的名称(如「日文」;简体 / 繁体要区分清楚)
    baidu   百度翻译 API 的 to 参数 —— 与 ISO 代码并不一致(日文是 jp、韩文是 kor)
    iso     文件名后缀,沿用 ISO 639-1,便于导出后一眼看出语种

顺序即下拉框顺序:先中英(主场景),再其它常见语种。
"""

from __future__ import annotations

LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"label": "英文", "prompt": "英文", "baidu": "en", "iso": "en"},
    "zh": {"label": "中文", "prompt": "简体中文", "baidu": "zh", "iso": "zh"},
    "zh-TW": {"label": "繁體中文", "prompt": "繁体中文", "baidu": "cht", "iso": "zh-TW"},
    "ja": {"label": "日文", "prompt": "日文", "baidu": "jp", "iso": "ja"},
    "ko": {"label": "韩文", "prompt": "韩文", "baidu": "kor", "iso": "ko"},
    "fr": {"label": "法文", "prompt": "法文", "baidu": "fra", "iso": "fr"},
    "de": {"label": "德文", "prompt": "德文", "baidu": "de", "iso": "de"},
    "es": {"label": "西班牙文", "prompt": "西班牙文", "baidu": "spa", "iso": "es"},
    "pt": {"label": "葡萄牙文", "prompt": "葡萄牙文", "baidu": "pt", "iso": "pt"},
    "it": {"label": "意大利文", "prompt": "意大利文", "baidu": "it", "iso": "it"},
    "ru": {"label": "俄文", "prompt": "俄文", "baidu": "ru", "iso": "ru"},
    "ar": {"label": "阿拉伯文", "prompt": "阿拉伯文", "baidu": "ara", "iso": "ar"},
}

# 中文名 → 代码;历史记录里 translated_lang 存的是「英文」「中文」这类名字,
# 读旧记录时靠它反查代码(见 code_of)。
_LABEL_TO_CODE = {v["label"]: k for k, v in LANGUAGES.items()}

DEFAULT = "en"


def labels() -> list[str]:
    """所有目标语种的中文名,顺序与 LANGUAGES 一致(下拉框选项)。"""
    return [v["label"] for v in LANGUAGES.values()]


def code_of(value: str | None) -> str | None:
    """接受语言代码或中文名,返回代码;不认识则返回 None。

    例:code_of("ja") → "ja",code_of("日文") → "ja"(兼容旧历史记录)。
    """
    if not value:
        return None
    if value in LANGUAGES:
        return value
    return _LABEL_TO_CODE.get(value)


def label_of(value: str | None) -> str:
    """返回中文名;无法识别时原样返回(旧记录里已经是中文名,直接透传)。"""
    code = code_of(value)
    return LANGUAGES[code]["label"] if code else (value or "")


def prompt_name(target: str | None) -> str:
    """写进 LLM 提示词的语言名(如「日文」「简体中文」)。"""
    code = code_of(target) or DEFAULT
    return LANGUAGES[code]["prompt"]


def baidu_code(target: str | None) -> str:
    """百度翻译 API 的 to 参数。"""
    code = code_of(target) or DEFAULT
    return LANGUAGES[code]["baidu"]


def iso_of(value: str | None) -> str:
    """文件名后缀用的 ISO 代码。"""
    code = code_of(value) or DEFAULT
    return LANGUAGES[code]["iso"]


def markdown_lang(value: str | None) -> str:
    """Markdown 标题语言:目前只做了中 / 英两套模板(zh-TW 沿用中文标题)。
    其它语种正文照常是目标语言,标题退回英文(国际通用的折中方案)。
    """
    code = code_of(value) or DEFAULT
    if code.startswith("zh"):
        return "zh"
    return "en"
