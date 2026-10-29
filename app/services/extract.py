"""从上传文件里抽取纯文本(供术语导入用)。

只做文本抽取、不做语义理解,尽量不引入重量级依赖:
- .pptx / .docx 用标准库 zipfile + ElementTree 解析 OOXML
  (两者本质都是 zip 包里的 XML,免装 python-pptx / python-docx)
- .txt / .md / .csv 直接解码(UTF-8 优先,兜底 GB18030,兼容 Windows 记事本)
- .pdf 需要可选依赖 pypdf;未安装时给出明确提示而不是静默失败
"""

from __future__ import annotations

import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

# OOXML 的命名空间前缀
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# 前端 file_uploader 的 type 参数与之一致
SUPPORTED_EXTS = (".pptx", ".docx", ".pdf", ".txt", ".md", ".csv")


def _slide_index(name: str) -> int:
    """ppt/slides/slide12.xml → 12,用于按页码排序(避免 slide10 排在 slide2 前)。"""
    tail = Path(name).stem[5:]  # 去掉 "slide" 前缀
    return int(tail) if tail.isdigit() else 0


def _pptx_text(data: bytes) -> str:
    """抽取 pptx 每页幻灯片里的文本(含标题、正文、表格单元格)。"""
    chunks: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [
            n
            for n in zf.namelist()
            if n.startswith("ppt/slides/slide") and n.endswith(".xml")
        ]
        for name in sorted(names, key=_slide_index):
            root = ET.fromstring(zf.read(name))
            # <a:t> 是 DrawingML 里的文本节点,覆盖标题/正文/表格
            texts = [t.text for t in root.iter(_A + "t") if t.text]
            if texts:
                chunks.append("\n".join(texts))
    return "\n\n".join(chunks)


def _docx_text(data: bytes) -> str:
    """抽取 docx 正文段落文本。"""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    paras: list[str] = []
    for p in root.iter(_W + "p"):
        txt = "".join(t.text or "" for t in p.iter(_W + "t"))
        if txt.strip():
            paras.append(txt)
    return "\n".join(paras)


def _pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # 可选依赖:按需提示,不写进 requirements 强制安装
        raise RuntimeError("读取 PDF 需要 pypdf,请先执行:pip install pypdf") from exc
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _plain_text(data: bytes) -> str:
    """纯文本解码:UTF-8(BOM)优先,其次 GB18030(Windows 中文默认编码)。"""
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text(filename: str, data: bytes) -> str:
    """按扩展名抽取文本;返回已 strip 的正文。读不到内容时返回空串。"""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pptx":
        text = _pptx_text(data)
    elif suffix == ".docx":
        text = _docx_text(data)
    elif suffix == ".pdf":
        text = _pdf_text(data)
    elif suffix in (".txt", ".md", ".csv"):
        text = _plain_text(data)
    elif suffix == ".ppt":
        raise RuntimeError("旧版 .ppt 无法直接解析,请在 PowerPoint 里另存为 .pptx 后再导入")
    elif suffix == ".doc":
        raise RuntimeError("旧版 .doc 无法直接解析,请另存为 .docx 后再导入")
    else:
        raise RuntimeError(f"不支持的文件类型:{suffix or '(无扩展名)'}")
    return text.strip()
