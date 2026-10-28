# CLAUDE.md

本文件为 Claude Code 在此仓库中工作时提供指导。

## 项目概述

AI 比赛赛题 **A4「会议智治与跨语言沟通平台」** 参赛作品。

输入一段会议录音 / 会议文本,输出结构化会议纪要(议题、结论、待办),并支持中英双语互译与跨语言问答。

- 赛题难度:低~中
- 团队规模:5 人
- 核心能力:ASR + TTS + LLM + 翻译 API
- 交付要求:可运行的 Demo + 3-5 段样例输出 + 技术文档 + 答辩材料
- 产品边界:以「录音/文本文件」为主链路,实时字幕为加分项

## 技术栈(已定方案)

| 层 | 选型 | 备选 |
| --- | --- | --- |
| 语言 | Python 3.11 | — |
| 后端 | FastAPI + uvicorn | Flask |
| 前端 | Streamlit(app/web.py,已跑通) | React(如需更复杂交互) |
| ASR | faster-whisper(`small`,CPU,已跑通) | — |
| 说话人分离 | FunASR cam++(集成式:fsmn-vad + cam++,已跑通) | pyannote |
| LLM | DeepSeek(deepseek-chat,已跑通纠错+摘要) | 通义千问 |
| 翻译 | LLM 直译 + 自定义术语表 | DeepL / 百度翻译 API |
| 音频处理 | ffmpeg(统一转 16k mono wav) | — |
| TTS(加分) | edge-tts / 阿里云语音合成 | — |
| 存储 | SQLite + 本地文件 | — |

## 常用命令(随项目建立逐步补齐)

```bash
# 环境
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt

# 生成测试音频(Windows TTS,离线)
.venv/Scripts/python.exe scripts/make_sample.py

# 转写音频(输出 data/output/<名>.json / .txt)
.venv/Scripts/python.exe scripts/transcribe.py data/samples/sample_zh.wav --language zh

# 端到端:音频→转写→纠错→摘要(输出 data/output/<名>.summary.json)
.venv/Scripts/python.exe scripts/pipeline.py data/samples/测试1.m4a --glossary "DeepSeek" --language zh

# 前端(Streamlit,双击 start_web.bat 或)
.venv/Scripts/python.exe -m streamlit run app/web.py

# 后端(实时转写 WebSocket 中继,讯飞 rtasr;start_web.bat / run.py 会自动拉起,一般无需手动跑)
uvicorn app.main:app --port 8000

# 测试
pytest
```

## 目标目录结构

```
app/
  main.py            # FastAPI 入口
  api/               # 路由:上传、任务状态、结果导出
  services/
    audio.py         # ffmpeg 预处理、VAD 切分
    asr.py           # 转写 + 说话人分离
    llm.py           # 摘要 / 结构化 / 翻译(统一封装)
    export.py        # Markdown / JSON / SRT 导出
  prompts/           # 提示词模板
  schemas/           # Pydantic 模型(纪要 JSON Schema)
web/                 # React 前端
data/                # 样例音频、输出结果
tests/               # 单元测试 + 评测脚本
```

## 约定与规范

- ASR 输出契约(全队共用,勿改字段):`app/services/asr.py` 中 `Transcript.to_dict()`,即 `{model, language, language_probability, duration, elapsed_seconds, text, segments:[{start, end, speaker, text}]}`。
- 所有外部模型调用统一封装在 `app/services/` 下,供应商可替换。
- LLM 输出必须是受 Schema 约束的 JSON,禁止用正则解析自由文本。
- 长音频按 VAD 分块,采用 map-reduce 两段式摘要,避免超长上下文。
- API Key 一律放 `.env`,不提交仓库;`.env.example` 提供模板。
- 代码注释、提交信息、文档使用中文。
- 会议内容涉及隐私,架构上保留「全本地部署」选项作为卖点。

## 注意事项

- 比赛截止:自 2026-10-26 起仅剩约 1 周,以交付可演示 MVP 为唯一目标
- 需准备 3-5 段中英会议音频(含多人对话)作为固定评测集
- 验收指标:CER、纪要人工评分(准确/完整/可用)、中英翻译一致性、1 小时会议处理耗时
- 说话人分离(funasr 集成式)依赖两个非 PyPI 常规项:项目自带的 `bin/ffmpeg.exe`(由 `imageio-ffmpeg` 提供,已由 `app/config.py` 前置到 PATH)和 `kaldi-native-fbank`(fbank 后端)。首次跑分离会自动从 ModelScope 下载 paraformer-zh / fsmn-vad / cam++ 模型(约 400MB)。
- 「实时字幕」页:在「⚙️ 设置」页「🔊 实时转写 (讯飞)」填你自己的讯飞 AppID/接口密钥走云端实时转写(速度接近讯飞听见),记住到 `data/xfyun_settings.json`(打包时会剔除);留空则回退浏览器 Web Speech API。讯飞 key 只经浏览器传给中继后端(`app/main.py`,由 `run.py` 自动拉起,默认 8000 端口),服务端不内置 key,便于分发后各用各的 key。
- LLM 同样只认「⚙️ 设置」页「🔑 大模型」里用户自己填的 key(记住到 `data/llm_settings.json`,打包时剔除);留空则回退本机 Ollama 免费模型(`OLLAMA_BASE_URL`/`OLLAMA_MODEL`,默认 `qwen2.5:3b`,需已装并启动 Ollama),不回退 `.env`。`.env` 里的 `LLM_*` 仅用于命令行脚本(`scripts/pipeline.py` 等)开发者本地跑,网页端不读。
