# 第三方组件来源与许可声明（Third-Party Notices）

本项目为 AI 比赛教学演示用途，非商业发布。以下列出所依赖的第三方组件及其许可，
使用前请以各项目官方页面为准。

## 二进制文件

| 组件 | 用途 | 许可 | 来源 |
| --- | --- | --- | --- |
| ffmpeg | 音频预处理（转 16k 单声道） | LGPL-2.1+（由 imageio-ffmpeg 提供预编译版本） | https://ffmpeg.org / https://github.com/imageio/imageio-ffmpeg |

## 模型权重（首次运行时自动下载）

| 模型 | 用途 | 许可 | 来源 |
| --- | --- | --- | --- |
| Whisper（small 等） | 语音识别 | MIT | https://github.com/openai/whisper |
| paraformer-zh / fsmn-vad / cam++（FunASR） | 说话人分离 | 研究 / 学术用途（ModelScope 模型许可，商用需另行确认） | https://github.com/modelscope/FunASR |

## 主要 Python 依赖

| 包 | 许可 |
| --- | --- |
| faster-whisper | MIT |
| torch | BSD-3-Clause |
| streamlit | Apache-2.0 |
| fastapi / uvicorn | MIT / BSD-3-Clause |
| openai | Apache-2.0 |
| opencc-python-reimplemented | MIT |
| kaldi-native-fbank | Apache-2.0 |
| imageio-ffmpeg | BSD-2-Clause |
| pypdf / websockets | BSD-3-Clause |

## 说明

- 完整依赖及版本见 `requirements.txt`。
- 本清单为整理性声明，各组件最终许可以其官方仓库为准。
- 若项目转为商用，请逐一确认 FunASR 模型与 ffmpeg 构建的具体授权条款。
- 本项目的 UI 主题配色完全使用 Streamlit 自带主题，未内置任何第三方字体或图片素材。
