# 会议智治与跨语言沟通平台

AI 比赛赛题 A4 参赛作品。输入会议录音或文本，输出结构化会议纪要（议题 / 结论 / 待办），并支持中英双语互译。

## 快速开始

### Windows（推荐双击启动）

1. **解压**：把 zip 解压到**英文路径、不含空格**的文件夹，例如 `D:\meeting-ai`。**不要直接在压缩包里双击**。
2. **装 Python**（只需一次）：安装 Python **3.11 ~ 3.13**（别装太新的 3.14+，部分依赖暂无安装包），安装时勾选 `Add python.exe to PATH`。
3. **双击 `start_web.bat`**：首次运行会自动创建虚拟环境并安装依赖（需联网，约几分钟）；之后每次双击即可启动。
4. 浏览器会自动打开 `http://localhost:8501`，也可手动访问。

### Mac / Linux

```bash
cd 项目目录
bash 启动.sh
```

（首次运行同样会自动建环境、装依赖。）

## 填入你自己的大模型 Key

- 打开页面后，在左侧边栏选择模型、填入你的 API Key。
- 只需填一次，程序会记住（保存在本地 `data/llm_settings.json`，不会随分享包泄露）。
- 不想填侧边栏的话，也可以复制 `.env.example` 为 `.env` 并填入，效果相同。

## 常见问题（打不开 / 闪退怎么办）

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| 双击后黑窗一闪就没了 | 在压缩包里直接双击了 | 先**完整解压**到文件夹再双击 |
| 双击 `bin\ffmpeg.exe` 刷出一堆英文后退出码 1 | 点错文件了 | 那是程序内部调用的工具，**不要双击**，请双击 `start_web.bat` |
| 提示 `Python not found` | 没装 Python / 没勾 PATH | 装 Python 3.11~3.13，勾选 `Add python.exe to PATH` |
| 依赖安装失败 | 网络不通 / Python 版本太新 | 换网络重试；或改装 Python 3.11~3.13 |
| 提示「创建虚拟环境失败」 | 路径带中文/空格 | 移到 `D:\meeting-ai` 这类纯英文路径 |
| 网页报 `ModuleNotFoundError: No module named 'xxx'` | 之前依赖装了一半（网络中断/窗口被关） | 重新双击 `start_web.bat`，启动器会自动体检并补齐；仍不行就删掉 `.venv` 文件夹再双击 |
| 想看清报错 | 黑窗被关闭 | 打开 CMD，`cd` 到项目目录，手动运行 `run.py`（或 `python run.py`） |

> 想手动看详细报错：按 `Win+R` 输入 `cmd`，然后
> ```
> cd /d D:\meeting-ai
> python run.py
> ```

## 注意事项

- **首次使用「说话人识别」**会自动下载约 400MB 模型（paraformer-zh / fsmn-vad / cam++），请耐心等待。
- 语音转写模型（whisper `small`）首次也会自动下载。
- 本项目不依赖系统 ffmpeg（自带 `bin/ffmpeg.exe`）。

## 目录说明

| 路径 | 说明 |
| --- | --- |
| `run.py` | 跨平台一键启动器（建环境、装依赖、启动） |
| `start_web.bat` / `启动.sh` | Windows / Mac·Linux 启动入口 |
| `app/web.py` | Streamlit 前端入口 |
| `app/services/` | ASR / 说话人分离 / 纠错 / 摘要 / 翻译 |
| `scripts/` | 命令行脚本（转写、流水线、打包） |
| `data/samples/` | 样例音频 |
| `data/output/` | 处理结果（自动创建） |
