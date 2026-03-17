# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AudioGen 是一个游戏语音合成工具，基于 **Qwen3-TTS** 本地大模型，使用 **Gradio** 提供 Web 界面。支持预设人声、音色设计、参考音频克隆、角色音色管理、CSV/Excel 批量处理、视频音频提取、音效合成和音乐合成。

## 启动

```bash
pip install -r requirements.txt   # 首次安装依赖
python app.py                      # 启动，访问 http://localhost:7860
```

> 项目自带 `ffmpeg/` 目录，`config.py` 启动时自动将其加入 PATH，无需额外安装。模型权重首次运行时从 HuggingFace 自动下载，缓存在项目根目录 `models/`（通过 `HF_HUB_CACHE` 环境变量设置）。Windows 下启动时会出现 SoX 找不到的警告，可忽略，不影响功能。

## 架构

### 数据流

```
Gradio UI (app.py)
    └── ui/common.py                         ← 全局模型实例/model_manager/batch_processor/task_queue 单例
    └── ui/tab_*.py                          ← 每个 Tab 一个文件
    └── TaskQueue (core/task_queue.py)       ← 单 worker 线程，保证同一时间只执行一个推理任务
    └── ModelManager (core/model_manager.py) ← 统一管理所有模型的加载/卸载生命周期
    └── PresetModel (core/preset_model.py)   ← CustomVoice 预设人声
    └── CloneModel (core/clone_model.py)     ← Base 参考音频克隆
    └── DesignModel (core/design_model.py)   ← VoiceDesign 音色设计合成
    └── SfxModel (core/sfx_model.py)         ← Stable Audio 音效合成
    └── MusicModel (core/music_model.py)     ← ACE-Step 音乐合成
    └── AsrModel (core/asr_model.py)         ← Whisper 语音识别
    └── BatchProcessor (core/batch_processor.py)
            └── 直接调用 PresetModel/CloneModel/DesignModel，逐行合成
    └── audio_utils (core/audio_utils.py)
            └── numpy array → WAV/MP3 bytes
    └── app_logger (core/app_logger.py)      ← 业务事件日志，按天轮转写入 logs/
    └── config.py                            ← ffmpeg PATH + HF 环境变量 + AUTH_USERS + 模型缓存目录，最早导入
```

### 模型管理（core/model_manager.py）

`ModelManager` 统一管理所有模型的生命周期。同一时刻只允许一个模型占用 VRAM，加载新模型前自动卸载其他模型。

所有模型在 `ui/common.py` 中注册到 ModelManager：
- `tts_preset` → PresetModel（CustomVoice）
- `tts_clone` → CloneModel（Base）
- `tts_design` → DesignModel（VoiceDesign）
- `sfx` → SfxModel（Stable Audio）
- `music` → MusicModel（ACE-Step）
- `asr` → AsrModel（Whisper）

### 模型

TTS 使用 3 个 Qwen3-TTS 模型，每个模型独立一个文件：

| 模型文件 | 模型 ID | 用途 |
|---------|---------|------|
| `core/preset_model.py` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 预设人声 |
| `core/clone_model.py` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 参考音频克隆 |
| `core/design_model.py` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 音色设计合成 |
| `core/sfx_model.py` | `stabilityai/stable-audio-open-1.0` | 音效合成 |
| `core/music_model.py` | `ACE-Step/Ace-Step1.5` | 音乐合成 |
| `core/asr_model.py` | `openai/whisper-large-v3` | 语音识别 |

- `config.py` 设置 `HF_HUB_CACHE`（指向 `models/`）、`HF_ENDPOINT`（默认 `https://hf-mirror.com`）、`HF_HUB_DISABLE_XET`、`AUTH_USERS`（Gradio 登录账号），并将项目内 `ffmpeg/` 目录加入 PATH，将 `ACE-Step-1.5/` 加入 `sys.path`。必须在其他模块之前导入。
- 模型以 `torch.bfloat16` 加载，自动检测 cuda / cpu。
- `qwen_tts.Qwen3TTSModel` 的方法返回 `(List[np.ndarray], int)`，取 `[0]` 得到单条音频。

### 语音识别（core/asr_model.py）

`AsrModel` 使用 `openai/whisper-large-v3` 模型进行语音识别，用于识别参考音频的文字内容。延迟加载，CUDA 使用 `float16`，CPU 使用 `float32`。通过 ModelManager 统一管理加载/卸载。

### 合成路由

各 Tab 直接调用对应的模型实例（在 `ui/common.py` 中创建）：

| 场景 | 调用模型 |
|------|---------|
| 预设人声 | `preset_model.synthesize(text, voice_name)` |
| 参考音频克隆 | `clone_model.synthesize(text, ref_audio, ref_text)` |
| 音色设计合成 | `design_model.synthesize(text, instruct)` |
| 音效合成 | `sfx_model.generate(prompt, ...)` |
| 音乐合成 | `music_model.generate(caption, lyrics, ...)` |
| 语音识别 | `asr_model.recognize(audio_path)` |

`synth_to_file()` 在 `ui/common.py` 中内联路由逻辑：有 `ref_audio` → CloneModel，否则 → PresetModel。

加载前各模型通过 `ModelManager.request_load()` 自动卸载其他模型释放 VRAM。

### 任务队列（core/task_queue.py）

`TaskQueue` 使用单 worker 线程，保证同一时间只有一个模型推理任务在执行。各 Tab 提交合成任务时通过 `task_queue.submit()` 入队，任务类型包括 `"preset"` / `"clone"` / `"design"` / `"sfx"` / `"music"` / `"batch"`。任务状态：`queued` → `running` → `done` / `error`。

### Gradio UI（app.py）

共 9 个 Tab，`build_app()` 中依次调用：

| 函数 | Tab | 返回值 |
|------|-----|--------|
| `tab_single_synth()` | 预设人声 | `(audio_out, send_to_clone_btn)` |
| `tab_voice_design()` | 自定义人声 | `(design_char_dd, design_save_btn, design_audio_out, design_send_to_clone_btn)` |
| `tab_clone()` | 克隆人声 | `(clone_char_dd, save_to_char_btn, ref_audio_in)` |
| `tab_character_manager()` | 角色管理 | `(char_table, design_table, delete_clone_btn, delete_design_btn)` |
| `tab_batch()` | 批量处理 | 无 |
| `tab_tools()` | 工具（视频音频提取） | `(tool_audio_out, tool_send_to_clone_btn)` |
| `tab_audio_browser()` | 音频浏览 | `(browser_audio_out, browser_send_to_clone_btn)` |
| `tab_batch_download()` | 批量下载 | 无 |
| `tab_sfx()` | 音效合成 | `audio_out` |
| `tab_music()` | 音乐合成 | 无 |

`build_app()` 顶部还包含一个**任务队列状态面板**（`gr.Timer` 每秒轮询），显示当前执行和等待中的推理任务。

各 Tab 函数返回需要跨 Tab 交互的组件，在 `build_app()` 中统一连接事件。

### 跨 Tab 交互

- **发送到克隆人声**：预设人声、自定义人声、工具提取音频、音频浏览下方各有按钮，点击后将音频传到克隆人声 Tab 的 `ref_audio_in`
- **保存角色 → 刷新管理表格**：克隆人声/自定义人声 Tab 保存后自动刷新角色管理表格
- **删除角色 → 刷新下拉**：角色管理删除后自动刷新合成 Tab 下拉列表
- 所有音频输出组件设为 `interactive=False`，仅播放不可上传

### 角色配置

**克隆角色**（`data/characters.json`）：

```json
{
  "角色名": {
    "description": "描述",
    "voice_type": "clone",
    "ref_audio_path": "data/reference_audio/xxx.wav",
    "ref_text": "参考文字"
  }
}
```

- `ref_audio_path` 使用**相对路径**（相对于项目根目录 BASE_DIR），读取时自动解析为绝对路径，兼容旧的绝对路径
- 参考音频文件存放于 `data/reference_audio/`，通过克隆 Tab 保存时自动拷贝至此目录

**设计角色**（`data/design_characters.json`）：

```json
{
  "角色名": {
    "description": "描述",
    "voice_type": "design",
    "instruct": "音色描述文本"
  }
}
```

## 关键常量（core/preset_model.py）

- `PRESET_VOICES`：静态预设说话人列表（`["Vivian", "Serena", "Uncle_Fu", ...]`），模型加载后尝试通过 `model.get_supported_speakers()` 动态更新。
- `CUSTOM_VOICE_MODEL_ID`：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

## 批量处理 CSV 格式

默认列名：`text`（必填）、`character`（可选）、`filename`（可选）。列名可在 UI 中自定义。`character` 列的值需与 `characters.json` 中的角色名完全匹配。输出文件名：有 `filename` 列时用其值，否则按行号 `0001.wav` 命名。

## 输出目录

- `output/` — 单条合成音频，按日期子目录 `output/YYYY-MM-DD/` 存放
- `output_batch/` — 批量合成输出
- `logs/` — 业务事件日志（`app.log`），按天轮转，保留 90 天

## Commit 风格

使用中文 commit message，简洁描述改动目的。
