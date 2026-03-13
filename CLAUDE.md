# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AudioGen 是一个游戏语音合成工具，基于 **Qwen3-TTS** 本地大模型，使用 **Gradio** 提供 Web 界面。支持预设音色合成、音色设计、参考音频克隆、角色音色管理、CSV/Excel 批量处理和视频音频提取。

## 启动

```bash
pip install -r requirements.txt   # 首次安装依赖
python app.py                      # 启动，访问 http://localhost:7860
```

> MP3 输出依赖系统安装的 `ffmpeg`（需在 PATH 中）。模型权重首次运行时从 HuggingFace 自动下载（已缓存在 `~/.cache/huggingface/hub/`）。Windows 下启动时会出现 SoX 找不到的警告，可忽略，不影响功能。

## 架构

### 数据流

```
Gradio UI (app.py)
    └── TTSEngine (core/tts_engine.py)      ← 单例，延迟加载
            └── CustomVoice 模型（三种合成模式）
                  ├── generate_custom_voice   预设音色（_preset）
                  ├── generate_voice_clone    参考克隆（_clone）
                  └── voice_design            音色设计（自然语言描述）
    └── BatchProcessor (core/batch_processor.py)
            └── 调用 TTSEngine.synthesize()，逐行合成
    └── audio_utils (core/audio_utils.py)
            └── numpy array → WAV/MP3 bytes
    └── config.py                            ← HF 环境变量，最早导入
```

### 模型

只使用 **一个**模型：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`，通过 `TTSEngine._get_model()` 延迟加载。

- `config.py` 设置 `HF_ENDPOINT`（默认 `http://hf-mirror.com`）和 `HF_HUB_DISABLE_XET`，必须在其他模块之前导入。
- 模型以 `torch.bfloat16` 加载，自动检测 cuda / cpu。
- `qwen_tts.Qwen3TTSModel` 的方法返回 `(List[np.ndarray], int)`，取 `[0]` 得到单条音频。

### 合成路由（TTSEngine.synthesize）

| 条件 | 内部方法 |
|------|------|
| `ref_audio` 不为 None | `_clone` → `generate_voice_clone`（non_streaming_mode=True） |
| 否则 | `_preset` → `generate_custom_voice`（non_streaming_mode=True） |

`generate_voice_clone` 默认 `non_streaming_mode=False`，**调用时必须显式传 `True`**。

### Gradio UI（app.py）

共 6 个 Tab，`build_app()` 中依次调用：

| 函数 | Tab | 返回值 |
|------|-----|--------|
| `tab_single_synth()` | 单条合成 | `(audio_out, send_to_clone_btn)` |
| `tab_voice_design()` | 音色设计 | `(design_char_dd, design_save_btn, design_audio_out, design_send_to_clone_btn)` |
| `tab_clone()` | 参考音频克隆 | `(clone_char_dd, save_to_char_btn, ref_audio_in)` |
| `tab_character_manager()` | 角色音色管理 | `(char_table, design_table, delete_clone_btn, delete_design_btn)` |
| `tab_batch()` | 批量处理 | 无 |
| `tab_tools()` | 工具（视频音频提取） | `(tool_audio_out, tool_send_to_clone_btn)` |

各 Tab 函数返回需要跨 Tab 交互的组件，在 `build_app()` 中统一连接事件。

### 跨 Tab 交互

- **发送到参考音频克隆**：单条合成、音色设计、工具提取音频下方各有按钮，点击后将音频传到克隆 Tab 的 `ref_audio_in`
- **保存角色 → 刷新管理表格**：克隆/设计 Tab 保存后自动刷新角色管理表格
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

## 关键常量（core/tts_engine.py）

- `PRESET_VOICES`：静态预设说话人列表（`["Vivian", "Serena", "Uncle_Fu", ...]`），模型加载后尝试通过 `model.get_supported_speakers()` 动态更新。
- `CUSTOM_VOICE_MODEL_ID`：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

## 批量处理 CSV 格式

默认列名：`text`（必填）、`character`（可选）、`filename`（可选）。列名可在 UI 中自定义。`character` 列的值需与 `characters.json` 中的角色名完全匹配。输出文件名：有 `filename` 列时用其值，否则按行号 `0001.wav` 命名。

## Commit 风格

使用中文 commit message，简洁描述改动目的。
