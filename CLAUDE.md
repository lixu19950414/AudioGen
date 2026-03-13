# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DPAudio 是一个游戏语音合成工具，基于 **Qwen3-TTS** 本地大模型，使用 **Gradio** 提供 Web 界面。支持预设音色合成、音色设计、参考音频克隆、角色音色管理和 CSV/Excel 批量处理。

## 启动

```bash
pip install -r requirements.txt   # 首次安装依赖
python app.py                      # 启动，访问 http://localhost:7860
```

> MP3 输出依赖系统安装的 `ffmpeg`（需在 PATH 中）。模型权重首次运行时从 HuggingFace 自动下载（已缓存在 `~/.cache/huggingface/hub/`）。

## 架构

### 数据流

```
Gradio UI (app.py)
    └── TTSEngine (core/tts_engine.py)      ← 单例，延迟加载
            └── CustomVoice 模型（三种合成模式）
                  ├── generate_custom_voice   预设音色
                  ├── generate_voice_clone    参考克隆
                  └── generate_voice_design   音色设计
    └── BatchProcessor (core/batch_processor.py)
            └── 调用 TTSEngine.synthesize()，逐行合成
    └── audio_utils (core/audio_utils.py)
            └── numpy array → WAV/MP3 bytes
```

### 模型

只使用 **一个**模型：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`，通过 `TTSEngine._get_model()` 延迟加载。

`qwen_tts.Qwen3TTSModel` 的三个主要方法都返回 `(List[np.ndarray], int)`，取 `[0]` 得到单条音频。

### 合成路由（TTSEngine.synthesize）

| 条件 | 方法 |
|------|------|
| `ref_audio` 不为 None | `generate_voice_clone`（non_streaming_mode=True） |
| 否则 | `generate_custom_voice`（non_streaming_mode=True） |

音色设计通过单独的 `TTSEngine.synthesize_design(text, instruct)` 调用。

### Gradio UI（app.py）

| 函数 | Tab |
|------|-----|
| `tab_single_synth()` | 单条合成（预设音色 + 角色库） |
| `tab_voice_design()` | 音色设计（自然语言描述） |
| `tab_clone()` | 参考音频克隆 |
| `tab_character_manager()` | 角色音色管理 |
| `tab_batch()` | 批量处理（CSV/Excel） |

### 角色配置（data/characters.json）

```json
{
  "角色名": {
    "description": "描述",
    "voice_type": "preset" | "clone",
    "voice_name": "Vivian",        // preset 时有效
    "ref_audio_path": "/绝对路径", // clone 时有效
    "ref_text": "参考文字"
  }
}
```

## 关键常量（core/tts_engine.py）

- `PRESET_VOICES`：静态预设说话人列表（模型加载后会尝试动态更新）
- `CUSTOM_VOICE_MODEL_ID`：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

## 批量处理 CSV 格式

默认列名：`text`（必填）、`character`（可选）、`filename`（可选）。列名可在 UI 中自定义。`character` 列的值需与 `characters.json` 中的角色名完全匹配。
