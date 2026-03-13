# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

DPAudio 是一个游戏语音合成工具，基于 **Qwen3-TTS** 本地大模型，使用 **Gradio** 提供 Web 界面。支持预设音色合成、参考音频克隆、角色音色管理和 CSV/Excel 批量处理。

## 启动

```bash
pip install -r requirements.txt   # 首次安装依赖
python app.py                      # 启动，访问 http://localhost:7860
```

> MP3 输出依赖系统安装的 `ffmpeg`（需在 PATH 中）。模型权重首次运行时从 HuggingFace 自动下载（约 3-7 GB）。

## 架构

### 数据流

```
Gradio UI (app.py)
    └── TTSEngine (core/tts_engine.py)      ← 单例，延迟加载
            ├── Base 模型（预设音色）
            └── CustomVoice 模型（参考克隆）
    └── BatchProcessor (core/batch_processor.py)
            └── 调用 TTSEngine，逐行合成
    └── audio_utils (core/audio_utils.py)
            └── numpy array → WAV/MP3 bytes
```

### 两个 Qwen3-TTS 模型

- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` — 无说话人纯文本合成（暂未使用），在 `TTSEngine._get_base_model()` 中延迟加载
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` — **同时**负责预设音色（`generate_custom_voice`）和参考音频克隆（`generate_voice_clone`），在 `TTSEngine._get_custom_model()` 中延迟加载

路由逻辑：`TTSEngine.synthesize()` 根据是否有 `ref_audio` 参数决定调用 `generate_custom_voice` 还是 `generate_voice_clone`，两者都走 CustomVoice 模型。

### 角色配置持久化

角色音色配置存储在 `data/characters.json`，格式：

```json
{
  "角色名": {
    "description": "描述",
    "voice_type": "preset" | "clone",
    "voice_name": "Ethan",          // voice_type=preset 时有效
    "ref_audio_path": "/绝对路径",   // voice_type=clone 时有效
    "ref_text": "参考文字"
  }
}
```

参考音频文件统一存放在 `data/reference_audio/`，通过 Tab4 的"保存到角色管理"按钮自动复制并写入配置。

### Gradio UI 结构（app.py）

每个 Tab 是独立函数，在 `build_app()` 内的 `gr.Blocks` 上下文中调用：

| 函数 | Tab |
|------|-----|
| `tab_single_synth()` | 单条合成 |
| `tab_character_manager()` | 角色音色管理 |
| `tab_batch()` | 批量处理 |
| `tab_clone()` | 参考音频克隆 |

Gradio 回调函数以闭包形式定义在各 Tab 函数内部，通过 `.click()` 绑定。

### 音频处理管道

合成结果统一为 `(np.ndarray float32, sample_rate: int)`，经 `normalize_audio()` 峰值归一化后：
- WAV：`soundfile.write()` 写入 PCM_16
- MP3：先写 WAV，再 `pydub.AudioSegment` 转码

Gradio Audio 组件通过 `bytes_to_gradio_audio()` 写入系统临时文件（`tempfile.NamedTemporaryFile`）来播放。

## 关键常量

- `PRESET_VOICES`（`core/tts_engine.py`）：预设说话人列表（`Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee`），通过 `model.get_supported_speakers()` 可动态获取
- `BASE_MODEL_ID` / `CUSTOM_VOICE_MODEL_ID`：HuggingFace 模型 ID
- `OUTPUT_DIR`：默认输出目录（`output/`，相对于项目根）

## 批量处理 CSV 格式

默认列名：`text`（必填）、`character`（可选）、`filename`（可选）。列名可在 UI 中自定义映射。`character` 列的值需与 `characters.json` 中的角色名完全匹配。
