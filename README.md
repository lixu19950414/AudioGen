# AudioGen - 游戏语音合成工具

基于 **Qwen3-TTS** 本地大模型的游戏语音合成工具，使用 Gradio 提供 Web 界面。

## 功能

- **预设人声** — 内置多种说话人，开箱即用
- **音色设计** — 通过文字描述生成自定义音色
- **参考音频克隆** — 上传参考音频克隆说话人音色
- **角色管理** — 保存、管理已创建的角色音色
- **批量处理** — 支持 CSV / Excel 批量合成
- **视频音频提取** — 从视频中提取音频
- **音频浏览** — 浏览和管理已生成的音频
- **音效合成** — 基于 Stable Audio 生成音效
- **音乐合成** — 基于 ACE-Step 生成音乐

## 环境要求

- Python 3.10+
- CUDA 兼容 GPU（推荐，CPU 也可运行）
- Windows / Linux

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 ACE-Step 音乐合成（可选，需要音乐合成功能时安装）
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync
cd ..

# 启动
python app.py
```

启动后访问 http://localhost:7860

> 项目自带 `ffmpeg/` 目录，无需额外安装。模型权重首次运行时自动从 HuggingFace 下载，缓存在 `models/` 目录。

## 使用的模型

| 模型 | 用途 |
|------|------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 预设人声 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 参考音频克隆 |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 音色设计 |
| `stabilityai/stable-audio-open-1.0` | 音效合成 |
| `ACE-Step/Ace-Step1.5` | 音乐合成 |
| `openai/whisper-large-v3` | 语音识别 |

同一时刻仅加载一个模型，切换时自动卸载释放显存。

## 项目结构

```
├── app.py                  # Gradio 主入口
├── config.py               # 全局配置（环境变量、路径）
├── core/                   # 核心逻辑
│   ├── model_manager.py    # 模型生命周期管理
│   ├── preset_model.py     # 预设人声
│   ├── clone_model.py      # 克隆人声
│   ├── design_model.py     # 音色设计
│   ├── sfx_model.py        # 音效合成
│   ├── music_model.py      # 音乐合成
│   ├── asr_model.py        # 语音识别
│   ├── task_queue.py       # 任务队列
│   ├── batch_processor.py  # 批量处理
│   └── audio_utils.py      # 音频工具
├── ui/                     # Gradio 界面
│   ├── common.py           # 全局实例与工具函数
│   └── tab_*.py            # 各 Tab 页面
├── data/                   # 角色配置与参考音频
├── output/                 # 合成输出（按日期）
└── models/                 # 模型缓存（自动下载）
```

## 批量处理

支持 CSV / Excel 文件，列说明：

| 列名 | 必填 | 说明 |
|------|------|------|
| `text` | 是 | 待合成文本 |
| `character` | 否 | 角色名（需与已保存角色匹配） |
| `filename` | 否 | 输出文件名（缺省按行号命名） |

列名可在 UI 中自定义。

## 许可证

仅供学习研究使用，请遵守各模型的许可协议。
