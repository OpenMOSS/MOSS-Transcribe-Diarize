# MOSS-Transcribe-Diarize 0.9B

<br>

<p align="center">
  <img src="./assets/OpenMOSS_Logo.svg" height="70" align="middle" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./assets/mosi-logo.png" height="50" align="middle" />
</p>

<div align="center">
<a href="https://trendshift.io/repositories/78061" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Trendshift-Peak%20%2311%20Python%20Repository%20of%20the%20Day-6670CC?style=for-the-badge" alt="Peak #11 Python Repository of the Day on Trendshift"/></a>
</div>
<div align="center">
  <a href="https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize"><img src="https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface"></a>
  <a href="https://arxiv.org/abs/2601.01554"><img src="https://img.shields.io/badge/arXiv-2601.01554-b31b1b?logo=arxiv"></a>
  <a href="https://x.com/MosiAI_Official"><img src="https://img.shields.io/badge/Twitter-Follow-black?logo=x&amp"></a>
</div>

<p align="center">
  <b>中文</b> | <a href="./README.md">English</a>
</p>

MOSS-Transcribe-Diarize 0.9B 是一个开源的 SOTA 端到端音频理解模型，面向长时、多说话人场景，支持语音转写、说话人分离（diarization）、时间戳标注以及声学事件感知。[MOSS-Transcribe-Diarize Pro](https://platform.mosi.cn/app/playground) 是性能更强的版本，整体表现更优，现已可通过在线 Playground 使用。

MOSS-Transcribe-Diarize 0.9B 支持 50+ 种语言。

## 最新动态

* 2026-07-22：字幕 Web 界面现已支持简体中文和英文。
* 2026-07-14：🏆 MOSS-Transcribe-Diarize 在 INTERSPEECH 2026 [第二届 MLC-SLM Challenge](https://www.nexdata.ai/competition/mlc-slm) 中夺得第一名，覆盖 14 种语言。
* 2026-07-09：开源 MOSS-Transcribe-Diarize 0.9B。

## 目录

- [简介](#简介)
- [模型架构](#模型架构)
- [评测](#评测)
  - [客观评测](#客观评测)
- [快速开始](#快速开始)
  - [环境准备](#环境准备)
  - [Python 用法](#python-用法)
  - [使用 SGLang Omni 部署](#使用-sglang-omni-部署)
  - [使用 vLLM 部署](#使用-vllm-部署)
  - [自定义 Prompt 与热词](#自定义-prompt-与热词)
  - [字幕 Web 应用](#字幕-web-应用)
- [引用](#引用)
- [Star 趋势](#star-趋势)

## 简介

MOSS-Transcribe-Diarize 是我们的旗舰级 SOTA 模型系列，能够一次性将真实世界的长音频转换为结构化、带说话人标注的转写结果。它不再依赖将独立的 ASR 与说话人分离系统拼接起来的方案，而是联合完成语音转写与说话人分离，直接输出带精确时间戳、说话人标签（如 `[S01]`、`[S02]` 等）一致的时间对齐文本。

MOSS-Transcribe-Diarize 面向会议、通话、播客、访谈、讲座和视频内容而设计，能够可靠地处理长时、嘈杂、多说话人的录音。它还可以选择性地输出声学事件标注，让下游系统更全面地理解“发生了什么、谁在说话、何时说话”。

模型接收原始音频，输出紧凑的带时间戳转写文本。标准输出格式为：

```text
[start_time][Sxx]transcribed speech[end_time]
```

时间戳以秒为单位，相邻片段会拼接成单条连续序列，例如：

```text
[0.48][S01]Welcome everyone[1.66][12.26][S02]The new transcription pipeline is ready for evaluation[13.81][14.36][S01]Great, include the diarization results in the report[18.76]
```

## 模型架构

<p align="center">
  <img src="./assets/Model_Architecture.png" alt="MOSS-Transcribe-Diarize model architecture" width="90%" />
</p>

| 组件 | 规格说明 |
|---|---|
| 文本主干 | Qwen3-0.6B 风格的因果解码器（causal decoder） |
| 音频编码器 | Whisper-Medium encoder 配置 |
| 音频前端 | `WhisperFeatureExtractor`，16 kHz，80 个 mel 频带，30 秒分块 |
| 音频-文本桥接 | 4 倍时间维度合并（temporal merge）+ MLP 适配器 |
| 特征融合 | 音频特征通过 `masked_scatter` 替换 <code>&lt;&#124;audio_pad&#124;&gt;</code> 的 embedding |
| 输出格式 | 紧凑的 `[start][Sxx]text[end]` 转写文本，含 `[S01]` 等说话人标签 |

## 评测

### 客观评测

我们使用三个客观指标评测 MOSS-Transcribe-Diarize：字符错误率（CER）、拼接后最小排列字符错误率（cpCER）以及 Δcp。所有指标均为越低越好。最优结果以**加粗**表示，次优结果以<ins>下划线</ins>表示。短横线（`-`）表示该结果不可用。

<div style="overflow-x: auto;">
<table style="white-space: nowrap;">
  <thead>
    <tr>
      <th rowspan="2" style="min-width: 220px;">Model</th>
      <th colspan="3" style="text-align:center;">AISHELL&#8209;4</th>
      <th colspan="3" style="text-align:center;">Alimeeting</th>
      <th colspan="3" style="text-align:center;">Podcast</th>
      <th colspan="3" style="text-align:center;">Movies</th>
    </tr>
    <tr>
      <th>CER↓</th><th>cpCER↓</th><th>Δcp↓</th>
      <th>CER↓</th><th>cpCER↓</th><th>Δcp↓</th>
      <th>CER↓</th><th>cpCER↓</th><th>Δcp↓</th>
      <th>CER↓</th><th>cpCER↓</th><th>Δcp↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap;">Doubao</td>
      <td>18.18</td><td>27.86</td><td>9.68</td>
      <td>25.25</td><td>37.57</td><td>12.31</td>
      <td>7.93</td><td>10.54</td><td>2.61</td>
      <td>9.94</td><td>30.88</td><td>20.94</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">ElevenLabs</td>
      <td>19.58</td><td>37.95</td><td>18.36</td>
      <td>25.70</td><td>36.69</td><td>10.99</td>
      <td>8.50</td><td>11.34</td><td>2.85</td>
      <td>11.49</td><td>17.85</td><td>6.37</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">GPT-4o</td>
      <td>-</td><td>-</td><td>-</td>
      <td>-</td><td>-</td><td>-</td>
      <td>-</td><td>-</td><td>-</td>
      <td>14.37</td><td>23.67</td><td>9.31</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">Gemini 2.5 Pro</td>
      <td>42.70</td><td>53.42</td><td>10.72</td>
      <td>27.43</td><td>41.64</td><td>14.21</td>
      <td>7.38</td><td>10.23</td><td>2.85</td>
      <td>15.46</td><td>24.15</td><td>8.69</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">Gemini 3 Pro</td>
      <td>22.75</td><td>27.43</td><td>4.68</td>
      <td>26.75</td><td>32.84</td><td>6.09</td>
      <td>-</td><td>-</td><td>-</td>
      <td>8.62</td><td>14.73</td><td><ins>6.11</ins></td>
    </tr>
    <tr>
      <td style="white-space: nowrap;">VIBEVOICE ASR</td>
      <td>21.40</td><td>24.99</td><td>3.59</td>
      <td>27.40</td><td>29.33</td><td>1.93</td>
      <td>27.94</td><td>48.30</td><td>20.36</td>
      <td>14.59</td><td>42.54</td><td>27.94</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><b>MOSS Transcribe Diarize 0.9B</b></td>
      <td><ins>14.84</ins></td><td><ins>15.83</ins></td><td><ins>0.99</ins></td>
      <td><ins>24.86</ins></td><td><ins>22.17</ins></td><td><ins>-2.69</ins></td>
      <td><ins>5.97</ins></td><td><ins>7.37</ins></td><td><b>1.40</b></td>
      <td><ins>6.36</ins></td><td><ins>12.76</ins></td><td>6.40</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><b>MOSS Transcribe Diarize Pro</b></td>
      <td><b>13.78</b></td><td><b>14.02</b></td><td><b>0.24</b></td>
      <td><b>18.22</b></td><td><b>13.94</b></td><td><b>-4.27</b></td>
      <td><b>4.46</b></td><td><b>6.97</b></td><td><ins>2.51</ins></td>
      <td><b>5.86</b></td><td><b>11.78</b></td><td><b>5.92</b></td>
    </tr>
  </tbody>
</table>
</div>

## 快速开始

### 环境准备

请使用干净的 Python 环境。本项目在 Python 3.12 与 Transformers 5.x 上测试通过。

```bash
git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git
cd MOSS-Transcribe-Diarize
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[torch-runtime]" --torch-backend=auto
```

微调说明请参阅 [FINETUNING.md](FINETUNING.md)。

### Python 用法

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import (
    build_transcription_messages,
    generate_transcription,
    resolve_device,
)

model_id = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
audio_path = "audio.wav"

device = resolve_device("auto")
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
).to(dtype=dtype).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

messages = build_transcription_messages(audio_path)
result = generate_transcription(
    model,
    processor,
    messages,
    max_new_tokens=2048,
    do_sample=False,
    device=device,
    dtype=dtype,
)

print(result["text"])

for segment in parse_transcript(result["text"]):
    print(segment.start, segment.end, segment.speaker, segment.text)
```

消息流程遵循常见的 Qwen 多模态范式。对话模板由 `AutoProcessor` 从模型侧加载：

1. `processor.apply_chat_template(messages, tokenize=False)` 渲染文本并插入音频占位符。
2. `process_audio_info(messages, sampling_rate)` 从同一份 messages 中加载音频波形。
3. `processor(text=text, audio=audios)` 计算 Whisper 输入特征并展开音频占位符。
4. `model.generate(...)` 生成带时间戳的转写与说话人分离文本。

### 使用 SGLang Omni 部署

[SGLang Omni](https://github.com/sgl-project/sglang-omni) 是 MOSS-Transcribe-Diarize 推荐的部署后端，通过兼容 OpenAI 的 `/v1/audio/transcriptions` 接口，为长音频提供经过优化的推理能力。

SGLang Omni 目前面向 CUDA 13 环境。请参考官方[安装指南](https://github.com/sgl-project/sglang-omni/blob/main/docs/get_started/installation.md)完成受支持的配置。若为 CUDA 12 环境，也可使用下文的 vLLM 方案。

下载模型：

```bash
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize
```

启动服务：

```bash
sgl-omni serve \
  --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --port 8000 \
  --max-running-requests 16 \
  --cuda-graph-max-bs 16 \
  --mem-fraction-static 0.80
```

当你需要解析后的说话人分段时，请使用 `response_format=verbose_json`。`json` 仅返回原始转写文本。

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@audio.wav \
  -F response_format=verbose_json
```

```python
import requests

with open("audio.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            "response_format": "verbose_json",
        },
        files={"file": ("audio.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
payload = resp.json()
print(payload["text"])
for segment in payload.get("segments", []):
    print(f"[{segment['start']:.2f}-{segment['end']:.2f}] {segment['text']}")
```

对于更长的多说话人音频，请调大 `max_new_tokens`，以便解码器完整生成整段分离后的转写：

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=OpenMOSS-Team/MOSS-Transcribe-Diarize \
  -F file=@audio.wav \
  -F response_format=verbose_json \
  -F max_new_tokens=65536
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `file` | file | 必填 | 以 multipart form data 上传的音频文件 |
| `model` | string | 服务端默认 | 模型标识符 |
| `language` | string | 未设置 | 可选的语言提示 |
| `response_format` | string | `json` | `json`、`verbose_json` 或 `text` |
| `temperature` | float | 模型默认（`0.0`） | 采样温度 |
| `max_new_tokens` | int | `5120` | 最大生成 token 数；长音频请调大，例如 `65536` |
| `prompt` | string | 未设置 | 可选的指令覆盖；留空则使用内置的转写+分离 prompt |

关于基准测试、性能数据与实现细节，请参阅 [SGLang Omni cookbook](https://github.com/sgl-project/sglang-omni/blob/main/docs/cookbook/moss_transcribe_diarize.md)。下列结果基于单张 H100，分别针对短序列与长序列的多说话人 ASR 任务。

`movies` 短序列 ASR：

| 并发数 | 吞吐（req/s） | 平均延迟（s） | RTF 均值 | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 2.57 | 0.388 | 0.0612 | 29.76 |
| 2 | 4.89 | 0.409 | 0.0659 | 56.55 |
| 4 | 6.62 | 0.513 | 0.0790 | 76.64 |
| 8 | 6.80 | 0.533 | 0.0810 | 78.70 |
| 16 | 7.08 | 0.659 | 0.0922 | 81.98 |

`aishell4_long` 长序列 ASR：

| 并发数 | 吞吐（req/s） | 平均延迟（s） | RTF 均值 | audio_s/s |
|---:|---:|---:|---:|---:|
| 1 | 0.022 | 45.2 | 0.0197 | 50.64 |
| 2 | 0.032 | 60.7 | 0.0265 | 74.25 |
| 4 | 0.036 | 105.6 | 0.0461 | 81.64 |
| 8 | 0.040 | 172.6 | 0.0754 | 90.62 |
| 16 | 0.043 | 282.8 | 0.1237 | 98.83 |

### 使用 vLLM 部署

MOSS-Transcribe-Diarize 通过兼容 OpenAI 的转写 API 支持 vLLM 部署。请使用已包含 MOSS-Transcribe-Diarize 模型注册的固定版 vLLM nightly 构建。从下列命令中任选其一：CUDA 12 环境使用 `cu129`，CUDA 13 环境使用 `cu130`。

```bash
uv pip install -U vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129
```

或：

```bash
uv pip install -U vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu130
```

```bash
vllm serve OpenMOSS-Team/MOSS-Transcribe-Diarize --trust-remote-code
```

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F model="OpenMOSS-Team/MOSS-Transcribe-Diarize" \
  -F file=@"audio.wav" \
  -F response_format="json" \
  -F temperature="0"
```

### 自定义 Prompt 与热词

默认 prompt 针对带时间戳的转写与说话人分离进行了优化：

```text
请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。
```

若要添加热词，只需在默认 prompt 后追加简短提示：

```text
请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。热词提示：热词1, 热词2, 热词3
```

更多 prompt 用例见 [examples/prompts.md](examples/prompts.md)。同一个 prompt 可传入 `build_transcription_messages`、`mtd-subtitle` 与 `mtd-subtitle-web`。

### 字幕 Web 应用

本工具包还内置了一个本地字幕工作流，支持上传、审阅、字幕导出，以及可选的 FFmpeg 压制（burn-in）：

```bash
mtd-subtitle-web \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --host 127.0.0.1 \
  --port 7860
```

打开 `http://127.0.0.1:7860`，上传音频/视频文件，审阅解析出的字幕分段，然后下载 JSON/SRT/ASS；若 `PATH` 中存在 `ffmpeg` 与 `ffprobe`，还可压制生成 MP4。
Web 界面支持简体中文和英文。首次使用时会跟随浏览器语言，也可通过页头的语言选择器即时切换并记住选择。

批量处理：

```bash
mtd-subtitle /path/to/input.mp4 \
  --model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --out-dir runs/example \
  --render
```

## 引用

如果你使用了 MOSS-Transcribe-Diarize，请引用我们的技术报告：

```bibtex
@misc{moss_transcribe_diarize_2026,
  title={MOSS Transcribe Diarize Technical Report},
  author={{MOSI.AI}},
  year={2026},
  eprint={2601.01554},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2601.01554}
}
```

## Star 趋势

<p align="center">
  <a href="https://www.star-history.com/#OpenMOSS/MOSS-Transcribe-Diarize&amp;Date">
    <img width="700" alt="Star History Chart" src="https://api.star-history.com/svg?repos=OpenMOSS%2FMOSS-Transcribe-Diarize&amp;type=Date">
  </a>
</p>
