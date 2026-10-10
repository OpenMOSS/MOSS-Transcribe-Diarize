# 实时会议语音转写工具设计

日期：2026-09-24
状态：待实现
基线版本：`moss-transcribe-diarize` 0.1.0

## 1. 背景与目标

MOSS-Transcribe-Diarize 是一个端到端的音频理解模型，能一次性把长音频转成带时间戳和说话人标签的结构化文本。仓库里已有的 `mtd-subtitle` / `mtd-subtitle-web` 面向**文件**：上传、转写、校对、导出字幕。本设计新增的是一个面向**正在进行中的会议**的工具：边开会边出字。

### 要达成的结果

1. 在浏览器里抓取会议音频（麦克风 + 会议软件/系统声音），边说边出字。
2. 两级并存显示：已定稿的稳定段落，加上仍在修正中的临时文字。
3. 说话人编号在整个会话内全局一致——同一个人全程是同一个 `Sxx`，并支持改成真实姓名。
4. 会话结束后能导出全文（Markdown / SRT / JSON）。

### 已确定的约束

| 决策 | 选择 |
|---|---|
| 音频来源 | 浏览器捕获（`getUserMedia` + `getDisplayMedia`），WebSocket 送后端 |
| 实时策略 | 两级并存：定稿段 + 临时段 |
| 推理后端 | 复用 OpenAI 兼容的 vLLM / SGLang 服务（同时保留本地 HF 后端） |
| 说话人一致性 | 跨窗口全局一致，靠声纹嵌入 |
| 声纹实现 | ONNX + `onnxruntime`（实时进程不引入 torch） |
| 服务形态 | 独立服务 `mtd-realtime`，与字幕工坊解耦 |

### 非目标（明确不做）

- 跨会话的说话人识别（同一批人下次开会仍是新编号）
- 跨窗口的文本上下文条件（窗口只喂音频，保持确定性和可测性）
- 音频内存直传（v1 统一走临时 wav 文件适配层）
- 实时翻译、声学事件标注
- 多用户、鉴权、TLS

## 2. 核心约束：模型不是流式的

这是整个设计的地基，必须先说清楚。

`MossTranscribeDiarizeProcessor` 的音频前端按固定 `n_samples`（30 秒）切块、补齐、堆叠成 `input_features`，模型对整段音频做一次性全序列生成。**没有增量解码，没有跨段 KV 复用**。

因此"实时"只能靠**滑动窗口重复推理**实现：每隔 `hop` 秒，对最近 `W` 秒音频重跑一次完整转写，再从结果里挑出稳定的部分定稿。这带来两个直接后果：

1. **算力开销是 `W / hop` 倍实时**。默认 `W=20, hop=5` 就是 4 倍。0.9B 模型配合 vLLM 批处理应该撑得住，但这必须实测，且 `W`/`hop`/`tail` 必须是可调旋钮。
2. **已定稿的段落不会被回改**。定稿意味着我们放弃了后续修正。`tail` 参数就是拿延迟换稳定性的旋钮。

## 3. 总体架构

```
浏览器 realtime.html
  mic(getUserMedia) ─────┐
  系统声(getDisplayMedia) ┴→ AudioContext({sampleRate:16000})
                              → AudioWorklet → Float32 PCM
                                    ↓ WebSocket 二进制帧
mtd-realtime (FastAPI + uvicorn)
  RealtimeSession（每连接一个）
    ├ AudioRingBuffer    保留最近 180 秒 16k 单声道
    ├ WindowPolicy       决定何时、对哪一段跑推理
    ├ WindowTranscriber  窗口写临时 wav → 调现有 runner
    ├ TranscriptStreamParser（直接复用）
    ├ Stitcher           局部→绝对时间、commit 水位线、去重
    └ SpeakerGallery     fbank → ONNX 嵌入 → 余弦匹配 → 全局说话人
  SessionStore           runs/realtime/<session-id>/ 落盘
```

### 关键性质：基本增量，两处上游改动用于隔离 torch

`ModelRunner`、`server.py`、`JobManager` 一行都不改。

**torch 隔离问题。** 包顶层 `moss_transcribe_diarize/__init__.py` 无条件 import 了 `modeling_*` 和 `processing_*`，而这两个模块以及 `inference_utils.py` 都在模块顶层 `import torch`。因此只要 `import moss_transcribe_diarize.realtime.*`，父包的 `__init__.py` 会先执行，torch 和 transformers 就被一并拉起来——实时进程即便一行模型代码都不跑，也要为此装上 2.5 GB 的 torch。同样地，现有 `vllm_runner.py` 因为 `from .model_runner import ...` 和 `from ...inference_utils import ...`，本身也不是 torch-free 的，所以它不能被实时路径直接复用。

设计要兑现"实时进程不引入 torch"，需要三处上游改动，都服务于同一个目标：

- `moss_transcribe_diarize/__init__.py` 改为 PEP 562 惰性导入（模块级 `__getattr__`）。现有用法 `from moss_transcribe_diarize import TranscriptSegment` 继续工作，而 `import moss_transcribe_diarize.realtime.config` 不再拖入 torch。
- `DEFAULT_PROMPT` 从 `inference_utils.py` 提到新的 torch-free 模块 `moss_transcribe_diarize/prompts.py`，`inference_utils` 改为从那里导入并保持同名再导出。实时管线需要这个默认 prompt，但它不该为了一个字符串去 import 一个顶层 `import torch` 的模块。
- 把 `vllm_runner.py` 里的 multipart 构造、wav 字节编码、SSE 消费、响应文本提取抽到新的 torch-free 模块 `app/openai_audio_client.py`。`VllmRunner` 和实时转写器共用同一份实现，既没有重复代码，实时路径也不碰 torch。现有 `tests/test_vllm_runner.py` 是这次抽取的回归保障。

三处都是小改，且都有现成测试兜底。除此之外，改动只限于 `pyproject.toml`（新增 `[project.optional-dependencies] realtime` 和 `[project.scripts] mtd-realtime`）和 README 新增一节。

## 4. 模块设计

### 4.1 `realtime/buffer.py` — `AudioRingBuffer`

固定容量的 16 kHz 单声道 float32 环形缓冲。

```python
class AudioRingBuffer:
    def __init__(self, capacity_seconds: float = 180.0, sample_rate: int = 16000): ...
    def append(self, pcm: np.ndarray) -> None: ...
    def slice(self, start_sec: float, end_sec: float) -> np.ndarray | None: ...
    @property
    def total_seconds(self) -> float: ...
```

- `append` 接受任意长度的 float32 一维数组，超出容量的最旧数据被丢弃。
- `slice` 返回窗口音频；若请求区间有任意部分已被丢弃，返回 `None`（调用方据此跳过本次推理）。
- 容量 180 秒约 11.5 MB，可忽略。默认容量须 ≥ `W + 2 * hop`。
- 线程安全：接收音频在事件循环线程，推理在工作线程，内部用 `threading.Lock`。

### 4.2 `realtime/window.py` — `WindowPolicy`

纯逻辑，无 IO，可完全单测。

```python
@dataclass(frozen=True)
class WindowDecision:
    should_run: bool
    start_sec: float
    end_sec: float

class WindowPolicy:
    def __init__(self, *, window: float, hop: float, min_first_window: float): ...
    def decide(self, *, total_seconds: float, last_run_sec: float | None) -> WindowDecision: ...
```

规则：

- 首次运行：`total_seconds >= min_first_window`（默认 8.0）时触发
- 后续运行：`total_seconds - last_run_sec >= hop`（默认 5.0）时触发
- 决策成立时 `end_sec = total_seconds`；`start_sec` 分两种：
  - **首次运行恒为 0**
  - 后续运行为 `max(0.0, total_seconds - window)`
- 决策不成立时：`should_run=False`，`start_sec`/`end_sec` 无意义

首次运行不等满 `W` 秒，是为了避免开场 20 秒一片空白。窗口短一些对模型输出质量影响有限，而且首个窗口左边界没有更早的音频，本来也不需要左侧上下文。

**首次运行的 `start_sec` 必须是 0，不能套用后续运行那条公式。** 若统一用 `max(0.0, total_seconds - window)`，当 `min_first_window > window` 时首次窗口的左边界会落在 0 之后；而后续窗口都是 `[t - window, t]`，永远够不到开头。于是 `[0, total_seconds - window)` 这段音频不会被任何一次推理覆盖，被**静默丢弃**——转写看起来一切正常，只是开头少了内容，下游没有任何东西能发现。

这条约束的代价：首次窗口可能长达 `min_first_window` 秒，超过 `window`。会话的输出 token 预算按 `window` 推算，所以当 `min_first_window` 明显大于 `window` 时首个窗口的预算会偏紧。默认配置（8.0 < 20.0）下两者一致，不受影响。

### 4.3 `realtime/stitch.py` — `Stitcher`

把一次窗口推理的原始文本，变成定稿段和临时快照。

```python
@dataclass(frozen=True)
class Segment:
    start: float          # 绝对秒，相对会话开始
    end: float
    speaker: str          # 该窗口内的局部标签，如 "S02"
    text: str
    window_id: int

class Stitcher:
    def __init__(self, *, window: float, tail: float): ...
    @property
    def committed_until(self) -> float: ...
    def ingest(self, *, window_start: float, window_end: float,
               raw_text: str, window_id: int) -> StitchResult: ...
    def flush(self) -> list[Segment]: ...
```

`StitchResult` 含 `committed: list[Segment]`（本次新增的定稿段，可能为空）和 `provisional: list[Segment]`（整体替换用的临时快照）。

`ingest` 的步骤：

1. 用现成的 `TranscriptStreamParser` 解析 `raw_text`，`close()` 收尾，得到局部时间的段。

   解析器会丢弃它无法解析的部分，所以多数格式偏离只损失对应片段、不会让整次推理作废。

   **必须知道的例外**，全部源于解析器的 `_after_end`：`[end]` 之后若跟着非空白的杂散文本，它会把 `[end]` **折回正文**并退回"读取正文"状态。后果取决于杂散文本的位置：

   - **流尾**（后面没有更多内容）：该片段永不闭合，而 `close_into` 只在 `_AFTER_END` 状态才吐出片段——于是**窗口的最后一段丢失**。
   - **中间位置**：紧随其后的 `[时间戳]` 被 `_read_text` 当作**本片段**的结束时间（只要它 `>= start` 就会被 `_read_end` 接受）。该片段的文字因此被污染（正文里混入 `[1.5]垃圾`）、结束时间被顶到下一段的起点，而那个 `[Sxx]` 在 `_READ_START` 状态下无法解析成时间戳、被 reset 丢弃，**下一段随之静默消失**。这比单纯丢失更糟。

   注意"后续片段照常存活"**只在杂散文本由空白分隔时成立**——空白走 `_pending_after_end` 分支，下一个 `[` 到来即干净地 emit。无分隔时不成立。

   实时管线**不做修补**：既不修改解析器（它是已发布模块，`subtitle/` 与文件工坊都在依赖，且本 spec 只允许三处上游改动），也不在 `parse_window_segments` 里加"正文含方括号就丢弃"之类的判据。

   **为什么不加判据**：解析器把"正文里本来就有的方括号"和"被折回的时间戳"处理成**完全同一种结果**——两者都走 `_after_end` 折回正文，后面跟的都是任意文本，所以**原理上无法区分**。第 2 种例外（污染）与 `transcript_parser` 刻意保留正文方括号的契约（`tests/test_transcript_parser.py` 的 `test_numeric_brackets_inside_text_are_preserved` 明确声明 `第[2024]年，编号[001]继续` 是正确输出）在结果上同形。任何基于方括号的判据都会误伤合法内容，而误伤是**静默丢内容**——比带可见 `[1.5]` 痕迹的污染更隐蔽。

   所以两种例外都只能由**下一个窗口重新覆盖该段音频**来恢复：代价是暂时性的、可见的。这是本设计在"宁可留下可见的瑕疵"与"宁可静默少内容"之间的明确选择。
2. 绝对时间换算：`abs_start = window_start + local.start`，`abs_end = window_start + local.end`。
3. 边界处理：丢弃完全落在窗口外的段；跨出窗口**左**边界的段把 `abs_start` 裁剪到 `window_start`（模型偶尔会给出早于窗口起点的时间戳，裁剪比丢弃少丢内容）；跨出窗口**右**边界的段直接丢弃（下一轮会重新看到它）；裁剪后长度短于 0.05 秒的段丢弃。
4. 按 `abs_start` 升序排序。
5. 丢弃 `abs_end <= committed_until` 的段（已经定稿过）。
6. 丢弃**跨定稿水位线**的段（`abs_start < committed_until < abs_end`）。这类段几乎总是上一次运行已定稿内容的重叠部分，保留会造成重复——对实时字幕来说，重复比少量丢失更刺眼。
7. 定稿区：`abs_end <= window_end - tail` 的段全部定稿，推进 `committed_until = 最后一个定稿段的 end`。
8. 临时区：`abs_end > window_end - tail` 的段构成临时快照，按时间排序。

`flush()` 在会话结束时调用，把临时区里剩下的段全部定稿并返回。调用顺序是：会话先用正常的 `tail` 对最后 `W` 秒再跑一次窗口推理并 `ingest`，然后 `flush()`。这样临时区用的是最新一次推理的结果，而不是可能已经过时的旧快照。

**为什么水位线能天然去重**：第 7 步只定稿 `abs_end <= window_end - tail` 的段，而 `window_end` 每次前进 `hop`。相邻两次运行的定稿区在时间上重叠，但第 5、6 步保证了同一条内容只会被定稿一次。

### 4.4 `realtime/speaker.py` — 声纹与说话人库

```python
class SpeakerEmbedder(Protocol):
    embedding_dim: int
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None: ...

class OnnxCampplusEmbedder:
    """kaldi-native-fbank + onnxruntime，不依赖 torch。"""

class SpeakerGallery:
    def __init__(self, embedder: SpeakerEmbedder | None, *, threshold: float = 0.55,
                 min_segment_sec: float = 0.4): ...
    def assign(self, segments: list[Segment], audio_of: Callable[[Segment], np.ndarray | None]
               ) -> list[Assignment]: ...
    def rename(self, global_id: str, name: str) -> None: ...
    def speakers(self) -> list[dict]: ...
```

**`OnnxCampplusEmbedder`**：`kaldi_native_fbank` 提取 80 维 fbank（`frame_opts.samp_freq=16000`、`dither=0.0`、`mel_opts.num_bins=80`，保持 Kaldi 默认的 25ms 窗 / 10ms 移），送 onnxruntime session（输入名为 `session.get_inputs()[0].name`，形状 `[1, T, 80]`），输出 192 维嵌入，做 L2 归一化。单条音频过短（有效帧数 < 10）时返回 `None`。

**`SpeakerGallery.assign`** 的关键是**利用模型自己的局部标签作为强先验**。同一窗口内局部标签相同的段，几乎肯定是同一个人。因此按 (窗口, 局部标签) 分组处理：

1. 每个段按自己的 `[start, end]` 切音频。短于 `min_segment_sec` 的段拿不到可靠嵌入，跳过声纹。
2. 组内取有效嵌入中最长段的那个作为该组的代表嵌入（最长段信噪比最好）。
3. 代表嵌入与库中每个说话人的质心算余弦相似度：
   - 最高分 ≥ `threshold` → 归入该说话人，用滑动平均更新质心 `centroid = normalize(centroid * n + e)`、`n += 1`
   - 否则新建全局说话人 `S{k}`，质心为该嵌入、`n = 1`
4. 组内其余段（含因过短而跳过声纹的段）直接继承该组的全局 ID。
5. 组内**没有任何**段拿到有效嵌入时，整组继承不上——此时该组全部沿用局部标签，并在事件里标记 `speaker_confident=false`，前端以灰色显示，提示用户确认。

全局说话人 ID 从 `S01` 起顺序分配，与局部标签无关。`rename` 把 ID 映射到真实姓名，持久化到 `session.json`。

**诚实的局限**：远场、短片段（< 1 秒）、重叠说话人时 CAM++ 会不稳，误合并和误分裂都会发生。阈值和同窗口约束是两道保险，但**前端手工改说话人归属不是可选项，是必要补偿**。设计上必须保证改归属的成本足够低（下拉选择，改一段可批量应用到同组）。

### 4.5 `realtime/transcriber.py` — 窗口推理适配层

```python
class WindowTranscriber(Protocol):
    def transcribe_window(self, audio: np.ndarray, *, prompt: str) -> str: ...

class VllmWindowTranscriber:
    """torch-free。经 app/openai_audio_client.py 直连 OpenAI 兼容端点。"""

class HfWindowTranscriber:
    """把窗口写成临时 wav，调用本地 ModelRunner.transcribe(path)。"""
```

两个实现对应两个后端，选择 `WindowTranscriber` 协议是为了让 `RealtimeSession` 不关心后端差异，测试时可以直接注入假实现。

**`VllmWindowTranscriber`**：把窗口 PCM 编码成 wav 字节，调用 `app/openai_audio_client.py` 的请求函数 POST 到 `/v1/audio/transcriptions`，返回文本。全程不 import torch，也不 import `vllm_runner`——它只依赖那个新的 torch-free 模块。

**`HfWindowTranscriber`**：每次覆写 `scratch_dir/window.wav`（20 秒 16k 单声道约 640 KB），调用 `runner.transcribe(path)`，取 `TranscriptionResult.text`。这个实现会 import torch（本地跑模型本来就要），但只在 `--backend hf` 时才会被构造，`--backend vllm` 下永远不加载。

参数映射：`max_new_tokens` 按 `W` 缩放（默认 `W=20` 时约 1020），`decoding="greedy"`（实时场景要确定性），`temperature=0`。

**接线要求（阶段二必须做）**：`RealtimeConfig.effective_max_new_tokens()` 就是上面这个数（`max(256, round(window * 51))`），但阶段一里它**没有消费者**——`WindowTranscriber` 协议是 `transcribe_window(audio, *, prompt)`，不传预算，预算是构造转写器时固定的，而构造者是阶段二的 CLI。与此同时 `HfWindowTranscriber` 把同一个数硬编码了第二份（`max_new_tokens=1020`）。**CLI 必须把 `config.effective_max_new_tokens()` 传给转写器**，否则用户用 `--window 40` 时 config 算 2040、HF 后端仍用 1020，**输出被静默截断**。

**而这个按 `window` 算出的预算是错的，还有第二个原因**：4.7 的覆盖性保证会让**部分窗口长于 `window`**（被迫放行的窗口、驱动停摆后的窗口，实测可达 100 秒），按 `window` 定死的预算会把它们的尾部截掉。预算应当**按实际窗口长度**估算——转写器本来就拿得到那段音频，所以最省事的做法是让它在构造时不接死预算，改为按传入音频的长度推算（`max(256, round(实际秒数 * 51))`），`config` 的值作为上限或回退。

选择临时文件而非内存直传，是为了让两个后端共用同一条数据通路、且不改动 `ModelRunner`。写一帧不到 1 MB 的代价相对模型推理可以忽略；若后续 profiling 显示不是，再单独做内存直传优化。

### 4.6 `realtime/store.py` — `SessionStore`

```
runs/realtime/<session-id>/
  session.json       开始/结束时间、prompt、热词、参数、说话人表
  audio.wav          完整录音，边写边追加
  transcript.jsonl   定稿事件流，一条一段
  provisional.json   最后一份临时快照，仅用于崩溃恢复展示
```

- `audio.wav` 用 `soundfile` 以 `SoundFile(...)` 追加模式写入，会话中途崩溃也能保留已录部分。
- `transcript.jsonl` 每条定稿段追加一行 JSON，是崩溃恢复的权威来源。
- `SessionStore.list_sessions()` 扫描目录读 `session.json`，供历史列表使用。

### 4.7 `realtime/session.py` — `RealtimeSession`

编排层，持有上列所有组件，对外暴露三个方法。

```python
class RealtimeSession:
    def __init__(self, config: RealtimeConfig, *, transcriber: WindowTranscriber,
                 store: SessionStore, embedder: SpeakerEmbedder | None = None,
                 prompt: str = DEFAULT_PROMPT): ...
    def push_audio(self, pcm: np.ndarray) -> None: ...      # 同步
    async def run_pending(self) -> list[dict]: ...
    async def close(self) -> list[dict]: ...
```

事件是 `dict`（形状见 4.9），不是自定义类型。

**调用契约**：`run_pending` 与 `close` 都是异步的，且**两者不得并发**——驱动方必须先 await 完 `run_pending` 再调 `close`。若让它们同时在途（例如把 `run_pending` 当 task 起、不 await 就 `close()`），两个窗口会落在两个 executor 线程里并改 `Stitcher` 的水位线、`SpeakerGallery` 的表、段落计数器与 `SessionStore`，后果是重复定稿或临时区错乱，**不会报错**。不把"在 `close()` 开头检查 `_window_running`"当作修法：那会让 `close()` 直接返回空、store 永不收尾、尾部全丢，比现状更糟。

`run_pending` 每隔 0.5 秒由后台任务调用：

1. 问 `WindowPolicy.decide`。不跑就返回空。
2. **静音门控**：对窗口算 20ms 帧的 RMS，若高于 -45 dBFS 的帧占比 < 5%，判定为静音——**跳过推理但推进 `last_run_sec`**。会议里大量时间是静音，这道门控直接把实际算力开销砍下来，是成本上最划算的一个开关（可用 `--no-silence-gate` 关闭——该开关属阶段二的 CLI，阶段一里无法关闭）。

   **这道门控有代价，实现必须给它设上界。** 判据是"整窗活跃帧占比 < 5%"，20 秒窗口要越过门限需要约 1 秒以上的有声内容。设想"每 15 秒说半秒"的场合（安静演示、旁听、只答"嗯/对"）：覆盖该句的**每个**窗口活跃占比都在 2.5% 上下，于是**每个**都被跳过；而缓冲一旦过去超过 `window` 秒，后续窗口的左边界就移到它之上、**再也覆盖不到**——这不是别的失败路径那种"被下一个窗口重新覆盖"，是**永久丢失**。

   因此实现必须保证**覆盖性**。而实现方式**不是调阈值——调阈值原理上做不到**：实际的决策网格步长是 `ceil(hop / poll_interval) * poll_interval`，只有 `hop` 恰为 `poll_interval` 的整数倍时才等于 `hop`；网格还会被**音频到达粒度**拉长（`total_seconds` 按到达块推进，不按轮询推进）。会话无法预知这个步长，所以任何固定阈值都会在某些节奏下开缝——实测默认配置下"一次迟到的轮询"留下 0.5 秒永久空隙，音频按 1.5 秒成块到达（WebSocket 传输下的正常形态）留下 1 秒。

   真正的修法是**把每个执行窗口的起点钳到"已覆盖前沿"**：`start = min(decision.start_sec, 已执行的最后一个窗口的 end)`。于是窗口 *i* 的起点不晚于窗口 *i-1* 的终点，相邻执行窗口必相接或重叠；首窗起点恒为 0，故并集**无条件**从 0 起连续覆盖，与网格步长、轮询节奏、到达粒度全都无关。门控的阈值只剩一个作用：决定多久放行一次。实测 72 组参数 × 周期性迟到轮询 × 语音静音混合下无一空隙。

   `status` 事件要报告被门控跳过的窗口数，使跳过**可观测**——否则全程 `state: "running"`，与健康会话无法区分，用户无从知道自己的会议被跳过过。

   **代价**：被迫放行或**驱动停摆**时的窗口可以远长于 `window`，其上界是停顿长度而非一个步长（实测停顿到 `total=100` 时收尾窗口为 `(0.0, 100.0)`）。输出 token 预算在构造时按 `window` 固定，所以这类窗口**尾部可能被模型截断**——与首窗超预算同源（见 4.5 的接线要求）。这是"覆盖性"与"输出长度"之间的真实张力：覆盖交给调度层保证，预算留给阶段二按**实际窗口长度**估算。`audio.wav` 全量留存，所以被截断的只是转写文本，不是录音。

   另有两个已知的窄路径：驱动停摆超过环形缓冲容量时，钳位起点已不可读，实现回退到决策自身的窗口（覆盖性与修复前**逐值相同**，不会更差）；该回退的存活性隐含依赖 `buffer_capacity ≥ window + 2 * hop`，而 `RealtimeConfig` 强制了这一条。

   `silence_frame_ratio` 的默认值（0.05）本身是产品取舍：调低它减少跳过、增加算力。默认值下"不足 1 秒的孤立发声"仍可能被单窗判为静音，但**上面的覆盖性保证使这种窗口最终一定会被某次推理覆盖**（延迟有界，不是丢失）。门控放行间隔的**下界**是 `window - hop`；**上界不是 `window`**——实测 12 秒一块、`window=20`/`hop=5` 时间隔为 24 秒，因为网格步长可被拉长到大于 `hop`。
3. 取窗口音频，`loop.run_in_executor` 里跑转写（阻塞的 HTTP 和文件 IO 不能占住事件循环）。
4. 已有窗口在跑时跳过本次，不改 `last_run_sec`；下一次循环再判断。模型是瓶颈，堆队列只会让延迟持续增长。
5. `Stitcher.ingest` 切分定稿和临时。
6. 定稿段交给 `SpeakerGallery.assign` 拿全局说话人。
7. 追加写 `SessionStore`，返回事件。

兜底与降级：

- 单次窗口推理抛异常：记 `error` 事件，不终止会话，`last_run_sec` 照常推进，下次重试。
- 连续失败达到 `max_consecutive_failures`（默认 3）次：发 `status` 事件把会话标为 `degraded`，前端显性提示。成功一次即复位。
- 落后超过 `3 * W`：发 `status` 事件报告积压，前端提示"算力跟不上"。

### 4.8 `app/realtime_server.py` — FastAPI 应用

```python
def create_realtime_app(*, config: RealtimeConfig, runner, embedder) -> FastAPI: ...
```

路由：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 返回 `realtime.html` |
| WS | `/ws/realtime` | 音频 + 控制 |
| GET | `/api/runtime` | 后端信息、配置、声纹模型状态、vLLM 连通性 |
| GET | `/api/sessions` | 历史会话列表 |
| GET | `/api/sessions/{id}` | 单个会话详情（含定稿段和说话人表） |
| GET | `/api/sessions/{id}/export?format=srt\|json\|ass\|md\|txt` | 导出。`srt`/`json`/`ass` 直接复用 `moss_transcribe_diarize/subtitle/export.py`；`md`/`txt` 在实时模块里各是一个小格式化函数（该包没有这两种） |
| GET | `/api/sessions/{id}/audio` | 完整录音 |
| POST | `/api/sessions/{id}/retranscribe` | 用落盘录音按文件模式重转一遍 |

**导出复用说明**：`subtitle` 包里的 `SubtitleSegment` 与 `realtime.stitch.Segment` 字段不完全一致，用一个转换函数桥接（`Segment` → `SubtitleSegment`），说话人显示名从 `session.json` 的说话人表取。

### 4.9 WebSocket 协议

二进制帧 = PCM：float32 小端、单声道、16 kHz，长度任意（客户端按 ~100 ms 一帧发送）。

文本帧 = JSON 控制指令（客户端 → 服务端）：

```json
{"type": "start", "prompt": "...", "hotwords": ["..."], "session_name": "..."}
{"type": "stop"}
{"type": "set_prompt", "prompt": "..."}
{"type": "set_hotwords", "hotwords": ["..."]}
{"type": "rename_speaker", "speaker_id": "S02", "name": "张总"}
{"type": "reassign_segment", "segment_id": "seg-42", "speaker_id": "S01"}
```

JSON 事件（服务端 → 客户端）：

```json
{"type": "session", "session_id": "...", "started_at": ..., "config": {...}}

{"type": "provisional", "segments": [
  {"start": 41.2, "end": 43.8, "speaker": "S03", "text": "那么我们下周"}
]}

{"type": "committed", "segments": [
  {"id": "seg-42", "start": 38.0, "end": 41.2, "speaker": "S03",
   "speaker_name": "张总", "text": "这个方案我来跟进", "speaker_confident": true}
]}

{"type": "speaker", "speakers": [{"id": "S01", "name": "李工", "samples": 12}]}

{"type": "status", "state": "running", "buffered_sec": 63.4, "rtf": 0.42,
 "last_window_ms": 2140, "lag_sec": 3.2, "degraded": false,
 "gated_windows": 4, "gated_sec": 80.0}

{"type": "error", "code": "transcribe_failed", "detail": "..."}
```

`provisional` 是**整体替换**语义，客户端收到就丢掉旧的临时段。`committed` 是**增量追加**语义。

`rtf`（real-time factor）= 单次窗口推理耗时 / `hop`，小于 1 表示算力跟得上。这是前端最该显性展示的指标。

`lag_sec` 的定义是**字幕落后实时多少秒**，即 `buffered_sec - committed_until`。它天然包含 `window + tail` 的开销，所以稳态下不会低于 `tail`——这是用户实际感受到的延迟，也是前端该展示的数。

不要用"缓冲里有多少秒音频还没跑过推理"来定义它：`last_run_sec` 每次运行后都会被设成窗口末尾，那个差值恒等于零，测不出任何东西。真正能反映"算力跟不上"的是 `rtf`。

### 4.10 `app/realtime_cli.py` — `mtd-realtime`

```
mtd-realtime \
  --backend vllm \
  --vllm-base-url http://127.0.0.1:8000 \
  --vllm-model OpenMOSS-Team/MOSS-Transcribe-Diarize \
  --host 127.0.0.1 --port 7870 \
  --window 20 --hop 5 --tail 6 \
  --speaker-threshold 0.55 \
  --runs-dir runs/realtime
```

本地后端同样可用：`--backend hf --model <path> --device cuda:0 --dtype bf16`，直接复用现有的 `ModelRunner`。

附加开关：

- `--speaker-model <path>`：本地声纹模型路径；不指定则自动下载到 `~/.cache/mtd-speaker/`
- `--no-speaker`：关闭全局说话人一致，退化为每窗口局部标签（测试和降级用）
- `--no-silence-gate`：关闭静音门控
- `--no-record`：不落盘完整录音，只保留转写文本（见第 9 节）
- `--max-new-tokens`：覆盖按窗口长度推算的默认值

## 5. 浏览器侧

### 5.1 音频采集（`audio-worklet.js` + `realtime.js`）

1. 麦克风：`getUserMedia({audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}})`。**这三项必须关掉**——AEC/NS/AGC 是为通话设计的，远场人声会被当噪声削掉，对转写是纯损失。
2. 系统声音：`getDisplayMedia({audio: true, video: true})`。Windows 上选"分享整个屏幕"并勾选"分享系统音频"，或直接分享会议所在标签页。用户在页面上点按钮才触发（浏览器要求用户手势）。
3. 混音：每路 `MediaStreamAudioSourceNode` → `GainNode`（独立音量/静音）→ 汇总到同一个 `GainNode`。只开麦克风也能用（面对面会议），系统声是可选加装。
4. 采样率：`new AudioContext({sampleRate: 16000})`，让浏览器直接重采样到 16 kHz，省掉自己写重采样。不支持时回退到默认采样率，在 worklet 里做线性插值重采样并上报一次提示。
5. `AudioWorkletProcessor` 在音频线程累积 128 帧的渲染量子，攒够 1600 样本（100 ms）发一帧 `Float32Array`。跑在音频线程的好处是不受主线程渲染卡顿影响。
6. 传输：WebSocket 定长二进制帧。`ws.bufferedAmount` 超过 2 MB 时丢弃本帧并累加计数器，`status` 里上报"网络拥塞"——实时工具宁可丢音也不能在浏览器里无限堆内存。

### 5.2 界面

单页，两栏：

- **左栏（定稿区）**：按时间顺序的定稿段，每段显示时间、说话人（颜色 + 可改的名字）、文字。自动滚到底部，用户手动向上滚则暂停跟随。段右侧有"改说话人"下拉，可勾选"应用到本组全部段"。
- **右栏（临时区）**：最后几条临时段，以降低不透明度和斜体呈现，明确区别于定稿内容。整体替换时不做逐字动画，避免闪动。
- **顶栏**：会话名、录音/暂停/停止、输入源状态（麦克风、系统声各自的电平和静音开关）、`rtf` 与积压秒数的实时指示、后端信息。
- **底栏**：导出（Markdown / SRT / JSON / TXT）、"重跑本次会话"、"历史会话"。

说话人颜色从固定调色板按 ID 稳定分配。i18n 沿用现有 `assets/i18n.js` 的写法，新增 `realtime` 命名空间到 `locales/zh-CN.json` 和 `locales/en.json`。

## 6. 测试策略

原则：所有有算法含量的部分都用纯逻辑测试覆盖，外部依赖全部可注入。

| 测试文件 | 覆盖内容 |
|---|---|
| `test_realtime_buffer.py` | 环形回绕、容量淘汰、`slice` 越界返回 `None`、并发追加 |
| `test_realtime_window.py` | 首次运行门槛、`hop` 节流、窗口左边界钳位 |
| `test_realtime_stitch.py` | 绝对时间换算、跨水位线丢弃、定稿去重、`tail` 边界、`flush` 收尾、解析器最后一段不丢 |
| `test_realtime_speaker.py` | 用**注入的假 embedder**（确定性向量）测：阈值命中/落空、质心滑动平均、同窗口同标签强制归并、全组无有效嵌入时的降级标记 |
| `test_realtime_session.py` | 用**假 transcriber**（按窗口返回预置文本）跑端到端：喂合成 PCM，断言定稿序列、绝对时间、说话人归属、静音门控跳过、连续失败降级 |
| `test_realtime_api.py` | FastAPI `TestClient` 的 WebSocket 支持：连接握手、发音频收事件、控制指令、断线清理、导出端点 |
| `test_realtime_speaker_onnx.py` | 真实声纹链路，标记 `slow`，需要模型文件，默认跳过 |

`SpeakerEmbedder` 和 `WindowTranscriber` 都是 Protocol，测试里注入假实现即可——**核心测试完全不依赖 onnxruntime、模型文件或 vLLM 服务**。

`scripts/realtime_client.py`：无浏览器冒烟客户端，读一个 wav 文件按实时速度推 `WS /ws/realtime`，打印事件流。没有麦克风也能验证整条链路，也是手工验证的起点。

一条测试需要有**真实时长**的合成音频（约 60 秒），由脚本按确定性规则生成（交替静音段和正弦音段，模拟说话/停顿节奏），保证时序类断言可复现。

## 7. 依赖

`pyproject.toml` 新增：

```toml
[project.optional-dependencies]
realtime = ["onnxruntime", "kaldi-native-fbank", "websockets"]

[project.scripts]
mtd-realtime = "moss_transcribe_diarize.app.realtime_cli:main"
```

`websockets` 是必需的——现在 pyproject 里是裸 `uvicorn`，不带 WebSocket 支持。

**声纹模型权重**（首次启动自动下载，缓存到 `~/.cache/mtd-speaker/`）：

| 用途 | 文件 |
|---|---|
| 中文 | `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` |
| 英文 | `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` |

两者均约 28 MB，输出 192 维嵌入，输入 80 维 fbank。默认按 `--lang` 选择，`--speaker-model` 可覆盖。

来源：`github.com/k2-fsa/sherpa-onnx` 的 `speaker-recongition-models` release（tag 名原文如此）。下载必须**锁定到具体 release 资源并校验 SHA-256**，不能跟随浮动分支——声纹模型被替换会静默破坏说话人一致性。地址和哈希写进代码常量，`--speaker-model` 可覆盖为本地路径以支持离线部署。

## 8. 默认参数与调优

| 参数 | 默认 | 含义与影响 |
|---|---|---|
| `window` (W) | 20.0 s | 每次推理的音频长度。越长上下文越足、分离越准，但算力开销线性增长 |
| `hop` | 5.0 s | 推理触发间隔。越小越跟手，算力开销线性增长 |
| `tail` | 6.0 s | 窗口末尾多少秒判为"不稳定"。越大定稿越稳、延迟越高 |
| `min_first_window` | 8.0 s | 首次推理前至少要攒多少音频 |
| `silence_rms_db` | -45 dBFS | 静音判定阈值 |
| `silence_frame_ratio` | 0.05 | 高于阈值的帧占比低于此值即判静音 |
| `speaker_threshold` | 0.55 | 声纹余弦相似度匹配阈值 |
| `min_segment_sec` | 0.4 s | 短于此长度的段不做声纹，继承同组判定 |
| `buffer_capacity` | 180.0 s | 环形缓冲容量 |

**算力开销 = `W / hop` = 4 倍实时**（忽略静音门控的节省）。这是最需要实测的一项：在目标 GPU 上跑一次真实会议音频，看 vLLM 的 `rtf`。若 `rtf > 1`，调大 `hop` 或调小 `W`。

## 9. 安全与运维

- 默认绑定 `127.0.0.1`。服务**没有鉴权**，绑到 `0.0.0.0` 等于把会议录音的转写接口开放给同网段，README 和 CLI 都要显性警告。
- 音频默认落盘到 `runs/realtime/<session-id>/audio.wav`。会议录音是敏感数据，README 里要说明并提供 `--no-record` 开关（仅保留转写文本）。
- 会话目录复用现有 `runs/` 的约定，`.gitignore` 已覆盖。

## 10. 已考虑并否决的方案

- **跨窗口文本上下文条件**：把上一窗口的尾部文本拼进 prompt。可能提升连贯性，但让输出变成非确定性的、难以复现，也让测试无法写固定断言。v1 不做，留作后续实验。
- **内存直传音频**：省一次临时文件写入。收益是每窗口毫秒级，代价是要修改 `VllmRunner` 和 `ModelRunner`。不划算。
- **前端用 `MediaRecorder` 编码后传输**：省带宽，但拿到的是容器分片，需要服务端解封装，且编码延迟与丢帧行为不可控。裸 PCM 在本机/局域网场景完全够用。
- **依赖 SGLang 的 `verbose_json` 直接拿分段**：会绑定到单一后端。自己用 `TranscriptStreamParser` 解析，vLLM / SGLang / HF 三后端行为一致。
- **服务端 WASAPI loopback 采集**：用户选择了浏览器方案，跨平台且零额外依赖。

## 11. 待实测确认的风险

1. **`rtf` 是否 < 1**：在目标 GPU 上以 `W=20, hop=5` 跑真实会议音频。这是整个设计能否成立的前提。
2. **浏览器混音可行性**：`AudioContext({sampleRate:16000})` 配合 `getDisplayMedia` 音轨在目标浏览器（Chrome/Edge）上的实际行为，特别是系统声音音轨能否稳定接入 `MediaStreamAudioSourceNode`。
3. **声纹在真实会议音频上的表现**：远场短片段上的误合并率。若明显不可用，降级路径是 `--no-speaker`，并用"重跑本次会话"走文件模式的高质量分离作为补偿。
4. **vLLM 服务的并发能力**：多个会话同时跑时的表现。v1 假设单会话为主，服务端不做多会话调度优化。

## 12. 实现顺序

实现分三个阶段，每个阶段都能独立交付、独立验证。

**阶段一：核心实时管线**（纯 Python，无服务、无浏览器）

1. torch 隔离（上游改动）：`__init__.py` 惰性导入 + `DEFAULT_PROMPT` 提到 `prompts.py`。必须最先做，否则阶段一的所有测试和模块都还在拖 torch
2. `RealtimeConfig`
3. 纯逻辑层：`buffer` / `window`，配套测试
4. `stitch`：先做解析与时间换算，再做定稿水位线与去重
5. `speaker`：`SpeakerEmbedder` 协议 + `SpeakerGallery`（用假 embedder 跑通匹配逻辑）
6. `transcriber`：`WindowTranscriber` 协议 + `HfWindowTranscriber`
7. `store` 落盘
8. `session` 编排
9. 静音门控与降级

交付物：一个能吃 PCM、吐出定稿段的库，全部用假 transcriber / 假 embedder 测试，且导入时不碰 torch。

**阶段二：服务与协议**

10. ~~`OnnxCampplusEmbedder` + 模型下载与 SHA-256 校验~~ —— **已完成**（CAM++ ONNX，接上后两人会议的说话人标注从 2 段错变为 0 段错）
11. 抽出 `app/openai_audio_client.py`（上游改动，是 `VllmWindowTranscriber` 的前置：现有的 `vllm_runner` 会 import torch，不能被实时路径复用）
12. `VllmWindowTranscriber`；并把 token 预算改成**按实际窗口长度**推算（4.5 那两条接线要求，两个后端都要）
13. `realtime/export.py`：`CommittedSegment` → `SubtitleSegment` 的桥接，加上 `md`/`txt` 两个格式化函数（`subtitle` 包里没有这两种）
14. `realtime_server`：HTTP 路由 + WebSocket 端点 + 每会话的驱动任务
15. `realtime_cli` + `pyproject.toml` 入口（含 `websockets` 依赖）
16. `scripts/realtime_client.py` 冒烟客户端

交付物：一个能跑起来的 `mtd-realtime`，可以用脚本推音频并收到事件。

阶段一收尾时已顺带完成、不必再做：`SessionStore` 的 `session_id` 校验与路径穿越拒绝；`status` 事件的 `gated_windows` / `gated_sec` 可观测字段；静音门控的覆盖性保证与超预算窗口的记录。

**阶段三：浏览器前端**

15. `audio-worklet.js` + 采集与混音
16. `realtime.html` / `realtime.js` 渲染与交互
17. i18n 词条
18. README、真实 vLLM + 真实麦克风端到端实测与参数调优

交付物：可用的实时会议转写工具。
