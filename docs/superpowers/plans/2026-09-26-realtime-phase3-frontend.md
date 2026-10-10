# 实时会议转写 — 阶段三：浏览器前端 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把阶段二那个"能用脚本推音频"的服务变成能用的工具：浏览器里开麦（或共享系统声），边说边出字，定稿区与临时区两级显示，说话人可以改名与改归属，会话结束后能导出、能重跑、能翻历史。

**Architecture:** 服务端只多两条路由（静态资源、重跑），其余全靠阶段二的 WS 协议。前端是三个文件：`audio-worklet.js`（音频线程里攒 100 ms 帧）、`realtime.js`（采集/混音/传输/渲染/交互）、`realtime.html` + `realtime.css`（骨架与样式）。i18n 沿用 `static/i18n.js` 的 `t()` / `data-i18n` 机制，新增 `realtime.*` 词条。

**Tech Stack:** 浏览器原生 API（`getUserMedia` / `getDisplayMedia` / `AudioContext` / `AudioWorklet` / `WebSocket`），无框架、无构建步骤。服务端：FastAPI（阶段二已有）。

**Spec:** `docs/superpowers/specs/2026-09-24-realtime-meeting-transcription-design.md`（阶段三相关：§4.8 的 retranscribe 与静态资源、§5 全部、§9、§12）

**前序：** `docs/superpowers/plans/2026-09-25-realtime-phase2-service.md` 已交付 `mtd-realtime`。阶段三的协议就是 `scripts/realtime_client.py` 已经在说的那套。

## Global Constraints

- **服务端不得 import torch**（沿用阶段二）。阶段三新增的服务端代码只有 `realtime_server.py` 里两条路由，以及 CLI 里为 `--backend hf` 接上重跑用的 runner——那个 import 留在工厂内部。
- **前端不得引入构建步骤、框架或 CDN 依赖。** 三个文件直接 `import`/`<script type="module">`。理由：这个服务是"跑在本地、给人临时开个会用的"，没有人会为一个 700 行的页面配一套打包器。
- **`static/index.html` 与 `static/app.js` 是字幕工坊的**，本阶段一个字节都不动。字幕工坊的 i18n 测试会扫这两个文件，实时页面的词条进同一份 locale 目录，所以**两份 locale 的键必须严格一致**（现有测试已经在钉这条）。
- **`AudioContext` 的采样率**：优先 `{sampleRate: 16000}`；浏览器不支持时回退默认采样率，在 worklet 里线性插值重采样，并在界面上明确提示一次。
- **麦克风的 AEC / NS / AGC 三项必须关掉**：它们是为通话设计的，会把远场人声当噪声削掉，对转写是纯损失。
- 测试用 `unittest.TestCase` 风格，pytest 运行。`python` 一律指 `.venv/Scripts/python.exe`。
- 浏览器侧无法用 pytest 覆盖，所以**能被纯函数表达的判定都抽成纯函数**（事件合并、说话人配色、时间格式化、需要发送的帧切分、`bufferedAmount` 背压判定），服务端起一个真服务，用浏览器面板逐条走查并把结果记进报告。

## Review Focus

1. **背压丢帧必须可观测。** `ws.bufferedAmount > 2 MB` 时丢帧是设计（宁可丢音也不在浏览器里无限堆内存），但**丢了多少必须显示出来**——静默丢帧会被读成"模型漏听了"。
2. **自动滚动不能跟用户抢。** 定稿区自动滚到底；用户往上滚则暂停跟随，滚回底部再恢复。滚动位置用"距底部多少像素"而不是 `scrollTop`（`scrollTop` 在内容增长时会误判）。
3. **临时区是整体替换语义。** 每次收到 `provisional` 就整体换掉右栏内容，不做逐字动画——逐字动画在整体替换下会闪。
4. **改名与改归属是两条不同的路。** `rename_speaker` 只改显示名（说话人表），`reassign_segment` 改的是**某一段的归属**（写进 `transcript.jsonl`）。界面上必须是两个动作，且改名后**已经显示过的段落也要跟着变**——否则用户改了名字却只看到新段落变了。
5. **重跑失败要说清楚为什么。** `POST /api/sessions/{id}/retranscribe` 在没有可用重跑后端时必须是 501 加一句人话，而不是 500。

---

### Task 8: `POST /api/sessions/{id}/retranscribe`

**Files:**
- Modify: `moss_transcribe_diarize/app/realtime_server.py`
- Modify: `moss_transcribe_diarize/app/realtime_cli.py`
- Test: `tests/test_realtime_api.py`（追加）

**Interfaces:**
- Consumes: `SessionStore`（阶段二）、`openai_audio_client.transcribe_bytes`（阶段一 Task 1）
- Produces:
  - `create_realtime_app(..., retranscribe: Callable[[Path, str], str] | None = None)` —— 收到 `(audio_path, prompt)`，返回整份转写文本
  - `POST /api/sessions/{id}/retranscribe` → `{"session_id": ..., "text": ...}`；没有注入重跑后端时 501 `retranscribe_unavailable`；会话不存在 404；没有录音 404 `audio_missing`

**说明**：spec §4.8 的路线表里就有这条，§5.2 的底栏也有"重跑本次会话"。它走的是**文件模式**：把 `audio.wav` 整段交给当前后端跑一次。`--backend vllm` 下就是 `transcribe_bytes` 一次整文件请求（那个函数本来就是文件模式的）；`--backend hf` 下是 `ModelRunner.transcribe`。两者都用**构造时注入的 callable**，服务端因此仍然不知道后端是什么。

- [ ] **Step 1: 写失败测试**

在 `tests/test_realtime_api.py` 追加：

```python
class RetranscribeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        store = SessionStore(self.runs, "s1", name="周会")
        store.append_committed([_Row()])
        store.append_audio(np.zeros(16000, dtype=np.float32))
        store.finalize([])

    def _client(self, retranscribe=None):
        return TestClient(_app(Path(self._tmp.name), retranscribe=retranscribe))

    def test_runs_the_whole_recording_through_the_injected_backend(self):
        seen = []

        def fake(path, prompt):
            seen.append((Path(path).name, prompt))
            return "[0.5][S01]重跑的结果[3.0]"

        response = self._client(fake).post("/api/sessions/s1/retranscribe")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "[0.5][S01]重跑的结果[3.0]")
        self.assertEqual(seen, [("audio.wav", "")])

    def test_a_failing_backend_is_502_with_the_reason(self):
        def broken(path, prompt):
            raise RuntimeError("vLLM 没起来")

        response = self._client(broken).post("/api/sessions/s1/retranscribe")

        self.assertEqual(response.status_code, 502)
        self.assertIn("vLLM 没起来", response.json()["detail"])

    def test_without_a_backend_it_says_so_instead_of_crashing(self):
        response = self._client(None).post("/api/sessions/s1/retranscribe")

        self.assertEqual(response.status_code, 501)
        self.assertEqual(response.json()["code"], "retranscribe_unavailable")

    def test_unknown_session_is_404(self):
        self.assertEqual(self._client(lambda p, q: "x").post("/api/sessions/nope/retranscribe").status_code, 404)

    def test_a_session_without_a_recording_is_404(self):
        SessionStore(self.runs, "s2", name="没录音").finalize([])
        response = self._client(lambda p, q: "x").post("/api/sessions/s2/retranscribe")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "audio_missing")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_api.py -q -k Retranscribe`
Expected: FAIL —— `create_realtime_app() got an unexpected keyword argument 'retranscribe'`

- [ ] **Step 3: 实现**

`realtime_server.py`：签名加 `retranscribe=None`，并在 `export` 路由旁边加：

```python
    @app.post("/api/sessions/{session_id}/retranscribe")
    def retranscribe_session(session_id: str):
        if retranscribe is None:
            return error("retranscribe_unavailable",
                         "this deployment cannot re-run a session: no file-mode backend was provided", 501)
        meta = load_meta(session_id)
        if not meta:
            return error("session_not_found", f"no such session: {session_id}", 404)
        path = SessionStore.session_dir(runs, session_id) / "audio.wav"
        if not path.exists():
            return error("audio_missing", "this session has no recording", 404)
        try:
            text = retranscribe(path, str(meta.get("prompt") or ""))
        except Exception as exc:
            return error("retranscribe_failed", f"{type(exc).__name__}: {exc}", 502)
        return JSONResponse({"session_id": session_id, "text": text})
```

`realtime_cli.py`：加 `build_retranscriber(args)`，两个后端各一份，`main()` 里传进去。

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_api.py -q`
Expected: 全部 PASS

---

### Task 9: 实时页面的静态资源路由

**Files:**
- Modify: `moss_transcribe_diarize/app/realtime_server.py`
- Test: `tests/test_realtime_api.py`（追加）

**Interfaces:**
- Produces: `GET /assets/{path:path}` → `static_dir` 下的文件；越界 404；未知类型 `application/octet-stream`

**说明**：阶段二的 `create_realtime_app` 只服务 `/` 与 API，前端要用的 `.js` / `.css` / `.json` 没有出口。字幕工坊的 `server.py` 里有一份同样的实现，照它的路由形状与媒体类型表写，避免两个服务对同类文件给出不同的 `Content-Type`。

- [ ] **Step 1: 写失败测试**

```python
class StaticAssetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self.assets = Path(self._tmp.name) / "assets"
        self.assets.mkdir(parents=True)
        (self.assets / "realtime.js").write_text("export const x = 1;\n", encoding="utf-8")
        (self.assets / "realtime.css").write_text("body { margin: 0 }\n", encoding="utf-8")
        (self.assets / "audio-worklet.js").write_text("registerProcessor('x', class {});\n", encoding="utf-8")
        (self.assets / "realtime.html").write_text("<!doctype html><title>rt</title>\n", encoding="utf-8")
        (self._tmp.name and (Path(self._tmp.name) / "secret.txt")).write_text("nope", encoding="utf-8")
        self.client = TestClient(_app(Path(self._tmp.name), static_dir=self.assets))

    def test_serves_the_frontend_files_with_useful_content_types(self):
        for name, media in (("realtime.js", "text/javascript"),
                            ("realtime.css", "text/css"),
                            ("audio-worklet.js", "text/javascript")):
            with self.subTest(name=name):
                response = self.client.get(f"/assets/{name}")
                self.assertEqual(response.status_code, 200)
                self.assertIn(media, response.headers["content-type"])

    def test_serves_the_page_at_the_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("rt", response.text)

    def test_a_missing_asset_is_404(self):
        self.assertEqual(self.client.get("/assets/nope.js").status_code, 404)

    def test_a_traversal_attempt_cannot_read_outside_the_assets_dir(self):
        response = self.client.get("/assets/..%2F..%2Fsecret.txt")
        self.assertIn(response.status_code, (400, 404))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_api.py -q -k StaticAsset`
Expected: FAIL —— 404（`/assets/...` 没有路由）

- [ ] **Step 3: 实现**

`create_realtime_app` 加 `static_dir` 已存在（阶段二），补一条路由：

```python
    @app.get("/assets/{asset_path:path}")
    def asset(asset_path: str):
        target = (assets / asset_path).resolve()
        if not target.is_file() or assets.resolve() not in target.parents:
            return JSONResponse({"detail": "asset not found"}, status_code=404)
        return FileResponse(target, media_type=_MEDIA_TYPES.get(target.suffix.lower()))
```

模块级：

```python
_MEDIA_TYPES = {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".html": "text/html",
    ".ico": "image/x-icon",
}
```

（`assets.resolve() not in target.parents` 是路径穿越的守卫：`resolve()` 之后再比父目录，`..%2F..%2Fsecret.txt` 会被解析到 assets 之外。）

- [ ] **Step 4: 运行测试确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_realtime_api.py -q`
Expected: 全部 PASS

---

### Task 10: `audio-worklet.js` —— 采集与成帧

**Files:**
- Create: `moss_transcribe_diarize/app/static/audio-worklet.js`

**Interfaces（worklet 侧）：**
- `registerProcessor("mtd-capture", MtdCaptureProcessor)`
- options：`{ frameSamples: number, inputSampleRate: number }`；`inputSampleRate` 与 worklet 全局 `sampleRate` 不同时，在 worklet 里做线性插值重采样
- 每攒够 `frameSamples` 个 16 kHz 样本，`port.postMessage({ type: "frame", pcm: Float32Array })`

**为什么在 worklet 里而不是主线程**：主线程的渲染卡顿会直接变成丢音；worklet 跑在音频线程上，那里的调度是硬实时的。这正是 §5.1 第 5 条要求的。

- [ ] **Step 1: 对照 spec §5.1 逐条实现**（本任务无法用 pytest 覆盖，验收在 Task 14 的浏览器走查）

要点：
1. 128 帧的渲染量子累积，够 `frameSamples`（默认 1600 = 100 ms）就发一帧，**余数留下**（不是丢掉）。
2. 采样率不一致时线性插值到 16 kHz。位置游标跨量子保留，否则每个量子边界都会出现一次相位不连续。
3. 输入静音时也要照常发帧——静音门控在服务端，客户端不做这个判断（客户端不知道 `window` / `hop`）。
4. 单声道：`inputs[0]` 的每个声道直接相加再平均。

- [ ] **Step 2: 用浏览器面板确认 worklet 真的加载并成帧**（Task 14 一起做）

---

### Task 11: `realtime.js` —— 传输、渲染与交互

**Files:**
- Create: `moss_transcribe_diarize/app/static/realtime.js`

**Interfaces：**
- 纯函数（可被浏览器控制台直接调用验证，也是本任务唯一能"断言"的部分）：
  - `speakerColor(index: number) -> string` —— 固定调色板按 id 稳定分配
  - `formatClock(seconds: number) -> string` —— `mm:ss` / `hh:mm:ss`
  - `applyRename(segments, speakerId, name) -> segments` —— 改名后**已显示的段落**也要跟着变
  - `shouldFollow(container) -> boolean` —— 距底部 ≤ 24 px 才算"用户在跟随"
  - `mergeCommitted(segments, incoming) -> segments` —— 按 id 原地替换，否则追加（协议是增量追加语义，改归属时同一个 id 会再来一次）
- 主流程：`start()` 采集 → `ws.onmessage` 分派 → 渲染；`stop()` 收尾；`exportAs(fmt)`；`loadHistory()`；`retranscribe()`

- [ ] **Step 1: 实现纯函数**，并把它们挂在 `window.mtdRealtime` 上，供控制台与 Task 14 的走查直接调。
- [ ] **Step 2: 实现采集与传输**（spec §5.1 第 1–4、6 条）
  - `getUserMedia({audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}})`
  - `getDisplayMedia({audio: true, video: true})` 由按钮触发；拿到的 `video` track 立刻 `stop()`（只要声音）
  - 两路各接 `MediaStreamAudioSourceNode` → `GainNode`（各自的音量/静音）→ 汇总 `GainNode` → worklet
  - `ws.bufferedAmount > 2 MB` 时丢帧、计数、在顶栏显示
- [ ] **Step 3: 实现渲染**（spec §5.2）
  - 定稿区：时间 + 说话人（颜色 + 可改名）+ 文字；自动滚到底（`shouldFollow` 判定）；每段右侧一个"改说话人"下拉，可勾选"应用到本次会话里同一说话人的全部段"
  - 临时区：整体替换、降透明度、斜体
  - 顶栏：会话名、开始/暂停/停止、两路输入的电平与静音开关、`rtf` 与积压秒数、丢帧计数、后端信息
  - 底栏：导出五格、重跑本次会话、历史会话
- [ ] **Step 4: 服务端事件里 `speaker` 到达时要刷新所有段的显示名**（Review Focus 4）

---

### Task 12: `realtime.html` 与 `realtime.css`

**Files:**
- Create: `moss_transcribe_diarize/app/static/realtime.html`
- Create: `moss_transcribe_diarize/app/static/realtime.css`
- Modify: `moss_transcribe_diarize/app/static/index.html`（加一个指回实时页面的入口链接——**只加一个链接**）

**说明**：样式沿用 `styles.css` 的调色板（`--bg #f7f5f0`、`--teal #007d77`、`--panel`、`--line`、`--muted`），不引入第二套视觉语言。布局用 grid：顶栏 56 px + 两栏主体 + 底栏。

- [ ] **Step 1: 写 `realtime.html` 骨架**（两栏、顶栏、底栏；所有静态文案挂 `data-i18n`）
- [ ] **Step 2: 写 `realtime.css`**（含两栏在窄屏下变单栏）
- [ ] **Step 3: `GET /` 已经会返回 `realtime.html`**（阶段二实现），确认浏览器能打开

---

### Task 13: i18n 词条

**Files:**
- Modify: `moss_transcribe_diarize/app/static/locales/zh-CN.json`
- Modify: `moss_transcribe_diarize/app/static/locales/en.json`
- Test: `tests/test_app_i18n.py`（追加）

**Interfaces:** 键一律 `realtime.*` 前缀；错误文案 `errors.retranscribe_unavailable` / `errors.retranscribe_failed` / `errors.audio_missing`（与 Task 8 的 code 对齐）。

- [ ] **Step 1: 写失败测试**

```python
    def test_realtime_frontend_uses_only_known_keys(self):
        english = self.catalogs["en"]
        html = (STATIC_DIR / "realtime.html").read_text(encoding="utf-8")
        js = (STATIC_DIR / "realtime.js").read_text(encoding="utf-8")
        html_keys = set(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', html))
        js_keys = set(re.findall(r"\bt\('([^']+)'", js))
        self.assertFalse((html_keys | js_keys) - set(english))

    def test_every_realtime_string_exists_in_both_locales(self):
        for key in self.catalogs["en"]:
            if key.startswith("realtime.") or key.startswith("errors.retranscribe"):
                with self.subTest(key=key):
                    self.assertIn(key, self.catalogs["zh-CN"])
```

- [ ] **Step 2: 运行确认失败**；**Step 3: 补两份词条**；**Step 4: 运行确认通过**

Run: `.venv/Scripts/python.exe -m pytest tests/test_app_i18n.py -q`
Expected: 全部 PASS（**现有那三条也必须仍然 PASS**——两份 locale 的键必须严格一致）

---

### Task 14: 浏览器走查与 README

**Files:**
- Modify: `README.md`（新增"实时会议转写"一节）
- Modify: `README_zh.md`（同上）

**走查清单**（用浏览器面板，起真服务 + 假端点或真模型，逐条记录结果）：

1. 打开 `/`，页面渲染正常，切语言后静态文案跟着变。
2. 在控制台把这个会话喂进去（用 `window.mtdRealtime` 暴露的纯函数 + 一个 scripted 假 socket），确认：定稿区按序追加、临时区整体替换、改名后**旧段落也换名**、改归属后同一 id 原地更新。
3. 导出五格各下载一次，内容非空。
4. 历史会话列表能列出阶段二 E2E 造出来的那些会话。
5. 点"重跑本次会话"，拿到 `{"text": ...}`（用假端点）。
6. 断网/端点不可达时顶栏出现降级提示。

**README 必须写到的三件事**（spec §9）：默认只绑 `127.0.0.1` 且**无鉴权**；录音默认落盘 `runs/realtime/<id>/audio.wav`，是敏感数据，`--no-record` 可关；`--backend vllm` 要先起 vLLM 服务，忘了起的话 `/api/runtime` 会说。

---

## 阶段三完成标志

```bash
.venv/Scripts/python.exe -m pytest tests/ -q      # 全绿
```

加上 Task 14 那六条走查逐条符合预期，README 两节都写好。
