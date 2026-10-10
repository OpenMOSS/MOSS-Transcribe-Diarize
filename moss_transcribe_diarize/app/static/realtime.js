import {
  applyDocumentTranslations,
  getLocale,
  initI18n,
  localizedError,
  setLocale,
  t,
} from "./i18n.js";
import {
  applyRename,
  formatClock,
  mergeCommitted,
  pipelineHint,
  segmentsForSpeaker,
  shouldFollow,
  speakerColor,
  speakerIndex,
  statusAlert,
} from "./realtime_logic.js";

const WS_PATH = "/ws/realtime";
const FRAME_SAMPLES = 1600;          // 100 ms @ 16 kHz
const TARGET_RATE = 16000;
const BACKPRESSURE_BYTES = 2 * 1024 * 1024;

// 能被纯函数表达的判定都住在 realtime_logic.js 里，由 tests/js/ 用 node --test 覆盖；
// 这个文件只管 DOM 与网络。

// ---------------------------------------------------------------- 应用

const state = {
  socket: null,
  sessionId: null,
  committed: [],
  provisional: [],
  roster: new Map(),        // speaker_id -> 显示名
  follow: true,
  sending: false,
  paused: false,
  phase: "idle",
  applyToAll: false,
  droppedFrames: 0,
  clockTimer: null,
  startedAt: 0,
  audio: null,
  lastStatus: null,
  hasStatus: false,
  failures: 0,
  degraded: false,
  stickyNotice: false,
  framesSent: 0,
  peakLevel: 0,
  micInfo: null,
  micSilent: false,
};

const dom = {};

function cacheDom() {
  const ids = [
    "runtime", "localeSelect", "sessionName", "statePill", "sessionId",
    "startButton", "pauseButton", "stopButton",
    "micButton", "micMeter", "systemButton", "systemMeter", "micInfo",
    "statRtf", "statLag", "statGated", "statDropped", "statClock",
    "committedList", "committedEmpty", "provisionalList", "provisionalEmpty",
    "exportBar", "retranscribeButton", "historyButton", "historyPanel", "historyList",
    "historyClose", "retranscribePanel", "retranscribeText", "retranscribeClose",
    "notice", "backendInfo",
  ];
  for (const id of ids) dom[id] = document.getElementById(id);
}

/**
 * `sticky` 的消息不会被计时器收走。
 *
 * 默认 8 秒自动消失是为了不挡路，但**故障**不能这样：用户往往正看着别处，回来时告警
 * 已经没了，屏幕上只剩一片空白——那和"什么都没发生"没有区别。故障留到它自己好为止。
 */
function notice(key, params, { sticky = false } = {}) {
  if (!dom.notice) return;
  dom.notice.textContent = t(key, params);
  dom.notice.hidden = false;
  window.clearTimeout(notice.timer);
  state.stickyNotice = sticky;
  if (!sticky) notice.timer = window.setTimeout(() => { dom.notice.hidden = true; }, 8000);
}

function clearStickyNotice() {
  if (!state.stickyNotice) return;
  state.stickyNotice = false;
  window.clearTimeout(notice.timer);
  if (dom.notice) dom.notice.hidden = true;
}

const STATE_MESSAGES = {
  idle: "realtime.status.idle",
  connecting: "realtime.status.connecting",
  live: "realtime.status.live",
  stopped: "realtime.status.stopped",
};

/**
 * 状态标签只由**逻辑状态**渲染。
 *
 * 不能各处各自 `t(...)` 一句话塞进去：切语言时要把当前状态重新翻译一遍，而"当前是
 * 什么状态"与"那句话怎么翻译"是两件事——分开之后，切语言不会再覆盖掉会话的真实状态。
 */
function setPhase(phase) {
  state.phase = phase;
  if (!dom.statePill) return;
  dom.statePill.textContent = t(STATE_MESSAGES[phase] || STATE_MESSAGES.idle);
  dom.statePill.dataset.tone = phase === "live" ? "good" : "";
  // 会话还在录的时候重跑读到的是一份半成品录音，服务端会 409；按钮先禁掉，别让用户
  // 点出一条注定失败的请求。
  if (dom.retranscribeButton) dom.retranscribeButton.disabled = phase === "live";
}

// ---------------------------------------------------------------- 渲染

function speakerLabel(speakerId, fallback) {
  const name = state.roster.get(speakerId);
  return name || fallback || speakerId || t("realtime.speaker.unknown");
}

function renderCommitted() {
  if (!dom.committedList) return;
  const container = dom.committedList;
  const keepFollowing = state.follow;
  container.textContent = "";
  for (const segment of state.committed) {
    container.appendChild(committedRow(segment));
  }
  if (dom.committedEmpty) {
    dom.committedEmpty.hidden = state.committed.length > 0;
    // 空的时候要说清**为什么**空。这片空白有五种完全不同的原因（还没开始、正在攒够
    // 第一个窗口、窗口跑过但还没稳定内容、输入一直是静音被门控跳过、后端在连续失败），
    // 而它们过去长得一模一样：两边空白、状态写着"转写中"——用户只能得出"它不工作"。
    const hint = pipelineHint({
      sending: state.sending,
      hasStatus: state.hasStatus,
      committed: state.committed.length,
      provisional: state.provisional.length,
      gatedWindows: state.lastStatus ? state.lastStatus.gated_windows : 0,
      degraded: state.degraded,
      failures: state.failures,
      framesSent: state.framesSent,
      peakLevel: state.peakLevel,
    });
    dom.committedEmpty.textContent = hint
      ? t(hint.key, hint.params)
      : t("realtime.committed.empty");
  }
  if (keepFollowing) container.scrollTop = container.scrollHeight;
}

function committedRow(segment) {
  const row = document.createElement("article");
  row.className = "segment";
  row.dataset.segmentId = segment.id;
  row.dataset.speaker = segment.speaker || "";

  const time = document.createElement("span");
  time.className = "segment-time";
  time.textContent = formatClock(segment.start);
  row.appendChild(time);

  const speaker = document.createElement("span");
  speaker.className = "segment-speaker";
  speaker.style.setProperty("--speaker", speakerColor(segment.speaker));
  speaker.textContent = speakerLabel(segment.speaker, segment.speaker_name);
  if (segment.speaker_confident === false) speaker.classList.add("uncertain");
  row.appendChild(speaker);

  const text = document.createElement("span");
  text.className = "segment-text";
  text.textContent = segment.text || "";
  row.appendChild(text);

  if (state.roster.size > 1) {
    row.appendChild(speakerPicker(segment));
  }
  return row;
}

function speakerPicker(segment) {
  const wrap = document.createElement("span");
  wrap.className = "segment-actions";

  const select = document.createElement("select");
  select.className = "speaker-select";
  select.setAttribute("aria-label", t("realtime.speaker.change"));
  for (const [id, name] of state.roster) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = name;
    option.selected = id === segment.speaker;
    select.appendChild(option);
  }
  const applyAll = document.createElement("input");
  applyAll.type = "checkbox";
  applyAll.className = "apply-all";
  applyAll.checked = state.applyToAll;
  applyAll.setAttribute("aria-label", t("realtime.speaker.applyToAll"));
  applyAll.addEventListener("change", () => { state.applyToAll = applyAll.checked; });

  select.addEventListener("change", () => {
    const from = segment.speaker;
    const to = select.value;
    const ids = applyAll.checked ? segmentsForSpeaker(state.committed, from) : [segment.id];
    for (const id of ids) {
      send({ type: "reassign_segment", segment_id: id, speaker_id: to });
    }
  });
  wrap.appendChild(select);

  const label = document.createElement("label");
  label.className = "apply-all-label";
  label.title = t("realtime.speaker.applyToAll");
  label.append(applyAll, document.createTextNode(t("realtime.speaker.applyToAll")));
  wrap.appendChild(label);

  const rename = document.createElement("button");
  rename.type = "button";
  rename.className = "ghost small";
  rename.textContent = t("realtime.speaker.rename");
  rename.addEventListener("click", () => {
    const current = speakerLabel(segment.speaker, segment.speaker_name);
    const next = window.prompt(t("realtime.speaker.renamePrompt", { name: current }), current);
    if (next === null || !next.trim() || next.trim() === current) return;
    send({ type: "rename_speaker", speaker_id: segment.speaker, name: next.trim() });
  });
  wrap.appendChild(rename);
  return wrap;
}

function renderProvisional() {
  if (!dom.provisionalList) return;
  // 整体替换，不做逐字动画——整体替换下的逐字动画只会闪。
  dom.provisionalList.textContent = "";
  for (const segment of state.provisional) {
    const row = document.createElement("div");
    row.className = "provisional-row";
    const time = document.createElement("span");
    time.className = "segment-time";
    time.textContent = formatClock(segment.start);
    const speaker = document.createElement("span");
    speaker.className = "segment-speaker";
    speaker.style.setProperty("--speaker", speakerColor(segment.speaker));
    speaker.textContent = speakerLabel(segment.speaker, segment.speaker);
    const text = document.createElement("span");
    text.textContent = segment.text || "";
    row.append(time, speaker, text);
    dom.provisionalList.appendChild(row);
  }
  if (dom.provisionalEmpty) dom.provisionalEmpty.hidden = state.provisional.length > 0;
}

function renderRoster() {
  renderCommitted();
}
function renderStats() {
  const status = state.lastStatus;
  if (dom.statRtf) dom.statRtf.textContent = status ? status.rtf.toFixed(2) : "—";
  if (dom.statLag) dom.statLag.textContent = status ? `${status.lag_sec.toFixed(1)}s` : "—";
  if (dom.statGated) dom.statGated.textContent = status ? String(status.gated_windows) : "0";
  if (dom.statDropped) dom.statDropped.textContent = String(state.droppedFrames);
  // 积压数字的颜色分两档：后端失败是故障（红），算力跟不上只是提醒（琥珀）——
  // 两者都让这个数变红的话，用户分不出"要去看日志"还是"等它追上来"。
  const alert = statusAlert(status || {});
  if (dom.statLag && alert) dom.statLag.dataset.tone = alert.tone;
  else if (dom.statLag) delete dom.statLag.dataset.tone;
}

// ---------------------------------------------------------------- 协议

function handleEvent(event) {
  switch (event.type) {
    case "session":
      state.sessionId = event.session_id;
      if (dom.sessionId) dom.sessionId.textContent = event.session_id;
      dom.exportBar.hidden = false;
      setPhase("live");
      break;
    case "committed":
      state.committed = mergeCommitted(state.committed, event.segments);
      renderCommitted();
      break;
    case "provisional":
      state.provisional = event.segments || [];
      renderProvisional();
      break;
    case "speaker":
      state.roster = new Map((event.speakers || []).map((item) => [item.id, item.name || item.id]));
      // 连数据一起改写，不只是重画：`committed` 里的旧名字是定稿那一刻冻结的，而
      // 说话人表是**当下**的。只改视图的话，两份状态会一直不一致。
      for (const [id, name] of state.roster) {
        state.committed = applyRename(state.committed, id, name);
      }
      renderCommitted();
      break;
    case "status": {
      state.lastStatus = event;
      state.hasStatus = true;
      state.degraded = Boolean(event.degraded);
      if (!state.degraded) state.failures = 0;
      const alert = statusAlert(event);
      // 积压秒数一起报出去：只说"算力跟不上"用户没法判断是刚发生还是已经积了很久。
      if (alert) notice(alert.key, { lag: event.lag_sec }, { sticky: true });
      else clearStickyNotice();
      renderStats();
      renderCommitted();                 // 空态的解释要跟着门控计数一起更新
      break;
    }
    case "error":
      state.failures = Number(event.failures) || state.failures + 1;
      // 故障不留 8 秒就消失：用户回来时告警已经没了，屏幕上只剩空白。
      notice("realtime.notice.error", { detail: `${event.code}: ${event.detail}` },
             { sticky: true });
      renderCommitted();
      break;
    default:
      break;
  }
}

function send(payload) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  state.socket.send(JSON.stringify(payload));
}

function sendFrame(pcm) {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  // 实时工具宁可丢音也不能在浏览器里无限堆内存；但丢了多少必须显示出来，否则会被
  // 读成"模型漏听了"。
  if (state.socket.bufferedAmount > BACKPRESSURE_BYTES) {
    state.droppedFrames += 1;
    renderStats();
    return;
  }
  // 量的是**真正发出去的这批样本**的峰值。麦克风"拿到了但送的是数字静音"（系统层禁麦、
  // 选错设备）在 getUserMedia 那一步是完全成功的，只有看样本才能发现。取衰减最大值，
  // 界面上才是一条能看的电平，而不是每帧乱跳的数字。
  let peak = 0;
  for (let i = 0; i < pcm.length; i += 1) {
    const value = pcm[i] < 0 ? -pcm[i] : pcm[i];
    if (value > peak) peak = value;
  }
  state.framesSent += 1;
  state.peakLevel = Math.max(peak, state.peakLevel * 0.85);
  state.socket.send(pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength));
}

/** 顶栏那行麦克风信息：哪台设备、什么参数、以及**现在有没有在送声音**。 */
function renderMicInfo() {
  if (!dom.micInfo) return;
  const info = state.micInfo;
  if (!info) {
    dom.micInfo.textContent = "";
    return;
  }
  const seconds = (state.framesSent / 10).toFixed(0);
  const level = Math.min(100, Math.round(state.peakLevel * 400));
  dom.micInfo.textContent = t("realtime.mic.info", {
    label: info.label || t("realtime.mic.unknown"),
    rate: info.rate || "?",
    seconds,
    level,
  });
  const silent = state.framesSent >= 20 && state.peakLevel < 1e-4;
  dom.micInfo.dataset.silent = silent ? "yes" : "no";
  // "麦克风在送静音"这件事**在第一个窗口跑起来之前**就已经能看出来了，而那时候还没有
  // 任何服务端事件、也就没有任何东西会触发重画——空态提示会一直停在"正在攒第一个窗口"。
  // 所以状态一变就自己重画一次。
  if (silent !== state.micSilent) {
    state.micSilent = silent;
    renderCommitted();
  }
}

// ---------------------------------------------------------------- 音频采集

function killAudio() {
  if (state.audio) {
    for (const stop of state.audio.stops) {
      try { stop(); } catch (err) { /* 已经停了 */ }
    }
    // 轨道必须显式 stop：不 stop 的话浏览器标签页上的录音指示会一直亮着。
    for (const source of state.audio.sources.values()) {
      for (const track of source.stream.getTracks()) {
        try { track.stop(); } catch (err) { /* 已经停了 */ }
      }
    }
    state.audio.sources.clear();
    try { state.audio.context.close(); } catch (err) { /* 已经关了 */ }
  }
  state.audio = null;
}

async function ensureAudio() {
  if (state.audio) return state.audio;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    notice("realtime.notice.audioUnsupported");
    return null;
  }

  let context;
  let resampled = false;
  try {
    context = new AudioContextClass({ sampleRate: TARGET_RATE });
  } catch (err) {
    context = new AudioContextClass();
    resampled = true;
  }
  if (context.sampleRate !== TARGET_RATE) resampled = true;
  if (resampled) notice("realtime.notice.resampled", { rate: context.sampleRate });
  // 浏览器的自动播放策略会让新建的 AudioContext 停在 suspended，那样音频线程不渲染、
  // 一帧都出不来。这个调用点就在"开始"按钮的点击里，所以 resume 一定被允许。
  if (context.state === "suspended") {
    try { await context.resume(); } catch (err) { /* 没有用户手势时会被拒，让用户再点一次 */ }
  }

  await context.audioWorklet.addModule("/assets/audio-worklet.js?in=" + context.sampleRate);
  const node = new AudioWorkletNode(context, "mtd-capture", {
    numberOfInputs: 1,
    numberOfOutputs: 0,
    processorOptions: {
      frameSamples: FRAME_SAMPLES,
      inputSampleRate: context.sampleRate,
      outputSampleRate: TARGET_RATE,
    },
  });
  node.port.onmessage = (message) => {
    if (state.paused) return;
    if (message.data && message.data.type === "frame") sendFrame(message.data.pcm);
  };

  const master = context.createGain();
  master.gain.value = 1;
  master.connect(node);

  state.audio = { context, node, master, sources: new Map(), stops: [], resampled };
  state.audio.stops.push(() => node.port.close(), () => master.disconnect());
  return state.audio;
}

function attachMeter(track, meterElement) {
  if (!meterElement || !state.audio) return;
  const analyser = state.audio.context.createAnalyser();
  analyser.fftSize = 512;
  const source = state.audio.context.createMediaStreamSource(new MediaStream([track]));
  source.connect(analyser);
  const buffer = new Uint8Array(analyser.frequencyBinCount);
  const tick = () => {
    if (!state.audio || !meterElement.isConnected) return;
    analyser.getByteTimeDomainData(buffer);
    let peak = 0;
    for (const value of buffer) peak = Math.max(peak, Math.abs(value - 128) / 128);
    meterElement.style.setProperty("--level", String(Math.min(1, peak * 1.6)));
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  state.audio.stops.push(() => source.disconnect(), () => analyser.disconnect());
}

async function enableMicrophone() {
  const audio = await ensureAudio();
  if (!audio || audio.sources.has("mic")) return;
  let stream;
  try {
    // 这三项必须关掉：AEC/NS/AGC 是为通话设计的，会把远场人声当噪声削掉。
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (err) {
    // 把浏览器的原始原因带上：过约束（某台设备的 AEC/NS/AGC 开关组合它不支持）、设备被
    // 别的程序独占、权限被拒——都不是同一件事，只报"拿不到权限"会把人引到错的方向。
    notice("realtime.notice.noMic", { detail: `${err.name || "Error"}: ${err.message || err}` });
    return;
  }
  const source = audio.context.createMediaStreamSource(stream);
  const gain = audio.context.createGain();
  gain.gain.value = 1;
  source.connect(gain).connect(audio.master);
  audio.sources.set("mic", { stream, source, gain });
  const track = stream.getAudioTracks()[0];
  const settings = (track && track.getSettings) ? track.getSettings() : {};
  state.micInfo = {
    label: track ? track.label : "",
    rate: settings.sampleRate || audio.context.sampleRate,
  };
  renderMicInfo();
  attachMeter(track, dom.micMeter);
  dom.micButton?.classList.add("active");
}

async function enableSystemAudio() {
  const audio = await ensureAudio();
  if (!audio || audio.sources.has("system")) return;
  let stream;
  try {
    // 浏览器要求用户手势；调用点就在按钮的 click 里。
    stream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
  } catch (err) {
    return;
  }
  // 只要声音：视频轨立刻停掉，否则浏览器会一直显示"正在共享屏幕"。
  for (const track of stream.getVideoTracks()) track.stop();
  if (!stream.getAudioTracks().length) {
    notice("realtime.notice.shareHint");
    return;
  }
  const source = audio.context.createMediaStreamSource(stream);
  const gain = audio.context.createGain();
  gain.gain.value = 1;
  source.connect(gain).connect(audio.master);
  audio.sources.set("system", { stream, source, gain });
  attachMeter(stream.getAudioTracks()[0], dom.systemMeter);
  dom.systemButton?.classList.add("active");
  // 用户点浏览器的"停止共享"时，把这一路一并收掉——否则界面还显示着它在采集。
  for (const track of stream.getAudioTracks()) {
    track.addEventListener("ended", () => {
      try { source.disconnect(); gain.disconnect(); } catch (err) { /* 已经断开 */ }
      audio.sources.delete("system");
      dom.systemButton?.classList.remove("active", "muted");
    });
  }
}

function toggleSource(kind) {
  const source = state.audio && state.audio.sources.get(kind);
  if (!source) return;
  source.gain.gain.value = source.gain.gain.value > 0 ? 0 : 1;
  const button = kind === "mic" ? dom.micButton : dom.systemButton;
  button?.classList.toggle("muted", source.gain.gain.value === 0);
}

// ---------------------------------------------------------------- 会话控制

function openSocket() {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${scheme}//${window.location.host}${WS_PATH}`);
  socket.binaryType = "arraybuffer";
  socket.addEventListener("open", () => setPhase("connecting"));
  socket.addEventListener("message", (message) => {
    if (typeof message.data !== "string") return;
    try {
      handleEvent(JSON.parse(message.data));
    } catch (err) {
      notice("realtime.notice.badEvent", { detail: String(err) });
    }
  });
  socket.addEventListener("close", () => {
    state.socket = null;
    state.sending = false;
    state.paused = false;
    window.clearInterval(state.clockTimer);       // 否则"已录"在掉线后一直涨
    state.clockTimer = null;
    setPhase("stopped");
    dom.startButton.disabled = false;
    dom.pauseButton.disabled = true;
    dom.stopButton.disabled = true;
    dom.pauseButton.textContent = t("realtime.controls.pause");
  });
  socket.addEventListener("error", () => notice("realtime.notice.wsClosed"));
  return socket;
}

async function start() {
  // 这道守卫必须在**任何 await 之前**立起来。放在后面的话，浏览器弹麦克风授权框期间
  // 用户再点一次"开始"就能穿过去：第二条 WebSocket 被建出来，`state.socket` 指向新的
  // 那条、旧的那条再没有代码能引用到它——服务端那条会话永远收不到 stop，麦克风也被接进
  // 混音两遍。
  if (state.sending) return;
  state.sending = true;
  dom.startButton.disabled = true;

  state.committed = [];
  state.provisional = [];
  state.roster = new Map();
  state.droppedFrames = 0;
  state.lastStatus = null;
  state.hasStatus = false;
  state.failures = 0;
  state.degraded = false;
  state.framesSent = 0;
  state.peakLevel = 0;
  state.micSilent = false;
  state.follow = true;
  clearStickyNotice();
  renderCommitted();
  renderProvisional();
  renderStats();

  await enableMicrophone();
  if (!state.audio || !state.audio.sources.size) {
    notice("realtime.notice.needSource");
    state.sending = false;
    dom.startButton.disabled = false;
    return;
  }
  state.socket = openSocket();
  state.startedAt = Date.now();
  const name = (dom.sessionName?.value || "").trim();
  state.socket.addEventListener("open", () => {
    send({ type: "start", session_name: name });
    dom.pauseButton.disabled = false;
    dom.stopButton.disabled = false;
  });
  state.clockTimer = window.setInterval(() => {
    if (dom.statClock) {
      dom.statClock.textContent = formatClock((Date.now() - state.startedAt) / 1000);
    }
    renderMicInfo();          // 电平要跟着走，否则"麦克风在不在送声音"看不出来
  }, 500);
}

async function stop() {
  if (!state.socket) return;
  send({ type: "stop" });
  // 收尾那几段在 stop **之后**才由服务端发出（它还要跑完一个完整窗口），所以监听不动，
  // 等服务端关连接——连接关闭时会复位状态与时钟。
  state.sending = false;
  state.paused = false;
  window.clearInterval(state.clockTimer);        // 连接关闭时还会再清一次，幂等
  state.clockTimer = null;
  // 把麦克风真的放掉：只是不再发帧的话，浏览器标签页上的录音指示会一直亮着，而这是个
  // 记录会议的工具——"停止"应当确实停止采集。
  killAudio();
  dom.micButton?.classList.remove("active", "muted");
  dom.systemButton?.classList.remove("active", "muted");
  dom.pauseButton.textContent = t("realtime.controls.pause");
}

function togglePause() {
  if (!state.audio) return;
  state.paused = !state.paused;
  if (state.paused) state.audio.context.suspend();
  else state.audio.context.resume();
  dom.pauseButton.textContent = state.paused
    ? t("realtime.controls.resume")
    : t("realtime.controls.pause");
}

// ---------------------------------------------------------------- 导出 / 历史 / 重跑

async function exportAs(format) {
  if (!state.sessionId) return;
  const response = await fetch(`/api/sessions/${state.sessionId}/export?format=${format}`);
  if (!response.ok) {
    notice("realtime.notice.exportFailed", { detail: await response.text() });
    return;
  }
  const payload = await response.json();
  const blob = new Blob([payload.text], { type: "text/plain;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${state.sessionId}.${format}`;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function loadHistory() {
  const response = await fetch("/api/sessions");
  const payload = await response.json();
  dom.historyList.textContent = "";
  const sessions = payload.sessions || [];
  if (!sessions.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = t("realtime.history.empty");
    dom.historyList.appendChild(empty);
    return;
  }
  for (const session of sessions) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "history-row ghost";
    row.dataset.sessionId = session.session_id;
    const when = session.started_at ? new Date(session.started_at * 1000).toLocaleString() : "";
    row.textContent = [session.name || session.session_id, when,
                       t("realtime.history.speakers", { count: (session.speakers || []).length })]
      .filter(Boolean).join(" · ");
    row.addEventListener("click", () => loadSession(session.session_id));
    dom.historyList.appendChild(row);
  }
}

async function loadSession(sessionId) {
  const response = await fetch(`/api/sessions/${sessionId}`);
  if (!response.ok) return;
  const payload = await response.json();
  state.sessionId = sessionId;
  state.roster = new Map(
    (payload.session.speakers || []).map((item) => [item.id, item.name || item.id]),
  );
  state.committed = payload.segments || [];
  state.provisional = [];
  renderRoster();
  renderProvisional();
  dom.sessionId.textContent = sessionId;
  dom.exportBar.hidden = false;
  dom.historyPanel.hidden = true;
}

async function retranscribe() {
  if (!state.sessionId) return;
  dom.retranscribePanel.hidden = false;
  dom.retranscribeText.textContent = t("realtime.retranscribe.running");
  let response;
  try {
    response = await fetch(`/api/sessions/${state.sessionId}/retranscribe`, { method: "POST" });
  } catch (err) {
    // 网络断了 / 服务重启：不接住的话面板会永远停在"正在重跑…"。
    dom.retranscribeText.textContent = t("realtime.notice.error", { detail: String(err) });
    return;
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const friendly = localizedError(payload, "errors.retranscribe_failed");
    // 词汇表只给一句话，而**为什么**失败（模型路径、显存、录音坏了）全在 detail 里。
    // 只说"重跑失败。"等于把用户唯一的线索丢掉。
    dom.retranscribeText.textContent = payload.detail
      ? `${friendly}\n\n${payload.detail}`
      : friendly;
    return;
  }
  dom.retranscribeText.textContent = payload.text || t("realtime.retranscribe.empty");
}

// ---------------------------------------------------------------- 装配

async function loadRuntime() {
  try {
    const payload = await (await fetch("/api/runtime")).json();
    if (dom.runtime) {
      dom.runtime.textContent = t("realtime.runtime.available");
      dom.runtime.dataset.tone = "good";
    }
    if (dom.backendInfo) {
      const backend = payload.backend || {};
      const reachable = backend.reachable;
      dom.backendInfo.textContent = [
        `window ${payload.config.window}s`,
        `hop ${payload.config.hop}s`,
        `tail ${payload.config.tail}s`,
        payload.speaker && payload.speaker.enabled
          ? t("realtime.runtime.speakerOn")
          : t("realtime.runtime.speakerOff"),
      ].join(" · ");
      if (reachable === false) notice("realtime.notice.backendDown", { detail: backend.detail || "" });
    }
  } catch (err) {
    if (dom.runtime) {
      dom.runtime.textContent = t("realtime.runtime.unavailable");
      dom.runtime.dataset.tone = "bad";
    }
  }
}

function wire() {
  dom.startButton?.addEventListener("click", start);
  dom.stopButton?.addEventListener("click", stop);
  dom.pauseButton?.addEventListener("click", togglePause);
  dom.micButton?.addEventListener("click", () => (
    state.audio && state.audio.sources.has("mic") ? toggleSource("mic") : enableMicrophone()
  ));
  dom.systemButton?.addEventListener("click", () => (
    state.audio && state.audio.sources.has("system") ? toggleSource("system") : enableSystemAudio()
  ));
  dom.committedList?.addEventListener("scroll", () => {
    state.follow = shouldFollow(dom.committedList);
  });
  dom.retranscribeButton?.addEventListener("click", retranscribe);
  dom.historyButton?.addEventListener("click", () => {
    dom.historyPanel.hidden = !dom.historyPanel.hidden;
    if (!dom.historyPanel.hidden) loadHistory();
  });
  dom.historyClose?.addEventListener("click", () => { dom.historyPanel.hidden = true; });
  dom.retranscribeClose?.addEventListener("click", () => { dom.retranscribePanel.hidden = true; });
  for (const button of document.querySelectorAll("[data-export]")) {
    button.addEventListener("click", () => exportAs(button.dataset.export));
  }
  dom.localeSelect?.addEventListener("change", async (event) => {
    await setLocale(event.target.value);
    renderCommitted();
    renderProvisional();
    setPhase(state.phase);
  });
}

async function main() {
  cacheDom();
  await initI18n();
  applyDocumentTranslations();
  if (dom.localeSelect) dom.localeSelect.value = getLocale();
  wire();
  await loadRuntime();
  setPhase("idle");
  renderCommitted();          // 开机就说清"还没有开始"，而不是留一句无信息量的占位
  dom.pauseButton.disabled = true;
  dom.stopButton.disabled = true;
  window.addEventListener("beforeunload", killAudio);
}

// 走查与调试入口：把事件直接喂给渲染层，不需要麦克风也能验界面。
window.mtdRealtime = {
  feed(events) {
    for (const event of events) handleEvent(event);
  },
  reloadRuntime: loadRuntime,
  helpers: { formatClock, speakerColor, speakerIndex, mergeCommitted, applyRename,
             shouldFollow, segmentsForSpeaker, pipelineHint },
  state: () => ({
    sessionId: state.sessionId,
    committed: state.committed.length,
    provisional: state.provisional.length,
    roster: Object.fromEntries(state.roster),
    droppedFrames: state.droppedFrames,
  }),
};

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", main);
else main();
