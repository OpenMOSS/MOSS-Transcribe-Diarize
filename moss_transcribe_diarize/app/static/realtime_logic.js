/**
 * 实时页面里"能被纯函数表达"的那部分判定。
 *
 * 用 `node --test tests/js/` 跑。之所以单独一个文件、且**不 import 任何 DOM 或 i18n**：
 * 这样它能在 Node 里直接测，而不用为了测一条判定去架一个浏览器。
 */

const SPEAKER_COLORS = [
  "#007d77", "#c94b35", "#2f7d4f", "#7b5ea7", "#a8721a", "#1f6fb2", "#a3336b", "#5a6b3b",
];

/** `mm:ss`，超过一小时才带上小时。 */
export function formatClock(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const pad = (value) => String(value).padStart(2, "0");
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${pad(minutes)}:${pad(secs)}`;
}

/** 说话人 id 的数字部分决定颜色，所以同一个 S02 在两次会话里颜色一致。 */
export function speakerIndex(speakerId) {
  const text = String(speakerId || "");
  const match = /^[A-Za-z]*(\d+)$/.exec(text);
  if (match) return Number(match[1]) - 1;
  let hash = 0;
  for (const ch of text) hash = (hash * 31 + ch.codePointAt(0)) % 997;
  return hash;
}

export function speakerColor(speakerId) {
  const index = Math.abs(Math.trunc(speakerIndex(speakerId))) % SPEAKER_COLORS.length;
  return SPEAKER_COLORS[index];
}

/**
 * `committed` 是**增量追加**语义，但改归属时同一个 id 会再来一次——那次要原地替换。
 */
export function mergeCommitted(segments, incoming) {
  const list = segments.slice();
  for (const segment of incoming || []) {
    const at = list.findIndex((item) => item.id === segment.id);
    if (at >= 0) list[at] = segment;
    else list.push(segment);
  }
  return list;
}

/**
 * 改名之后**已经显示过的段落也要跟着变**——否则用户改了名字却只看到新段落换了名。
 */
export function applyRename(segments, speakerId, name) {
  return (segments || []).map((segment) => (
    segment.speaker === speakerId ? { ...segment, speaker_name: name } : segment
  ));
}

/**
 * 滚动位置用"距底部多少像素"判断，不用 `scrollTop`：内容在增长时 `scrollTop` 不变而
 * 实际已经不在底部，自动滚动会跟用户抢。
 */
export function shouldFollow(container, threshold = 24) {
  if (!container) return true;
  const distance = container.scrollHeight - container.scrollTop - container.clientHeight;
  return distance <= threshold;
}

/** 定稿区里所有属于某个说话人的段落 id（"应用到同组全部段"用）。 */
export function segmentsForSpeaker(segments, speakerId) {
  return (segments || []).filter((segment) => segment.speaker === speakerId)
    .map((segment) => segment.id);
}

/**
 * 麦克风在送**数字静音**的判据。
 *
 * 系统层禁掉麦克风时，Chrome 常常**不是**报错，而是正常 resolve 一条全程为 0 的轨道；
 * 设备选错、耳机没插好也会给出同样的东西。这时 getUserMedia 是成功的、界面显示"已开麦"、
 * 一帧不少地往后端送——送的全是 0。所以判据只能看样本本身：数字静音是**恰好** 0，
 * 而任何真实麦克风（哪怕很安静的房间）都有自己的底噪，峰值不会掉到 1e-4 以下。
 *
 * 攒够 2 秒再下结论：刚点开始的那一瞬间本来就没有声音。
 */
export const MIC_SILENCE_PEAK = 1e-4;
export const MIC_SILENCE_FRAMES = 20;

/**
 * 转写区还是空的时候，**为什么**它空着。
 *
 * 这一条是这张页面最容易骗人的地方：它的空屏有一串完全不同的原因——还没开始、正在攒够
 * 第一个窗口（默认要 8 秒）、窗口跑过但还没攒出稳定内容、输入的音频一直是静音被门控全部
 * 跳过、后端连续失败。界面上它们长得一模一样：两边空白，状态写着"转写中"。用户因此只能
 * 得出"这东西不工作"。把原因算出来，界面才有话可说。
 *
 * 返回 `null`（有内容可看，不需要解释）或 `{ key, params }`；**返回的是词条键而不是译好的
 * 句子**，所以这个函数保持纯净、可单独测。
 */
export function pipelineHint({
  sending = false,
  hasStatus = false,
  committed = 0,
  provisional = 0,
  gatedWindows = 0,
  degraded = false,
  failures = 0,
  framesSent = 0,
  peakLevel = 0,
} = {}) {
  if (committed > 0 || provisional > 0) return null;
  if (!sending) return { key: "realtime.hint.idle", params: {} };
  if (degraded || failures > 0) return { key: "realtime.hint.degraded", params: { count: failures } };
  // 这一条排在"静音窗口"之前：门控跳过是**结果**，送的是静音才是原因。而且它比第一个窗口
  // 还早就能下结论——攒够 2 秒就够判断了，不必等 8 秒。
  if (framesSent >= MIC_SILENCE_FRAMES && peakLevel < MIC_SILENCE_PEAK) {
    return { key: "realtime.hint.micSilent", params: {} };
  }
  if (!hasStatus) return { key: "realtime.hint.warmingUp", params: {} };
  if (gatedWindows > 0) return { key: "realtime.hint.silentWindows", params: { count: gatedWindows } };
  return { key: "realtime.hint.waiting", params: {} };
}

/**
 * 一条 status 该不该挂告警、挂哪一条、顶栏那个积压数字该是什么颜色。
 *
 * 后端已经算好了 `overloaded`（落后超过 `3 * W`，spec §4.7）。它和 `degraded` 的区别是
 * **还在出内容、只是追不上**：所以它用提醒色而不是故障色——用户据此才知道该等着，还是
 * 该去看看后端。这不是空态提示（那个由 `pipelineHint` 负责）：算力吃紧时屏幕上往往是
 * 一堆旧内容，而不是空白。
 *
 * 两条同时成立时只报降级：后端连续失败时落后必然跟着涨，`overloaded` 只是它的结果，
 * 一起说出来会把真正要看的那条稀释掉。
 *
 * 返回 `null` 表示"没有告警"。调用方正是靠这个 `null` 把粘性提示收掉，所以健康时不能
 * 返回一个空对象。
 */
export function statusAlert({ degraded = false, overloaded = false } = {}) {
  if (degraded) return { key: "realtime.notice.degraded", tone: "bad" };
  if (overloaded) return { key: "realtime.notice.overloaded", tone: "warn" };
  return null;
}
