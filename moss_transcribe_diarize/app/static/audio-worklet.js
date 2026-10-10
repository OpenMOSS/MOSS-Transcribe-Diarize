/**
 * 采集侧的成帧器，跑在**音频线程**上。
 *
 * 主线程的渲染卡顿会直接变成丢音；音频线程的调度是硬实时的，所以攒帧这件事必须在
 * 这里做。每攒够 frameSamples 个 16 kHz 样本就 postMessage 一帧出去，主线程只负责
 * 把它推进 WebSocket。
 *
 * 采样率：优先让 AudioContext 直接跑 16 kHz，省掉自己重采样。浏览器不支持时回退到
 * 默认采样率，在这里做线性插值。插值游标必须**跨量子保留**——每个量子重置一次的话，
 * 128 帧的边界上会出现一次相位跳变，听不出来但会污染转写。
 */

const DEFAULT_FRAME_SAMPLES = 1600; // 100 ms @ 16 kHz

class MtdCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.frameSamples = Math.max(1, Math.trunc(opts.frameSamples) || DEFAULT_FRAME_SAMPLES);
    this.outputRate = Math.trunc(opts.outputSampleRate) || 16000;
    this.ratio = (options && options.inputSampleRate ? options.inputSampleRate : sampleRate) / this.outputRate;

    this.frame = new Float32Array(this.frameSamples);
    this.filled = 0;
    this.pending = new Float32Array(0);
    this.phase = 0;
  }

  /** 把若干声道的这一量子混成单声道。 */
  toMono(input) {
    const channels = input.length;
    const frames = channels > 0 ? input[0].length : 0;
    if (channels === 0 || frames === 0) return null;
    if (channels === 1) return input[0];
    const mono = new Float32Array(frames);
    for (let channel = 0; channel < channels; channel += 1) {
      const data = input[channel];
      for (let i = 0; i < frames; i += 1) mono[i] += data[i];
    }
    for (let i = 0; i < frames; i += 1) mono[i] /= channels;
    return mono;
  }

  process(inputs) {
    const mono = this.toMono(inputs[0] || []);
    // 输入还没接上（或这一量子是空的）时照常返回 true：worklet 得活着等下一个人说话。
    if (!mono) return true;

    const merged = new Float32Array(this.pending.length + mono.length);
    merged.set(this.pending, 0);
    merged.set(mono, this.pending.length);

    let pos = this.phase;
    while (pos + 1 < merged.length) {
      const index = Math.floor(pos);
      const frac = pos - index;
      this.frame[this.filled] = merged[index] * (1 - frac) + merged[index + 1] * frac;
      this.filled += 1;
      if (this.filled === this.frameSamples) {
        // slice() 出一份拷贝：这块缓冲会被复用，postMessage 是转移引用而不是深拷贝。
        this.port.postMessage({ type: "frame", pcm: this.frame.slice() });
        this.filled = 0;
      }
      pos += this.ratio;
    }

    const consumed = Math.floor(pos);
    this.pending = merged.slice(Math.min(consumed, merged.length));
    this.phase = pos - consumed;
    return true;
  }
}

registerProcessor("mtd-capture", MtdCaptureProcessor);
