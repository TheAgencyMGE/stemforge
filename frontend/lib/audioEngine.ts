import type { Note } from "./types";

/**
 * One AudioContext drives everything. All stem sources are started with the same `when`
 * and offset, so they stay sample-locked; the MIDI preview synth is scheduled against the
 * same clock with a short lookahead.
 */

export interface LaneMix {
  volume: number; // 0..1.5 linear
  muted: boolean;
  soloed: boolean;
  audio: boolean; // monitor separated audio
  midi: boolean; // monitor MIDI preview synth
}

interface Lane {
  buffer: AudioBuffer | null;
  audioGain: GainNode;
  midiGain: GainNode;
  mix: LaneMix;
  isDrum: boolean;
  isOriginal: boolean;
}

type NotesProvider = (laneId: string) => Note[] | null;

const LOOKAHEAD = 0.25;
const TICK_MS = 40;

export class AudioEngine {
  ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private lanes = new Map<string, Lane>();
  private sources: AudioBufferSourceNode[] = [];
  private playing = false;
  private startCtxTime = 0;
  private startOffset = 0;
  private pausedAt = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private scheduledUntil = 0; // timeline seconds
  private voices = new Set<AudioScheduledSourceNode>();
  private noise: AudioBuffer | null = null;
  duration = 0;
  loop: { start: number; end: number } | null = null;
  compare: "stems" | "original" = "stems";
  notesProvider: NotesProvider = () => null;
  onEnded: (() => void) | null = null;

  configure(duration: number, notesProvider: NotesProvider) {
    this.duration = duration;
    this.notesProvider = notesProvider;
  }

  ensureContext(): AudioContext {
    if (!this.ctx) {
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new Ctx({ latencyHint: "interactive" });
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.9;
      const comp = this.ctx.createDynamicsCompressor(); // safety limiter when many lanes sum
      comp.threshold.value = -3;
      comp.ratio.value = 12;
      this.master.connect(comp).connect(this.ctx.destination);
      const len = this.ctx.sampleRate;
      this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const d = this.noise.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    }
    return this.ctx;
  }

  addLane(id: string, opts: { isDrum?: boolean; isOriginal?: boolean; mix: LaneMix }) {
    const ctx = this.ensureContext();
    if (this.lanes.has(id)) return;
    const audioGain = ctx.createGain();
    const midiGain = ctx.createGain();
    audioGain.connect(this.master!);
    midiGain.connect(this.master!);
    this.lanes.set(id, { buffer: null, audioGain, midiGain, mix: opts.mix, isDrum: !!opts.isDrum, isOriginal: !!opts.isOriginal });
    this.applyGains();
  }

  async loadLane(id: string, url: string, signal?: AbortSignal): Promise<AudioBuffer> {
    const ctx = this.ensureContext();
    const res = await fetch(url, { signal });
    if (!res.ok) throw new Error(`could not load audio (${res.status})`);
    const data = await res.arrayBuffer();
    const buffer = await ctx.decodeAudioData(data);
    const lane = this.lanes.get(id);
    if (lane) lane.buffer = buffer;
    this.duration = Math.max(this.duration, buffer.duration);
    if (this.playing) this.restartAt(this.currentTime);
    return buffer;
  }

  setMix(id: string, mix: Partial<LaneMix>) {
    const lane = this.lanes.get(id);
    if (!lane) return;
    lane.mix = { ...lane.mix, ...mix };
    this.applyGains();
  }

  setCompare(mode: "stems" | "original") {
    this.compare = mode;
    this.applyGains();
  }

  private audible(lane: Lane): boolean {
    if (this.compare === "original") return lane.isOriginal;
    const anySolo = [...this.lanes.values()].some((l) => l.mix.soloed);
    if (anySolo) return lane.mix.soloed;
    return !lane.mix.muted;
  }

  private applyGains() {
    const ctx = this.ctx;
    if (!ctx) return;
    const t = ctx.currentTime;
    for (const lane of this.lanes.values()) {
      const on = this.audible(lane);
      const a = on && lane.mix.audio ? lane.mix.volume : 0;
      const m = on && lane.mix.midi && this.compare !== "original" ? lane.mix.volume * 0.8 : 0;
      lane.audioGain.gain.setTargetAtTime(a, t, 0.01);
      lane.midiGain.gain.setTargetAtTime(m, t, 0.01);
    }
  }

  get isPlaying() {
    return this.playing;
  }

  get currentTime(): number {
    if (!this.playing || !this.ctx) return this.pausedAt;
    const raw = this.startOffset + (this.ctx.currentTime - this.startCtxTime);
    if (this.loop && raw >= this.loop.end) {
      const len = this.loop.end - this.loop.start;
      return this.loop.start + ((raw - this.loop.start) % len);
    }
    return Math.min(raw, this.duration);
  }

  async play(from?: number) {
    const ctx = this.ensureContext();
    if (ctx.state === "suspended") await ctx.resume();
    this.restartAt(from ?? this.pausedAt);
  }

  pause() {
    if (!this.playing) return;
    this.pausedAt = this.currentTime;
    this.stopSources();
    this.playing = false;
  }

  stop() {
    this.pause();
    this.pausedAt = this.loop ? this.loop.start : 0;
  }

  seek(t: number) {
    const clamped = Math.max(0, Math.min(t, this.duration || t));
    if (this.playing) this.restartAt(clamped);
    else this.pausedAt = clamped;
  }

  setLoop(loop: { start: number; end: number } | null) {
    this.loop = loop && loop.end - loop.start > 0.05 ? loop : null;
    if (this.playing) this.restartAt(this.currentTime);
  }

  private restartAt(offset: number) {
    const ctx = this.ensureContext();
    this.stopSources();
    if (this.loop && (offset < this.loop.start || offset >= this.loop.end)) offset = this.loop.start;
    if (offset >= this.duration) offset = 0;
    const when = ctx.currentTime + 0.05;
    for (const lane of this.lanes.values()) {
      if (!lane.buffer) continue;
      const src = ctx.createBufferSource();
      src.buffer = lane.buffer;
      if (this.loop) {
        src.loop = true;
        src.loopStart = this.loop.start;
        src.loopEnd = this.loop.end;
      }
      src.connect(lane.audioGain);
      src.start(when, offset);
      this.sources.push(src);
    }
    this.startCtxTime = when;
    this.startOffset = offset;
    this.playing = true;
    this.scheduledUntil = offset;
    this.timer = setInterval(() => this.tick(), TICK_MS);
    this.tick();
  }

  private stopSources() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    for (const s of this.sources) {
      try {
        s.stop();
      } catch {
        /* already stopped */
      }
      s.disconnect();
    }
    this.sources = [];
    for (const v of this.voices) {
      try {
        v.stop();
      } catch {
        /* ignore */
      }
    }
    this.voices.clear();
  }

  /** AudioContext time at which unwrapped playback position `raw` is heard. */
  private ctxTimeForRaw(raw: number): number {
    return this.startCtxTime + (raw - this.startOffset);
  }

  private tick() {
    if (!this.playing || !this.ctx) return;
    const now = this.currentTime;
    if (!this.loop && now >= this.duration - 0.01 && this.duration > 0) {
      this.pause();
      this.pausedAt = 0;
      this.onEnded?.();
      return;
    }
    // Schedule MIDI preview notes in (scheduledUntil, now + LOOKAHEAD].
    const elapsed = this.ctx.currentTime - this.startCtxTime;
    const rawNow = this.startOffset + elapsed;
    const rawUntil = rawNow + LOOKAHEAD;
    let from = this.scheduledUntil;
    if (from < rawNow) from = rawNow;
    if (rawUntil <= from) return;
    for (const [id, lane] of this.lanes) {
      if (!lane.mix.midi || lane.isOriginal) continue;
      const notes = this.notesProvider(id);
      if (!notes) continue;
      this.scheduleWindow(lane, notes, from, rawUntil);
    }
    this.scheduledUntil = rawUntil;
  }

  /**
   * Schedule notes whose onsets fall into the unwrapped interval [a, b). With a loop, pass k >= 1
   * covers raw times [end + (k-1)*len, end + k*len) and maps to timeline t = raw - k*len.
   */
  private scheduleWindow(lane: Lane, notes: Note[], a: number, b: number) {
    const ctx = this.ctx!;
    const segments: { lo: number; hi: number; shift: number }[] = [];
    if (this.loop) {
      const { start, end } = this.loop;
      const len = end - start;
      let r = a;
      while (r < b) {
        const k = r < end ? 0 : Math.floor((r - end) / len) + 1;
        const hi = Math.min(b, end + k * len);
        segments.push({ lo: r - k * len, hi: hi - k * len, shift: k * len });
        r = hi;
      }
    } else {
      segments.push({ lo: a, hi: b, shift: 0 });
    }
    for (const seg of segments) {
      for (const n of notes) {
        if (n.start < seg.lo || n.start >= seg.hi) continue;
        const end = this.loop ? Math.min(n.end, this.loop.end) : n.end;
        const t0 = this.ctxTimeForRaw(n.start + seg.shift);
        const t1 = this.ctxTimeForRaw(end + seg.shift);
        if (t0 < ctx.currentTime - 0.01) continue;
        if (lane.isDrum) this.drumVoice(lane.midiGain, n, t0);
        else this.toneVoice(lane.midiGain, n, t0, Math.max(t0 + 0.03, t1));
      }
    }
  }

  private track(node: AudioScheduledSourceNode) {
    this.voices.add(node);
    node.onended = () => this.voices.delete(node);
  }

  private toneVoice(out: AudioNode, n: Note, t0: number, t1: number) {
    const ctx = this.ctx!;
    const freq = 440 * Math.pow(2, (n.pitch - 69) / 12);
    const osc = ctx.createOscillator();
    osc.type = n.pitch < 52 ? "sawtooth" : "triangle";
    osc.frequency.setValueAtTime(freq, t0);
    if (n.bends?.length) {
      for (const [dt, semis] of n.bends) osc.detune.setValueAtTime(semis * 100, t0 + dt);
    }
    const filt = ctx.createBiquadFilter();
    filt.type = "lowpass";
    filt.frequency.value = Math.min(8000, freq * 6);
    const g = ctx.createGain();
    const peak = 0.18 * (n.velocity / 127);
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(peak, t0 + 0.008);
    g.gain.setTargetAtTime(peak * 0.6, t0 + 0.01, 0.15);
    g.gain.setTargetAtTime(0, t1, 0.03);
    osc.connect(filt).connect(g).connect(out);
    osc.start(t0);
    osc.stop(t1 + 0.2);
    this.track(osc);
  }

  private drumVoice(out: AudioNode, n: Note, t0: number) {
    const ctx = this.ctx!;
    const v = n.velocity / 127;
    const g = ctx.createGain();
    g.connect(out);
    if (n.pitch === 35 || n.pitch === 36) {
      const osc = ctx.createOscillator();
      osc.frequency.setValueAtTime(150, t0);
      osc.frequency.exponentialRampToValueAtTime(45, t0 + 0.12);
      g.gain.setValueAtTime(0.9 * v, t0);
      g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.3);
      osc.connect(g);
      osc.start(t0);
      osc.stop(t0 + 0.32);
      this.track(osc);
      return;
    }
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const f = ctx.createBiquadFilter();
    const isHat = n.pitch === 42 || n.pitch === 44 || n.pitch === 46 || n.pitch >= 49;
    f.type = isHat ? "highpass" : "bandpass";
    f.frequency.value = isHat ? 7000 : 1800;
    const len = isHat ? (n.pitch === 46 ? 0.25 : 0.05) : 0.16;
    g.gain.setValueAtTime(0.5 * v, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + len);
    src.connect(f).connect(g);
    src.start(t0, Math.random() * 0.5);
    src.stop(t0 + len + 0.02);
    this.track(src);
  }

  /** Audition a single note immediately (used when inserting/clicking notes). */
  preview(laneId: string, n: Note) {
    const lane = this.lanes.get(laneId);
    const ctx = this.ensureContext();
    if (!lane) return;
    if (ctx.state === "suspended") void ctx.resume();
    const t0 = ctx.currentTime + 0.01;
    const out = ctx.createGain();
    out.gain.value = Math.max(0.3, lane.mix.volume);
    out.connect(this.master!);
    if (lane.isDrum) this.drumVoice(out, n, t0);
    else this.toneVoice(out, { ...n, bends: null }, t0, t0 + Math.min(0.6, n.end - n.start));
  }

  dispose() {
    this.stopSources();
    this.playing = false;
    void this.ctx?.close();
    this.ctx = null;
    this.lanes.clear();
  }
}
