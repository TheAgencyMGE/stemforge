"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useWorkspace } from "./Workspace";
import { STEM_COLOR, STEM_HEX } from "@/lib/format";
import { secondsPerBeat } from "@/lib/notes";
import { ORIGINAL_LANE, useEditor } from "@/lib/store";
import type { Note, Peaks } from "@/lib/types";

const RULER = 24;

interface LaneDef {
  id: string;
  stem: string;
  name: string;
  trackId: string | null;
}

export function useCanvasSize(ref: React.RefObject<HTMLElement | null>) {
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, [ref]);
  return size;
}

export function setupCanvas(c: HTMLCanvasElement, w: number, h: number) {
  const dpr = window.devicePixelRatio || 1;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
  }
  const g = c.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return g;
}

/** Ruler with bars/beats from the tempo grid, section markers and the loop region. */
export function drawRuler(g: CanvasRenderingContext2D, w: number, x0: number, scrollX: number, pps: number,
  grid: ReturnType<typeof useEditor.getState>["grid"] extends () => infer R ? R : never,
  sections: { start: number; end: number; label: string }[], offset: number, h = RULER) {
  g.fillStyle = "#111419";
  g.fillRect(x0, 0, w - x0, h);
  const spb = secondsPerBeat(grid);
  const beatPx = spb * pps;
  const showBeats = beatPx > 14;
  const barEvery = Math.max(1, Math.ceil(60 / (beatPx * grid.beatsPerBar)));
  const t0 = scrollX, t1 = scrollX + (w - x0) / pps;
  const k0 = Math.floor((t0 - grid.origin) / spb) - 1;
  const k1 = Math.ceil((t1 - grid.origin) / spb) + 1;
  g.font = "10.5px ui-monospace, Consolas, monospace";
  for (let k = k0; k <= k1; k++) {
    const t = grid.origin + k * spb;
    if (t < 0) continue;
    const x = x0 + (t - scrollX) * pps;
    const isBar = ((k % grid.beatsPerBar) + grid.beatsPerBar) % grid.beatsPerBar === 0;
    const bar = Math.floor(k / grid.beatsPerBar) + 1;
    if (isBar && (bar - 1) % barEvery === 0) {
      g.fillStyle = "#56606c";
      g.fillRect(x, h - 12, 1, 12);
      g.fillStyle = "#9aa3ad";
      g.fillText(String(bar), x + 3, h - 4);
    } else if (showBeats) {
      g.fillStyle = "#353d47";
      g.fillRect(x, h - 5, 1, 5);
    }
  }
  for (const s of sections) {
    const x = x0 + (s.start - scrollX) * pps;
    if (x < x0 - 40 || x > w) continue;
    g.fillStyle = "#ff7a3d";
    g.fillRect(x, 0, 1, 9);
    g.fillStyle = "#c98a6a";
    g.fillText(s.label, x + 3, 9);
  }
  void offset;
}

export function TrackLanes() {
  const { project, peaks, engine, duration, laneLoad } = useWorkspace();
  const mix = useEditor((s) => s.mix);
  const setMix = useEditor((s) => s.setMix);
  const selectedTrackId = useEditor((s) => s.selectedTrackId);
  const selectTrack = useEditor((s) => s.selectTrack);
  const notes = useEditor((s) => s.notes);
  const pps = useEditor((s) => s.pxPerSec);
  const scrollX = useEditor((s) => s.scrollX);
  const loop = useEditor((s) => s.loop);
  const division = useEditor((s) => s.division);
  const tsig = project.time_signature;
  const tempo = project.tempo_bpm;

  const lanes: LaneDef[] = useMemo(
    () => [
      { id: ORIGINAL_LANE, stem: "original", name: "Original mix", trackId: null },
      ...project.tracks.map((t) => ({ id: t.id, stem: t.stem, name: t.display_name, trackId: t.id })),
    ],
    [project.tracks],
  );

  const wrapRef = useRef<HTMLDivElement>(null);
  const size = useCanvasSize(wrapRef);
  const laneH = Math.max(38, Math.min(84, Math.floor((size.h - RULER) / lanes.length)));
  const staticRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);

  // ---- static layer ----
  useEffect(() => {
    const c = staticRef.current;
    if (!c || !size.w) return;
    const g = setupCanvas(c, size.w, size.h);
    g.clearRect(0, 0, size.w, size.h);
    const grid = useEditor.getState().grid();
    drawRuler(g, size.w, 0, scrollX, pps, grid, project.analysis?.sections ?? [], project.offset_seconds);
    // beat grid lines
    const spb = secondsPerBeat(grid);
    if (spb * pps > 10) {
      const k0 = Math.floor((scrollX - grid.origin) / spb);
      const k1 = Math.ceil((scrollX + size.w / pps - grid.origin) / spb);
      for (let k = k0; k <= k1; k++) {
        const x = (grid.origin + k * spb - scrollX) * pps;
        const isBar = ((k % grid.beatsPerBar) + grid.beatsPerBar) % grid.beatsPerBar === 0;
        g.fillStyle = isBar ? "rgba(255,255,255,0.07)" : "rgba(255,255,255,0.025)";
        g.fillRect(x, RULER, 1, size.h - RULER);
      }
    }
    lanes.forEach((lane, i) => {
      const y = RULER + i * laneH;
      g.fillStyle = lane.trackId === selectedTrackId ? "rgba(255,255,255,0.035)" : "transparent";
      g.fillRect(0, y, size.w, laneH);
      g.fillStyle = "#232931";
      g.fillRect(0, y + laneH - 1, size.w, 1);
      const color = STEM_HEX[lane.stem] ?? "#9aa3ad";
      const m = mix[lane.id];
      const dim = m && (m.muted || (Object.values(mix).some((x) => x.soloed) && !m.soloed));
      const p = peaks[lane.id];
      if (p) drawWave(g, p, y + 4, laneH - 8, size.w, scrollX, pps, color, dim ? 0.3 : 0.85);
      const ns = lane.trackId ? notes[lane.trackId] : null;
      if (ns?.length) drawMiniNotes(g, ns, y + 3, laneH - 6, size.w, scrollX, pps, dim ? 0.35 : 0.95);
    });
    if (duration) {
      const xe = (duration - scrollX) * pps;
      if (xe < size.w) {
        g.fillStyle = "rgba(0,0,0,0.45)";
        g.fillRect(xe, RULER, size.w - xe, size.h - RULER);
      }
    }
  }, [size, scrollX, pps, lanes, peaks, notes, mix, selectedTrackId, laneH, duration, project.analysis, project.offset_seconds, division, tsig, tempo]);

  // ---- overlay: playhead + loop, every frame ----
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      const c = overlayRef.current;
      if (c && size.w) {
        const g = setupCanvas(c, size.w, size.h);
        g.clearRect(0, 0, size.w, size.h);
        const st = useEditor.getState();
        const lp = st.loop;
        if (lp) {
          const a = (lp.start - st.scrollX) * st.pxPerSec, b = (lp.end - st.scrollX) * st.pxPerSec;
          g.fillStyle = "rgba(255,122,61,0.10)";
          g.fillRect(a, 0, b - a, size.h);
          g.fillStyle = "rgba(255,122,61,0.75)";
          g.fillRect(a, 0, b - a, 4);
        }
        const t = engine.currentTime;
        const x = (t - st.scrollX) * st.pxPerSec;
        g.fillStyle = "#ffe9a8";
        g.fillRect(Math.round(x), 0, 1.5, size.h);
        // follow playhead
        const visible = size.w / st.pxPerSec;
        if (engine.isPlaying && (t > st.scrollX + visible * 0.92 || t < st.scrollX)) {
          st.set({ scrollX: Math.max(0, t - visible * 0.08) });
        }
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [engine, size]);

  // ---- interaction ----
  const drag = useRef<{ x: number; t: number; mode: "ruler" | "lane"; lane?: LaneDef } | null>(null);
  const timeAt = (clientX: number) => {
    const r = overlayRef.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, scrollX + (clientX - r.left) / pps));
  };

  return (
    <div className="lanes" style={{ ["--lane-h" as string]: `${laneH}px` }}>
      <div className="lane-head-col" role="list" aria-label="Tracks">
        <div style={{ height: RULER, borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", padding: "0 10px" }} className="faint">
          <span style={{ fontSize: 11 }}>TRACKS</span>
        </div>
        {lanes.map((lane) => {
          const m = mix[lane.id];
          const track = project.tracks.find((t) => t.id === lane.trackId);
          const load = laneLoad[lane.id];
          return (
            <div
              key={lane.id}
              role="listitem"
              className={`lane-head ${lane.trackId && lane.trackId === selectedTrackId ? "selected" : ""}`}
              style={{ ["--lane-color" as string]: STEM_COLOR[lane.stem] ?? "var(--stem-original)" }}
              onClick={() => lane.trackId && selectTrack(lane.trackId)}
            >
              <div className="lname">
                <span>{lane.name}</span>
                {track?.warnings.length ? <span title={track.warnings.join("\n")} style={{ color: "var(--warn)" }} aria-label="Has warnings">⚠</span> : null}
                {load === undefined && project.artifacts.some((x) => x.kind === "preview" && x.stem === lane.stem) && <span className="faint" style={{ fontSize: 11 }}>loading…</span>}
                {load === "error" && <span style={{ color: "var(--err)", fontSize: 11 }}>audio error</span>}
                <span className="spacer" />
                {track && <span className="faint num" style={{ fontSize: 11 }}>{track.transcription_status === "done" ? `${notes[track.id]?.length ?? track.note_count}♪` : track.transcription_status}</span>}
              </div>
              {laneH >= 50 && m && (
                <div className="ctrls" onClick={(e) => e.stopPropagation()}>
                  <button className="tbtn m" aria-pressed={m.muted} aria-label={`Mute ${lane.name}`} title="Mute (M)" onClick={() => setMix(lane.id, { muted: !m.muted })}>M</button>
                  <button className="tbtn s" aria-pressed={m.soloed} aria-label={`Solo ${lane.name}`} title="Solo (S)" onClick={() => setMix(lane.id, { soloed: !m.soloed })}>S</button>
                  {lane.trackId && (
                    <>
                      <button className="tbtn a" aria-pressed={m.audio} aria-label={`Monitor ${lane.name} audio`} title="Hear separated audio" onClick={() => setMix(lane.id, { audio: !m.audio })}>A</button>
                      <button className="tbtn mi" aria-pressed={m.midi} aria-label={`Monitor ${lane.name} MIDI`} title="Hear MIDI preview synth" disabled={track?.transcription_status !== "done"} onClick={() => setMix(lane.id, { midi: !m.midi })}>MIDI</button>
                    </>
                  )}
                  <input
                    type="range" min={0} max={1.5} step={0.01} value={m.volume}
                    aria-label={`${lane.name} volume`}
                    onChange={(e) => setMix(lane.id, { volume: Number(e.target.value) })}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div className="timeline-col" ref={wrapRef}>
        <canvas ref={staticRef} aria-hidden="true" style={{ position: "absolute", inset: 0 }} />
        <canvas
          ref={overlayRef}
          style={{ position: "absolute", inset: 0, cursor: "pointer", touchAction: "none" }}
          role="slider"
          aria-label="Timeline: click to seek, drag on the ruler to set a loop, Ctrl+wheel to zoom"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={0}
          tabIndex={0}
          onPointerDown={(e) => {
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            const r = overlayRef.current!.getBoundingClientRect();
            const y = e.clientY - r.top;
            const mode = y < RULER ? "ruler" : "lane";
            const idx = Math.floor((y - RULER) / laneH);
            drag.current = { x: e.clientX, t: timeAt(e.clientX), mode, lane: lanes[idx] };
          }}
          onPointerMove={(e) => {
            const d = drag.current;
            if (!d || d.mode !== "ruler" || Math.abs(e.clientX - d.x) < 4) return;
            const t = timeAt(e.clientX);
            useEditor.getState().set({ loop: { start: Math.min(d.t, t), end: Math.max(d.t, t) } });
          }}
          onPointerUp={(e) => {
            const d = drag.current;
            drag.current = null;
            if (!d) return;
            if (Math.abs(e.clientX - d.x) < 4) {
              engine.seek(d.t);
              if (d.mode === "lane" && d.lane?.trackId) selectTrack(d.lane.trackId);
            }
          }}
          onWheel={(e) => {
            const st = useEditor.getState();
            if (e.ctrlKey || e.metaKey) {
              const t = timeAt(e.clientX);
              const np = Math.max(8, Math.min(1200, st.pxPerSec * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
              const r = overlayRef.current!.getBoundingClientRect();
              st.set({ pxPerSec: np, scrollX: Math.max(0, t - (e.clientX - r.left) / np) });
            } else {
              const d = (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / st.pxPerSec;
              st.set({ scrollX: Math.max(0, Math.min(Math.max(0, duration - 1), st.scrollX + d)) });
            }
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
              e.stopPropagation();
              e.preventDefault();
              engine.seek(engine.currentTime + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 10 : 1));
            }
          }}
        />
      </div>
    </div>
  );
}

function drawWave(g: CanvasRenderingContext2D, p: Peaks, y: number, h: number, w: number, scrollX: number, pps: number, color: string, alpha: number) {
  const bucketSec = p.samples_per_bucket / p.sample_rate;
  const mid = y + h / 2;
  g.globalAlpha = alpha;
  g.fillStyle = color;
  for (let x = 0; x < w; x++) {
    const t0 = scrollX + x / pps, t1 = scrollX + (x + 1) / pps;
    let i0 = Math.floor(t0 / bucketSec), i1 = Math.max(i0 + 1, Math.ceil(t1 / bucketSec));
    if (i0 >= p.max.length) break;
    i1 = Math.min(i1, p.max.length);
    let mn = 0, mx = 0;
    for (let i = i0; i < i1; i++) {
      if (p.min[i] < mn) mn = p.min[i];
      if (p.max[i] > mx) mx = p.max[i];
    }
    const a = (mx / 127) * (h / 2), b = (mn / 127) * (h / 2);
    g.fillRect(x, mid - a, 1, Math.max(1, a - b));
    if (i0 < 0) i0 = 0;
  }
  g.globalAlpha = 1;
}

function drawMiniNotes(g: CanvasRenderingContext2D, notes: Note[], y: number, h: number, w: number, scrollX: number, pps: number, alpha: number) {
  let lo = 127, hi = 0;
  for (const n of notes) {
    if (n.pitch < lo) lo = n.pitch;
    if (n.pitch > hi) hi = n.pitch;
  }
  const span = Math.max(12, hi - lo + 1);
  const t1 = scrollX + w / pps;
  g.globalAlpha = alpha;
  g.fillStyle = "#f4f6f8";
  const rh = Math.max(1.5, Math.min(4, h / span));
  for (const n of notes) {
    if (n.end < scrollX || n.start > t1) continue;
    const x = (n.start - scrollX) * pps;
    const ww = Math.max(1.5, (n.end - n.start) * pps);
    const yy = y + h - ((n.pitch - lo + 0.5) / span) * h;
    g.fillRect(x, yy - rh / 2, ww, rh);
  }
  g.globalAlpha = 1;
}
