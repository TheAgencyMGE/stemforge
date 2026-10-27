"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { setupCanvas, useCanvasSize } from "./TrackLanes";
import { useWorkspace } from "./Workspace";
import { STEM_HEX, STEM_COLOR, stemKey } from "@/lib/format";
import {
  DRUM_NAMES, deleteNotes, gridStep, insertNote, moveNotes, notesInRect, pitchName, quantize, resizeNotes,
  secondsPerBeat, setVelocity, snapTime, snapToScale, transpose, MIN_NOTE,
} from "@/lib/notes";
import { useEditor } from "@/lib/store";
import type { Note } from "@/lib/types";

const KEY_W = 58;
const TOP = 18;
const VEL_H = 64;
const EDGE = 6;
const BLACK = new Set([1, 3, 6, 8, 10]);
const DRUM_ROWS = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 54, 70];

type Drag =
  | { mode: "move"; x: number; y: number; t: number; p: number; anchor: Note; base: Note[]; ids: Set<string>; moved: boolean; toggleId?: string }
  | { mode: "resize-end" | "resize-start"; x: number; t: number; anchor: Note; base: Note[]; ids: Set<string> }
  | { mode: "marquee"; t0: number; p0: number; t1: number; p1: number; additive: boolean; prior: Set<string> }
  | { mode: "draw"; id: string; t0: number; base: Note[] }
  | { mode: "velocity"; ids: Set<string> | null }
  | { mode: "seek" };

export function PianoRoll() {
  const { project, engine, duration, spectrograms } = useWorkspace();
  const trackId = useEditor((s) => s.selectedTrackId);
  const track = project.tracks.find((t) => t.id === trackId) ?? null;
  const notes = useEditor((s) => (trackId ? s.notes[trackId] : undefined)) ?? EMPTY;
  const selection = useEditor((s) => s.selection);
  const tool = useEditor((s) => s.tool);
  const st = useEditor;

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const size = useCanvasSize(wrapRef);
  const view = useRef({ pitchTop: 84, rowH: 12 });
  const drag = useRef<Drag | null>(null);
  const hover = useRef<{ t: number; p: number } | null>(null);
  const [, force] = useState(0);
  const isDrum = !!track?.is_drum;
  const color = STEM_HEX[stemKey(track)] ?? "#4fd1c5";

  const noteAreaH = Math.max(40, size.h - TOP - VEL_H - 6);

  // Fit the vertical view to the track's notes when switching tracks.
  const fit = useCallback(() => {
    const ns = st.getState().notes[trackId ?? ""] ?? [];
    if (isDrum) {
      view.current.rowH = Math.max(10, Math.min(24, noteAreaH / DRUM_ROWS.length));
      view.current.pitchTop = DRUM_ROWS.length - 1 + 0.0;
    } else {
      let lo = 48, hi = 72;
      if (ns.length) {
        lo = Math.min(...ns.map((n) => n.pitch));
        hi = Math.max(...ns.map((n) => n.pitch));
      }
      const span = hi - lo + 7;
      view.current.rowH = Math.max(5, Math.min(18, noteAreaH / span));
      view.current.pitchTop = hi + 3 + Math.max(0, (noteAreaH / view.current.rowH - span) / 2);
    }
    force((x) => x + 1);
  }, [isDrum, noteAreaH, trackId, st]);

  const loadedFor = useRef<string | null>(null);
  useEffect(() => {
    if (!trackId || !size.h) return;
    const key = `${trackId}:${notes.length > 0}`;
    if (loadedFor.current === key) return;
    loadedFor.current = key;
    fit();
  }, [trackId, size.h, notes.length, fit]);

  // ---- coordinate helpers (drum view maps rows to an index into DRUM_ROWS) ----
  const rowToPitch = useCallback((row: number) => (isDrum ? DRUM_ROWS[Math.max(0, Math.min(DRUM_ROWS.length - 1, row))] : row), [isDrum]);
  const pitchToRow = useCallback((p: number) => (isDrum ? DRUM_ROWS.indexOf(p) : p), [isDrum]);
  const yOfRow = (row: number) => TOP + (view.current.pitchTop - row) * view.current.rowH;
  const xOf = (t: number) => KEY_W + (t - st.getState().scrollX) * st.getState().pxPerSec;
  const tAt = (x: number) => st.getState().scrollX + (x - KEY_W) / st.getState().pxPerSec;
  const rowAt = (y: number) => Math.floor(view.current.pitchTop - (y - TOP) / view.current.rowH) + 1;

  const hitNote = useCallback(
    (x: number, y: number): { note: Note; edge: "start" | "end" | null } | null => {
      const row = rowAt(y);
      const t = tAt(x);
      const pps = st.getState().pxPerSec;
      for (let i = notes.length - 1; i >= 0; i--) {
        const n = notes[i];
        if (pitchToRow(n.pitch) !== row) continue;
        const pad = EDGE / pps;
        if (t < n.start - 1 / pps || t > n.end + pad / 2) continue;
        const w = (n.end - n.start) * pps;
        if (w > 10 && (n.end - t) * pps <= EDGE) return { note: n, edge: "end" };
        if (w > 16 && (t - n.start) * pps <= 4) return { note: n, edge: "start" };
        if (t >= n.start && t <= n.end) return { note: n, edge: null };
      }
      return null;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [notes, pitchToRow],
  );

  // ---- render loop ----
  useEffect(() => {
    let raf = 0;
    const render = () => {
      const c = canvasRef.current;
      if (c && size.w && size.h) draw(c);
      raf = requestAnimationFrame(render);
    };
    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  });

  function draw(c: HTMLCanvasElement) {
    const g = setupCanvas(c, size.w, size.h);
    const s = st.getState();
    const { rowH, pitchTop } = view.current;
    const pps = s.pxPerSec;
    const grid = s.grid();
    const areaBottom = TOP + noteAreaH;
    g.fillStyle = "#101318";
    g.fillRect(0, 0, size.w, size.h);
    if (!track) {
      g.fillStyle = "#7d8792";
      g.font = "13px system-ui";
      g.fillText("Select a track to edit its notes.", KEY_W + 16, TOP + 30);
      return;
    }
    const rowLo = Math.floor(pitchTop - noteAreaH / rowH) - 1;
    const rowHi = Math.ceil(pitchTop) + 1;
    // rows
    g.save();
    g.beginPath();
    g.rect(KEY_W, TOP, size.w - KEY_W, noteAreaH);
    g.clip();
    for (let row = rowLo; row <= rowHi; row++) {
      const y = yOfRow(row);
      const p = rowToPitch(row);
      if (!isDrum && (row < 0 || row > 127)) continue;
      if (isDrum && (row < 0 || row >= DRUM_ROWS.length)) continue;
      g.fillStyle = !isDrum && BLACK.has(p % 12) ? "#14181e" : "#191d24";
      g.fillRect(KEY_W, y, size.w - KEY_W, rowH);
      if (!isDrum && p % 12 === 0) {
        g.fillStyle = "#2a313a";
        g.fillRect(KEY_W, y + rowH - 1, size.w - KEY_W, 1);
      }
    }
    // pitch spectrogram underlay (rows align 1:1 with MIDI pitches)
    const spec = trackId ? spectrograms[trackId] : undefined;
    if (spec && s.showSpectrogram && !isDrum) {
      const t0 = s.scrollX, t1 = s.scrollX + (size.w - KEY_W) / pps;
      const pHi = Math.min(spec.midiHigh, Math.ceil(pitchTop) + 1), pLo = Math.max(spec.midiLow, rowLo);
      if (pHi >= pLo) {
        const sx = Math.max(0, t0 / spec.hop), sw = Math.min(spec.img.width - sx, (t1 - t0) / spec.hop);
        const sy = spec.midiHigh - pHi, sh = pHi - pLo + 1;
        if (sw > 0) {
          g.globalAlpha = 0.5;
          g.imageSmoothingEnabled = true;
          g.drawImage(spec.img, sx, sy, sw, sh, xOf(sx * spec.hop), yOfRow(pHi), sw * spec.hop * pps, sh * rowH);
          g.globalAlpha = 1;
        }
      }
    }
    // grid
    const spb = secondsPerBeat(grid);
    const step = gridStep(grid);
    const tStart = s.scrollX, tEnd = s.scrollX + (size.w - KEY_W) / pps;
    if (step > 0 && step * pps > 6) {
      const k0 = Math.floor((tStart - grid.origin) / step), k1 = Math.ceil((tEnd - grid.origin) / step);
      g.fillStyle = "rgba(255,255,255,0.035)";
      for (let k = k0; k <= k1; k++) g.fillRect(Math.round(xOf(grid.origin + k * step)), TOP, 1, noteAreaH);
    }
    if (spb * pps > 8) {
      const k0 = Math.floor((tStart - grid.origin) / spb), k1 = Math.ceil((tEnd - grid.origin) / spb);
      for (let k = k0; k <= k1; k++) {
        const isBar = ((k % grid.beatsPerBar) + grid.beatsPerBar) % grid.beatsPerBar === 0;
        g.fillStyle = isBar ? "rgba(255,255,255,0.13)" : "rgba(255,255,255,0.06)";
        g.fillRect(Math.round(xOf(grid.origin + k * spb)), TOP, 1, noteAreaH);
      }
    }
    // loop region
    if (s.loop) {
      g.fillStyle = "rgba(255,122,61,0.07)";
      g.fillRect(xOf(s.loop.start), TOP, (s.loop.end - s.loop.start) * pps, noteAreaH);
    }
    // notes
    g.font = "10px ui-monospace, Consolas, monospace";
    for (const n of notes) {
      if (n.end < tStart || n.start > tEnd) continue;
      const row = pitchToRow(n.pitch);
      if (row < rowLo || row > rowHi) continue;
      const x = xOf(n.start), y = yOfRow(row);
      const w = Math.max(2, (isDrum ? Math.max(n.end - n.start, 0.05) : n.end - n.start) * pps);
      const selected = s.selection.has(n.id);
      const conf = n.confidence;
      const low = conf !== null && conf < s.confThreshold;
      g.globalAlpha = s.showConfidence && conf !== null ? 0.3 + 0.7 * conf : 0.92;
      g.fillStyle = color;
      g.fillRect(x, y + 1, w, rowH - 2);
      g.globalAlpha = 1;
      if (s.showConfidence && low) {
        g.strokeStyle = "#f3c25b";
        g.setLineDash([3, 2]);
        g.lineWidth = 1.2;
        g.strokeRect(x + 0.5, y + 1.5, w - 1, rowH - 3);
        g.setLineDash([]);
      } else if (conf === null) {
        g.strokeStyle = "rgba(255,255,255,0.7)";
        g.lineWidth = 1;
        g.strokeRect(x + 0.5, y + 1.5, w - 1, rowH - 3);
      }
      if (selected) {
        g.strokeStyle = "#ffe9a8";
        g.lineWidth = 2;
        g.strokeRect(x + 1, y + 2, w - 2, rowH - 4);
      }
      // velocity tick on the left edge
      g.fillStyle = "rgba(0,0,0,0.35)";
      g.fillRect(x, y + 1, 2, (rowH - 2) * (1 - n.velocity / 127));
      if (n.bends?.length && rowH >= 6) {
        g.strokeStyle = "rgba(255,255,255,0.85)";
        g.lineWidth = 1;
        g.beginPath();
        n.bends.forEach(([dt, semis], i) => {
          const bx = x + dt * pps, by = y + rowH / 2 - semis * rowH;
          if (i === 0) g.moveTo(bx, by);
          else g.lineTo(bx, by);
        });
        g.stroke();
      }
      if (rowH >= 11 && w > 30) {
        g.fillStyle = "rgba(10,12,15,0.85)";
        g.fillText(isDrum ? DRUM_NAMES[n.pitch] ?? String(n.pitch) : pitchName(n.pitch), x + 4, y + rowH - 3);
      }
    }
    // marquee
    const d = drag.current;
    if (d?.mode === "marquee") {
      const x0 = xOf(d.t0), x1 = xOf(d.t1);
      const y0 = yOfRow(d.p0) + (d.p0 >= d.p1 ? 0 : rowH), y1 = yOfRow(d.p1) + (d.p1 > d.p0 ? rowH : 0);
      g.fillStyle = "rgba(124,196,255,0.12)";
      g.strokeStyle = "rgba(124,196,255,0.8)";
      g.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
      g.strokeRect(Math.min(x0, x1) + 0.5, Math.min(y0, y1) + 0.5, Math.abs(x1 - x0), Math.abs(y1 - y0));
    }
    // hover ghost for draw tool
    if (s.tool === "draw" && hover.current && !d) {
      const ht = snapTime(hover.current.t, grid, "floor");
      g.fillStyle = "rgba(255,255,255,0.12)";
      g.fillRect(xOf(ht), yOfRow(hover.current.p) + 1, Math.max(6, (step || spb / 4) * pps), rowH - 2);
    }
    // playhead
    const ph = xOf(engine.currentTime);
    g.fillStyle = "#ffe9a8";
    g.fillRect(Math.round(ph), TOP, 1.5, noteAreaH);
    g.restore();

    // keyboard
    for (let row = rowLo; row <= rowHi; row++) {
      if (isDrum && (row < 0 || row >= DRUM_ROWS.length)) continue;
      if (!isDrum && (row < 0 || row > 127)) continue;
      const y = yOfRow(row);
      if (y > areaBottom || y + rowH < TOP) continue;
      const p = rowToPitch(row);
      const black = !isDrum && BLACK.has(p % 12);
      g.fillStyle = black ? "#2b3038" : "#d9dde3";
      g.fillRect(0, Math.max(TOP, y), KEY_W - 2, Math.min(rowH - 1, areaBottom - y));
      if (isDrum || p % 12 === 0 || rowH >= 14) {
        g.fillStyle = black ? "#cfd5dc" : "#2b3038";
        g.font = "10px ui-monospace, Consolas, monospace";
        const label = isDrum ? (DRUM_NAMES[p] ?? String(p)).slice(0, 9) : pitchName(p);
        if (y + rowH - 3 > TOP && y + rowH - 3 < areaBottom) g.fillText(label, 4, y + rowH - 3);
      }
    }
    // top ruler
    g.fillStyle = "#111419";
    g.fillRect(KEY_W, 0, size.w - KEY_W, TOP);
    if (spb * pps > 6) {
      const k0 = Math.floor((tStart - grid.origin) / spb), k1 = Math.ceil((tEnd - grid.origin) / spb);
      g.font = "10px ui-monospace, Consolas, monospace";
      for (let k = k0; k <= k1; k++) {
        if (((k % grid.beatsPerBar) + grid.beatsPerBar) % grid.beatsPerBar !== 0) continue;
        const x = xOf(grid.origin + k * spb);
        if (x < KEY_W) continue;
        g.fillStyle = "#56606c";
        g.fillRect(x, 6, 1, TOP - 6);
        g.fillStyle = "#9aa3ad";
        g.fillText(String(Math.floor(k / grid.beatsPerBar) + 1), x + 3, 13);
      }
    }
    g.fillStyle = "#ffe9a8";
    if (ph >= KEY_W) g.fillRect(Math.round(ph), 0, 1.5, TOP);
    g.fillStyle = "#14171c";
    g.fillRect(0, 0, KEY_W, TOP);

    // velocity lane
    const vy = areaBottom + 6;
    g.fillStyle = "#0c0e12";
    g.fillRect(0, vy - 2, size.w, size.h - vy + 2);
    g.fillStyle = "#7d8792";
    g.font = "10px system-ui";
    g.fillText("velocity", 6, vy + 12);
    for (const n of notes) {
      if (n.end < tStart || n.start > tEnd) continue;
      const x = xOf(n.start);
      if (x < KEY_W) continue;
      const hgt = (n.velocity / 127) * (VEL_H - 8);
      g.fillStyle = s.selection.has(n.id) ? "#ffe9a8" : color;
      g.globalAlpha = s.selection.has(n.id) ? 1 : 0.75;
      g.fillRect(x, vy + VEL_H - 4 - hgt, 3, hgt);
      g.globalAlpha = 1;
    }
  }

  // ---- pointer handling ----
  const pos = (e: React.PointerEvent | React.WheelEvent | React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  const audition = (n: Note) => trackId && engine.preview(trackId, n);

  function onPointerDown(e: React.PointerEvent) {
    if (!track || !trackId) return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    canvasRef.current?.focus();
    const { x, y } = pos(e);
    const s = st.getState();
    const areaBottom = TOP + noteAreaH;
    const t = tAt(x);
    if (y < TOP) {
      drag.current = { mode: "seek" };
      engine.seek(Math.max(0, t));
      return;
    }
    if (y > areaBottom + 2) {
      // velocity lane: drag adjusts the notes under the pointer (or the whole selection)
      const near = notes.find((n) => Math.abs(xOf(n.start) - x) < 5);
      const ids = near && s.selection.has(near.id) ? new Set(s.selection) : null;
      drag.current = { mode: "velocity", ids };
      applyVelocity(x, y, ids, true);
      return;
    }
    const row = rowAt(y);
    if (x < KEY_W) {
      audition({ id: "k", pitch: rowToPitch(row), start: 0, end: 0.4, velocity: 100, confidence: null, bends: null });
      return;
    }
    const hit = hitNote(x, y);
    if (hit && e.altKey) {
      s.edit(trackId, (cur) => deleteNotes(cur, new Set([hit.note.id])));
      return;
    }
    if (hit) {
      let ids = new Set(s.selection);
      let toggleId: string | undefined;
      if (e.shiftKey || e.ctrlKey || e.metaKey) {
        if (ids.has(hit.note.id)) toggleId = hit.note.id;
        else ids.add(hit.note.id);
      } else if (!ids.has(hit.note.id)) {
        ids = new Set([hit.note.id]);
      }
      s.setSelection(ids);
      if (hit.edge) {
        drag.current = { mode: hit.edge === "end" ? "resize-end" : "resize-start", x, t, anchor: hit.note, base: notes, ids };
      } else {
        drag.current = { mode: "move", x, y, t, p: row, anchor: hit.note, base: notes, ids, moved: false, toggleId };
        audition(hit.note);
      }
      return;
    }
    if (s.tool === "draw") {
      insertAt(t, row, true);
      return;
    }
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    drag.current = { mode: "marquee", t0: t, p0: row, t1: t, p1: row, additive, prior: additive ? new Set(s.selection) : new Set() };
    if (!additive) s.setSelection(new Set());
  }

  /** Insert a note at (time, row); with drag=true the pointer then sets its length. */
  function insertAt(t: number, row: number, withDrag: boolean) {
    if (!trackId) return;
    const s = st.getState();
    {
      const grid = s.grid();
      const start = Math.max(0, snapTime(t, grid, "floor"));
      const len = gridStep(grid) || secondsPerBeat(grid) / 4;
      const vel = notes.length ? Math.round(notes.reduce((a, n) => a + n.velocity, 0) / notes.length) : 96;
      let created: Note | null = null;
      s.edit(trackId, (cur) => {
        const r = insertNote(cur, { pitch: rowToPitch(row), start, end: Math.min(duration, start + len), velocity: vel, confidence: null, bends: null }, duration);
        created = r.note;
        return r.notes;
      });
      if (created) {
        const c = created as Note;
        s.setSelection(new Set([c.id]));
        audition(c);
        if (withDrag) drag.current = { mode: "draw", id: c.id, t0: start, base: st.getState().notes[trackId] ?? [] };
      }
    }
  }

  function applyVelocity(x: number, y: number, ids: Set<string> | null, first: boolean) {
    if (!trackId) return;
    const vy = TOP + noteAreaH + 6;
    const v = Math.max(1, Math.min(127, Math.round(((vy + VEL_H - 4 - y) / (VEL_H - 8)) * 127)));
    const target = ids ?? new Set(notes.filter((n) => Math.abs(xOf(n.start) - x) < 4).map((n) => n.id));
    if (!target.size) return;
    st.getState().edit(trackId, (cur) => setVelocity(cur, target, v), { coalesce: first ? undefined : "velocity" });
  }

  function onPointerMove(e: React.PointerEvent) {
    const { x, y } = pos(e);
    const d = drag.current;
    const s = st.getState();
    const cnv = canvasRef.current!;
    if (!d) {
      hover.current = x > KEY_W && y > TOP && y < TOP + noteAreaH ? { t: tAt(x), p: rowAt(y) } : null;
      const hit = y > TOP && y < TOP + noteAreaH && x > KEY_W ? hitNote(x, y) : null;
      cnv.style.cursor = hit ? (hit.edge ? "ew-resize" : "grab") : s.tool === "draw" ? "crosshair" : x < KEY_W ? "pointer" : "default";
      return;
    }
    if (!trackId) return;
    const grid = s.grid();
    const snap = !e.altKey && gridStep(grid) > 0;
    const t = tAt(x);
    if (d.mode === "seek") {
      engine.seek(Math.max(0, t));
    } else if (d.mode === "velocity") {
      applyVelocity(x, y, d.ids, false);
    } else if (d.mode === "move") {
      if (!d.moved && Math.abs(x - d.x) < 3 && Math.abs(y - d.y) < 3) return;
      d.moved = true;
      cnv.style.cursor = "grabbing";
      let dt = t - d.t;
      if (snap) dt = snapTime(d.anchor.start + dt, grid) - d.anchor.start;
      const dRow = rowAt(y) - d.p;
      const dp = isDrum ? rowToPitch(pitchToRow(d.anchor.pitch) + dRow) - d.anchor.pitch : dRow;
      const prevPitch = (s.notes[trackId] ?? []).find((n) => n.id === d.anchor.id)?.pitch;
      s.edit(trackId, () => moveNotes(d.base, d.ids, dt, dp, duration), { coalesce: "drag" });
      if (prevPitch !== undefined && prevPitch !== d.anchor.pitch + dp) audition({ ...d.anchor, pitch: d.anchor.pitch + dp });
    } else if (d.mode === "resize-end" || d.mode === "resize-start") {
      let dt = t - d.t;
      if (snap) {
        const edgeT = d.mode === "resize-end" ? d.anchor.end : d.anchor.start;
        dt = snapTime(edgeT + dt, grid) - edgeT;
      }
      s.edit(trackId, () => resizeNotes(d.base, d.ids, d.mode === "resize-end" ? "end" : "start", dt, duration), { coalesce: "drag" });
    } else if (d.mode === "draw") {
      const end = Math.max(d.t0 + (snap ? gridStep(grid) : MIN_NOTE), snap ? snapTime(t, grid) : t);
      s.edit(trackId, () => d.base.map((n) => (n.id === d.id ? { ...n, end: Math.min(duration, end) } : n)), { coalesce: "drag" });
    } else if (d.mode === "marquee") {
      d.t1 = t;
      d.p1 = rowAt(y);
      const lo = Math.min(d.p0, d.p1), hi = Math.max(d.p0, d.p1);
      const inRect = isDrum
        ? new Set(notes.filter((n) => { const r = pitchToRow(n.pitch); return r >= lo && r <= hi && n.end >= Math.min(d.t0, t) && n.start <= Math.max(d.t0, t); }).map((n) => n.id))
        : notesInRect(notes, d.t0, t, lo, hi);
      s.setSelection(new Set([...d.prior, ...inRect]));
    }
  }

  function onPointerUp() {
    const d = drag.current;
    drag.current = null;
    if (d?.mode === "move" && !d.moved && d.toggleId) {
      const ids = new Set(st.getState().selection);
      ids.delete(d.toggleId);
      st.getState().setSelection(ids);
    }
  }

  function onDoubleClick(e: React.MouseEvent) {
    if (!trackId) return;
    const { x, y } = pos(e);
    if (y < TOP || y > TOP + noteAreaH || x < KEY_W) return;
    const hit = hitNote(x, y);
    if (hit) {
      st.getState().edit(trackId, (cur) => deleteNotes(cur, new Set([hit.note.id])));
      st.getState().setSelection(new Set());
    } else if (st.getState().tool === "select") {
      drag.current = null;
      insertAt(tAt(x), rowAt(y), false);
    }
  }

  function onWheel(e: React.WheelEvent) {
    const s = st.getState();
    const { x } = pos(e);
    if (e.ctrlKey || e.metaKey) {
      const t = tAt(x);
      const np = Math.max(8, Math.min(1200, s.pxPerSec * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
      s.set({ pxPerSec: np, scrollX: Math.max(0, t - (x - KEY_W) / np) });
    } else if (e.altKey) {
      view.current.rowH = Math.max(4, Math.min(28, view.current.rowH * (e.deltaY < 0 ? 1.1 : 1 / 1.1)));
      force((v) => v + 1);
    } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      const d = (e.shiftKey ? e.deltaY : e.deltaX) / s.pxPerSec;
      s.set({ scrollX: Math.max(0, Math.min(Math.max(0, duration - 1), s.scrollX + d)) });
    } else {
      const maxTop = isDrum ? DRUM_ROWS.length - 1 : 127;
      view.current.pitchTop = Math.max(noteAreaH / view.current.rowH - 1, Math.min(maxTop + 1, view.current.pitchTop - e.deltaY / 40));
      force((v) => v + 1);
    }
  }

  return (
    <div className="roll-wrap">
      <RollToolbar fit={fit} />
      <div className="roll-canvas-wrap" ref={wrapRef}>
        <canvas
          ref={canvasRef}
          tabIndex={0}
          role="application"
          aria-label={track ? `Piano roll for ${track.display_name}: ${notes.length} notes. Click to select, drag to move, drag edges to resize, double-click empty space to add a note, double-click a note to delete it.` : "Piano roll"}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerLeave={() => (hover.current = null)}
          onDoubleClick={onDoubleClick}
          onWheel={onWheel}
          onContextMenu={(e) => e.preventDefault()}
        />
      </div>
    </div>
  );
}

const EMPTY: Note[] = [];

function RollToolbar({ fit }: { fit: () => void }) {
  const { project } = useWorkspace();
  const s = useEditor();
  const track = project.tracks.find((t) => t.id === s.selectedTrackId);
  const notes = useMemo(() => (track ? s.notes[track.id] ?? [] : []), [s.notes, track]);
  const sel = s.selection;
  const [strength, setStrength] = useState(100);
  const key = project.analysis?.key;
  const [keyPc, setKeyPc] = useState<number>(key?.tonic_pc ?? 0);
  const [keyMode, setKeyMode] = useState<"major" | "minor">(key?.mode ?? "major");
  if (!track) return <div className="roll-toolbar faint">No track selected</div>;
  const tid = track.id;
  const target = sel.size ? sel : new Set(notes.map((n) => n.id));
  const scope = sel.size ? `${sel.size} selected` : "all notes";
  const selNotes = notes.filter((n) => sel.has(n.id));
  const avgVel = selNotes.length ? Math.round(selNotes.reduce((a, n) => a + n.velocity, 0) / selNotes.length) : null;

  return (
    <div className="roll-toolbar" role="toolbar" aria-label="Note editing tools">
      <span className="chip" style={{ borderColor: STEM_COLOR[stemKey(track)], color: STEM_COLOR[stemKey(track)] }}>{track.display_name}</span>
      <span className="faint num" style={{ fontSize: 12 }}>{notes.length} notes · {scope}</span>
      <span style={{ width: 8 }} />
      <button className="btn sm" aria-pressed={s.tool === "select"} onClick={() => s.set({ tool: "select" })} title="Select / move tool (V)">Select</button>
      <button className="btn sm" aria-pressed={s.tool === "draw"} onClick={() => s.set({ tool: "draw" })} title="Draw tool (D)">Draw</button>
      <span style={{ width: 8 }} />
      <button
        className="btn sm"
        disabled={gridStep(s.grid()) <= 0 || !notes.length}
        title="Quantize note starts toward the grid. This is an undoable edit. Reset restores the model timing, which is not quantized."
        onClick={() => s.edit(tid, (cur) => quantize(cur, target, s.grid(), strength / 100))}
      >
        Quantize
      </button>
      <label className="row" style={{ gap: 4 }} title="Quantize strength">
        <input type="range" min={10} max={100} step={10} value={strength} onChange={(e) => setStrength(Number(e.target.value))} aria-label="Quantize strength" style={{ width: 60 }} />
        <span className="num faint" style={{ fontSize: 11, width: 30 }}>{strength}%</span>
      </label>
      {!track.is_drum && (
        <>
          <button className="btn sm" disabled={!sel.size} onClick={() => s.edit(tid, (cur) => transpose(cur, sel, -12))} title="Octave down (Shift+↓)">−12</button>
          <button className="btn sm" disabled={!sel.size} onClick={() => s.edit(tid, (cur) => transpose(cur, sel, -1))} title="Semitone down (↓)">−1</button>
          <button className="btn sm" disabled={!sel.size} onClick={() => s.edit(tid, (cur) => transpose(cur, sel, 1))} title="Semitone up (↑)">+1</button>
          <button className="btn sm" disabled={!sel.size} onClick={() => s.edit(tid, (cur) => transpose(cur, sel, 12))} title="Octave up (Shift+↑)">+12</button>
          <select aria-label="Scale tonic" value={keyPc} onChange={(e) => setKeyPc(Number(e.target.value))} style={{ height: 26 }}>
            {["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"].map((n, i) => <option key={n} value={i}>{n}</option>)}
          </select>
          <select aria-label="Scale mode" value={keyMode} onChange={(e) => setKeyMode(e.target.value as "major" | "minor")} style={{ height: 26 }}>
            <option value="major">major</option>
            <option value="minor">minor</option>
          </select>
          <button className="btn sm" disabled={!notes.length} title="Pitch correction: move out-of-key notes to the nearest scale note" onClick={() => s.edit(tid, (cur) => snapToScale(cur, target, keyPc, keyMode))}>
            Snap to key
          </button>
        </>
      )}
      <label className="row" style={{ gap: 4 }}>
        <span className="faint" style={{ fontSize: 12 }}>Vel</span>
        <input
          type="number" min={1} max={127} disabled={!sel.size} aria-label="Velocity of selected notes"
          value={avgVel ?? ""} placeholder="–" style={{ width: 58, height: 26 }}
          onChange={(e) => { const v = Number(e.target.value); if (v >= 1 && v <= 127) s.edit(tid, (cur) => setVelocity(cur, sel, v), { coalesce: "vel-input" }); }}
        />
      </label>
      <button className="btn sm danger" disabled={!sel.size} onClick={() => { s.edit(tid, (cur) => deleteNotes(cur, sel)); s.setSelection(new Set()); }} title="Delete selected (Del)">Delete</button>
      <span className="spacer" />
      <label className="check" style={{ fontSize: 12 }} title="Shade notes by model confidence; dashed outline = below threshold; white outline = drawn by you">
        <input type="checkbox" checked={s.showConfidence} onChange={(e) => s.set({ showConfidence: e.target.checked })} />
        Confidence
      </label>
      <input type="range" min={0} max={0.9} step={0.05} value={s.confThreshold} aria-label="Low-confidence threshold" title="Low-confidence threshold" onChange={(e) => s.set({ confThreshold: Number(e.target.value) })} style={{ width: 70 }} />
      <span className="num faint" style={{ fontSize: 11, width: 28 }}>{s.confThreshold.toFixed(2)}</span>
      {!track.is_drum && (
        <label className="check" style={{ fontSize: 12 }} title="Constant-Q spectrogram of this stem, one row per semitone">
          <input type="checkbox" checked={s.showSpectrogram} onChange={(e) => s.set({ showSpectrogram: e.target.checked })} />
          Spectrogram
        </label>
      )}
      <button className="btn sm" onClick={fit} title="Fit notes vertically">Fit</button>
    </div>
  );
}
