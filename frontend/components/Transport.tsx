"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { BrandMark } from "./Brand";
import { useWorkspace } from "./Workspace";
import { api, errorMessage } from "@/lib/api";
import { fmtTime } from "@/lib/format";
import { DIVISIONS, secondsPerBeat } from "@/lib/notes";
import { useEditor } from "@/lib/store";

export function Transport() {
  const { engine, project, openExport, openShortcuts, duration } = useWorkspace();
  const st = useEditor();
  const clockRef = useRef<HTMLSpanElement>(null);
  const barRef = useRef<HTMLSpanElement>(null);
  const [playing, setPlaying] = useState(false);
  const [tempoText, setTempoText] = useState(String((project.tempo_bpm ?? project.analysis?.tempo ?? 120).toFixed(2)));
  const [name, setName] = useState(project.name);

  useEffect(() => {
    let raf = 0;
    const loop = () => {
      const t = engine.currentTime;
      const g = useEditor.getState().grid();
      if (clockRef.current) clockRef.current.textContent = fmtTime(t + project.offset_seconds, 2);
      if (barRef.current) {
        const beats = (t - g.origin) / secondsPerBeat(g);
        const bar = Math.floor(beats / g.beatsPerBar) + 1;
        const beat = Math.floor(((beats % g.beatsPerBar) + g.beatsPerBar) % g.beatsPerBar) + 1;
        barRef.current.textContent = beats < 0 ? "pickup" : `${bar}.${beat}`;
      }
      setPlaying((p) => (p !== engine.isPlaying ? engine.isPlaying : p));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [engine, project.offset_seconds]);

  async function saveProject(body: { tempo_bpm?: number; time_signature?: string; name?: string }) {
    try {
      const p = await api.patchProject(project.id, body);
      st.setProject(p);
    } catch (e) {
      st.toast("error", errorMessage(e));
    }
  }

  function commitTempo(v: number) {
    if (!(v >= 20 && v <= 400)) {
      setTempoText(String(project.tempo_bpm ?? 120));
      return st.toast("error", "Tempo must be between 20 and 400 BPM");
    }
    setTempoText(v.toFixed(2));
    void saveProject({ tempo_bpm: Math.round(v * 100) / 100 });
  }

  const saveLabel = { saved: "All saved", pending: "Unsaved…", saving: "Saving…", error: "Save failed", conflict: "Conflict" }[st.saveState];
  const tempo = project.tempo_bpm ?? project.analysis?.tempo ?? 120;
  void st.historyTick; // re-render on history changes so undo/redo enablement is current

  return (
    <div className="topbar" role="toolbar" aria-label="Transport and project controls">
      <div className="group">
        <Link href="/" aria-label="Back to projects" className="btn ghost icon"><BrandMark size={22} /></Link>
        <input
          type="text"
          aria-label="Project name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => name.trim() && name !== project.name && saveProject({ name: name.trim() })}
          style={{ width: 170, fontWeight: 600 }}
        />
      </div>
      <div className="group">
        <button className="btn icon" aria-label="Back to start" title="Back to start (Enter)" onClick={() => engine.stop()}>⏮</button>
        <button
          className="btn primary icon"
          aria-label={playing ? "Pause" : "Play"}
          title="Play / pause (Space)"
          onClick={() => (engine.isPlaying ? engine.pause() : void engine.play())}
        >
          {playing ? "❚❚" : "▶"}
        </button>
        <span className="clock" aria-label="Playback position (original timeline)">
          <span ref={clockRef}>0:00.00</span>
          <small ref={barRef}>1.1</small>
        </span>
        <button
          className="btn sm"
          aria-pressed={!!st.loop}
          title="Loop: drag on the ruler to set a region, or press L with notes selected"
          onClick={() => st.set({ loop: st.loop ? null : { start: Math.max(0, engine.currentTime), end: Math.min(duration, engine.currentTime + 4 * secondsPerBeat(st.grid()) * st.grid().beatsPerBar) } })}
        >
          Loop
        </button>
        <button
          className="btn sm"
          aria-pressed={st.compare === "original"}
          title="A/B: hear the original mix instead of the stems (C)"
          onClick={() => st.set({ compare: st.compare === "original" ? "stems" : "original" })}
        >
          {st.compare === "original" ? "A: Original" : "B: Stems"}
        </button>
      </div>
      <div className="group">
        <label className="row" style={{ gap: 4 }}>
          <span className="faint" style={{ fontSize: 12 }}>BPM</span>
          <input
            className="tempo-input"
            type="number"
            min={20}
            max={400}
            step={0.01}
            aria-label="Tempo (BPM)"
            value={tempoText}
            onChange={(e) => setTempoText(e.target.value)}
            onBlur={() => commitTempo(Number(tempoText))}
            onKeyDown={(e) => e.key === "Enter" && commitTempo(Number(tempoText))}
          />
        </label>
        <button className="btn sm" title="Halve tempo (beat tracker picked double time)" onClick={() => commitTempo(tempo / 2)}>÷2</button>
        <button className="btn sm" title="Double tempo (beat tracker picked half time)" onClick={() => commitTempo(tempo * 2)}>×2</button>
        <select aria-label="Time signature" value={project.time_signature} onChange={(e) => saveProject({ time_signature: e.target.value })}>
          {["2/4", "3/4", "4/4", "5/4", "6/8", "7/8", "12/8"].map((t) => <option key={t}>{t}</option>)}
        </select>
      </div>
      <div className="group">
        <label className="row" style={{ gap: 4 }}>
          <span className="faint" style={{ fontSize: 12 }}>Snap</span>
          <select aria-label="Grid snap" value={st.division} onChange={(e) => st.set({ division: Number(e.target.value) })}>
            {DIVISIONS.map((d) => <option key={d.label} value={d.value}>{d.label}</option>)}
          </select>
        </label>
        <button className="btn sm icon" aria-label="Zoom out" title="Zoom out (-)" onClick={() => st.set({ pxPerSec: Math.max(8, st.pxPerSec / 1.25) })}>−</button>
        <button className="btn sm icon" aria-label="Zoom in" title="Zoom in (+)" onClick={() => st.set({ pxPerSec: Math.min(1200, st.pxPerSec * 1.25) })}>+</button>
      </div>
      <div className="group">
        <button className="btn sm" disabled={!st.canUndo()} onClick={st.undo} title="Undo (Ctrl+Z)">Undo</button>
        <button className="btn sm" disabled={!st.canRedo()} onClick={st.redo} title="Redo (Ctrl+Shift+Z)">Redo</button>
        <span className={`save-state ${st.saveState}`} role="status" title={st.saveError ?? undefined}>{saveLabel}</span>
      </div>
      <span className="spacer" />
      <div className="group">
        <button className="btn ghost sm" onClick={openShortcuts} aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)">?</button>
        <button className="btn primary" onClick={openExport}>Export</button>
      </div>
    </div>
  );
}
