"use client";

import { useState } from "react";
import { useWorkspace } from "./Workspace";
import { downloadFromApi, errorMessage } from "@/lib/api";
import { flushSaves, useEditor } from "@/lib/store";

export function ExportDialog({ onClose }: { onClose: () => void }) {
  const { project } = useWorkspace();
  const toast = useEditor((s) => s.toast);
  const [stems, setStems] = useState<Set<string>>(new Set(project.tracks.map((t) => t.stem)));
  const [includeStems, setIncludeStems] = useState(true);
  const [includeMidi, setIncludeMidi] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run(label: string, path: string, name: string) {
    setBusy(label);
    setErr(null);
    try {
      await flushSaves();
      await downloadFromApi(path, name);
      toast("success", `${label} ready`);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setBusy(null);
    }
  }

  const stemParam = [...stems].join(",");
  const transcribed = project.tracks.filter((t) => stems.has(t.stem) && t.transcription_status === "done");

  return (
    <div className="backdrop" role="dialog" aria-modal="true" aria-labelledby="export-title" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" onKeyDown={(e) => e.key === "Escape" && onClose()}>
        <header>
          <h2 id="export-title">Export</h2>
          <button className="btn ghost sm icon" aria-label="Close" onClick={onClose}>×</button>
        </header>
        <button
          className="btn primary"
          disabled={!!busy}
          onClick={() => run("Full-song export", `/projects/${project.id}/export/bundle`, "export.zip")}
        >
          {busy === "Full-song export" ? "Preparing…" : "Export everything (all stems + all MIDI)"}
        </button>
        <div className="faint" style={{ fontSize: 12 }}>
          Stems are 24-bit FLAC, sample-aligned to the original. MIDI is rendered from your current edits and verified by re-parsing.
          {project.offset_seconds > 0 && ` MIDI events are offset by ${project.offset_seconds.toFixed(2)} s so they line up with the original file.`}
        </div>
        <h3>Selective export</h3>
        <div className="stack" style={{ gap: 6 }}>
          {project.tracks.map((t) => (
            <label key={t.id} className="check">
              <input
                type="checkbox"
                checked={stems.has(t.stem)}
                onChange={(e) => {
                  const n = new Set(stems);
                  if (e.target.checked) n.add(t.stem);
                  else n.delete(t.stem);
                  setStems(n);
                }}
              />
              {t.display_name}
              <span className="faint" style={{ fontSize: 12 }}>{t.transcription_status === "done" ? `${t.note_count} notes` : "audio only"}</span>
            </label>
          ))}
        </div>
        <div className="row wrap">
          <label className="check"><input type="checkbox" checked={includeStems} onChange={(e) => setIncludeStems(e.target.checked)} />Stem audio</label>
          <label className="check"><input type="checkbox" checked={includeMidi} onChange={(e) => setIncludeMidi(e.target.checked)} />MIDI (per track + multitrack)</label>
        </div>
        <div className="row wrap">
          <button
            className="btn"
            disabled={!!busy || !stems.size || (!includeStems && !includeMidi)}
            onClick={() => run("Selected export", `/projects/${project.id}/export/bundle?stems=${stemParam}&include_stems=${includeStems}&include_midi=${includeMidi}`, "export.zip")}
          >
            {busy === "Selected export" ? "Preparing…" : "Export selected (.zip)"}
          </button>
          <button
            className="btn"
            disabled={!!busy || !transcribed.length}
            onClick={() => run("Multitrack MIDI", `/projects/${project.id}/export/midi?stems=${transcribed.map((t) => t.stem).join(",")}`, "multitrack.mid")}
          >
            {busy === "Multitrack MIDI" ? "Preparing…" : "Multitrack MIDI only (.mid)"}
          </button>
        </div>
        <h3>Save project</h3>
        <button
          className="btn"
          disabled={!!busy}
          onClick={() => run("Project bundle", `/projects/${project.id}/export/bundle?full=true`, "project.zip")}
        >
          {busy === "Project bundle" ? "Preparing…" : "Full project bundle (restorable, includes edits)"}
        </button>
        <div className="faint" style={{ fontSize: 12 }}>Open it later from the home page with “Open project bundle…”.</div>
        {err && <div className="errbox" role="alert">{err}</div>}
      </div>
    </div>
  );
}
