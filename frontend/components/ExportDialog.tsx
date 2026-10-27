"use client";

import { useMemo, useState } from "react";
import { useWorkspace } from "./Workspace";
import { downloadFromApi, errorMessage } from "@/lib/api";
import { audibleStems, exportedByDefault } from "@/lib/exports";
import { flushSaves, useEditor } from "@/lib/store";

export function ExportDialog({ onClose }: { onClose: () => void }) {
  const { project } = useWorkspace();
  const toast = useEditor((s) => s.toast);
  const mix = useEditor((s) => s.mix);
  const [stems, setStems] = useState<Set<string>>(
    () => new Set(project.tracks.filter((t) => exportedByDefault(t) === null).map((t) => t.stem)),
  );
  const [mode, setMode] = useState<"synced" | "trimmed">("synced");
  const [includeStems, setIncludeStems] = useState(true);
  const [includeMidi, setIncludeMidi] = useState(true);
  const [includeMix, setIncludeMix] = useState(false);
  const [tempoMap, setTempoMap] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // A lane is heard when it is not muted and, if anything is soloed, it is soloed too.
  const audible = useMemo(() => audibleStems(project.tracks, mix), [mix, project.tracks]);

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

  const selected = project.tracks.filter((t) => stems.has(t.stem));
  const mixStems = selected.filter((t) => audible.has(t.stem) && t.stem !== "mix").map((t) => t.stem);
  const transcribed = selected.filter((t) => t.transcription_status === "done");
  const recon = project.analysis?.discovery?.reconstruction;
  const query = new URLSearchParams({
    stems: [...stems].join(","), include_stems: String(includeStems), include_midi: String(includeMidi), mode,
  });
  if (includeMix && mixStems.length) query.set("mix", mixStems.join(","));
  if (tempoMap) query.set("tempo_map", "true");

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
          {busy === "Full-song export" ? "Preparing…" : "Export every detected source (FLAC + MIDI, DAW-aligned)"}
        </button>
        <div className="faint" style={{ fontSize: 12 }}>
          Every source detected as audible is included (unresolved sources too, marked in the manifest). Every FLAC is 24-bit, full length and starts at the same instant, so the files line up when placed at 0:00 in a DAW.
          {project.offset_seconds > 0 && ` Audio and MIDI are placed on the original file's timeline (offset ${project.offset_seconds.toFixed(2)} s).`}
        </div>

        <h3>Selective export</h3>
        <div className="stack" style={{ gap: 6 }}>
          {project.tracks.map((t) => {
            const why = exportedByDefault(t);
            return (
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
                <span className="faint" style={{ fontSize: 12 }}>
                  {t.transcription_status === "done" ? `${t.note_count} notes` : "audio only"}
                  {why && ` · off by default: ${why}`}
                  {!audible.has(t.stem) && " · muted in the mixer"}
                </span>
              </label>
            );
          })}
        </div>

        <fieldset className="stack" style={{ gap: 4, border: 0, padding: 0, margin: 0 }}>
          <legend className="faint" style={{ fontSize: 12 }}>Audio timeline</legend>
          <label className="check">
            <input type="radio" name="mode" checked={mode === "synced"} onChange={() => setMode("synced")} />
            Full timeline, synchronized
            <span className="faint" style={{ fontSize: 12 }}>every file is the full song length; for multitrack/DAW use</span>
          </label>
          <label className="check">
            <input type="radio" name="mode" checked={mode === "trimmed"} onChange={() => setMode("trimmed")} />
            Trimmed individual files
            <span className="faint" style={{ fontSize: 12 }}>leading/trailing silence removed; each start time is in export_manifest.json. Silence inside the music is never cut.</span>
          </label>
        </fieldset>

        <div className="row wrap">
          <label className="check"><input type="checkbox" checked={includeStems} onChange={(e) => setIncludeStems(e.target.checked)} />Source audio (FLAC)</label>
          <label className="check"><input type="checkbox" checked={includeMidi} onChange={(e) => setIncludeMidi(e.target.checked)} />MIDI (per track + multitrack)</label>
          <label className="check" title="Sum of the selected tracks that are audible in the mixer. Muted tracks stay out of the mix but their own files are still exported if selected.">
            <input type="checkbox" checked={includeMix} onChange={(e) => setIncludeMix(e.target.checked)} />
            Rendered mix of unmuted tracks ({mixStems.length})
          </label>
          <label className="check" title="A tempo change on every detected beat, so bar lines follow a song that speeds up or slows down. Note timing is the same either way. Soundtrap uses one project tempo and ignores tempo maps.">
            <input type="checkbox" checked={tempoMap} onChange={(e) => setTempoMap(e.target.checked)} />
            MIDI tempo map from detected beats (DAWs that support it)
          </label>
        </div>
        <div className="row wrap">
          <button
            className="btn"
            disabled={!!busy || !stems.size || (!includeStems && !includeMidi && !(includeMix && mixStems.length))}
            onClick={() => run("Selected export", `/projects/${project.id}/export/bundle?${query.toString()}`, "export.zip")}
          >
            {busy === "Selected export" ? "Preparing…" : "Export selected (.zip)"}
          </button>
          <button
            className="btn"
            disabled={!!busy || !transcribed.length}
            onClick={() => run("Multitrack MIDI", `/projects/${project.id}/export/midi?stems=${transcribed.map((t) => t.stem).join(",")}${tempoMap ? "&tempo_map=true" : ""}`, "multitrack.mid")}
          >
            {busy === "Multitrack MIDI" ? "Preparing…" : "Multitrack MIDI only (.mid)"}
          </button>
        </div>
        <h3>Rebuild in Soundtrap</h3>
        <button
          className="btn"
          disabled={!!busy || !stems.size}
          onClick={() => run("Soundtrap package", `/projects/${project.id}/export/soundtrap?stems=${[...stems].join(",")}`, "soundtrap.zip")}
        >
          {busy === "Soundtrap package" ? "Preparing…" : "Soundtrap package (.zip)"}
        </button>
        <div className="faint" style={{ fontSize: 12 }}>
          One MIDI file per track at a whole-number tempo you can type into Soundtrap, full-length 24-bit WAV stems
          (Soundtrap does not import FLAC), and step-by-step instructions with an instrument suggestion per track.
          The MIDI carries the notes, not the original sound, effects or mix.
        </div>
        {recon && (
          <div className="reason" style={{ fontSize: 12 }}>
            <strong>What the files guarantee.</strong> FLAC is lossless: each file decodes to exactly the samples Stemforge produced (checked on export).
            Separation is an estimate, not the studio multitrack. All sources together reconstruct the original mix to {recon.residual_db} dB
            {recon.internal_residual_db != null && `; content that matched no source (${recon.internal_residual_db} dB) is not exported as a track`}.
            MIDI is a transcription and is checked for structure and timing, not musical correctness.
          </div>
        )}

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
