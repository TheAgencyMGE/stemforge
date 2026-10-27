"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { api, errorMessage, uploadProject } from "@/lib/api";
import { fmtBytes, fmtTime } from "@/lib/format";
import type { SystemInfo } from "@/lib/types";

const EXT_TO_FORMAT: Record<string, string> = {
  wav: "wav", wave: "wav", flac: "flac", mp3: "mp3", ogg: "ogg", oga: "ogg", opus: "ogg",
  aif: "aiff", aiff: "aiff", m4a: "m4a", aac: "m4a", mp4: "m4a", webm: "webm",
};

interface Picked {
  file: File;
  duration: number | null;
  peaks: Float32Array | null;
  decodeNote: string | null;
}

async function inspectLocally(file: File): Promise<Pick<Picked, "duration" | "peaks" | "decodeNote">> {
  if (file.size > 120 * 1024 * 1024) {
    return { duration: null, peaks: null, decodeNote: "Large file: preview skipped; the server will validate it." };
  }
  try {
    const ctx = new OfflineAudioContext(1, 1, 44100);
    const buf = await ctx.decodeAudioData(await file.arrayBuffer());
    const ch = buf.getChannelData(0);
    const n = 600;
    const step = Math.max(1, Math.floor(ch.length / n));
    const peaks = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let m = 0;
      for (let j = i * step; j < Math.min(ch.length, (i + 1) * step); j += 8) m = Math.max(m, Math.abs(ch[j]));
      peaks[i] = m;
    }
    return { duration: buf.duration, peaks, decodeNote: null };
  } catch {
    return { duration: null, peaks: null, decodeNote: "Your browser can't preview this file; the server will still try to decode it." };
  }
}

export function UploadPanel({ system }: { system: SystemInfo | null }) {
  const router = useRouter();
  const [over, setOver] = useState(false);
  const [picked, setPicked] = useState<Picked | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [sixStem, setSixStem] = useState(false);
  const [drumKit, setDrumKit] = useState(false);
  const [transcribeMix, setTranscribeMix] = useState(false);
  const [compareEngines, setCompareEngines] = useState(false);
  const [trim, setTrim] = useState<[number, number] | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const maxMb = system?.max_upload_mb ?? 200;
  const maxDur = system?.max_duration_seconds ?? 900;
  const formats = useMemo(() => system?.formats ?? ["wav", "flac", "mp3", "ogg", "aiff"], [system]);

  const choose = useCallback(
    async (file: File) => {
      setError(null);
      setTrim(null);
      const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
      const fmt = EXT_TO_FORMAT[ext];
      if (!fmt) {
        setError(`“.${ext}” doesn't look like an audio file. Supported: ${formats.map((f) => f.toUpperCase()).join(", ")}.`);
        return;
      }
      if (!formats.includes(fmt)) {
        setError(`${fmt.toUpperCase()} needs FFmpeg, which this server doesn't have. Convert to WAV, FLAC or MP3 first.`);
        return;
      }
      if (file.size > maxMb * 1024 * 1024) {
        setError(`File is ${fmtBytes(file.size)}; the limit is ${maxMb} MB.`);
        return;
      }
      if (file.size === 0) {
        setError("That file is empty.");
        return;
      }
      setName(file.name.replace(/\.[^.]+$/, "").replace(/[_]+/g, " "));
      setPicked({ file, duration: null, peaks: null, decodeNote: "Reading audio…" });
      const info = await inspectLocally(file);
      setPicked({ file, ...info });
      if (info.duration && info.duration > maxDur) {
        setTrim([0, maxDur]);
      }
    },
    [formats, maxDur, maxMb],
  );

  const tooLong = !!picked?.duration && (trim ? trim[1] - trim[0] : picked.duration) > maxDur;

  async function submit() {
    if (!picked) return;
    setBusy(true);
    setError(null);
    setProgress(0);
    abortRef.current = new AbortController();
    try {
      const p = await uploadProject(
        picked.file,
        {
          name: name.trim() || undefined, sixStem, drumKit, transcribeMix, compareEngines,
          trimStart: trim?.[0], trimEnd: trim && picked.duration && trim[1] < picked.duration - 0.01 ? trim[1] : null,
        },
        (f) => setProgress(f),
        abortRef.current.signal,
      );
      router.push(`/projects/${p.id}`);
    } catch (e) {
      setError(errorMessage(e));
      setBusy(false);
      setProgress(null);
    }
  }

  return (
    <div
      className={`dropzone card ${over ? "over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f && !busy) void choose(f);
      }}
      aria-label="Upload area"
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*,.flac,.wav,.mp3,.ogg,.opus,.aif,.aiff,.m4a"
        className="sr-only"
        aria-label="Choose an audio file"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void choose(f);
          e.target.value = "";
        }}
      />
      {!picked ? (
        <>
          <div className="big">Drop a song here</div>
          <div className="muted">
            {formats.map((f) => f.toUpperCase()).join(" · ")}. Up to {maxMb} MB and {Math.round(maxDur / 60)} min (trim longer files
            to a section).
          </div>
          <div className="row">
            <button className="btn primary" onClick={() => inputRef.current?.click()}>
              Choose file
            </button>
          </div>
        </>
      ) : (
        <div className="stack">
          <div className="row">
            <div style={{ minWidth: 0 }}>
              <div className="big" style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{picked.file.name}</div>
              <div className="faint num">
                {fmtBytes(picked.file.size)}
                {picked.duration ? ` · ${fmtTime(picked.duration)}` : ""}
              </div>
            </div>
            <span className="spacer" />
            {!busy && (
              <button className="btn ghost sm" onClick={() => setPicked(null)}>
                Change
              </button>
            )}
          </div>
          {picked.decodeNote && <div className="faint">{picked.decodeNote}</div>}
          {picked.peaks && picked.duration && (
            <TrimSelector peaks={picked.peaks} duration={picked.duration} trim={trim} onChange={setTrim} />
          )}
          <div className="row wrap" style={{ gap: 14 }}>
            <label className="field" style={{ flex: 1, minWidth: 180 }}>
              Project name
              <input type="text" value={name} maxLength={200} onChange={(e) => setName(e.target.value)} />
            </label>
            <label className="field">
              Section start (s)
              <input
                type="number" min={0} step={0.1} style={{ width: 100 }}
                value={trim ? trim[0].toFixed(1) : "0.0"}
                onChange={(e) => {
                  const v = Math.max(0, Number(e.target.value) || 0);
                  setTrim([v, trim?.[1] ?? picked.duration ?? v + 30]);
                }}
              />
            </label>
            <label className="field">
              Section end (s)
              <input
                type="number" min={0} step={0.1} style={{ width: 100 }}
                value={trim ? trim[1].toFixed(1) : picked.duration ? picked.duration.toFixed(1) : ""}
                placeholder="end"
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (v > 0) setTrim([trim?.[0] ?? 0, v]);
                }}
              />
            </label>
          </div>
          <label className="check">
            <input type="checkbox" checked={sixStem} onChange={(e) => setSixStem(e.target.checked)} />
            <span>
              Always try guitar and piano extraction (htdemucs_6s)
              <span className="faint"> · it already runs automatically when guitar or piano is heard; a part is kept only if its own tags confirm it</span>
            </span>
          </label>
          <details>
            <summary className="faint" style={{ cursor: "pointer", fontSize: 13 }}>More separation and transcription options</summary>
            <div className="stack" style={{ gap: 8, marginTop: 8 }}>
              <label className="check">
                <input type="checkbox" checked={drumKit} onChange={(e) => setDrumKit(e.target.checked)} />
                <span>
                  Split the drum stem into kick, snare, hat and residual masks
                  <span className="faint"> · listening only. MIDI stays on the Drums track. Not a second neural model.</span>
                </span>
              </label>
              <label className="check">
                <input type="checkbox" checked={transcribeMix} onChange={(e) => setTranscribeMix(e.target.checked)} />
                <span>
                  Also transcribe the full mix
                  <span className="faint"> · one Basic Pitch track of everything at once, for comparison.</span>
                </span>
              </label>
              <label className="check">
                <input type="checkbox" checked={compareEngines} onChange={(e) => setCompareEngines(e.target.checked)} />
                <span>
                  Compare a second engine on bass and vocals
                  <span className="faint"> · disagreements are listed. Notes are not averaged or deleted. Slower.</span>
                </span>
              </label>
            </div>
          </details>
          {tooLong && <div className="warnbox">The selection is longer than {Math.round(maxDur / 60)} min. Drag on the waveform to choose a section.</div>}
          {progress !== null && (
            <div className="stack" style={{ gap: 4 }}>
              <div className="progress" aria-label="Upload progress"><div style={{ width: `${Math.round(progress * 100)}%` }} /></div>
              <div className="faint num">{progress < 1 ? `Uploading ${Math.round(progress * 100)}%` : "Validating on the server…"}</div>
            </div>
          )}
          <div className="row">
            {busy && progress !== null && progress < 1 && (
              <button className="btn ghost" onClick={() => abortRef.current?.abort()}>
                Cancel upload
              </button>
            )}
            <span className="spacer" />
            <button className="btn primary" disabled={busy || tooLong} onClick={submit}>
              {busy ? "Uploading…" : trim ? "Process section" : "Process song"}
            </button>
          </div>
        </div>
      )}
      {error && <div className="errbox" role="alert">{error}</div>}
    </div>
  );
}

function TrimSelector({ peaks, duration, trim, onChange }: {
  peaks: Float32Array; duration: number; trim: [number, number] | null; onChange: (t: [number, number] | null) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drag = useRef<number | null>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    c.width = w * dpr;
    c.height = h * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    g.clearRect(0, 0, w, h);
    const sel = trim ?? [0, duration];
    const x0 = (sel[0] / duration) * w, x1 = (sel[1] / duration) * w;
    g.fillStyle = "rgba(255,122,61,0.12)";
    g.fillRect(x0, 0, x1 - x0, h);
    for (let i = 0; i < peaks.length; i++) {
      const x = (i / peaks.length) * w;
      const t = (i / peaks.length) * duration;
      const a = peaks[i] * (h / 2 - 4);
      g.fillStyle = t >= sel[0] && t <= sel[1] ? "#ff9a66" : "#4a5562";
      g.fillRect(x, h / 2 - a, Math.max(1, w / peaks.length - 0.5), a * 2 || 1);
    }
    g.fillStyle = "#ffe9a8";
    g.fillRect(x0, 0, 1.5, h);
    g.fillRect(x1 - 1.5, 0, 1.5, h);
  }, [peaks, duration, trim]);
  const tAt = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return Math.max(0, Math.min(duration, ((e.clientX - r.left) / r.width) * duration));
  };
  return (
    <div className="stack" style={{ gap: 4 }}>
      <canvas
        ref={ref}
        className="trim-canvas"
        role="slider"
        aria-label="Section to process (drag to select)"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(trim?.[0] ?? 0)}
        onPointerDown={(e) => {
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          drag.current = tAt(e);
        }}
        onPointerMove={(e) => {
          if (drag.current == null) return;
          const t = tAt(e);
          const a = Math.min(drag.current, t), b = Math.max(drag.current, t);
          if (b - a > 0.5) onChange([a, b]);
        }}
        onPointerUp={() => (drag.current = null)}
      />
      <div className="row faint" style={{ fontSize: 12 }}>
        <span>Drag to process only a section</span>
        <span className="spacer" />
        {trim && (
          <>
            <span className="num">{fmtTime(trim[0])} – {fmtTime(trim[1])}</span>
            <button className="btn ghost sm" onClick={() => onChange(null)}>Whole song</button>
          </>
        )}
      </div>
    </div>
  );
}

export function SampleButton() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="stack" style={{ gap: 6 }}>
      <button
        className="btn"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          setErr(null);
          try {
            const p = await api.createSample();
            router.push(`/projects/${p.id}`);
          } catch (e) {
            setErr(errorMessage(e));
            setBusy(false);
          }
        }}
      >
        {busy ? "Starting…" : "Try the demo song"}
      </button>
      {err && <div className="errbox" role="alert">{err}</div>}
    </div>
  );
}
