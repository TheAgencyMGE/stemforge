// Mirrors backend/stemforge/schemas.py (the OpenAPI contract at /api/openapi.json).

export interface Note {
  id: string;
  pitch: number;
  start: number; // seconds, relative to the processed region
  end: number;
  velocity: number;
  /** Model confidence 0..1, or null for notes the user drew. */
  confidence: number | null;
  /** [seconds from note start, semitones] */
  bends: [number, number][] | null;
  /** Pitch before an octave correction. Absent when the engine's pitch was kept. */
  raw_pitch?: number | null;
  flags?: string[] | null;
}

export interface NotesDoc {
  version: number;
  notes: Note[];
}

export interface Artifact {
  id: string;
  kind: "master" | "preview" | "stem" | "peaks" | "spectrogram" | "midi" | "midi_combined" | "residual" | "discovery" | "midi_render";
  stem: string | null;
  filename: string;
  content_type: string;
  size: number;
  url: string;
  meta: Record<string, unknown>;
}

export interface SourceCandidate {
  id: string;
  name: string;
  score: number;
}

/** What discovery established about one source (backend: pipeline/discovery.py). */
export interface SourceInfo {
  status: "accepted" | "unresolved";
  identity: string | null;
  family: string;
  display: string;
  /** single = one instrument named; group = several, not separable; unresolved = no confident name. */
  kind: "single" | "group" | "unresolved";
  /** Tag probability that supports the identity. null when the tagger is not installed. */
  confidence: number | null;
  family_score: number | null;
  candidates: SourceCandidate[];
  contains: SourceCandidate[];
  activity: {
    active_seconds: number;
    active_fraction: number;
    first_active: number | null;
    last_active: number | null;
    median_active_rel_db: number | null;
    segments: [number, number][];
  };
  /** How the audio was obtained, e.g. "htdemucs:other" or an HPSS / ratio-mask description. */
  separation: string;
  /** broad = a separator stem; targeted = split from a parent; residual/group/unresolved = what remained. */
  isolation: "broad" | "targeted" | "residual" | "group" | "unresolved";
  limitations: string[];
  absorbed?: { from: string; rel_db: number }[];
}

export interface TrackStats {
  rms_db: number;
  peak_db: number;
  clipped_ratio: number;
  relative_db?: number;
  silent?: boolean;
  mix_residual_db?: number;
  source?: SourceInfo;
  /** The track's MIDI rendered through a GM soundfont, compared with its own stem (backend render.py). */
  render_fidelity?: { chroma_similarity: number | null; onset_correlation: number | null; logmel_l1_db: number | null };
}

export interface DiscoveryLogEntry {
  step: string;
  [k: string]: unknown;
}

export interface Track {
  id: string;
  stem: string;
  display_name: string;
  order_index: number;
  source_model: string;
  is_drum: boolean;
  gm_program: number;
  stats: TrackStats;
  warnings: string[];
  engine: EngineId | null;
  engine_reason: string | null;
  engine_params: Record<string, number | null>;
  notes_version: number;
  note_count: number;
  edited: boolean;
  transcription_status: "pending" | "done" | "skipped" | "failed";
  /** Engines the server accepts for this track (installed ones only), in preference order. */
  allowed_engines?: EngineId[];
  transcription_summary: {
    count?: number;
    mean_confidence?: number | null;
    low_confidence?: number;
    pitch_range?: [number, number];
    detail?: string;
    coverage?: string;
    suggestions?: { code: string; severity: "review" | "info" | string; message: string; note_ids?: string[] }[];
    tablature?: {
      estimate: boolean;
      tuning: string;
      disclaimer: string;
      assigned: number;
      unassigned: number;
    };
  } | null;
}

export interface StageState {
  status: "running" | "done" | "skipped" | "failed";
  progress: number;
  message?: string;
  seconds?: number;
  [k: string]: unknown;
}

export interface Job {
  id: string;
  project_id: string;
  kind: "process" | "retranscribe";
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  stage: string | null;
  progress: number;
  eta_seconds: number | null;
  stages: Record<string, StageState>;
  stage_order: string[];
  message: string | null;
  error: string | null;
  attempts: number;
  max_attempts: number;
  params: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface KeyInfo {
  tonic: string;
  tonic_pc: number;
  mode: "major" | "minor";
  label: string;
  correlation: number;
  margin: number;
  runner_up: string;
}

export interface Analysis {
  duration: number;
  sample_rate: number;
  rms_db: number;
  peak_db: number;
  clipped_ratio: number;
  tempo: number | null;
  tempo_confidence?: number;
  beats: number[];
  first_beat?: number;
  key: KeyInfo | null;
  sections: { start: number; end: number; label: string }[];
  silent: boolean;
  discovery?: {
    version: number;
    tagger: string;
    log: DiscoveryLogEntry[];
    reconstruction: { residual_db: number; unexplained_by_tracks_db: number; internal_residual_db: number | null; tracks: number };
  };
  meter?: {
    time_signature: string | null;
    beats_per_bar: number | null;
    confidence: number;
    downbeat: number | null;
  };
}

export interface ProjectSummary {
  id: string;
  name: string;
  status: "queued" | "processing" | "ready" | "failed" | "cancelled";
  original_filename: string;
  duration: number | null;
  tempo: number | null;
  key: string | null;
  stems: number;
  is_sample: boolean;
  error: string | null;
  created_at: string;
  expires_at: string | null;
}

export interface ProjectDetail extends ProjectSummary {
  options: {
    six_stem?: boolean;
    drum_kit?: boolean;
    transcribe_mix?: boolean;
    compare_engines?: boolean;
    trim_start?: number;
    trim_end?: number | null;
    source_duration?: number;
    [k: string]: unknown;
  };
  analysis: Analysis | null;
  tempo_bpm: number | null;
  time_signature: string;
  offset_seconds: number;
  tracks: Track[];
  artifacts: Artifact[];
  job: Job | null;
}

export type EngineId = "basic-pitch" | "pyin" | "drum-nmf" | "onset-events" | "transkun" | "yourmt3" | "muscriptor";

export interface EngineInfo {
  id: EngineId;
  label: string;
  polyphonic: boolean;
  description: string;
  defaults: Record<string, number | null>;
}

export interface SystemInfo {
  version: string;
  device: string;
  device_name: string | null;
  models: { name: string; installed: boolean; sources: string[]; size_bytes: number; license: string; caveat: string | null }[];
  engines: EngineInfo[];
  formats: string[];
  m4a_supported: boolean;
  max_upload_mb: number;
  max_duration_seconds: number;
  retention_days: number;
  auth_required: boolean;
  queue: Record<string, number>;
}

export interface Peaks {
  sample_rate: number;
  samples_per_bucket: number;
  min: number[];
  max: number[];
}
