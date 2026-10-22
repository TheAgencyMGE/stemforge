import type { Job, NotesDoc, Note, ProjectDetail, ProjectSummary, SystemInfo, Track, EngineId } from "./types";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string, public requestId?: string) {
    super(message);
  }
}

const TOKEN_KEY = "stemforge.apiToken";

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: token lives for this page only */
  }
}

function authHeaders(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function parseError(res: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `${res.status} ${res.statusText}`;
  let rid: string | undefined;
  try {
    const body = await res.json();
    if (body?.error) {
      code = body.error.code;
      message = body.error.message;
      rid = body.error.request_id ?? undefined;
    }
  } catch {
    /* non-JSON error body */
  }
  if (res.status === 401 && typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent("stemforge:auth-required"));
  }
  return new ApiError(res.status, code, message, rid);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { ...(init.body && typeof init.body === "string" ? { "Content-Type": "application/json" } : {}), ...authHeaders(), ...(init.headers ?? {}) },
  });
  if (!res.ok) throw await parseError(res);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export interface UploadOptions {
  name?: string;
  sixStem?: boolean;
  drumKit?: boolean;
  transcribeMix?: boolean;
  compareEngines?: boolean;
  trimStart?: number;
  trimEnd?: number | null;
}

export interface InstrumentInfo {
  id: string;
  family: string;
  name: string;
  separate: "isolated" | "shared" | "masked" | "unavailable" | string;
  transcribe: string;
  stem: string | null;
  engines: string[];
  model: string | null;
  limitation: string;
  summary: string;
  gm_program: number | null;
  gm_note: number | null;
}

export interface InstrumentCatalog {
  families: { id: string; instruments: InstrumentInfo[] }[];
  excluded_models: { id: string; task: string; reason: string }[];
}

/** Upload raw bytes with progress (XHR, because fetch has no upload progress events). */
export function uploadProject(
  file: File,
  opts: UploadOptions,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<ProjectDetail> {
  const q = new URLSearchParams();
  if (opts.name) q.set("name", opts.name);
  if (opts.sixStem) q.set("six_stem", "true");
  if (opts.drumKit) q.set("drum_kit", "true");
  if (opts.transcribeMix) q.set("transcribe_mix", "true");
  if (opts.compareEngines) q.set("compare_engines", "true");
  if (opts.trimStart) q.set("trim_start", opts.trimStart.toFixed(3));
  if (opts.trimEnd) q.set("trim_end", opts.trimEnd.toFixed(3));
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/projects?${q.toString()}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    for (const [k, v] of Object.entries(authHeaders())) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onerror = () => reject(new ApiError(0, "network", "network error while uploading; is the API running?"));
    xhr.onabort = () => reject(new ApiError(0, "aborted", "upload cancelled"));
    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as ProjectDetail);
      else {
        const err = (body as { error?: { code: string; message: string; request_id?: string } })?.error;
        if (xhr.status === 401) window.dispatchEvent(new CustomEvent("stemforge:auth-required"));
        reject(new ApiError(xhr.status, err?.code ?? "http_error", err?.message ?? `upload failed (${xhr.status})`, err?.request_id));
      }
    };
    signal?.addEventListener("abort", () => xhr.abort());
    xhr.send(file);
  });
}

export const api = {
  system: () => request<SystemInfo>("/system"),
  instruments: () => request<InstrumentCatalog>("/instruments"),
  listProjects: () => request<ProjectSummary[]>("/projects"),
  getProject: (id: string) => request<ProjectDetail>(`/projects/${id}`),
  createSample: (sixStem = false) =>
    request<ProjectDetail>("/projects/sample", { method: "POST", body: JSON.stringify({ six_stem: sixStem }) }),
  patchProject: (id: string, body: { name?: string; tempo_bpm?: number; time_signature?: string }) =>
    request<ProjectDetail>(`/projects/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteProject: (id: string) => request<void>(`/projects/${id}`, { method: "DELETE" }),
  reprocess: (id: string, body: { six_stem?: boolean; drum_kit?: boolean; transcribe_mix?: boolean; compare_engines?: boolean }) =>
    request<Job>(`/projects/${id}/reprocess`, { method: "POST", body: JSON.stringify(body) }),
  getJob: (id: string) => request<Job>(`/jobs/${id}`),
  cancelJob: (id: string) => request<Job>(`/jobs/${id}/cancel`, { method: "POST" }),
  retryJob: (id: string) => request<Job>(`/jobs/${id}/retry`, { method: "POST" }),
  getNotes: (pid: string, tid: string) => request<NotesDoc>(`/projects/${pid}/tracks/${tid}/notes`),
  putNotes: (pid: string, tid: string, baseVersion: number, notes: Note[]) =>
    request<NotesDoc>(`/projects/${pid}/tracks/${tid}/notes`, {
      method: "PUT",
      body: JSON.stringify({ base_version: baseVersion, notes }),
    }),
  resetNotes: (pid: string, tid: string) => request<NotesDoc>(`/projects/${pid}/tracks/${tid}/reset`, { method: "POST" }),
  patchTrack: (pid: string, tid: string, body: { display_name?: string; gm_program?: number }) =>
    request<Track>(`/projects/${pid}/tracks/${tid}`, { method: "PATCH", body: JSON.stringify(body) }),
  retranscribe: (pid: string, tid: string, engine: EngineId | null, params: Record<string, number | null>) =>
    request<Job>(`/projects/${pid}/tracks/${tid}/transcribe`, { method: "POST", body: JSON.stringify({ engine, params }) }),
};

/** Download an authenticated endpoint as a file (works with or without an API token). */
export async function downloadFromApi(path: string, fallbackName: string): Promise<void> {
  const res = await fetch(`/api${path}`, { headers: authHeaders() });
  if (!res.ok) throw await parseError(res);
  const blob = await res.blob();
  const cd = res.headers.get("content-disposition") ?? "";
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd);
  saveBlob(blob, m ? decodeURIComponent(m[1]) : fallbackName);
}

export function saveBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

export async function importBundle(file: File): Promise<ProjectDetail> {
  const res = await fetch("/api/projects/import", {
    method: "POST",
    headers: { "Content-Type": "application/zip", ...authHeaders() },
    body: file,
  });
  if (!res.ok) throw await parseError(res);
  return res.json();
}

export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.requestId ? `${e.message} (request ${e.requestId})` : e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}
