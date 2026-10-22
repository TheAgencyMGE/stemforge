export function fmtTime(sec: number, decimals = 1): string {
  if (!Number.isFinite(sec)) return "–";
  const sign = sec < 0 ? "-" : "";
  const s = Math.abs(sec);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  const whole = Math.floor(rest);
  const frac = decimals > 0 ? "." + Math.floor((rest - whole) * 10 ** decimals).toString().padStart(decimals, "0") : "";
  return `${sign}${m}:${whole.toString().padStart(2, "0")}${frac}`;
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

/** The API stores UTC; SQLite drops the zone suffix, so treat naive timestamps as UTC. */
export function parseUtc(iso: string): Date {
  return new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : `${iso}Z`);
}

export function fmtAgo(iso: string): string {
  const d = parseUtc(iso).getTime();
  const s = Math.round((Date.now() - d) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return parseUtc(iso).toLocaleDateString();
}

export function fmtEta(sec: number | null): string | null {
  if (sec == null || !Number.isFinite(sec)) return null;
  if (sec < 60) return `~${Math.max(1, Math.round(sec))} s left in this stage`;
  return `~${Math.round(sec / 60)} min left in this stage`;
}

export const STEM_COLOR: Record<string, string> = {
  vocals: "var(--stem-vocals)",
  drums: "var(--stem-drums)",
  bass: "var(--stem-bass)",
  guitar: "var(--stem-guitar)",
  piano: "var(--stem-piano)",
  other: "var(--stem-other)",
  original: "var(--stem-original)",
  mix: "var(--stem-original)",
  kit_kick: "var(--stem-drums)",
  kit_snare: "var(--stem-drums)",
  kit_hats: "var(--stem-drums)",
  kit_perc: "var(--stem-drums)",
};

/** Hex versions for canvas drawing (CSS variables are not resolvable inside canvas). */
export const STEM_HEX: Record<string, string> = {
  vocals: "#f2c14e",
  drums: "#ef6f6c",
  bass: "#6c9cff",
  guitar: "#7bd389",
  piano: "#c792ea",
  other: "#4fd1c5",
  original: "#9aa3ad",
  mix: "#9aa3ad",
  kit_kick: "#ef6f6c",
  kit_snare: "#ef6f6c",
  kit_hats: "#ef6f6c",
  kit_perc: "#e0a090",
};
