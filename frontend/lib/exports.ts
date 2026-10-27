import type { Track } from "./types";

/** Why a track is left out of the default export, or null if it is included.
 * Same rule as backend exports.is_exportable_by_default: every audible source, including unresolved ones
 * (real audio, needed to rebuild the mix, flagged for review in the UI). */
export function exportedByDefault(t: Pick<Track, "stem" | "stats">): string | null {
  if (t.stats.silent) return "no audible content";
  if (t.stem.startsWith("kit_")) return "drum-stem mask";
  if (t.stem === "mix") return "MIDI of the original mix, no separate audio";
  return null;
}

/** Stems heard in the mixer: not muted, and soloed when anything is soloed (mute/solo are keyed by track id). */
export function audibleStems(
  tracks: Pick<Track, "id" | "stem">[],
  mix: Record<string, { muted: boolean; soloed: boolean } | undefined>,
): Set<string> {
  const anySolo = tracks.some((t) => mix[t.id]?.soloed);
  return new Set(
    tracks
      .filter((t) => {
        const m = mix[t.id];
        if (anySolo) return !!m?.soloed;
        return !m?.muted;
      })
      .map((t) => t.stem),
  );
}
