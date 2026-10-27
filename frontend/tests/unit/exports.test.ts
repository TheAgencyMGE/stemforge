import { describe, expect, it } from "vitest";
import { audibleStems, exportedByDefault } from "@/lib/exports";

const track = (stem: string, stats: Record<string, unknown> = {}) => ({ id: `id-${stem}`, stem, stats: { rms_db: -20, peak_db: -3, clipped_ratio: 0, ...stats } });

describe("export defaults", () => {
  it("leaves out silent tracks, kit masks and the mix transcription, keeps unresolved sources", () => {
    expect(exportedByDefault(track("drums"))).toBeNull();
    expect(exportedByDefault(track("vocals", { silent: true }))).toMatch(/no audible/);
    expect(exportedByDefault(track("other", { source: { status: "unresolved" } } as never))).toBeNull();
    expect(exportedByDefault(track("kit_kick"))).toMatch(/mask/);
    expect(exportedByDefault(track("mix"))).toMatch(/original mix/);
  });

  it("follows mute, and solo overrides mute", () => {
    const tracks = [track("drums"), track("bass"), track("piano")];
    expect([...audibleStems(tracks, { "id-bass": { muted: true, soloed: false } })].sort()).toEqual(["drums", "piano"]);
    expect([...audibleStems(tracks, { "id-piano": { muted: false, soloed: true } })]).toEqual(["piano"]);
  });
});
