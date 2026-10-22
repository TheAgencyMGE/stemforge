import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushSaves, useEditor } from "@/lib/store";
import type { Note, ProjectDetail } from "@/lib/types";

const note = (id: string, pitch = 60): Note => ({ id, pitch, start: 0, end: 1, velocity: 90, confidence: 0.9, bends: null });

const project = {
  id: "p1", name: "P", status: "ready", original_filename: "a.wav", duration: 10, tempo: 120, key: null, stems: 1,
  is_sample: false, error: null, created_at: "2026-01-01T00:00:00", expires_at: null, options: {},
  analysis: { duration: 10, sample_rate: 44100, rms_db: -20, peak_db: -1, clipped_ratio: 0, tempo: 120, beats: [], first_beat: 0.5, key: null, sections: [], silent: false },
  tempo_bpm: 100, time_signature: "3/4", offset_seconds: 0, artifacts: [], job: null,
  tracks: [{ id: "t1", stem: "bass", display_name: "Bass", order_index: 0, source_model: "demucs:htdemucs", is_drum: false, gm_program: 33,
    stats: { rms_db: -20, peak_db: -3, clipped_ratio: 0 }, warnings: [], engine: "basic-pitch", engine_reason: "x", engine_params: {},
    notes_version: 1, note_count: 1, edited: false, transcription_status: "done", transcription_summary: null }],
} as unknown as ProjectDetail;

describe("editor store", () => {
  beforeEach(() => {
    useEditor.setState({ project: null, notes: {}, versions: {}, dirty: {}, saveState: "saved", selectedTrackId: null });
    useEditor.getState().setProject(project);
    useEditor.getState().setTrackNotes("t1", [note("a")], 1);
  });

  it("derives the grid from project tempo, meter and beat phase", () => {
    const g = useEditor.getState().grid();
    expect(g.bpm).toBe(100);
    expect(g.beatsPerBar).toBe(3);
    expect(g.origin).toBe(0.5);
  });

  it("selects the first transcribed track and mutes the original lane by default", () => {
    const s = useEditor.getState();
    expect(s.selectedTrackId).toBe("t1");
    expect(s.mix.original.muted).toBe(true);
    expect(s.mix.t1.muted).toBe(false);
  });

  it("records undo history and marks tracks dirty", () => {
    const s = useEditor.getState();
    s.edit("t1", (ns) => [...ns, note("b", 62)]);
    expect(useEditor.getState().notes.t1).toHaveLength(2);
    expect(useEditor.getState().dirty.t1).toBe(true);
    useEditor.getState().undo();
    expect(useEditor.getState().notes.t1).toHaveLength(1);
    useEditor.getState().redo();
    expect(useEditor.getState().notes.t1).toHaveLength(2);
  });

  it("coalesces drag edits into one undo step", () => {
    const s = useEditor.getState();
    s.edit("t1", (ns) => ns.map((x) => ({ ...x, pitch: 61 })), { coalesce: "drag" });
    s.edit("t1", (ns) => ns.map((x) => ({ ...x, pitch: 62 })), { coalesce: "drag" });
    s.edit("t1", (ns) => ns.map((x) => ({ ...x, pitch: 63 })), { coalesce: "drag" });
    useEditor.getState().undo();
    expect(useEditor.getState().notes.t1[0].pitch).toBe(60);
  });

  it("saves dirty tracks with the base version and handles conflicts", async () => {
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ version: 2, notes: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    useEditor.getState().edit("t1", (ns) => [...ns, note("c", 64)]);
    await flushSaves();
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.base_version).toBe(1);
    expect(body.notes).toHaveLength(2);
    expect(useEditor.getState().versions.t1).toBe(2);
    expect(useEditor.getState().saveState).toBe("saved");

    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ error: { code: "version_conflict", message: "changed" } }), { status: 409 }));
    useEditor.getState().edit("t1", (ns) => ns.slice(0, 1));
    await expect(flushSaves()).rejects.toThrow("changed");
    expect(useEditor.getState().saveState).toBe("conflict");
    vi.unstubAllGlobals();
  });
});
