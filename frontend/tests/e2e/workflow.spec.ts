import { expect, test } from "@playwright/test";
import { Midi } from "@tonejs/midi";
import fs from "node:fs";
import path from "node:path";

const SAMPLE = path.resolve(__dirname, "../../../samples/stemforge-demo.flac");

test("upload → process → edit → export with real inference", async ({ page, request }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Take any song apart" })).toBeVisible();

  // Upload the CC0 demo through the real file input.
  await page.getByLabel("Choose an audio file").setInputFiles(SAMPLE);
  await expect(page.getByText("stemforge-demo.flac")).toBeVisible();
  await page.getByLabel("Project name").fill("E2E demo");
  await page.getByRole("button", { name: "Process song" }).click();

  // Processing dashboard with real stage progress, then the workspace.
  await page.waitForURL(/\/projects\/[a-f0-9]{32}$/);
  const projectId = page.url().split("/").pop()!;
  await expect(page.getByRole("toolbar", { name: "Transport and project controls" })).toBeVisible({ timeout: 200_000 });

  const project = await (await request.get(`/api/projects/${projectId}`)).json();
  expect(project.status).toBe("ready");
  expect(project.tracks.map((t: { stem: string }) => t.stem).sort()).toEqual(["bass", "drums", "other", "vocals"]);

  // Track lanes with mixer controls.
  for (const name of ["Original mix", "Vocals", "Drums", "Bass", "Other"]) {
    await expect(page.getByRole("listitem").filter({ hasText: name }).first()).toBeVisible();
  }
  await page.getByRole("button", { name: "Solo Bass" }).click();
  await expect(page.getByRole("button", { name: "Solo Bass" })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Solo Bass" }).click();

  // Edit: select Bass, add a note by double-clicking empty space high in the roll.
  await page.getByRole("listitem").filter({ hasText: "Bass" }).locator(".lname").click();
  await expect(page.getByRole("toolbar", { name: "Note editing tools" })).toContainText("Bass");
  const bass = project.tracks.find((t: { stem: string }) => t.stem === "bass");
  const before = (await (await request.get(`/api/projects/${projectId}/tracks/${bass.id}/notes`)).json()).notes.length;
  const roll = page.getByRole("application");
  const box = (await roll.boundingBox())!;
  await page.mouse.dblclick(box.x + box.width * 0.5, box.y + 40);
  await expect(page.getByRole("status").filter({ hasText: "All saved" })).toBeVisible({ timeout: 10_000 });
  await expect
    .poll(async () => (await (await request.get(`/api/projects/${projectId}/tracks/${bass.id}/notes`)).json()).notes.length)
    .toBe(before + 1);

  // Undo removes it again (and autosaves).
  await page.keyboard.press("Control+z");
  await expect
    .poll(async () => (await (await request.get(`/api/projects/${projectId}/tracks/${bass.id}/notes`)).json()).notes.length)
    .toBe(before);

  // Quantize via shortcut, then export multitrack MIDI through the dialog.
  await page.keyboard.press("q");
  await page.getByRole("button", { name: "Export", exact: true }).click();
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Multitrack MIDI only (.mid)" }).click(),
  ]);
  const file = await download.path();
  const midi = new Midi(fs.readFileSync(file!));
  const names = midi.tracks.map((t) => t.name).filter(Boolean);
  expect(names).toEqual(expect.arrayContaining(["Drums", "Bass", "Other"]));
  expect(midi.tracks.find((t) => t.name === "Drums")!.channel).toBe(9);
  expect(midi.tracks.find((t) => t.name === "Bass")!.notes.length).toBe(before);
  expect(midi.header.tempos[0].bpm).toBeGreaterThan(90);

  // Clean up
  await request.delete(`/api/projects/${projectId}`);
});

test("rejects a non-audio file with an actionable message", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Choose an audio file").setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("hello") });
  await expect(page.locator(".errbox")).toContainText("doesn't look like an audio file");
});
