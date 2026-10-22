import { describe, expect, it } from "vitest";
import { fmtBytes, fmtEta, fmtTime, parseUtc } from "@/lib/format";

describe("format helpers", () => {
  it("formats times", () => {
    expect(fmtTime(0)).toBe("0:00.0");
    expect(fmtTime(75.25, 2)).toBe("1:15.25");
    expect(fmtTime(-3, 0)).toBe("-0:03");
  });
  it("formats sizes", () => {
    expect(fmtBytes(512)).toBe("512 B");
    expect(fmtBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
  it("treats naive API timestamps as UTC", () => {
    expect(parseUtc("2026-01-01T00:00:00").toISOString()).toBe("2026-01-01T00:00:00.000Z");
    expect(parseUtc("2026-01-01T00:00:00+02:00").toISOString()).toBe("2025-12-31T22:00:00.000Z");
  });
  it("only states ETAs that exist", () => {
    expect(fmtEta(null)).toBeNull();
    expect(fmtEta(30)).toContain("30 s");
  });
});
