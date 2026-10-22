"use client";

const SHORTCUTS: [string, string][] = [
  ["Space", "Play / pause"],
  ["Enter / Home", "Stop and return to start"],
  ["C", "A/B: original mix ↔ stems"],
  ["Ctrl+Z / Ctrl+Shift+Z", "Undo / redo (per track)"],
  ["Ctrl+S", "Save now (edits also autosave)"],
  ["V / D", "Select tool / draw tool"],
  ["Double-click", "Add note (empty space) / delete note"],
  ["Alt+click", "Delete note"],
  ["Drag note edge", "Resize (hold Alt to bypass snap)"],
  ["Shift+drag / Shift+click", "Add to selection"],
  ["Ctrl+A / Esc", "Select all / clear selection"],
  ["Delete / Backspace", "Delete selection"],
  ["↑ / ↓ (Shift)", "Transpose semitone (octave)"],
  ["← / → (Shift)", "Nudge by grid step (×4)"],
  ["Q / Shift+Q", "Quantize starts / starts and ends"],
  ["Ctrl+C / Ctrl+V", "Copy / paste at playhead"],
  ["Ctrl+D", "Duplicate selection after itself"],
  ["L", "Loop selected notes"],
  ["M / S", "Mute / solo selected track"],
  ["1–9", "Select track"],
  ["+ / −, Ctrl+wheel", "Zoom time"],
  ["Wheel / Shift+wheel / Alt+wheel", "Scroll pitch / scroll time / zoom pitch"],
  ["?", "This help"],
];

export function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="backdrop" role="dialog" aria-modal="true" aria-labelledby="sc-title" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog" onKeyDown={(e) => e.key === "Escape" && onClose()}>
        <header>
          <h2 id="sc-title">Keyboard shortcuts</h2>
          <button className="btn ghost sm icon" aria-label="Close" onClick={onClose} autoFocus>×</button>
        </header>
        <div className="shortcuts">
          {SHORTCUTS.map(([k, v]) => (
            <div key={k} style={{ display: "contents" }}>
              <kbd>{k}</kbd>
              <span>{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
