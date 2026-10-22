"use client";

import { useEffect, useState } from "react";
import { setToken } from "@/lib/api";

/** Shown when the API answers 401 (only when STEMFORGE_API_TOKEN is configured on the server). */
export function AuthPrompt() {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState("");
  useEffect(() => {
    const onAuth = () => setOpen(true);
    window.addEventListener("stemforge:auth-required", onAuth);
    return () => window.removeEventListener("stemforge:auth-required", onAuth);
  }, []);
  if (!open) return null;
  return (
    <div className="backdrop" role="dialog" aria-modal="true" aria-labelledby="auth-title">
      <form
        className="dialog"
        onSubmit={(e) => {
          e.preventDefault();
          setToken(value.trim() || null);
          setOpen(false);
          window.location.reload();
        }}
      >
        <header>
          <h2 id="auth-title">API token required</h2>
        </header>
        <p className="muted">
          This Stemforge server requires an access token (set by the administrator as STEMFORGE_API_TOKEN). It is kept in
          this browser only.
        </p>
        <input type="password" aria-label="API token" value={value} onChange={(e) => setValue(e.target.value)} autoFocus />
        <div className="row">
          <span className="spacer" />
          <button type="button" className="btn ghost" onClick={() => setOpen(false)}>
            Cancel
          </button>
          <button type="submit" className="btn primary">
            Save token
          </button>
        </div>
      </form>
    </div>
  );
}
