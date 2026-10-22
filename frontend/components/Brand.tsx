import Link from "next/link";

export function BrandMark({ size = 26 }: { size?: number }) {
  // Stacked stems splitting from one waveform: the product in one glyph.
  return (
    <svg className="brand-mark" width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect x="1" y="1" width="30" height="30" rx="8" fill="#1f242b" stroke="#3a434e" />
      <path d="M5 16h4l2-6 3 12 3-9 2 5h8" fill="none" stroke="#ff7a3d" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <rect x="21" y="8" width="6" height="2" rx="1" fill="#f2c14e" />
      <rect x="21" y="22" width="6" height="2" rx="1" fill="#6c9cff" />
    </svg>
  );
}

export function AppHeader({ children }: { children?: React.ReactNode }) {
  return (
    <header className="app-header">
      <Link href="/" className="brand" aria-label="Stemforge home">
        <BrandMark />
        Stemforge
      </Link>
      <span className="spacer" />
      {children}
    </header>
  );
}
