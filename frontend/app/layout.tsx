import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Toasts } from "@/components/Toasts";
import { AuthPrompt } from "@/components/AuthPrompt";

export const metadata: Metadata = {
  title: "Stemforge",
  description: "Separate songs into stems and editable MIDI.",
};

export const viewport: Viewport = { themeColor: "#0e1013", width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        {children}
        <Toasts />
        <AuthPrompt />
      </body>
    </html>
  );
}
