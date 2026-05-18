import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Ollama Dashboard",
  description: "Live Ollama usage and throughput",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
