import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CyberML — Cyber Attack Classification",
  description: "AI-based network intrusion detection · CICIDS2017 · KMITL Senior Project",
};

/* Runs synchronously before first paint — prevents theme flash.
   Must be a plain string (no JSX expressions inside the script). */
const themeScript = `(function(){try{var t=localStorage.getItem('theme');if(t==='light'||t==='dark'){document.documentElement.setAttribute('data-theme',t);return;}if(window.matchMedia('(prefers-color-scheme:light)').matches){document.documentElement.setAttribute('data-theme','light');}else{document.documentElement.setAttribute('data-theme','dark');}}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Flash-prevention script — must be before any CSS or body content */}
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link rel="preconnect" href="http://localhost:8000" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"
        />
      </head>
      <body className="antialiased bg-canvas text-ink-0">{children}</body>
    </html>
  );
}
