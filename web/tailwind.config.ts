import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      colors: {
        canvas:  "#07090E",
        surface: {
          DEFAULT:  "#0B0E15",
          raised:   "#11151F",
          elevated: "#161B27",
          hover:    "#1A2030",
        },
        ink: {
          0: "#E6E9F2",
          1: "#A8AFC0",
          2: "#6C7488",
          3: "#4A5163",
        },
        line: {
          subtle: "rgba(255,255,255,0.05)",
          base:   "rgba(255,255,255,0.08)",
          strong: "rgba(255,255,255,0.14)",
        },
        brand: {
          blue:   "#3B82F6",
          cyan:   "#22D3EE",
          indigo: "#6366F1",
        },
        ok:       { DEFAULT: "#10B981" },
        info:     { DEFAULT: "#38BDF8" },
        warn:     { DEFAULT: "#F59E0B" },
        danger:   { DEFAULT: "#EF4444" },
        critical: { DEFAULT: "#F43F5E" },
      },
      boxShadow: {
        raised:  "0 0 0 1px rgba(255,255,255,0.06)",
        popover: "0 8px 24px -4px rgba(0,0,0,.6), 0 0 0 1px rgba(255,255,255,.06)",
        ring:    "0 0 0 2px rgba(56,189,248,.5)",
      },
      keyframes: {
        pulseDot: { "0%,100%": { opacity: "1" }, "50%": { opacity: ".35" } },
        fadeIn: { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "translateY(0)" } },
      },
      animation: {
        pulseDot: "pulseDot 2s ease-in-out infinite",
        fadeIn:   "fadeIn .2s ease-out forwards",
      },
    },
  },
  plugins: [],
};
export default config;
