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
        /* All colors reference CSS vars — opacity modifiers (bg-ok/10) work
           because Tailwind replaces <alpha-value> at build time. */
        canvas: "rgb(var(--color-canvas) / <alpha-value>)",
        surface: {
          DEFAULT:  "rgb(var(--color-surface) / <alpha-value>)",
          raised:   "rgb(var(--color-surface-raised) / <alpha-value>)",
          elevated: "rgb(var(--color-surface-elevated) / <alpha-value>)",
          hover:    "rgb(var(--color-surface-hover) / <alpha-value>)",
        },
        ink: {
          0: "rgb(var(--color-ink-0) / <alpha-value>)",
          1: "rgb(var(--color-ink-1) / <alpha-value>)",
          2: "rgb(var(--color-ink-2) / <alpha-value>)",
          3: "rgb(var(--color-ink-3) / <alpha-value>)",
        },
        line: {
          /* These use rgba() values — no opacity modifier support needed */
          subtle: "var(--border-subtle)",
          base:   "var(--border-base)",
          strong: "var(--border-strong)",
        },
        brand: {
          blue:   "rgb(var(--color-brand-blue) / <alpha-value>)",
          cyan:   "rgb(var(--color-brand-cyan) / <alpha-value>)",
          indigo: "rgb(var(--color-brand-indigo) / <alpha-value>)",
        },
        ok:       { DEFAULT: "rgb(var(--color-ok) / <alpha-value>)"       },
        info:     { DEFAULT: "rgb(var(--color-info) / <alpha-value>)"     },
        warn:     { DEFAULT: "rgb(var(--color-warn) / <alpha-value>)"     },
        danger:   { DEFAULT: "rgb(var(--color-danger) / <alpha-value>)"   },
        critical: { DEFAULT: "rgb(var(--color-critical) / <alpha-value>)" },
      },
      boxShadow: {
        /* Reference CSS vars so shadows adapt per theme */
        raised:  "0 0 0 1px var(--border-base)",
        popover: "0 8px 24px -4px rgba(0,0,0,.18), 0 0 0 1px var(--border-base)",
        ring:    "0 0 0 2px rgb(var(--border-focus) / 0.5)",
      },
      keyframes: {
        pulseDot: { "0%,100%": { opacity: "1" }, "50%": { opacity: ".35" } },
        fadeIn:   { from: { opacity: "0", transform: "translateY(4px)" }, to: { opacity: "1", transform: "translateY(0)" } },
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
