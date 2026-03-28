import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        line: "hsl(var(--line))",
        panel: "hsl(var(--panel))",
        "panel-alt": "hsl(var(--panel-alt))",
        "panel-elevated": "hsl(var(--panel-elevated))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent-secondary))",
          foreground: "hsl(var(--foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        warning: "hsl(var(--warning))",
      },
      boxShadow: {
        brutal: "0 0 0 1px rgba(255,255,255,0.08), 0 18px 48px rgba(0,0,0,0.45)",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
export default config;
