export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0E1117",
        surface: "#161C28",
        surface2: "#1D2534",
        hair: "#262F42",
        ink: "#E8ECF5",
        muted: "#8892A6",
        energyFrom: "#FF5E62",
        energyTo: "#FF9E5C",
        teal: "#22D3A8",
        violet: "#7C6BFF",
      },
      fontFamily: {
        display: ['"Space Grotesk"', "system-ui", "sans-serif"],
        sans: ['"Plus Jakarta Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 12px 40px -12px rgba(255,94,98,0.45)",
        card: "0 8px 24px -16px rgba(0,0,0,0.6)",
      },
    },
  },
  plugins: [],
};
