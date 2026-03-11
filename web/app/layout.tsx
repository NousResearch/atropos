import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Atropos Environments Hub",
  description:
    "A Nous Research-aligned archive for discovering, evaluating, and operationalizing Atropos environments.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen overflow-x-hidden bg-[hsl(var(--background))] text-[hsl(var(--foreground))]">
        {children}
      </body>
    </html>
  );
}
