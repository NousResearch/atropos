import fs from "fs";
import path from "path";
import type { Environment } from "@/types/env";

function getManifestPath() {
  return path.join(process.cwd(), "public", "environments.json");
}

export function readStaticManifest(): Environment[] {
  try {
    const raw = fs.readFileSync(getManifestPath(), "utf-8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Environment[]) : [];
  } catch {
    return [];
  }
}
