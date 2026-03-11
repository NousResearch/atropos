import fs from "fs";
import path from "path";
import type { Environment, EnvironmentDetail, EnvironmentFile } from "@/types/env";

function slugifyEnvId(envId: string) {
  return envId.replace(/\//g, "--");
}

function getManifestPath() {
  return (
    process.env.ENVIRONMENTS_MANIFEST_PATH ?? path.join(process.cwd(), "public", "environments.json")
  );
}

function getEnvironmentDataPath(envId: string) {
  return path.join(process.cwd(), "public", "env-data", `${slugifyEnvId(envId)}.json`);
}

function getEnvironmentsRoot() {
  return process.env.ENVIRONMENTS_PATH ?? path.resolve(process.cwd(), "..", "environments");
}

function isWithinRoot(root: string, candidate: string) {
  const relative = path.relative(root, candidate);
  return relative !== "" && !relative.startsWith("..") && !path.isAbsolute(relative);
}

export function loadManifest(): Environment[] {
  try {
    const raw = fs.readFileSync(getManifestPath(), "utf-8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as Environment[]) : [];
  } catch {
    return [];
  }
}

export function getEnvById(id: string) {
  return loadManifest().find((environment) => environment.id === id) ?? null;
}

export function getEnvDirPath(id: string) {
  const root = getEnvironmentsRoot();
  const fullPath = path.resolve(root, id);
  if (!isWithinRoot(root, fullPath)) {
    return null;
  }
  if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isDirectory()) {
    return null;
  }
  return fullPath;
}

export function getEnvFilePath(id: string, relativeFilePath: string) {
  const dirPath = getEnvDirPath(id);
  if (!dirPath) {
    return null;
  }
  const fullPath = path.resolve(dirPath, relativeFilePath);
  if (!isWithinRoot(dirPath, fullPath)) {
    return null;
  }
  if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
    return null;
  }
  return fullPath;
}

function readEnvironmentDetail(id: string): EnvironmentDetail | null {
  try {
    const raw = fs.readFileSync(getEnvironmentDataPath(id), "utf-8");
    return JSON.parse(raw) as EnvironmentDetail;
  } catch {
    return null;
  }
}

function walkFiles(dirPath: string, current = ""): EnvironmentFile[] {
  const entries = fs.readdirSync(path.join(dirPath, current), { withFileTypes: true });
  const files: EnvironmentFile[] = [];

  for (const entry of entries) {
    if (entry.name.startsWith(".")) {
      continue;
    }
    const relativePath = current ? `${current}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      files.push(...walkFiles(dirPath, relativePath));
      continue;
    }
    const fullPath = path.join(dirPath, relativePath);
    files.push({
      path: relativePath.replace(/\\/g, "/"),
      size: fs.statSync(fullPath).size,
      previewable: true,
    });
  }

  return files.sort((a, b) => a.path.localeCompare(b.path));
}

export function listEnvFiles(id: string) {
  const detail = readEnvironmentDetail(id);
  if (detail) {
    return detail.files.map((file) => file.path);
  }
  const dirPath = getEnvDirPath(id);
  if (!dirPath) {
    return [];
  }
  return walkFiles(dirPath).map((file) => file.path);
}

export function listEnvFilesWithSizes(id: string) {
  const detail = readEnvironmentDetail(id);
  if (detail) {
    return detail.files;
  }
  const dirPath = getEnvDirPath(id);
  if (!dirPath) {
    return [];
  }
  return walkFiles(dirPath);
}
