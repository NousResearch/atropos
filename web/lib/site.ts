const FALLBACK_REPO = "NousResearch/atropos";
const FALLBACK_REF = "main";

export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export const GITHUB_REPO = process.env.NEXT_PUBLIC_GITHUB_REPO ?? FALLBACK_REPO;
export const GITHUB_REF = process.env.NEXT_PUBLIC_GITHUB_REF ?? FALLBACK_REF;

/** Resolve the public site URL at runtime (works in client components). */
export function getSiteUrl(): string {
  if (typeof window !== "undefined") {
    return window.location.origin + BASE_PATH;
  }
  return process.env.NEXT_PUBLIC_SITE_URL ?? `http://localhost:3000${BASE_PATH}`;
}

function encodePath(path: string) {
  return path
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

export function withBasePath(pathname: string) {
  if (!BASE_PATH) {
    return pathname;
  }
  return `${BASE_PATH}${pathname.startsWith("/") ? pathname : `/${pathname}`}`;
}

export function getManifestUrl() {
  return withBasePath("/environments.json");
}

export function getEnvironmentDataUrl(slug: string) {
  return withBasePath(`/env-data/${encodeURIComponent(slug)}.json`);
}

export function getEnvironmentRoute(slug: string) {
  return `/environments/${encodeURIComponent(slug)}`;
}

export function getGithubTreeUrl(envId: string) {
  return `https://github.com/${GITHUB_REPO}/tree/${GITHUB_REF}/environments/${encodePath(envId)}`;
}

export function getGithubRawUrl(envId: string, filePath: string) {
  return `https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_REF}/environments/${encodePath(
    envId
  )}/${encodePath(filePath)}`;
}

export function getRepositoryUrl() {
  return `https://github.com/${GITHUB_REPO}`;
}

export function getContributingUrl() {
  return `https://github.com/${GITHUB_REPO}/blob/${GITHUB_REF}/CONTRIBUTING.md`;
}
