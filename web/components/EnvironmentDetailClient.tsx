"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowLeft,
  ChevronRight,
  ExternalLink,
  FileText,
  Folder,
  Orbit,
  ScanSearch,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

import { InstallModal } from "@/components/InstallModal";
import { FileViewer } from "@/components/FileViewer";
import {
  getEnvironmentDataUrl,
  getGithubRawUrl,
  getGithubTreeUrl,
  getRepositoryUrl,
} from "@/lib/site";
import { cn } from "@/lib/utils";
import type { EnvironmentDetail } from "@/types/env";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function DetailPanel({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="screen-frame-alt p-4">
      <div className="data-label">{label}</div>
      <div className="mt-3 text-sm font-semibold uppercase tracking-[0.16em] text-foreground">
        {value}
      </div>
    </div>
  );
}

function PlaceholderPanel({
  label,
  message,
}: {
  label: string;
  message: string;
}) {
  return (
    <div className="screen-frame scanlines p-8">
      <div className="section-kicker">{label}</div>
      <p className="mt-4 max-w-2xl text-sm leading-7 text-muted-foreground">{message}</p>
    </div>
  );
}

async function fetchText(url: string) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error("Request failed");
  }
  return response.text();
}

export function EnvironmentDetailClient({ slug }: { slug: string }) {
  const [detail, setDetail] = useState<EnvironmentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [installOpen, setInstallOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileLoading, setFileLoading] = useState(false);
  const [readmeContent, setReadmeContent] = useState<string | null>(null);
  const [readmeLoading, setReadmeLoading] = useState(false);

  useEffect(() => {
    fetch(getEnvironmentDataUrl(slug))
      .then((response) => {
        if (!response.ok) {
          throw new Error("Not found");
        }
        return response.json();
      })
      .then((data) => setDetail(data as EnvironmentDetail))
      .catch(() => setDetail(null))
      .finally(() => setLoading(false));
  }, [slug]);

  const env = detail?.environment ?? null;
  const files = detail?.files ?? [];
  const readmePath = detail?.readmePath ?? null;

  useEffect(() => {
    if (files.length === 0 || selectedFile) return;
    const preferred = files.find((file) => /readme\.md$/i.test(file.path)) ?? files[0];
    setSelectedFile(preferred.path);
  }, [files, selectedFile]);

  const selectedFileMeta = useMemo(
    () => files.find((file) => file.path === selectedFile) ?? null,
    [files, selectedFile]
  );

  useEffect(() => {
    if (!env || !selectedFileMeta) {
      setFileContent(null);
      return;
    }
    if (!selectedFileMeta.previewable) {
      setFileContent(null);
      return;
    }

    setFileLoading(true);
    fetchText(getGithubRawUrl(env.id, selectedFileMeta.path))
      .then(setFileContent)
      .catch(() => setFileContent(null))
      .finally(() => setFileLoading(false));
  }, [env, selectedFileMeta]);

  useEffect(() => {
    if (!env || !readmePath) {
      setReadmeContent(null);
      return;
    }
    setReadmeLoading(true);
    fetchText(getGithubRawUrl(env.id, readmePath))
      .then(setReadmeContent)
      .catch(() => setReadmeContent(null))
      .finally(() => setReadmeLoading(false));
  }, [env, readmePath]);

  if (loading) {
    return (
      <div className="ui-shell flex min-h-screen items-center justify-center px-6">
        <div className="screen-frame scanlines w-full max-w-xl p-8 text-center">
          <p className="text-sm uppercase tracking-[0.24em] text-muted-foreground">
            Loading environment dossier
          </p>
        </div>
      </div>
    );
  }

  if (!env || !detail) {
    return (
      <div className="ui-shell flex min-h-screen flex-col items-center justify-center gap-4 px-6">
        <div className="screen-frame scanlines w-full max-w-xl p-8 text-center">
          <p className="text-sm uppercase tracking-[0.24em] text-muted-foreground">
            Environment not found
          </p>
        </div>
        <Link href="/">
          <Button variant="outline">Back to Explore</Button>
        </Link>
      </div>
    );
  }

  const githubTreeUrl = getGithubTreeUrl(env.id);
  const readmeUrl = readmePath ? getGithubRawUrl(env.id, readmePath) : null;

  return (
    <div className="ui-shell">
      <div className="mx-auto flex max-w-7xl flex-col gap-8 px-6 py-8">
        <header className="label-panel overflow-hidden">
          <div className="grid gap-6 p-6 sm:p-8 xl:grid-cols-[1.18fr_0.82fr]">
            <div className="space-y-6">
              <div className="flex flex-wrap items-center gap-3">
                <span className="panel-tag">Environment Dossier</span>
                <span className="warning-tag">Static Inspection</span>
                <Badge variant="outline">{env.id}</Badge>
              </div>

              <div className="flex items-start gap-4">
                <Link href="/">
                  <Button variant="ghost" size="icon" aria-label="Back to explore">
                    <ArrowLeft className="h-4 w-4" />
                  </Button>
                </Link>
                <div className="space-y-4">
                  <div>
                    <p className="poster-caption">Atropos // environment archive</p>
                    <h1 className="mt-3 document-title sm:text-4xl">{env.name}</h1>
                  </div>
                  <p className="max-w-3xl text-sm leading-7 text-muted-foreground sm:text-base">
                    {env.description || "No environment description is currently available."}
                  </p>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                <DetailPanel label="Visible Files" value={`${files.length}`} />
                <DetailPanel label="Top Tag" value={env.tags[0] ?? "General"} />
                <DetailPanel label="README Status" value={readmePath ? "Indexed" : "Unavailable"} />
              </div>
            </div>

            <div className="grid gap-4">
              <div className="screen-frame-elevated p-5">
                <div className="flex items-start justify-between gap-6">
                  <div>
                    <div className="section-kicker">Reference Mark</div>
                    <div className="mt-3 text-lg font-semibold uppercase tracking-[0.14em] text-foreground">
                      Source Orbit
                    </div>
                  </div>
                  <div className="orbital-mark shrink-0">
                    <div className="orbital-grid" />
                  </div>
                </div>
                <div className="mt-6 caution-rule" />
              </div>

              <div className="screen-frame-elevated terminal-code p-5">
                <div className="data-label text-primary">Access Pattern</div>
                <pre className="mt-3 overflow-x-auto text-xs leading-6 text-foreground">
                  {`inspect manifest\npreview tracked text files\nopen source tree on GitHub`}
                </pre>
              </div>

              <div className="screen-frame-alt p-5">
                <div className="data-label">Action Cluster</div>
                <div className="mt-4 flex flex-col gap-3">
                  <Button className="justify-between" size="sm" onClick={() => setInstallOpen(true)}>
                    <span className="flex items-center gap-2">
                      <Folder className="h-4 w-4" />
                      Acquire Source
                    </span>
                    <ChevronRight className="h-4 w-4" />
                  </Button>

                </div>
              </div>
            </div>
          </div>

        </header>

        <div className="mt-0 space-y-8">
          <div className="grid gap-6 xl:grid-cols-[0.9fr_1.4fr]">
            <section className="screen-frame scanlines overflow-hidden">
              <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
                <div>
                  <div className="section-kicker">Package Archive</div>
                  <div className="mt-2 text-lg font-semibold uppercase tracking-[0.14em] text-foreground">
                    Files
                  </div>
                </div>
                <Folder className="h-5 w-5 text-primary" />
              </div>

              <ul className="max-h-[40rem] overflow-y-auto">
                {files.length === 0 && (
                  <li className="px-5 py-4 text-sm text-muted-foreground">
                    No files indexed for this environment.
                  </li>
                )}
                {files.map((file) => (
                  <li key={file.path} className="border-b border-white/8 last:border-b-0">
                    <button
                      type="button"
                      onClick={() => setSelectedFile(file.path)}
                      className={cn(
                        "flex w-full items-center gap-3 px-5 py-4 text-left transition-colors hover:bg-white/5",
                        selectedFile === file.path && "bg-primary/10"
                      )}
                    >
                      <FileText className="h-4 w-4 shrink-0 text-primary" />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-medium text-foreground">{file.path}</div>
                        <div className="mt-1 text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                          {formatSize(file.size)} {file.previewable ? " // previewable" : " // binary or large"}
                        </div>
                      </div>
                      <ChevronRight className="h-4 w-4 shrink-0 text-muted-foreground" />
                    </button>
                  </li>
                ))}
              </ul>
            </section>

            <section className="screen-frame scanlines overflow-hidden">
              <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
                <div>
                  <div className="section-kicker">Inspection Surface</div>
                  <div className="mt-2 text-lg font-semibold uppercase tracking-[0.14em] text-foreground">
                    {selectedFile ? "File Viewer" : "Awaiting File Selection"}
                  </div>
                </div>
                <ScanSearch className="h-5 w-5 text-primary" />
              </div>

              {selectedFile ? (
                <div className="space-y-0">
                  <div className="border-b border-white/8 bg-black/30 px-5 py-4">
                    <div className="data-label">Selected Path</div>
                    <div className="mt-2 truncate font-mono text-sm text-foreground">{selectedFile}</div>
                  </div>
                  <div className="max-h-[42rem] overflow-auto p-5">
                    {fileLoading ? (
                      <div className="text-sm text-muted-foreground">Loading file contents…</div>
                    ) : selectedFileMeta && !selectedFileMeta.previewable ? (
                      <div className="space-y-4 text-sm text-muted-foreground">
                        <p>This file is not previewable in the static mirror. Open the source tree to inspect it directly.</p>
                        <a href={githubTreeUrl} target="_blank" rel="noopener noreferrer">
                          <Button variant="outline" size="sm">
                            Open Source Folder
                            <ExternalLink className="h-4 w-4" />
                          </Button>
                        </a>
                      </div>
                    ) : fileContent !== null ? (
                      <FileViewer path={selectedFile} content={fileContent} />
                    ) : (
                      <div className="text-sm text-muted-foreground">
                        Could not load this file preview from GitHub raw content.
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="p-8 text-sm leading-7 text-muted-foreground">
                  Select a file from the archive panel to inspect its contents.
                </div>
              )}
            </section>
          </div>

          <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
            <section className="screen-frame scanlines p-5 sm:p-6">
              <div className="section-kicker">Operational Summary</div>
              <div className="mt-4 grid gap-6 lg:grid-cols-[1fr_0.95fr]">
                <div className="space-y-6">
                  <div>
                    <h2 className="text-lg font-semibold uppercase tracking-[0.14em] text-foreground">
                      Overview
                    </h2>
                    <ul className="mt-4 space-y-3 text-sm leading-7 text-muted-foreground">
                      <li>
                        <span className="text-foreground">Environment ID:</span> <code>{env.id}</code>
                      </li>
                      <li>
                        <span className="text-foreground">Description:</span>{" "}
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            p: ({ children }) => <span>{children}</span>,
                          }}
                        >
                          {env.description || "—"}
                        </ReactMarkdown>
                      </li>
                      <li>
                        <span className="text-foreground">Research Tags:</span>{" "}
                        {(env.tags || []).join(", ") || "—"}
                      </li>
                      <li>
                        <span className="text-foreground">Tracked Size:</span> {formatSize(detail.totalSize)}
                      </li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-sm font-semibold uppercase tracking-[0.16em] text-foreground">
                      Usage Direction
                    </h3>
                    <p className="mt-3 text-sm leading-7 text-muted-foreground">
                      This hosted mirror emphasizes file inspection and acquisition. Use the source
                      folder link or the sparse-checkout flow in the install panel when you need the
                      actual environment package on disk.
                    </p>
                  </div>
                </div>

                <aside className="space-y-4">
                  <div className="screen-frame-alt p-4">
                    <div className="data-label">About</div>
                    <p className="mt-3 text-sm leading-6 text-muted-foreground">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: ({ children }) => <>{children}</>,
                        }}
                      >
                        {env.description || "No additional notes available."}
                      </ReactMarkdown>
                    </p>
                  </div>
                  <div className="screen-frame-alt p-4">
                    <div className="data-label">Research Tags</div>
                    <div className="mt-4 flex flex-wrap gap-2">
                      {(env.tags || ["general"]).map((tag) => (
                        <Badge key={tag} variant="secondary">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div className="screen-frame-alt p-4">
                    <div className="data-label">Runtime Baseline</div>
                    <div className="mt-3 flex items-center justify-between gap-4">
                      <div className="space-y-2 text-sm leading-6 text-muted-foreground">
                        <p>
                          <span className="text-foreground">Python:</span> &gt;=3.10
                        </p>
                        <p>
                          <span className="text-foreground">Surface:</span> GitHub raw + static JSON
                        </p>
                      </div>
                      <Orbit className="h-5 w-5 text-primary" />
                    </div>
                  </div>
                </aside>
              </div>
            </section>

            {readmePath ? (
              <section className="screen-frame scanlines overflow-hidden">
                <div className="border-b border-white/10 px-5 py-4">
                  <div className="section-kicker">Documentation Surface</div>
                  <div className="mt-2 text-lg font-semibold uppercase tracking-[0.14em] text-foreground">
                    README
                  </div>
                  <div className="mt-2 truncate text-xs uppercase tracking-[0.22em] text-muted-foreground">
                    {readmePath}
                  </div>
                </div>
                <div className="max-h-[44rem] overflow-auto p-5">
                  {readmeLoading ? (
                    <div className="text-sm text-muted-foreground">Loading README…</div>
                  ) : readmeContent !== null ? (
                    <FileViewer path={readmePath} content={readmeContent} />
                  ) : (
                    <div className="space-y-4 text-sm text-muted-foreground">
                      <p>Could not load the README preview from GitHub raw content.</p>
                      {readmeUrl && (
                        <a href={readmeUrl} target="_blank" rel="noopener noreferrer">
                          <Button variant="outline" size="sm">
                            Open Raw README
                            <ExternalLink className="h-4 w-4" />
                          </Button>
                        </a>
                      )}
                    </div>
                  )}
                </div>
              </section>
            ) : (
              <PlaceholderPanel
                label="Documentation Surface"
                message="No README file was detected for this package archive."
              />
            )}
          </div>
        </div>
      </div>

      <InstallModal
        open={installOpen}
        onOpenChange={setInstallOpen}
        env={env}
        sourceUrl={githubTreeUrl}
        readmeUrl={readmeUrl}
      />
    </div>
  );
}
