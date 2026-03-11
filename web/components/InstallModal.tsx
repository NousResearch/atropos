"use client";

import { useState } from "react";
import { Check, Copy, ExternalLink, FolderGit2, TerminalSquare, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from "@/components/ui/dialog";
import { GITHUB_REF, GITHUB_REPO, getSiteUrl } from "@/lib/site";
import type { Environment } from "@/types/env";

const STEPS = [
  {
    title: "Install Atropos",
    note: "Install the CLI directly from the repository.",
    code: "pip install git+https://github.com/NousResearch/atropos.git",
  },
  {
    title: "Install Environment",
    note: "Install a specific environment from your base URL.",
    code: (id: string) =>
      `atropos install ${id} --base-url ${getSiteUrl()}`,
  },
  {
    title: "List Installed Environments",
    note: "Verify available environments.",
    code: "atropos list",
  },
];

function CodeBlock({
  code,
  onCopy,
}: {
  code: string;
  onCopy?: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    onCopy?.();
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div className="terminal-code relative border border-white/10 bg-black/55 p-4">
      <div className="data-label mb-3 text-primary">Command</div>
      <pre className="overflow-x-auto whitespace-pre-wrap break-all text-xs leading-6">{code}</pre>
      <Button
        variant="ghost"
        size="icon"
        className="absolute right-3 top-3 h-8 w-8"
        onClick={handleCopy}
      >
        {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
      </Button>
    </div>
  );
}

interface InstallModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  env: Environment;
  sourceUrl: string;
  readmeUrl?: string | null;
}

export function InstallModal({ open, onOpenChange, env, sourceUrl, readmeUrl }: InstallModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] max-w-3xl overflow-y-auto p-0">
        <DialogClose>
          <X className="h-4 w-4" />
        </DialogClose>
        <div className="hero-grid border-b border-white/10 px-6 py-6 sm:px-8">
          <DialogHeader>
            <div className="flex flex-wrap items-center gap-2">
              <span className="section-kicker">Acquisition Procedure</span>
              <span className="data-label text-primary">{env.id}</span>
            </div>
            <DialogTitle className="mt-3">Acquire & Use Environment</DialogTitle>
            <DialogDescription>
              Use the hosted dossier to inspect files, then retrieve the environment directly from
              the repository with a sparse checkout when you need the package on disk.
            </DialogDescription>
          </DialogHeader>
        </div>

        <div className="space-y-8 px-6 py-6 sm:px-8">
          <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
            <div className="screen-frame-alt p-5">
              <div className="data-label">Environment</div>
              <div className="mt-3 text-xl font-semibold uppercase tracking-[0.12em] text-foreground">
                {env.name}
              </div>
              <p className="mt-4 text-sm leading-6 text-muted-foreground">
                {env.description || "No environment description is currently available."}
              </p>
            </div>

            <div className="screen-frame-alt p-5">
              <div className="data-label">Actions</div>
              <div className="mt-4 flex flex-col gap-3">
                <Button
                  onClick={() => {
                    window.open(sourceUrl, "_blank");
                  }}
                >
                  <FolderGit2 className="h-4 w-4" />
                  Open Source Folder
                </Button>
                <Button
                  variant="outline"
                  onClick={() => {
                    if (readmeUrl) {
                      window.open(readmeUrl, "_blank");
                    }
                  }}
                  disabled={!readmeUrl}
                >
                  <ExternalLink className="h-4 w-4" />
                  Open Raw README
                </Button>
              </div>
            </div>
          </div>

          <div className="screen-frame scanlines p-5">
            <div className="flex items-center justify-between gap-4 border-b border-white/10 pb-4">
              <div>
                <div className="section-kicker">Execution Steps</div>
                <div className="mt-2 text-lg font-semibold uppercase tracking-[0.14em] text-foreground">
                  Operator Checklist
                </div>
              </div>
              <TerminalSquare className="h-5 w-5 text-primary" />
            </div>

            <div className="mt-6 space-y-6">
              {STEPS.map((step, i) => (
                <div
                  key={i}
                  className="grid gap-4 border-b border-white/8 pb-6 last:border-b-0 last:pb-0 lg:grid-cols-[13rem_1fr]"
                >
                  <div>
                    <div className="data-label">Step {String(i + 1).padStart(2, "0")}</div>
                    <h4 className="mt-2 text-sm font-semibold uppercase tracking-[0.16em] text-foreground">
                      {step.title}
                    </h4>
                    {step.note && (
                      <p className="mt-3 text-sm leading-6 text-muted-foreground">{step.note}</p>
                    )}
                  </div>
                  <CodeBlock
                    code={
                      typeof step.code === "function"
                        ? step.code(env.id)
                        : step.code
                    }
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
