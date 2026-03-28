"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { cn } from "@/lib/utils";

const CODE_EXT_TO_LANG: Record<string, string> = {
  py: "python",
  js: "javascript",
  ts: "typescript",
  tsx: "tsx",
  jsx: "jsx",
  json: "json",
  yaml: "yaml",
  yml: "yaml",
  toml: "toml",
  md: "markdown",
  sh: "bash",
  bash: "bash",
  css: "css",
  html: "html",
};

function getLanguage(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  return CODE_EXT_TO_LANG[ext] ?? "text";
}

function isMarkdown(path: string): boolean {
  const lower = path.toLowerCase();
  return lower.endsWith(".md") || lower.endsWith(".markdown");
}

export function FileViewer({
  path,
  content,
  className,
}: {
  path: string;
  content: string;
  className?: string;
}) {
  if (isMarkdown(path)) {
    return (
      <div
        className={cn(
          "prose prose-invert prose-sm max-w-none dark:prose-invert prose-headings:uppercase prose-headings:tracking-[0.12em] prose-headings:text-foreground prose-p:text-muted-foreground prose-a:text-primary prose-strong:text-foreground prose-code:text-primary prose-pre:border prose-pre:border-white/10 prose-pre:bg-black/55 prose-blockquote:border-primary/40 prose-blockquote:text-muted-foreground",
          className
        )}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ node, className: codeClassName, children, ...props }) {
              const match = /language-(\w+)/.exec(codeClassName ?? "");
              if (!match) {
                return (
                  <code className={codeClassName} {...props}>
                    {children}
                  </code>
                );
              }
              return (
                <SyntaxHighlighter
                  style={oneDark}
                  language={match[1]}
                  PreTag="div"
                  customStyle={{
                    margin: 0,
                    borderRadius: 0,
                    background: "rgba(0, 0, 0, 0.55)",
                    border: "1px solid rgba(255,255,255,0.08)",
                  }}
                  codeTagProps={{ style: {} }}
                >
                  {String(children).replace(/\n$/, "")}
                </SyntaxHighlighter>
              );
            },
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
    );
  }

  const language = getLanguage(path);
  return (
    <SyntaxHighlighter
      style={oneDark}
      language={language}
      PreTag="div"
      showLineNumbers
      customStyle={{
        margin: 0,
        padding: "1rem",
        background: "rgba(0, 0, 0, 0.55)",
        fontSize: "0.875rem",
        border: "1px solid rgba(255,255,255,0.08)",
      }}
      codeTagProps={{ style: {} }}
      className={cn("terminal-code", className)}
    >
      {content}
    </SyntaxHighlighter>
  );
}
