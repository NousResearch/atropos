"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  ChevronRight,
  Database,
  Folder,
  Globe,
  Grid2X2,
  List,
  Orbit,
  Search,
  Sparkles,
  Star,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getContributingUrl, getEnvironmentRoute, getManifestUrl, getRepositoryUrl } from "@/lib/site";
import { cn } from "@/lib/utils";
import type { Environment } from "@/types/env";

const CATEGORIES = [
  { label: "Coding", value: "coding" },
  { label: "Games", value: "games" },
  { label: "Math", value: "math" },
  { label: "Eval", value: "eval" },
  { label: "Community", value: "community" },
];

function formatBytes(bytes?: number) {
  if (!bytes) {
    return "--";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function EnvironmentCard({
  env,
  view,
  index,
}: {
  env: Environment;
  view: "grid" | "list";
  index: number;
}) {
  return (
    <Link href={getEnvironmentRoute(env.slug)} className="block h-full">
      <Card
        className={cn(
          "group h-full transition-all duration-200 hover:border-primary/60 hover:bg-[hsl(var(--panel-elevated))]",
          view === "list" && "flex h-auto flex-col sm:flex-row"
        )}
      >
        <CardHeader
          className={cn(
            "flex h-full flex-col justify-between gap-6 border-b border-white/8 pb-4",
            view === "list" && "w-full border-b sm:w-[19rem] sm:border-b-0 sm:border-r"
          )}
        >
          <div className="space-y-4">
            <div className="flex items-start justify-between gap-4">
              <CardTitle className="text-base">{env.name}</CardTitle>
              <ChevronRight className="h-4 w-4 shrink-0 text-primary transition-transform group-hover:translate-x-1" />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="panel-tag">Unit {String(index + 1).padStart(2, "0")}</span>
              <Badge variant="outline" className="max-w-full truncate">
                {env.id}
              </Badge>
            </div>
            <CardDescription className="line-clamp-3">
              {env.description || "No descriptive abstract has been logged for this environment."}
            </CardDescription>
          </div>

          <div className="space-y-3">
            <div className="barcode-rule opacity-80" />
            <div className="grid gap-2 text-xs uppercase tracking-[0.22em] text-muted-foreground sm:grid-cols-2">
              <div className="screen-frame-alt p-3">
                <div className="data-label">Files</div>
                <div className="mt-2 text-sm font-semibold text-foreground">{env.fileCount ?? "--"}</div>
              </div>
              <div className="screen-frame-alt p-3">
                <div className="data-label">Mass</div>
                <div className="mt-2 text-sm font-semibold text-foreground">
                  {formatBytes(env.totalSize)}
                </div>
              </div>
            </div>
          </div>
        </CardHeader>

        {view === "list" && (
          <CardContent className="flex flex-1 flex-col justify-between gap-6 py-5 sm:py-6">
            <div className="flex flex-wrap gap-2">
              {(env.tags.length ? env.tags : ["general"]).slice(0, 5).map((tag) => (
                <Badge key={tag} variant="secondary">
                  {tag}
                </Badge>
              ))}
            </div>
            <div className="spec-list">
              <div className="spec-row">
                <span>Registry</span>
                <span>Atropos</span>
              </div>
              <div className="spec-row">
                <span>README</span>
                <span>{env.readmePath ? "Indexed" : "Missing"}</span>
              </div>
              <div className="spec-row">
                <span>Surface</span>
                <span>Dossier</span>
              </div>
            </div>
          </CardContent>
        )}
      </Card>
    </Link>
  );
}

function EmptyPanel({ message }: { message: string }) {
  return (
    <div className="screen-frame scanlines p-8 text-center">
      <p className="text-sm uppercase tracking-[0.22em] text-muted-foreground">{message}</p>
    </div>
  );
}

function SectionHeader({
  label,
  title,
  count,
}: {
  label: string;
  title: string;
  count: number;
}) {
  return (
    <div className="mb-5 flex flex-col gap-3 border-b border-white/10 pb-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <p className="section-kicker">{label}</p>
        <h2 className="mt-2 document-title">{title}</h2>
      </div>
      <div className="fine-print">{count} entries surfaced</div>
    </div>
  );
}

function SignalTile({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Database;
  label: string;
  value: string;
}) {
  return (
    <div className="screen-frame-alt p-4">
      <div className="flex items-center justify-between gap-3">
        <span className="data-label">{label}</span>
        <Icon className="h-4 w-4 text-primary" />
      </div>
      <div className="mt-4 text-lg font-semibold uppercase tracking-[0.14em] text-foreground">
        {value}
      </div>
    </div>
  );
}

export default function ExplorePage() {
  const [envs, setEnvs] = useState<Environment[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<string | null>(null);
  const [view, setView] = useState<"grid" | "list">("grid");
  const [activeTab, setActiveTab] = useState("explore");

  useEffect(() => {
    fetch(getManifestUrl())
      .then((response) => response.json())
      .then((data) => {
        setEnvs(Array.isArray(data) ? (data as Environment[]) : []);
      })
      .catch(() => setEnvs([]))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    return envs.filter((env) => {
      const query = search.trim().toLowerCase();
      const matchesSearch =
        !query ||
        env.name.toLowerCase().includes(query) ||
        env.id.toLowerCase().includes(query) ||
        env.description.toLowerCase().includes(query) ||
        env.tags.some((tag) => tag.toLowerCase().includes(query));
      const matchesCategory = !category || env.tags.includes(category);
      return matchesSearch && matchesCategory;
    });
  }, [category, envs, search]);

  const featured = filtered.slice(0, 6);
  const archiveMass = filtered.reduce((sum, env) => sum + (env.totalSize ?? 0), 0);

  return (
    <div className="ui-shell">
      <main className="mx-auto flex max-w-7xl flex-col gap-8 px-6 py-8">
        <section className="grid gap-6 xl:grid-cols-[1.45fr_0.55fr]">
          <div className="label-panel p-6 sm:p-8">
            <div className="relative z-10 flex h-full flex-col gap-8">
              <div className="flex flex-wrap items-center gap-2">
                <span className="panel-tag">Atropos Archive</span>
                <span className="warning-tag">Handle With Care</span>
                <span className="panel-tag">Static Mirror Ready</span>
              </div>

              <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
                <div>
                  <p className="poster-caption">Nous Research // environment registry</p>
                  <h1 className="poster-title mt-5">
                    Human
                    <br />
                    Training
                    <br />
                    Worlds
                  </h1>
                  <p className="research-copy mt-6 max-w-2xl">
                    A poster-styled inspection surface for the Atropos environment archive. Browse
                    candidate worlds, inspect tracked files, and move from discovery to repo-level
                    retrieval without standing up backend infrastructure.
                  </p>
                </div>

                <div className="flex flex-col justify-between gap-6">
                  <div className="screen-frame-elevated p-5">
                    <div className="flex items-start justify-between gap-6">
                      <div>
                        <div className="section-kicker">Visual Motif</div>
                        <div className="mt-3 text-lg font-semibold uppercase tracking-[0.14em] text-foreground">
                          Research Label
                        </div>
                      </div>
                      <div className="orbital-mark shrink-0">
                        <div className="orbital-grid" />
                      </div>
                    </div>
                    <div className="mt-6 caution-rule" />
                  </div>

                  <div className="terminal-code screen-frame-elevated p-5">
                    <div className="data-label text-primary">Acquisition Path</div>
                    <pre className="mt-3 overflow-x-auto text-xs leading-6 text-foreground">
                      {`git clone --filter=blob:none --sparse\ncd atropos\nbrowse dossier -> fetch source`}
                    </pre>
                  </div>
                </div>
              </div>

              <div className="grid gap-4 border-t border-white/10 pt-6 sm:grid-cols-3">
                <SignalTile icon={Database} label="Registry Size" value={`${envs.length}`} />
                <SignalTile icon={Sparkles} label="Filtered Set" value={loading ? "..." : `${filtered.length}`} />
                <SignalTile icon={Folder} label="Visible Mass" value={formatBytes(archiveMass)} />
              </div>
            </div>
          </div>

          <aside className="grid gap-4">
            <div className="screen-frame-alt p-5">
              <div className="data-label">Poster Notes</div>
              <div className="mt-4 space-y-4 text-sm leading-6 text-muted-foreground">
                <p>Warm monochrome framing, caution accents, and dense metadata blocks mirror the supplied label/poster references.</p>
                <p>The hosted build reads generated JSON plus GitHub raw content, so the UI can be published as a static mirror.</p>
              </div>
            </div>

            <div className="screen-frame-alt p-5">
              <div className="data-label">Surface Mode</div>
              <div className="mt-4 flex items-center justify-between gap-4">
                <div>
                  <div className="text-lg font-semibold uppercase tracking-[0.14em] text-foreground">
                    {view === "grid" ? "Grid Scan" : "List Audit"}
                  </div>
                  <div className="mt-2 text-xs uppercase tracking-[0.22em] text-muted-foreground">
                    {category ? `${category} tag filter active` : "All tags online"}
                  </div>
                </div>
                <Orbit className="h-5 w-5 text-primary" />
              </div>
            </div>

            <div className="screen-frame-alt p-5">
              <div className="data-label">Outbound</div>
              <div className="mt-4 flex flex-col gap-3">
                <a href={getRepositoryUrl()} target="_blank" rel="noopener noreferrer">
                  <Button variant="outline" className="w-full justify-between">
                    Repository
                    <ArrowUpRight className="h-4 w-4" />
                  </Button>
                </a>
                <a href={getContributingUrl()} target="_blank" rel="noopener noreferrer">
                  <Button className="w-full justify-between">
                    Submit Environment
                    <ArrowUpRight className="h-4 w-4" />
                  </Button>
                </a>
              </div>
            </div>
          </aside>
        </section>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <div className="screen-frame scanlines p-4 sm:p-5">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <TabsList className="w-full justify-start xl:w-auto">
                <TabsTrigger value="explore" className="gap-2">
                  <Globe className="h-4 w-4" /> Explore
                </TabsTrigger>
                <TabsTrigger value="stars" className="gap-2">
                  <Star className="h-4 w-4" /> Signals
                </TabsTrigger>
                <TabsTrigger value="archive" className="gap-2">
                  <Folder className="h-4 w-4" /> Archive
                </TabsTrigger>
              </TabsList>

              <div className="flex flex-col gap-4 xl:flex-1 xl:flex-row xl:items-center xl:justify-end">
                <div className="relative w-full xl:max-w-md">
                  <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder="Search name, id, description, or tags"
                    className="pl-11"
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                  />
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    variant={view === "grid" ? "default" : "ghost"}
                    size="icon"
                    onClick={() => setView("grid")}
                    aria-label="Grid view"
                  >
                    <Grid2X2 className="h-4 w-4" />
                  </Button>
                  <Button
                    variant={view === "list" ? "default" : "ghost"}
                    size="icon"
                    onClick={() => setView("list")}
                    aria-label="List view"
                  >
                    <List className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-2 border-t border-white/10 pt-4">
              {CATEGORIES.map((cat) => (
                <Button
                  key={cat.value}
                  variant={category === cat.value ? "default" : "outline"}
                  size="sm"
                  onClick={() => setCategory(category === cat.value ? null : cat.value)}
                >
                  {cat.label}
                </Button>
              ))}
            </div>
          </div>

          <TabsContent value="explore" className="mt-6">
            {loading ? (
              <EmptyPanel message="Loading environment registry" />
            ) : filtered.length === 0 ? (
              <EmptyPanel message="No environments match the active filters" />
            ) : (
              <div className="space-y-10">
                <section>
                  <SectionHeader
                    label="Priority Signal"
                    title="Featured Dossiers"
                    count={featured.length}
                  />
                  <div
                    className={cn(
                      "grid gap-4",
                      view === "grid" ? "grid-cols-1 md:grid-cols-2 xl:grid-cols-3" : "grid-cols-1"
                    )}
                  >
                    {featured.map((env, index) => (
                      <EnvironmentCard key={env.slug} env={env} view={view} index={index} />
                    ))}
                  </div>
                </section>

                {filtered.length > featured.length && (
                  <section>
                    <SectionHeader
                      label="Extended Index"
                      title="Archive Continuation"
                      count={filtered.length - featured.length}
                    />
                    <div
                      className={cn(
                        "grid gap-4",
                        view === "grid" ? "grid-cols-1 md:grid-cols-2 xl:grid-cols-3" : "grid-cols-1"
                      )}
                    >
                      {filtered.slice(featured.length).map((env, index) => (
                        <EnvironmentCard
                          key={env.slug}
                          env={env}
                          view={view}
                          index={featured.length + index}
                        />
                      ))}
                    </div>
                  </section>
                )}
              </div>
            )}
          </TabsContent>

          <TabsContent value="stars" className="mt-6">
            <EmptyPanel message="Signal curation and saved environments can land here later" />
          </TabsContent>

          <TabsContent value="archive" className="mt-6">
            {loading ? (
              <EmptyPanel message="Preparing archive view" />
            ) : filtered.length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {filtered.map((env, index) => (
                  <EnvironmentCard key={env.slug} env={env} view="grid" index={index} />
                ))}
              </div>
            ) : (
              <EmptyPanel message="Archive is empty for this query" />
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
