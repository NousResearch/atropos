import { NextRequest, NextResponse } from "next/server";
import { loadManifest } from "@/lib/env-api";

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const q = searchParams.get("q")?.toLowerCase();
  const category = searchParams.get("category");
  let manifest = loadManifest();
  if (q) {
    manifest = manifest.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        (e.description || "").toLowerCase().includes(q) ||
        (e.tags || []).some((t) => t.toLowerCase().includes(q))
    );
  }
  if (category) {
    manifest = manifest.filter((e) => (e.tags || []).includes(category));
  }
  return NextResponse.json(manifest);
}
